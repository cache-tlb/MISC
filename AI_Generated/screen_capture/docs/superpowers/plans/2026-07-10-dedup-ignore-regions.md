# 去重忽略区域 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自动保存去重支持可配置的"忽略区域"——判断是否重复时，涂黑这些矩形区域后再比较，但实际保存的图片内容不受影响（不涂黑）。

**Architecture:** `ScreenCaptureService` 再拆一层为 `CaptureBitmap()`（只截图）+ `EncodeToPngBytes(Bitmap)`（只编码），方便同一次截图分别生成"保存用的完整图"和"比较用的涂黑图"。新建 `IgnoreRegion` 数据类和 `AppConfig.IgnoreRegions` 字段。`ScreenshotDeduplicator` 新增纯函数 `ToPixelRect`（比例坐标转像素矩形，含 clamp），`CaptureAndSaveIfNotDuplicate` 内部对克隆图涂黑后再走原有的 `IsDuplicate`/`Remember`（这两个方法本身不需要改动）。

**Tech Stack:** 沿用现有 .NET 8 / WinForms / xUnit 项目（`d:/tmp/screen_capture`），无新增依赖。

## Global Constraints

- 配置字段：`IgnoreRegions`（`List<IgnoreRegion>`，每个元素含 `X`/`Y`/`Width`/`Height` 四个 `double`），默认空列表
- 比例转像素规则：`X`/`Y` 各自 clamp 到 `[0,1]`；`Width` clamp 到 `[0, 1-clamped X]`；`Height` clamp 到 `[0, 1-clamped Y]`；再分别乘以截图实际宽/高并四舍五入取整
- 忽略区域只影响"判断是否重复"这一步的比较图，**从不影响实际保存的截图内容**
- 手动热键截图完全不受影响，不参与去重也不受忽略区域影响
- `IsDuplicate`/`Remember` 的现有实现和测试保持不变，不允许修改它们的语义（它们只关心传入的字节数组本身）

---

## File Structure

```
src/ScreenCaptureDaemon/
  ScreenCaptureService.cs      (修改：拆出 CaptureBitmap + EncodeToPngBytes)
  IgnoreRegion.cs                (新建：简单数据类)
  AppConfig.cs                    (修改：新增 IgnoreRegions 字段)
  ScreenshotDeduplicator.cs    (修改：新增 ToPixelRect + 涂黑比较逻辑)
  Program.cs                      (修改：传入 config.IgnoreRegions)
  appsettings.json               (修改：新增示例字段)
tests/ScreenCaptureDaemon.Tests/
  AppConfigTests.cs                    (修改：新增用例)
  ScreenshotDeduplicatorTests.cs      (修改：新增 ToPixelRect 用例)
```

---

### Task 1: `ScreenCaptureService` — 拆出 `CaptureBitmap` + `EncodeToPngBytes`

**Files:**
- Modify: `src/ScreenCaptureDaemon/ScreenCaptureService.cs`

**Interfaces:**
- Produces: `static Bitmap ScreenCaptureService.CaptureBitmap()`（调用方负责 `Dispose`）、`static byte[] ScreenCaptureService.EncodeToPngBytes(Bitmap bitmap)`
- `CaptureToBytes()`/`Capture(string)` 对外签名和行为保持不变，内部改为组合调用上面两个新方法

**Note:** 行为保持不变的重构，没有新增自动化测试（跟 Task 1 of 上一个 dedup 计划一样，只做人工验证）。

- [ ] **Step 1: 重构 `ScreenCaptureService.cs`**

把 `src/ScreenCaptureDaemon/ScreenCaptureService.cs` 整个文件替换为：

```csharp
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace ScreenCaptureDaemon;

public static class ScreenCaptureService
{
    public static string BuildFileName(DateTime timestamp)
    {
        return $"screenshot_{timestamp:yyyyMMdd_HHmmss_fff}.png";
    }

    // 调用方负责 Dispose 返回的 Bitmap。
    public static Bitmap CaptureBitmap()
    {
        var screen = Screen.FromPoint(Cursor.Position);
        var bounds = screen.Bounds;

        var bitmap = new Bitmap(bounds.Width, bounds.Height);
        using (var g = Graphics.FromImage(bitmap))
        {
            g.CopyFromScreen(bounds.Location, Point.Empty, bounds.Size);
        }

        return bitmap;
    }

    public static byte[] EncodeToPngBytes(Bitmap bitmap)
    {
        using var ms = new MemoryStream();
        bitmap.Save(ms, ImageFormat.Png);
        return ms.ToArray();
    }

    public static byte[] CaptureToBytes()
    {
        using var bitmap = CaptureBitmap();
        return EncodeToPngBytes(bitmap);
    }

    public static string Capture(string saveDirectory)
    {
        Directory.CreateDirectory(saveDirectory);
        var imageBytes = CaptureToBytes();
        var path = Path.Combine(saveDirectory, BuildFileName(DateTime.Now));
        File.WriteAllBytes(path, imageBytes);
        return path;
    }
}
```

- [ ] **Step 2: 运行既有测试确认没有回归**

Run: `dotnet test --filter ScreenCaptureServiceTests`
Expected: `Passed!` — 2 个既有测试（`BuildFileName_*`）依然通过

Run: `dotnet build`
Expected: `Build succeeded.`

- [ ] **Step 3: 人工验证 `Capture` 行为不变**

用之前任务用过的临时控制台脚本方式调用 `ScreenCaptureService.Capture(@"C:\temp\capture-test")`，确认生成的 PNG 内容、尺寸、文件名格式与重构前一致。

- [ ] **Step 4: Commit**

```bash
git add src/ScreenCaptureDaemon/ScreenCaptureService.cs
git commit -m "refactor: split ScreenCaptureService.CaptureToBytes into CaptureBitmap + EncodeToPngBytes"
```

---

### Task 2: `IgnoreRegion` + `AppConfig.IgnoreRegions`

**Files:**
- Create: `src/ScreenCaptureDaemon/IgnoreRegion.cs`
- Modify: `src/ScreenCaptureDaemon/AppConfig.cs`
- Modify: `src/ScreenCaptureDaemon/appsettings.json`
- Modify: `tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs`

**Interfaces:**
- Produces: `sealed class IgnoreRegion { double X; double Y; double Width; double Height; }`（均为 `{ get; set; }`）、`AppConfig.IgnoreRegions`（`List<IgnoreRegion>`，默认空列表）

- [ ] **Step 1: 写失败的测试**

在 `tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs` 的 `AppConfigTests` 类中，`Load_ReadsAutoSaveIntervalSecondsFromFile` 方法之后（`Load_FallsBackToDefaultInterval_WhenAutoSaveIntervalSecondsIsNotPositive` 这个 Theory 之前或之后均可，放在类末尾 `}` 之前）新增：

```csharp
    [Fact]
    public void Load_UsesEmptyIgnoreRegions_WhenFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Empty(config.IgnoreRegions);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsIgnoreRegionsFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"IgnoreRegions\": [{\"X\": 0.9, \"Y\": 0.95, \"Width\": 0.1, \"Height\": 0.05}]}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Single(config.IgnoreRegions);
            Assert.Equal(0.9, config.IgnoreRegions[0].X);
            Assert.Equal(0.95, config.IgnoreRegions[0].Y);
            Assert.Equal(0.1, config.IgnoreRegions[0].Width);
            Assert.Equal(0.05, config.IgnoreRegions[0].Height);
        }
        finally
        {
            File.Delete(path);
        }
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter AppConfigTests`
Expected: FAIL（编译错误：`AppConfig` 还没有 `IgnoreRegions` 属性，`IgnoreRegion` 类型不存在）

- [ ] **Step 3: 新建 `IgnoreRegion`**

`src/ScreenCaptureDaemon/IgnoreRegion.cs`（新建文件）：

```csharp
namespace ScreenCaptureDaemon;

public sealed class IgnoreRegion
{
    public double X { get; set; }
    public double Y { get; set; }
    public double Width { get; set; }
    public double Height { get; set; }
}
```

- [ ] **Step 4: 给 `AppConfig` 新增字段**

在 `src/ScreenCaptureDaemon/AppConfig.cs` 中，`AutoSaveIntervalSeconds` 属性之后新增：

```csharp
    public List<IgnoreRegion> IgnoreRegions { get; set; } = new();
```

- [ ] **Step 5: 更新示例配置文件**

`src/ScreenCaptureDaemon/appsettings.json` 改为：

```json
{
  "SaveDirectory": "F:\\Screenshots",
  "Port": 8080,
  "Hotkey": "Win+Shift+Z",
  "ToggleAutoSaveHotkey": "Win+Shift+X",
  "AutoSaveIntervalSeconds": 20,
  "IgnoreRegions": []
}
```

- [ ] **Step 6: 运行测试确认通过**

Run: `dotnet test --filter AppConfigTests`
Expected: `Passed!` — 全部 AppConfigTests 通过（既有 14 个 + 本任务新增 2 个 = 16 个）

- [ ] **Step 7: 运行完整测试套件确认无回归**

Run: `dotnet test`
Expected: 全部通过（既有 41 个 + 本任务新增 2 个 = 43 个）

- [ ] **Step 8: Commit**

```bash
git add src/ScreenCaptureDaemon/IgnoreRegion.cs src/ScreenCaptureDaemon/AppConfig.cs src/ScreenCaptureDaemon/appsettings.json tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs
git commit -m "feat: add IgnoreRegion type and AppConfig.IgnoreRegions field"
```

---

### Task 3: `ScreenshotDeduplicator` — `ToPixelRect` + 涂黑比较逻辑

**Files:**
- Modify: `src/ScreenCaptureDaemon/ScreenshotDeduplicator.cs`
- Modify: `tests/ScreenCaptureDaemon.Tests/ScreenshotDeduplicatorTests.cs`

**Interfaces:**
- Consumes: `IgnoreRegion`（Task 2 产出）、`ScreenCaptureService.CaptureBitmap()`/`EncodeToPngBytes(Bitmap)`（Task 1 产出）
- Produces: `ScreenshotDeduplicator` 的构造函数新增可选参数 `IReadOnlyList<IgnoreRegion>? ignoreRegions = null`（不传等价于空列表，保证既有的 `new ScreenshotDeduplicator()` 调用和测试无需改动）；新增 `internal static Rectangle ToPixelRect(IgnoreRegion region, int screenWidth, int screenHeight)`
- `IsDuplicate`/`Remember` 的方法签名和实现**完全不变**，不需要触碰

**Note:** `ToPixelRect` 是纯函数（不触碰屏幕/文件系统），可以完整自动化测试。`CaptureAndSaveIfNotDuplicate` 会调用真实的 `ScreenCaptureService.CaptureBitmap()`，跟项目里其他触碰真实屏幕的代码一样只做人工验证（本任务 Step 6）。

- [ ] **Step 1: 写失败的测试**

在 `tests/ScreenCaptureDaemon.Tests/ScreenshotDeduplicatorTests.cs` 顶部把 `using ScreenCaptureDaemon;` 之后新增一行 `using System.Drawing;`（即最终顶部两行 using 变为 `using System.Drawing;` 和 `using ScreenCaptureDaemon;`，按字母序排列），然后在类末尾 `}` 之前新增：

```csharp
    [Fact]
    public void ToPixelRect_ConvertsNormalRegionToPixelCoordinates()
    {
        var region = new IgnoreRegion { X = 0.9, Y = 0.95, Width = 0.1, Height = 0.05 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 3840, 2160);

        Assert.Equal(new Rectangle(3456, 2052, 384, 108), rect);
    }

    [Fact]
    public void ToPixelRect_ClampsNegativeXAndY_ToZero()
    {
        var region = new IgnoreRegion { X = -0.5, Y = -0.2, Width = 0.1, Height = 0.1 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 1000, 1000);

        Assert.Equal(0, rect.X);
        Assert.Equal(0, rect.Y);
    }

    [Fact]
    public void ToPixelRect_ClampsWidthAndHeight_WhenTheyWouldExceedScreenBounds()
    {
        var region = new IgnoreRegion { X = 0.9, Y = 0.9, Width = 0.5, Height = 0.5 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 1000, 1000);

        Assert.Equal(900, rect.X);
        Assert.Equal(900, rect.Y);
        Assert.Equal(100, rect.Width);
        Assert.Equal(100, rect.Height);
    }

    [Fact]
    public void ToPixelRect_ClampsXGreaterThanOne_ToZeroWidthAtScreenEdge()
    {
        var region = new IgnoreRegion { X = 1.5, Y = 0, Width = 0.2, Height = 0.2 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 1000, 1000);

        Assert.Equal(1000, rect.X);
        Assert.Equal(0, rect.Width);
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter ScreenshotDeduplicatorTests`
Expected: FAIL（编译错误：找不到 `ScreenshotDeduplicator.ToPixelRect`）；既有的 5 个 `IsDuplicate`/`Remember` 测试应该仍然编译通过（`ScreenshotDeduplicator` 的构造函数这一步还没改，等 Step 3 一起改）

- [ ] **Step 3: 实现涂黑比较逻辑**

把 `src/ScreenCaptureDaemon/ScreenshotDeduplicator.cs` 整个文件替换为：

```csharp
using System.Drawing;

namespace ScreenCaptureDaemon;

public sealed class ScreenshotDeduplicator
{
    private readonly IReadOnlyList<IgnoreRegion> _ignoreRegions;
    private byte[]? _lastImageBytes;

    public ScreenshotDeduplicator(IReadOnlyList<IgnoreRegion>? ignoreRegions = null)
    {
        _ignoreRegions = ignoreRegions ?? Array.Empty<IgnoreRegion>();
    }

    public bool IsDuplicate(byte[] imageBytes)
    {
        return _lastImageBytes != null && imageBytes.AsSpan().SequenceEqual(_lastImageBytes);
    }

    public void Remember(byte[] imageBytes)
    {
        _lastImageBytes = imageBytes;
    }

    internal static Rectangle ToPixelRect(IgnoreRegion region, int screenWidth, int screenHeight)
    {
        var x = Math.Clamp(region.X, 0.0, 1.0);
        var y = Math.Clamp(region.Y, 0.0, 1.0);
        var width = Math.Clamp(region.Width, 0.0, 1.0 - x);
        var height = Math.Clamp(region.Height, 0.0, 1.0 - y);

        return new Rectangle(
            (int)Math.Round(x * screenWidth),
            (int)Math.Round(y * screenHeight),
            (int)Math.Round(width * screenWidth),
            (int)Math.Round(height * screenHeight));
    }

    public string? CaptureAndSaveIfNotDuplicate(string saveDirectory)
    {
        Directory.CreateDirectory(saveDirectory);
        using var bitmap = ScreenCaptureService.CaptureBitmap();

        using var comparisonBitmap = (Bitmap)bitmap.Clone();
        if (_ignoreRegions.Count > 0)
        {
            using var g = Graphics.FromImage(comparisonBitmap);
            foreach (var region in _ignoreRegions)
            {
                g.FillRectangle(Brushes.Black, ToPixelRect(region, bitmap.Width, bitmap.Height));
            }
        }

        var comparisonBytes = ScreenCaptureService.EncodeToPngBytes(comparisonBitmap);

        if (IsDuplicate(comparisonBytes))
        {
            return null;
        }

        var imageBytes = ScreenCaptureService.EncodeToPngBytes(bitmap);
        var path = Path.Combine(saveDirectory, ScreenCaptureService.BuildFileName(DateTime.Now));
        File.WriteAllBytes(path, imageBytes);
        Remember(comparisonBytes);
        return path;
    }
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `dotnet test --filter ScreenshotDeduplicatorTests`
Expected: `Passed!` — 全部通过（既有 5 个 `IsDuplicate`/`Remember` 用例 + 本任务新增 4 个 `ToPixelRect` 用例 = 9 个）

- [ ] **Step 5: 运行完整测试套件确认无回归**

Run: `dotnet test`
Expected: 全部通过（既有 43 个 + 本任务新增 4 个 = 47 个）

- [ ] **Step 6: 人工验证涂黑比较不影响保存内容**

用临时控制台脚本方式，构造一个 `new ScreenshotDeduplicator(new[] { new IgnoreRegion { X = 0, Y = 0, Width = 0.2, Height = 0.2 } })`，调用一次 `CaptureAndSaveIfNotDuplicate(@"C:\temp\ignore-region-test")`，打开生成的 PNG 文件确认：
- 左上角那 20%×20% 的区域在**保存的文件里**依然是正常屏幕内容（没有被涂黑）
- 文件能正常打开，不是损坏的图片

Expected: 以上两项符合预期（涂黑只发生在内存里用来比较的临时图，从不写入磁盘）

- [ ] **Step 7: Commit**

```bash
git add src/ScreenCaptureDaemon/ScreenshotDeduplicator.cs tests/ScreenCaptureDaemon.Tests/ScreenshotDeduplicatorTests.cs
git commit -m "feat: mask configured ignore regions when comparing auto-saved screenshots"
```

---

### Task 4: `Program.cs` — 接线 `IgnoreRegions` + 端到端验证

**Files:**
- Modify: `src/ScreenCaptureDaemon/Program.cs`

**Interfaces:**
- Consumes: `AppConfig.IgnoreRegions`（Task 2）、`ScreenshotDeduplicator` 新构造函数（Task 3）

- [ ] **Step 1: 接线 `Program.cs`**

将 `src/ScreenCaptureDaemon/Program.cs` 中的：

```csharp
        var deduplicator = new ScreenshotDeduplicator();
```

改为：

```csharp
        var deduplicator = new ScreenshotDeduplicator(config.IgnoreRegions);
```

文件其余部分保持不变。

- [ ] **Step 2: 构建并运行完整测试套件**

Run: `dotnet build`
Expected: `Build succeeded.`，无编译错误

Run: `dotnet test`
Expected: 全部通过（47 个测试，数量与 Task 3 结束时一致，本任务不新增测试用例）

- [ ] **Step 3: 人工验证**

1. 在 `src/ScreenCaptureDaemon/bin/Debug/net8.0-windows/appsettings.json`（构建输出目录的副本）里，把 `AutoSaveIntervalSeconds` 改成一个方便观察的小值（比如 `5`），并配置一个覆盖任务栏时钟位置的 `IgnoreRegions`（比如 `[{"X": 0.85, "Y": 0.9, "Width": 0.15, "Height": 0.1}]`，具体数值按实际任务栏时钟在屏幕上的相对位置调整）
2. 运行 `SCDaemon.exe`，开启自动保存，保持屏幕其余内容不动，观察至少 3-4 个间隔周期（时钟会正常跳动）
3. 确认保存目录里**只有一张**截图（时钟跳动不再触发重复保存了）
4. 打开这张截图，确认时钟区域的内容是正常的（没有被涂黑——涂黑只在内存里用于比较，不影响保存的文件）
5. 改变一下屏幕其他区域的内容（跟时钟无关的地方），确认下一个周期正常产生新截图
6. 把 `IgnoreRegions` 改回空数组 `[]`，重启程序，确认行为恢复成"时钟跳动也会触发保存"（验证不配置忽略区域时行为跟之前完全一致）

Expected: 以上 6 项均符合预期

- [ ] **Step 4: Commit**

```bash
git add src/ScreenCaptureDaemon/Program.cs
git commit -m "feat: wire configured IgnoreRegions into the auto-save deduplicator"
```
