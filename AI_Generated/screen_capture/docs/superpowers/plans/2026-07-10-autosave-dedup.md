# 自动保存去重 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自动保存模式下，若某次定时截图与最近一次已保存的自动截图字节级完全相同，则静默丢弃，不落盘；手动热键截图不受影响。

**Architecture:** `ScreenCaptureService` 拆出一个不落盘的 `CaptureToBytes()`，`Capture()` 内部改为复用它（行为不变）。新建 `ScreenshotDeduplicator`：`IsDuplicate(byte[])`（只读比较）和 `Remember(byte[])`（更新基准）是可自动化测试的纯逻辑，`CaptureAndSaveIfNotDuplicate(string)` 是触碰真实屏幕/文件系统的整体入口，且只在 `File.WriteAllBytes` 成功之后才调用 `Remember`。`Program.cs` 只把自动保存那一路的回调换成走 `ScreenshotDeduplicator`，手动热键那一路保持不变。

**Tech Stack:** 沿用现有 .NET 8 / WinForms / xUnit 项目（`d:/tmp/screen_capture`），无新增依赖。

## Global Constraints

- 比较方式：PNG 编码后的字节数组逐字节完全相等，不做模糊/相似度判断
- 比较基准：只跟"最近一次成功保存的自动截图"比较，不扫描保存目录里的历史文件
- 命中重复：不落盘、不写日志，完全静默
- 未命中重复：正常保存，并把这次的字节内容记为新的比较基准
- "最近一次"记忆存活在 `ScreenshotDeduplicator` 实例的整个生命周期内（进程运行期间），不随自动保存开关的开/关重置
- 手动热键截图（`Program.cs` 里 `hotkeyManager.HotkeyPressed` 的回调）完全不参与去重，`ScreenCaptureService.Capture` 的对外行为必须保持不变

---

## File Structure

```
src/ScreenCaptureDaemon/
  ScreenCaptureService.cs     (修改：拆出 CaptureToBytes，Capture 内部复用)
  ScreenshotDeduplicator.cs   (新建：判重逻辑 + 整体入口)
  Program.cs                    (修改：自动保存回调改走 ScreenshotDeduplicator)
tests/ScreenCaptureDaemon.Tests/
  ScreenshotDeduplicatorTests.cs  (新建：IsDuplicate/Remember 的用例)
```

---

### Task 1: `ScreenCaptureService` — 拆出 `CaptureToBytes`

**Files:**
- Modify: `src/ScreenCaptureDaemon/ScreenCaptureService.cs`

**Interfaces:**
- Produces: `static byte[] ScreenCaptureService.CaptureToBytes()` — 截取鼠标所在显示器画面，编码为 PNG 字节数组，不写文件
- `static string ScreenCaptureService.Capture(string saveDirectory)` 对外签名和行为保持不变（仍然写文件并返回路径），内部实现改为复用 `CaptureToBytes()`

**Note:** 这是一次行为保持不变的重构，没有新增自动化测试（`Capture`/`CaptureToBytes` 都依赖真实 Windows 桌面会话，和现有的做法一致，只做人工验证）；现有 `BuildFileName` 的两个测试不受影响，用它们验证没有破坏编译/既有行为。

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

    public static byte[] CaptureToBytes()
    {
        var screen = Screen.FromPoint(Cursor.Position);
        var bounds = screen.Bounds;

        using var bitmap = new Bitmap(bounds.Width, bounds.Height);
        using (var g = Graphics.FromImage(bitmap))
        {
            g.CopyFromScreen(bounds.Location, Point.Empty, bounds.Size);
        }

        using var ms = new MemoryStream();
        bitmap.Save(ms, ImageFormat.Png);
        return ms.ToArray();
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

调用 `ScreenCaptureService.Capture(@"C:\temp\capture-test")`（可以用之前任务用过的临时控制台项目/脚本方式），确认：
- 生成的 PNG 文件依然是正确的截图内容，尺寸与当前鼠标所在显示器分辨率一致
- 文件名格式依然是 `screenshot_yyyyMMdd_HHmmss_fff.png`

Expected: 与重构前行为完全一致

- [ ] **Step 4: Commit**

```bash
git add src/ScreenCaptureDaemon/ScreenCaptureService.cs
git commit -m "refactor: extract ScreenCaptureService.CaptureToBytes for reuse without writing to disk"
```

---

### Task 2: `ScreenshotDeduplicator` — 判重逻辑

**Files:**
- Create: `src/ScreenCaptureDaemon/ScreenshotDeduplicator.cs`
- Create: `tests/ScreenCaptureDaemon.Tests/ScreenshotDeduplicatorTests.cs`

**Interfaces:**
- Consumes: `ScreenCaptureService.CaptureToBytes()`、`ScreenCaptureService.BuildFileName(DateTime)`（Task 1 产出）
- Produces: `sealed class ScreenshotDeduplicator { bool IsDuplicate(byte[] imageBytes); void Remember(byte[] imageBytes); string? CaptureAndSaveIfNotDuplicate(string saveDirectory); }`

**Note:** `IsDuplicate`/`Remember` 是纯逻辑（不触碰屏幕/文件系统），可以完整自动化测试。`CaptureAndSaveIfNotDuplicate` 会调用真实的 `ScreenCaptureService.CaptureToBytes()`，跟项目里其他触碰真实屏幕的代码一样只做人工验证（本任务 Step 6），且只在 `File.WriteAllBytes` 成功之后才调用 `Remember`，保证判重基准只反映真正落盘成功的截图。

- [ ] **Step 1: 写失败的测试**

`tests/ScreenCaptureDaemon.Tests/ScreenshotDeduplicatorTests.cs`（新建文件）：

```csharp
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class ScreenshotDeduplicatorTests
{
    [Fact]
    public void IsDuplicate_ReturnsFalse_WhenNothingRememberedYet()
    {
        var deduplicator = new ScreenshotDeduplicator();

        var result = deduplicator.IsDuplicate(new byte[] { 1, 2, 3 });

        Assert.False(result);
    }

    [Fact]
    public void IsDuplicate_ReturnsTrue_WhenBytesMatchRememberedValue()
    {
        var deduplicator = new ScreenshotDeduplicator();
        deduplicator.Remember(new byte[] { 1, 2, 3 });

        var result = deduplicator.IsDuplicate(new byte[] { 1, 2, 3 });

        Assert.True(result);
    }

    [Fact]
    public void IsDuplicate_ReturnsFalse_WhenBytesDifferFromRememberedValue()
    {
        var deduplicator = new ScreenshotDeduplicator();
        deduplicator.Remember(new byte[] { 1, 2, 3 });

        var result = deduplicator.IsDuplicate(new byte[] { 4, 5, 6 });

        Assert.False(result);
    }

    [Fact]
    public void IsDuplicate_ReturnsFalse_WhenBytesDifferOnlyInLength()
    {
        var deduplicator = new ScreenshotDeduplicator();
        deduplicator.Remember(new byte[] { 1, 2, 3 });

        var result = deduplicator.IsDuplicate(new byte[] { 1, 2, 3, 4 });

        Assert.False(result);
    }

    [Fact]
    public void IsDuplicate_ReflectsMostRecentlyRememberedValue()
    {
        var deduplicator = new ScreenshotDeduplicator();

        deduplicator.Remember(new byte[] { 9, 9, 9 });
        var stillMatchesFirst = deduplicator.IsDuplicate(new byte[] { 9, 9, 9 });

        deduplicator.Remember(new byte[] { 1, 1, 1 });
        var matchesUpdatedValue = deduplicator.IsDuplicate(new byte[] { 1, 1, 1 });
        var noLongerMatchesOldValue = deduplicator.IsDuplicate(new byte[] { 9, 9, 9 });

        Assert.True(stillMatchesFirst);
        Assert.True(matchesUpdatedValue);
        Assert.False(noLongerMatchesOldValue);
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter ScreenshotDeduplicatorTests`
Expected: FAIL（编译错误：找不到类型 `ScreenshotDeduplicator`）

- [ ] **Step 3: 实现 `ScreenshotDeduplicator`**

`src/ScreenCaptureDaemon/ScreenshotDeduplicator.cs`:

```csharp
namespace ScreenCaptureDaemon;

public sealed class ScreenshotDeduplicator
{
    private byte[]? _lastImageBytes;

    public bool IsDuplicate(byte[] imageBytes)
    {
        return _lastImageBytes != null && imageBytes.AsSpan().SequenceEqual(_lastImageBytes);
    }

    public void Remember(byte[] imageBytes)
    {
        _lastImageBytes = imageBytes;
    }

    public string? CaptureAndSaveIfNotDuplicate(string saveDirectory)
    {
        Directory.CreateDirectory(saveDirectory);
        var imageBytes = ScreenCaptureService.CaptureToBytes();

        if (IsDuplicate(imageBytes))
        {
            return null;
        }

        var path = Path.Combine(saveDirectory, ScreenCaptureService.BuildFileName(DateTime.Now));
        File.WriteAllBytes(path, imageBytes);
        Remember(imageBytes);
        return path;
    }
}
```

`Remember` 必须放在 `File.WriteAllBytes` 成功之后调用，不能在写入之前推进去重基准——否则一次写入失败（磁盘满/权限问题）会污染基准，导致下一次即使是全新内容也可能被误判为重复而丢弃。

- [ ] **Step 4: 运行测试确认通过**

Run: `dotnet test --filter ScreenshotDeduplicatorTests`
Expected: `Passed!` — 5 个测试全部通过

- [ ] **Step 5: 运行完整测试套件确认无回归**

Run: `dotnet test`
Expected: 全部通过（既有 36 个 + 本任务新增 5 个 = 41 个）

- [ ] **Step 6: 人工验证 `CaptureAndSaveIfNotDuplicate`**

用之前任务验证 `Capture()` 用过的方式（临时控制台脚本/项目，引用本项目），连续调用两次 `new ScreenshotDeduplicator().CaptureAndSaveIfNotDuplicate(@"C:\temp\dedup-test")`（不要在两次调用之间改变屏幕内容），确认：
- 第一次调用返回一个非空路径，且该路径下确实生成了 PNG 文件
- 第二次调用返回 `null`，且没有生成第二个文件（目录里文件数量不变）

Expected: 以上两项符合预期

- [ ] **Step 7: Commit**

```bash
git add src/ScreenCaptureDaemon/ScreenshotDeduplicator.cs tests/ScreenCaptureDaemon.Tests/ScreenshotDeduplicatorTests.cs
git commit -m "feat: add ScreenshotDeduplicator to skip byte-identical auto-saves"
```

---

### Task 3: `Program.cs` — 接线去重 + 端到端验证

**Files:**
- Modify: `src/ScreenCaptureDaemon/Program.cs`

**Interfaces:**
- Consumes: `ScreenshotDeduplicator`（Task 2）

- [ ] **Step 1: 接线 `Program.cs`**

将 `src/ScreenCaptureDaemon/Program.cs` 中原来的：

```csharp
        var autoSaveController = new AutoSaveController(config.AutoSaveIntervalSeconds, () =>
        {
            try
            {
                ScreenCaptureService.Capture(config.SaveDirectory);
            }
            catch (Exception ex)
            {
                Logger.Error($"自动保存截图失败: {ex.Message}");
            }
        });
```

改为：

```csharp
        var deduplicator = new ScreenshotDeduplicator();
        var autoSaveController = new AutoSaveController(config.AutoSaveIntervalSeconds, () =>
        {
            try
            {
                deduplicator.CaptureAndSaveIfNotDuplicate(config.SaveDirectory);
            }
            catch (Exception ex)
            {
                Logger.Error($"自动保存截图失败: {ex.Message}");
            }
        });
```

手动热键的回调（`hotkeyManager.HotkeyPressed` 那一段，调用 `ScreenCaptureService.Capture(config.SaveDirectory)`）保持完全不变。文件其余部分（Mutex/配置加载/HTTP 启动/两个热键的解析注册/`ApplicationExit`/`Application.Run()`）也保持不变。

- [ ] **Step 2: 构建并运行完整测试套件**

Run: `dotnet build`
Expected: `Build succeeded.`，无编译错误

Run: `dotnet test`
Expected: 全部通过（41 个测试，数量与 Task 2 结束时一致，本任务不新增测试用例）

- [ ] **Step 3: 人工验证**

1. 在 `src/ScreenCaptureDaemon/bin/Debug/net8.0-windows/appsettings.json`（构建输出目录的副本）里把 `AutoSaveIntervalSeconds` 改成一个方便观察的小值（比如 `5`），运行 `SCDaemon.exe`
2. 按下开关热键开启自动保存，**不要移动/改变屏幕内容**，等待至少 15-20 秒（跨越 3-4 个间隔周期），确认保存目录里**只有一张**截图（后续重复的都被静默丢弃了，没有产生新文件）
3. 改变一下屏幕内容（比如移动一下某个窗口、切换一下桌面壁纸或打开个新窗口），等下一个间隔周期，确认保存目录里**出现了第二张**截图
4. 确认整个过程中 `log.txt` 没有因为"丢弃重复截图"产生任何新的日志行（静默，符合预期）
5. 确认手动截图热键在这个过程中依然每次都正常保存（不受去重影响），哪怕连续两次按下热键时屏幕内容完全没变

Expected: 以上 5 项均符合预期

- [ ] **Step 4: Commit**

```bash
git add src/ScreenCaptureDaemon/Program.cs
git commit -m "feat: wire ScreenshotDeduplicator into the auto-save capture path"
```
