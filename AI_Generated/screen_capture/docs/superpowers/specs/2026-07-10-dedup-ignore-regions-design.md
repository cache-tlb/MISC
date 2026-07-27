# 去重忽略区域 设计文档

日期：2026-07-10

## 背景

实测发现，自动保存去重（逐字节比对 PNG）在屏幕右下角有系统时钟这类持续跳动的小区域时基本失效——每隔一个周期时钟就会变化几个像素，导致 PNG 压缩后的字节大范围不同，去重判断为"不重复"，每次都保存。这是逐字节精确比对的既定权衡（详见 `2026-07-10-autosave-dedup-design.md` 的排除范围），本次给它加一个可配置的"忽略区域"机制来缓解。

## 目标

新增可配置的矩形区域列表。**保存的截图内容不受影响**（时钟等正常显示），但判断"是否与上一张重复"时，忽略这些区域内的像素变化。

## 配置格式

`appsettings.json` 新增 `IgnoreRegions` 字段，矩形数组，默认空数组：

```json
{
  "IgnoreRegions": [
    { "X": 0.9, "Y": 0.95, "Width": 0.1, "Height": 0.05 }
  ]
}
```

`X`/`Y`/`Width`/`Height` 都是 0-1 之间的比例值（相对屏幕宽高），实际使用时换算成像素矩形：

1. `X`、`Y` 各自 clamp 到 `[0, 1]`
2. `Width` clamp 到 `[0, 1 - clamped X]`（保证矩形右边界不超出屏幕）
3. `Height` clamp 到 `[0, 1 - clamped Y]`（保证矩形下边界不超出屏幕）
4. 四个比例值分别乘以截图的实际宽/高，四舍五入取整，得到像素矩形

不配置该字段（或配置为空数组）时行为与当前完全一致，不忽略任何区域。

## 作用范围

只影响自动保存去重的比较步骤。手动热键截图、实际保存到磁盘的图片内容，都不受任何影响——忽略区域只用来生成"用于比较的临时图"，从不影响真正保存的文件。

## 代码结构

### `ScreenCaptureService`（修改）

再拆一层，把"截图"和"编码成 PNG 字节"彻底分开，方便复用同一次截图分别生成"用于保存的完整图"和"用于比较的涂黑图"：

```csharp
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
```

`CaptureToBytes`/`Capture` 对外行为完全不变（又一次行为保持不变的重构），手动热键那条路径不受任何影响。

### `IgnoreRegion`（新建，小数据类）

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

### `AppConfig`（修改）

新增：

```csharp
public List<IgnoreRegion> IgnoreRegions { get; set; } = new();
```

### `ScreenshotDeduplicator`（修改）

```csharp
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

要点：
- `IsDuplicate`/`Remember` 完全不变，不需要改动，也不需要改动它们现有的测试——它们只关心字节数组本身，不关心字节代表的是"完整图"还是"涂黑图"
- `Remember` 依然只在 `File.WriteAllBytes` 成功之后调用，延续之前修复过的那条不变量
- `bitmap`（原图，用来保存）和 `comparisonBitmap`（涂黑后的克隆，只用来比较）分开处理，保存的文件内容永远是完整原图
- 构造函数的 `ignoreRegions` 参数带默认值 `null`（等价于空列表），保证已有的 `new ScreenshotDeduplicator()` 调用和测试不需要改动
- `ToPixelRect` 是纯函数（不触碰屏幕/文件系统），可以完整自动化测试

### `Program.cs`（修改）

```csharp
var deduplicator = new ScreenshotDeduplicator(config.IgnoreRegions);
```

其余部分不变。

## 明确排除的范围（YAGNI）

- 不支持圆形/多边形等非矩形忽略区域
- 不支持"忽略区域内允许多大差异"这种模糊阈值，忽略区域内完全不参与比较（100% 忽略）
- 不做忽略区域之间重叠的特殊处理（重叠只是多涂一次黑，无副作用）
- 不提供图形化框选忽略区域的工具，仍然是手改 JSON 里的比例数字

## 测试计划

- 自动化：`ScreenshotDeduplicator.ToPixelRect` 用手工构造的 `IgnoreRegion` 单测覆盖：正常区域换算、X/Y 为负数时 clamp 到 0、X/Y 超过 1 时 clamp 到边界（宽高变 0）、Width/Height 超出屏幕边界时被 clamp
- 人工验证：配置一个覆盖任务栏时钟的忽略区域，开启自动保存，确认时钟跳动不再触发新的重复保存；确认保存的图片里时钟内容依然正常显示（没有被涂黑）；再改变一下屏幕其他区域内容，确认依然能正常触发保存
