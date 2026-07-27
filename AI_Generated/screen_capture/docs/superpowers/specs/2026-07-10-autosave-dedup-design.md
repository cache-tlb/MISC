# 自动保存去重 设计文档

日期：2026-07-10

## 背景与目标

自动保存模式下，如果某一次定时截图与最近一次已保存的自动截图完全一样（字节级相同），则丢弃这次截图，不落盘。手动热键截图不受影响，照常每次都保存。

## 作用范围

仅影响 `AutoSaveController` 触发的定时截图。`Program.cs` 里手动截图热键（`hotkeyManager.HotkeyPressed`）的回调保持不变，继续直接调用 `ScreenCaptureService.Capture`，不经过去重逻辑。

## 比较方式

- 比较对象：PNG 编码后的字节数组，逐字节完全相等才算"完全一样"，不做任何模糊/相似度判断
- 比较基准：只跟"最近一次成功保存的自动截图"比较，不扫描保存目录里的历史文件
- 命中重复：不写文件、不写日志，完全静默丢弃
- 未命中重复（第一张，或与上一张不同）：正常保存，并把这次的字节内容记为新的比较基准

## 状态生命周期

"最近一次保存的截图"这个记忆存活在内存里，跟随进程运行的整个生命周期：

- 关闭自动保存开关再重新打开，不会重置这个记忆——它代表的是"保存目录里最新一张自动截图长什么样"，与开关的开/关状态无关
- 程序重启后自然清空，从零开始（第一张自动截图必定会被保存）

## 代码结构

### `ScreenCaptureService`（修改）

新增一个不落盘的纯截图方法，现有 `Capture` 内部改为复用它，对外行为完全不变：

```csharp
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
```

### `ScreenshotDeduplicator`（新建）

```csharp
public sealed class ScreenshotDeduplicator
{
    private byte[]? _lastImageBytes;

    // 纯逻辑，只读比较，不修改状态，可自动化单测
    public bool IsDuplicate(byte[] imageBytes)
    {
        return _lastImageBytes != null && imageBytes.AsSpan().SequenceEqual(_lastImageBytes);
    }

    // 纯逻辑，只修改状态，可自动化单测
    public void Remember(byte[] imageBytes)
    {
        _lastImageBytes = imageBytes;
    }

    // 对外整体入口：截图 + 判重 + （不重复时）落盘
    // 注意：Remember 必须放在 File.WriteAllBytes 成功之后调用，不能在写入之前就推进去重基准——
    // 否则一次写入失败（磁盘满/权限问题）会污染基准，导致下一次即使是全新内容也可能被误判丢弃
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

`IsDuplicate`/`Remember` 是这个类里唯一值得自动化测试的部分（纯逻辑，不依赖真实屏幕）；`CaptureAndSaveIfNotDuplicate` 因为会调用真实的 `Screen.FromPoint`/`CopyFromScreen`，跟 `ScreenCaptureService.Capture` 一样只做人工验证。

### `Program.cs`（修改）

自动保存的回调从直接调用 `ScreenCaptureService.Capture` 改为通过一个 `ScreenshotDeduplicator` 实例调用 `CaptureAndSaveIfNotDuplicate`：

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

手动热键的回调（`hotkeyManager.HotkeyPressed`）保持不变，继续直接调用 `ScreenCaptureService.Capture(config.SaveDirectory)`。

## 明确排除的范围（YAGNI）

- 不做相似度/感知哈希之类的模糊比对，只做逐字节完全相等
- 不跟保存目录里的历史文件比较，只跟内存里记住的"上一张"比较
- 手动热键截图不参与去重，永远保存
- 不为"丢弃重复截图"这个事件写任何日志

## 测试计划

- 自动化：`ScreenshotDeduplicator.IsDuplicate`/`Remember` 用手工构造的 `byte[]` 单测覆盖：没有 `Remember` 过任何内容时 `IsDuplicate` 返回 false；`Remember` 之后传入相同字节 `IsDuplicate` 返回 true；传入不同字节（含仅长度不同）`IsDuplicate` 返回 false；多次 `Remember` 之后 `IsDuplicate` 始终反映最近一次 `Remember` 的内容。"关闭再打开"场景等价于——两次不同的 `Toggle()` 周期之间不清空 `ScreenshotDeduplicator` 实例，所以不需要专门测试这个，因为 `IsDuplicate`/`Remember` 本身根本不知道开关状态的存在
- 人工验证：开启自动保存后，保持屏幕内容不变，确认间隔时间内没有产生新的重复文件；改变一下屏幕内容（比如移动一下窗口），确认下一次自动保存正常产生新文件
