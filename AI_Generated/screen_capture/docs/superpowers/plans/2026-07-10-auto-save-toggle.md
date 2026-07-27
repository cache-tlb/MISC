# 定时自动保存开关 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增第二个可配置全局热键（默认 Win+Shift+X），按下后切换"定时自动保存截图"开关；开启状态下每 N 秒（默认 20，可配置）自动截一张图，关闭状态下不做任何事。

**Architecture:** 复用现有 `HotkeyManager` 类（不改动它），在 `Program.cs` 中再实例化一个 `HotkeyManager` 专门处理开关热键，因为 Win32 `RegisterHotKey` 的作用域是"窗口句柄级别"而不是"进程级别"，两个独立实例互不冲突。新增 `AutoSaveController` 类，单一职责是管理"开关状态 + 定时器"，通过构造函数传入的 `Action` 回调触发截图，不直接依赖 `ScreenCaptureService`。

**Tech Stack:** 沿用现有 .NET 8 / WinForms / xUnit 项目（`d:/tmp/screen_capture`），无新增外部依赖，新用到 `System.Windows.Forms.Timer`。

## Global Constraints

- 配置文件新增字段：`ToggleAutoSaveHotkey`（string，默认 `"Win+Shift+X"`，解析规则与现有 `Hotkey` 字段完全一致，复用 `HotkeyManager.TryParse`）、`AutoSaveIntervalSeconds`（int，默认 `20`）
- `AutoSaveIntervalSeconds` ≤0 视为非法值：`Logger.Error` 记录一条日志，回退到默认值 20，不弹窗
- 全程静默：开关切换、每次自动截图成功或失败，都不弹窗、不提示音、不写剪贴板；唯一的例外是"热键注册失败"这一种情况沿用已有的托盘气泡通知（与截图热键处理方式一致，是已批准的既有例外，不是本次新增行为）
- 开启开关的瞬间立即截一张，之后每 N 秒一张；关闭立即停止，不再产生新截图
- 程序重启后开关状态不持久化，每次启动默认"关"
- 自动保存截取范围与手动热键完全一致：截取鼠标当前所在的单个显示器，直接复用 `ScreenCaptureService.Capture`，不新增截图逻辑
- 若两个热键配置成同一个组合，后注册的那个会自然失败并走现有的"热键被占用"日志/气泡流程，不需要专门检测

---

## File Structure

```
src/ScreenCaptureDaemon/
  AppConfig.cs             (修改：新增 ToggleAutoSaveHotkey、AutoSaveIntervalSeconds 字段 + 校验)
  AutoSaveController.cs    (新建：开关状态 + 定时器)
  Program.cs                 (修改：接线第二个 HotkeyManager + AutoSaveController)
  appsettings.json          (修改：新增两个示例字段)
tests/ScreenCaptureDaemon.Tests/
  ScreenCaptureDaemon.Tests.csproj  (修改：新增 UseWindowsForms，AutoSaveController 依赖 System.Windows.Forms.Timer)
  AppConfigTests.cs                  (修改：新增用例)
  AutoSaveControllerTests.cs        (新建：Toggle 行为用例)
```

---

### Task 1: `AppConfig` — 新增 `ToggleAutoSaveHotkey` / `AutoSaveIntervalSeconds` 字段

**Files:**
- Modify: `src/ScreenCaptureDaemon/AppConfig.cs`
- Modify: `src/ScreenCaptureDaemon/appsettings.json`
- Modify: `tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs`

**Interfaces:**
- Produces: `AppConfig.ToggleAutoSaveHotkey`（string，默认 `"Win+Shift+X"`）、`AppConfig.AutoSaveIntervalSeconds`（int，默认 `20`，非正数时 `Load` 内部回退为 20 并记日志）

- [ ] **Step 1: 写失败的测试**

在 `tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs` 的 `AppConfigTests` 类中，`Load_ReadsHotkeyFromFile` 方法之后、类的结尾 `}` 之前，新增：

```csharp
    [Fact]
    public void Load_UsesDefaultToggleAutoSaveHotkey_WhenFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Win+Shift+X", config.ToggleAutoSaveHotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsToggleAutoSaveHotkeyFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"ToggleAutoSaveHotkey\": \"Ctrl+Alt+X\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Ctrl+Alt+X", config.ToggleAutoSaveHotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_UsesDefaultAutoSaveIntervalSeconds_WhenFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal(20, config.AutoSaveIntervalSeconds);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsAutoSaveIntervalSecondsFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"AutoSaveIntervalSeconds\": 45}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal(45, config.AutoSaveIntervalSeconds);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void Load_FallsBackToDefaultInterval_WhenAutoSaveIntervalSecondsIsNotPositive(int invalidInterval)
    {
        var configPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(configPath, $"{{\"AutoSaveIntervalSeconds\": {invalidInterval}}}");

        var logPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".log");
        var originalLogPath = Logger.LogPath;
        Logger.LogPath = logPath;

        try
        {
            var config = AppConfig.Load(configPath);

            Assert.Equal(20, config.AutoSaveIntervalSeconds);
            Assert.True(File.Exists(logPath));
            Assert.Contains("AutoSaveIntervalSeconds", File.ReadAllText(logPath));
        }
        finally
        {
            Logger.LogPath = originalLogPath;
            File.Delete(configPath);
            if (File.Exists(logPath)) File.Delete(logPath);
        }
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter AppConfigTests`
Expected: FAIL — 新增的 6 个测试执行（4 个 Fact + 1 个 Theory 的 2 组数据）失败（`AppConfig` 还没有 `ToggleAutoSaveHotkey`/`AutoSaveIntervalSeconds` 属性），其余既有测试仍然通过

- [ ] **Step 3: 实现字段与校验逻辑**

将 `src/ScreenCaptureDaemon/AppConfig.cs` 中 `Hotkey` 属性之后新增两个属性：

```csharp
    public string ToggleAutoSaveHotkey { get; set; } = "Win+Shift+X";

    public int AutoSaveIntervalSeconds { get; set; } = 20;
```

将 `Load` 方法里 `try` 块的内容从：

```csharp
        try
        {
            var json = File.ReadAllText(path);
            var loaded = JsonSerializer.Deserialize<AppConfig>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return loaded ?? new AppConfig();
        }
```

改为：

```csharp
        try
        {
            var json = File.ReadAllText(path);
            var loaded = JsonSerializer.Deserialize<AppConfig>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            if (loaded == null)
            {
                return new AppConfig();
            }

            if (loaded.AutoSaveIntervalSeconds <= 0)
            {
                Logger.Error($"AutoSaveIntervalSeconds 配置值 {loaded.AutoSaveIntervalSeconds} 无效，回退到默认值 20");
                loaded.AutoSaveIntervalSeconds = 20;
            }

            return loaded;
        }
```

`catch (JsonException ex) { ... }` 块保持不变。

- [ ] **Step 4: 更新示例配置文件**

`src/ScreenCaptureDaemon/appsettings.json` 改为：

```json
{
  "SaveDirectory": "F:\\Screenshots",
  "Port": 8080,
  "Hotkey": "Win+Shift+Z",
  "ToggleAutoSaveHotkey": "Win+Shift+X",
  "AutoSaveIntervalSeconds": 20
}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `dotnet test --filter AppConfigTests`
Expected: `Passed!` — 全部 AppConfigTests 通过（既有 8 个 + 本任务新增 6 个执行 = 14 个）

- [ ] **Step 6: Commit**

```bash
git add src/ScreenCaptureDaemon/AppConfig.cs src/ScreenCaptureDaemon/appsettings.json tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs
git commit -m "feat: add ToggleAutoSaveHotkey and AutoSaveIntervalSeconds config fields"
```

---

### Task 2: `AutoSaveController` — 开关状态 + 定时器

**Files:**
- Create: `src/ScreenCaptureDaemon/AutoSaveController.cs`
- Modify: `tests/ScreenCaptureDaemon.Tests/ScreenCaptureDaemon.Tests.csproj`
- Create: `tests/ScreenCaptureDaemon.Tests/AutoSaveControllerTests.cs`

**Interfaces:**
- Consumes: 无（构造函数接收 `int intervalSeconds` 和 `Action onTick`，不依赖任何项目内其他类型）
- Produces: `sealed class AutoSaveController : IDisposable { bool Enabled { get; }; void Toggle(); void Dispose(); }`

**Note:** `AutoSaveController` 内部使用 `System.Windows.Forms.Timer`。`Toggle()` 方法本身的状态切换和"开启瞬间立即触发一次回调"是可以自动化测试的纯行为（不依赖 Timer 真的 Tick，也不依赖挂钟时间）；但"每 N 秒真的会再触发一次"这种依赖真实消息循环和挂钟时间的行为，与项目里其他 Win32/WinForms 组件一样，只做人工验证（放在 Task 3）。

- [ ] **Step 1: 给测试项目启用 WinForms 引用**

`AutoSaveController` 会在测试进程里真正构造 `System.Windows.Forms.Timer`，测试项目需要能在运行时解析 `Microsoft.WindowsDesktop.App` 共享框架（与 Task 5 给 `HttpServerHostTests` 补 `FrameworkReference` 是同一类问题）。在 `tests/ScreenCaptureDaemon.Tests/ScreenCaptureDaemon.Tests.csproj` 的 `<PropertyGroup>` 中新增一行：

```xml
    <UseWindowsForms>true</UseWindowsForms>
```

即整个 `<PropertyGroup>` 变为：

```xml
  <PropertyGroup>
    <TargetFramework>net8.0-windows</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <IsPackable>false</IsPackable>
    <UseWindowsForms>true</UseWindowsForms>
  </PropertyGroup>
```

- [ ] **Step 2: 写失败的测试**

`tests/ScreenCaptureDaemon.Tests/AutoSaveControllerTests.cs`（新建文件）：

```csharp
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class AutoSaveControllerTests
{
    [Fact]
    public void Toggle_InvokesCallbackImmediately_WhenTurningOn()
    {
        var callCount = 0;
        using var controller = new AutoSaveController(20, () => callCount++);

        controller.Toggle();

        Assert.Equal(1, callCount);
        Assert.True(controller.Enabled);
    }

    [Fact]
    public void Toggle_DoesNotInvokeCallback_WhenTurningOff()
    {
        var callCount = 0;
        using var controller = new AutoSaveController(20, () => callCount++);

        controller.Toggle();
        controller.Toggle();

        Assert.Equal(1, callCount);
        Assert.False(controller.Enabled);
    }

    [Fact]
    public void Toggle_InvokesCallbackAgain_WhenTurnedOnASecondTime()
    {
        var callCount = 0;
        using var controller = new AutoSaveController(20, () => callCount++);

        controller.Toggle();
        controller.Toggle();
        controller.Toggle();

        Assert.Equal(2, callCount);
        Assert.True(controller.Enabled);
    }
}
```

- [ ] **Step 3: 运行测试确认失败**

Run: `dotnet test --filter AutoSaveControllerTests`
Expected: FAIL（编译错误：找不到类型 `AutoSaveController`）

- [ ] **Step 4: 实现 `AutoSaveController`**

`src/ScreenCaptureDaemon/AutoSaveController.cs`:

```csharp
using System.Windows.Forms;

namespace ScreenCaptureDaemon;

public sealed class AutoSaveController : IDisposable
{
    private readonly Timer _timer;
    private readonly Action _onTick;
    private bool _enabled;

    public AutoSaveController(int intervalSeconds, Action onTick)
    {
        _onTick = onTick;
        _timer = new Timer { Interval = intervalSeconds * 1000 };
        _timer.Tick += (_, _) => _onTick();
    }

    public bool Enabled => _enabled;

    public void Toggle()
    {
        _enabled = !_enabled;

        if (_enabled)
        {
            _onTick();
            _timer.Start();
        }
        else
        {
            _timer.Stop();
        }
    }

    public void Dispose()
    {
        _timer.Stop();
        _timer.Dispose();
    }
}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `dotnet test --filter AutoSaveControllerTests`
Expected: `Passed!` — 3 个测试全部通过

- [ ] **Step 6: 运行完整测试套件确认无回归**

Run: `dotnet test`
Expected: 全部通过（既有 27 个 + Task 1 新增 6 个 + 本任务新增 3 个 = 36 个）

- [ ] **Step 7: Commit**

```bash
git add src/ScreenCaptureDaemon/AutoSaveController.cs tests/ScreenCaptureDaemon.Tests/ScreenCaptureDaemon.Tests.csproj tests/ScreenCaptureDaemon.Tests/AutoSaveControllerTests.cs
git commit -m "feat: add AutoSaveController for the periodic auto-save toggle"
```

---

### Task 3: `Program.cs` — 接线第二个热键 + `AutoSaveController` + 端到端验证

**Files:**
- Modify: `src/ScreenCaptureDaemon/Program.cs`

**Interfaces:**
- Consumes: `AppConfig.ToggleAutoSaveHotkey`/`AppConfig.AutoSaveIntervalSeconds`（Task 1）、`AutoSaveController`（Task 2）、`HotkeyManager`（已有，不改动，直接再实例化一个）

- [ ] **Step 1: 接线 `Program.cs`**

将 `src/ScreenCaptureDaemon/Program.cs` 整个文件替换为：

```csharp
using System.Windows.Forms;

namespace ScreenCaptureDaemon;

internal static class Program
{
    private static Mutex? _mutex;

    [STAThread]
    private static void Main()
    {
        _mutex = new Mutex(true, "Local\\ScreenCaptureDaemon_SingleInstance", out var createdNew);
        if (!createdNew)
        {
            return;
        }

        Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        var config = AppConfig.Load(Path.Combine(AppContext.BaseDirectory, "appsettings.json"));

        var httpServer = new HttpServerHost();
        try
        {
            httpServer.StartAsync(config.SaveDirectory, config.Port).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Logger.Error($"HTTP 服务启动失败: {ex.Message}");
        }

        var hotkeyManager = new HotkeyManager();
        hotkeyManager.HotkeyPressed += () =>
        {
            try
            {
                ScreenCaptureService.Capture(config.SaveDirectory);
            }
            catch (Exception ex)
            {
                Logger.Error($"截图失败: {ex.Message}");
            }
        };

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

        var toggleHotkeyManager = new HotkeyManager();
        toggleHotkeyManager.HotkeyPressed += () => autoSaveController.Toggle();

        var trayIcon = new TrayIconManager(config.SaveDirectory, Application.Exit);

        var hotkeyText = config.Hotkey;
        if (!HotkeyManager.TryParse(hotkeyText, out var modifiers, out var vk))
        {
            Logger.Error($"热键配置 \"{config.Hotkey}\" 无效，回退到默认热键 Win+Shift+Z");
            hotkeyText = "Win+Shift+Z";
            HotkeyManager.TryParse(hotkeyText, out modifiers, out vk);
        }

        if (!hotkeyManager.Register(modifiers, vk))
        {
            var message = $"热键 {hotkeyText} 注册失败，可能已被其他程序占用。";
            Logger.Error(message);
            trayIcon.ShowBalloon("截图守护进程", message);
        }

        var toggleHotkeyText = config.ToggleAutoSaveHotkey;
        if (!HotkeyManager.TryParse(toggleHotkeyText, out var toggleModifiers, out var toggleVk))
        {
            Logger.Error($"热键配置 \"{config.ToggleAutoSaveHotkey}\" 无效，回退到默认热键 Win+Shift+X");
            toggleHotkeyText = "Win+Shift+X";
            HotkeyManager.TryParse(toggleHotkeyText, out toggleModifiers, out toggleVk);
        }

        if (!toggleHotkeyManager.Register(toggleModifiers, toggleVk))
        {
            var message = $"热键 {toggleHotkeyText} 注册失败，可能已被其他程序占用。";
            Logger.Error(message);
            trayIcon.ShowBalloon("截图守护进程", message);
        }

        Application.ApplicationExit += (_, _) =>
        {
            hotkeyManager.Dispose();
            toggleHotkeyManager.Dispose();
            autoSaveController.Dispose();
            trayIcon.Dispose();
            httpServer.DisposeAsync().AsTask().GetAwaiter().GetResult();
            _mutex?.ReleaseMutex();
        };

        Application.Run();
    }
}
```

- [ ] **Step 2: 构建并运行完整测试套件**

Run: `dotnet build`
Expected: `Build succeeded.`，无编译错误

Run: `dotnet test`
Expected: 全部通过（36 个测试，数量与 Task 2 结束时一致，本任务不新增测试用例——`Program.cs` 的接线行为是人工验证）

- [ ] **Step 3: 人工验证**

1. 在 `src/ScreenCaptureDaemon/bin/Debug/net8.0-windows/appsettings.json`（构建输出目录的副本，不是源码目录里的）中把 `AutoSaveIntervalSeconds` 改成一个方便观察的小值（比如 `5`），运行 `ScreenCaptureDaemon.exe`
2. 按下 Win+Shift+X（或配置的组合），确认配置的截图目录里**立即**出现一张新截图
3. 等待 5 秒以上，确认目录里出现**第二张**截图（间隔生效）
4. 再按一次 Win+Shift+X，确认之后不再有新截图产生（开关已关闭）
5. 确认原有的 Win+Shift+Z 截图热键在整个过程中依然正常工作，两个热键互不干扰
6. 把 `AutoSaveIntervalSeconds` 改成 `0`，重启程序，确认 `log.txt` 出现包含 "AutoSaveIntervalSeconds" 的日志行，且开关热键仍然可以正常切换（回退到 20 秒后依然有效）
7. 全程确认没有任何弹窗、提示音——包括开关切换本身和每一次自动截图

Expected: 以上 7 项均符合预期

- [ ] **Step 4: Commit**

```bash
git add src/ScreenCaptureDaemon/Program.cs
git commit -m "feat: wire second hotkey and AutoSaveController for periodic auto-save"
```

---

## Post-Plan Notes

- 两处热键解析/注册/失败处理的代码目前是重复的（各约 8 行）。如果将来再加第三个热键，值得把这段逻辑提取成一个小的辅助方法（比如 `RegisterHotkeyOrFallback(string configuredText, string defaultText, HotkeyManager manager, TrayIconManager trayIcon)`），但目前只有两处重复，按 YAGNI 原则不在本计划中做这个抽象
- 如果用户希望"重启后记住开关状态"，需要另起一次设计讨论（涉及是否写回配置文件或单独的状态文件），不在本计划范围内
