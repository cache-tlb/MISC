# 热键可配置 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把写死在 `HotkeyManager` 里的 Win+Shift+C 全局热键改成可通过 `appsettings.json` 配置，默认值改为 Win+Shift+Z，解析失败时静默回退默认值并写日志。

**Architecture:** `AppConfig` 新增 `Hotkey` 字符串字段；`HotkeyManager` 新增纯函数解析器 `TryParse`（字符串 → 修饰键位掩码 + 虚拟键码），`Register` 方法签名从无参改为接收解析结果；`Program.cs` 在启动时解析配置、失败则回退默认值再注册，注册失败的日志/气泡通知文案里带上实际尝试的热键名。

**Tech Stack:** 沿用现有 .NET 8 / WinForms / xUnit 项目（`d:/tmp/screen_capture`），无新增依赖。

## Global Constraints

- 配置文件字段：`Hotkey`（string），默认值 `"Win+Shift+Z"`，缺失时沿用 `AppConfig` 现有"缺字段用默认值"模式
- 热键字符串格式：`+` 分隔的 token，忽略大小写；修饰键 token 为 `Win`/`Ctrl`/`Alt`/`Shift` 的任意组合，**至少一个**；主键 token **必须恰好一个**，仅支持单个 `A-Z` 或 `0-9`
- 解析失败（空串/null、未知修饰键名、无主键、多个主键、主键非单字符字母数字）一律返回 `false`，不抛异常
- 解析失败时的处理：`Logger.Error` 记录一条包含原始配置值的日志，回退到默认热键 `Win+Shift+Z` 继续注册，不弹窗
- 热键注册失败（如被占用）的日志/托盘气泡通知文案中的热键名，必须是**实际尝试注册的那个热键**（配置有效时是用户配置的值，回退时是 `Win+Shift+Z`），不是写死的字符串
- 不支持功能键（F1-F12）或多字符键名作为主键；不支持运行时热更新（改配置需重启程序）

---

## File Structure

```
src/ScreenCaptureDaemon/
  AppConfig.cs          (修改：新增 Hotkey 属性)
  HotkeyManager.cs       (修改：新增 TryParse 静态方法，Register 签名变化)
  Program.cs              (修改：接线解析 + 回退 + 注册 + 文案)
  appsettings.json       (修改：新增 Hotkey 示例字段)
tests/ScreenCaptureDaemon.Tests/
  AppConfigTests.cs        (修改：新增 Hotkey 相关用例)
  HotkeyManagerTests.cs    (新建：TryParse 的用例)
```

---

### Task 1: `AppConfig` — 新增 `Hotkey` 配置字段

**Files:**
- Modify: `src/ScreenCaptureDaemon/AppConfig.cs`
- Modify: `src/ScreenCaptureDaemon/appsettings.json`
- Modify: `tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs`

**Interfaces:**
- Produces: `AppConfig.Hotkey` — `string`，默认值 `"Win+Shift+Z"`

- [ ] **Step 1: 写失败的测试**

在 `tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs` 的 `AppConfigTests` 类中新增两个测试方法（放在现有 `Load_ReturnsDefaultsAndLogs_WhenJsonIsMalformed` 方法之后，`}` 之前）：

```csharp
    [Fact]
    public void Load_UsesDefaultHotkey_WhenHotkeyFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Win+Shift+Z", config.Hotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsHotkeyFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"Hotkey\": \"Ctrl+Alt+9\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Ctrl+Alt+9", config.Hotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter AppConfigTests`
Expected: FAIL — `Load_UsesDefaultHotkey_WhenHotkeyFieldMissing` 和 `Load_ReadsHotkeyFromFile` 两个新测试失败（`AppConfig` 还没有 `Hotkey` 属性，`config.Hotkey` 编译不通过 / 或断言失败），其余既有测试仍然通过

- [ ] **Step 3: 实现 `AppConfig.Hotkey`**

在 `src/ScreenCaptureDaemon/AppConfig.cs` 中，`Port` 属性之后新增：

```csharp
    public string Hotkey { get; set; } = "Win+Shift+Z";
```

完整文件内容此时应为：

```csharp
using System.Text.Json;

namespace ScreenCaptureDaemon;

public sealed class AppConfig
{
    public string SaveDirectory { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "Screenshots");

    public int Port { get; set; } = 8080;

    public string Hotkey { get; set; } = "Win+Shift+Z";

    public static AppConfig Load(string path)
    {
        if (!File.Exists(path))
        {
            return new AppConfig();
        }

        try
        {
            var json = File.ReadAllText(path);
            var loaded = JsonSerializer.Deserialize<AppConfig>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return loaded ?? new AppConfig();
        }
        catch (JsonException ex)
        {
            Logger.Error("配置文件解析失败，使用默认配置: " + ex.Message);
            return new AppConfig();
        }
    }
}
```

- [ ] **Step 4: 更新示例配置文件**

`src/ScreenCaptureDaemon/appsettings.json` 改为：

```json
{
  "SaveDirectory": "F:\\Screenshots",
  "Port": 8080,
  "Hotkey": "Win+Shift+Z"
}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `dotnet test --filter AppConfigTests`
Expected: `Passed!` — 6 个测试全部通过（4 个既有 + 2 个新增）

- [ ] **Step 6: Commit**

```bash
git add src/ScreenCaptureDaemon/AppConfig.cs src/ScreenCaptureDaemon/appsettings.json tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs
git commit -m "feat: add configurable Hotkey field to AppConfig"
```

---

### Task 2: `HotkeyManager.TryParse` — 热键字符串解析器（纯函数）

**Files:**
- Modify: `src/ScreenCaptureDaemon/HotkeyManager.cs`
- Create: `tests/ScreenCaptureDaemon.Tests/HotkeyManagerTests.cs`

**Interfaces:**
- Produces: `static bool HotkeyManager.TryParse(string? text, out uint modifiers, out uint vk)`
- `Register()` 方法本任务**不改动**（签名变化和调用方接线放在 Task 3，避免这一步破坏编译）

- [ ] **Step 1: 写失败的测试**

`tests/ScreenCaptureDaemon.Tests/HotkeyManagerTests.cs`（新建文件）：

```csharp
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class HotkeyManagerTests
{
    [Theory]
    [InlineData("Win+Shift+Z", 0x0008u | 0x0004u, 0x5Au)]
    [InlineData("win+shift+z", 0x0008u | 0x0004u, 0x5Au)]
    [InlineData("Ctrl+Alt+9", 0x0002u | 0x0001u, 0x39u)]
    [InlineData("Shift+A", 0x0004u, 0x41u)]
    public void TryParse_ParsesValidHotkeyStrings(string text, uint expectedModifiers, uint expectedVk)
    {
        var result = HotkeyManager.TryParse(text, out var modifiers, out var vk);

        Assert.True(result);
        Assert.Equal(expectedModifiers, modifiers);
        Assert.Equal(expectedVk, vk);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("Z")]
    [InlineData("Win+Shift")]
    [InlineData("Win+Shift+Z+X")]
    [InlineData("Win+Shift+ZZ")]
    [InlineData("Banana+Z")]
    [InlineData("Win+Shift+F1")]
    public void TryParse_RejectsInvalidHotkeyStrings(string? text)
    {
        var result = HotkeyManager.TryParse(text, out _, out _);

        Assert.False(result);
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter HotkeyManagerTests`
Expected: FAIL（编译错误：`HotkeyManager` 还没有 `TryParse` 方法）

- [ ] **Step 3: 实现 `TryParse`**

将 `src/ScreenCaptureDaemon/HotkeyManager.cs` 顶部的修饰键常量区替换为（新增 `ModAlt`、`ModControl`）：

```csharp
    private const int WM_HOTKEY = 0x0312;
    private const int HotkeyId = 1;
    private const uint ModAlt = 0x0001;
    private const uint ModControl = 0x0002;
    private const uint ModShift = 0x0004;
    private const uint ModWin = 0x0008;
```

在 `UnregisterHotKey` 的 `[DllImport]` 声明之后、`Register` 方法之前，新增：

```csharp
    public static bool TryParse(string? text, out uint modifiers, out uint vk)
    {
        modifiers = 0;
        vk = 0;

        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var tokens = text.Split('+', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);

        uint parsedModifiers = 0;
        char? mainKey = null;

        foreach (var token in tokens)
        {
            switch (token.ToUpperInvariant())
            {
                case "WIN":
                    parsedModifiers |= ModWin;
                    break;
                case "CTRL":
                    parsedModifiers |= ModControl;
                    break;
                case "ALT":
                    parsedModifiers |= ModAlt;
                    break;
                case "SHIFT":
                    parsedModifiers |= ModShift;
                    break;
                default:
                    if (token.Length == 1 && (char.IsAsciiLetter(token[0]) || char.IsAsciiDigit(token[0])))
                    {
                        if (mainKey != null)
                        {
                            return false;
                        }

                        mainKey = char.ToUpperInvariant(token[0]);
                    }
                    else
                    {
                        return false;
                    }

                    break;
            }
        }

        if (parsedModifiers == 0 || mainKey == null)
        {
            return false;
        }

        modifiers = parsedModifiers;
        vk = mainKey.Value;
        return true;
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `dotnet test --filter HotkeyManagerTests`
Expected: `Passed!` — 13 个测试全部通过（4 个合法用例 + 9 个非法用例）

- [ ] **Step 5: 运行完整测试套件确认无回归**

Run: `dotnet test`
Expected: 全部通过（12 个既有 + 2 个 Task 1 新增 + 13 个本任务新增 = 27 个）

- [ ] **Step 6: Commit**

```bash
git add src/ScreenCaptureDaemon/HotkeyManager.cs tests/ScreenCaptureDaemon.Tests/HotkeyManagerTests.cs
git commit -m "feat: add HotkeyManager.TryParse for configurable hotkey strings"
```

---

### Task 3: `Register` 签名变化 + `Program.cs` 接线 + 端到端验证

**Files:**
- Modify: `src/ScreenCaptureDaemon/HotkeyManager.cs`
- Modify: `src/ScreenCaptureDaemon/Program.cs`

**Interfaces:**
- Consumes: `AppConfig.Hotkey`（Task 1）、`HotkeyManager.TryParse`（Task 2）
- Produces: `HotkeyManager.Register(uint modifiers, uint vk)`（替换原来的无参 `Register()`）

- [ ] **Step 1: 修改 `Register` 签名**

在 `src/ScreenCaptureDaemon/HotkeyManager.cs` 中，把：

```csharp
    public bool Register()
    {
        var cp = new CreateParams { Parent = (IntPtr)(-3) }; // HWND_MESSAGE：消息专用窗口，不可见
        CreateHandle(cp);

        _registered = RegisterHotKey(Handle, HotkeyId, ModWin | ModShift, VkC);
        return _registered;
    }
```

改为：

```csharp
    public bool Register(uint modifiers, uint vk)
    {
        var cp = new CreateParams { Parent = (IntPtr)(-3) }; // HWND_MESSAGE：消息专用窗口，不可见
        CreateHandle(cp);

        _registered = RegisterHotKey(Handle, HotkeyId, modifiers, vk);
        return _registered;
    }
```

（此时 `Program.cs` 里对 `Register()` 的旧调用会编译失败，属于预期中间状态，下一步一并修好。）

- [ ] **Step 2: 接线 `Program.cs`**

将 `src/ScreenCaptureDaemon/Program.cs` 中原来的：

```csharp
        var trayIcon = new TrayIconManager(config.SaveDirectory, Application.Exit);

        if (!hotkeyManager.Register())
        {
            var message = "热键 Win+Shift+C 注册失败，可能已被其他程序占用。";
            Logger.Error(message);
            trayIcon.ShowBalloon("截图守护进程", message);
        }
```

改为：

```csharp
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
```

文件其余部分（`Main` 方法开头的 Mutex/配置加载/HTTP 启动/`hotkeyManager.HotkeyPressed` 订阅、`Application.ApplicationExit` 处理、`Application.Run()`）保持不变。

- [ ] **Step 3: 构建并运行完整测试套件**

Run: `dotnet build`
Expected: `Build succeeded.`，无编译错误

Run: `dotnet test`
Expected: 全部通过（27 个测试，数量与 Task 2 结束时一致，本任务不新增测试用例——`Register`/`Program.cs` 的行为验证是人工验证）

- [ ] **Step 4: 人工验证**

1. 在 `src/ScreenCaptureDaemon/bin/Debug/net8.0-windows/appsettings.json`（构建输出目录的副本，不是源码目录里的）中设置 `"Hotkey": "Ctrl+Alt+9"`，运行 `ScreenCaptureDaemon.exe`，按下 Ctrl+Alt+9，确认能触发截图（若开发机上 Ctrl+Alt+9 恰好也被占用，换一个当前机器上确认空闲的组合，比如 `Ctrl+Alt+8`，不影响验证目的）
2. 把 `Hotkey` 改成一个非法字符串（比如 `"Win+Banana"`），重启程序，确认 `log.txt` 出现"热键配置 ... 无效，回退到默认热键 Win+Shift+Z"，并且实际生效的是 Win+Shift+Z（若 Win+Shift+Z 在当前机器上空闲，可按下验证触发截图；若也被占用，确认托盘气泡通知文案里显示的是"热键 Win+Shift+Z 注册失败"而不是旧的"热键 Win+Shift+C"）
3. 删除构建输出目录里 `appsettings.json` 中的 `Hotkey` 字段（或整个文件都删掉，让 `AppConfig.Load` 走"文件不存在"分支），确认默认生效的是 Win+Shift+Z

Expected: 以上 3 项均符合预期

- [ ] **Step 5: Commit**

```bash
git add src/ScreenCaptureDaemon/HotkeyManager.cs src/ScreenCaptureDaemon/Program.cs
git commit -m "feat: wire configurable hotkey with fallback to Win+Shift+Z"
```

---

## Post-Plan Notes

- 若用户想要功能键（F1-F12）或组合键之外的更复杂绑定，需要另起一次设计讨论，不在本计划范围内
- 配置热更新（不重启生效）同样不在本计划范围内，与 `SaveDirectory`/`Port` 的现有行为保持一致
