# 热键可配置 设计文档

日期：2026-07-09

## 背景与目标

原实现中触发截图的全局热键 Win+Shift+C 写死在 `HotkeyManager` 内部。需要改为可通过配置文件设置，默认热键改为 Win+Shift+Z。

## 改动范围

`AppConfig`、`HotkeyManager`、`Program.cs`、`TrayIconManager`（气泡通知文案）。其余组件（`Logger`、`ScreenCaptureService`、`HttpServerHost`）不受影响。

## 配置文件

`appsettings.json` 新增 `Hotkey` 字段（string），默认 `"Win+Shift+Z"`：

```json
{
  "SaveDirectory": "F:\\Screenshots",
  "Port": 8080,
  "Hotkey": "Win+Shift+Z"
}
```

配置文件缺失该字段时，沿用 `AppConfig` 现有的"缺字段用默认值"模式（`Hotkey` 属性初始化为 `"Win+Shift+Z"`），不需要额外处理逻辑。

## 热键字符串格式与解析

在 `HotkeyManager` 中新增纯函数：

```csharp
public static bool TryParse(string text, out uint modifiers, out uint vk)
```

规则：
- 按 `+` 分割 token，忽略大小写和首尾空白
- 修饰键 token：`Win`、`Ctrl`、`Alt`、`Shift`（对应 Win32 `MOD_WIN`/`MOD_CONTROL`/`MOD_ALT`/`MOD_SHIFT`），可任意组合，**至少需要一个**（不允许无修饰键的全局热键，避免劫持正常按键输入）
- 主键 token：**必须恰好一个**，且仅支持单个字符 `A-Z` 或 `0-9`（映射为对应的 Win32 虚拟键码，如 `'Z'` → `0x5A`，`'5'` → `0x35`）
- 以下情况均返回 `false`（不抛异常）：空字符串/null、无法识别的修饰键名、没有主键、主键不止一个、主键不是单个字母或数字
- 解析成功时 `modifiers`/`vk` 输出有效值；失败时输出值不做保证（调用方不应使用）

这是纯函数，不依赖任何 Win32 状态，可以像 `ScreenCaptureService.BuildFileName` 一样写自动化单元测试覆盖。

## HotkeyManager.Register 签名变化

```csharp
// 原来
public bool Register()

// 改为
public bool Register(uint modifiers, uint vk)
```

内部不再写死 `ModShift | ModWin` 和 `VkC`，改用调用方传入的值调用 `RegisterHotKey`。

## Program.cs 接线

启动时：
1. 调用 `HotkeyManager.TryParse(config.Hotkey, out var modifiers, out var vk)`
2. 若解析失败：`Logger.Error("热键配置 \"{config.Hotkey}\" 无效，回退到默认热键 Win+Shift+Z")`，并改用默认热键（`Win+Shift+Z` 对应的 modifiers/vk，作为常量保留在 `Program.cs` 或 `HotkeyManager` 中）继续注册
3. 用最终生效的 modifiers/vk 调用 `hotkeyManager.Register(modifiers, vk)`
4. 若 `Register` 返回 `false`（如被其他程序占用），沿用现有逻辑：写日志 + 托盘气泡通知一次；通知文案中的热键名改为**实际尝试注册的那个热键**（配置有效时是用户配置的热键，配置无效回退时是 Win+Shift+Z），而不是固定写死 "Win+Shift+C"

## 明确排除的范围（YAGNI）

- 不支持功能键 F1-F12 或其他特殊键（方向键、Tab 等）作为主键
- 不支持运行时热更新配置文件（改配置后仍需重启程序生效，与现有 `SaveDirectory`/`Port` 行为一致）
- 不提供图形化热键设置界面，仍然是手改 JSON 文件

## 测试计划

- `HotkeyManager.TryParse` 自动化单测：合法组合（含大小写混用）、多修饰键组合、每类失败场景（空串、无修饰键、无主键、多主键、主键非法字符、未知修饰键名）
- 手动验证（沿用 Task 8 已建立的验证方式）：配置一个自定义热键（如 `Ctrl+Alt+9`）确认能生效；配置一个非法字符串确认回退到 Win+Shift+Z 并写日志
