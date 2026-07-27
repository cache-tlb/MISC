# 定时自动保存开关 设计文档

日期：2026-07-10

## 背景与目标

新增第二个全局热键（默认 Win+Shift+X，可配置），按下后切换"定时自动保存屏幕截图"开关。开启状态下每 N 秒（默认 20，可配置）自动保存一次截图；关闭状态下不做任何事。

## 改动范围

`AppConfig`（新增两个字段）、新建 `AutoSaveController`、`Program.cs`（接线第二个 `HotkeyManager` 实例 + `AutoSaveController`）。`HotkeyManager` 本身不改动。

## 配置文件

`appsettings.json` 新增两个字段：

```json
{
  "ToggleAutoSaveHotkey": "Win+Shift+X",
  "AutoSaveIntervalSeconds": 20
}
```

- `ToggleAutoSaveHotkey`（string）：格式和解析规则与现有 `Hotkey` 字段完全一致（复用 `HotkeyManager.TryParse`），默认 `"Win+Shift+X"`
- `AutoSaveIntervalSeconds`（int）：自动保存间隔秒数，默认 `20`；若配置值 ≤0，视为非法值，回退到默认 20 并写日志（`Logger.Error`），不弹窗，与现有"非法配置回退默认值"的模式一致

## 热键架构：复用 `HotkeyManager`，不修改它

Win32 `RegisterHotKey` 的作用域是"窗口句柄 + id"，不是进程级别的单例限制，所以两个独立的 `HotkeyManager` 实例（各自拥有独立的 `HWND_MESSAGE` 消息专用窗口）可以分别注册不同的组合键，互不冲突。因此：

- `Program.cs` 中会有两个 `HotkeyManager` 实例：现有的截图热键一个，新增的开关热键一个
- 两处都复用同样的"`TryParse` 解析配置 → 失败则记日志并回退默认值 → `Register(modifiers, vk)` → 失败则记日志 + 托盘气泡通知"流程。这段逻辑会有两份几乎相同的代码，但目前只有两个热键，重复量很小，暂不提取公共方法（YAGNI，避免过早抽象）
- 若用户把两个热键配置成同一个组合，第二个 `Register` 调用会因为组合键已被第一个占用而失败（Win32 层面对同一组合键的注册是系统级互斥的，与是否同进程无关），自然落入"热键被占用"的现有处理流程，不需要专门检测

## `AutoSaveController`（新建）

单一职责：管理"自动保存开关状态 + 定时器"，不关心具体怎么截图。

```csharp
public sealed class AutoSaveController : IDisposable
{
    // 构造函数接收：间隔秒数、一个无参 Action 回调（每次该触发截图时调用）
    // Toggle()：翻转内部开关状态
    //   - 翻转为"开"：立即调用一次回调，然后启动 System.Windows.Forms.Timer
    //   - 翻转为"关"：停止 Timer，不再调用回调
    // Timer 的 Tick 事件也调用同一个回调
    // Dispose()：停止并释放 Timer
}
```

回调本身（实际截图 + 失败处理）由 `Program.cs` 提供，复用截图热键已有的 `try { ScreenCaptureService.Capture(config.SaveDirectory) } catch (Exception ex) { Logger.Error(...) }` 模式，保持 `AutoSaveController` 不依赖 `ScreenCaptureService`。

## 行为约束（明确、不可协商）

- **全程静默**：开关切换、每次自动截图成功或失败，都不弹窗、不提示音、不写剪贴板。仅在"热键注册失败"这一种情况下沿用已有的托盘气泡通知（与截图热键的处理方式完全一致，这是已经批准的例外，不是本次新增的例外）
- **首次触发时机**：开启的瞬间立即保存一张，之后每 N 秒一张
- **重启后不保留状态**：程序重启后开关默认恢复为"关"，不做任何持久化
- **自动保存范围**：与手动热键完全一致，截取"鼠标当前所在的单个显示器"，直接复用 `ScreenCaptureService.Capture`，不新增截图逻辑

## 明确排除的范围（YAGNI）

- 不持久化开关状态到配置文件或其他存储
- 不提供开关状态的任何可视化指示（托盘图标变色、气泡通知等）
- 自动保存不做防抖/去重（如果用户截图目录被外部程序清空，下一次定时截图仍正常进行，不做特殊处理）
- 两个热键之间不做冲突预检测，冲突时依赖现有的注册失败处理流程

## 测试计划

- 自动化：无法为 `AutoSaveController` 的定时器行为写有意义的 xUnit 测试（依赖真实的 WinForms 消息循环和挂钟时间），与 `HotkeyManager.Register`/`ScreenCaptureService.Capture` 一样，只做人工验证
- 若 `AppConfig` 新增字段的默认值/回退逻辑本身是纯函数式的（读取配置 → 决定最终生效的 interval 值），可以像现有 `AppConfig`/`HotkeyManager.TryParse` 测试一样写自动化单测覆盖
- 人工验证：按开关热键确认开启后立即产生一张截图；等待 N 秒确认产生第二张；再按一次热键确认停止产生新截图；配置非法 interval（如 0 或 -5）确认回退到 20 并写日志
