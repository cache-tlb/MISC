# Windows 截图守护进程 + HTTP 服务 设计文档

日期：2026-07-09

## 背景与目标

需要一个 Windows 后台常驻工具：监听全局热键 Win+Shift+C，将当前鼠标所在显示器的屏幕内容截图保存到指定目录，同时以该目录为根启动一个局域网可访问的 HTTP 静态文件服务器，方便从其他设备浏览/下载截图。

## 技术栈

C# / .NET 8，WinForms 项目类型（无控制台窗口，托盘图标常驻）。

## 运行方式

带托盘图标的后台程序，手动启动（不注册开机自启动）。托盘右键菜单提供：
- 打开截图目录
- 退出

不注册为 Windows 服务 —— Windows 服务运行在会话 0，无法监听当前登录用户会话的全局热键，不满足本需求。

## 架构总览

单进程应用，主线程运行隐藏窗口 + Windows 消息循环（用于接收全局热键消息 `WM_HOTKEY`），同时异步启动 ASP.NET Core 静态文件服务器（`WebApplication.RunAsync()`，非阻塞）。

## 组件划分

| 组件 | 职责 |
|---|---|
| `Program.cs` | 入口：单实例互斥锁（`Mutex`）检查、加载配置、启动各组件 |
| `AppConfig` | 绑定 `appsettings.json`（`SaveDirectory`、`Port`） |
| `HotkeyManager` | 隐藏 `Form`/`NativeWindow` 调用 Win32 `RegisterHotKey` 注册 Win+Shift+C，在 `WndProc` 中拦截 `WM_HOTKEY` 并触发截图事件 |
| `ScreenCaptureService` | `Screen.FromPoint(Cursor.Position)` 定位鼠标当前所在显示器 → `Graphics.CopyFromScreen` 截取该显示器画面 → 保存为 PNG |
| `TrayIconManager` | `NotifyIcon` + 右键菜单（打开截图目录 / 退出） |
| HTTP 服务 | ASP.NET Core `WebApplication`，`UseStaticFiles` + `UseDirectoryBrowser`，物理根目录指向配置的保存目录 |

## 数据流

1. 用户按下 Win+Shift+C → 系统发送 `WM_HOTKEY` 消息 → `HotkeyManager` 捕获并触发截图事件
2. `ScreenCaptureService` 截取鼠标所在显示器画面 → 保存为 `{SaveDirectory}\screenshot_yyyyMMdd_HHmmss_fff.png`（若目录不存在则自动创建）
3. 局域网内任意设备访问 `http://<本机局域网IP>:<Port>/`，看到目录文件列表，可点击预览/下载任意截图文件

## 配置文件

程序目录下的 `appsettings.json`，示例：

```json
{
  "SaveDirectory": "F:\\Screenshots",
  "Port": 8080
}
```

- `SaveDirectory`：截图保存目录，同时作为 HTTP 服务的根目录。修改后需重启程序生效。
- `Port`：HTTP 服务监听端口，默认 8080。

## 访问范围

HTTP 服务监听 `0.0.0.0`，局域网内其他设备可通过本机 IP 访问。不做身份鉴权，默认信任局域网环境。

## 截图反馈策略

按需求，截图成功或失败**均不产生任何弹窗、提示音或剪贴板操作**，完全静默。

## 错误处理与日志

以下"启动期"关键错误会写入程序目录下的 `log.txt`（简单追加写日志，不弹窗）：

- 热键注册失败（如 Win+Shift+C 已被其他程序占用）
- HTTP 服务启动失败（如端口被占用）
- 保存目录无写入权限

托盘图标本身持续存在，作为程序仍在运行的直观标志。

## 单实例保证

启动时使用具名 `Mutex` 检查是否已有实例运行，若已存在则直接退出，避免重复注册热键或端口冲突。

## 明确排除的范围（YAGNI）

- 开机自启动
- 截图成功/失败提示（气泡通知、提示音、剪贴板复制）
- 自定义缩略图画廊页面（使用系统默认目录列表即可）
- HTTP 服务身份鉴权
- 多显示器拼接截图（只截鼠标所在的单个显示器）

## 测试计划

- 手动验证：按热键后检查目标目录是否生成正确命名的 PNG 文件，内容与鼠标所在显示器画面一致
- 多显示器场景：将鼠标切换到不同显示器后按热键，验证截取的是对应显示器内容
- HTTP 服务：本机及局域网内另一设备通过浏览器访问 `http://<IP>:<Port>/`，验证能看到并下载截图文件
- 异常场景：保存目录不存在时自动创建；端口被占用时程序不崩溃且写入 `log.txt`；重复启动第二个实例时能正确检测并退出
