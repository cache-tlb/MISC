# Windows 截图守护进程 + HTTP 服务 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个 C# .NET 8 WinForms 托盘程序，监听全局热键 Win+Shift+C 截取鼠标所在显示器画面并保存到配置目录，同时用 ASP.NET Core 静态文件服务在局域网内暴露该目录。

**Architecture:** 单进程应用。主线程运行 WinForms 消息循环（承载托盘图标 `NotifyIcon` 与消息专用窗口 `HotkeyManager`），异步启动的 ASP.NET Core `WebApplication` 提供静态文件 + 目录浏览。各组件之间只通过构造函数参数和事件耦合，互相独立、可单独测试。

**Tech Stack:** .NET 8 (`net8.0-windows`), WinForms, ASP.NET Core Minimal Hosting (`Microsoft.AspNetCore.App` FrameworkReference), xUnit for tests.

## Global Constraints

- 目标框架：`net8.0-windows`（已确认本机安装了 .NET 9 SDK 与 .NET 8 / 9 Windows Desktop + AspNetCore 运行时，可构建/运行 net8.0-windows）
- 无控制台窗口，`OutputType=WinExe`
- 配置文件路径：程序目录下 `appsettings.json`，字段为 `SaveDirectory`（string）、`Port`（int），默认端口 8080
- 截图文件名格式：`screenshot_{yyyyMMdd_HHmmss_fff}.png`
- 截图成功/失败**不产生任何弹窗、提示音或剪贴板操作**（规格明确要求静默）
- 启动期错误（热键注册失败 / HTTP 服务启动失败）写入程序目录下 `log.txt`，一行一条，格式 `[yyyy-MM-dd HH:mm:ss] message`
- 单实例保证：具名 `Mutex`，第二个实例启动后直接退出
- HTTP 服务监听 `0.0.0.0:{Port}`（局域网可访问），无身份鉴权
- 全局热键：Win+Shift+C（`MOD_WIN | MOD_SHIFT` + 虚拟键码 `0x43`）
- 截图范围：`Screen.FromPoint(Cursor.Position)` 所在的单个显示器

---

## File Structure

```
ScreenCaptureDaemon.sln
src/ScreenCaptureDaemon/
  ScreenCaptureDaemon.csproj
  Program.cs
  AppConfig.cs
  Logger.cs
  ScreenCaptureService.cs
  HotkeyManager.cs
  TrayIconManager.cs
  HttpServerHost.cs
  appsettings.json
tests/ScreenCaptureDaemon.Tests/
  ScreenCaptureDaemon.Tests.csproj
  AppConfigTests.cs
  LoggerTests.cs
  ScreenCaptureServiceTests.cs
  HttpServerHostTests.cs
.gitignore
```

---

### Task 1: 项目脚手架（git、解决方案、csproj、appsettings.json）

**Files:**
- Create: `.gitignore`
- Create: `ScreenCaptureDaemon.sln`
- Create: `src/ScreenCaptureDaemon/ScreenCaptureDaemon.csproj`
- Create: `src/ScreenCaptureDaemon/appsettings.json`
- Create: `tests/ScreenCaptureDaemon.Tests/ScreenCaptureDaemon.Tests.csproj`

**Interfaces:**
- Produces: 两个可构建项目 `ScreenCaptureDaemon`（WinExe，`net8.0-windows`）与 `ScreenCaptureDaemon.Tests`（xUnit，`net8.0-windows`），测试项目通过 `ProjectReference` 依赖主项目

- [ ] **Step 1: 初始化 git 仓库并写 `.gitignore`**

```bash
git init
```

`.gitignore`:

```gitignore
bin/
obj/
*.user
```

- [ ] **Step 2: 创建目录结构**

```bash
mkdir -p src/ScreenCaptureDaemon
mkdir -p tests/ScreenCaptureDaemon.Tests
```

- [ ] **Step 3: 写主项目 csproj**

`src/ScreenCaptureDaemon/ScreenCaptureDaemon.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>WinExe</OutputType>
    <TargetFramework>net8.0-windows</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <UseWindowsForms>true</UseWindowsForms>
    <RootNamespace>ScreenCaptureDaemon</RootNamespace>
    <AssemblyName>ScreenCaptureDaemon</AssemblyName>
  </PropertyGroup>

  <ItemGroup>
    <FrameworkReference Include="Microsoft.AspNetCore.App" />
  </ItemGroup>

  <ItemGroup>
    <InternalsVisibleTo Include="ScreenCaptureDaemon.Tests" />
  </ItemGroup>

  <ItemGroup>
    <None Update="appsettings.json">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
  </ItemGroup>

</Project>
```

- [ ] **Step 4: 写默认配置文件**

`src/ScreenCaptureDaemon/appsettings.json`:

```json
{
  "SaveDirectory": "F:\\Screenshots",
  "Port": 8080
}
```

- [ ] **Step 5: 写测试项目 csproj**

`tests/ScreenCaptureDaemon.Tests/ScreenCaptureDaemon.Tests.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0-windows</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <IsPackable>false</IsPackable>
  </PropertyGroup>

  <ItemGroup>
    <FrameworkReference Include="Microsoft.AspNetCore.App" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.11.1" />
    <PackageReference Include="xunit" Version="2.9.2" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.8.2" />
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="..\..\src\ScreenCaptureDaemon\ScreenCaptureDaemon.csproj" />
  </ItemGroup>

</Project>
```

- [ ] **Step 6: 创建解决方案文件并加入两个项目**

```bash
dotnet new sln -n ScreenCaptureDaemon
dotnet sln add src/ScreenCaptureDaemon/ScreenCaptureDaemon.csproj
dotnet sln add tests/ScreenCaptureDaemon.Tests/ScreenCaptureDaemon.Tests.csproj
```

- [ ] **Step 7: 放一个占位 `Program.cs` 使主项目能编译**

`src/ScreenCaptureDaemon/Program.cs`:

```csharp
namespace ScreenCaptureDaemon;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
    }
}
```

- [ ] **Step 8: 还原并构建，确认脚手架可用**

Run: `dotnet build`
Expected: `Build succeeded.`，两个项目都成功编译，无错误

- [ ] **Step 9: Commit**

```bash
git add .gitignore ScreenCaptureDaemon.sln src tests
git commit -m "chore: scaffold solution with WinForms + AspNetCore host and test project"
```

---

### Task 2: `Logger` — 启动期错误日志

**Files:**
- Create: `src/ScreenCaptureDaemon/Logger.cs`
- Test: `tests/ScreenCaptureDaemon.Tests/LoggerTests.cs`

**Interfaces:**
- Produces: `static class Logger { static string LogPath { get; set; }; static void Error(string message); }`

- [ ] **Step 1: 写失败的测试**

`tests/ScreenCaptureDaemon.Tests/LoggerTests.cs`:

```csharp
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class LoggerTests
{
    [Fact]
    public void Error_AppendsTimestampedLineToLogFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".log");
        var original = Logger.LogPath;
        Logger.LogPath = path;

        try
        {
            Logger.Error("test failure message");

            var content = File.ReadAllText(path);
            Assert.Contains("test failure message", content);
        }
        finally
        {
            Logger.LogPath = original;
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Error_AppendsMultipleLines_WithoutOverwriting()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".log");
        var original = Logger.LogPath;
        Logger.LogPath = path;

        try
        {
            Logger.Error("first");
            Logger.Error("second");

            var lines = File.ReadAllLines(path);
            Assert.Equal(2, lines.Length);
            Assert.Contains("first", lines[0]);
            Assert.Contains("second", lines[1]);
        }
        finally
        {
            Logger.LogPath = original;
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter LoggerTests`
Expected: FAIL（编译错误：找不到类型 `Logger`）

- [ ] **Step 3: 实现 `Logger`**

`src/ScreenCaptureDaemon/Logger.cs`:

```csharp
namespace ScreenCaptureDaemon;

public static class Logger
{
    public static string LogPath { get; set; } = Path.Combine(AppContext.BaseDirectory, "log.txt");

    public static void Error(string message)
    {
        var line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}{Environment.NewLine}";
        File.AppendAllText(LogPath, line);
    }
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `dotnet test --filter LoggerTests`
Expected: `Passed!` — 2 个测试全部通过

- [ ] **Step 5: Commit**

```bash
git add src/ScreenCaptureDaemon/Logger.cs tests/ScreenCaptureDaemon.Tests/LoggerTests.cs
git commit -m "feat: add file-based Logger for startup errors"
```

---

### Task 3: `AppConfig` — 配置加载

**Files:**
- Create: `src/ScreenCaptureDaemon/AppConfig.cs`
- Test: `tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs`

**Interfaces:**
- Consumes: 无
- Produces: `sealed class AppConfig { string SaveDirectory { get; set; }; int Port { get; set; }; static AppConfig Load(string path); }`

- [ ] **Step 1: 写失败的测试**

`tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs`:

```csharp
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class AppConfigTests
{
    [Fact]
    public void Load_ReturnsDefaults_WhenFileMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");

        var config = AppConfig.Load(path);

        Assert.Equal(8080, config.Port);
        Assert.False(string.IsNullOrWhiteSpace(config.SaveDirectory));
    }

    [Fact]
    public void Load_ReadsValuesFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\", \"Port\": 9090}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("F:\\Screenshots", config.SaveDirectory);
            Assert.Equal(9090, config.Port);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_UsesDefaultPort_WhenPortFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("F:\\Screenshots", config.SaveDirectory);
            Assert.Equal(8080, config.Port);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter AppConfigTests`
Expected: FAIL（编译错误：找不到类型 `AppConfig`）

- [ ] **Step 3: 实现 `AppConfig`**

`src/ScreenCaptureDaemon/AppConfig.cs`:

```csharp
using System.Text.Json;

namespace ScreenCaptureDaemon;

public sealed class AppConfig
{
    public string SaveDirectory { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "Screenshots");

    public int Port { get; set; } = 8080;

    public static AppConfig Load(string path)
    {
        if (!File.Exists(path))
        {
            return new AppConfig();
        }

        var json = File.ReadAllText(path);
        var loaded = JsonSerializer.Deserialize<AppConfig>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });

        return loaded ?? new AppConfig();
    }
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `dotnet test --filter AppConfigTests`
Expected: `Passed!` — 3 个测试全部通过

- [ ] **Step 5: Commit**

```bash
git add src/ScreenCaptureDaemon/AppConfig.cs tests/ScreenCaptureDaemon.Tests/AppConfigTests.cs
git commit -m "feat: add AppConfig loader with defaults for missing file/fields"
```

---

### Task 4: `ScreenCaptureService` — 截图逻辑

**Files:**
- Create: `src/ScreenCaptureDaemon/ScreenCaptureService.cs`
- Test: `tests/ScreenCaptureDaemon.Tests/ScreenCaptureServiceTests.cs`

**Interfaces:**
- Consumes: 无
- Produces: `static class ScreenCaptureService { static string BuildFileName(DateTime timestamp); static string Capture(string saveDirectory); }`

**Note:** `BuildFileName` 是纯函数，可自动化测试。`Capture` 依赖真实的 Windows 桌面会话（`Screen.FromPoint` / `Graphics.CopyFromScreen`），无法在无头测试环境中可靠验证，本任务对它只做人工验证（见 Step 5）。

- [ ] **Step 1: 写失败的测试（只测 `BuildFileName`）**

`tests/ScreenCaptureDaemon.Tests/ScreenCaptureServiceTests.cs`:

```csharp
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class ScreenCaptureServiceTests
{
    [Fact]
    public void BuildFileName_FormatsTimestampCorrectly()
    {
        var timestamp = new DateTime(2026, 7, 9, 14, 5, 3, 250);

        var fileName = ScreenCaptureService.BuildFileName(timestamp);

        Assert.Equal("screenshot_20260709_140503_250.png", fileName);
    }

    [Fact]
    public void BuildFileName_IsUniqueForDifferentMilliseconds()
    {
        var t1 = new DateTime(2026, 7, 9, 14, 5, 3, 250);
        var t2 = new DateTime(2026, 7, 9, 14, 5, 3, 251);

        Assert.NotEqual(ScreenCaptureService.BuildFileName(t1), ScreenCaptureService.BuildFileName(t2));
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter ScreenCaptureServiceTests`
Expected: FAIL（编译错误：找不到类型 `ScreenCaptureService`）

- [ ] **Step 3: 实现 `ScreenCaptureService`**

`src/ScreenCaptureDaemon/ScreenCaptureService.cs`:

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

    public static string Capture(string saveDirectory)
    {
        Directory.CreateDirectory(saveDirectory);

        var screen = Screen.FromPoint(Cursor.Position);
        var bounds = screen.Bounds;

        using var bitmap = new Bitmap(bounds.Width, bounds.Height);
        using (var g = Graphics.FromImage(bitmap))
        {
            g.CopyFromScreen(bounds.Location, Point.Empty, bounds.Size);
        }

        var path = Path.Combine(saveDirectory, BuildFileName(DateTime.Now));
        bitmap.Save(path, ImageFormat.Png);
        return path;
    }
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `dotnet test --filter ScreenCaptureServiceTests`
Expected: `Passed!` — 2 个测试全部通过

- [ ] **Step 5: 人工验证 `Capture`**

在 `tests/ScreenCaptureDaemon.Tests` 目录下临时写一段一次性代码（或用 `dotnet fsi`/交互方式均可），调用：

```csharp
var path = ScreenCaptureDaemon.ScreenCaptureService.Capture(@"C:\temp\capture-test");
Console.WriteLine(path);
```

Expected: 打印出的路径下生成了一张 PNG，内容与当前鼠标所在显示器画面一致，图片尺寸与该显示器分辨率一致

- [ ] **Step 6: Commit**

```bash
git add src/ScreenCaptureDaemon/ScreenCaptureService.cs tests/ScreenCaptureDaemon.Tests/ScreenCaptureServiceTests.cs
git commit -m "feat: add ScreenCaptureService capturing the monitor under the cursor"
```

---

### Task 5: `HttpServerHost` — 静态文件 + 目录浏览服务

**Files:**
- Create: `src/ScreenCaptureDaemon/HttpServerHost.cs`
- Test: `tests/ScreenCaptureDaemon.Tests/HttpServerHostTests.cs`

**Interfaces:**
- Consumes: 无
- Produces: `sealed class HttpServerHost : IAsyncDisposable { int Port { get; }; Task StartAsync(string rootDirectory, int port, string host = "0.0.0.0"); ValueTask DisposeAsync(); }`

- [ ] **Step 1: 写失败的测试**

`tests/ScreenCaptureDaemon.Tests/HttpServerHostTests.cs`:

```csharp
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class HttpServerHostTests
{
    [Fact]
    public async Task ServesFileFromRootDirectory()
    {
        var tempDir = Directory.CreateTempSubdirectory().FullName;
        try
        {
            File.WriteAllText(Path.Combine(tempDir, "hello.txt"), "hello world");

            var host = new HttpServerHost();
            await host.StartAsync(tempDir, 0, "127.0.0.1");
            try
            {
                using var client = new HttpClient();
                var response = await client.GetStringAsync($"http://127.0.0.1:{host.Port}/hello.txt");
                Assert.Equal("hello world", response);
            }
            finally
            {
                await host.DisposeAsync();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task DirectoryListingShowsFileName()
    {
        var tempDir = Directory.CreateTempSubdirectory().FullName;
        try
        {
            File.WriteAllText(Path.Combine(tempDir, "screenshot_20260709_120000_000.png"), "fake-png-bytes");

            var host = new HttpServerHost();
            await host.StartAsync(tempDir, 0, "127.0.0.1");
            try
            {
                using var client = new HttpClient();
                var response = await client.GetStringAsync($"http://127.0.0.1:{host.Port}/");
                Assert.Contains("screenshot_20260709_120000_000.png", response);
            }
            finally
            {
                await host.DisposeAsync();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `dotnet test --filter HttpServerHostTests`
Expected: FAIL（编译错误：找不到类型 `HttpServerHost`）

- [ ] **Step 3: 实现 `HttpServerHost`**

`src/ScreenCaptureDaemon/HttpServerHost.cs`:

```csharp
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;

namespace ScreenCaptureDaemon;

public sealed class HttpServerHost : IAsyncDisposable
{
    private WebApplication? _app;

    public int Port { get; private set; }

    public async Task StartAsync(string rootDirectory, int port, string host = "0.0.0.0")
    {
        Directory.CreateDirectory(rootDirectory);

        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls($"http://{host}:{port}");
        builder.Logging.ClearProviders();

        _app = builder.Build();

        var fileProvider = new PhysicalFileProvider(rootDirectory);
        _app.UseStaticFiles(new StaticFileOptions { FileProvider = fileProvider });
        _app.UseDirectoryBrowser(new DirectoryBrowserOptions { FileProvider = fileProvider });

        await _app.StartAsync();

        var addressFeature = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>();
        var address = addressFeature!.Addresses.First();
        Port = int.Parse(address.Split(':').Last());
    }

    public async ValueTask DisposeAsync()
    {
        if (_app != null)
        {
            await _app.StopAsync();
            await _app.DisposeAsync();
        }
    }
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `dotnet test --filter HttpServerHostTests`
Expected: `Passed!` — 2 个测试全部通过

- [ ] **Step 5: Commit**

```bash
git add src/ScreenCaptureDaemon/HttpServerHost.cs tests/ScreenCaptureDaemon.Tests/HttpServerHostTests.cs
git commit -m "feat: add HttpServerHost serving static files with directory browsing"
```

---

### Task 6: `HotkeyManager` — 全局热键 Win+Shift+C

**Files:**
- Create: `src/ScreenCaptureDaemon/HotkeyManager.cs`

**Interfaces:**
- Consumes: 无
- Produces: `sealed class HotkeyManager : NativeWindow, IDisposable { event Action? HotkeyPressed; bool Register(); void Dispose(); }`

**Note:** 全局热键注册（`RegisterHotKey`）依赖真实的 Windows 交互式会话和消息循环，无法在 xUnit 无头测试进程中验证，本任务只做人工验证。

- [ ] **Step 1: 实现 `HotkeyManager`**

`src/ScreenCaptureDaemon/HotkeyManager.cs`:

```csharp
using System.Runtime.InteropServices;

namespace ScreenCaptureDaemon;

public sealed class HotkeyManager : NativeWindow, IDisposable
{
    private const int WM_HOTKEY = 0x0312;
    private const int HotkeyId = 1;
    private const uint ModShift = 0x0004;
    private const uint ModWin = 0x0008;
    private const uint VkC = 0x43;

    private bool _registered;

    public event Action? HotkeyPressed;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

    public bool Register()
    {
        var cp = new CreateParams { Parent = (IntPtr)(-3) }; // HWND_MESSAGE：消息专用窗口，不可见
        CreateHandle(cp);

        _registered = RegisterHotKey(Handle, HotkeyId, ModWin | ModShift, VkC);
        return _registered;
    }

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WM_HOTKEY && m.WParam.ToInt32() == HotkeyId)
        {
            HotkeyPressed?.Invoke();
        }

        base.WndProc(ref m);
    }

    public void Dispose()
    {
        if (_registered)
        {
            UnregisterHotKey(Handle, HotkeyId);
        }

        if (Handle != IntPtr.Zero)
        {
            DestroyHandle();
        }
    }
}
```

- [ ] **Step 2: 构建确认无编译错误**

Run: `dotnet build`
Expected: `Build succeeded.`

- [ ] **Step 3: Commit**

```bash
git add src/ScreenCaptureDaemon/HotkeyManager.cs
git commit -m "feat: add HotkeyManager registering Win+Shift+C as a global hotkey"
```

（人工验证放在 Task 8 的端到端验证中一并完成，因为热键触发的效果需要配合 `ScreenCaptureService` 才能观察。）

---

### Task 7: `TrayIconManager` — 托盘图标与菜单

**Files:**
- Create: `src/ScreenCaptureDaemon/TrayIconManager.cs`

**Interfaces:**
- Consumes: `string saveDirectory`, `Action onExit`（构造函数参数）
- Produces: `sealed class TrayIconManager : IDisposable { void Dispose(); }`

- [ ] **Step 1: 实现 `TrayIconManager`**

`src/ScreenCaptureDaemon/TrayIconManager.cs`:

```csharp
using System.Diagnostics;
using System.Windows.Forms;

namespace ScreenCaptureDaemon;

public sealed class TrayIconManager : IDisposable
{
    private readonly NotifyIcon _notifyIcon;

    public TrayIconManager(string saveDirectory, Action onExit)
    {
        var menu = new ContextMenuStrip();
        menu.Items.Add("打开截图目录", null, (_, _) =>
        {
            Directory.CreateDirectory(saveDirectory);
            Process.Start(new ProcessStartInfo(saveDirectory) { UseShellExecute = true });
        });
        menu.Items.Add("退出", null, (_, _) => onExit());

        _notifyIcon = new NotifyIcon
        {
            Icon = SystemIcons.Application,
            Text = "截图守护进程",
            Visible = true,
            ContextMenuStrip = menu
        };
    }

    public void Dispose()
    {
        _notifyIcon.Visible = false;
        _notifyIcon.Dispose();
    }
}
```

- [ ] **Step 2: 构建确认无编译错误**

Run: `dotnet build`
Expected: `Build succeeded.`

- [ ] **Step 3: Commit**

```bash
git add src/ScreenCaptureDaemon/TrayIconManager.cs
git commit -m "feat: add TrayIconManager with open-directory and exit menu items"
```

---

### Task 8: `Program.cs` — 组装所有组件 + 端到端验证

**Files:**
- Modify: `src/ScreenCaptureDaemon/Program.cs`

**Interfaces:**
- Consumes: `AppConfig`, `Logger`, `HttpServerHost`, `HotkeyManager`, `ScreenCaptureService`, `TrayIconManager`（前面所有任务产出的类型）

- [ ] **Step 1: 实现完整的 `Program.cs`**

`src/ScreenCaptureDaemon/Program.cs`:

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

        if (!hotkeyManager.Register())
        {
            Logger.Error("热键 Win+Shift+C 注册失败，可能已被其他程序占用。");
        }

        var trayIcon = new TrayIconManager(config.SaveDirectory, Application.Exit);

        Application.ApplicationExit += (_, _) =>
        {
            hotkeyManager.Dispose();
            trayIcon.Dispose();
            httpServer.DisposeAsync().AsTask().GetAwaiter().GetResult();
            _mutex?.ReleaseMutex();
        };

        Application.Run();
    }
}
```

- [ ] **Step 2: 构建整个解决方案**

Run: `dotnet build`
Expected: `Build succeeded.`，无编译错误

- [ ] **Step 3: 运行完整测试套件**

Run: `dotnet test`
Expected: 全部测试通过（`Logger`、`AppConfig`、`ScreenCaptureService.BuildFileName`、`HttpServerHost` 共 9 个测试）

- [ ] **Step 4: 人工端到端验证**

1. 在 `src/ScreenCaptureDaemon/bin/Debug/net8.0-windows/` 下确认 `appsettings.json` 已随构建复制，把其中 `SaveDirectory` 改成一个**不存在**的本机路径（例如 `C:\temp\ScreenshotsTest`，确保该目录当前不存在）
2. 运行 `src/ScreenCaptureDaemon/bin/Debug/net8.0-windows/ScreenCaptureDaemon.exe`
3. 确认系统托盘出现图标，且没有弹出任何窗口或提示
4. 确认 `C:\temp\ScreenshotsTest` 目录被自动创建（对应"保存目录不存在时自动创建"场景）
5. 按下 Win+Shift+C，确认该目录下生成了一张新的 `screenshot_*.png`，内容与当前鼠标所在显示器画面一致
6. 在同一台机器浏览器访问 `http://localhost:8080/`，确认能看到目录列表并能点击预览/下载刚才的截图
7. 在同一局域网的另一台设备上，用第 1 台机器的局域网 IP 访问 `http://<IP>:8080/`，确认同样能访问
8. 右键托盘图标 → "打开截图目录"，确认资源管理器打开了正确的目录
9. 不退出程序的情况下再次运行一次 `ScreenCaptureDaemon.exe`，确认第二个实例立即自行退出（`tasklist` 中只有一个 `ScreenCaptureDaemon.exe` 进程），且第一个实例不受影响
10. 退出当前运行的实例。用 `netstat -ano | findstr :8080` 确认端口空闲后，先用另一个临时进程占用 8080 端口（例如 `python -m http.server 8080` 或任意监听 8080 的程序），再启动 `ScreenCaptureDaemon.exe`，确认程序**没有崩溃**、托盘图标正常出现、`log.txt` 中新增一行包含"HTTP 服务启动失败"的记录；关闭占用端口的临时进程后重启程序确认 HTTP 服务恢复正常
11. 右键托盘图标 → "退出"，确认托盘图标消失、HTTP 服务不再可访问、进程完全退出

Expected: 以上 11 项全部符合预期

- [ ] **Step 5: Commit**

```bash
git add src/ScreenCaptureDaemon/Program.cs
git commit -m "feat: wire config, hotkey, capture, tray icon and http server together"
```

---

## Post-Plan Notes

- 若热键 Win+Shift+C 与已安装的其他工具（如 PowerToys）冲突，`Register()` 会返回 `false` 并写入 `log.txt`，需要手动排查/更换热键组合，不在本计划自动处理范围内
- 若需要开机自启动，可以在此计划完成后另起一个小任务，把程序快捷方式放入 `shell:startup`，不在本次范围
