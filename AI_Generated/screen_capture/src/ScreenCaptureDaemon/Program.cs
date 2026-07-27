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

        var deduplicator = new ScreenshotDeduplicator(config.IgnoreRegions);
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
            trayIcon.ShowBalloon("SC 守护进程", message);
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
            trayIcon.ShowBalloon("SC 守护进程", message);
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
