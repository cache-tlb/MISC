using System.Diagnostics;
using System.Windows.Forms;

namespace ScreenCaptureDaemon;

public sealed class TrayIconManager : IDisposable
{
    private readonly NotifyIcon _notifyIcon;

    public TrayIconManager(string saveDirectory, Action onExit)
    {
        var menu = new ContextMenuStrip();
        menu.Items.Add("打开 SC 目录", null, (_, _) =>
        {
            try
            {
                Directory.CreateDirectory(saveDirectory);
                Process.Start(new ProcessStartInfo(saveDirectory) { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                Logger.Error($"打开截图目录失败: {ex.Message}");
            }
        });
        menu.Items.Add("退出", null, (_, _) => onExit());

        _notifyIcon = new NotifyIcon
        {
            Icon = SystemIcons.Application,
            Text = "SC 守护进程",
            Visible = true,
            ContextMenuStrip = menu
        };
    }

    public void ShowBalloon(string title, string text)
    {
        _notifyIcon.BalloonTipTitle = title;
        _notifyIcon.BalloonTipText = text;
        _notifyIcon.ShowBalloonTip(5000);
    }

    public void Dispose()
    {
        _notifyIcon.Visible = false;
        _notifyIcon.Dispose();
    }
}
