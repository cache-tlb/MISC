using Timer = System.Windows.Forms.Timer;

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
