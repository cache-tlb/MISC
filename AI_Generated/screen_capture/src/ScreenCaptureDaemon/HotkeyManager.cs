using System.Runtime.InteropServices;

namespace ScreenCaptureDaemon;

public sealed class HotkeyManager : NativeWindow, IDisposable
{
    private const int WM_HOTKEY = 0x0312;
    private const int HotkeyId = 1;
    private const uint ModAlt = 0x0001;
    private const uint ModControl = 0x0002;
    private const uint ModShift = 0x0004;
    private const uint ModWin = 0x0008;

    private bool _registered;

    public event Action? HotkeyPressed;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

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

    public bool Register(uint modifiers, uint vk)
    {
        var cp = new CreateParams { Parent = (IntPtr)(-3) }; // HWND_MESSAGE：消息专用窗口，不可见
        CreateHandle(cp);

        _registered = RegisterHotKey(Handle, HotkeyId, modifiers, vk);
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
            _registered = false;
        }

        if (Handle != IntPtr.Zero)
        {
            DestroyHandle();
        }
    }
}
