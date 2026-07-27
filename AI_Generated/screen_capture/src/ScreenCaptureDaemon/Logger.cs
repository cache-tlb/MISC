namespace ScreenCaptureDaemon;

public static class Logger
{
    public static string LogPath { get; set; } = Path.Combine(AppContext.BaseDirectory, "log.txt");

    public static void Error(string message)
    {
        try
        {
            var line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}{Environment.NewLine}";
            File.AppendAllText(LogPath, line);
        }
        catch
        {
            // Logging must never crash the host application.
        }
    }
}
