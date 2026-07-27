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

    [Fact]
    public void Error_DoesNotThrow_WhenLogPathIsInvalid()
    {
        var original = Logger.LogPath;
        Logger.LogPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString(), "nonexistent-dir", "log.txt");

        try
        {
            var exception = Record.Exception(() => Logger.Error("should not throw"));
            Assert.Null(exception);
        }
        finally
        {
            Logger.LogPath = original;
        }
    }
}
