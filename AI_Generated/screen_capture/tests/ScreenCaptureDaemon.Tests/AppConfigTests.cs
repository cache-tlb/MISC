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

    [Fact]
    public void Load_ReturnsDefaultsAndLogs_WhenJsonIsMalformed()
    {
        var configPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(configPath, "{ this is not valid json");

        var logPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".log");
        var originalLogPath = Logger.LogPath;
        Logger.LogPath = logPath;

        try
        {
            var config = AppConfig.Load(configPath);

            Assert.Equal(8080, config.Port);
            Assert.False(string.IsNullOrWhiteSpace(config.SaveDirectory));
            Assert.True(File.Exists(logPath));
            Assert.Contains("配置文件解析失败", File.ReadAllText(logPath));
        }
        finally
        {
            Logger.LogPath = originalLogPath;
            File.Delete(configPath);
            if (File.Exists(logPath)) File.Delete(logPath);
        }
    }

    [Fact]
    public void Load_UsesDefaultHotkey_WhenHotkeyFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Win+Shift+Z", config.Hotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsHotkeyFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"Hotkey\": \"Ctrl+Alt+9\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Ctrl+Alt+9", config.Hotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_UsesDefaultToggleAutoSaveHotkey_WhenFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Win+Shift+X", config.ToggleAutoSaveHotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsToggleAutoSaveHotkeyFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"ToggleAutoSaveHotkey\": \"Ctrl+Alt+X\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal("Ctrl+Alt+X", config.ToggleAutoSaveHotkey);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_UsesDefaultAutoSaveIntervalSeconds_WhenFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal(20, config.AutoSaveIntervalSeconds);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsAutoSaveIntervalSecondsFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"AutoSaveIntervalSeconds\": 45}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Equal(45, config.AutoSaveIntervalSeconds);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void Load_FallsBackToDefaultInterval_WhenAutoSaveIntervalSecondsIsNotPositive(int invalidInterval)
    {
        var configPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(configPath, $"{{\"AutoSaveIntervalSeconds\": {invalidInterval}}}");

        var logPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".log");
        var originalLogPath = Logger.LogPath;
        Logger.LogPath = logPath;

        try
        {
            var config = AppConfig.Load(configPath);

            Assert.Equal(20, config.AutoSaveIntervalSeconds);
            Assert.True(File.Exists(logPath));
            Assert.Contains("AutoSaveIntervalSeconds", File.ReadAllText(logPath));
        }
        finally
        {
            Logger.LogPath = originalLogPath;
            File.Delete(configPath);
            if (File.Exists(logPath)) File.Delete(logPath);
        }
    }

    [Fact]
    public void Load_UsesEmptyIgnoreRegions_WhenFieldMissing()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"SaveDirectory\": \"F:\\\\Screenshots\"}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Empty(config.IgnoreRegions);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ReadsIgnoreRegionsFromFile()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        File.WriteAllText(path, "{\"IgnoreRegions\": [{\"X\": 0.9, \"Y\": 0.95, \"Width\": 0.1, \"Height\": 0.05}]}");

        try
        {
            var config = AppConfig.Load(path);

            Assert.Single(config.IgnoreRegions);
            Assert.Equal(0.9, config.IgnoreRegions[0].X);
            Assert.Equal(0.95, config.IgnoreRegions[0].Y);
            Assert.Equal(0.1, config.IgnoreRegions[0].Width);
            Assert.Equal(0.05, config.IgnoreRegions[0].Height);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
