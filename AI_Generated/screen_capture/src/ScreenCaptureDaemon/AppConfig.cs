using System.Text.Json;

namespace ScreenCaptureDaemon;

public sealed class AppConfig
{
    public string SaveDirectory { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "Screenshots");

    public int Port { get; set; } = 8080;

    public string Hotkey { get; set; } = "Win+Shift+Z";

    public string ToggleAutoSaveHotkey { get; set; } = "Win+Shift+X";

    public int AutoSaveIntervalSeconds { get; set; } = 20;

    public List<IgnoreRegion> IgnoreRegions { get; set; } = new();

    public static AppConfig Load(string path)
    {
        if (!File.Exists(path))
        {
            return new AppConfig();
        }

        try
        {
            var json = File.ReadAllText(path);
            var loaded = JsonSerializer.Deserialize<AppConfig>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            if (loaded == null)
            {
                return new AppConfig();
            }

            if (loaded.AutoSaveIntervalSeconds <= 0)
            {
                Logger.Error($"AutoSaveIntervalSeconds 配置值 {loaded.AutoSaveIntervalSeconds} 无效，回退到默认值 20");
                loaded.AutoSaveIntervalSeconds = 20;
            }

            return loaded;
        }
        catch (JsonException ex)
        {
            Logger.Error("配置文件解析失败，使用默认配置: " + ex.Message);
            return new AppConfig();
        }
    }
}
