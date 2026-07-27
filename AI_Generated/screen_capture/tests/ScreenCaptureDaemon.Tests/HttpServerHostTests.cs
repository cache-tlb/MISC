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

    [Fact]
    public async Task DisposeAsync_DoesNotThrow_WhenStartAsyncFailed()
    {
        var tempDir = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var blocker = new HttpServerHost();
            await blocker.StartAsync(tempDir, 0, "127.0.0.1");
            try
            {
                var occupiedPort = blocker.Port;

                var host = new HttpServerHost();
                await Assert.ThrowsAnyAsync<Exception>(() => host.StartAsync(tempDir, occupiedPort, "127.0.0.1"));

                await host.DisposeAsync();
            }
            finally
            {
                await blocker.DisposeAsync();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }
}
