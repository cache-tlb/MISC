using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;

namespace ScreenCaptureDaemon;

public sealed class HttpServerHost : IAsyncDisposable
{
    private WebApplication? _app;
    private bool _started;

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
        _started = true;

        var addressFeature = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>();
        var address = addressFeature!.Addresses.First();
        Port = new Uri(address).Port;
    }

    public async ValueTask DisposeAsync()
    {
        if (_app != null)
        {
            if (_started)
            {
                await _app.StopAsync();
            }

            await _app.DisposeAsync();
        }
    }
}
