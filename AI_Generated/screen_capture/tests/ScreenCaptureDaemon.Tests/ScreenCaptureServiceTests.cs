using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class ScreenCaptureServiceTests
{
    [Fact]
    public void BuildFileName_FormatsTimestampCorrectly()
    {
        var timestamp = new DateTime(2026, 7, 9, 14, 5, 3, 250);

        var fileName = ScreenCaptureService.BuildFileName(timestamp);

        Assert.Equal("screenshot_20260709_140503_250.png", fileName);
    }

    [Fact]
    public void BuildFileName_IsUniqueForDifferentMilliseconds()
    {
        var t1 = new DateTime(2026, 7, 9, 14, 5, 3, 250);
        var t2 = new DateTime(2026, 7, 9, 14, 5, 3, 251);

        Assert.NotEqual(ScreenCaptureService.BuildFileName(t1), ScreenCaptureService.BuildFileName(t2));
    }
}
