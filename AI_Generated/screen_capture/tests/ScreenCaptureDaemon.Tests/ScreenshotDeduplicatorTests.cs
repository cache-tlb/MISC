using System.Drawing;
using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class ScreenshotDeduplicatorTests
{
    [Fact]
    public void IsDuplicate_ReturnsFalse_WhenNothingRememberedYet()
    {
        var deduplicator = new ScreenshotDeduplicator();

        var result = deduplicator.IsDuplicate(new byte[] { 1, 2, 3 });

        Assert.False(result);
    }

    [Fact]
    public void IsDuplicate_ReturnsTrue_WhenBytesMatchRememberedValue()
    {
        var deduplicator = new ScreenshotDeduplicator();
        deduplicator.Remember(new byte[] { 1, 2, 3 });

        var result = deduplicator.IsDuplicate(new byte[] { 1, 2, 3 });

        Assert.True(result);
    }

    [Fact]
    public void IsDuplicate_ReturnsFalse_WhenBytesDifferFromRememberedValue()
    {
        var deduplicator = new ScreenshotDeduplicator();
        deduplicator.Remember(new byte[] { 1, 2, 3 });

        var result = deduplicator.IsDuplicate(new byte[] { 4, 5, 6 });

        Assert.False(result);
    }

    [Fact]
    public void IsDuplicate_ReturnsFalse_WhenBytesDifferOnlyInLength()
    {
        var deduplicator = new ScreenshotDeduplicator();
        deduplicator.Remember(new byte[] { 1, 2, 3 });

        var result = deduplicator.IsDuplicate(new byte[] { 1, 2, 3, 4 });

        Assert.False(result);
    }

    [Fact]
    public void IsDuplicate_ReflectsMostRecentlyRememberedValue()
    {
        var deduplicator = new ScreenshotDeduplicator();

        deduplicator.Remember(new byte[] { 9, 9, 9 });
        var stillMatchesFirst = deduplicator.IsDuplicate(new byte[] { 9, 9, 9 });

        deduplicator.Remember(new byte[] { 1, 1, 1 });
        var matchesUpdatedValue = deduplicator.IsDuplicate(new byte[] { 1, 1, 1 });
        var noLongerMatchesOldValue = deduplicator.IsDuplicate(new byte[] { 9, 9, 9 });

        Assert.True(stillMatchesFirst);
        Assert.True(matchesUpdatedValue);
        Assert.False(noLongerMatchesOldValue);
    }

    [Fact]
    public void ToPixelRect_ConvertsNormalRegionToPixelCoordinates()
    {
        var region = new IgnoreRegion { X = 0.9, Y = 0.95, Width = 0.1, Height = 0.05 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 3840, 2160);

        Assert.Equal(new Rectangle(3456, 2052, 384, 108), rect);
    }

    [Fact]
    public void ToPixelRect_ClampsNegativeXAndY_ToZero()
    {
        var region = new IgnoreRegion { X = -0.5, Y = -0.2, Width = 0.1, Height = 0.1 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 1000, 1000);

        Assert.Equal(0, rect.X);
        Assert.Equal(0, rect.Y);
    }

    [Fact]
    public void ToPixelRect_ClampsWidthAndHeight_WhenTheyWouldExceedScreenBounds()
    {
        var region = new IgnoreRegion { X = 0.9, Y = 0.9, Width = 0.5, Height = 0.5 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 1000, 1000);

        Assert.Equal(900, rect.X);
        Assert.Equal(900, rect.Y);
        Assert.Equal(100, rect.Width);
        Assert.Equal(100, rect.Height);
    }

    [Fact]
    public void ToPixelRect_ClampsXGreaterThanOne_ToZeroWidthAtScreenEdge()
    {
        var region = new IgnoreRegion { X = 1.5, Y = 0, Width = 0.2, Height = 0.2 };

        var rect = ScreenshotDeduplicator.ToPixelRect(region, 1000, 1000);

        Assert.Equal(1000, rect.X);
        Assert.Equal(0, rect.Width);
    }
}
