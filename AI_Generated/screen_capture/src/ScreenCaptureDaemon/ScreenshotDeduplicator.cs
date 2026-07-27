using System.Drawing;

namespace ScreenCaptureDaemon;

public sealed class ScreenshotDeduplicator
{
    private readonly IReadOnlyList<IgnoreRegion> _ignoreRegions;
    private byte[]? _lastImageBytes;

    public ScreenshotDeduplicator(IReadOnlyList<IgnoreRegion>? ignoreRegions = null)
    {
        _ignoreRegions = ignoreRegions ?? Array.Empty<IgnoreRegion>();
    }

    public bool IsDuplicate(byte[] imageBytes)
    {
        return _lastImageBytes != null && imageBytes.AsSpan().SequenceEqual(_lastImageBytes);
    }

    public void Remember(byte[] imageBytes)
    {
        _lastImageBytes = imageBytes;
    }

    internal static Rectangle ToPixelRect(IgnoreRegion region, int screenWidth, int screenHeight)
    {
        var x = Math.Clamp(region.X, 0.0, 1.0);
        var y = Math.Clamp(region.Y, 0.0, 1.0);
        var width = Math.Clamp(region.Width, 0.0, 1.0 - x);
        var height = Math.Clamp(region.Height, 0.0, 1.0 - y);

        return new Rectangle(
            (int)Math.Round(x * screenWidth),
            (int)Math.Round(y * screenHeight),
            (int)Math.Round(width * screenWidth),
            (int)Math.Round(height * screenHeight));
    }

    public string? CaptureAndSaveIfNotDuplicate(string saveDirectory)
    {
        Directory.CreateDirectory(saveDirectory);
        using var bitmap = ScreenCaptureService.CaptureBitmap();

        using var comparisonBitmap = (Bitmap)bitmap.Clone();
        if (_ignoreRegions.Count > 0)
        {
            using var g = Graphics.FromImage(comparisonBitmap);
            foreach (var region in _ignoreRegions)
            {
                g.FillRectangle(Brushes.Black, ToPixelRect(region, bitmap.Width, bitmap.Height));
            }
        }

        var comparisonBytes = ScreenCaptureService.EncodeToPngBytes(comparisonBitmap);

        if (IsDuplicate(comparisonBytes))
        {
            return null;
        }

        var imageBytes = ScreenCaptureService.EncodeToPngBytes(bitmap);
        var path = Path.Combine(saveDirectory, ScreenCaptureService.BuildFileName(DateTime.Now));
        File.WriteAllBytes(path, imageBytes);
        Remember(comparisonBytes);
        return path;
    }
}
