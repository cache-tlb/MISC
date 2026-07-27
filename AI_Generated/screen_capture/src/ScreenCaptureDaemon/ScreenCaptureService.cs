using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace ScreenCaptureDaemon;

public static class ScreenCaptureService
{
    public static string BuildFileName(DateTime timestamp)
    {
        return $"screenshot_{timestamp:yyyyMMdd_HHmmss_fff}.png";
    }

    // 调用方负责 Dispose 返回的 Bitmap。
    public static Bitmap CaptureBitmap()
    {
        var screen = Screen.FromPoint(Cursor.Position);
        var bounds = screen.Bounds;

        var bitmap = new Bitmap(bounds.Width, bounds.Height);
        using (var g = Graphics.FromImage(bitmap))
        {
            g.CopyFromScreen(bounds.Location, Point.Empty, bounds.Size);
        }

        return bitmap;
    }

    public static byte[] EncodeToPngBytes(Bitmap bitmap)
    {
        using var ms = new MemoryStream();
        bitmap.Save(ms, ImageFormat.Png);
        return ms.ToArray();
    }

    public static byte[] CaptureToBytes()
    {
        using var bitmap = CaptureBitmap();
        return EncodeToPngBytes(bitmap);
    }

    public static string Capture(string saveDirectory)
    {
        Directory.CreateDirectory(saveDirectory);
        var imageBytes = CaptureToBytes();
        var path = Path.Combine(saveDirectory, BuildFileName(DateTime.Now));
        File.WriteAllBytes(path, imageBytes);
        return path;
    }
}
