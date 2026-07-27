using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class HotkeyManagerTests
{
    [Theory]
    [InlineData("Win+Shift+Z", 0x0008u | 0x0004u, 0x5Au)]
    [InlineData("win+shift+z", 0x0008u | 0x0004u, 0x5Au)]
    [InlineData("Ctrl+Alt+9", 0x0002u | 0x0001u, 0x39u)]
    [InlineData("Shift+A", 0x0004u, 0x41u)]
    public void TryParse_ParsesValidHotkeyStrings(string text, uint expectedModifiers, uint expectedVk)
    {
        var result = HotkeyManager.TryParse(text, out var modifiers, out var vk);

        Assert.True(result);
        Assert.Equal(expectedModifiers, modifiers);
        Assert.Equal(expectedVk, vk);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("Z")]
    [InlineData("Win+Shift")]
    [InlineData("Win+Shift+Z+X")]
    [InlineData("Win+Shift+ZZ")]
    [InlineData("Banana+Z")]
    [InlineData("Win+Shift+F1")]
    public void TryParse_RejectsInvalidHotkeyStrings(string? text)
    {
        var result = HotkeyManager.TryParse(text, out _, out _);

        Assert.False(result);
    }
}
