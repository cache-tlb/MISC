using ScreenCaptureDaemon;
using Xunit;

namespace ScreenCaptureDaemon.Tests;

public class AutoSaveControllerTests
{
    [Fact]
    public void Toggle_InvokesCallbackImmediately_WhenTurningOn()
    {
        var callCount = 0;
        using var controller = new AutoSaveController(20, () => callCount++);

        controller.Toggle();

        Assert.Equal(1, callCount);
        Assert.True(controller.Enabled);
    }

    [Fact]
    public void Toggle_DoesNotInvokeCallback_WhenTurningOff()
    {
        var callCount = 0;
        using var controller = new AutoSaveController(20, () => callCount++);

        controller.Toggle();
        controller.Toggle();

        Assert.Equal(1, callCount);
        Assert.False(controller.Enabled);
    }

    [Fact]
    public void Toggle_InvokesCallbackAgain_WhenTurnedOnASecondTime()
    {
        var callCount = 0;
        using var controller = new AutoSaveController(20, () => callCount++);

        controller.Toggle();
        controller.Toggle();
        controller.Toggle();

        Assert.Equal(2, callCount);
        Assert.True(controller.Enabled);
    }
}
