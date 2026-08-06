/*
 * Unreal Engine style fly camera.
 *
 * The controls only engage while the right mouse button is held, exactly like
 * the UE viewport:
 *
 *   right mouse drag  yaw / pitch
 *   w / s             forward / back
 *   a / d             strafe left / right
 *   e / q             up / down
 *   mouse wheel       change movement speed
 *   shift             temporary speed boost
 *
 * Coordinate system is right handed, as OpenGL conventionally uses: +x right,
 * +y up, +z out of the screen toward the viewer, so the camera looks along -z.
 *
 * Yaw is a rotation about +y following the right hand rule, which means
 * positive yaw turns the camera to the *left*, and yaw 0 looks straight down
 * -z. Pitch is positive when looking up.
 *
 * @module camera
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./m4.js'], factory);
  } else {
    // Browser globals
    root.createCamera = factory(root.m4);
  }
}(this, function(m4) {
  "use strict";

  var MAX_PITCH = Math.PI * 0.5 - 0.01;
  var MIN_SPEED = 0.5;
  var MAX_SPEED = 200;

  /**
   * Creates a fly camera bound to an element.
   * @param {HTMLElement} element the element that captures the mouse
   * @param {Object} [options] position, yaw, pitch, speed, sensitivity
   * @return {Object} the camera
   * @memberOf module:camera
   */
  function createCamera(element, options) {
    options = options || {};

    var camera = {
      position: options.position ? options.position.slice() : [0, 5, 30],
      yaw: options.yaw === undefined ? 0 : options.yaw,
      pitch: options.pitch === undefined ? 0 : options.pitch,
      speed: options.speed === undefined ? 20 : options.speed,
      sensitivity: options.sensitivity === undefined ? 0.0022 : options.sensitivity,
      active: false,
    };

    var keys = Object.create(null);
    var usingPointerLock = false;

    /**
     * The unit vector the camera looks along.
     * @return {number[]} forward in world space
     */
    camera.getForward = function() {
      var cp = Math.cos(camera.pitch);
      return [
        -Math.sin(camera.yaw) * cp,
        Math.sin(camera.pitch),
        -Math.cos(camera.yaw) * cp,
      ];
    };

    /**
     * The unit vector pointing to the camera's right, ie. cross(forward, up)
     * in a right handed basis. At yaw 0, where forward is -z, this is +x.
     * @return {number[]} right in world space
     */
    camera.getRight = function() {
      return [Math.cos(camera.yaw), 0, -Math.sin(camera.yaw)];
    };

    /**
     * The camera's world matrix. Take the inverse for a view matrix.
     * @param {Matrix4} [dst] optional matrix to store result
     * @return {Matrix4} dst or a new matrix if none provided
     */
    camera.getWorldMatrix = function(dst) {
      var forward = camera.getForward();
      var target = [
        camera.position[0] + forward[0],
        camera.position[1] + forward[1],
        camera.position[2] + forward[2],
      ];
      return m4.lookAt(camera.position, target, [0, 1, 0], dst);
    };

    /**
     * The view matrix.
     * @param {Matrix4} [dst] optional matrix to store result
     * @return {Matrix4} dst or a new matrix if none provided
     */
    camera.getViewMatrix = function(dst) {
      return m4.inverse(camera.getWorldMatrix(), dst);
    };

    /**
     * Advances the camera by one frame of held-key movement.
     * @param {number} deltaTime seconds since the previous update
     */
    camera.update = function(deltaTime) {
      if (!camera.active) {
        return;
      }

      var forwardAxis = (keys.w ? 1 : 0) - (keys.s ? 1 : 0);
      var rightAxis = (keys.d ? 1 : 0) - (keys.a ? 1 : 0);
      var upAxis = (keys.e ? 1 : 0) - (keys.q ? 1 : 0);
      if (!forwardAxis && !rightAxis && !upAxis) {
        return;
      }

      var forward = camera.getForward();
      var right = camera.getRight();
      var distance = camera.speed * deltaTime * (keys.shift ? 4 : 1);

      camera.position[0] += (forward[0] * forwardAxis + right[0] * rightAxis) * distance;
      camera.position[1] += (forward[1] * forwardAxis + upAxis) * distance;
      camera.position[2] += (forward[2] * forwardAxis + right[2] * rightAxis) * distance;
    };

    function setActive(active) {
      if (camera.active === active) {
        return;
      }
      camera.active = active;
      if (!active) {
        // Drop every held key, otherwise a key released outside the capture
        // would stay stuck down.
        keys = Object.create(null);
        if (usingPointerLock && document.pointerLockElement === element) {
          document.exitPointerLock();
        }
        usingPointerLock = false;
      }
    }

    element.addEventListener('contextmenu', function(e) {
      e.preventDefault();
    });

    element.addEventListener('mousedown', function(e) {
      if (e.button !== 2) {
        return;
      }
      e.preventDefault();
      setActive(true);
      if (element.requestPointerLock) {
        usingPointerLock = true;
        // Chrome returns a promise that rejects if the lock is requested again
        // too quickly. Falling back to plain movementX/Y is fine.
        var result = element.requestPointerLock();
        if (result && result.catch) {
          result.catch(function() { usingPointerLock = false; });
        }
      }
    });

    // Listen on the window so releasing the button off-canvas still stops us.
    window.addEventListener('mouseup', function(e) {
      if (e.button === 2) {
        setActive(false);
      }
    });

    window.addEventListener('blur', function() {
      setActive(false);
    });

    window.addEventListener('mousemove', function(e) {
      if (!camera.active) {
        return;
      }
      // Subtracting, because a right handed yaw about +y turns to the left.
      camera.yaw -= e.movementX * camera.sensitivity;
      camera.pitch -= e.movementY * camera.sensitivity;
      camera.pitch = Math.min(MAX_PITCH, Math.max(-MAX_PITCH, camera.pitch));
    });

    element.addEventListener('wheel', function(e) {
      if (!camera.active) {
        return;
      }
      e.preventDefault();
      var factor = Math.pow(1.15, e.deltaY > 0 ? -1 : 1);
      camera.speed = Math.min(MAX_SPEED, Math.max(MIN_SPEED, camera.speed * factor));
    }, {passive: false});

    window.addEventListener('keydown', function(e) {
      if (!camera.active) {
        return;
      }
      keys[e.key.toLowerCase()] = true;
      keys.shift = e.shiftKey;
    });

    window.addEventListener('keyup', function(e) {
      keys[e.key.toLowerCase()] = false;
      keys.shift = e.shiftKey;
    });

    return camera;
  }

  return createCamera;

}));
