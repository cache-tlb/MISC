/*
 * The control panel, built with lil-gui (vendor/lil-gui.umd.js, MIT).
 *
 * Controls bind straight to the scene state and call back so the matrices get
 * rebuilt. Angles are stored in radians but presented in degrees, which is
 * what the small accessor proxies below are for.
 *
 * @module ui
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./scene.js', './lil-gui.umd.js'], factory);
  } else {
    // Browser globals
    root.ui = factory(root.scene, root.lil);
  }
}(this, function(scene, lil) {
  "use strict";

  var DEG = Math.PI / 180;
  var RAD = 180 / Math.PI;

  var gui = null;
  var info = {fps: 0};

  /**
   * Wraps a radian field so lil-gui can edit it in degrees.
   *
   * lil-gui writes through object[property], so a one-property object with an
   * accessor is all that is needed. The target is captured by reference, which
   * is why reset mutates the state in place rather than replacing it.
   *
   * @param {Object|number[]} target the object or array holding the angle
   * @param {string|number} key property name or array index
   * @return {Object} an object whose "value" reads and writes degrees
   */
  function degrees(target, key) {
    var proxy = {};
    Object.defineProperty(proxy, 'value', {
      // Rounded, because the radian round trip turns 30 degrees into
      // 29.999999999999996 and lil-gui would display every digit of it.
      get: function() { return Math.round(target[key] * RAD * 1e6) / 1e6; },
      set: function(v) { target[key] = v * DEG; },
      enumerable: true,
    });
    return proxy;
  }

  /**
   * Adds x, y and z controls for a vector.
   * @param {Object} folder the lil-gui folder
   * @param {number[]} vector the three components, bound by reference
   * @param {Object} options min, max, step and whether values are angles
   * @param {function} onChange called after any component changes
   */
  function addVector(folder, vector, options, onChange) {
    ['x', 'y', 'z'].forEach(function(axis, index) {
      var controller = options.degrees ?
          folder.add(degrees(vector, index), 'value',
              options.min, options.max, options.step) :
          folder.add(vector, String(index),
              options.min, options.max, options.step);
      controller.name(axis).onChange(onChange);
    });
  }

  /**
   * Copies source values into target without replacing any object or array,
   * so every reference lil-gui captured stays live.
   * @param {Object} target the object to update
   * @param {Object} source the values to copy in
   */
  function assignInto(target, source) {
    for (var key in source) {
      if (!Object.prototype.hasOwnProperty.call(source, key)) {
        continue;
      }
      var value = source[key];
      if (Array.isArray(value) && Array.isArray(target[key])) {
        for (var i = 0; i < value.length; i++) {
          target[key][i] = value[i];
        }
      } else if (value && typeof value === 'object' && target[key] &&
                 typeof target[key] === 'object') {
        assignInto(target[key], value);
      } else {
        target[key] = value;
      }
    }
  }

  /**
   * Builds the panel.
   * @param {Object} state the scene state
   * @param {function} onChange called whenever anything changes
   * @memberOf module:ui
   */
  function create(state, onChange) {
    if (gui) {
      gui.destroy();
    }
    // Hosted in our own container rather than lil-gui's autoPlace, which
    // pins a fixed, full-max-height layer over the canvas; that layer's
    // opaque background stopped the WebGL content compositing beneath it.
    gui = new lil.GUI({
      container: document.getElementById('gui'),
      title: 'Participating Media',
      width: 300,
    });

    var light = gui.addFolder('Directional light');
    light.add(degrees(state.light, 'yaw'), 'value', -180, 180, 1)
        .name('yaw').onChange(onChange);
    light.add(degrees(state.light, 'pitch'), 'value', -89, 89, 1)
        .name('pitch').onChange(onChange);
    light.add(state.light, 'intensity', 0, 10, 0.05)
        .name('intensity').onChange(onChange);

    var ambient = gui.addFolder('Ambient');
    ambient.add(state.ambient, 'intensity', 0, 2, 0.01)
        .name('intensity').onChange(onChange);

    var teapot = gui.addFolder('Teapot');
    teapot.add(state.teapot, 'visible').name('visible').onChange(onChange);
    var teapotPos = teapot.addFolder('position');
    addVector(teapotPos, state.teapot.position,
        {min: -40, max: 40, step: 0.5}, onChange);
    var teapotRot = teapot.addFolder('rotation');
    addVector(teapotRot, state.teapot.rotation,
        {min: -180, max: 180, step: 1, degrees: true}, onChange);
    teapot.add(state.teapot, 'scale', 0.05, 1.0, 0.01)
        .name('scale').onChange(onChange);

    var medium = gui.addFolder('Medium');
    medium.add(state.medium, 'visible').name('visible').onChange(onChange);
    var mediumPos = medium.addFolder('position');
    addVector(mediumPos, state.medium.position,
        {min: -50, max: 50, step: 0.5}, onChange);
    var mediumRot = medium.addFolder('rotation');
    addVector(mediumRot, state.medium.rotation,
        {min: -180, max: 180, step: 1, degrees: true}, onChange);
    medium.add(state.medium, 'scale', 5, 80, 0.5)
        .name('scale').onChange(onChange);
    medium.add(state.medium, 'density', 0, 3, 0.01)
        .name('density').onChange(onChange);
    medium.addColor(state.medium, 'color')
        .name('colour').onChange(onChange);

    // Nested folders start closed so the panel opens at a readable height.
    teapotPos.close();
    teapotRot.close();
    mediumPos.close();
    mediumRot.close();

    var actions = {
      reset: function() {
        var fresh = scene.createState(state.grid);
        // In place, so the arrays and objects lil-gui bound to stay the same.
        ['light', 'ambient', 'teapot', 'medium'].forEach(function(key) {
          assignInto(state[key], fresh[key]);
        });
        gui.controllersRecursive().forEach(function(c) { c.updateDisplay(); });
        onChange();
      },
    };

    gui.add(info, 'fps').name('fps').disable().listen();
    gui.add(actions, 'reset').name('Reset defaults');
  }

  /**
   * Updates the frame rate readout.
   * @param {number} fps frames per second
   * @memberOf module:ui
   */
  function setFps(fps) {
    info.fps = Math.round(fps);
  }

  return {
    create: create,
    setFps: setFps,
  };

}));
