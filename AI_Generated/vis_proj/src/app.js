/*
 * Wiring for the stereographic projection demo: context creation, the UI panel,
 * the camera and the frame loop. All of the drawing lives in renderer.js.
 *
 * @module app
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./m4.js', './webgl-utils.js', './camera.js', './renderer.js'], factory);
  } else {
    // Browser globals
    root.startApp = factory(root.m4, root.webglUtils, root.createCamera, root.createRenderer);
  }
}(this, function(m4, webglUtils, createCamera, createRenderer) {
  "use strict";

  var DEG_TO_RAD = Math.PI / 180;

  var DEFAULT_TRANSFORM = {
    // M sits behind the sphere by default, far enough that it never touches it.
    // Slide it forward through z = 0 to watch P blow up toward infinity.
    translate: [0, 0, -1.8],
    rotate: [0, 0, 0],
    scale: [1, 1, 1],
  };

  // h is open on both ends by definition, so the slider stops just short of 10.
  var DEFAULT_PROJECTION_HEIGHT = 1;

  var SLIDERS = [
    {title: '平移 Translate', key: 'translate', unit: '', min: -4, max: 4, step: 0.01},
    {title: '旋转 Rotate', key: 'rotate', unit: '°', min: -180, max: 180, step: 1},
    {title: '缩放 Scale', key: 'scale', unit: '×', min: 0.1, max: 3, step: 0.01},
    // Titles get upper cased by the stylesheet, so the variable name lives on
    // the row label instead of in the title.
    {title: '投影中心 Center', key: 'projectionHeight', unit: '', min: 0.05, max: 9.95,
     step: 0.05, axes: ['h'], scalar: true},
  ];

  var TOGGLES = [
    {key: 'm', label: 'M · 平面网格', swatch: '#f0f0ff'},
    {key: 's', label: 'S · 球面投影', swatch: '#ffbc52'},
    {key: 'p', label: 'P · 球极投影', swatch: '#66dbff'},
    {key: 'wireframe', label: '线框 (16×16)', swatch: null},
    {key: 'rays', label: '投影射线', swatch: null},
    {key: 'helpers', label: '参考体 (球 / 轴 / z = 0)', swatch: null},
  ];

  /**
   * Boots the demo.
   * @memberOf module:app
   */
  function startApp() {
    var canvas = document.getElementById('canvas');
    var gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      depth: true,
      powerPreference: 'high-performance',
    });

    if (!gl) {
      showError('这个浏览器没有可用的 WebGL2 上下文。');
      return;
    }

    var state = {
      translate: DEFAULT_TRANSFORM.translate.slice(),
      rotate: DEFAULT_TRANSFORM.rotate.slice(),
      scale: DEFAULT_TRANSFORM.scale.slice(),
      projectionHeight: DEFAULT_PROJECTION_HEIGHT,
      show: {m: true, s: true, p: true, wireframe: false, rays: true, helpers: true},
    };

    var renderer = createRenderer(gl, {
      halfSize: 2,
      segments: 16,
      rayStride: 4,
      checkerScale: 16,
    });

    // Off to the side rather than straight down -z: M, S and P are stacked along
    // the z axis, so an oblique view is the only one that separates all three.
    var camera = createCamera(canvas, {
      position: [4.2, 1.5, 2.8],
      yaw: 0.905,
      pitch: -0.273,
      speed: 3.2,
    });

    var lightDirection = m4.normalize([0.45, 0.75, 0.55]);
    var world = m4.identity();
    var projection = m4.identity();
    var view = m4.identity();

    var sync = buildUi(state);
    sync();

    document.getElementById('stat-vertices').textContent = renderer.numVertices;
    document.getElementById('stat-triangles').textContent = renderer.numTriangles;

    var fpsCounter = createFpsCounter();
    var then = 0;

    function frame(now) {
      now *= 0.001;
      var deltaTime = Math.min(now - then, 0.1);
      then = now;

      camera.update(deltaTime);
      webglUtils.resizeCanvasToDisplaySize(canvas, window.devicePixelRatio);

      var aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
      m4.perspective(60 * DEG_TO_RAD, aspect, 0.05, 1000, projection);
      camera.getViewMatrix(view);
      composeWorld(state, world);

      renderer.render({
        projection: projection,
        view: view,
        world: world,
        projectionHeight: state.projectionHeight,
        cameraPosition: camera.position,
        lightDirection: lightDirection,
        show: state.show,
      });

      updateHud(camera, fpsCounter.sample(deltaTime));
      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  }

  /**
   * Builds M's model matrix: translate, then rotate around x, y and z, then scale.
   * @param {Object} state the UI state
   * @param {Matrix4} dst matrix to fill
   * @return {Matrix4} dst
   */
  function composeWorld(state, dst) {
    m4.translation(state.translate[0], state.translate[1], state.translate[2], dst);
    m4.xRotate(dst, state.rotate[0] * DEG_TO_RAD, dst);
    m4.yRotate(dst, state.rotate[1] * DEG_TO_RAD, dst);
    m4.zRotate(dst, state.rotate[2] * DEG_TO_RAD, dst);
    m4.scale(dst, state.scale[0], state.scale[1], state.scale[2], dst);
    return dst;
  }

  /**
   * Creates every control in the panel and wires it to `state`.
   * @param {Object} state the UI state, mutated in place
   * @return {function} call to push `state` back into the controls
   */
  function buildUi(state) {
    var transforms = document.getElementById('transform-controls');
    var toggles = document.getElementById('display-controls');
    var refresh = [];

    SLIDERS.forEach(function(group) {
      var section = document.createElement('div');
      section.className = 'group';
      section.appendChild(labelled('h3', group.title));

      (group.axes || ['X', 'Y', 'Z']).forEach(function(axisName, axis) {
        var row = document.createElement('label');
        row.className = 'slider';
        row.appendChild(labelled('span', axisName));

        var input = document.createElement('input');
        input.type = 'range';
        input.min = group.min;
        input.max = group.max;
        input.step = group.step;

        var readout = labelled('output', '');
        function push() {
          var value = parseFloat(input.value);
          if (group.scalar) {
            state[group.key] = value;
          } else {
            state[group.key][axis] = value;
          }
          readout.textContent = formatNumber(value, group.step) + group.unit;
        }
        input.addEventListener('input', push);
        refresh.push(function() {
          input.value = group.scalar ? state[group.key] : state[group.key][axis];
          push();
        });

        row.appendChild(input);
        row.appendChild(readout);
        section.appendChild(row);
      });

      transforms.appendChild(section);
    });

    TOGGLES.forEach(function(toggle) {
      var row = document.createElement('label');
      row.className = 'toggle';

      var input = document.createElement('input');
      input.type = 'checkbox';
      input.addEventListener('change', function() {
        state.show[toggle.key] = input.checked;
      });
      refresh.push(function() {
        input.checked = state.show[toggle.key];
      });
      row.appendChild(input);

      if (toggle.swatch) {
        var swatch = document.createElement('i');
        swatch.className = 'swatch';
        swatch.style.background = toggle.swatch;
        row.appendChild(swatch);
      }

      row.appendChild(labelled('span', toggle.label));
      toggles.appendChild(row);
    });

    function sync() {
      refresh.forEach(function(fn) { fn(); });
    }

    document.getElementById('reset').addEventListener('click', function() {
      state.translate = DEFAULT_TRANSFORM.translate.slice();
      state.rotate = DEFAULT_TRANSFORM.rotate.slice();
      state.scale = DEFAULT_TRANSFORM.scale.slice();
      state.projectionHeight = DEFAULT_PROJECTION_HEIGHT;
      sync();
    });

    var panel = document.getElementById('panel');
    document.getElementById('collapse').addEventListener('click', function() {
      panel.classList.toggle('collapsed');
    });

    return sync;
  }

  /**
   * @param {string} tag element name
   * @param {string} text its text content
   * @return {HTMLElement} the element
   */
  function labelled(tag, text) {
    var element = document.createElement(tag);
    element.textContent = text;
    return element;
  }

  /**
   * @param {number} value the number
   * @param {number} step the slider step, which decides the precision shown
   * @return {string} the formatted number
   */
  function formatNumber(value, step) {
    return value.toFixed(step < 1 ? 2 : 0);
  }

  /**
   * A frame rate readout smoothed enough to be readable.
   * @return {Object} a counter with a `sample` method
   */
  function createFpsCounter() {
    var fps = 0;
    return {
      sample: function(deltaTime) {
        if (deltaTime > 0) {
          fps += (1 / deltaTime - fps) * 0.08;
        }
        return fps;
      },
    };
  }

  function updateHud(camera, fps) {
    document.getElementById('stat-fps').textContent = fps.toFixed(0);
    document.getElementById('stat-camera').textContent =
        camera.position[0].toFixed(1) + ', ' +
        camera.position[1].toFixed(1) + ', ' +
        camera.position[2].toFixed(1);
    document.getElementById('stat-speed').textContent = camera.speed.toFixed(1);
  }

  function showError(message) {
    var element = document.getElementById('error');
    element.textContent = message;
    element.hidden = false;
  }

  return startApp;

}));
