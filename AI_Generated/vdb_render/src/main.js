/*
 * Bootstrap: create the context, load the assets, run the frame loop.
 *
 * @module main
 */
(function() {
  "use strict";

  var canvas = document.getElementById('canvas');
  var overlay = document.getElementById('overlay');
  var overlayTitle = document.getElementById('overlay-title');
  var overlayText = document.getElementById('overlay-text');
  var barFill = document.getElementById('bar-fill');
  var bar = document.getElementById('bar');

  var VDB_URL = 'assets/explosion.vdb';
  var OBJ_URL = 'assets/teapot/teapot.obj';
  var PNG_URL = 'assets/teapot/default.png';

  /**
   * Shows a terminal error on the overlay and stops.
   * @param {string} title short headline
   * @param {string} detail what went wrong and what to do about it
   */
  function fail(title, detail) {
    overlay.className = 'error';
    overlayTitle.textContent = title;
    overlayText.innerHTML = detail;
    bar.style.display = 'none';
    throw new Error(title + ' - ' + detail.replace(/<[^>]*>/g, ''));
  }

  function setProgress(text, fraction) {
    overlayText.textContent = text;
    barFill.style.width = Math.round(fraction * 100) + '%';
  }

  var gl = canvas.getContext('webgl2', {antialias: false, depth: true});
  if (!gl) {
    fail('WebGL2 is not available',
        'This demo needs a WebGL2 context. Your browser or GPU driver did not ' +
        'provide one. Try a current Chrome, Edge or Firefox.');
  }
  if (!gl.getExtension('EXT_color_buffer_float')) {
    fail('EXT_color_buffer_float is missing',
        'This demo renders to half float targets, which needs the ' +
        '<code>EXT_color_buffer_float</code> extension. Your WebGL2 ' +
        'implementation does not expose it.');
  }
  // Not fatal: without it the volume texture just filters a little worse.
  gl.getExtension('OES_texture_float_linear');

  /**
   * Fetches an ArrayBuffer, reporting download progress when the server sends
   * a content length.
   * @param {string} url where to fetch from
   * @param {function(number)} onProgress receives 0..1
   * @return {Promise<ArrayBuffer>} the bytes
   */
  function fetchBuffer(url, onProgress) {
    return fetch(url).then(function(response) {
      if (!response.ok) {
        fail('Could not load ' + url,
            'The server answered ' + response.status + ' ' + response.statusText +
            '. Serve this directory over HTTP, for example ' +
            '<code>python -m http.server 8000</code>, rather than opening ' +
            'index.html as a file:// URL.');
      }
      var total = Number(response.headers.get('content-length')) || 0;
      if (!total || !response.body) {
        return response.arrayBuffer();
      }
      var reader = response.body.getReader();
      var chunks = [];
      var received = 0;
      return (function pump() {
        return reader.read().then(function(result) {
          if (result.done) {
            var merged = new Uint8Array(received);
            var at = 0;
            for (var i = 0; i < chunks.length; i++) {
              merged.set(chunks[i], at);
              at += chunks[i].length;
            }
            return merged.buffer;
          }
          chunks.push(result.value);
          received += result.value.length;
          onProgress(received / total);
          return pump();
        });
      }());
    }).catch(function(e) {
      if (e instanceof TypeError) {
        fail('Could not load ' + url,
            'The fetch failed outright. Serve this directory over HTTP, for ' +
            'example <code>python -m http.server 8000</code>, rather than ' +
            'opening index.html as a file:// URL.');
      }
      throw e;
    });
  }

  function fetchText(url) {
    return fetch(url).then(function(response) {
      if (!response.ok) {
        fail('Could not load ' + url,
            'The server answered ' + response.status + ' ' + response.statusText + '.');
      }
      return response.text();
    });
  }

  function fetchImage(url) {
    return new Promise(function(resolve, reject) {
      var image = new Image();
      image.onload = function() { resolve(image); };
      image.onerror = function() { reject(new TypeError('image ' + url)); };
      image.src = url;
    });
  }

  /** Yields to the browser so the overlay repaints before a blocking parse. */
  function nextFrame() {
    return new Promise(function(resolve) {
      requestAnimationFrame(function() { setTimeout(resolve, 0); });
    });
  }

  setProgress('Fetching assets…', 0);

  Promise.all([
    fetchBuffer(VDB_URL, function(f) {
      setProgress('Downloading volume… ' + Math.round(f * 100) + '%', f * 0.8);
    }),
    fetchText(OBJ_URL),
    fetchImage(PNG_URL),
  ]).then(function(assets) {
    setProgress('Decoding OpenVDB…', 0.85);
    return nextFrame().then(function() {
      return assets;
    });
  }).then(function(assets) {
    // Both of these block the main thread for a moment; the overlay above has
    // already been painted so the page does not look hung.
    var grid = vdbLoader.load(assets[0], {maxResolution: 256});
    var mesh = objLoader.parse(assets[1]);

    console.log('vdb: ' + grid.name + ' ' + grid.dims.join('x') +
        ', ' + grid.activeVoxelCount.toLocaleString() + ' active voxels in ' +
        grid.leafCount.toLocaleString() + ' leaves, maxDensity ' +
        grid.maxDensity.toFixed(4));
    console.log('obj: ' + (mesh.position.length / 3) + ' vertices, ' +
        mesh.indices.length + ' indices');

    setProgress('Building scene…', 0.95);
    return nextFrame().then(function() {
      start(grid, mesh, assets[2]);
    });
  }).catch(function(e) {
    console.error(e);
    if (overlay.className !== 'error') {
      fail('Something went wrong', String(e && e.message ? e.message : e));
    }
  });

  /**
   * Builds the renderer and runs the frame loop.
   * @param {Object} grid the decoded density grid
   * @param {Object} mesh the teapot
   * @param {HTMLImageElement} image the teapot texture
   */
  function start(grid, mesh, image) {
    var state = scene.createState(grid);
    var renderer = rendererModule.create(gl, {
      grid: grid,
      mesh: mesh,
      image: image,
    });

    var camera = createCamera(canvas, {
      position: [0, 26, 88],
      yaw: 0,
      pitch: -0.13,
      speed: 26,
    });

    ui.create(state, function() {
      scene.updateMatrices(state);
    });
    scene.updateMatrices(state);

    overlay.className = 'hidden';

    var last = 0;
    var fpsAccum = 0;
    var fpsFrames = 0;

    function frame(now) {
      var seconds = now * 0.001;
      // Clamp, so returning to a background tab does not teleport the camera.
      var dt = last ? Math.min(0.1, seconds - last) : 0;
      last = seconds;

      fpsAccum += dt;
      fpsFrames++;
      if (fpsAccum >= 0.5) {
        ui.setFps(fpsFrames / fpsAccum);
        fpsAccum = 0;
        fpsFrames = 0;
      }

      camera.update(dt);
      webglUtils.resizeCanvasToDisplaySize(gl.canvas, window.devicePixelRatio);
      renderer.render(state, camera, seconds);

      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  }

}());
