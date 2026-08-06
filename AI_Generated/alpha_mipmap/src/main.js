/*
 * Wiring for the alpha mask mipmap demo: loads the assets, builds the three
 * mask textures being compared, hooks up the UI, and runs the render loop.
 *
 * @module main
 */
(function() {
  "use strict";

  var ASSET_DIR = 'assets/';
  var MODEL_URL = ASSET_DIR + 'Tree.obj';
  // The one texture in this model whose alpha channel is an alpha test mask.
  var MASK_MAP = 'DB2X2_L01.png';
  // The model is authored ~7.6 units tall; scale it to this so it reads well
  // against the 100 unit room and its 20 unit walls.
  var TREE_HEIGHT = 15;
  // Roughly the middle of the canopy, used as the look-at point for presets.
  var TREE_FOCUS = [0, TREE_HEIGHT * 0.55, 0];

  var MODE_NO_MIPMAP = 0;
  var MODE_DEFAULT = 1;
  var MODE_CORRECTED = 2;

  // Mip levels shown as thumbnails in the details panel.
  var PREVIEW_LEVELS = [4, 5, 6, 7, 8];

  // The camera looks along -z, so these all sit on the +z side of the tree.
  var VIEW_PRESETS = {
    near: {position: [0, 5, 13], speed: 8},
    mid: {position: [0, 9, 34], speed: 18},
    // Far enough back that the mask is sampled from mip level 7-8, which is
    // where the plain box filtered chain falls apart.
    far: {position: [-160, 45, 160], speed: 60},
  };

  var elements = {
    canvas: document.getElementById('canvas'),
    modes: document.getElementById('modes'),
    threshold: document.getElementById('threshold'),
    thresholdValue: document.getElementById('threshold-value'),
    split: document.getElementById('split'),
    splitLabels: document.getElementById('split-labels'),
    statsBody: document.getElementById('stats-body'),
    statsNote: document.getElementById('stats-note'),
    mipPreview: document.getElementById('mip-preview'),
    fps: document.getElementById('fps'),
    distance: document.getElementById('distance'),
    speed: document.getElementById('speed'),
    loading: document.getElementById('loading'),
    loadingText: document.getElementById('loading-text'),
    panel: document.getElementById('panel'),
    panelToggle: document.getElementById('panel-toggle'),
  };

  var state = {
    mode: MODE_CORRECTED,
    alphaRef: parseFloat(elements.threshold.value),
    split: false,
  };

  // Filled in during init.
  var renderer = null;
  var camera = null;
  var mask = null;  // {levels, cdfs} from alphaMipmap.prepare
  var maskTextures = [];
  var defaultLevels = null;
  var correctedLevels = null;
  var previewCanvases = {};

  function setLoading(text) {
    elements.loadingText.textContent = text;
  }

  function fail(error) {
    console.error(error);
    elements.loading.classList.remove('hidden');
    elements.loadingText.innerHTML = '';
    var div = document.createElement('div');
    div.className = 'err';
    div.textContent = '加载失败：' + (error && error.message ? error.message : error) +
        '\n\n如果是 fetch / CORS 错误，请通过本地 HTTP 服务器打开本页面，' +
        '例如在本目录执行：\n\npython -m http.server 8080\n\n然后访问 http://localhost:8080/';
    elements.loadingText.appendChild(div);
  }

  // ---------------------------------------------------------------- textures

  /**
   * Converts one float mip level into the {width, height, data} form the
   * renderer uploads, applying an alpha scale on the way.
   * @param {Object} level a level from alphaMipmap
   * @param {number} scale the alpha scale for this level
   * @return {Object} an uploadable level
   */
  function toUploadable(level, scale) {
    return {
      width: level.width,
      height: level.height,
      data: alphaMipmap.toUint8(level, scale),
    };
  }

  /**
   * Re-solves the coverage preserving chain for the current threshold and
   * pushes it to the GPU. Cheap enough to run straight from the slider's input
   * event because the per level coverage histograms are computed only once.
   */
  function refreshCorrectedMask() {
    var solutions = alphaMipmap.solveChain(mask.cdfs, state.alphaRef);
    correctedLevels = mask.levels.map(function(level, i) {
      return toUploadable(level, solutions[i].scale);
    });
    renderer.updateMaskTexture(maskTextures[MODE_CORRECTED], correctedLevels);
    updateStats(solutions);
    updateMipPreview();
  }

  // --------------------------------------------------------------------- UI

  function updateStats(solutions) {
    // Level 0 is never rescaled, so its coverage is both the measured baseline
    // and the value every other level is being steered toward.
    var target = solutions[0].coverageBefore;

    // A level is off target if it has drifted more than 10% away, in either
    // direction: a low threshold makes the box filter fatten the mask, a high
    // one makes it erode.
    function className(coverage) {
      if (!target) {
        return '';
      }
      return Math.abs(coverage - target) / target > 0.1 ? 'bad' : 'good';
    }

    var rows = solutions.map(function(s) {
      var level = mask.levels[s.level];
      return '<tr>' +
          '<td>' + s.level + '</td>' +
          '<td>' + level.width + '&times;' + level.height + '</td>' +
          '<td class="' + className(s.coverageBefore) + '">' +
              (s.coverageBefore * 100).toFixed(1) + '%</td>' +
          '<td>' + s.scale.toFixed(3) + '</td>' +
          '<td class="' + className(s.coverageAfter) + '">' +
              (s.coverageAfter * 100).toFixed(1) + '%</td>' +
          '</tr>';
    });
    elements.statsBody.innerHTML = rows.join('');

    // Once a level has averaged down to a nearly uniform value its coverage
    // can only be 0% or 100%, so say so instead of pretending it was fixed.
    var unfixable = solutions.filter(function(s) {
      return target > 0 && Math.abs(s.coverageAfter - target) / target > 0.1;
    }).length;

    elements.statsNote.textContent =
        'Level 0 的 ' + (target * 100).toFixed(1) + '% 就是目标覆盖率。' +
        '覆盖率是按 GPU 实际采样的方式统计的（双线性超采样），' +
        '列出的是量化成 8bit 上传后的真实值。' +
        (unfixable
            ? ' 最粗的 ' + unfixable + ' 级已经退化成接近常数，覆盖率只能是 0% 或 100%，' +
              '缩放 alpha 已无法还原——这是该方法的固有上限。'
            : ' 每一级都被拉回了目标值。');
  }

  function drawLevelToCanvas(canvas, level) {
    canvas.width = level.width;
    canvas.height = level.height;
    var ctx = canvas.getContext('2d');
    var imageData = ctx.createImageData(level.width, level.height);
    for (var i = 0; i < level.data.length; ++i) {
      var v = level.data[i];
      imageData.data[i * 4 + 0] = v;
      imageData.data[i * 4 + 1] = v;
      imageData.data[i * 4 + 2] = v;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  function buildMipPreview() {
    var rows = [
      {key: 'default', tag: '默认'},
      {key: 'corrected', tag: '修正'},
    ];
    rows.forEach(function(row) {
      var div = document.createElement('div');
      div.className = 'row';
      var tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = row.tag;
      div.appendChild(tag);
      previewCanvases[row.key] = PREVIEW_LEVELS.map(function(levelIndex) {
        var canvas = document.createElement('canvas');
        canvas.title = 'level ' + levelIndex;
        div.appendChild(canvas);
        return canvas;
      });
      elements.mipPreview.appendChild(div);
    });
  }

  function updateMipPreview() {
    PREVIEW_LEVELS.forEach(function(levelIndex, i) {
      if (levelIndex >= defaultLevels.length) {
        return;
      }
      drawLevelToCanvas(previewCanvases.default[i], defaultLevels[levelIndex]);
      drawLevelToCanvas(previewCanvases.corrected[i], correctedLevels[levelIndex]);
    });
  }

  function setMode(mode) {
    state.mode = mode;
    Array.prototype.forEach.call(
        elements.modes.querySelectorAll('button'), function(button) {
          button.setAttribute('aria-pressed',
              String(Number(button.dataset.mode) === mode));
        });
  }

  function setSplit(split) {
    state.split = split;
    elements.split.checked = split;
    elements.splitLabels.classList.toggle('visible', split);
  }

  function applyPreset(name) {
    var preset = VIEW_PRESETS[name];
    camera.position = preset.position.slice();
    camera.speed = preset.speed;
    var dx = TREE_FOCUS[0] - camera.position[0];
    var dy = TREE_FOCUS[1] - camera.position[1];
    var dz = TREE_FOCUS[2] - camera.position[2];
    // Inverts camera.getForward(), which is (-sin(yaw)cos(pitch), sin(pitch),
    // -cos(yaw)cos(pitch)) in the right handed basis.
    camera.yaw = Math.atan2(-dx, -dz);
    camera.pitch = Math.atan2(dy, Math.sqrt(dx * dx + dz * dz));
  }

  function bindUi() {
    elements.modes.addEventListener('click', function(e) {
      var button = e.target.closest('button[data-mode]');
      if (button) {
        setMode(Number(button.dataset.mode));
      }
    });

    elements.threshold.addEventListener('input', function() {
      state.alphaRef = parseFloat(elements.threshold.value);
      elements.thresholdValue.textContent = state.alphaRef.toFixed(2);
      refreshCorrectedMask();
    });

    elements.split.addEventListener('change', function() {
      setSplit(elements.split.checked);
    });

    document.querySelectorAll('[data-preset]').forEach(function(button) {
      button.addEventListener('click', function() {
        applyPreset(button.dataset.preset);
      });
    });

    elements.panelToggle.addEventListener('click', function() {
      elements.panel.classList.toggle('collapsed');
    });

    window.addEventListener('keydown', function(e) {
      if (e.key === '1') { setMode(MODE_NO_MIPMAP); }
      else if (e.key === '2') { setMode(MODE_DEFAULT); }
      else if (e.key === '3') { setMode(MODE_CORRECTED); }
      else if (e.key === 'c' || e.key === 'C') { setSplit(!state.split); }
      else if (e.key === 'h' || e.key === 'H') {
        elements.panel.classList.toggle('collapsed');
      }
    });
  }

  // ------------------------------------------------------------------ assets

  /**
   * Loads every map_Kd referenced by the parsed materials and turns them into
   * textures, plus the raw pixels of the alpha mask texture.
   * @param {Object} model from objLoader.loadObj
   * @return {Promise<{materialTextures: Object, maskPixels: Object}>} the result
   */
  function buildMaterials(model) {
    var gl = renderer.gl;
    var names = Object.keys(model.materials);
    var jobs = names.map(function(name) {
      var mapKd = model.materials[name].map_Kd;
      if (!mapKd) {
        return Promise.resolve({name: name, texture: null, isMask: false});
      }
      var isMask = mapKd === MASK_MAP;
      return objLoader.loadImage(ASSET_DIR + mapKd).then(function(image) {
        return {
          name: name,
          // The leaf atlas fills 0..1 exactly, so clamp it; the bark tiles.
          texture: renderer.createAlbedoTexture(
              image, isMask ? gl.CLAMP_TO_EDGE : gl.REPEAT),
          isMask: isMask,
        };
      });
    });

    return Promise.all(jobs).then(function(results) {
      var materialTextures = {};
      results.forEach(function(result) {
        materialTextures[result.name] = {
          albedo: result.texture,
          useMask: result.isMask,
          // In this model the alpha masked material is also the one built from
          // single sided leaf cards, so it needs two sided shading too.
          twoSided: result.isMask,
        };
      });
      return materialTextures;
    });
  }

  // -------------------------------------------------------------- main loop

  var lastTime = 0;
  var fpsAccumulator = 0;
  var fpsFrames = 0;

  function frame(now) {
    var deltaTime = lastTime ? Math.min(0.1, (now - lastTime) / 1000) : 0;
    lastTime = now;

    camera.update(deltaTime);

    renderer.render({
      camera: camera,
      alphaRef: state.alphaRef,
      maskTexture: state.split ? undefined : maskTextures[state.mode],
      maskTextures: state.split ? maskTextures : undefined,
    });

    fpsAccumulator += deltaTime;
    fpsFrames += 1;
    if (fpsAccumulator >= 0.5) {
      elements.fps.textContent = (fpsFrames / fpsAccumulator).toFixed(0);
      var dx = camera.position[0] - TREE_FOCUS[0];
      var dy = camera.position[1] - TREE_FOCUS[1];
      var dz = camera.position[2] - TREE_FOCUS[2];
      elements.distance.textContent = Math.sqrt(dx * dx + dy * dy + dz * dz).toFixed(0);
      elements.speed.textContent = camera.speed.toFixed(0);
      fpsAccumulator = 0;
      fpsFrames = 0;
    }

    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------- init

  function init() {
    try {
      renderer = createRenderer(elements.canvas);
    } catch (e) {
      fail(e);
      return;
    }

    camera = createCamera(elements.canvas);
    applyPreset('mid');

    setLoading('正在加载模型 Tree.obj …');
    objLoader.loadObj(MODEL_URL).then(function(model) {
      setLoading('正在加载贴图 …');
      return Promise.all([
        buildMaterials(model),
        objLoader.loadImageData(ASSET_DIR + MASK_MAP),
      ]).then(function(results) {
        var materialTextures = results[0];
        var maskPixels = results[1];

        // Normalise the model to a known height rather than trusting whatever
        // units it happens to be authored in.
        var bounds = objLoader.getBounds(model.geometries);
        var scale = TREE_HEIGHT / (bounds.max[1] - bounds.min[1]);
        renderer.setModel(model.geometries, materialTextures,
            m4.scaling(scale, scale, scale));

        setLoading('正在生成 alpha mipmap …');
        // Yield once so the message actually paints before the (~100ms)
        // histogram pass blocks the main thread.
        return new Promise(function(resolve) {
          requestAnimationFrame(function() { resolve(maskPixels); });
        });
      });
    }).then(function(maskPixels) {
      var start = performance.now();
      mask = alphaMipmap.prepare(maskPixels.data, maskPixels.width, maskPixels.height);
      console.log('alpha mip chain + coverage histograms in',
          (performance.now() - start).toFixed(1), 'ms');

      defaultLevels = mask.levels.map(function(level) {
        return toUploadable(level, 1);
      });

      // 1. No mipmap at all: level 0 only.
      // 2. Plain box filtered chain, ie. what generateMipmap would give.
      // 3. Same chain, alpha rescaled per level to preserve coverage. Its
      //    contents are filled in by refreshCorrectedMask below.
      maskTextures[MODE_NO_MIPMAP] = renderer.createMaskTexture([defaultLevels[0]]);
      maskTextures[MODE_DEFAULT] = renderer.createMaskTexture(defaultLevels);
      maskTextures[MODE_CORRECTED] = renderer.createMaskTexture(defaultLevels);

      buildMipPreview();
      refreshCorrectedMask();

      setMode(state.mode);
      setSplit(state.split);
      elements.thresholdValue.textContent = state.alphaRef.toFixed(2);
      bindUi();

      // Handy from the console when poking at the demo.
      window.demo = {renderer: renderer, camera: camera, state: state, mask: mask};

      elements.loading.classList.add('hidden');
      requestAnimationFrame(frame);
    }).catch(fail);
  }

  init();

}());
