/*
 * The four render passes.
 *
 *   0  shadow map    opaque casters only, into a depth texture
 *   1  opaque        forward shading into an HDR target plus a depth texture
 *   2  volume        half resolution ray march through the medium
 *   3  composite     depth aware upsample, combine, tonemap, to the canvas
 *
 * The medium shadows opaque surfaces analytically during pass 1, and the
 * opaque geometry shadows the medium by sampling the same shadow map inside
 * pass 2, which is what makes the interaction go both ways.
 *
 * @module rendererModule
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./m4.js', './webgl-utils.js', './gl-helpers.js', './shaders.js',
        './scene.js'], factory);
  } else {
    // Browser globals
    root.rendererModule = factory(root.m4, root.webglUtils, root.glHelpers,
        root.shaders, root.scene);
  }
}(this, function(m4, webglUtils, glHelpers, shaders, scene) {
  "use strict";

  var SHADOW_SIZE = 2048;
  var NEAR = 0.5;
  var FAR = 600;
  var VOLUME_STEPS = 64;
  var LIGHT_STEPS = 10;
  var OCTAVES = 3;

  /**
   * Builds the renderer.
   * @param {WebGL2RenderingContext} gl the context
   * @param {Object} resources grid, mesh and image
   * @return {Object} an object with a render method
   * @memberOf module:rendererModule
   */
  function create(gl, resources) {
    var depthProgram = glHelpers.createProgram(gl,
        shaders.depthVS, shaders.depthFS);
    var opaqueProgram = glHelpers.createProgram(gl,
        shaders.opaqueVS, shaders.opaqueFS);
    var volumeProgram = glHelpers.createProgram(gl,
        shaders.fullscreenVS, shaders.volumeFS);
    var compositeProgram = glHelpers.createProgram(gl,
        shaders.fullscreenVS, shaders.compositeFS);

    var room = scene.buildRoom(gl);
    var mesh = resources.mesh;
    var teapot = webglUtils.createBufferInfoFromArrays(gl, {
      position: {numComponents: 3, data: mesh.position},
      normal: {numComponents: 3, data: mesh.normal},
      texcoord: {numComponents: 2, data: mesh.texcoord},
      indices: {numComponents: 3, data: mesh.indices},
    });
    var teapotIndexType = (mesh.indices instanceof Uint32Array) ?
        gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    var quad = glHelpers.createFullscreenQuad(gl);

    var grid = resources.grid;
    var densityTexture = glHelpers.createTexture3D(gl, {
      width: grid.dims[0],
      height: grid.dims[1],
      depth: grid.dims[2],
      data: grid.data,
    });
    var teapotTexture = glHelpers.createTextureFromImage(gl, resources.image);

    var shadowTarget = glHelpers.createFramebuffer(gl, {
      depth: glHelpers.createDepthTexture(gl, SHADOW_SIZE, SHADOW_SIZE),
      width: SHADOW_SIZE,
      height: SHADOW_SIZE,
    });

    var opaqueTarget = null;
    var volumeTarget = null;

    // Reused each frame so the loop does not allocate.
    var viewMatrix = m4.identity();
    var projectionMatrix = m4.identity();
    var viewProjection = m4.identity();
    var inverseViewProjection = m4.identity();
    var lightViewMatrix = m4.identity();
    var lightViewProjection = m4.identity();
    var roomMatrix = m4.identity();
    var normalMatrix = m4.identity();

    /**
     * (Re)creates the size dependent targets.
     * @param {number} width drawing buffer width
     * @param {number} height drawing buffer height
     */
    function resizeTargets(width, height) {
      if (opaqueTarget && opaqueTarget.width === width &&
          opaqueTarget.height === height) {
        return;
      }
      glHelpers.deleteFramebuffer(gl, opaqueTarget);
      glHelpers.deleteFramebuffer(gl, volumeTarget);

      opaqueTarget = glHelpers.createFramebuffer(gl, {
        color: glHelpers.createTexture2D(gl, {
          width: width, height: height,
          internalFormat: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT,
        }),
        depth: glHelpers.createDepthTexture(gl, width, height),
        width: width,
        height: height,
      });

      var halfWidth = Math.max(1, width >> 1);
      var halfHeight = Math.max(1, height >> 1);
      volumeTarget = glHelpers.createFramebuffer(gl, {
        color: glHelpers.createTexture2D(gl, {
          width: halfWidth, height: halfHeight,
          internalFormat: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT,
        }),
        width: halfWidth,
        height: halfHeight,
      });
    }

    /**
     * Fits an orthographic light frustum around everything that matters.
     *
     * The scene corners are transformed into light space and the bounds taken
     * from there, which keeps the shadow map's resolution concentrated on the
     * scene however the light is angled.
     *
     * @param {Object} state the scene state
     * @return {Matrix4} the light view projection
     */
    function fitLightFrustum(state) {
      var dir = state.lightDirection;
      var corners = scene.sceneCorners(state, mesh.bbox);

      var centre = [0, 0, 0];
      var i;
      for (i = 0; i < corners.length; i++) {
        centre[0] += corners[i][0];
        centre[1] += corners[i][1];
        centre[2] += corners[i][2];
      }
      centre[0] /= corners.length;
      centre[1] /= corners.length;
      centre[2] /= corners.length;

      var radius = 0;
      for (i = 0; i < corners.length; i++) {
        radius = Math.max(radius, m4.distance(centre, corners[i]));
      }

      // lookAt degenerates when the view direction is parallel to up, which
      // happens whenever the light points straight down.
      var up = Math.abs(dir[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
      var eye = [
        centre[0] - dir[0] * radius * 2,
        centre[1] - dir[1] * radius * 2,
        centre[2] - dir[2] * radius * 2,
      ];
      m4.inverse(m4.lookAt(eye, centre, up), lightViewMatrix);

      var min = [Infinity, Infinity, Infinity];
      var max = [-Infinity, -Infinity, -Infinity];
      for (i = 0; i < corners.length; i++) {
        var p = m4.transformPoint(lightViewMatrix, corners[i]);
        for (var a = 0; a < 3; a++) {
          if (p[a] < min[a]) { min[a] = p[a]; }
          if (p[a] > max[a]) { max[a] = p[a]; }
        }
      }

      // near and far are distances along -z, and the near plane is pushed out
      // so casters just outside the fitted box still register.
      var near = -max[2] - 1.0;
      var far = -min[2] + 1.0;
      var ortho = m4.orthographic(min[0], max[0], min[1], max[1], near, far);
      return m4.multiply(ortho, lightViewMatrix, lightViewProjection);
    }

    /**
     * Draws the opaque geometry with whichever program is bound.
     * @param {Object} programInfo the program to draw with
     * @param {Object} state the scene state
     * @param {Object} extra uniforms shared by both draws
     */
    function drawOpaque(programInfo, state, extra) {
      var uniforms = {};
      var key;

      m4.identity(roomMatrix);
      for (key in extra) { uniforms[key] = extra[key]; }
      uniforms.u_model = roomMatrix;
      uniforms.u_normalMatrix = roomMatrix;
      uniforms.u_albedo = [1, 1, 1];
      uniforms.u_useChecker = 1;
      uniforms.u_useTexture = 0;
      uniforms.u_texture = teapotTexture;

      webglUtils.setBuffersAndAttributes(gl, programInfo.attribSetters, room);
      glHelpers.setUniforms(gl, programInfo, uniforms);
      gl.drawElements(gl.TRIANGLES, room.numElements, gl.UNSIGNED_SHORT, 0);

      // Hiding the teapot here covers both callers, so it stops being shaded
      // and stops casting a shadow in the same breath.
      if (!state.teapot.visible) {
        return;
      }

      m4.transpose(m4.inverse(state.teapotMatrix), normalMatrix);
      uniforms.u_model = state.teapotMatrix;
      uniforms.u_normalMatrix = normalMatrix;
      uniforms.u_albedo = [0.85, 0.80, 0.72];
      uniforms.u_useChecker = 0;
      uniforms.u_useTexture = 1;

      webglUtils.setBuffersAndAttributes(gl, programInfo.attribSetters, teapot);
      glHelpers.setUniforms(gl, programInfo, uniforms);
      gl.drawElements(gl.TRIANGLES, teapot.numElements, teapotIndexType, 0);
    }

    /** Uniforms describing the medium, shared by the opaque and volume passes. */
    function mediumUniforms(state) {
      return {
        u_density: densityTexture,
        u_mediumInverse: state.mediumInverse,
        u_maxDensity: state.grid.maxDensity,
        // Hiding the medium is a density of zero, which the shaders already
        // treat as "no medium at all": every transmittance march returns 1 and
        // the opaque pass skips them outright.
        u_densityScale: state.medium.visible ? state.medium.density : 0,
        u_mediumColor: state.medium.color,
      };
    }

    /** Uniforms describing the shadow map, shared by two passes. */
    function shadowUniforms(state) {
      return {
        u_shadowMap: shadowTarget.depth,
        u_lightViewProjection: lightViewProjection,
        u_shadowTexel: [1 / SHADOW_SIZE, 1 / SHADOW_SIZE],
        u_shadowBias: [0.0012, 0.0045],
        u_lightDirection: state.lightDirection,
      };
    }

    /**
     * Renders one frame.
     * @param {Object} state the scene state
     * @param {Object} camera the fly camera
     * @param {number} time seconds since start
     * @memberOf module:rendererModule
     */
    function render(state, camera, time) {
      var width = gl.drawingBufferWidth;
      var height = gl.drawingBufferHeight;
      resizeTargets(width, height);

      camera.getViewMatrix(viewMatrix);
      m4.perspective(60 * Math.PI / 180, width / height, NEAR, FAR,
          projectionMatrix);
      m4.multiply(projectionMatrix, viewMatrix, viewProjection);
      m4.inverse(viewProjection, inverseViewProjection);

      fitLightFrustum(state);

      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.CULL_FACE);

      // ---- Pass 0: shadow map, opaque casters only ----
      glHelpers.bindFramebuffer(gl, shadowTarget);
      gl.colorMask(false, false, false, false);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      gl.useProgram(depthProgram.program);
      drawOpaque(depthProgram, state, {
        u_lightViewProjection: lightViewProjection,
      });
      gl.colorMask(true, true, true, true);

      // ---- Pass 1: opaque, into the HDR target ----
      glHelpers.bindFramebuffer(gl, opaqueTarget);
      gl.clearColor(0.02, 0.03, 0.05, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(opaqueProgram.program);

      var opaqueUniforms = {
        u_viewProjection: viewProjection,
        u_lightIntensity: state.light.intensity,
        u_ambientIntensity: state.ambient.intensity,
      };
      var key;
      var medium = mediumUniforms(state);
      var shadow = shadowUniforms(state);
      for (key in medium) { opaqueUniforms[key] = medium[key]; }
      for (key in shadow) { opaqueUniforms[key] = shadow[key]; }
      drawOpaque(opaqueProgram, state, opaqueUniforms);

      // ---- Pass 2: volume, at half resolution ----
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      glHelpers.bindFramebuffer(gl, volumeTarget);

      if (!state.medium.visible) {
        // Nothing to march. Clearing to zero in-scatter and full
        // transmittance is exactly what an empty medium contributes, so the
        // composite below hands the opaque colour through untouched, and the
        // whole ray march is skipped rather than run against a zero density.
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
      } else {
        gl.useProgram(volumeProgram.program);

        var volumeUniforms = {
          u_inverseViewProjection: inverseViewProjection,
          u_cameraPosition: camera.position,
          u_sceneDepth: opaqueTarget.depth,
          u_lightIntensity: state.light.intensity,
          u_ambientIntensity: state.ambient.intensity,
          u_time: time,
          u_steps: VOLUME_STEPS,
          u_lightSteps: LIGHT_STEPS,
          u_octaves: OCTAVES,
        };
        for (key in medium) { volumeUniforms[key] = medium[key]; }
        for (key in shadow) { volumeUniforms[key] = shadow[key]; }

        webglUtils.setBuffersAndAttributes(gl, volumeProgram.attribSetters,
            quad);
        glHelpers.setUniforms(gl, volumeProgram, volumeUniforms);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }

      // ---- Pass 3: composite to the canvas ----
      glHelpers.bindFramebuffer(gl, null);
      gl.useProgram(compositeProgram.program);
      webglUtils.setBuffersAndAttributes(gl, compositeProgram.attribSetters, quad);
      glHelpers.setUniforms(gl, compositeProgram, {
        u_opaque: opaqueTarget.color,
        u_volume: volumeTarget.color,
        u_sceneDepth: opaqueTarget.depth,
        u_halfResolution: [volumeTarget.width, volumeTarget.height],
        u_clipPlanes: [NEAR, FAR],
      });
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    return {
      render: render,
    };
  }

  return {
    create: create,
  };

}));
