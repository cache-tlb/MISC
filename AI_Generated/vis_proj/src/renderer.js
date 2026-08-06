/*
 * The renderer for the stereographic projection demo.
 *
 * It owns every GPU resource in the scene and draws one frame from a plain
 * description of what should be visible. There is no external library involved:
 * webgl-utils.js supplies the WebGL boilerplate, m4.js the matrices, and
 * geometry.js the vertex data.
 *
 * The three meshes M, S and P share one topology and one vertex buffer. What
 * separates them is a single `deform()` call in the vertex shader, selected at
 * compile time by the STAGE define:
 *
 *   M   the grid where the model matrix leaves it,  (x, y, z)
 *   S   M pushed onto the unit sphere,              (x, y, z) / L
 *   P   S projected from (0, 0, h),                 (xh/(h-z), yh/(h-z), 0)
 *
 * @module renderer
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./m4.js', './webgl-utils.js', './geometry.js'], factory);
  } else {
    // Browser globals
    root.createRenderer = factory(root.m4, root.webglUtils, root.geometry);
  }
}(this, function(m4, webglUtils, geometry) {
  "use strict";

  var STAGE = geometry.STAGE;

  /*
   * Shared by every shader that has to place a vertex on one of the meshes.
   * `stage` matches module:geometry.STAGE. STAGE.RAW never reaches this
   * function, its vertices are already in world space.
   */
  var DEFORM_GLSL = [
    'vec3 deform(vec3 world, int stage, float h) {',
    '  if (stage == 0) {',
    '    return world;                                  // M',
    '  }',
    '  float len = max(length(world), 1e-5);',
    '  vec3 s = world / len;                            // S, on the unit sphere',
    '  if (stage == 1) {',
    '    return s;',
    '  }',
    '  // Generalised stereographic projection: where the ray from the centre',
    '  // (0,0,h) through s meets the plane z = 0. h = 1 is the classic case.',
    '  //',
    '  // h - s.z is negative for vertices beyond the centre, which is a real',
    '  // case once h < 1 puts the centre inside the sphere, so the guard has to',
    '  // keep the sign instead of clamping to a positive floor. Only h == s.z',
    '  // is singular there: the ray runs parallel to the plane.',
    '  float d = h - s.z;',
    '  d = (d < 0.0 ? -1.0 : 1.0) * max(abs(d), 1e-4);',
    '  vec2 q = clamp(s.xy * h / d, vec2(-1e4), vec2(1e4));',
    '  return vec3(q, 0.0);                             // P, on z = 0',
    '}',
  ].join('\n');

  var MESH_VS = [
    'in vec4 a_position;',
    'in vec2 a_texcoord;',
    '',
    'uniform mat4 u_world;',
    'uniform mat4 u_viewProjection;',
    'uniform float u_projectionHeight;',
    '',
    'out vec3 v_worldPosition;',
    'out vec2 v_texcoord;',
    '',
    DEFORM_GLSL,
    '',
    'void main() {',
    '  vec3 p = deform((u_world * a_position).xyz, STAGE, u_projectionHeight);',
    '  v_worldPosition = p;',
    '  v_texcoord = a_texcoord;',
    '  gl_Position = u_viewProjection * vec4(p, 1.0);',
    '}',
  ].join('\n');

  var MESH_FS = [
    'precision highp float;',
    '',
    'in vec3 v_worldPosition;',
    'in vec2 v_texcoord;',
    '',
    'uniform vec3 u_cameraPosition;',
    'uniform vec3 u_lightDirection;',
    'uniform vec3 u_checkerLight;',
    'uniform vec3 u_checkerDark;',
    'uniform vec3 u_tint;',
    'uniform float u_checkerScale;',
    'uniform vec4 u_flatColor;   // alpha > 0 switches to the flat wireframe colour',
    '',
    'out vec4 outColor;',
    '',
    '// Box filtered checkerboard. Where the cells shrink below a pixel, which is',
    '// exactly what happens in the middle of P, this fades to flat grey instead',
    '// of boiling into noise.',
    'float filteredChecker(vec2 p, vec2 w) {',
    '  vec2 i = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5) -',
    '                  abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;',
    '  return 0.5 - 0.5 * i.x * i.y;',
    '}',
    '',
    'void main() {',
    '  if (u_flatColor.a > 0.0) {',
    '    outColor = u_flatColor;',
    '    return;',
    '  }',
    '',
    '  vec2 p = v_texcoord * u_checkerScale;',
    '  vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 1e-4;',
    '  vec3 base = mix(u_checkerDark, u_checkerLight, filteredChecker(p, w)) * u_tint;',
    '',
    '  // The vertex shader moves the vertices, so any normal we shipped from the',
    '  // CPU would be wrong. Screen space derivatives give the exact face normal',
    '  // of the deformed triangle for free.',
    '  vec3 faceNormal = cross(dFdx(v_worldPosition), dFdy(v_worldPosition));',
    '  float faceLength = length(faceNormal);',
    '  vec3 n = faceLength > 1e-12 ? faceNormal / faceLength : vec3(0.0, 0.0, 1.0);',
    '  vec3 toEye = normalize(u_cameraPosition - v_worldPosition);',
    '  if (dot(n, toEye) < 0.0) {',
    '    n = -n;                                   // the sheets are lit on both sides',
    '  }',
    '',
    '  float diffuse = max(dot(n, u_lightDirection), 0.0);',
    '  float ambient = 0.34 + 0.16 * (n.y * 0.5 + 0.5);',
    '  vec3 color = base * (ambient + 0.62 * diffuse);',
    '  color += vec3(0.18) * pow(max(dot(normalize(u_lightDirection + toEye), n), 0.0), 40.0);',
    '  if (!gl_FrontFacing) {',
    '    color *= vec3(0.74, 0.80, 0.92);          // cool tint on the back side',
    '  }',
    '  outColor = vec4(color, 1.0);',
    '}',
  ].join('\n');

  var RAY_VS = [
    'in vec4 a_position;',
    'in float a_stage;',
    '',
    'uniform mat4 u_world;',
    'uniform mat4 u_viewProjection;',
    'uniform float u_projectionHeight;',
    '',
    DEFORM_GLSL,
    '',
    'void main() {',
    '  int stage = int(a_stage + 0.5);',
    '  vec3 p;',
    '  if (stage == 3) {',
    '    p = a_position.xyz;             // STAGE.RAW, a world space anchor',
    '  } else if (stage == 4) {',
    '    p = vec3(0.0, 0.0, u_projectionHeight);   // STAGE.CENTER',
    '  } else {',
    '    p = deform((u_world * a_position).xyz, stage, u_projectionHeight);',
    '  }',
    '  gl_Position = u_viewProjection * vec4(p, 1.0);',
    '}',
  ].join('\n');

  var RAY_FS = [
    'precision highp float;',
    'uniform vec4 u_color;',
    'out vec4 outColor;',
    'void main() {',
    '  outColor = u_color;',
    '}',
  ].join('\n');

  var HELPER_VS = [
    'in vec4 a_position;',
    'in vec4 a_color;',
    '',
    'uniform mat4 u_viewProjection;',
    'uniform vec3 u_center;   // lets one buffer be parked anywhere in the scene',
    '',
    'out vec4 v_color;',
    '',
    'void main() {',
    '  v_color = a_color;',
    '  gl_Position = u_viewProjection * vec4(a_position.xyz + u_center, 1.0);',
    '}',
  ].join('\n');

  var HELPER_FS = [
    'precision highp float;',
    'in vec4 v_color;',
    'uniform float u_opacity;',
    'out vec4 outColor;',
    'void main() {',
    '  outColor = vec4(v_color.rgb, v_color.a * u_opacity);',
    '}',
  ].join('\n');

  var BACKGROUND_VS = [
    'out vec2 v_uv;',
    'void main() {',
    '  // One oversized triangle, no vertex buffer needed.',
    '  v_uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);',
    '  gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);',
    '}',
  ].join('\n');

  var BACKGROUND_FS = [
    'precision highp float;',
    'in vec2 v_uv;',
    'uniform vec3 u_top;',
    'uniform vec3 u_bottom;',
    'out vec4 outColor;',
    'void main() {',
    '  vec3 color = mix(u_bottom, u_top, smoothstep(0.0, 1.0, v_uv.y));',
    '  // A soft glow behind the middle of the scene, then a dither so the very',
    '  // shallow gradient does not band on 8 bit displays.',
    '  color += 0.05 * (1.0 - smoothstep(0.0, 0.75, distance(v_uv, vec2(0.5, 0.55))));',
    '  float noise = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);',
    '  outColor = vec4(color + (noise - 0.5) / 255.0, 1.0);',
    '}',
  ].join('\n');

  /*
   * Per mesh look. The checker stays grey and white as specified, the tints are
   * only strong enough to tell the three meshes apart at a glance.
   */
  var MESHES = [
    {key: 'm', stage: STAGE.M, tint: [1.00, 1.00, 1.00], wire: [0.96, 0.96, 1.00, 0.55]},
    {key: 's', stage: STAGE.S, tint: [1.00, 0.84, 0.58], wire: [1.00, 0.74, 0.32, 0.65]},
    {key: 'p', stage: STAGE.P, tint: [0.58, 0.85, 1.00], wire: [0.40, 0.86, 1.00, 0.70]},
  ];

  var RAY_COLORS = {
    radial: [1.00, 0.72, 0.30, 0.32],
    polar: [0.40, 0.86, 1.00, 0.32],
  };

  /**
   * @typedef {Object} RendererOptions
   * @property {number} [halfSize] half the side length of M, default 2
   * @property {number} [segments] quads along one side of M, default 16
   * @property {number} [rayStride] sample every n-th vertex for the rays, default 4
   * @property {number} [checkerScale] checker cells across M, default 16
   * @memberOf module:renderer
   */

  /**
   * Creates the renderer and uploads every buffer the scene needs.
   *
   * @param {WebGL2RenderingContext} gl
   * @param {module:renderer.RendererOptions} [options]
   * @return {Object} the renderer
   * @memberOf module:renderer
   */
  function createRenderer(gl, options) {
    options = options || {};
    var halfSize = options.halfSize === undefined ? 2 : options.halfSize;
    var segments = options.segments === undefined ? 16 : options.segments;
    var rayStride = options.rayStride === undefined ? 4 : options.rayStride;
    var checkerScale = options.checkerScale === undefined ? 16 : options.checkerScale;

    var plane = geometry.createSubdividedPlane(halfSize, segments);
    var rays = geometry.createProjectionRays(halfSize, segments, rayStride);

    // One program per stage. Compiling the same source three times with a
    // different STAGE is what makes "the vertex shader of S and P" a real thing
    // instead of a runtime branch.
    var meshPrograms = MESHES.map(function(mesh) {
      return createProgram(gl, MESH_VS, MESH_FS, {STAGE: mesh.stage});
    });
    var rayProgram = createProgram(gl, RAY_VS, RAY_FS);
    var helperProgram = createProgram(gl, HELPER_VS, HELPER_FS);
    var backgroundProgram = createProgram(gl, BACKGROUND_VS, BACKGROUND_FS);

    var surfaceBuffer = webglUtils.createBufferInfoFromArrays(gl, plane.surface);
    var wireframeBuffer = webglUtils.createBufferInfoFromArrays(gl, plane.wireframe);
    var rayBuffers = {
      radial: webglUtils.createBufferInfoFromArrays(gl, rays.radial),
      polar: webglUtils.createBufferInfoFromArrays(gl, rays.polar),
    };
    var helperBuffers = [
      webglUtils.createBufferInfoFromArrays(gl, geometry.createSphereWireframe(
          1, 12, 6, 48, [0.42, 0.55, 0.72, 0.42], [0.55, 0.72, 0.92, 0.75])),
      // The projection plane, which is the equatorial plane through the sphere.
      webglUtils.createBufferInfoFromArrays(gl, geometry.createPlaneGrid(
          0, 1.5, 12, [0.34, 0.48, 0.62, 0.30], [0.45, 0.66, 0.85, 0.60])),
      webglUtils.createBufferInfoFromArrays(gl, geometry.createAxes(2.6)),
    ];
    // Drawn separately because it rides on the projection centre.
    var centerMarker = webglUtils.createBufferInfoFromArrays(
        gl, geometry.createCrossMarker(0.09, [1.0, 0.88, 0.35, 1.0]));

    // The background draws without attributes, so it gets an empty vertex array
    // of its own rather than inheriting whatever the last mesh left enabled.
    var backgroundVao = gl.createVertexArray();

    var viewProjection = m4.identity();

    /**
     * Draws one frame.
     *
     * @param {Object} scene projection, view, cameraPosition, world, show
     * @memberOf module:renderer
     */
    function render(scene) {
      m4.multiply(scene.projection, scene.view, viewProjection);

      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      // The depth mask has to be open here, a masked depth buffer does not clear.
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.depthMask(false);

      gl.useProgram(backgroundProgram.program);
      gl.bindVertexArray(backgroundVao);
      webglUtils.setUniforms(backgroundProgram, {
        u_top: [0.075, 0.090, 0.125],
        u_bottom: [0.020, 0.026, 0.038],
      });
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);

      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(true);
      // Both sides of every sheet are visible, so nothing is culled.
      gl.disable(gl.CULL_FACE);

      drawMeshes(scene);
      drawWireframes(scene);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);

      drawHelpers(scene);
      drawRays(scene);

      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }

    function drawMeshes(scene) {
      // Push the filled triangles back a hair so the wireframe on top of them
      // does not fight for the same depth values.
      gl.enable(gl.POLYGON_OFFSET_FILL);
      gl.polygonOffset(1, 1);

      MESHES.forEach(function(mesh, i) {
        if (!scene.show[mesh.key]) {
          return;
        }
        var programInfo = meshPrograms[i];
        gl.useProgram(programInfo.program);
        webglUtils.setBuffersAndAttributes(gl, programInfo, surfaceBuffer);
        webglUtils.setUniforms(programInfo, {
          u_world: scene.world,
          u_viewProjection: viewProjection,
          u_projectionHeight: scene.projectionHeight,
          u_cameraPosition: scene.cameraPosition,
          u_lightDirection: scene.lightDirection,
          u_checkerLight: [0.92, 0.92, 0.93],
          u_checkerDark: [0.38, 0.39, 0.42],
          u_tint: mesh.tint,
          u_checkerScale: checkerScale,
          u_flatColor: [0, 0, 0, 0],
        });
        webglUtils.drawBufferInfo(gl, surfaceBuffer, gl.TRIANGLES);
      });

      gl.disable(gl.POLYGON_OFFSET_FILL);
    }

    function drawWireframes(scene) {
      if (!scene.show.wireframe) {
        return;
      }
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);

      MESHES.forEach(function(mesh, i) {
        if (!scene.show[mesh.key]) {
          return;
        }
        var programInfo = meshPrograms[i];
        gl.useProgram(programInfo.program);
        webglUtils.setBuffersAndAttributes(gl, programInfo, wireframeBuffer);
        webglUtils.setUniforms(programInfo, {
          u_world: scene.world,
          u_viewProjection: viewProjection,
          u_projectionHeight: scene.projectionHeight,
          u_flatColor: mesh.wire,
        });
        webglUtils.drawBufferInfo(gl, wireframeBuffer, gl.LINES);
      });

      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }

    function drawHelpers(scene) {
      if (!scene.show.helpers) {
        return;
      }
      gl.useProgram(helperProgram.program);
      webglUtils.setUniforms(helperProgram, {
        u_viewProjection: viewProjection,
        u_opacity: 1,
        u_center: [0, 0, 0],
      });
      helperBuffers.forEach(function(bufferInfo) {
        webglUtils.setBuffersAndAttributes(gl, helperProgram, bufferInfo);
        webglUtils.drawBufferInfo(gl, bufferInfo, gl.LINES);
      });

      webglUtils.setBuffersAndAttributes(gl, helperProgram, centerMarker);
      webglUtils.setUniforms(helperProgram, {u_center: [0, 0, scene.projectionHeight]});
      webglUtils.drawBufferInfo(gl, centerMarker, gl.LINES);
    }

    function drawRays(scene) {
      if (!scene.show.rays) {
        return;
      }
      gl.useProgram(rayProgram.program);
      Object.keys(rayBuffers).forEach(function(key) {
        var bufferInfo = rayBuffers[key];
        webglUtils.setBuffersAndAttributes(gl, rayProgram, bufferInfo);
        webglUtils.setUniforms(rayProgram, {
          u_world: scene.world,
          u_viewProjection: viewProjection,
          u_projectionHeight: scene.projectionHeight,
          u_color: RAY_COLORS[key],
        });
        webglUtils.drawBufferInfo(gl, bufferInfo, gl.LINES);
      });
    }

    return {
      render: render,
      numVertices: plane.numVertices,
      numTriangles: plane.numTriangles,
    };
  }

  /**
   * Compiles a program, prepending the GLSL ES 3.00 version line and any
   * defines. `#version` has to stay on the very first line, which is why the
   * shader sources above do not carry it themselves.
   *
   * @param {WebGL2RenderingContext} gl
   * @param {string} vs vertex shader body
   * @param {string} fs fragment shader body
   * @param {Object.<string, number>} [defines]
   * @return {module:webgl-utils.ProgramInfo}
   */
  function createProgram(gl, vs, fs, defines) {
    var header = '#version 300 es\n';
    if (defines) {
      Object.keys(defines).forEach(function(name) {
        header += '#define ' + name + ' ' + defines[name] + '\n';
      });
    }
    return webglUtils.createProgramInfo(gl, [header + vs, header + fs]);
  }

  return createRenderer;

}));
