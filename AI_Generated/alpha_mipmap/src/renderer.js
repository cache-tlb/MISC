/*
 * WebGL2 renderer for the alpha mask mipmap demo.
 *
 * No external dependencies beyond m4 (matrix math) and webglUtils (buffer /
 * program / uniform plumbing), both of which live next to this file.
 *
 * The scene is a 100x100 checkerboard room with a single alpha tested tree in
 * the middle. Everything is drawn with one program; the only difference
 * between the three mipmap strategies being compared is which texture is bound
 * to u_mask, so the comparison is not muddied by anything else changing.
 *
 * Right handed coordinate system, as OpenGL conventionally uses: +x right,
 * +y up, +z out of the screen, so the camera looks along -z.
 *
 * @module renderer
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./m4.js', './webgl-utils.js'], factory);
  } else {
    // Browser globals
    root.createRenderer = factory(root.m4, root.webglUtils);
  }
}(this, function(m4, webglUtils) {
  "use strict";

  var ROOM_SIZE = 500;
  var WALL_HEIGHT = 20;
  // World units covered by one tile of the checker texture.
  var CHECKER_TILE = 8;

  var vertexShaderSource = `#version 300 es
in vec4 a_position;
in vec3 a_normal;
in vec2 a_texcoord;

uniform mat4 u_worldViewProjection;
uniform mat4 u_world;

out vec3 v_normal;
out vec2 v_texcoord;
out vec3 v_worldPosition;

void main() {
  gl_Position = u_worldViewProjection * a_position;
  v_normal = mat3(u_world) * a_normal;
  v_texcoord = a_texcoord;
  v_worldPosition = (u_world * a_position).xyz;
}
`;

  var fragmentShaderSource = `#version 300 es
precision highp float;

in vec3 v_normal;
in vec2 v_texcoord;
in vec3 v_worldPosition;

uniform sampler2D u_albedo;
uniform sampler2D u_mask;
// 0 for opaque surfaces, 1 for alpha tested ones.
uniform float u_useMask;
// 1 for surfaces built from single sided cards that are drawn from both sides.
uniform float u_twoSided;
uniform float u_alphaRef;
uniform vec3 u_lightDirection;
uniform vec3 u_ambient;
uniform vec3 u_cameraPosition;

out vec4 outColor;

void main() {
  vec4 albedo = texture(u_albedo, v_texcoord);
  float mask = texture(u_mask, v_texcoord).r;

  // The whole point of the demo: what survives this test depends entirely on
  // how the mip chain of u_mask was built.
  if (u_useMask > 0.5 && mask < u_alphaRef) {
    discard;
  }

  vec3 normal = normalize(v_normal);

  // Foliage is modelled as single sided cards that are drawn from both sides,
  // so a leaf seen from behind needs its normal flipped or it goes black.
  //
  // Orienting against the view vector rather than testing gl_FrontFacing.
  // gl_FrontFacing reports which side of the *card* you are on, which is only
  // the same question when the vertex normals are perpendicular to it. These
  // leaf normals are deliberately bent away from their card to soften the
  // canopy, so the two disagree on ~15% of the foliage. Comparing against the
  // view vector asks about the normal directly and is right every time.
  //
  // Only for cards. The trunk is a closed opaque surface whose normals are
  // already correct, and flipping near its silhouette would seam the shading.
  if (u_twoSided > 0.5 && dot(normal, u_cameraPosition - v_worldPosition) < 0.0) {
    normal = -normal;
  }

  // Ambient sets the floor and the directional term fills the rest, so a fully
  // lit surface lands at exactly 1.0 instead of clipping to white.
  float diffuse = max(dot(normal, u_lightDirection), 0.0);
  vec3 light = u_ambient + (vec3(1.0) - u_ambient) * diffuse;
  outColor = vec4(albedo.rgb * light, 1.0);
}
`;

  /**
   * Builds a grey/white checkerboard as RGBA8 texels, two cells across.
   * @param {number} size width and height in texels
   * @return {Uint8Array} the pixels
   * @memberOf module:renderer
   */
  function makeCheckerPixels(size) {
    var pixels = new Uint8Array(size * size * 4);
    var half = size >> 1;
    for (var y = 0; y < size; ++y) {
      for (var x = 0; x < size; ++x) {
        var light = ((x < half) !== (y < half));
        var v = light ? 240 : 170;
        var offset = (y * size + x) * 4;
        pixels[offset + 0] = v;
        pixels[offset + 1] = v;
        pixels[offset + 2] = v;
        pixels[offset + 3] = 255;
      }
    }
    return pixels;
  }

  /**
   * Appends a quad to the arrays being built for the room.
   * @param {Object} arrays position/normal/texcoord/indices being filled
   * @param {number[][]} corners four world space corners, counter clockwise
   * @param {number[]} normal the quad's normal
   * @param {number[][]} uvs four texture coordinates matching the corners
   * @memberOf module:renderer
   */
  function addQuad(arrays, corners, normal, uvs) {
    var base = arrays.position.length / 3;
    for (var i = 0; i < 4; ++i) {
      arrays.position.push(corners[i][0], corners[i][1], corners[i][2]);
      arrays.normal.push(normal[0], normal[1], normal[2]);
      arrays.texcoord.push(uvs[i][0], uvs[i][1]);
    }
    arrays.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  /**
   * Builds the floor and the four walls as one mesh.
   * @return {Object} arrays for webglUtils.createBufferInfoFromArrays
   * @memberOf module:renderer
   */
  function makeRoomArrays() {
    var arrays = {position: [], normal: [], texcoord: [], indices: []};
    var h = ROOM_SIZE * 0.5;
    var t = ROOM_SIZE / CHECKER_TILE;
    var tv = WALL_HEIGHT / CHECKER_TILE;

    // Floor, normal up.
    addQuad(arrays,
        [[-h, 0, -h], [-h, 0, h], [h, 0, h], [h, 0, -h]],
        [0, 1, 0],
        [[0, 0], [0, t], [t, t], [t, 0]]);

    // Four walls, each facing into the room. Face culling is off so winding
    // does not matter here, only the normals do.
    addQuad(arrays,
        [[-h, 0, h], [-h, WALL_HEIGHT, h], [h, WALL_HEIGHT, h], [h, 0, h]],
        [0, 0, -1],
        [[0, 0], [0, tv], [t, tv], [t, 0]]);
    addQuad(arrays,
        [[h, 0, -h], [h, WALL_HEIGHT, -h], [-h, WALL_HEIGHT, -h], [-h, 0, -h]],
        [0, 0, 1],
        [[0, 0], [0, tv], [t, tv], [t, 0]]);
    addQuad(arrays,
        [[h, 0, h], [h, WALL_HEIGHT, h], [h, WALL_HEIGHT, -h], [h, 0, -h]],
        [-1, 0, 0],
        [[0, 0], [0, tv], [t, tv], [t, 0]]);
    addQuad(arrays,
        [[-h, 0, -h], [-h, WALL_HEIGHT, -h], [-h, WALL_HEIGHT, h], [-h, 0, h]],
        [1, 0, 0],
        [[0, 0], [0, tv], [t, tv], [t, 0]]);

    return arrays;
  }

  /**
   * Creates the renderer and everything that does not depend on loaded assets.
   * @param {HTMLCanvasElement} canvas the canvas to render into
   * @return {Object} the renderer
   * @memberOf module:renderer
   */
  function createRenderer(canvas) {
    var gl = canvas.getContext('webgl2', {antialias: true, alpha: false});
    if (!gl) {
      throw new Error('WebGL2 is not available in this browser.');
    }

    var programInfo = webglUtils.createProgramInfo(
        gl, [vertexShaderSource, fragmentShaderSource]);
    if (!programInfo) {
      throw new Error('failed to compile the scene shaders, see the console.');
    }

    // R8 mip levels get down to 1 and 2 texels wide, which are not a multiple
    // of the default 4 byte row alignment.
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.enable(gl.DEPTH_TEST);
    // Leaves are single sided cards that have to be visible from both sides.
    gl.disable(gl.CULL_FACE);

    var roomBufferInfo = webglUtils.createBufferInfoFromArrays(gl, makeRoomArrays());

    var checkerTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, checkerTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 256, 0, gl.RGBA,
        gl.UNSIGNED_BYTE, makeCheckerPixels(256));
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

    // A 1x1 opaque mask so u_mask always has something bound, even while the
    // real masks are still loading.
    var whiteTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, whiteTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE,
        new Uint8Array([255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    var renderer = {
      gl: gl,
      roomSize: ROOM_SIZE,
      wallHeight: WALL_HEIGHT,
      // Filled in by setModel.
      meshes: [],
      modelMatrix: m4.identity(),
      maskTexture: whiteTexture,
      fieldOfView: 60 * Math.PI / 180,
      // The room's diagonal is ~707 units, so the far plane has to clear that
      // with room to spare. Near is pulled back off 0.1 to keep enough depth
      // precision at the far end of a range this wide.
      near: 0.5,
      far: 1500,
      // Direction toward the light. Biased to +z, the side the camera starts
      // on, so the face of the tree you are looking at is the lit one.
      lightDirection: m4.normalize([0.4, 0.85, 0.35]),
      ambient: [0.45, 0.45, 0.5],
    };

    /**
     * Creates a mipmapped colour texture from a decoded image.
     *
     * Uploaded as RGB8 on purpose: the alpha channel of the leaf atlas is the
     * mask, and it lives in its own single channel texture so that switching
     * mipmap strategy switches nothing but the mask.
     *
     * @param {HTMLImageElement} image the source image
     * @param {number} wrap gl.REPEAT or gl.CLAMP_TO_EDGE
     * @return {WebGLTexture} the texture
     */
    renderer.createAlbedoTexture = function(image, wrap) {
      var texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      // Row 0 of the image is v = 0; the loader already flipped the OBJ's v.
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, image);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap || gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap || gl.REPEAT);
      return texture;
    };

    /**
     * Creates a single channel R8 mask texture from explicit mip levels.
     * Passing a single level produces a texture with no mipmaps at all.
     * @param {Array<{width: number, height: number, data: Uint8Array}>} levels
     *     the mip chain, level 0 first
     * @return {WebGLTexture} the texture
     */
    renderer.createMaskTexture = function(levels) {
      var texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      for (var i = 0; i < levels.length; ++i) {
        gl.texImage2D(gl.TEXTURE_2D, i, gl.R8, levels[i].width, levels[i].height,
            0, gl.RED, gl.UNSIGNED_BYTE, levels[i].data);
      }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, levels.length - 1);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
          levels.length > 1 ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      // The leaf atlas uses the full 0..1 range, so clamping avoids bleeding
      // the opposite edge in at the coarse levels.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return texture;
    };

    /**
     * Replaces the contents of an existing mask texture in place. Used when the
     * alpha test threshold changes, which changes the coverage corrections.
     * @param {WebGLTexture} texture a texture from createMaskTexture
     * @param {Array<{width: number, height: number, data: Uint8Array}>} levels
     *     the new mip chain, same dimensions as the original
     */
    renderer.updateMaskTexture = function(texture, levels) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      for (var i = 0; i < levels.length; ++i) {
        gl.texSubImage2D(gl.TEXTURE_2D, i, 0, 0, levels[i].width,
            levels[i].height, gl.RED, gl.UNSIGNED_BYTE, levels[i].data);
      }
    };

    /**
     * Uploads the parsed model and binds each chunk to its material's textures.
     * @param {Object[]} geometries from objLoader.parseObj
     * @param {Object.<string, Object>} materialTextures per material name,
     *     {albedo: WebGLTexture, useMask: boolean}
     * @param {Matrix4} modelMatrix where to place the model
     */
    renderer.setModel = function(geometries, materialTextures, modelMatrix) {
      renderer.meshes = geometries.map(function(geometry) {
        return {
          bufferInfo: webglUtils.createBufferInfoFromArrays(gl, geometry.arrays),
          material: materialTextures[geometry.material] || {},
        };
      });
      renderer.modelMatrix = modelMatrix || m4.identity();
    };

    // Scratch matrices, reused every frame to keep the render loop off the
    // allocator.
    var projection = m4.identity();
    var view = m4.identity();
    var viewProjection = m4.identity();
    var worldViewProjection = m4.identity();
    var identityMatrix = m4.identity();

    /**
     * Draws the whole scene into the current viewport.
     * @param {Object} camera a camera from createCamera
     * @param {number} aspect width / height of the viewport
     * @param {WebGLTexture} maskTexture the mask to alpha test the tree against
     * @param {number} alphaRef the alpha test threshold in [0, 1]
     */
    function drawScene(camera, aspect, maskTexture, alphaRef) {
      m4.perspective(renderer.fieldOfView, aspect, renderer.near, renderer.far, projection);
      camera.getViewMatrix(view);
      m4.multiply(projection, view, viewProjection);

      gl.useProgram(programInfo.program);

      // The room: opaque, no alpha test.
      m4.copy(viewProjection, worldViewProjection);
      webglUtils.setBuffersAndAttributes(gl, programInfo, roomBufferInfo);
      webglUtils.setUniforms(programInfo, {
        u_worldViewProjection: worldViewProjection,
        u_world: identityMatrix,
        u_albedo: checkerTexture,
        u_mask: whiteTexture,
        u_useMask: 0,
        u_twoSided: 0,
        u_alphaRef: alphaRef,
        u_lightDirection: renderer.lightDirection,
        u_ambient: renderer.ambient,
        u_cameraPosition: camera.position,
      });
      webglUtils.drawBufferInfo(gl, roomBufferInfo);

      // The model.
      m4.multiply(viewProjection, renderer.modelMatrix, worldViewProjection);
      renderer.meshes.forEach(function(mesh) {
        webglUtils.setBuffersAndAttributes(gl, programInfo, mesh.bufferInfo);
        webglUtils.setUniforms(programInfo, {
          u_worldViewProjection: worldViewProjection,
          u_world: renderer.modelMatrix,
          u_albedo: mesh.material.albedo || checkerTexture,
          u_mask: maskTexture,
          u_useMask: mesh.material.useMask ? 1 : 0,
          u_twoSided: mesh.material.twoSided ? 1 : 0,
          u_alphaRef: alphaRef,
          u_lightDirection: renderer.lightDirection,
          u_ambient: renderer.ambient,
          u_cameraPosition: camera.position,
        });
        webglUtils.drawBufferInfo(gl, mesh.bufferInfo);
      });
    }

    /**
     * Renders one frame.
     * @param {Object} state camera, alphaRef, and either a single maskTexture
     *     or a `maskTextures` array to show side by side
     */
    renderer.render = function(state) {
      webglUtils.resizeCanvasToDisplaySize(gl.canvas, window.devicePixelRatio);

      var width = gl.canvas.width;
      var height = gl.canvas.height;
      gl.viewport(0, 0, width, height);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(0.08, 0.09, 0.11, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      var masks = state.maskTextures || [state.maskTexture];
      if (masks.length === 1) {
        drawScene(state.camera, width / height, masks[0], state.alphaRef);
        return;
      }

      // Side by side: the same camera, the same everything, one column per
      // strategy so the difference is the only thing you can see.
      gl.enable(gl.SCISSOR_TEST);
      var columnWidth = Math.floor(width / masks.length);
      for (var i = 0; i < masks.length; ++i) {
        var x = i * columnWidth;
        // Give the last column any pixels lost to rounding.
        var w = (i === masks.length - 1) ? width - x : columnWidth;
        gl.viewport(x, 0, w, height);
        gl.scissor(x, 0, w, height);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        drawScene(state.camera, w / height, masks[i], state.alphaRef);
      }
      gl.disable(gl.SCISSOR_TEST);
      gl.viewport(0, 0, width, height);
    };

    return renderer;
  }

  return createRenderer;

}));
