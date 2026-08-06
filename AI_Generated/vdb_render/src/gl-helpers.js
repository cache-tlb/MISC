/*
 * The WebGL2 objects webgl-utils.js does not cover: textures, 3D textures,
 * framebuffers and a fullscreen triangle.
 *
 * webgl-utils.js handles programs, buffers and uniforms; it has no texture or
 * framebuffer helpers at all, and it must not be modified, so those live here.
 *
 * @module glHelpers
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./webgl-utils.js'], factory);
  } else {
    // Browser globals
    root.glHelpers = factory(root.webglUtils);
  }
}(this, function(webglUtils) {
  "use strict";

  /**
   * Builds a program and introspects its uniforms.
   *
   * webgl-utils' createProgramInfo cannot be used here: its uniform setter
   * factory only knows SAMPLER_2D and SAMPLER_CUBE and throws
   * "unknown type: 0x8b5f" on SAMPLER_3D, which every pass in this demo needs.
   * That file must not be modified, so linking is delegated to it and uniform
   * handling is done here.
   *
   * @param {WebGL2RenderingContext} gl the context
   * @param {string} vertexSource the vertex shader
   * @param {string} fragmentSource the fragment shader
   * @return {Object} program, uniforms and attribSetters
   * @memberOf module:glHelpers
   */
  function createProgram(gl, vertexSource, fragmentSource) {
    var program = webglUtils.createProgramFromSources(gl,
        [vertexSource, fragmentSource]);

    var uniforms = {};
    var textureUnit = 0;
    var count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);

    for (var i = 0; i < count; i++) {
      var info = gl.getActiveUniform(program, i);
      if (!info) {
        continue;
      }
      // Array uniforms are reported as "name[0]".
      var name = info.name.replace(/\[0\]$/, '');
      var entry = {
        location: gl.getUniformLocation(program, info.name),
        type: info.type,
        size: info.size,
        unit: -1,
      };
      if (isSampler(gl, info.type)) {
        entry.unit = textureUnit;
        textureUnit += 1;
      }
      uniforms[name] = entry;
    }

    return {
      program: program,
      uniforms: uniforms,
      attribSetters: webglUtils.createAttributeSetters(gl, program),
    };
  }

  function isSampler(gl, type) {
    return type === gl.SAMPLER_2D || type === gl.SAMPLER_3D ||
        type === gl.SAMPLER_CUBE || type === gl.SAMPLER_2D_SHADOW ||
        type === gl.SAMPLER_2D_ARRAY;
  }

  function samplerTarget(gl, type) {
    if (type === gl.SAMPLER_3D) {
      return gl.TEXTURE_3D;
    }
    if (type === gl.SAMPLER_CUBE) {
      return gl.TEXTURE_CUBE_MAP;
    }
    if (type === gl.SAMPLER_2D_ARRAY) {
      return gl.TEXTURE_2D_ARRAY;
    }
    return gl.TEXTURE_2D;
  }

  /**
   * Sets uniforms on a program built by createProgram. Names the program does
   * not declare are ignored, so callers can pass one shared block of uniforms
   * to several programs.
   * @param {WebGL2RenderingContext} gl the context
   * @param {Object} programInfo from createProgram
   * @param {Object} values name to value
   * @memberOf module:glHelpers
   */
  function setUniforms(gl, programInfo, values) {
    var uniforms = programInfo.uniforms;
    for (var name in values) {
      var uniform = uniforms[name];
      if (!uniform) {
        continue;
      }
      var value = values[name];
      var type = uniform.type;
      var location = uniform.location;

      if (uniform.unit >= 0) {
        gl.activeTexture(gl.TEXTURE0 + uniform.unit);
        gl.bindTexture(samplerTarget(gl, type), value);
        gl.uniform1i(location, uniform.unit);
      } else if (type === gl.FLOAT_MAT4) {
        gl.uniformMatrix4fv(location, false, value);
      } else if (type === gl.FLOAT_MAT3) {
        gl.uniformMatrix3fv(location, false, value);
      } else if (type === gl.FLOAT_VEC2) {
        gl.uniform2fv(location, value);
      } else if (type === gl.FLOAT_VEC3) {
        gl.uniform3fv(location, value);
      } else if (type === gl.FLOAT_VEC4) {
        gl.uniform4fv(location, value);
      } else if (type === gl.INT || type === gl.BOOL) {
        gl.uniform1i(location, value);
      } else if (type === gl.INT_VEC2 || type === gl.BOOL_VEC2) {
        gl.uniform2iv(location, value);
      } else if (type === gl.INT_VEC3 || type === gl.BOOL_VEC3) {
        gl.uniform3iv(location, value);
      } else if (type === gl.INT_VEC4 || type === gl.BOOL_VEC4) {
        gl.uniform4iv(location, value);
      } else if (type === gl.FLOAT) {
        gl.uniform1f(location, value);
      } else {
        throw new Error('glHelpers: no setter for uniform ' + name +
            ' of type 0x' + type.toString(16));
      }
    }
  }

  /**
   * Creates a 2D texture.
   * @param {WebGL2RenderingContext} gl the context
   * @param {Object} options width, height, internalFormat, format, type, data,
   *     min, mag, wrap
   * @return {WebGLTexture} the texture
   * @memberOf module:glHelpers
   */
  function createTexture2D(gl, options) {
    var internalFormat = options.internalFormat || gl.RGBA8;
    var format = options.format || gl.RGBA;
    var type = options.type || gl.UNSIGNED_BYTE;
    var min = options.min || gl.LINEAR;
    var mag = options.mag || gl.LINEAR;
    var wrap = options.wrap || gl.CLAMP_TO_EDGE;

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, options.width,
        options.height, 0, format, type, options.data || null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, min);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, mag);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  /**
   * Creates a texture from an already loaded image, with mipmaps.
   * @param {WebGL2RenderingContext} gl the context
   * @param {HTMLImageElement} image the decoded image
   * @return {WebGLTexture} the texture
   * @memberOf module:glHelpers
   */
  function createTextureFromImage(gl, image) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  /**
   * Creates a 3D texture. Note that UNPACK_ALIGNMENT is forced to 1: a density
   * grid's row length is rarely a multiple of four, and the default alignment
   * of 4 would shear the volume.
   * @param {WebGL2RenderingContext} gl the context
   * @param {Object} options width, height, depth, internalFormat, format,
   *     type, data
   * @return {WebGLTexture} the texture
   * @memberOf module:glHelpers
   */
  function createTexture3D(gl, options) {
    var internalFormat = options.internalFormat || gl.R8;
    var format = options.format || gl.RED;
    var type = options.type || gl.UNSIGNED_BYTE;

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, texture);
    var previousAlignment = gl.getParameter(gl.UNPACK_ALIGNMENT);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, internalFormat, options.width,
        options.height, options.depth, 0, format, type, options.data || null);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, previousAlignment);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_3D, null);
    return texture;
  }

  /**
   * Creates a depth texture, sampled directly rather than through hardware
   * comparison so the same PCF code can serve both the opaque and volume
   * passes.
   * @param {WebGL2RenderingContext} gl the context
   * @param {number} width in pixels
   * @param {number} height in pixels
   * @return {WebGLTexture} the texture
   * @memberOf module:glHelpers
   */
  function createDepthTexture(gl, width, height) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, width, height, 0,
        gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.NONE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  /**
   * Creates a framebuffer from existing attachments.
   * @param {WebGL2RenderingContext} gl the context
   * @param {Object} options color (texture or null), depth (texture or null),
   *     width, height
   * @return {Object} framebuffer, width, height and the attachments
   * @memberOf module:glHelpers
   */
  function createFramebuffer(gl, options) {
    var framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    if (options.color) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D, options.color, 0);
    } else {
      // A depth only framebuffer still has to say it draws nothing.
      gl.drawBuffers([gl.NONE]);
      gl.readBuffer(gl.NONE);
    }
    if (options.depth) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
          gl.TEXTURE_2D, options.depth, 0);
    }

    var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('framebuffer incomplete: ' +
          webglUtils.glEnumToString(gl, status));
    }

    return {
      framebuffer: framebuffer,
      color: options.color || null,
      depth: options.depth || null,
      width: options.width,
      height: options.height,
    };
  }

  /**
   * Binds a framebuffer created above, or the canvas when passed null, and
   * sets the viewport to match.
   * @param {WebGL2RenderingContext} gl the context
   * @param {Object} target the framebuffer, or null for the canvas
   * @memberOf module:glHelpers
   */
  function bindFramebuffer(gl, target) {
    if (target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, target.width, target.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    }
  }

  /**
   * A single oversized triangle covering clip space. One triangle rather than
   * two avoids the seam along a quad's diagonal.
   * @param {WebGL2RenderingContext} gl the context
   * @return {Object} a bufferInfo for webglUtils
   * @memberOf module:glHelpers
   */
  function createFullscreenQuad(gl) {
    return webglUtils.createBufferInfoFromArrays(gl, {
      position: {
        numComponents: 2,
        data: [-1, -1, 3, -1, -1, 3],
      },
    });
  }

  /**
   * Deletes a framebuffer and the textures it owns.
   * @param {WebGL2RenderingContext} gl the context
   * @param {Object} target the framebuffer to delete, may be null
   * @memberOf module:glHelpers
   */
  function deleteFramebuffer(gl, target) {
    if (!target) {
      return;
    }
    if (target.color) {
      gl.deleteTexture(target.color);
    }
    if (target.depth) {
      gl.deleteTexture(target.depth);
    }
    gl.deleteFramebuffer(target.framebuffer);
  }

  return {
    createProgram: createProgram,
    setUniforms: setUniforms,
    createTexture2D: createTexture2D,
    createTextureFromImage: createTextureFromImage,
    createTexture3D: createTexture3D,
    createDepthTexture: createDepthTexture,
    createFramebuffer: createFramebuffer,
    bindFramebuffer: bindFramebuffer,
    createFullscreenQuad: createFullscreenQuad,
    deleteFramebuffer: deleteFramebuffer,
  };

}));
