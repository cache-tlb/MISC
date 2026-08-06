/*
 * Minimal Wavefront OBJ / MTL loader and image loader.
 *
 * Deliberately dependency free. Only the subset of the format used by the
 * assets in this demo is supported: v / vt / vn / f / g / o / usemtl / mtllib
 * for OBJ and newmtl / Ka / Kd / Ks / Ns / d / map_* for MTL.
 *
 * Faces with more than three vertices are triangulated as a fan, and vertices
 * are de-duplicated on their "v/vt/vn" key so the result can be drawn with an
 * index buffer.
 *
 * @module obj-loader
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define([], factory);
  } else {
    // Browser globals
    root.objLoader = factory();
  }
}(this, function() {
  "use strict";

  /**
   * A chunk of geometry sharing a single material.
   * @typedef {Object} Geometry
   * @property {string} name the "g"/"o" name it came from
   * @property {string} material the "usemtl" name, or "default"
   * @property {Object} arrays position/normal/texcoord/indices, ready for
   *     webglUtils.createBufferInfoFromArrays
   * @memberOf module:obj-loader
   */

  /**
   * Resolves a path relative to the directory another file lives in.
   * @param {string} baseUrl url of the file the reference was found in
   * @param {string} relativeUrl the reference
   * @return {string} the resolved url
   * @memberOf module:obj-loader
   */
  function resolveUrl(baseUrl, relativeUrl) {
    var dir = baseUrl.substring(0, baseUrl.lastIndexOf('/') + 1);
    return dir + relativeUrl.replace(/\\/g, '/');
  }

  /**
   * Parses the text of an OBJ file.
   * @param {string} text contents of the .obj file
   * @return {{geometries: Geometry[], materialLibs: string[]}} parsed result
   * @memberOf module:obj-loader
   */
  function parseObj(text) {
    // OBJ indices are 1 based, so index 0 is a placeholder that is never used.
    var objPositions = [[0, 0, 0]];
    var objTexcoords = [[0, 0]];
    var objNormals = [[0, 0, 0]];
    var objVertexData = [objPositions, objTexcoords, objNormals];

    var materialLibs = [];
    var geometries = [];
    var geometry = null;
    var groupName = 'default';
    var material = 'default';

    // Maps a "v/vt/vn" triple to its index in the current geometry so that
    // vertices shared between faces are only emitted once.
    var vertexMap = null;

    function newGeometry() {
      // Force the next face to start a fresh geometry.
      geometry = null;
    }

    function setGeometry() {
      if (geometry) {
        return;
      }
      var position = [];
      var texcoord = [];
      var normal = [];
      var indices = [];
      vertexMap = Object.create(null);
      geometry = {
        name: groupName,
        material: material,
        arrays: {
          position: {numComponents: 3, data: position},
          texcoord: {numComponents: 2, data: texcoord},
          normal: {numComponents: 3, data: normal},
          indices: indices,
        },
      };
      geometries.push(geometry);
    }

    // Returns the index of the vertex described by "1/2/3" inside the current
    // geometry, adding it if it has not been seen yet.
    function addVertex(vert) {
      setGeometry();
      var existing = vertexMap[vert];
      if (existing !== undefined) {
        return existing;
      }

      var arrays = geometry.arrays;
      var dst = [arrays.position.data, arrays.texcoord.data, arrays.normal.data];
      var parts = vert.split('/');
      for (var i = 0; i < parts.length; ++i) {
        var objIndexStr = parts[i];
        if (!objIndexStr) {
          continue;
        }
        var objIndex = parseInt(objIndexStr, 10);
        var src = objVertexData[i];
        // Negative indices count backwards from the most recent element.
        var index = objIndex + (objIndex >= 0 ? 0 : src.length);
        var value = src[index];
        for (var j = 0; j < value.length; ++j) {
          dst[i].push(value[j]);
        }
      }

      var newIndex = arrays.position.data.length / 3 - 1;
      vertexMap[vert] = newIndex;
      return newIndex;
    }

    var keywords = {
      v: function(parts) {
        objPositions.push([parseFloat(parts[0]), parseFloat(parts[1]), parseFloat(parts[2])]);
      },
      vn: function(parts) {
        objNormals.push([parseFloat(parts[0]), parseFloat(parts[1]), parseFloat(parts[2])]);
      },
      vt: function(parts) {
        // OBJ puts the texture origin at the bottom left, GL sampling in this
        // demo treats row 0 of the image as v = 0, so flip v here once and
        // never think about it again.
        objTexcoords.push([parseFloat(parts[0]), 1 - parseFloat(parts[1])]);
      },
      f: function(parts) {
        setGeometry();
        // webglUtils.drawBufferInfo always draws with UNSIGNED_SHORT indices,
        // so split into another chunk of the same material before a geometry
        // could grow past what a 16 bit index can reach.
        if (geometry.arrays.position.data.length / 3 + parts.length > 65535) {
          newGeometry();
          setGeometry();
        }
        var indices = geometry.arrays.indices;
        var a = addVertex(parts[0]);
        // Triangulate any n-gon as a fan around the first vertex.
        for (var i = 1; i < parts.length - 1; ++i) {
          indices.push(a, addVertex(parts[i]), addVertex(parts[i + 1]));
        }
      },
      g: function(parts) {
        groupName = parts.join(' ') || 'default';
        newGeometry();
      },
      o: function(parts) {
        groupName = parts.join(' ') || 'default';
        newGeometry();
      },
      usemtl: function(parts) {
        material = parts.join(' ') || 'default';
        newGeometry();
      },
      mtllib: function(parts) {
        materialLibs.push(parts.join(' '));
      },
      s: function() {},  // smoothing group, ignored
    };

    var lines = text.split('\n');
    for (var lineNo = 0; lineNo < lines.length; ++lineNo) {
      var line = lines[lineNo].trim();
      if (line === '' || line.charAt(0) === '#') {
        continue;
      }
      var parts = line.split(/\s+/);
      var keyword = parts.shift();
      var handler = keywords[keyword];
      if (!handler) {
        console.warn('obj-loader: unhandled keyword "' + keyword + '" at line ' + (lineNo + 1));
        continue;
      }
      handler(parts, line);
    }

    return {
      geometries: geometries,
      materialLibs: materialLibs,
    };
  }

  /**
   * Parses the text of an MTL file.
   * @param {string} text contents of the .mtl file
   * @return {Object.<string, Object>} materials keyed by name
   * @memberOf module:obj-loader
   */
  function parseMtl(text) {
    var materials = {};
    var material = null;

    function parseFloats(parts) {
      return parts.map(parseFloat);
    }

    var keywords = {
      newmtl: function(parts) {
        material = {};
        materials[parts.join(' ')] = material;
      },
      Ka: function(parts) { material.ambient = parseFloats(parts); },
      Kd: function(parts) { material.diffuse = parseFloats(parts); },
      Ks: function(parts) { material.specular = parseFloats(parts); },
      Ke: function(parts) { material.emissive = parseFloats(parts); },
      Ns: function(parts) { material.shininess = parseFloat(parts[0]); },
      Ni: function(parts) { material.opticalDensity = parseFloat(parts[0]); },
      d: function(parts) { material.opacity = parseFloat(parts[0]); },
      illum: function(parts) { material.illum = parseInt(parts[0], 10); },
    };

    // All map_* / bump / disp keywords store the last token as a file name.
    ['map_Ka', 'map_Kd', 'map_Ks', 'map_Ke', 'map_Ns', 'map_d', 'map_Bump',
     'bump', 'disp', 'refl', 'norm'].forEach(function(keyword) {
      keywords[keyword] = function(parts) {
        material[keyword] = parts[parts.length - 1];
      };
    });

    var lines = text.split('\n');
    for (var lineNo = 0; lineNo < lines.length; ++lineNo) {
      var line = lines[lineNo].trim();
      if (line === '' || line.charAt(0) === '#') {
        continue;
      }
      var parts = line.split(/\s+/);
      var keyword = parts.shift();
      var handler = keywords[keyword];
      if (!handler) {
        console.warn('obj-loader: unhandled mtl keyword "' + keyword + '" at line ' + (lineNo + 1));
        continue;
      }
      handler(parts, line);
    }

    return materials;
  }

  /**
   * Fetches an OBJ plus every MTL it references.
   * @param {string} url url of the .obj file
   * @return {Promise<{geometries: Geometry[], materials: Object, url: string}>}
   * @memberOf module:obj-loader
   */
  function loadObj(url) {
    return fetch(url).then(function(response) {
      if (!response.ok) {
        throw new Error('failed to load ' + url + ': ' + response.status);
      }
      return response.text();
    }).then(function(text) {
      var obj = parseObj(text);
      var mtlPromises = obj.materialLibs.map(function(name) {
        return fetch(resolveUrl(url, name)).then(function(response) {
          if (!response.ok) {
            throw new Error('failed to load ' + name + ': ' + response.status);
          }
          return response.text();
        }).then(parseMtl);
      });
      return Promise.all(mtlPromises).then(function(mtls) {
        var materials = {};
        mtls.forEach(function(mtl) {
          Object.keys(mtl).forEach(function(name) {
            materials[name] = mtl[name];
          });
        });
        return {
          geometries: obj.geometries,
          materials: materials,
          url: url,
        };
      });
    });
  }

  /**
   * Loads an image.
   * @param {string} url url of the image
   * @return {Promise<HTMLImageElement>} the decoded image
   * @memberOf module:obj-loader
   */
  function loadImage(url) {
    return new Promise(function(resolve, reject) {
      var img = new Image();
      img.onload = function() { resolve(img); };
      img.onerror = function() { reject(new Error('failed to load image ' + url)); };
      img.src = url;
    });
  }

  /**
   * Loads an image and returns its raw RGBA8 pixels, top row first.
   * @param {string} url url of the image
   * @return {Promise<{width: number, height: number, data: Uint8Array}>} pixels
   * @memberOf module:obj-loader
   */
  function loadImageData(url) {
    return loadImage(url).then(function(img) {
      var canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      var ctx = canvas.getContext('2d', {willReadFrequently: true});
      ctx.drawImage(img, 0, 0);
      var imageData = ctx.getImageData(0, 0, img.width, img.height);
      return {
        width: img.width,
        height: img.height,
        data: new Uint8Array(imageData.data.buffer.slice(0)),
      };
    });
  }

  /**
   * Computes the axis aligned bounding box of a set of geometries.
   * @param {Geometry[]} geometries the geometries to measure
   * @return {{min: number[], max: number[]}} the bounds
   * @memberOf module:obj-loader
   */
  function getBounds(geometries) {
    var min = [Infinity, Infinity, Infinity];
    var max = [-Infinity, -Infinity, -Infinity];
    geometries.forEach(function(geometry) {
      var position = geometry.arrays.position.data;
      for (var i = 0; i < position.length; i += 3) {
        for (var j = 0; j < 3; ++j) {
          var v = position[i + j];
          if (v < min[j]) { min[j] = v; }
          if (v > max[j]) { max[j] = v; }
        }
      }
    });
    return {min: min, max: max};
  }

  return {
    parseObj: parseObj,
    parseMtl: parseMtl,
    loadObj: loadObj,
    loadImage: loadImage,
    loadImageData: loadImageData,
    getBounds: getBounds,
    resolveUrl: resolveUrl,
  };

}));
