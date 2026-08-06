/*
 * A Wavefront OBJ parser producing arrays ready for
 * webglUtils.createBufferInfoFromArrays.
 *
 * Faces of any vertex count are triangulated as a fan, vertices are
 * deduplicated on their v/vt/vn triple, and normals are generated when the
 * file does not supply them, which is the case for the teapot.
 *
 * @module objLoader
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS, so the Node tests can require this directly.
    module.exports = factory();
  } else {
    // Browser globals
    root.objLoader = factory();
  }
}(this, function() {
  "use strict";

  /**
   * Resolves a 1 based OBJ index, where negatives count back from the end.
   * @param {string} token the index text
   * @param {number} count how many of that element have been read so far
   * @return {number} a 0 based index, or -1 when the token is empty
   */
  function resolveIndex(token, count) {
    if (!token) {
      return -1;
    }
    var i = parseInt(token, 10);
    if (isNaN(i)) {
      return -1;
    }
    return i < 0 ? count + i : i - 1;
  }

  /**
   * Parses OBJ text.
   * @param {string} text the file contents
   * @return {Object} position, normal, texcoord, indices and bbox
   * @memberOf module:objLoader
   */
  function parse(text) {
    var rawPosition = [];
    var rawTexcoord = [];
    var rawNormal = [];

    var position = [];
    var texcoord = [];
    var normal = [];
    var indices = [];

    var seen = Object.create(null);
    var hasNormals = false;

    var lines = text.split(/\r?\n/);

    for (var li = 0; li < lines.length; li++) {
      var line = lines[li].trim();
      if (!line || line.charAt(0) === '#') {
        continue;
      }
      var parts = line.split(/\s+/);
      var keyword = parts[0];

      if (keyword === 'v') {
        rawPosition.push(parseFloat(parts[1]), parseFloat(parts[2]),
            parseFloat(parts[3]));
      } else if (keyword === 'vt') {
        rawTexcoord.push(parseFloat(parts[1]), parseFloat(parts[2]));
      } else if (keyword === 'vn') {
        rawNormal.push(parseFloat(parts[1]), parseFloat(parts[2]),
            parseFloat(parts[3]));
        hasNormals = true;
      } else if (keyword === 'f') {
        var face = [];
        for (var vi = 1; vi < parts.length; vi++) {
          var token = parts[vi];
          var index = seen[token];
          if (index === undefined) {
            var fields = token.split('/');
            var pi = resolveIndex(fields[0], rawPosition.length / 3);
            var ti = resolveIndex(fields[1], rawTexcoord.length / 2);
            var ni = resolveIndex(fields[2], rawNormal.length / 3);

            index = position.length / 3;
            position.push(rawPosition[pi * 3], rawPosition[pi * 3 + 1],
                rawPosition[pi * 3 + 2]);
            if (ti >= 0) {
              texcoord.push(rawTexcoord[ti * 2], rawTexcoord[ti * 2 + 1]);
            } else {
              texcoord.push(0, 0);
            }
            if (ni >= 0) {
              normal.push(rawNormal[ni * 3], rawNormal[ni * 3 + 1],
                  rawNormal[ni * 3 + 2]);
            } else {
              normal.push(0, 0, 0);
            }
            seen[token] = index;
          }
          face.push(index);
        }
        // Fan triangulation, which is correct for the convex quads OBJ uses.
        for (var f = 1; f + 1 < face.length; f++) {
          indices.push(face[0], face[f], face[f + 1]);
        }
      }
      // Everything else, including g, s, usemtl and mtllib, is ignored.
    }

    if (!hasNormals) {
      generateNormals(position, normal, indices);
    }

    var bbox = recentre(position);

    // 16 bit indices where they fit, because webgl-utils' drawBufferInfo
    // always draws with UNSIGNED_SHORT and must not be modified.
    var vertexCount = position.length / 3;
    var IndexArray = vertexCount > 65535 ? Uint32Array : Uint16Array;

    return {
      position: new Float32Array(position),
      normal: new Float32Array(normal),
      texcoord: new Float32Array(texcoord),
      indices: new IndexArray(indices),
      bbox: bbox,
    };
  }

  /**
   * Accumulates face normals into their vertices and normalises.
   *
   * The face normal is deliberately left un-normalised while accumulating: the
   * raw cross product is proportional to twice the triangle's area, so larger
   * triangles carry proportionally more weight, which is what makes a curved
   * surface like the teapot come out smooth.
   *
   * @param {number[]} position xyz per vertex
   * @param {number[]} normal xyz per vertex, filled in place
   * @param {number[]} indices triangle list
   */
  function generateNormals(position, normal, indices) {
    var i;
    for (i = 0; i < normal.length; i++) {
      normal[i] = 0;
    }

    for (i = 0; i < indices.length; i += 3) {
      var a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;

      var abx = position[b] - position[a];
      var aby = position[b + 1] - position[a + 1];
      var abz = position[b + 2] - position[a + 2];
      var acx = position[c] - position[a];
      var acy = position[c + 1] - position[a + 1];
      var acz = position[c + 2] - position[a + 2];

      var nx = aby * acz - abz * acy;
      var ny = abz * acx - abx * acz;
      var nz = abx * acy - aby * acx;

      normal[a] += nx; normal[a + 1] += ny; normal[a + 2] += nz;
      normal[b] += nx; normal[b + 1] += ny; normal[b + 2] += nz;
      normal[c] += nx; normal[c + 1] += ny; normal[c + 2] += nz;
    }

    for (i = 0; i < normal.length; i += 3) {
      var x = normal[i], y = normal[i + 1], z = normal[i + 2];
      var len = Math.sqrt(x * x + y * y + z * z);
      if (len > 0) {
        normal[i] = x / len;
        normal[i + 1] = y / len;
        normal[i + 2] = z / len;
      } else {
        normal[i + 1] = 1;
      }
    }
  }

  /**
   * Centres the model on x and z and drops it onto y = 0, so a UI position of
   * the origin stands it on the floor in the middle of the room.
   * @param {number[]} position xyz per vertex, modified in place
   * @return {Object} the bbox after recentring
   */
  function recentre(position) {
    if (position.length === 0) {
      return {min: [0, 0, 0], max: [0, 0, 0]};
    }

    var min = [Infinity, Infinity, Infinity];
    var max = [-Infinity, -Infinity, -Infinity];
    var i, a;

    for (i = 0; i < position.length; i += 3) {
      for (a = 0; a < 3; a++) {
        if (position[i + a] < min[a]) min[a] = position[i + a];
        if (position[i + a] > max[a]) max[a] = position[i + a];
      }
    }

    var offset = [
      (min[0] + max[0]) * 0.5,
      min[1],
      (min[2] + max[2]) * 0.5,
    ];

    for (i = 0; i < position.length; i += 3) {
      position[i] -= offset[0];
      position[i + 1] -= offset[1];
      position[i + 2] -= offset[2];
    }

    return {
      min: [min[0] - offset[0], min[1] - offset[1], min[2] - offset[2]],
      max: [max[0] - offset[0], max[1] - offset[1], max[2] - offset[2]],
    };
  }

  return {
    parse: parse,
  };

}));
