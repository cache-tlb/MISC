/*
 * Geometry for the stereographic projection demo.
 *
 * Nothing here talks to WebGL. Every builder returns a plain "arrays" object in
 * the shape `webglUtils.createBufferInfoFromArrays` expects, so the renderer can
 * upload it without any further massaging.
 *
 * The scene is right handed like OpenGL: +x right, +y up, +z toward the viewer.
 * The sphere the demo projects onto is the unit sphere at the origin and the
 * projection centre is (0, 0, h) for a user controlled h.
 *
 * @module geometry
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define([], factory);
  } else {
    // Browser globals
    root.geometry = factory();
  }
}(this, function() {
  "use strict";

  /**
   * Stage codes. They travel all the way into the shaders, where they select
   * which deformation to apply to a vertex, so keep them in sync with the
   * `deform()` function in renderer.js.
   * @enum {number}
   * @memberOf module:geometry
   */
  var STAGE = {
    M: 0,       // the grid where the model matrix puts it
    S: 1,       // M pushed onto the unit sphere along the ray from the origin
    P: 2,       // S projected from (0,0,h) onto the equatorial plane z = 0
    RAW: 3,     // a plain world space point, used by the helper rays
    CENTER: 4,  // the projection centre (0,0,h); the vertex position is ignored
  };

  /**
   * Builds the subdivided square plane M.
   *
   * With `segments` = 16 this is the mesh the demo asks for: 17 * 17 = 289
   * vertices and 16 * 16 * 2 = 512 triangles. The plane lies in the local xy
   * plane facing +z, and its texture coordinates run 0..1 across each axis so
   * the checker pattern can be generated procedurally.
   *
   * @param {number} halfSize half the side length of the square
   * @param {number} segments quads along one side
   * @return {Object} `surface` and `wireframe` arrays plus vertex/triangle counts
   * @memberOf module:geometry
   */
  function createSubdividedPlane(halfSize, segments) {
    var across = segments + 1;
    var numVertices = across * across;
    var positions = new Float32Array(numVertices * 3);
    var texcoords = new Float32Array(numVertices * 2);

    for (var iy = 0; iy < across; ++iy) {
      for (var ix = 0; ix < across; ++ix) {
        var v = iy * across + ix;
        var u0 = ix / segments;
        var v0 = iy / segments;
        positions[v * 3 + 0] = (u0 * 2 - 1) * halfSize;
        positions[v * 3 + 1] = (v0 * 2 - 1) * halfSize;
        positions[v * 3 + 2] = 0;
        texcoords[v * 2 + 0] = u0;
        texcoords[v * 2 + 1] = v0;
      }
    }

    // Two counter clockwise triangles per quad, as seen from +z.
    var indices = new Uint16Array(segments * segments * 6);
    var t = 0;
    for (iy = 0; iy < segments; ++iy) {
      for (ix = 0; ix < segments; ++ix) {
        var a = iy * across + ix;
        var b = a + 1;
        var c = a + across;
        var d = c + 1;
        indices[t++] = a; indices[t++] = b; indices[t++] = d;
        indices[t++] = a; indices[t++] = d; indices[t++] = c;
      }
    }

    // Every grid edge exactly once, for the wireframe overlay.
    var lineIndices = new Uint16Array(2 * across * segments * 2);
    var e = 0;
    for (iy = 0; iy < across; ++iy) {
      for (ix = 0; ix < segments; ++ix) {
        var h = iy * across + ix;
        lineIndices[e++] = h; lineIndices[e++] = h + 1;
      }
    }
    for (ix = 0; ix < across; ++ix) {
      for (iy = 0; iy < segments; ++iy) {
        var w = iy * across + ix;
        lineIndices[e++] = w; lineIndices[e++] = w + across;
      }
    }

    return {
      numVertices: numVertices,
      numTriangles: segments * segments * 2,
      surface: {
        position: {numComponents: 3, data: positions},
        texcoord: {numComponents: 2, data: texcoords},
        indices: indices,
      },
      wireframe: {
        position: {numComponents: 3, data: positions},
        indices: lineIndices,
      },
    };
  }

  /**
   * Builds the construction rays that explain how S and P are obtained.
   *
   * Only every `stride`-th grid vertex gets a ray, otherwise the picture turns
   * into a hairball. Each vertex carries a stage code, so a single shader can
   * place both ends of a segment on different meshes. Each family is drawn as
   * the whole ray, and the mesh it constructs lies on it:
   *
   *   radial  origin -> M     crosses the unit sphere exactly at S
   *   polar   (0,0,h) -> S    crosses the plane z = 0 exactly at P
   *
   * @param {number} halfSize half the side length of the square
   * @param {number} segments quads along one side
   * @param {number} stride sample every stride-th vertex in both directions
   * @return {Object} `radial` and `polar` arrays
   * @memberOf module:geometry
   */
  function createProjectionRays(halfSize, segments, stride) {
    var radial = createStagedLineBuilder();
    var polar = createStagedLineBuilder();

    for (var iy = 0; iy <= segments; iy += stride) {
      for (var ix = 0; ix <= segments; ix += stride) {
        var x = (ix / segments * 2 - 1) * halfSize;
        var y = (iy / segments * 2 - 1) * halfSize;
        radial.line([0, 0, 0], STAGE.RAW, [x, y, 0], STAGE.M);
        polar.line([0, 0, 0], STAGE.CENTER, [x, y, 0], STAGE.S);
      }
    }

    return {
      radial: radial.arrays(),
      polar: polar.arrays(),
    };
  }

  /**
   * Collects line segments whose endpoints each live on a different mesh.
   * @return {Object} a builder with `line` and `arrays`
   */
  function createStagedLineBuilder() {
    var positions = [];
    var stages = [];

    return {
      line: function(from, fromStage, to, toStage) {
        positions.push(from[0], from[1], from[2], to[0], to[1], to[2]);
        stages.push(fromStage, toStage);
      },
      arrays: function() {
        return {
          position: {numComponents: 3, data: new Float32Array(positions)},
          stage: {numComponents: 1, data: new Float32Array(stages)},
        };
      },
    };
  }

  /**
   * Collects plain coloured line segments for the reference helpers.
   * @return {Object} a builder with `line` and `arrays`
   */
  function createLineBuilder() {
    var positions = [];
    var colors = [];

    return {
      line: function(from, to, color) {
        positions.push(from[0], from[1], from[2], to[0], to[1], to[2]);
        colors.push(color[0], color[1], color[2], color[3]);
        colors.push(color[0], color[1], color[2], color[3]);
      },
      arrays: function() {
        return {
          position: {numComponents: 3, data: new Float32Array(positions)},
          color: {numComponents: 4, data: new Float32Array(colors)},
        };
      },
    };
  }

  /**
   * A latitude / longitude wireframe of the unit sphere. The poles sit on the
   * z axis, matching the projection pole at (0, 0, 1).
   *
   * @param {number} radius sphere radius
   * @param {number} meridians lines of constant azimuth
   * @param {number} parallels bands between the poles
   * @param {number} segments segments used to draw one full circle
   * @param {number[]} color rgba
   * @param {number[]} highlight rgba used for the equator
   * @return {Object} arrays for gl.LINES
   * @memberOf module:geometry
   */
  function createSphereWireframe(radius, meridians, parallels, segments, color, highlight) {
    var builder = createLineBuilder();

    function point(theta, phi) {
      var st = Math.sin(theta);
      return [
        radius * st * Math.cos(phi),
        radius * st * Math.sin(phi),
        radius * Math.cos(theta),
      ];
    }

    for (var i = 0; i < meridians; ++i) {
      var phi = i / meridians * Math.PI * 2;
      for (var j = 0; j < segments; ++j) {
        builder.line(
            point(j / segments * Math.PI, phi),
            point((j + 1) / segments * Math.PI, phi),
            color);
      }
    }

    for (var k = 1; k < parallels; ++k) {
      var theta = k / parallels * Math.PI;
      var bandColor = Math.abs(theta - Math.PI * 0.5) < 1e-6 ? highlight : color;
      for (var s = 0; s < segments * 2; ++s) {
        builder.line(
            point(theta, s / (segments * 2) * Math.PI * 2),
            point(theta, (s + 1) / (segments * 2) * Math.PI * 2),
            bandColor);
      }
    }

    return builder.arrays();
  }

  /**
   * A grid drawn on a plane of constant z, so the viewer can see where P lands.
   *
   * @param {number} z the plane
   * @param {number} halfSize half the side length of the grid
   * @param {number} divisions cells along one side
   * @param {number[]} color rgba
   * @param {number[]} borderColor rgba for the outline
   * @return {Object} arrays for gl.LINES
   * @memberOf module:geometry
   */
  function createPlaneGrid(z, halfSize, divisions, color, borderColor) {
    var builder = createLineBuilder();

    for (var i = 0; i <= divisions; ++i) {
      var t = (i / divisions * 2 - 1) * halfSize;
      var edge = (i === 0 || i === divisions) ? borderColor : color;
      builder.line([t, -halfSize, z], [t, halfSize, z], edge);
      builder.line([-halfSize, t, z], [halfSize, t, z], edge);
    }

    return builder.arrays();
  }

  /**
   * The world axes. The negative half of each axis is drawn dimmer so the
   * orientation stays readable.
   *
   * @param {number} length how far each axis reaches
   * @return {Object} arrays for gl.LINES
   * @memberOf module:geometry
   */
  function createAxes(length) {
    var builder = createLineBuilder();
    var axes = [
      {dir: [1, 0, 0], color: [0.90, 0.32, 0.34, 0.85]},
      {dir: [0, 1, 0], color: [0.40, 0.82, 0.40, 0.85]},
      {dir: [0, 0, 1], color: [0.34, 0.56, 0.95, 0.85]},
    ];

    axes.forEach(function(axis) {
      var d = axis.dir;
      var dim = [axis.color[0], axis.color[1], axis.color[2], 0.28];
      builder.line([0, 0, 0], [d[0] * length, d[1] * length, d[2] * length], axis.color);
      builder.line([0, 0, 0], [-d[0] * length, -d[1] * length, -d[2] * length], dim);
    });

    return builder.arrays();
  }

  /**
   * A three armed cross built around the origin. The renderer slides it onto
   * the projection centre, which moves whenever h changes.
   *
   * @param {number} size half length of each arm
   * @param {number[]} color rgba
   * @return {Object} arrays for gl.LINES
   * @memberOf module:geometry
   */
  function createCrossMarker(size, color) {
    var builder = createLineBuilder();
    builder.line([-size, 0, 0], [size, 0, 0], color);
    builder.line([0, -size, 0], [0, size, 0], color);
    builder.line([0, 0, -size], [0, 0, size], color);
    return builder.arrays();
  }

  return {
    STAGE: STAGE,
    createSubdividedPlane: createSubdividedPlane,
    createProjectionRays: createProjectionRays,
    createSphereWireframe: createSphereWireframe,
    createPlaneGrid: createPlaneGrid,
    createAxes: createAxes,
    createCrossMarker: createCrossMarker,
  };

}));
