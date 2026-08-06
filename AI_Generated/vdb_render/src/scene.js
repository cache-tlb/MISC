/*
 * Scene contents and the mutable state the UI writes into.
 *
 * @module scene
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./m4.js', './webgl-utils.js'], factory);
  } else {
    // Browser globals
    root.scene = factory(root.m4, root.webglUtils);
  }
}(this, function(m4, webglUtils) {
  "use strict";

  var DEG = Math.PI / 180;

  var ROOM_HALF = 50;      // the floor is 100 x 100, centred on the origin
  var WALL_HEIGHT = 50;

  /**
   * The scene defaults.
   * @param {Object} grid the decoded density grid, for its aspect ratio
   * @return {Object} freshly defaulted state
   * @memberOf module:scene
   */
  function createState(grid) {
    var longest = Math.max(grid.dims[0], grid.dims[1], grid.dims[2]);
    return {
      grid: grid,
      // The medium's unit cube is stretched by this before the UI transform,
      // so a uniform scale keeps the volume undistorted.
      gridAspect: [
        grid.dims[0] / longest,
        grid.dims[1] / longest,
        grid.dims[2] / longest,
      ],
      light: {yaw: 30 * DEG, pitch: -50 * DEG, intensity: 1.6},
      ambient: {intensity: 0.35},
      teapot: {visible: true, position: [0, 0, 0], rotation: [0, 0, 0],
               scale: 0.3},
      medium: {
        visible: true,
        position: [0, 38, 0],
        rotation: [0, 0, 0],
        scale: 32,
        density: 1.0,
        color: [0.95, 0.96, 1.0],
      },
      // Filled in by updateMatrices.
      teapotMatrix: m4.identity(),
      mediumMatrix: m4.identity(),
      mediumInverse: m4.identity(),
      lightDirection: [0, -1, 0],
    };
  }

  /**
   * Recomputes everything derived from the UI values. Cheap enough to run on
   * every change rather than tracking dirty flags.
   * @param {Object} state the scene state
   * @memberOf module:scene
   */
  function updateMatrices(state) {
    var light = state.light;
    var cp = Math.cos(light.pitch);
    // The direction light travels, using the same formula camera.js uses for
    // forward so the two conventions agree.
    state.lightDirection = [
      -Math.sin(light.yaw) * cp,
      Math.sin(light.pitch),
      -Math.cos(light.yaw) * cp,
    ];

    var teapot = state.teapot;
    var m = m4.translation(teapot.position[0], teapot.position[1],
        teapot.position[2]);
    m = m4.yRotate(m, teapot.rotation[1]);
    m = m4.xRotate(m, teapot.rotation[0]);
    m = m4.zRotate(m, teapot.rotation[2]);
    state.teapotMatrix = m4.scale(m, teapot.scale, teapot.scale, teapot.scale);

    var medium = state.medium;
    var v = m4.translation(medium.position[0], medium.position[1],
        medium.position[2]);
    v = m4.yRotate(v, medium.rotation[1]);
    v = m4.xRotate(v, medium.rotation[0]);
    v = m4.zRotate(v, medium.rotation[2]);
    state.mediumMatrix = m4.scale(v,
        medium.scale * state.gridAspect[0],
        medium.scale * state.gridAspect[1],
        medium.scale * state.gridAspect[2]);
    state.mediumInverse = m4.inverse(state.mediumMatrix);
  }

  /**
   * The medium's eight world space corners, for fitting the shadow frustum.
   * @param {Object} state the scene state
   * @return {number[][]} the corners
   * @memberOf module:scene
   */
  function mediumCorners(state) {
    var corners = [];
    for (var i = 0; i < 8; i++) {
      corners.push(m4.transformPoint(state.mediumMatrix, [
        (i & 1) ? 0.5 : -0.5,
        (i & 2) ? 0.5 : -0.5,
        (i & 4) ? 0.5 : -0.5,
      ]));
    }
    return corners;
  }

  /**
   * Builds the room. The floor and the four walls share one buffer: they use
   * the same shader and never move, so there is nothing to gain from splitting
   * them.
   * @param {WebGL2RenderingContext} gl the context
   * @return {Object} a bufferInfo for webglUtils
   * @memberOf module:scene
   */
  function buildRoom(gl) {
    var position = [];
    var normal = [];
    var indices = [];

    /**
     * Adds a quad spanned by u and v from corner o. The winding follows
     * cross(u, v), so passing u and v in that order fixes the facing.
     */
    function addQuad(o, u, v) {
      var base = position.length / 3;
      var n = m4.normalize(m4.cross(u, v));
      var corners = [
        o,
        [o[0] + u[0], o[1] + u[1], o[2] + u[2]],
        [o[0] + u[0] + v[0], o[1] + u[1] + v[1], o[2] + u[2] + v[2]],
        [o[0] + v[0], o[1] + v[1], o[2] + v[2]],
      ];
      for (var i = 0; i < 4; i++) {
        position.push(corners[i][0], corners[i][1], corners[i][2]);
        normal.push(n[0], n[1], n[2]);
      }
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }

    var s = ROOM_HALF;
    var h = WALL_HEIGHT;

    // Floor, facing up.
    addQuad([-s, 0, -s], [0, 0, 2 * s], [2 * s, 0, 0]);
    // Walls, each facing inward.
    addQuad([-s, 0, s], [0, 0, -2 * s], [0, h, 0]);   // x = -s, faces +x
    addQuad([s, 0, -s], [0, 0, 2 * s], [0, h, 0]);    // x = +s, faces -x
    addQuad([-s, 0, -s], [2 * s, 0, 0], [0, h, 0]);   // z = -s, faces +z
    addQuad([s, 0, s], [-2 * s, 0, 0], [0, h, 0]);    // z = +s, faces -z

    return webglUtils.createBufferInfoFromArrays(gl, {
      position: {numComponents: 3, data: position},
      normal: {numComponents: 3, data: normal},
      indices: {numComponents: 3, data: new Uint16Array(indices)},
    });
  }

  /**
   * The scene's world space bounds, used to fit the shadow map frustum.
   * @param {Object} state the scene state
   * @param {Object} teapotBbox the teapot's local bbox
   * @return {number[][]} the eight corners of the scene bounds
   * @memberOf module:scene
   */
  function sceneCorners(state, teapotBbox) {
    var points = [];
    var i;

    // The room.
    for (i = 0; i < 8; i++) {
      points.push([
        (i & 1) ? ROOM_HALF : -ROOM_HALF,
        (i & 2) ? WALL_HEIGHT : 0,
        (i & 4) ? ROOM_HALF : -ROOM_HALF,
      ]);
    }
    // The teapot and the medium, but only while they are shown: a hidden
    // object must not keep stretching the frustum and coarsening the map.
    if (state.teapot.visible) {
      for (i = 0; i < 8; i++) {
        points.push(m4.transformPoint(state.teapotMatrix, [
          (i & 1) ? teapotBbox.max[0] : teapotBbox.min[0],
          (i & 2) ? teapotBbox.max[1] : teapotBbox.min[1],
          (i & 4) ? teapotBbox.max[2] : teapotBbox.min[2],
        ]));
      }
    }
    if (state.medium.visible) {
      points = points.concat(mediumCorners(state));
    }

    return points;
  }

  return {
    createState: createState,
    updateMatrices: updateMatrices,
    buildRoom: buildRoom,
    sceneCorners: sceneCorners,
    mediumCorners: mediumCorners,
    ROOM_HALF: ROOM_HALF,
    WALL_HEIGHT: WALL_HEIGHT,
  };

}));
