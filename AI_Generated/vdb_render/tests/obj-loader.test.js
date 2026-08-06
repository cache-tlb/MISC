'use strict';
var test = require('node:test');
var assert = require('node:assert');
var fs = require('node:fs');
var path = require('node:path');
var objLoader = require('../src/obj-loader.js');

var OBJ = path.join(__dirname, '..', 'assets', 'teapot', 'teapot.obj');

var teapot = null;
function loadTeapot() {
  if (!teapot) {
    teapot = objLoader.parse(fs.readFileSync(OBJ, 'utf8'));
  }
  return teapot;
}

test('triangulates the teapot faces', function() {
  // The file is mixed: 7676 quads (2 triangles each) and 352 triangles.
  assert.strictEqual(loadTeapot().indices.length, (7676 * 2 + 352) * 3);
});

test('generates unit length normals when the file has none', function() {
  var mesh = loadTeapot();
  assert.strictEqual(mesh.normal.length, mesh.position.length);
  for (var i = 0; i < mesh.normal.length; i += 3) {
    var x = mesh.normal[i], y = mesh.normal[i + 1], z = mesh.normal[i + 2];
    var len = Math.sqrt(x * x + y * y + z * z);
    assert.ok(Math.abs(len - 1) < 1e-4, 'normal ' + (i / 3) + ' length ' + len);
  }
});

test('recentres on x/z and rests on y = 0', function() {
  var mesh = loadTeapot();
  // Source bbox is (-71.893, 0, -47.928)..(82.294, 79.007, 47.928).
  assert.ok(Math.abs(mesh.bbox.min[0] + mesh.bbox.max[0]) < 1e-3, 'x not centred');
  assert.ok(Math.abs(mesh.bbox.min[2] + mesh.bbox.max[2]) < 1e-3, 'z not centred');
  assert.ok(Math.abs(mesh.bbox.min[1]) < 1e-3, 'does not rest on y=0');
  assert.ok(Math.abs((mesh.bbox.max[0] - mesh.bbox.min[0]) - 154.187) < 1e-2);
  assert.ok(Math.abs((mesh.bbox.max[1] - mesh.bbox.min[1]) - 79.007) < 1e-2);
  assert.ok(Math.abs((mesh.bbox.max[2] - mesh.bbox.min[2]) - 95.857) < 1e-2);
});

test('smooths shared vertices rather than faceting', function() {
  // Two quads meeting along an edge, bent 90 degrees. The shared vertices must
  // average both face normals, which is what keeps the teapot smooth.
  var mesh = objLoader.parse([
    'v 0 0 0', 'v 1 0 0', 'v 1 0 -1', 'v 0 0 -1',   // flat quad
    'v 1 1 0', 'v 1 1 -1',                          // raised edge
    'f 1 2 3 4',
    'f 2 5 6 3',
  ].join('\n'));
  // Vertex 2 (index 1) is shared by both faces, whose normals point along
  // different axes, so the averaged normal must be diagonal. Faceting would
  // leave it aligned to one axis.
  var nx = mesh.normal[1 * 3], ny = mesh.normal[1 * 3 + 1];
  assert.ok(Math.abs(nx) > 0.3 && Math.abs(ny) > 0.3,
      'shared normal was not averaged: ' + nx + ',' + ny);
});

test('parses face index forms', function() {
  var mesh = objLoader.parse([
    'v 0 0 0', 'v 1 0 0', 'v 0 1 0', 'v 1 1 0',
    'vt 0 0', 'vt 1 0', 'vt 0 1', 'vt 1 1',
    'f 1/1 2/2 4/4 3/3',
  ].join('\n'));
  assert.strictEqual(mesh.indices.length, 6);
  assert.strictEqual(mesh.position.length, 4 * 3);
});

test('resolves negative indices', function() {
  var mesh = objLoader.parse([
    'v 0 0 0', 'v 1 0 0', 'v 0 1 0', 'f -3 -2 -1',
  ].join('\n'));
  assert.deepStrictEqual(Array.from(mesh.indices), [0, 1, 2]);
});

test('keeps supplied normals instead of generating them', function() {
  var mesh = objLoader.parse([
    'v 0 0 0', 'v 1 0 0', 'v 0 1 0',
    'vn 0 0 1', 'vn 0 0 1', 'vn 0 0 1',
    'f 1//1 2//2 3//3',
  ].join('\n'));
  assert.deepStrictEqual(Array.from(mesh.normal.slice(0, 3)), [0, 0, 1]);
});
