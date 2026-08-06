'use strict';
var test = require('node:test');
var assert = require('node:assert');
var fs = require('node:fs');
var path = require('node:path');
var vdbLoader = require('../src/vdb-loader.js');

var VDB = path.join(__dirname, '..', 'assets', 'bunny_cloud.vdb');

/** Loads any asset by file name, with no caching. */
function loadFile(fileName, options) {
  var buf = fs.readFileSync(path.join(__dirname, '..', 'assets', fileName));
  options = options || {};
  options.maxResolution = options.maxResolution || 256;
  return vdbLoader.load(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
      options);
}

var loaded = null;
function load() {
  if (!loaded) {
    var buf = fs.readFileSync(VDB);
    loaded = vdbLoader.load(
        buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
        {maxResolution: 256});
  }
  return loaded;
}

test('decodes the exact active voxel count from the file metadata', function() {
  assert.strictEqual(load().activeVoxelCount, 19210271);
});

test('decodes the exact active bounding box from the file metadata', function() {
  var g = load();
  assert.deepStrictEqual(g.bboxMin, [-300, -47, -208]);
  assert.deepStrictEqual(g.bboxMax, [276, 524, 229]);
});

// Measured from the file: the active values span 0 .. 2.7922983, mean 0.3044,
// with 4.5% above 1.0. This Houdini fog volume is NOT normalised to 0..1.
var SOURCE_MAX_DENSITY = 2.7922983169555664;

test('box filtering never exceeds the source maximum density', function() {
  var g = load();
  assert.ok(g.maxDensity > 0, 'grid is empty');
  // An average over a cell cannot exceed the largest value in that cell, so a
  // result above the source maximum means the filter divisor is wrong.
  assert.ok(g.maxDensity <= SOURCE_MAX_DENSITY + 1e-4,
      'maxDensity ' + g.maxDensity + ' exceeds the source maximum');
  // ...and it should be close to it, or the filter is washing the grid out.
  assert.ok(g.maxDensity > SOURCE_MAX_DENSITY * 0.5,
      'maxDensity ' + g.maxDensity + ' is suspiciously low');
});

test('every leaf value is finite and non-negative', function() {
  var g = load();
  assert.ok(Number.isFinite(g.maxDensity), 'maxDensity is not finite');
  // data is Uint8 so it cannot be negative; check the grid actually resolves
  // the full 0..255 range rather than collapsing to a couple of buckets.
  var seen = new Set();
  for (var i = 0; i < g.data.length; i += 97) {
    seen.add(g.data[i]);
  }
  assert.ok(seen.size > 32, 'only ' + seen.size + ' distinct density levels');
});

test('target grid keeps the source aspect ratio', function() {
  var g = load();
  assert.strictEqual(g.name, 'density');
  assert.strictEqual(Math.max(g.dims[0], g.dims[1], g.dims[2]), 256);
  assert.strictEqual(g.data.length, g.dims[0] * g.dims[1] * g.dims[2]);
  // 577 x 572 x 438 scaled so the longest axis is 256.
  assert.deepStrictEqual(g.dims, [256, 254, 194]);
});

// smoke.vdb is Tree_float_5_4_3_HalfFloat: its leaf and tile buffers hold 16
// bit values, and HalfReader skips an empty buffer entirely instead of reading
// the compressed size header the full width path always reads.
test('decodes a half float grid', function() {
  var g = loadFile('smoke.vdb');
  // Both numbers come from the file's own metadata, so they are exact.
  assert.strictEqual(g.activeVoxelCount, 1049275);
  assert.deepStrictEqual(g.bboxMin, [1, 2, 1]);
  assert.deepStrictEqual(g.bboxMax, [111, 223, 112]);
  assert.ok(g.maxDensity > 0 && Number.isFinite(g.maxDensity),
      'maxDensity = ' + g.maxDensity);
});

// maxResolution caps the grid, it does not stretch a small one up to it.
// Upsampling would scatter each source voxel into one target cell and leave
// regular planes of empty cells behind, which shows up as a lattice of holes.
test('never upsamples a grid smaller than maxResolution', function() {
  var g = loadFile('smoke.vdb');
  // Source bbox is 111 x 222 x 112, so the grid must come out 1:1.
  assert.deepStrictEqual(g.dims, [111, 222, 112]);
});

/** Counts fully empty planes perpendicular to one axis. */
function emptyPlanes(g, axis) {
  var nx = g.dims[0], ny = g.dims[1], nz = g.dims[2];
  var occupied = new Uint8Array(g.dims[axis]);
  for (var z = 0; z < nz; z++) {
    for (var y = 0; y < ny; y++) {
      var row = y * nx + z * nx * ny;
      for (var x = 0; x < nx; x++) {
        if (g.data[row + x] > 0) {
          occupied[axis === 0 ? x : (axis === 1 ? y : z)] = 1;
        }
      }
    }
  }
  var empty = 0;
  for (var i = 0; i < occupied.length; i++) {
    if (!occupied[i]) empty++;
  }
  return empty;
}

test('leaves no empty lattice in the decoded grid', function() {
  var g = loadFile('smoke.vdb');
  for (var axis = 0; axis < 3; axis++) {
    var empty = emptyPlanes(g, axis);
    // A few empty planes at the padded edges of the bbox are normal; the
    // upsampling bug instead left about one plane in eight blank, spread
    // evenly through the volume, which is what read as a grid on screen.
    assert.ok(empty < g.dims[axis] * 0.08,
        'axis ' + 'xyz'[axis] + ': ' + empty + ' of ' + g.dims[axis] +
        ' planes empty');
  }
});

// explosion.vdb carries density, temperature and a vec3s velocity field.
test('picks the density grid out of a multi grid file', function() {
  var g = loadFile('explosion.vdb');
  assert.strictEqual(g.name, 'density');
  assert.strictEqual(g.activeVoxelCount, 3948553);
  assert.deepStrictEqual(g.bboxMin, [64, 39, 41]);
  assert.deepStrictEqual(g.bboxMax, [264, 310, 270]);
});

test('can be pointed at a specific grid by name', function() {
  var g = loadFile('explosion.vdb', {gridName: 'temperature'});
  assert.strictEqual(g.name, 'temperature');
  assert.strictEqual(g.activeVoxelCount, 3948353);
});

test('rejects a grid name that is not a usable float grid', function() {
  // "v" exists but is Tree_vec3s_5_4_3, so it is not eligible.
  assert.throws(function() { loadFile('explosion.vdb', {gridName: 'v'}); },
      /no float grid named/i);
});

test('the grid is not empty and not saturated', function() {
  var g = load();
  var nonZero = 0;
  for (var i = 0; i < g.data.length; i++) {
    if (g.data[i] > 0) nonZero++;
  }
  var frac = nonZero / g.data.length;
  assert.ok(frac > 0.02, 'only ' + (frac * 100).toFixed(2) + '% non-zero');
  assert.ok(frac < 0.90, (frac * 100).toFixed(2) + '% non-zero, suspiciously full');
});
