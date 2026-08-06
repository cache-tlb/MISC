/*
 * A reader for OpenVDB files, producing a dense downsampled density grid ready
 * to upload as a 3D texture.
 *
 * Supports file format version 222 and newer, scalar float trees in either
 * full or half precision, blosc plus active-mask compression, and the scale,
 * translate and affine transform maps. Files may hold several grids; the
 * density one is chosen unless options.gridName says otherwise. Anything else
 * throws with the offending value rather than quietly producing a wrong grid,
 * and the leaf walk is checked against the grid's recorded end offset so a
 * misread value layout cannot pass silently.
 *
 * The dense grid a file describes can be enormous, 144 million voxels for
 * bunny_cloud.vdb, so it is never materialised: each 8^3 leaf is box-filter
 * accumulated straight into the smaller target grid as it is read.
 *
 * @module vdbLoader
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['./blosc.js'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS, so the Node tests can require this directly.
    module.exports = factory(require('./blosc.js'));
  } else {
    // Browser globals
    root.vdbLoader = factory(root.blosc);
  }
}(this, function(blosc) {
  "use strict";

  // Metadata flags for io::readCompressedValues.
  var NO_MASK_OR_INACTIVE_VALS = 0;
  var NO_MASK_AND_MINUS_BG = 1;
  var NO_MASK_AND_ONE_INACTIVE_VAL = 2;
  var MASK_AND_NO_INACTIVE_VALS = 3;
  var MASK_AND_ONE_INACTIVE_VAL = 4;
  var MASK_AND_TWO_INACTIVE_VALS = 5;
  var NO_MASK_AND_ALL_VALS = 6;

  var COMPRESS_ZIP = 0x1;
  var COMPRESS_ACTIVE_MASK = 0x2;
  var COMPRESS_BLOSC = 0x4;

  // Doubles serialised by each math::Map subclass, measured against the file.
  // ScaleMap and friends write their precomputed reciprocals too.
  var MAP_DOUBLE_COUNTS = {
    'UniformScaleMap': 15,
    'ScaleMap': 15,
    'UniformScaleTranslateMap': 18,
    'ScaleTranslateMap': 18,
    'TranslationMap': 3,
    'AffineMap': 16,
  };

  var HALF_TO_FLOAT = null;

  var POPCOUNT = new Uint8Array(256);
  for (var pcI = 0; pcI < 256; pcI++) {
    POPCOUNT[pcI] = (pcI & 1) + POPCOUNT[pcI >> 1];
  }

  /**
   * A little-endian cursor over an ArrayBuffer.
   * @param {ArrayBuffer} arrayBuffer the whole file
   * @constructor
   */
  function Reader(arrayBuffer) {
    this.view = new DataView(arrayBuffer);
    this.bytes = new Uint8Array(arrayBuffer);
    this.pos = 0;
  }

  Reader.prototype.uint8 = function() {
    return this.bytes[this.pos++];
  };
  Reader.prototype.uint32 = function() {
    var v = this.view.getUint32(this.pos, true); this.pos += 4; return v;
  };
  Reader.prototype.int32 = function() {
    var v = this.view.getInt32(this.pos, true); this.pos += 4; return v;
  };
  Reader.prototype.uint64 = function() {
    var v = Number(this.view.getBigUint64(this.pos, true)); this.pos += 8; return v;
  };
  Reader.prototype.int64 = function() {
    var v = Number(this.view.getBigInt64(this.pos, true)); this.pos += 8; return v;
  };
  Reader.prototype.float32 = function() {
    var v = this.view.getFloat32(this.pos, true); this.pos += 4; return v;
  };
  Reader.prototype.float64 = function() {
    var v = this.view.getFloat64(this.pos, true); this.pos += 8; return v;
  };
  Reader.prototype.vec3i = function() {
    return [this.int32(), this.int32(), this.int32()];
  };
  Reader.prototype.skip = function(n) {
    this.pos += n;
  };
  /** Reads a length prefixed string. */
  Reader.prototype.string = function() {
    var n = this.uint32();
    var s = '';
    for (var i = 0; i < n; i++) {
      s += String.fromCharCode(this.bytes[this.pos + i]);
    }
    this.pos += n;
    return s;
  };
  /** Returns the file offset of an n bit mask and steps over it. */
  Reader.prototype.maskOffset = function(bits) {
    var at = this.pos;
    this.pos += bits >> 3;
    return at;
  };

  /**
   * Counts the set bits of a mask stored at a file offset.
   * @param {Uint8Array} bytes the file
   * @param {number} at offset of the mask
   * @param {number} nbytes mask length in bytes
   * @return {number} number of set bits
   */
  function countOn(bytes, at, nbytes) {
    var n = 0;
    for (var i = 0; i < nbytes; i++) {
      n += POPCOUNT[bytes[at + i]];
    }
    return n;
  }

  /**
   * Reads a metadata map. Every entry's payload is exactly the byte count that
   * precedes it, strings included, so unknown types cost nothing to skip.
   * @param {Reader} r the cursor
   * @return {Object} name to value, for the types we care about
   */
  function readMetadata(r) {
    var meta = {};
    var count = r.uint32();
    for (var i = 0; i < count; i++) {
      var name = r.string();
      var type = r.string();
      var size = r.uint32();
      var at = r.pos;
      if (type === 'string') {
        meta[name] = '';
        for (var c = 0; c < size; c++) {
          meta[name] += String.fromCharCode(r.bytes[at + c]);
        }
      } else if (type === 'vec3i') {
        meta[name] = [r.view.getInt32(at, true), r.view.getInt32(at + 4, true),
                      r.view.getInt32(at + 8, true)];
      } else if (type === 'int64') {
        meta[name] = Number(r.view.getBigInt64(at, true));
      } else if (type === 'int32') {
        meta[name] = r.view.getInt32(at, true);
      } else if (type === 'bool') {
        meta[name] = !!r.bytes[at];
      }
      r.pos = at + size;
    }
    return meta;
  }

  /**
   * Reads a transform and returns its uniform voxel size.
   * @param {Reader} r the cursor
   * @return {number[]} voxel size per axis
   */
  function readTransform(r) {
    var type = r.string();
    var doubles = MAP_DOUBLE_COUNTS[type];
    if (doubles === undefined) {
      throw new Error('vdb: unsupported transform map "' + type + '"');
    }
    var at = r.pos;
    r.skip(doubles * 8);

    if (type === 'TranslationMap') {
      return [1, 1, 1];
    }
    if (type === 'AffineMap') {
      // Column lengths of the upper 3x3 give the scale.
      var m = [];
      for (var i = 0; i < 16; i++) {
        m.push(r.view.getFloat64(at + i * 8, true));
      }
      return [
        Math.hypot(m[0], m[1], m[2]),
        Math.hypot(m[4], m[5], m[6]),
        Math.hypot(m[8], m[9], m[10]),
      ];
    }
    // The Scale family writes translation first when it has one.
    var base = (doubles === 18) ? at + 24 : at;
    return [
      r.view.getFloat64(base, true),
      r.view.getFloat64(base + 8, true),
      r.view.getFloat64(base + 16, true),
    ];
  }

  /**
   * Decodes IEEE 754 binary16 into a lookup table, built once on first use.
   * A table costs 256 KB and turns every later conversion into one index,
   * which matters when a grid holds millions of values.
   * @return {Float32Array} value for each of the 65536 half bit patterns
   */
  function halfTable() {
    if (HALF_TO_FLOAT) {
      return HALF_TO_FLOAT;
    }
    var lut = new Float32Array(65536);
    for (var h = 0; h < 65536; h++) {
      var sign = (h & 0x8000) ? -1 : 1;
      var exponent = (h >> 10) & 0x1f;
      var fraction = h & 0x3ff;
      if (exponent === 0) {
        // Subnormal: fraction * 2^-24.
        lut[h] = sign * fraction * 5.9604644775390625e-8;
      } else if (exponent === 31) {
        lut[h] = fraction ? NaN : sign * Infinity;
      } else {
        lut[h] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
      }
    }
    HALF_TO_FLOAT = lut;
    return lut;
  }

  /**
   * Copies count values out of a DataView, converting from half when needed.
   * @param {DataView} view where to read from
   * @param {number} at byte offset of the first value
   * @param {number} count how many values
   * @param {Float32Array} dest destination, may be null to skip the copy
   * @param {boolean} fromHalf true when values are 16 bit
   */
  function copyValues(view, at, count, dest, fromHalf) {
    if (!dest) {
      return;
    }
    var i;
    if (fromHalf) {
      var lut = halfTable();
      var available = Math.floor((view.byteLength - at) / 2);
      for (i = 0; i < count && i < available; i++) {
        dest[i] = lut[view.getUint16(at + i * 2, true)];
      }
    } else {
      for (i = 0; i < count; i++) {
        dest[i] = view.getFloat32(at + i * 4, true);
      }
    }
  }

  /**
   * Reads one io::readCompressedValues block.
   *
   * With active-mask compression only the values whose value-mask bit is set
   * are stored, so the count on the wire is the mask's population count, not
   * the node's value count.
   *
   * @param {Reader} r the cursor
   * @param {number} destCount values the node holds
   * @param {number} maskAt file offset of the node's value mask
   * @param {number} compression the grid's compression flags
   * @param {Float32Array} dest destination for the active values, in mask
   *     order, or null to step over the block without decoding it
   * @param {boolean} fromHalf true when the grid was saved as half float
   * @return {number} number of active values, ie. values written to dest
   */
  function readCompressedValues(r, destCount, maskAt, compression, dest,
      fromHalf) {
    var maskBytes = destCount >> 3;
    var valueBytes = fromHalf ? 2 : 4;
    var metadata = r.uint8();

    // Inactive values, which a fog volume never needs but must be stepped
    // over. These stay full width even in a half grid: OpenVDB reads them with
    // sizeof(ValueType), outside the HalfReader path that narrows the buffer.
    if (metadata === NO_MASK_AND_ONE_INACTIVE_VAL ||
        metadata === MASK_AND_ONE_INACTIVE_VAL ||
        metadata === MASK_AND_TWO_INACTIVE_VALS) {
      r.skip(4);
      if (metadata === MASK_AND_TWO_INACTIVE_VALS) {
        r.skip(4);
      }
    }

    // The selection mask picks between the two inactive values.
    if (metadata === MASK_AND_NO_INACTIVE_VALS ||
        metadata === MASK_AND_ONE_INACTIVE_VAL ||
        metadata === MASK_AND_TWO_INACTIVE_VALS) {
      r.skip(maskBytes);
    }

    var count = destCount;
    if ((compression & COMPRESS_ACTIVE_MASK) && metadata !== NO_MASK_AND_ALL_VALS) {
      count = countOn(r.bytes, maskAt, maskBytes);
    }

    // HalfReader bails out on an empty buffer before reading anything, while
    // the full width path still reads its compressed size header. Internal
    // nodes with no active tiles are the common case, so getting this wrong
    // desynchronises a half grid almost immediately.
    if (fromHalf && count < 1) {
      return 0;
    }

    if (!(compression & (COMPRESS_BLOSC | COMPRESS_ZIP))) {
      copyValues(r.view, r.pos, count, dest, fromHalf);
      r.skip(count * valueBytes);
      return count;
    }

    var numCompressedBytes = r.int64();
    if (numCompressedBytes <= 0) {
      // Stored uncompressed. The count is only a flag, never a length: read
      // exactly count * sizeof(value) bytes.
      copyValues(r.view, r.pos, count, dest, fromHalf);
      r.skip(count * valueBytes);
      return count;
    }

    if (dest) {
      var out = blosc.decompress(r.bytes, r.pos);
      // OpenVDB pads inputs under 128 bytes before compressing, so the
      // decompressed buffer can be longer than needed.
      copyValues(new DataView(out.buffer, out.byteOffset, out.byteLength), 0,
          count, dest, fromHalf);
    }
    r.skip(numCompressedBytes);
    return count;
  }

  /**
   * Walks the tree topology, collecting every leaf's origin and value mask
   * offset. Internal node tile values are stepped over, not decoded: this grid
   * is a fog volume whose tiles are all background.
   *
   * @param {Reader} r cursor positioned at the tree topology
   * @param {number} compression the grid's compression flags
   * @param {boolean} fromHalf true when the grid was saved as half float
   * @return {Object} leaves, activeVoxelCount, bboxMin, bboxMax
   */
  function readTopology(r, compression, fromHalf) {
    var bufferCount = r.uint32();
    if (bufferCount !== 1) {
      throw new Error('vdb: multi-buffer trees are not supported, got ' + bufferCount);
    }

    r.float32();                       // background
    var numTiles = r.uint32();
    var numChildren = r.uint32();

    for (var t = 0; t < numTiles; t++) {
      r.skip(12 + 4 + 1);              // origin, value, active flag
    }

    // Four ints per leaf: origin x, y, z and the value mask's file offset.
    var leaves = [];
    // Five numbers per active tile: origin x, y, z, span and constant value.
    var tiles = [];
    var activeVoxelCount = 0;
    var bboxMin = [Infinity, Infinity, Infinity];
    var bboxMax = [-Infinity, -Infinity, -Infinity];

    // Scratch for tile values, one per internal node level so a parent's
    // buffer is never clobbered by a child's.
    var tileScratch = {5: new Float32Array(32768), 4: new Float32Array(4096)};

    function extendBounds(x0, y0, z0, x1, y1, z1) {
      if (x0 < bboxMin[0]) bboxMin[0] = x0;
      if (y0 < bboxMin[1]) bboxMin[1] = y0;
      if (z0 < bboxMin[2]) bboxMin[2] = z0;
      if (x1 > bboxMax[0]) bboxMax[0] = x1;
      if (y1 > bboxMax[1]) bboxMax[1] = y1;
      if (z1 > bboxMax[2]) bboxMax[2] = z1;
    }

    /**
     * @param {number[]} origin index space origin of this node
     * @param {number} log2Dim children per axis, as a power of two
     * @param {number} childLog2 remaining levels below this one
     */
    function readInternalNode(origin, log2Dim, childLog2) {
      var dim = 1 << log2Dim;
      var numValues = dim * dim * dim;
      var childSpan = 1 << childLog2;
      var i;

      var childMaskAt = r.maskOffset(numValues);
      var valueMaskAt = r.maskOffset(numValues);

      // A set value-mask bit on an entry with no child is an active tile: a
      // whole childSpan^3 block of constant value. OpenVDB clears the value
      // mask bit whenever it stores a child, so the two never overlap.
      var values = tileScratch[log2Dim];
      var count = readCompressedValues(r, numValues, valueMaskAt, compression,
          values, fromHalf);
      var indexedDirectly = (count === numValues);

      var next = 0;
      for (i = 0; i < numValues; i++) {
        if (!(r.bytes[valueMaskAt + (i >> 3)] & (1 << (i & 7)))) {
          continue;
        }
        var value = indexedDirectly ? values[i] : values[next++];
        var tx = origin[0] + ((i >> (2 * log2Dim)) & (dim - 1)) * childSpan;
        var ty = origin[1] + ((i >> log2Dim) & (dim - 1)) * childSpan;
        var tz = origin[2] + (i & (dim - 1)) * childSpan;
        activeVoxelCount += childSpan * childSpan * childSpan;
        extendBounds(tx, ty, tz,
            tx + childSpan - 1, ty + childSpan - 1, tz + childSpan - 1);
        tiles.push(tx, ty, tz, childSpan, value);
      }

      for (i = 0; i < numValues; i++) {
        if (!(r.bytes[childMaskAt + (i >> 3)] & (1 << (i & 7)))) {
          continue;
        }
        // OpenVDB packs offsets x-major, z-minor.
        var childOrigin = [
          origin[0] + ((i >> (2 * log2Dim)) & (dim - 1)) * childSpan,
          origin[1] + ((i >> log2Dim) & (dim - 1)) * childSpan,
          origin[2] + (i & (dim - 1)) * childSpan,
        ];
        if (childLog2 === 3) {
          readLeafTopology(childOrigin);
        } else {
          readInternalNode(childOrigin, 4, 3);
        }
      }
    }

    function readLeafTopology(origin) {
      var maskAt = r.maskOffset(512);
      var on = countOn(r.bytes, maskAt, 64);
      activeVoxelCount += on;

      leaves.push(origin[0], origin[1], origin[2], maskAt);

      // A leaf's active voxels can sit anywhere inside its 8^3 span, so take
      // the exact bounds from the mask rather than the node origin.
      for (var i = 0; i < 512; i++) {
        if (!(r.bytes[maskAt + (i >> 3)] & (1 << (i & 7)))) {
          continue;
        }
        var x = origin[0] + ((i >> 6) & 7);
        var y = origin[1] + ((i >> 3) & 7);
        var z = origin[2] + (i & 7);
        extendBounds(x, y, z, x, y, z);
      }
    }

    for (var c = 0; c < numChildren; c++) {
      readInternalNode(r.vec3i(), 5, 7);
    }

    return {
      leaves: leaves,
      tiles: tiles,
      activeVoxelCount: activeVoxelCount,
      bboxMin: bboxMin,
      bboxMax: bboxMax,
    };
  }

  /**
   * Picks which grid to render from a file that may hold several.
   *
   * Vector grids like a velocity field cannot drive a density texture, so only
   * scalar float trees are eligible. Among those, a grid actually called
   * "density" wins, since a simulation cache usually also carries temperature
   * or fuel on the same topology.
   *
   * @param {Object[]} descriptors every grid in the file
   * @param {string} [wanted] an explicit grid name to use instead
   * @return {Object} the chosen descriptor, or null when none is usable
   */
  function chooseGrid(descriptors, wanted) {
    var eligible = descriptors.filter(function(d) {
      // "Tree_float_5_4_3" and "Tree_float_5_4_3_HalfFloat" qualify;
      // "Tree_vec3s_5_4_3" and "Tree_double_5_4_3" do not.
      return /^Tree_float_/.test(d.type);
    });
    if (eligible.length === 0) {
      return null;
    }
    var i;
    if (wanted) {
      for (i = 0; i < eligible.length; i++) {
        if (eligible[i].name === wanted) {
          return eligible[i];
        }
      }
      throw new Error('vdb: no float grid named "' + wanted + '", found ' +
          eligible.map(function(d) { return '"' + d.name + '"'; }).join(', '));
    }
    for (i = 0; i < eligible.length; i++) {
      if (eligible[i].name.toLowerCase() === 'density') {
        return eligible[i];
      }
    }
    return eligible[0];
  }

  /**
   * Number of source voxels falling into each target cell along one axis.
   * @param {number} srcDim source voxels on this axis
   * @param {number} dstDim target cells on this axis
   * @return {Uint16Array} per cell source voxel count
   */
  function axisCounts(srcDim, dstDim) {
    var counts = new Uint16Array(dstDim);
    var prev = 0;
    for (var c = 0; c < dstDim; c++) {
      var next = Math.ceil((c + 1) * srcDim / dstDim);
      // Never zero: it would divide a cell's sum by nothing and produce NaN,
      // which turns into a silent hole once stored as Uint8. With dstDim
      // clamped to at most srcDim this cannot trigger, but a zero divisor is
      // not something to leave to an invariant holding elsewhere.
      counts[c] = Math.max(1, next - prev);
      prev = next;
    }
    return counts;
  }

  /**
   * Reads an OpenVDB file into a dense downsampled density grid.
   * @param {ArrayBuffer} arrayBuffer the whole file
   * @param {Object} [options] maxResolution, default 256
   * @return {Object} the decoded grid
   * @memberOf module:vdbLoader
   */
  function load(arrayBuffer, options) {
    options = options || {};
    var maxResolution = options.maxResolution || 256;

    var r = new Reader(arrayBuffer);

    r.skip(8);                         // magic
    var version = r.uint32();
    r.uint32();                        // library major
    r.uint32();                        // library minor
    r.uint8();                         // hasGridOffsets
    if (version < 222) {
      throw new Error('vdb: file format version ' + version +
          ' is too old, 222 or newer is required');
    }
    r.skip(36);                        // uuid
    readMetadata(r);                   // file metadata

    var gridCount = r.int32();
    if (gridCount < 1) {
      throw new Error('vdb: the file contains no grids');
    }

    // Grid descriptors are not contiguous: each is immediately followed by its
    // own data, so the next one starts at the previous grid's endPos. Walk
    // them all and pick the one to render, because a simulation cache often
    // carries temperature and velocity grids alongside the density.
    var descriptors = [];
    var at = r.pos;
    for (var g = 0; g < gridCount; g++) {
      r.pos = at;
      var entry = {
        name: r.string(),
        type: r.string(),
      };
      r.string();                      // instance parent
      entry.gridPos = r.uint64();
      r.uint64();                      // blockPos, reached by reading in order
      entry.endPos = r.uint64();
      // Names are made unique with a trailing separator when they collide.
      var sep = entry.name.indexOf('\x1e');
      if (sep >= 0) {
        entry.name = entry.name.substring(0, sep);
      }
      descriptors.push(entry);
      at = entry.endPos;
    }

    var chosen = chooseGrid(descriptors, options.gridName);
    if (!chosen) {
      throw new Error('vdb: no scalar float grid found, the file has ' +
          descriptors.map(function(d) {
            return '"' + d.name + '" (' + d.type + ')';
          }).join(', '));
    }
    var name = chosen.name;

    r.pos = chosen.gridPos;
    var compression = r.uint32();
    var meta = readMetadata(r);
    // Half float grids store their leaf and tile buffers as 16 bit values.
    var fromHalf = !!meta.is_saved_as_half_float ||
        chosen.type.indexOf('HalfFloat') >= 0;
    var voxelSize = readTransform(r);

    var topology = readTopology(r, compression, fromHalf);
    var leaves = topology.leaves;
    var bboxMin = topology.bboxMin;
    var bboxMax = topology.bboxMax;

    var srcDim = [
      bboxMax[0] - bboxMin[0] + 1,
      bboxMax[1] - bboxMin[1] + 1,
      bboxMax[2] - bboxMin[2] + 1,
    ];
    var longest = Math.max(srcDim[0], srcDim[1], srcDim[2]);
    // maxResolution is a cap, never a target. Upsampling here would be worse
    // than useless: the accumulation scatters each source voxel into exactly
    // one target cell, so a target grid larger than the source leaves regular
    // planes of cells with no source voxel at all, which reads as a lattice of
    // holes through the volume. Clamping to 1 keeps small grids at 1:1.
    var factor = Math.max(1, longest / maxResolution);
    var dims = [
      Math.max(1, Math.round(srcDim[0] / factor)),
      Math.max(1, Math.round(srcDim[1] / factor)),
      Math.max(1, Math.round(srcDim[2] / factor)),
    ];

    var cellsX = axisCounts(srcDim[0], dims[0]);
    var cellsY = axisCounts(srcDim[1], dims[1]);
    var cellsZ = axisCounts(srcDim[2], dims[2]);

    var sums = new Float32Array(dims[0] * dims[1] * dims[2]);
    var values = new Float32Array(512);
    var strideY = dims[0];
    var strideZ = dims[0] * dims[1];

    for (var l = 0; l < leaves.length; l += 4) {
      var ox = leaves[l], oy = leaves[l + 1], oz = leaves[l + 2];
      // LeafNode::readBuffers writes the value mask a second time, ahead of
      // the values. It repeats the one already read during the topology walk,
      // but reading the local copy keeps this pass self contained.
      var maskAt = r.maskOffset(512);
      var n = readCompressedValues(r, 512, maskAt, compression, values,
          fromHalf);
      // When every value was stored the buffer is indexed by voxel; when only
      // the active ones were, it is indexed by set-bit order.
      var direct = (n === 512);

      var next = 0;
      for (var i = 0; i < 512; i++) {
        if (!(r.bytes[maskAt + (i >> 3)] & (1 << (i & 7)))) {
          continue;
        }
        var d = direct ? values[i] : values[next++];
        if (d === 0) {
          continue;
        }
        var cx = Math.floor((ox + ((i >> 6) & 7) - bboxMin[0]) * dims[0] / srcDim[0]);
        var cy = Math.floor((oy + ((i >> 3) & 7) - bboxMin[1]) * dims[1] / srcDim[1]);
        var cz = Math.floor((oz + (i & 7) - bboxMin[2]) * dims[2] / srcDim[2]);
        sums[cx + cy * strideY + cz * strideZ] += d;
      }
    }

    // The buffer walk must land exactly on the grid's recorded end. Any
    // mismatch means a value width or block header was misread, which would
    // otherwise show up only as a plausible looking but wrong cloud.
    if (r.pos !== chosen.endPos) {
      throw new Error('vdb: leaf buffers ended at ' + r.pos + ' but the grid ' +
          'descriptor says ' + chosen.endPos + ' (off by ' +
          (r.pos - chosen.endPos) + '); the ' +
          (fromHalf ? 'half float' : 'full width') + ' value layout is wrong');
    }

    // Active tiles are constant valued blocks with no leaf of their own.
    var tiles = topology.tiles;
    for (var tI = 0; tI < tiles.length; tI += 5) {
      var tv = tiles[tI + 4];
      if (tv === 0) {
        continue;
      }
      var span = tiles[tI + 3];
      for (var tz = 0; tz < span; tz++) {
        var tcz = Math.floor((tiles[tI + 2] + tz - bboxMin[2]) * dims[2] / srcDim[2]);
        for (var ty = 0; ty < span; ty++) {
          var tcy = Math.floor((tiles[tI + 1] + ty - bboxMin[1]) * dims[1] / srcDim[1]);
          for (var tx = 0; tx < span; tx++) {
            var tcx = Math.floor((tiles[tI] + tx - bboxMin[0]) * dims[0] / srcDim[0]);
            sums[tcx + tcy * strideY + tcz * strideZ] += tv;
          }
        }
      }
    }

    // Divide by the number of source voxels the cell covers, counting the
    // inactive ones, so this is a true box filter and total density is
    // conserved rather than the shell being over-thickened.
    var maxDensity = 0;
    var idx = 0;
    for (var z = 0; z < dims[2]; z++) {
      for (var y = 0; y < dims[1]; y++) {
        var rowDiv = cellsY[y] * cellsZ[z];
        for (var x = 0; x < dims[0]; x++) {
          var value = sums[idx] / (cellsX[x] * rowDiv);
          sums[idx++] = value;
          if (value > maxDensity) {
            maxDensity = value;
          }
        }
      }
    }

    var data = new Uint8Array(sums.length);
    if (maxDensity > 0) {
      var scale = 255 / maxDensity;
      for (var k = 0; k < sums.length; k++) {
        data[k] = Math.round(sums[k] * scale);
      }
    }

    return {
      name: name,
      dims: dims,
      data: data,
      maxDensity: maxDensity,
      worldSize: [srcDim[0] * voxelSize[0], srcDim[1] * voxelSize[1],
                  srcDim[2] * voxelSize[2]],
      bboxMin: bboxMin,
      bboxMax: bboxMax,
      activeVoxelCount: topology.activeVoxelCount,
      leafCount: leaves.length / 4,
    };
  }

  return {
    load: load,
  };

}));
