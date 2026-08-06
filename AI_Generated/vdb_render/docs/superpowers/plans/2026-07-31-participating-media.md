# Participating Media Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-page WebGL2 demo that decodes `assets/bunny_cloud.vdb` in the browser and renders it as a participating medium inside a shadow-mapped room containing a teapot, with full two-way light interaction between the medium and the opaque geometry.

**Architecture:** Four passes per frame — shadow map (opaque only), opaque forward shading into an HDR target, a half-resolution volumetric ray march, and a fullscreen composite. The medium is a dense `R8` `TEXTURE_3D` downsampled from the sparse VDB tree at load time. The medium shadows opaque surfaces analytically via short transmittance marches during shading; the opaque geometry shadows the medium via the same shadow map sampled inside the volume march.

**Tech Stack:** Plain ES5-style JavaScript in UMD-ish IIFEs, WebGL2 / GLSL ES 3.00, no external libraries, no build step. Node's built-in `node:test` for the loader tests.

**Spec:** `docs/superpowers/specs/2026-07-31-participating-media-design.md`

## Global Constraints

- **No external JavaScript libraries.** Not for rendering, not for OBJ, not for VDB, not for the UI. Everything is hand-written.
- **`src/webgl-utils.js` must not be modified.** It provides `createProgramInfo`, `createBufferInfoFromArrays`, `setBuffersAndAttributes`, `setUniforms`, `drawBufferInfo`, `resizeCanvasToDisplaySize` and nothing else. Textures and framebuffers are our job.
- **Three `webgl-utils` limits found during implementation, all worked around rather than patched:** its uniform setters throw `unknown type: 0x8b5f` on `SAMPLER_3D`, so `createProgramInfo` cannot be used at all and `glHelpers.createProgram`/`setUniforms` replace it; array keys become attribute names with an `a_` prefix, so shaders declare `a_position`/`a_normal`/`a_texcoord`; and `drawBufferInfo` hardcodes `UNSIGNED_SHORT`, so the renderer calls `drawElements` itself.
- **`src/m4.js` is expected to need no changes.** If something is genuinely missing, add it at the bottom in the file's existing style and register it in the export table at the end. Do not restructure it.
- **`src/camera.js` is expected to need no changes.** It already implements the required UE-style controls exactly.
- **Coordinate system:** right handed, +x right, +y up, +z toward the viewer, camera looks down −z.
- **Every new module** follows the UMD-ish IIFE pattern in `src/camera.js`, and additionally supports `module.exports` so Node tests can `require` it. Two-space indent, `"use strict"`, JSDoc on public functions, `var` not `let`/`const`, matching the existing files.
- **GLSL version:** `#version 300 es` on line 1 of every shader, `precision highp float;` and `precision highp sampler3D;` in fragment shaders.
- **This is not a git repository.** Either run `git init` before starting and keep the commit steps, or skip every commit step. Do not `git init` without asking.
- **Serving:** the page must be served over HTTP (`python -m http.server 8000`), never opened as `file://`.

### Verified asset facts — do not re-derive, do not assume otherwise

| Fact | Value |
|---|---|
| VDB file version | 223, library 12.1 |
| VDB grid | one grid, name `density`, type `Tree_float_5_4_3` |
| VDB compression flags | `6` = `COMPRESS_ACTIVE_MASK(2) \| COMPRESS_BLOSC(4)` |
| VDB half float | **no** — 4-byte floats |
| VDB active bbox | `[-300,-47,-208] .. [276,524,229]` (577 × 572 × 438) |
| VDB active voxel count | `19210271` |
| VDB `gridPos` / `blockPos` / `endPos` | `174` / `4574480` / `76328146` |
| OBJ | 7850 `v`, 8334 `vt`, 8028 faces — **7676 quads and 352 triangles**, `v/vt` only, **no `vn`** → 15704 triangles, 47112 indices |
| OBJ bbox | `(-71.893, 0, -47.928) .. (82.294, 79.007, 47.928)` |
| `default.png` | 128 × 128 PNG, decoded natively by the browser |

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/blosc.js` | Blosc 1.x container + LZ4 block decode + byte unshuffle | 1 |
| `src/vdb-loader.js` | OpenVDB reader → downsampled dense density grid | 2 |
| `src/obj-loader.js` | OBJ parse, triangulate, generate normals | 3 |
| `src/gl-helpers.js` | textures, 3D textures, FBOs, fullscreen quad | 4 |
| `index.html` | canvas, UI panel markup, script tags | 4, 10 |
| `src/main.js` | bootstrap, asset load, frame loop | 4 |
| `src/scene.js` | room geometry, scene state, defaults | 5 |
| `src/shaders.js` | all GLSL sources | 5–9 |
| `src/renderer.js` | the four passes | 5–9 |
| `src/ui.js` | control panel wiring | 10 |
| `tests/blosc.test.js` | Blosc decode against the real file | 1 |
| `tests/vdb-loader.test.js` | the three decode invariants | 2 |
| `tests/obj-loader.test.js` | parse/triangulate/normals | 3 |

Run all tests with `node --test "tests/*.test.js"` (the bare directory form
does not glob on Windows).

---

## Task 1: Blosc / LZ4 decoder

**Files:**
- Create: `src/blosc.js`
- Test: `tests/blosc.test.js`

**Interfaces:**
- Consumes: nothing.
- Produces: `blosc.decompress(bytes, offset) -> Uint8Array` where `bytes` is a
  `Uint8Array` and `offset` is the index of the 16-byte Blosc header. Returns
  the full decompressed buffer (`nbytes` long). Throws `Error` on an
  unsupported compressor, bitshuffle, or delta filter.

**Background the implementer needs:**

A Blosc 1.x buffer is a 16-byte header, then an `int32` offset table, then blocks.

| Offset | Type | Meaning |
|---|---|---|
| 0 | `uint8` | version (expect 1 or 2) |
| 1 | `uint8` | versionlz |
| 2 | `uint8` | flags |
| 3 | `uint8` | typesize |
| 4 | `uint32` | `nbytes` — uncompressed size |
| 8 | `uint32` | `blocksize` |
| 12 | `uint32` | `cbytes` — total compressed size |

Flags: bit 0 (`0x01`) byte shuffle, bit 1 (`0x02`) whole buffer is memcpy-ed,
bit 2 (`0x04`) bitshuffle, bit 3 (`0x08`) delta. Bits 5–7 (`flags >> 5`) are
the compressor id: 0 blosclz, 1 lz4, 2 lz4hc, 3 snappy, 4 zlib, 5 zstd.

OpenVDB always uses **lz4 with byte shuffle**, so only compressor id 1 and 2
(both decode with the same LZ4 block format) need to work. Anything else
throws.

If `BLOSC_MEMCPYED` is set, the payload is a raw copy starting at
`offset + 16` and there is no offset table.

Otherwise `nblocks = ceil(nbytes / blocksize)` and the offset table is
`nblocks` `int32`s at `offset + 16`, each an absolute offset from `offset`.
Block `i` covers output bytes `[i*blocksize, min((i+1)*blocksize, nbytes))`;
call that length `bsize`, and `leftoverblock` is true when `bsize != blocksize`.

**Block splitting is recomputed by the decompressor, not stored in the file.**
This is the single most error-prone detail in the task. c-blosc 1.x uses:

```
nstreams = (typesize <= 16 && (blocksize / typesize) >= 128 && !leftoverblock)
           ? typesize : 1
neblock  = bsize / nstreams
```

Then for each of the `nstreams` sub-streams: read an `int32` compressed length
`cb`; if `cb == neblock` the sub-stream is stored raw, so copy `neblock` bytes;
otherwise LZ4-decompress `cb` bytes into `neblock` bytes. Advance by `cb`.

For OpenVDB's typical 2048-byte leaf buffer with `typesize = 4` and
`blocksize = 2048`, this gives `nstreams = 4` — the split path is the normal
path here, not an edge case.

Byte shuffle is **per block**. For a block of `bsize` bytes with `ts = typesize`
and `n = floor(bsize / ts)` elements, the shuffled layout is all byte-0s, then
all byte-1s, and so on. Unshuffle is therefore
`dst[i*ts + j] = src[j*n + i]` for `i` in `[0,n)`, `j` in `[0,ts)`. Any trailing
`bsize - n*ts` bytes are copied straight across.

- [ ] **Step 1: Write the failing test**

> **Revised during execution.** The original test decoded the first leaf buffer
> of the real VDB file and asserted it was Blosc compressed. It is not — that
> leaf has `numCompressedBytes = 0`, the uncompressed marker. The test was
> asserting an assumption about the file rather than testing the decoder.
> `tests/blosc.test.js` now builds Blosc containers by hand, covering the LZ4
> literal/overlapping-match path, the byte unshuffle, the memcpy-ed path, a
> non-zero start offset, and both error paths — six deterministic tests with no
> file dependency. End-to-end validation against all ~300k real leaf buffers is
> Task 2's invariants, which is stronger evidence than any single leaf.
> See the file for the current tests; the block below is the original.

Create `tests/blosc.test.js`:

```js
'use strict';
var test = require('node:test');
var assert = require('node:assert');
var fs = require('node:fs');
var path = require('node:path');
var blosc = require('../src/blosc.js');

var VDB = path.join(__dirname, '..', 'assets', 'bunny_cloud.vdb');
var BLOCK_POS = 4574480;

test('decodes the first leaf buffer of bunny_cloud.vdb', function() {
  var buf = fs.readFileSync(VDB);
  var bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  var p = BLOCK_POS;
  var metadata = bytes[p]; p += 1;
  assert.ok(metadata === 0 || metadata === 6,
      'expected NO_MASK_OR_INACTIVE_VALS or NO_MASK_AND_ALL_VALS, got ' + metadata);

  var numCompressedBytes = Number(view.getBigInt64(p, true)); p += 8;
  assert.ok(numCompressedBytes > 0, 'first leaf should be blosc compressed');

  var out = blosc.decompress(bytes, p);

  // A leaf is 8^3 = 512 float values. Active-mask compression stores only the
  // active ones, and OpenVDB pads inputs under 128 bytes, so the decompressed
  // size is a multiple of 4 and at most 2048.
  assert.strictEqual(out.length % 4, 0);
  assert.ok(out.length <= 2048, 'got ' + out.length);

  var floats = new Float32Array(out.buffer, out.byteOffset, out.length / 4);
  for (var i = 0; i < floats.length; i++) {
    assert.ok(Number.isFinite(floats[i]), 'value ' + i + ' is not finite');
    assert.ok(floats[i] >= 0 && floats[i] <= 1,
        'fog density out of range at ' + i + ': ' + floats[i]);
  }
});

test('rejects an unsupported compressor', function() {
  var header = new Uint8Array(16);
  header[0] = 2;
  header[2] = 4 << 5;      // compressor id 4 = zlib
  header[3] = 4;
  new DataView(header.buffer).setUint32(4, 64, true);
  assert.throws(function() { blosc.decompress(header, 0); }, /compressor/i);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test tests/blosc.test.js`
Expected: FAIL — `Cannot find module '../src/blosc.js'`.

- [ ] **Step 3: Implement `src/blosc.js`**

```js
/*
 * Minimal Blosc 1.x decompressor, enough for the buffers OpenVDB writes.
 *
 * OpenVDB always compresses with LZ4 and byte shuffle at typesize =
 * sizeof(ValueType), so that is the only path implemented. Anything else
 * throws rather than quietly producing garbage.
 *
 * @module blosc
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.blosc = factory();
  }
}(this, function() {
  "use strict";

  var FLAG_SHUFFLE = 0x01;
  var FLAG_MEMCPYED = 0x02;
  var FLAG_BITSHUFFLE = 0x04;
  var FLAG_DELTA = 0x08;

  var COMPRESSOR_NAMES = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd'];

  // c-blosc 1.x recomputes the split on read; these are its constants.
  var MAX_SPLITS = 16;
  var MIN_BUFFERSIZE = 128;

  /**
   * Decompresses one LZ4 block.
   * @param {Uint8Array} src source bytes
   * @param {number} sp first byte of the block
   * @param {number} srcEnd one past the last byte of the block
   * @param {Uint8Array} dst destination bytes
   * @param {number} dp first byte to write
   * @return {number} number of bytes written
   */
  function lz4Block(src, sp, srcEnd, dst, dp) {
    var start = dp;
    var token, litLen, matchLen, offset, b, i, ref;

    while (sp < srcEnd) {
      token = src[sp++];

      litLen = token >> 4;
      if (litLen === 15) {
        do { b = src[sp++]; litLen += b; } while (b === 255);
      }
      for (i = 0; i < litLen; i++) {
        dst[dp++] = src[sp++];
      }

      // The final sequence is literals only, with no match to follow.
      if (sp >= srcEnd) {
        break;
      }

      offset = src[sp] | (src[sp + 1] << 8);
      sp += 2;
      if (offset === 0) {
        throw new Error('blosc: corrupt lz4 stream, zero match offset');
      }

      matchLen = token & 0x0f;
      if (matchLen === 15) {
        do { b = src[sp++]; matchLen += b; } while (b === 255);
      }
      matchLen += 4;

      // Matches may overlap the bytes being written, so copy one at a time.
      ref = dp - offset;
      for (i = 0; i < matchLen; i++) {
        dst[dp++] = dst[ref++];
      }
    }

    return dp - start;
  }

  /**
   * Reverses Blosc's byte shuffle over one block.
   * @param {Uint8Array} src shuffled bytes
   * @param {Uint8Array} dst destination, may not alias src
   * @param {number} dp first byte to write
   * @param {number} bsize block length in bytes
   * @param {number} typesize element size in bytes
   */
  function unshuffle(src, dst, dp, bsize, typesize) {
    var n = Math.floor(bsize / typesize);
    var i, j, s;
    for (j = 0; j < typesize; j++) {
      s = j * n;
      for (i = 0; i < n; i++) {
        dst[dp + i * typesize + j] = src[s + i];
      }
    }
    // Whatever does not divide evenly is stored unshuffled at the end.
    for (i = n * typesize; i < bsize; i++) {
      dst[dp + i] = src[i];
    }
  }

  /**
   * Decompresses a Blosc buffer.
   * @param {Uint8Array} bytes buffer containing the blosc data
   * @param {number} offset index of the 16 byte blosc header
   * @return {Uint8Array} the decompressed bytes, nbytes long
   * @memberOf module:blosc
   */
  function decompress(bytes, offset) {
    var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    var version = bytes[offset];
    var flags = bytes[offset + 2];
    var typesize = bytes[offset + 3];
    var nbytes = view.getUint32(offset + 4, true);
    var blocksize = view.getUint32(offset + 8, true);

    if (version !== 1 && version !== 2) {
      throw new Error('blosc: unsupported format version ' + version);
    }
    if (flags & FLAG_BITSHUFFLE) {
      throw new Error('blosc: bitshuffle is not supported');
    }
    if (flags & FLAG_DELTA) {
      throw new Error('blosc: the delta filter is not supported');
    }

    var dst = new Uint8Array(nbytes);

    if (flags & FLAG_MEMCPYED) {
      dst.set(bytes.subarray(offset + 16, offset + 16 + nbytes));
      return dst;
    }

    var compressor = flags >> 5;
    if (compressor !== 1 && compressor !== 2) {
      throw new Error('blosc: unsupported compressor ' +
          (COMPRESSOR_NAMES[compressor] || compressor) + ', only lz4 is supported');
    }

    var nblocks = Math.ceil(nbytes / blocksize);
    var tableAt = offset + 16;

    // Shuffled bytes land here first, then get unshuffled into dst.
    var tmp = (flags & FLAG_SHUFFLE) ? new Uint8Array(blocksize) : null;

    for (var i = 0; i < nblocks; i++) {
      var blockAt = offset + view.getInt32(tableAt + i * 4, true);
      var bsize = Math.min(blocksize, nbytes - i * blocksize);
      var leftover = bsize !== blocksize;

      var nstreams = (typesize <= MAX_SPLITS &&
                      Math.floor(blocksize / typesize) >= MIN_BUFFERSIZE &&
                      !leftover) ? typesize : 1;
      var neblock = Math.floor(bsize / nstreams);

      var out = tmp || dst;
      var outAt = tmp ? 0 : i * blocksize;

      for (var j = 0; j < nstreams; j++) {
        var cb = view.getInt32(blockAt, true);
        blockAt += 4;
        if (cb === neblock) {
          out.set(bytes.subarray(blockAt, blockAt + neblock), outAt);
        } else {
          var written = lz4Block(bytes, blockAt, blockAt + cb, out, outAt);
          if (written !== neblock) {
            throw new Error('blosc: block ' + i + ' stream ' + j + ' produced ' +
                written + ' bytes, expected ' + neblock);
          }
        }
        blockAt += cb;
        outAt += neblock;
      }

      if (tmp) {
        unshuffle(tmp, dst, i * blocksize, bsize, typesize);
      }
    }

    return dst;
  }

  return {
    decompress: decompress,
  };

}));
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test tests/blosc.test.js`
Expected: PASS, 2 tests.

If it fails, the likely culprits in order are: the split rule (`nstreams`), the
offset table being relative to `offset` rather than absolute, and the
per-block unshuffle using `bsize` rather than `blocksize`. Debug by printing
the header fields — `nbytes` for a full leaf should be `2048`.

- [ ] **Step 5: Commit**

```bash
git add src/blosc.js tests/blosc.test.js && git commit -m "feat: blosc/lz4 decoder"
```

---

## Task 2: OpenVDB reader

**Files:**
- Create: `src/vdb-loader.js`
- Test: `tests/vdb-loader.test.js`

**Interfaces:**
- Consumes: `blosc.decompress(bytes, offset)` from Task 1.
- Produces:

```js
vdbLoader.load(arrayBuffer, options) -> {
  name: string,          // 'density'
  dims: [nx, ny, nz],    // target grid dimensions, e.g. [256, 254, 194]
  data: Uint8Array,      // nx*ny*nz density samples, normalised to 0..255
  maxDensity: number,    // multiply the sampled 0..1 back up by this
  worldSize: [x, y, z],  // extent in VDB world units
  bboxMin: [x, y, z],    // active bbox in index space
  bboxMax: [x, y, z],
  activeVoxelCount: number,
}
```

`options.maxResolution` defaults to `256` and caps the **longest** axis; the
other axes stay proportional so the grid is not distorted.

`worldSize` is informational only — no later task consumes it. Because this
grid's voxels are cubic, `dims` already carries the correct aspect ratio, and
Task 5 derives the medium's shape from `dims`.

**Background the implementer needs:**

Read order mirrors `openvdb::io::Archive`. All integers are little endian.
Strings are a `uint32` length followed by that many raw bytes.

*File header:* 8-byte magic, `uint32` version (`223`), `uint32` library major,
`uint32` library minor, `uint8` `hasGridOffsets`, then a 36-character UUID.
**Version 223 stores compression per grid, so there is no file-level
compression word** — the UUID follows `hasGridOffsets` immediately.

*Metadata map:* `uint32` count, then per entry a name string, a type string, a
`uint32` byte count, and that many bytes of payload. Skipping by the byte count
means unknown types cost nothing.

*Grid descriptors:* `int32` grid count, then per grid a unique-name string, a
type string, an instance-parent string, and three `uint64` offsets `gridPos`,
`blockPos`, `endPos`.

*Grid body at `gridPos`:* `uint32` compression flags, metadata map, transform.

*Transform:* a type string then its parameters. Handle `UniformScaleMap`
(one `float64`), `UniformScaleTranslateMap` (`float64` scale then 3 `float64`
translation), `ScaleMap` (3 `float64`), `ScaleTranslateMap` (3 + 3 `float64`),
`TranslationMap` (3 `float64`), and `AffineMap` (16 `float64`, row major,
scale is the length of each of the first three rows). Any other type: fall back
to voxel size `[1,1,1]` and carry on — the demo places the medium by UI
transform anyway, so only the aspect ratio matters.

*Tree topology:* `uint32` buffer count (expect 1), then the root node:
- background value (one `float32`)
- `uint32` number of tiles
- `uint32` number of children
- per tile: `vec3i` origin, `float32` value, `int32` active flag
- per child: `vec3i` origin, then the child node

*`InternalNode<N>` topology* (N = 5 then 4, so 32³ then 16³ children):
- child mask: `N³ / 8` bytes
- value mask: `N³ / 8` bytes
- tile values via `readCompressedValues` with `N³` entries
- then recurse into each set child-mask bit, **in bit index order**

*`LeafNode<3>` topology:* value mask only, 512 bits = 64 bytes.

*Buffers* (`Tree::readBuffers`, starting at `blockPos`): for each leaf **in the
same traversal order**, first **the 512-bit value mask again** — OpenVDB writes
it twice, once in `readTopology` and once in `readBuffers` — then
`readCompressedValues` with 512 entries. Internal nodes contribute nothing to
this section; only leaves carry data.

> **Discovered during implementation.** The repeated value mask was missing
> from the original plan. Without it the buffer walk desynchronises after the
> first few leaves, and because the file opens with a run of zero-valued
> leaves the corruption does not surface immediately.

*Active tiles:* a set value-mask bit on an internal-node entry whose child-mask
bit is clear is an **active tile** — a whole `childSpan³` block of constant
value with no leaf of its own. OpenVDB clears the value-mask bit whenever it
stores a child, so the two masks never overlap. This file has two such tiles at
the `InternalNode<4>` level, contributing exactly `2 × 8³ = 1024` voxels. Miss
them and the active voxel count lands on `19209247` instead of `19210271`.

*`readCompressedValues(count, valueMask)`* with our flags
(`COMPRESS_ACTIVE_MASK | COMPRESS_BLOSC`) and file version ≥ 222:

1. `uint8` metadata flag.
2. Depending on the flag, read inactive values:
   `NO_MASK_AND_MINUS_BG(1)` and `NO_MASK_AND_ONE_INACTIVE_VAL(2)` and
   `MASK_AND_ONE_INACTIVE_VAL(4)` read one `float32`;
   `MASK_AND_TWO_INACTIVE_VALS(5)` reads two.
   `NO_MASK_OR_INACTIVE_VALS(0)`, `MASK_AND_NO_INACTIVE_VALS(3)` and
   `NO_MASK_AND_ALL_VALS(6)` read none.
3. For flags 4 and 5, read a selection mask of `count / 8` bytes.
4. If the flag is `NO_MASK_AND_ALL_VALS(6)`, read all `count` values.
   Otherwise read only as many values as there are set bits in `valueMask`.
5. The payload is `int64` `numCompressedBytes`, then:
   - **positive** — a Blosc buffer of that many bytes. OpenVDB pads inputs
     under 128 bytes before compressing, so the decompressed buffer may be
     **longer** than needed; take the first `n` values.
   - **zero or negative** — the values are stored uncompressed, and the count
     is only a flag. Read exactly `n * 4` bytes, where `n` is the value count
     computed in step 4. **Do not read `-numCompressedBytes` bytes** — that
     number is not a length, and `0` is a perfectly normal value here
     (the very first leaf in this file has it). Getting this wrong
     desynchronises every subsequent leaf.
6. Scatter the read values back through `valueMask` into the destination,
   filling unset positions with the appropriate inactive value.

For this fog volume the inactive value is the background, `0`, and only the
active values matter — but implement the scatter properly anyway, because
getting it wrong shifts every subsequent value and the invariants will catch it
in a way that is hard to localise.

*Index-space origin of a leaf:* leaves carry their origin implicitly through the
traversal. Track it: root children give an explicit `vec3i` origin; an
`InternalNode<5>` child at bit index `i` sits at local offset
`((i >> 10) & 31, (i >> 5) & 31, i & 31)` scaled by the child's index-space
span. For `5_4_3`, an `InternalNode<5>`'s children span `16 * 8 = 128` voxels
each, an `InternalNode<4>`'s children span `8` voxels each, and a leaf's voxel
at bit index `j` is at local `((j >> 6) & 7, (j >> 3) & 7, j & 7)`.
**OpenVDB bit ordering is x-major, z-minor** — this is the second most likely
place to introduce a bug, and it shows up as a mirrored or transposed cloud.

*Downsampling:* do not build the dense 144 M voxel grid. Compute the target
dims from the active bbox and `maxResolution`, allocate a `Float32Array` of
`nx*ny*nz` sums, and for each active voxel add its value into the target cell
`floor((index - bboxMin) / factor)`. Divide by `factor³` at the end so total
density is conserved, track the maximum, and normalise into a `Uint8Array`.

- [ ] **Step 1: Write the failing test**

Create `tests/vdb-loader.test.js`. These invariants are the gate for the
whole project — do not proceed to Task 4 until they pass.

> **Corrected during implementation.** The `fog volume densities stay within
> 0..1` test below is wrong and was replaced. Measured from the file, the
> active values span `0 .. 2.7922983` with a mean of `0.3044`, and 4.5% of them
> exceed `1.0` — Houdini fog volumes are not normalised. The replacement
> asserts what actually constrains the filter: box filtering can never produce
> a value above the source maximum, and should land reasonably close to it.
> See the file for the current tests; the block below is the original.

```js
'use strict';
var test = require('node:test');
var assert = require('node:assert');
var fs = require('node:fs');
var path = require('node:path');
var vdbLoader = require('../src/vdb-loader.js');

var VDB = path.join(__dirname, '..', 'assets', 'bunny_cloud.vdb');

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

test('fog volume densities stay within 0..1', function() {
  var g = load();
  assert.ok(g.maxDensity > 0, 'grid is empty');
  assert.ok(g.maxDensity <= 1.0 + 1e-5, 'maxDensity = ' + g.maxDensity);
});

test('target grid keeps the source aspect ratio', function() {
  var g = load();
  assert.strictEqual(g.name, 'density');
  assert.strictEqual(Math.max(g.dims[0], g.dims[1], g.dims[2]), 256);
  assert.strictEqual(g.data.length, g.dims[0] * g.dims[1] * g.dims[2]);
  // 577 x 572 x 438 scaled so the longest axis is 256.
  assert.deepStrictEqual(g.dims, [256, 254, 194]);
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test tests/vdb-loader.test.js`
Expected: FAIL — `Cannot find module '../src/vdb-loader.js'`.

- [ ] **Step 3: Implement `src/vdb-loader.js`**

Structure it as a `Reader` object wrapping a `DataView` with cursor methods:

```js
function Reader(arrayBuffer) {
  this.view = new DataView(arrayBuffer);
  this.bytes = new Uint8Array(arrayBuffer);
  this.pos = 0;
}
Reader.prototype.uint8   = function() { return this.bytes[this.pos++]; };
Reader.prototype.uint32  = function() { var v = this.view.getUint32(this.pos, true); this.pos += 4; return v; };
Reader.prototype.int32   = function() { var v = this.view.getInt32(this.pos, true); this.pos += 4; return v; };
Reader.prototype.uint64  = function() { var v = Number(this.view.getBigUint64(this.pos, true)); this.pos += 8; return v; };
Reader.prototype.int64   = function() { var v = Number(this.view.getBigInt64(this.pos, true)); this.pos += 8; return v; };
Reader.prototype.float32 = function() { var v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; };
Reader.prototype.float64 = function() { var v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; };
Reader.prototype.vec3i   = function() { return [this.int32(), this.int32(), this.int32()]; };
Reader.prototype.skip    = function(n) { this.pos += n; };
Reader.prototype.mask    = function(n) {  // n bits, returned as a Uint8Array
  var m = this.bytes.subarray(this.pos, this.pos + (n >> 3));
  this.pos += n >> 3;
  return m;
};
Reader.prototype.string  = function() {
  var n = this.uint32();
  var s = '';
  for (var i = 0; i < n; i++) { s += String.fromCharCode(this.bytes[this.pos + i]); }
  this.pos += n;
  return s;
};
```

Then the read functions described above:
`readHeader`, `readMetadata`, `readGridDescriptors`, `readTransform`,
`readCompressedValues`, `readRootNode`, `readInternalNode`, `readLeafTopology`,
`readLeafBuffers`, and finally `accumulate`/`normalise` for the downsample.

Do the two-phase read the format requires: a topology pass that collects every
leaf's origin and value mask into a flat array, then a buffer pass that walks
that same array in order reading voxel data from `blockPos` onward. Keeping the
leaf list flat is what makes the two phases provably consistent.

Compute the active bbox and the active voxel count from the leaf value masks
during the topology pass, so the invariant test exercises the topology walk
independently of the buffer walk.

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test tests/vdb-loader.test.js`
Expected: PASS, 5 tests.

If the voxel count is right but the bbox is wrong, the leaf origin arithmetic
is wrong. If both are right but densities are out of range, the buffer walk has
drifted out of sync with the topology walk. If the count is off by a small
amount, root **tiles** are probably being ignored — this grid may have none,
but handle them.

- [ ] **Step 5: Print the density statistics for later tuning**

Run a one-off script that prints `maxDensity`, the mean of the non-zero cells,
and the estimated optical depth through the middle of the cloud. Record the
value of the medium density multiplier that gives an optical depth of roughly
4–6; that becomes the UI default in Task 10.

- [ ] **Step 6: Commit**

```bash
git add src/vdb-loader.js tests/vdb-loader.test.js && git commit -m "feat: openvdb reader"
```

---

## Task 3: OBJ loader

**Files:**
- Create: `src/obj-loader.js`
- Test: `tests/obj-loader.test.js`

**Interfaces:**
- Consumes: nothing.
- Produces: `objLoader.parse(text) -> {position, normal, texcoord, indices, bbox}`
  where `position`/`normal` are `Float32Array` with 3 components per vertex,
  `texcoord` is a `Float32Array` with 2, `indices` is a `Uint32Array`, and
  `bbox` is `{min: [x,y,z], max: [x,y,z]}` **after** recentring. The three
  array names match what `webglUtils.createBufferInfoFromArrays` expects.

**Requirements:**

- Handle `v`, `vt`, `vn`, `f`; ignore every other line including `#`, `g`,
  `usemtl`, `mtllib`, and `s`.
- Faces may have any vertex count ≥ 3; triangulate as a fan
  (`0,i,i+1` for `i` in `1..n-2`). The teapot is **mixed** — 7676 quads and
  352 triangles — so its 8028 faces become 15704 triangles, not 16056.
  Do not assume a uniform face size.
- Accept index forms `v`, `v/vt`, `v//vn`, and `v/vt/vn`. Resolve negative
  indices relative to the current count. OBJ indices are 1-based.
- Deduplicate vertices on the `v/vt/vn` triple string.
- **If the file has no normals** — which is the case here — accumulate
  face normals into each of the face's vertices, then normalise. Do not
  normalise the face normal before accumulating: the un-normalised cross
  product is already area weighted, which is what makes the teapot's curved
  surfaces smooth.
- Recentre positions on X and Z about the bbox centre and move the bbox
  minimum Y to 0, so UI position `(0,0,0)` stands the model on the floor at
  the origin.
- Missing texcoords default to `(0,0)`.

- [ ] **Step 1: Write the failing test**

```js
'use strict';
var test = require('node:test');
var assert = require('node:assert');
var fs = require('node:fs');
var path = require('node:path');
var objLoader = require('../src/obj-loader.js');

var OBJ = path.join(__dirname, '..', 'assets', 'teapot', 'teapot.obj');

test('triangulates the teapot quads', function() {
  var mesh = objLoader.parse(fs.readFileSync(OBJ, 'utf8'));
  // 8028 quads -> 2 triangles each.
  assert.strictEqual(mesh.indices.length, 8028 * 2 * 3);
});

test('generates unit length normals when the file has none', function() {
  var mesh = objLoader.parse(fs.readFileSync(OBJ, 'utf8'));
  assert.strictEqual(mesh.normal.length, mesh.position.length);
  for (var i = 0; i < mesh.normal.length; i += 3) {
    var x = mesh.normal[i], y = mesh.normal[i + 1], z = mesh.normal[i + 2];
    var len = Math.sqrt(x * x + y * y + z * z);
    assert.ok(Math.abs(len - 1) < 1e-4, 'normal ' + (i / 3) + ' length ' + len);
  }
});

test('recentres on x/z and rests on y = 0', function() {
  var mesh = objLoader.parse(fs.readFileSync(OBJ, 'utf8'));
  // Source bbox is (-71.893, 0, -47.928)..(82.294, 79.007, 47.928).
  assert.ok(Math.abs(mesh.bbox.min[0] + mesh.bbox.max[0]) < 1e-3, 'x not centred');
  assert.ok(Math.abs(mesh.bbox.min[2] + mesh.bbox.max[2]) < 1e-3, 'z not centred');
  assert.ok(Math.abs(mesh.bbox.min[1]) < 1e-3, 'does not rest on y=0');
  assert.ok(Math.abs((mesh.bbox.max[0] - mesh.bbox.min[0]) - 154.187) < 1e-2);
  assert.ok(Math.abs((mesh.bbox.max[1] - mesh.bbox.min[1]) - 79.007) < 1e-2);
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test tests/obj-loader.test.js`
Expected: FAIL — `Cannot find module '../src/obj-loader.js'`.

- [ ] **Step 3: Implement `src/obj-loader.js`**

Split the text on `/\r?\n/`, then split each line on `/\s+/` after trimming.
Keep raw `v`/`vt`/`vn` in plain arrays, build the deduplicated vertex list in a
map keyed by the face-vertex token, then generate normals if `vn` was absent,
then recentre.

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test tests/obj-loader.test.js`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/obj-loader.js tests/obj-loader.test.js && git commit -m "feat: obj loader"
```

---

## Task 4: WebGL2 bootstrap and GL helpers

**Files:**
- Create: `src/gl-helpers.js`, `index.html`, `src/main.js`
- Test: manual, in the browser

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:

```js
glHelpers.createTexture2D(gl, opts)     // {width, height, internalFormat, format,
                                        //  type, min, mag, wrap, data} -> WebGLTexture
glHelpers.createTexture3D(gl, opts)     // {width, height, depth, internalFormat,
                                        //  format, type, data} -> WebGLTexture
glHelpers.createFramebuffer(gl, opts)   // {colorAttachments: [tex], depthAttachment: tex}
                                        //  -> {framebuffer, width, height}
glHelpers.createDepthTexture(gl, w, h)  // DEPTH_COMPONENT24 -> WebGLTexture
glHelpers.createFullscreenQuad(gl)      // -> bufferInfo for webglUtils
glHelpers.bindFramebuffer(gl, fbo)      // fbo or null, sets the viewport too
```

`createFullscreenQuad` returns a `bufferInfo` for a single triangle covering
clip space (`position` at `(-1,-1)`, `(3,-1)`, `(-1,3)`) — one triangle, not two,
so there is no diagonal seam and no wasted quad-helper invocations.

**Requirements:**

- `index.html` gets a full-window `<canvas id="canvas">`, a `<div id="ui">`
  panel (empty for now, filled in Task 10), a `<div id="loading">` overlay, and
  script tags in dependency order: `m4.js`, `webgl-utils.js`, `camera.js`,
  `blosc.js`, `vdb-loader.js`, `obj-loader.js`, `gl-helpers.js`, `shaders.js`,
  `scene.js`, `renderer.js`, `ui.js`, `main.js`.
- `main.js` gets the `webgl2` context, checks it, requests
  `EXT_color_buffer_float` and checks it, and on failure replaces the canvas
  with a readable message naming exactly what is missing.
- `main.js` fetches all three assets in parallel with `Promise.all`, showing the
  loading overlay. A failed fetch shows the URL, the status, and the reminder
  that the page must be served over HTTP rather than opened as `file://`.
  The VDB is a 76 MB fetch, so show download progress if it is cheap to do so,
  otherwise a plain "decoding VDB…" message.
- `main.js` creates the camera on the canvas, runs a `requestAnimationFrame`
  loop calling `camera.update(dt)` and `webglUtils.resizeCanvasToDisplaySize`,
  and for now just clears to a recognisable colour.
- Clamp `dt` to at most `0.1` seconds so a background tab does not teleport the
  camera on return.

- [ ] **Step 1: Write `src/gl-helpers.js`**

Follow the `camera.js` UMD pattern. Every creator restores no state it did not
set and leaves the bound texture at `null`.

- [ ] **Step 2: Write `index.html` and `src/main.js`**

- [ ] **Step 3: Serve and verify**

Run: `python -m http.server 8000` and open `http://localhost:8000/`.
Expected: the canvas fills the window in the clear colour, the console shows
the decoded VDB dims `[256, 254, 194]` and the teapot index count `47112`,
no errors, and holding right mouse + `w` logs camera position changes.

- [ ] **Step 4: Commit**

```bash
git add index.html src/gl-helpers.js src/main.js && git commit -m "feat: webgl2 bootstrap"
```

---

## Task 5: Room, teapot, and the opaque pass

**Files:**
- Create: `src/scene.js`, `src/shaders.js`, `src/renderer.js`
- Modify: `src/main.js`

**Interfaces:**
- Consumes: `objLoader.parse`, `glHelpers.*`, `webglUtils.createProgramInfo`,
  `webglUtils.createBufferInfoFromArrays`.
- Produces:

```js
scene.createState()       // -> the mutable state object read every frame
scene.buildRoom(gl)       // -> {floor: bufferInfo, walls: bufferInfo}
scene.updateMatrices(state)   // recomputes state.teapotMatrix, state.mediumMatrix,
                              // state.mediumInvMatrix, state.lightDirection
renderer.create(gl, resources) // -> {render(state, camera, time)}
```

`scene.createState()` returns exactly the defaults from the spec's default table:

```js
{
  light: {yaw: 30 * DEG, pitch: -50 * DEG, intensity: 3.0},
  ambient: {intensity: 0.35},
  teapot: {position: [0, 0, 0], rotation: [0, 0, 0], scale: 0.3},
  medium: {position: [0, 38, 0], rotation: [0, 0, 0], scale: 32,
           density: DENSITY_DEFAULT, color: [0.95, 0.96, 1.0]},
}
```

`DENSITY_DEFAULT = 1.0`, measured in Task 2 Step 5. At that value the optical
depth through the cloud along the grid's short axis is median `1.47`, p90
`3.10`, p99 `6.62` — dense but still translucent, inside the target band. The
slider range is `0..3`.

`state.lightDirection` is the direction light **travels**, using the same
formula `camera.js` uses for forward so the two conventions agree:

```js
dir = [-sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch)]
```

The medium's model matrix must fold in the grid aspect ratio before the UI
transform, so uniform scale does not distort it:

```js
aspect = dims / max(dims)                      // e.g. [1, 0.9922, 0.7578]
mediumMatrix = translate(position)
             * rotateY(rot.y) * rotateX(rot.x) * rotateZ(rot.z)
             * scale(uniformScale * aspect)
```

Local space for the medium is the **unit cube centred at the origin**,
`[-0.5, 0.5]³`, so the 3D texture lookup is `localPos + 0.5`.

**Requirements:**

- Floor: 100 × 100 on `y = 0`, normal `+y`, centred at the origin.
- Walls: four, inner faces at `x = ±50` and `z = ±50`, 50 tall, normals facing
  inward. Open top.
- Both carry a `vec3` world position and a `vec3` normal; the checkerboard is
  computed from world position in the fragment shader, so no UVs are needed for
  the room.
- Checkerboard: 5-unit cells, greys `0.6` and `0.9`. Pick the two axes to
  checker on from the dominant axis of the normal, so walls checker in their own
  plane rather than smearing. Anti-alias by comparing the cell size against
  `fwidth` of the checker coordinate and fading to the average grey when a pixel
  spans more than a cell — otherwise grazing angles alias violently.
- Teapot: albedo `(0.85, 0.80, 0.72)` modulated by `default.png`, loaded via an
  `Image` and `texImage2D` with mipmaps and `LINEAR_MIPMAP_LINEAR`.
- Shading for this task is Lambert with a **hardcoded** directional term plus a
  flat ambient — no shadows, no medium. Those arrive in Tasks 6 and 8.
- Render into an `RGBA16F` colour texture plus a `DEPTH_COMPONENT24` **texture**
  (not a renderbuffer — Task 7 reads it), then blit to the screen with a
  temporary passthrough shader. The composite pass replaces that blit in Task 9.
- Resize the offscreen targets when the canvas size changes.

- [ ] **Step 1: Build the room geometry in `scene.js` and verify vertex counts**

The floor is 4 vertices and 6 indices; the walls are 16 vertices and 24 indices.
Log them once and check.

- [ ] **Step 2: Write the opaque vertex and fragment shaders in `shaders.js`**

Uniforms: `u_viewProjection`, `u_model`, `u_normalMatrix`, `u_lightDirection`,
`u_lightIntensity`, `u_ambientIntensity`, `u_useTexture`, `u_texture`,
`u_albedo`, `u_checker` (0 = solid albedo, 1 = procedural checker).

- [ ] **Step 3: Wire the opaque pass in `renderer.js` and draw**

- [ ] **Step 4: Serve and verify visually**

Expected: a grey checkered room seen from inside, walls correctly lit from
different angles, teapot in the centre standing on the floor at roughly a
quarter of the room's width, no z-fighting, checker not shimmering when the
camera moves. The teapot must look **smooth**, not faceted — faceted means the
normal accumulation in Task 3 is wrong.

- [ ] **Step 5: Commit**

```bash
git add src/scene.js src/shaders.js src/renderer.js src/main.js && git commit -m "feat: room, teapot, opaque pass"
```

---

## Task 6: Shadow map

**Files:**
- Modify: `src/renderer.js`, `src/shaders.js`

**Interfaces:**
- Consumes: `state.lightDirection`, the room and teapot buffer infos.
- Produces: `u_lightViewProjection` and a bound `u_shadowMap` for the opaque
  shader; a `shadowFactor(vec3 worldPos, vec3 N)` GLSL function reused by the
  volume pass in Task 7.

**Requirements:**

- 2048 × 2048 `DEPTH_COMPONENT24` texture, no colour attachment. Set
  `TEXTURE_COMPARE_MODE` to `NONE` and sample it as a plain depth texture — we
  do our own PCF so we can share the function with the volume pass, where
  hardware comparison sampling is awkward.
- Refit the light frustum every frame: build a light view matrix with
  `m4.lookAt` from a point along `-lightDirection`, transform the eight corners
  of the scene AABB (the 100 × 100 × 50 room union the transformed teapot bbox
  union the transformed medium bbox) into light space, and take the min/max as
  the `m4.orthographic` bounds. Pad the near plane outward so casters behind the
  fitted volume still register.
- `m4.lookAt` degenerates when the light points straight down, because the up
  vector becomes parallel to the view direction. Pick the up vector as `[0,0,1]`
  when `abs(lightDirection.y) > 0.99`, otherwise `[0,1,0]`.
- Render **opaque only** — floor, walls, teapot. The medium is never in the
  shadow map.
- 3 × 3 PCF over `1/2048` texel offsets, with a slope-scaled bias:
  `bias = max(baseBias, slopeBias * (1 - dot(N, L)))` with `baseBias` around
  `0.0015` and `slopeBias` around `0.005` in normalised depth units. Tune until
  neither acne nor peter-panning is visible.
- Fragments outside the shadow map must count as **lit**, not shadowed.

- [ ] **Step 1: Add the depth-only shader and the shadow framebuffer**

- [ ] **Step 2: Add the frustum fit and render the shadow pass**

- [ ] **Step 3: Add `shadowFactor` to the opaque fragment shader and multiply the directional term by it**

- [ ] **Step 4: Serve and verify visually**

Expected: the teapot casts a shadow on the floor that moves correctly when the
light yaw/pitch change; walls shadow the floor at low sun angles; no acne on the
floor at grazing light; no visible gap between the teapot and its shadow. Sweep
the light through a full yaw rotation and both pitch extremes, checking that the
shadow never pops or disappears — popping means the frustum fit is wrong.

- [ ] **Step 5: Commit**

```bash
git add src/renderer.js src/shaders.js && git commit -m "feat: shadow map"
```

---

## Task 7: Volumetric pass — single scattering

**Files:**
- Modify: `src/renderer.js`, `src/shaders.js`, `src/main.js`

**Interfaces:**
- Consumes: the decoded grid from Task 2, the depth texture from Task 5, the
  shadow map from Task 6.
- Produces: a half-resolution `RGBA16F` texture holding
  `vec4(inScatter, transmittance)`.

**Requirements:**

- Upload the grid as an `R8` `TEXTURE_3D`, `LINEAR` filtering,
  `CLAMP_TO_EDGE` on all three axes. `UNPACK_ALIGNMENT` must be set to `1`
  before `texImage3D` because the row length is not a multiple of 4.
- Target is `RGBA16F` at half the drawing-buffer size in each dimension.
- Reconstruct the world ray from `u_inverseViewProjection` applied to the
  clip-space corners of the fullscreen triangle.
- Transform the ray origin and direction into medium local space with
  `u_mediumInverse` and intersect the `[-0.5, 0.5]³` box with the standard slab
  test. **Do not normalise the local-space direction** — keeping it in world
  parameterisation means `t` stays in world units, which is what the optical
  depth needs.
- Clamp `tmax` by the scene depth: read the depth texture, convert to view depth
  with the projection constants, and convert to a distance along the ray.
- 64 steps. Jitter the start by a per-pixel hash of `gl_FragCoord` and the
  frame index, scaled by one step length.
- Per step:

  ```glsl
  float D  = texture(u_density, localPos + 0.5).r * u_maxDensity;
  float st = D * u_densityScale;               // extinction
  vec3  ss = st * u_mediumColor;               // scattering albedo, 0..1
  float trL = transmittanceThrough(worldPos, -u_lightDirection, 12);
  float vis = shadowFactor(worldPos, vec3(0.0));
  float ph  = henyeyGreenstein(dot(viewDir, -u_lightDirection), 0.3);

  vec3 S    = ss * ph * u_lightColor * u_lightIntensity * trL * vis
            + ss * u_ambientIntensity * u_skyColor;
  vec3 Sint = (S - S * exp(-st * dt)) / max(st, 1e-6);
  L += T * Sint;
  T *= exp(-st * dt);
  ```

  The `Sint` line is the energy-conserving analytic integration over the step —
  a plain `L += T * S * dt` rectangle rule visibly loses energy at these step
  counts and makes the cloud thin and flat.
- `transmittanceThrough(vec3 worldPos, vec3 dir, int steps)` marches from the
  sample along `dir` with exponentially growing step lengths, accumulating
  `exp(-sum(st * dt))`, and stops when it leaves the box. **Give it this exact
  name and signature now** — Task 8 promotes this same function into a shared
  chunk used by the opaque shader, where it is also called with `+Y` for sky
  visibility.
- Early out when `T < 0.003`.
- Henyey-Greenstein:
  `(1 - g²) / (4π * pow(1 + g² - 2g*cosθ, 1.5))`.
- Sign check: `u_lightDirection` is the direction light **travels**, so the
  vector pointing from the sample toward the light is `-u_lightDirection`.
  Getting this backwards produces a cloud that is bright on the wrong side —
  exactly the directionality trap the reference article calls out.
- Composite temporarily with `final = opaque * T + inScatter` and a naive
  bilinear upsample. The bilateral upsample and tonemapping arrive in Task 9.

- [ ] **Step 1: Upload the 3D texture and verify it with a debug shader that shows max density along the ray**

This isolates sampling and box intersection from the scattering maths. The
bunny silhouette must be recognisable before going further.

- [ ] **Step 2: Implement the box intersection and the depth clamp**

- [ ] **Step 3: Implement the march with the analytic step integration**

- [ ] **Step 4: Implement `lightTransmittance` and the shadow map term**

- [ ] **Step 5: Serve and verify visually**

Expected: a recognisable bunny-shaped cloud, bright on the side facing the
light and dark on the far side, the teapot correctly occluding it when in
front, and the teapot's shadow visible **inside** the cloud when the teapot is
between the light and the cloud. The cloud must be correctly hidden behind
walls when the camera is outside the room looking in.

- [ ] **Step 6: Commit**

```bash
git add src/renderer.js src/shaders.js src/main.js && git commit -m "feat: volumetric single scattering"
```

---

## Task 8: The medium shadows the opaque scene

**Files:**
- Modify: `src/shaders.js`, `src/renderer.js`

**Interfaces:**
- Consumes: `lightTransmittance` from Task 7, promoted into a shared GLSL
  chunk in `shaders.js` and concatenated into both the opaque and volume
  fragment shaders.
- Produces: nothing new for later tasks.

**Requirements:**

- Move the box intersection, the density sample, and `transmittanceThrough`
  (already named and signed correctly in Task 7) into a chunk
  `shaders.mediumChunk`, and concatenate it into both the opaque and volume
  fragment shaders rather than duplicating the code.
- The opaque fragment shader gains `u_mediumInverse`, `u_density`,
  `u_maxDensity`, `u_densityScale`, and `u_mediumColor`.
- Directional term becomes:

  ```glsl
  float shadow = shadowFactor(worldPos, N);
  vec3 direct = vec3(0.0);
  if (shadow > 0.0) {
    float trL = transmittanceThrough(worldPos, -u_lightDirection, 12);
    direct = albedo / PI * u_lightColor * u_lightIntensity * trL * shadow * max(dot(N, L), 0.0);
  }
  ```

  Skipping the march entirely when `shadow == 0.0` is both the spec's stated
  behaviour and a real saving over the shadowed half of the scene.
- Ambient becomes hemispheric, occluded by the medium:

  ```glsl
  float skyVis = transmittanceThrough(worldPos, vec3(0.0, 1.0, 0.0), 8);
  vec3 hemi = mix(u_groundColor, u_skyColor, N.y * 0.5 + 0.5);
  vec3 ambient = u_ambientIntensity * hemi * albedo * skyVis;
  ```

  `u_skyColor` is a cool light blue, `u_groundColor` a warm neutral; both are
  fixed constants, since the brief exposes only the ambient intensity.

- [ ] **Step 1: Extract the shared medium chunk and confirm the volume pass still renders identically**

- [ ] **Step 2: Add the directional transmittance term to the opaque shader**

- [ ] **Step 3: Add the hemispheric ambient with sky visibility**

- [ ] **Step 4: Serve and verify visually**

Expected: the cloud now casts a soft coloured shadow on the floor and walls,
distinct from the teapot's hard shadow. Raising the density multiplier darkens
that shadow. Setting the light intensity to 0 leaves the ambient term alone, and
the floor directly under the cloud is visibly darker than the open floor —
that is `skyVis` working. Setting the density multiplier to 0 must make both
effects vanish completely, leaving exactly the Task 6 image.

- [ ] **Step 5: Commit**

```bash
git add src/shaders.js src/renderer.js && git commit -m "feat: medium shadows opaque surfaces"
```

---

## Task 9: Multiple scattering, bilateral upsample, tonemap

**Files:**
- Modify: `src/shaders.js`, `src/renderer.js`

**Interfaces:**
- Consumes: everything from Tasks 7 and 8.
- Produces: the final composite pass.

**Requirements:**

- **Multiple scattering**, three Wrenninge octaves. Inside the existing march,
  after computing `D` and `trL` once, loop three octaves with
  `a = pow(0.5, i)` scaling scattering, `b = pow(0.5, i)` scaling extinction,
  and `c = pow(0.5, i)` scaling the phase eccentricity `g`:

  ```glsl
  vec3 Ltotal = vec3(0.0);
  for (int i = 0; i < 3; i++) {
    float a = pow(0.5, float(i));
    float b = a;
    float c = a;
    float phase = henyeyGreenstein(cosTheta, 0.3 * c);
    float trLi  = pow(trL, b);
    Ltotal += a * ss * phase * u_lightColor * u_lightIntensity * trLi * vis;
  }
  ```

  Reuse the already-sampled `D` and `trL`; do not re-march per octave. Higher
  octaves are cheaper light that scatters more isotropically and is absorbed
  less, which is what fills in the dead-black cloud core the reference article
  warns about.
- **Bilateral upsample** in the composite: for each full-res pixel, take the
  four contributing half-res texels, weight each by
  `exp(-abs(halfResDepth - fullResDepth) * k)` times the bilinear weight,
  and normalise. Fall back to plain bilinear when all four weights underflow.
  This is what removes the halo around the teapot silhouette.
- Store linear view depth in a second half-res render target, or reconstruct it
  from the full-res depth texture at half-res sample positions — either is fine,
  but be consistent.
- **ACES-style tonemap** then gamma 2.2:

  ```glsl
  vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
  }
  ```
- The composite replaces the temporary blit from Task 5.

- [ ] **Step 1: Add the multi-scatter octave loop and compare against single scattering**

Toggle the octave count between 1 and 3 and confirm the cloud core brightens
without the silhouette blowing out.

- [ ] **Step 2: Implement the bilateral upsample in the composite shader**

- [ ] **Step 3: Add the ACES tonemap and gamma, and remove the temporary blit**

- [ ] **Step 4: Serve and verify visually**

Expected: no dark halo where the cloud meets the teapot or the walls; the cloud
core is soft grey rather than black; no banding at the default step count; the
image is not blown out at light intensity 3.

- [ ] **Step 5: Commit**

```bash
git add src/shaders.js src/renderer.js && git commit -m "feat: multi-scattering, bilateral upsample, tonemap"
```

---

## Task 10: UI

**Files:**
- Create: `src/ui.js`
- Modify: `index.html`, `src/main.js`

**Interfaces:**
- Consumes: the state object from `scene.createState()`.
- Produces: `ui.create(state, onChange)` — builds the panel, writes directly
  into `state`, and calls `onChange()` after each change so the renderer can
  recompute matrices.

**Requirements:**

Hand-rolled HTML and CSS, no library. Controls, matching the spec's table:

| Group | Control | Range | Default |
|---|---|---|---|
| Directional light | yaw | −180°..180° | 30° |
| | pitch | −89°..89° | −50° |
| | intensity | 0..10 | 3.0 |
| Ambient | intensity | 0..2 | 0.35 |
| Teapot | position x/y/z | ±40 | 0, 0, 0 |
| | rotation x/y/z | −180°..180° | 0, 0, 0 |
| | scale | 0.05..1.0 | 0.3 |
| Medium | position x/y/z | ±50 | 0, 38, 0 |
| | rotation x/y/z | −180°..180° | 0, 0, 0 |
| | scale | 5..80 | 32 |
| | density multiplier | 0..3 | `DENSITY_DEFAULT` from Task 5 |
| | colour | colour picker | `#F2F5FF` |

- Each slider is a labelled `<input type="range">` with a live numeric readout;
  colour is `<input type="color">`.
- Angles are presented in degrees and stored in radians.
- The panel must not steal the camera's right mouse drag — put it in a corner
  and let the canvas keep its own listeners.
- Add an FPS readout, averaged over roughly half a second, and a reset button
  that restores `scene.createState()` defaults and refreshes every control.
- Add a short controls legend so the camera scheme is discoverable: right mouse
  to look, `w`/`s`/`a`/`d` to move, `e`/`q` up/down, wheel for speed, shift to
  boost.

- [ ] **Step 1: Write the panel markup and CSS in `index.html`**

- [ ] **Step 2: Write `src/ui.js` to build the controls and bind them to state**

- [ ] **Step 3: Wire `onChange` to `scene.updateMatrices` in `main.js`**

- [ ] **Step 4: Serve and verify every control**

Expected: every slider visibly changes the image in the correct direction;
rotating the medium rotates the cloud without shearing it; the colour picker
tints both the cloud and the shadow it casts; reset restores the initial view;
right-dragging over the canvas still flies the camera.

- [ ] **Step 5: Commit**

```bash
git add index.html src/ui.js src/main.js && git commit -m "feat: control panel"
```

---

## Final verification checklist

Run through all of these before calling the demo done.

- [ ] `node --test "tests/*.test.js"` passes, 19 tests.
- [ ] Active voxel count is exactly `19210271` and the bbox is exactly
      `[-300,-47,-208]..[276,524,229]`.
- [ ] Teapot is smooth-shaded, standing on the floor, centred at the origin.
- [ ] Teapot and walls cast shadows on the floor and receive them.
- [ ] The teapot's shadow is visible inside the cloud.
- [ ] The cloud casts a soft shadow on the floor, separate from the teapot's.
- [ ] Setting the density multiplier to 0 removes every trace of the medium.
- [ ] Setting light intensity to 0 leaves a plausible ambient-only image.
- [ ] The cloud core is not black.
- [ ] No halo at the cloud/geometry silhouettes.
- [ ] No banding at default quality; no checker shimmer at grazing angles.
- [ ] Camera controls match the UE scheme exactly.
- [ ] No console errors, no WebGL warnings.
- [ ] `src/webgl-utils.js` is byte-identical to the original.
