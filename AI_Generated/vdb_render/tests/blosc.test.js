'use strict';
var test = require('node:test');
var assert = require('node:assert');
var blosc = require('../src/blosc.js');

var FLAG_SHUFFLE = 0x01;
var FLAG_MEMCPYED = 0x02;
var LZ4 = 1 << 5;

/**
 * Builds a single-block blosc buffer around an already-encoded payload.
 * The offset table has one entry, so the block starts 20 bytes in.
 */
function container(opts) {
  var payload = opts.payload;
  var total = 16 + 4 + 4 + payload.length;
  var out = new Uint8Array(total);
  var view = new DataView(out.buffer);
  out[0] = 2;                        // version
  out[1] = 1;                        // versionlz
  out[2] = opts.flags;
  out[3] = opts.typesize;
  view.setUint32(4, opts.nbytes, true);
  view.setUint32(8, opts.blocksize, true);
  view.setUint32(12, total, true);
  view.setInt32(16, 20, true);       // offset table: block 0 starts at 20
  view.setInt32(20, opts.cbytes, true);
  out.set(payload, 24);
  return out;
}

/** Blosc's forward byte shuffle, so the test can build shuffled input. */
function shuffle(src, typesize) {
  var n = Math.floor(src.length / typesize);
  var dst = new Uint8Array(src.length);
  for (var j = 0; j < typesize; j++) {
    for (var i = 0; i < n; i++) {
      dst[j * n + i] = src[i * typesize + j];
    }
  }
  for (var k = n * typesize; k < src.length; k++) {
    dst[k] = src[k];
  }
  return dst;
}

test('decodes an lz4 stream with literals and an overlapping match', function() {
  // "abc" literals, then a 12 byte match at offset 3, then the literal "X".
  // The match overlaps the bytes being written, which is the case a naive
  // block copy gets wrong.
  var lz4 = new Uint8Array([
    0x38,                    // token: litLen 3, matchLen field 8 (-> 12)
    0x61, 0x62, 0x63,        // "abc"
    0x03, 0x00,              // match offset 3
    0x10,                    // token: litLen 1, no match follows
    0x58,                    // "X"
  ]);
  var buf = container({
    flags: LZ4, typesize: 4, nbytes: 16, blocksize: 16,
    cbytes: lz4.length, payload: lz4,
  });

  var out = blosc.decompress(buf, 0);
  assert.strictEqual(Buffer.from(out).toString('latin1'), 'abcabcabcabcabcX');
});

test('reverses the byte shuffle', function() {
  var plain = new Uint8Array(16);
  for (var i = 0; i < 16; i++) {
    plain[i] = i + 1;
  }
  var shuffled = shuffle(plain, 4);
  assert.notDeepStrictEqual(Array.from(shuffled), Array.from(plain),
      'test setup is wrong, shuffling did nothing');

  // cbytes === neblock marks a sub-stream as stored raw, skipping lz4.
  var buf = container({
    flags: LZ4 | FLAG_SHUFFLE, typesize: 4, nbytes: 16, blocksize: 16,
    cbytes: 16, payload: shuffled,
  });

  var out = blosc.decompress(buf, 0);
  assert.deepStrictEqual(Array.from(out), Array.from(plain));
});

test('handles a memcpy-ed buffer', function() {
  var plain = new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2]);
  var buf = new Uint8Array(16 + plain.length);
  var view = new DataView(buf.buffer);
  buf[0] = 2;
  buf[2] = LZ4 | FLAG_MEMCPYED;
  buf[3] = 4;
  view.setUint32(4, plain.length, true);
  view.setUint32(8, plain.length, true);
  buf.set(plain, 16);

  assert.deepStrictEqual(Array.from(blosc.decompress(buf, 0)), Array.from(plain));
});

test('decodes at a non-zero offset', function() {
  var lz4 = new Uint8Array([0x40, 0x61, 0x62, 0x63, 0x64]);  // 4 literals "abcd"
  var inner = container({
    flags: LZ4, typesize: 4, nbytes: 4, blocksize: 4,
    cbytes: lz4.length, payload: lz4,
  });
  var padded = new Uint8Array(inner.length + 7);
  padded.set(inner, 7);

  var out = blosc.decompress(padded, 7);
  assert.strictEqual(Buffer.from(out).toString('latin1'), 'abcd');
});

test('rejects an unsupported compressor', function() {
  var header = new Uint8Array(16);
  header[0] = 2;
  header[2] = 4 << 5;      // compressor id 4 = zlib
  header[3] = 4;
  new DataView(header.buffer).setUint32(4, 64, true);
  assert.throws(function() { blosc.decompress(header, 0); }, /compressor/i);
});

test('rejects bitshuffle', function() {
  var header = new Uint8Array(16);
  header[0] = 2;
  header[2] = LZ4 | 0x04;
  header[3] = 4;
  new DataView(header.buffer).setUint32(4, 64, true);
  assert.throws(function() { blosc.decompress(header, 0); }, /bitshuffle/i);
});
