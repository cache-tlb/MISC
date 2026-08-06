/*
 * Minimal Blosc 1.x decompressor, enough for the buffers OpenVDB writes.
 *
 * OpenVDB always compresses with LZ4 and byte shuffle at typesize =
 * sizeof(ValueType), so that is the only path implemented. Anything else
 * throws rather than quietly producing garbage.
 *
 * A Blosc buffer is a 16 byte header, an int32 offset table with one entry per
 * block, then the blocks themselves. Whether a block is split into typesize
 * sub-streams is *not* stored in the file: the decompressor recomputes it with
 * the same rule the compressor used.
 *
 * @module blosc
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
    root.blosc = factory();
  }
}(this, function() {
  "use strict";

  var FLAG_SHUFFLE = 0x01;
  var FLAG_MEMCPYED = 0x02;
  var FLAG_BITSHUFFLE = 0x04;
  var FLAG_DELTA = 0x08;

  var COMPRESSOR_NAMES = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd'];

  // c-blosc 1.x recomputes the block split on read; these are its constants.
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
   * Reverses Blosc's byte shuffle over one block. The shuffled layout is all
   * of the elements' byte 0, then all of their byte 1, and so on.
   * @param {Uint8Array} src shuffled bytes, read from index 0
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
          (COMPRESSOR_NAMES[compressor] || compressor) +
          ', only lz4 is supported');
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
            throw new Error('blosc: block ' + i + ' stream ' + j +
                ' produced ' + written + ' bytes, expected ' + neblock);
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
