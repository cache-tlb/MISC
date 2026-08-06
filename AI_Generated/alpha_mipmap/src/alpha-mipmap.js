/*
 * Alpha mask mipmap generation, with and without alpha test coverage
 * preservation.
 *
 * Background
 * ----------
 * An alpha tested (clipped) texture is drawn with `if (a < Ar) discard;`. The
 * quantity that decides what the surface looks like is therefore not the
 * average alpha but the *coverage*: the fraction of the texture that passes the
 * test.
 *
 *     coverage = count(a_i >= Ar) / N
 *
 * A plain box filter averages alpha, and averaging pulls values toward the
 * middle. For foliage-style masks, where most texels are either 0 or 1 and the
 * mask covers well under half the texture, every mip level loses coverage, so
 * a tree thins out and eventually vanishes as it recedes from the camera.
 *
 * The fix (Ignacio Castano, "Computing Alpha Mipmaps",
 * https://www.ludicon.com/castano/blog/articles/computing-alpha-mipmaps/) is to
 * rescale the alpha of every mip level so its coverage matches level 0's.
 * Rather than searching for the scale directly, which is unbounded, the search
 * is for the alpha reference value `ar` that the *unscaled* level would need in
 * order to reproduce the original coverage:
 *
 *     count(a_i >= ar) / N = coverage
 *
 * `ar` lives in [0, 1], so it can be found by bisection. The scale that moves
 * that level onto the real threshold is then simply:
 *
 *     scale = Ar / ar
 *
 * Implementation notes
 * --------------------
 * Coverage is estimated the way the GPU actually reads the texture: by
 * bilinearly supersampling each 2x2 texel quad (`SUBSAMPLES` samples per axis)
 * rather than by counting texels. That matters because a bilinear tap between
 * a covered and an uncovered texel can still pass the test.
 *
 * Because the demo lets the threshold be changed interactively, the expensive
 * part of that estimate is factored out: each level's supersampled alpha values
 * are accumulated once into a 256 bin histogram, and coverage at any threshold
 * is then a suffix sum lookup. Re-solving the whole chain for a new threshold
 * costs microseconds.
 *
 * The mip chain itself is always box filtered from the *unscaled* parent, so
 * the corrections applied to each level never compound.
 *
 * @module alpha-mipmap
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define([], factory);
  } else {
    // Browser globals
    root.alphaMipmap = factory();
  }
}(this, function() {
  "use strict";

  // Bilinear samples per axis taken inside every texel quad when estimating
  // coverage. 2 (ie. 4 samples per quad) is what NVTT uses.
  var SUBSAMPLES = 2;

  // Number of bisection steps used to solve for the alpha reference value.
  var BISECTION_STEPS = 24;

  // The scale is clamped to this range. Without a clamp a level whose coverage
  // can never reach the target drives the solved reference toward 0 and the
  // scale toward infinity.
  var MIN_SCALE = 1 / 16;
  var MAX_SCALE = 16;

  /**
   * One mip level of a single channel image.
   * @typedef {Object} Level
   * @property {number} width width in texels
   * @property {number} height height in texels
   * @property {Float32Array} data alpha in [0, 1], row 0 first
   * @memberOf module:alpha-mipmap
   */

  /**
   * Extracts one channel of an interleaved RGBA8 image as floats in [0, 1].
   * @param {Uint8Array|Uint8ClampedArray} rgba the source pixels
   * @param {number} width width in texels
   * @param {number} height height in texels
   * @param {number} [channel] 0=R, 1=G, 2=B, 3=A. Defaults to alpha.
   * @return {Level} the extracted channel as mip level 0
   * @memberOf module:alpha-mipmap
   */
  function extractChannel(rgba, width, height, channel) {
    channel = channel === undefined ? 3 : channel;
    var data = new Float32Array(width * height);
    for (var i = 0; i < data.length; ++i) {
      data[i] = rgba[i * 4 + channel] / 255;
    }
    return {width: width, height: height, data: data};
  }

  /**
   * Box filters one mip level down to the next.
   * @param {Level} level the level to reduce
   * @return {Level} a level half the size on each axis
   * @memberOf module:alpha-mipmap
   */
  function downsample(level) {
    var srcWidth = level.width;
    var srcHeight = level.height;
    var src = level.data;
    var dstWidth = Math.max(1, srcWidth >> 1);
    var dstHeight = Math.max(1, srcHeight >> 1);
    var dst = new Float32Array(dstWidth * dstHeight);

    // Odd sizes cannot happen for the power of two textures this demo uses,
    // but clamping the second tap keeps the filter well defined anyway.
    for (var y = 0; y < dstHeight; ++y) {
      var y0 = Math.min(y * 2, srcHeight - 1) * srcWidth;
      var y1 = Math.min(y * 2 + 1, srcHeight - 1) * srcWidth;
      for (var x = 0; x < dstWidth; ++x) {
        var x0 = Math.min(x * 2, srcWidth - 1);
        var x1 = Math.min(x * 2 + 1, srcWidth - 1);
        dst[y * dstWidth + x] =
            (src[y0 + x0] + src[y0 + x1] + src[y1 + x0] + src[y1 + x1]) * 0.25;
      }
    }

    return {width: dstWidth, height: dstHeight, data: dst};
  }

  /**
   * Builds a complete box filtered mip chain down to 1x1. This is the "default
   * mipmap" of the demo and is exactly what gl.generateMipmap would produce.
   * @param {Level} level0 the base level
   * @return {Level[]} the chain, level 0 first
   * @memberOf module:alpha-mipmap
   */
  function buildChain(level0) {
    var levels = [level0];
    var level = level0;
    while (level.width > 1 || level.height > 1) {
      level = downsample(level);
      levels.push(level);
    }
    return levels;
  }

  /**
   * Accumulates the bilinearly supersampled alpha of a level into a 256 bin
   * histogram, then turns it into a suffix sum so coverage at a given threshold
   * is a single lookup.
   *
   * The returned array has 257 entries where `cdf[b]` is the fraction of
   * samples whose alpha, quantized to 8 bits, is >= b. `cdf[0]` is 1 and
   * `cdf[256]` is 0.
   *
   * @param {Level} level the level to measure
   * @param {number} [subsamples] bilinear samples per axis per texel quad
   * @return {Float64Array} the suffix summed histogram
   * @memberOf module:alpha-mipmap
   */
  function buildCoverageCdf(level, subsamples) {
    subsamples = subsamples || SUBSAMPLES;
    var width = level.width;
    var height = level.height;
    var data = level.data;
    var hist = new Float64Array(256);
    var total = 0;

    if (width < 2 || height < 2) {
      // Too small for a texel quad, so fall back to counting texels.
      for (var i = 0; i < data.length; ++i) {
        hist[Math.min(255, Math.max(0, Math.round(data[i] * 255)))] += 1;
        total += 1;
      }
    } else {
      var step = 1 / subsamples;
      for (var y = 0; y < height - 1; ++y) {
        var row0 = y * width;
        var row1 = row0 + width;
        for (var x = 0; x < width - 1; ++x) {
          var a00 = data[row0 + x];
          var a10 = data[row0 + x + 1];
          var a01 = data[row1 + x];
          var a11 = data[row1 + x + 1];
          for (var sy = 0; sy < subsamples; ++sy) {
            var fy = (sy + 0.5) * step;
            for (var sx = 0; sx < subsamples; ++sx) {
              var fx = (sx + 0.5) * step;
              var top = a00 + (a10 - a00) * fx;
              var bottom = a01 + (a11 - a01) * fx;
              var a = top + (bottom - top) * fy;
              hist[Math.min(255, Math.max(0, Math.round(a * 255)))] += 1;
              total += 1;
            }
          }
        }
      }
    }

    var cdf = new Float64Array(257);
    cdf[256] = 0;
    for (var b = 255; b >= 0; --b) {
      cdf[b] = cdf[b + 1] + hist[b] / total;
    }
    return cdf;
  }

  /**
   * Reads coverage at an arbitrary threshold out of a suffix summed histogram,
   * interpolating between bins so the result is continuous in `alphaRef`.
   * @param {Float64Array} cdf from buildCoverageCdf
   * @param {number} alphaRef the alpha test threshold in [0, 1]
   * @return {number} the fraction of the level that passes the test
   * @memberOf module:alpha-mipmap
   */
  function coverageAt(cdf, alphaRef) {
    var x = alphaRef * 255;
    if (x <= 0) {
      return cdf[0];
    }
    if (x >= 255) {
      return cdf[255];
    }
    var b = Math.floor(x);
    var f = x - b;
    return cdf[b] * (1 - f) + cdf[b + 1] * f;
  }

  /**
   * Coverage of a level after its alpha has been scaled and requantized to the
   * 8 bits that actually get uploaded.
   *
   * This is what the GPU ends up testing, and it can differ sharply from the
   * idealised `coverageAt(cdf, alphaRef / scale)` on the coarsest levels: once
   * a 4x4 level has averaged down to a nearly uniform value, its coverage can
   * only ever be 0% or 100%, no matter what scale is chosen. Used for
   * reporting, not for solving.
   *
   * @param {Float64Array} cdf from buildCoverageCdf
   * @param {number} alphaRef the alpha test threshold in [0, 1]
   * @param {number} scale the alpha scale applied to the level
   * @return {number} the fraction of the uploaded level that passes the test
   * @memberOf module:alpha-mipmap
   */
  function coverageAfterScale(cdf, alphaRef, scale) {
    var threshold = alphaRef * 255;
    // Quantization is monotonic in the source value, so the test still splits
    // the histogram at a single bin: find the lowest one that survives.
    for (var b = 0; b <= 255; ++b) {
      if (Math.min(255, Math.round(b * scale)) >= threshold) {
        return cdf[b];
      }
    }
    return 0;
  }

  /**
   * Solves for the alpha scale that makes one level's coverage match a target.
   *
   * Bisects on the alpha reference value `ar` (bounded to [0, 1]) rather than
   * on the scale (unbounded), then converts. Coverage is non increasing in
   * `ar`, so too much coverage means `ar` must go up.
   *
   * @param {Float64Array} cdf from buildCoverageCdf
   * @param {number} targetCoverage level 0's coverage
   * @param {number} alphaRef the alpha test threshold the shader uses
   * @return {{scale: number, alphaRefFound: number}} the solution
   * @memberOf module:alpha-mipmap
   */
  function solveAlphaScale(cdf, targetCoverage, alphaRef) {
    var minRef = 0;
    var maxRef = 1;
    var ref = alphaRef;

    for (var i = 0; i < BISECTION_STEPS; ++i) {
      var coverage = coverageAt(cdf, ref);
      if (coverage > targetCoverage) {
        minRef = ref;
      } else if (coverage < targetCoverage) {
        maxRef = ref;
      } else {
        break;
      }
      ref = (minRef + maxRef) * 0.5;
    }

    var scale = ref > 0 ? alphaRef / ref : MAX_SCALE;
    return {
      scale: Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale)),
      alphaRefFound: ref,
    };
  }

  /**
   * Solves the whole chain against level 0's coverage.
   * The solve itself works on the continuous coverage estimate, which is what
   * makes the result move smoothly as the threshold changes. The reported
   * coverages are measured on the quantized texels that get uploaded instead,
   * so the numbers describe what the GPU will really do.
   *
   * @param {Float64Array[]} cdfs one per level, from buildCoverageCdf
   * @param {number} alphaRef the alpha test threshold in [0, 1]
   * @return {Object[]} per level {level, scale, alphaRefFound, coverageBefore,
   *     coverageAfter, targetCoverage}
   * @memberOf module:alpha-mipmap
   */
  function solveChain(cdfs, alphaRef) {
    var targetCoverage = coverageAt(cdfs[0], alphaRef);
    return cdfs.map(function(cdf, level) {
      // With nothing covered there is nothing to preserve, and scaling up an
      // all-transparent level would only amplify noise.
      var solution = targetCoverage > 0 && level > 0
          ? solveAlphaScale(cdf, targetCoverage, alphaRef)
          : {scale: 1, alphaRefFound: alphaRef};
      return {
        level: level,
        scale: solution.scale,
        alphaRefFound: solution.alphaRefFound,
        coverageBefore: coverageAfterScale(cdf, alphaRef, 1),
        coverageAfter: coverageAfterScale(cdf, alphaRef, solution.scale),
        targetCoverage: targetCoverage,
      };
    });
  }

  /**
   * Quantizes a level to 8 bits, optionally scaling it on the way.
   * @param {Level} level the level to convert
   * @param {number} [scale] alpha scale to apply, saturating at 1
   * @return {Uint8Array} the level as R8 texels
   * @memberOf module:alpha-mipmap
   */
  function toUint8(level, scale) {
    scale = scale === undefined ? 1 : scale;
    var src = level.data;
    var dst = new Uint8Array(src.length);
    for (var i = 0; i < src.length; ++i) {
      var v = src[i] * scale * 255;
      dst[i] = v < 0 ? 0 : (v > 255 ? 255 : Math.round(v));
    }
    return dst;
  }

  /**
   * Prepares everything about one alpha mask that does not depend on the alpha
   * test threshold. Call once at load time, then call `solveChain` whenever the
   * threshold changes.
   * @param {Uint8Array|Uint8ClampedArray} rgba interleaved RGBA8 pixels
   * @param {number} width width in texels
   * @param {number} height height in texels
   * @param {number} [channel] which channel holds the mask. Defaults to alpha.
   * @return {{levels: Level[], cdfs: Float64Array[]}} the mip chain plus the
   *     per level coverage histograms
   * @memberOf module:alpha-mipmap
   */
  function prepare(rgba, width, height, channel) {
    var levels = buildChain(extractChannel(rgba, width, height, channel));
    var cdfs = levels.map(function(level) {
      return buildCoverageCdf(level);
    });
    return {levels: levels, cdfs: cdfs};
  }

  return {
    extractChannel: extractChannel,
    downsample: downsample,
    buildChain: buildChain,
    buildCoverageCdf: buildCoverageCdf,
    coverageAt: coverageAt,
    coverageAfterScale: coverageAfterScale,
    solveAlphaScale: solveAlphaScale,
    solveChain: solveChain,
    toUint8: toUint8,
    prepare: prepare,
  };

}));
