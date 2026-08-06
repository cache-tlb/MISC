import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadOt, supersampledCoverage } from './_helpers.mjs';
import { wrapFont } from '../js/font-loader.js';
import { preprocessGlyph } from '../js/sweeper-preprocess.js';
import { bboxOfCommands, flattenCommands, evalQuad } from '../js/geom.js';
import { coverage, detectSign, intersectMonotonic } from '../js/sweeper-core.js';

test('intersectMonotonic linear crossing', () => {
  assert.ok(Math.abs(intersectMonotonic(0, 0, 0.5, 1, 0.5) - 0.5) < 1e-6);
});

// The reference guards its divide-by-qa with a linear shortcut under |qa|<1e-3.
// That threshold is absolute, so in em space it admits a crossing error of up to
// qa/4 em — a few percent of a pixel at 500px, which showed as faint horizontal
// seams. The crossing must land on the target however small qa gets.
test('intersectMonotonic is exact for near-linear curves', () => {
  const c0 = -0.00109, c1 = 0.0099, c2 = 0.02113;   // 'S' spine, box-relative at 500px/em
  const qa = c0 + c2 - 2*c1;
  assert.ok(Math.abs(qa) < 1e-3, `exercises the near-linear case (qa=${qa})`);
  for (const target of [0, 1/500]) {
    const t = intersectMonotonic(qa, c0, c1, c2, target);
    const y = evalQuad([0,c0], [0,c1], [0,c2], t)[1];
    assert.ok(Math.abs(y - target) < 1e-12, `y(${t}) = ${y}, expected ${target}`);
  }
});

const square = [
  {type:'M',x:0,y:0},{type:'L',x:1,y:0},{type:'L',x:1,y:1},{type:'L',x:0,y:1},{type:'Z'}
];
const sqCurves = preprocessGlyph(square);
const sqSign = detectSign(sqCurves, {minX:0,minY:0,maxX:1,maxY:1});
const cov = (cx, cy, w) => coverage(sqCurves, [w, w], [cx - w/2, cy - w/2], sqSign);

test('unit square coverage inside/outside/straddle', () => {
  const w = 0.02;
  assert.ok(cov(0.5, 0.5, w) > 0.98, `inside ${cov(0.5,0.5,w)}`);
  assert.ok(cov(1.5, 0.5, w) < 0.02, 'outside right');
  assert.ok(cov(-0.5, 0.5, w) < 0.02, 'outside left');
  const edge = cov(1.0, 0.5, w);
  assert.ok(edge > 0.35 && edge < 0.65, `edge ${edge}`);
});

// Ground truth (shared with the footprint tests): supersampled non-zero winding.
test('glyph coverage matches supersampled ground truth', () => {
  const font = wrapFont(loadOt('fonts/Tinos-Regular.ttf'));
  const gid = font.glyphIdForChar('e');
  const cmds = font.outlineEm(gid);
  const curves = preprocessGlyph(cmds);
  const bbox = bboxOfCommands(cmds);
  const sign = detectSign(curves, bbox);
  const edges = flattenCommands(cmds, 24);
  const w = 1/64;
  let err = 0, n = 0;
  for (let gy=0; gy<20; gy++) for (let gx=0; gx<20; gx++) {
    const cx = bbox.minX + (gx+0.5)/20*(bbox.maxX-bbox.minX);
    const cy = bbox.minY + (gy+0.5)/20*(bbox.maxY-bbox.minY);
    const a = coverage(curves, [w,w], [cx-w/2, cy-w/2], sign);
    const b = supersampledCoverage(edges, cx, cy, w, 6);
    err += Math.abs(a-b); n++;
  }
  const mae = err/n;
  assert.ok(mae < 0.05, `mean abs coverage error ${mae}`);
});

// A stroke's interior is solid, so every scanline crossing one has at least one
// fully covered pixel. A row that falls short while both neighbours are solid is
// the faint horizontal seam an inexact curve/box crossing leaves behind.
test('no faint seam rows at 500 px/em', () => {
  const font = wrapFont(loadOt('fonts/Tinos-Regular.ttf'));
  const ppem = 500, w = 1/ppem;
  for (const ch of ['S', 'g']) {
    const cmds = font.outlineEm(font.glyphIdForChar(ch));
    const curves = preprocessGlyph(cmds);
    const bbox = bboxOfCommands(cmds);
    const sign = detectSign(curves, bbox);
    const full = (px, py) => coverage(curves, [w,w], [px*w, py*w], sign) >= 0.9995;
    const x0 = Math.floor(bbox.minX*ppem), x1 = Math.ceil(bbox.maxX*ppem);
    const y0 = Math.floor(bbox.minY*ppem), y1 = Math.ceil(bbox.maxY*ppem);
    const solid = [];
    let hint = null;                      // stems are vertically coherent: retry last hit first
    for (let py = y0; py <= y1; py++) {
      let ok = hint !== null && full(hint, py);
      if (!ok) for (let px = x0; px <= x1; px++) if (full(px, py)) { hint = px; ok = true; break; }
      solid.push(ok);
    }
    for (let i = 1; i < solid.length-1; i++)
      assert.ok(solid[i] || !solid[i-1] || !solid[i+1],
        `'${ch}': row y=${y0+i} never reaches full coverage, but the rows above and below do`);
  }
});
