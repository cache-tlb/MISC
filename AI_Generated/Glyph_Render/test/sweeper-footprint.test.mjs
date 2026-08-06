import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadOt, windingInside } from './_helpers.mjs';
import { wrapFont } from '../js/font-loader.js';
import { preprocessGlyph } from '../js/sweeper-preprocess.js';
import { bboxOfCommands, flattenCommands } from '../js/geom.js';
import { coverage, detectSign } from '../js/sweeper-core.js';
import { footprintWindows, anisotropy, coverageFootprint } from '../js/sweeper-footprint.js';

const close = (a,b,eps=1e-12) => assert.ok(Math.abs(a-b) < eps, `${a} != ${b}`);

// The 2D orthographic footprint: +1 screen px in x is +e em in x, +1 screen px
// in y is -e em in y (em is y-up). This must survive the new code path untouched.
const iso = (e) => ({ ddx: [e, 0], ddy: [0, -e] });

test('isotropic footprint collapses to a single fwidth-sized window', () => {
  const e = 1/64, { ddx, ddy } = iso(e);
  close(anisotropy(ddx, ddy), 1);
  const w = footprintWindows([0.3, 0.4], ddx, ddy, 8);
  assert.equal(w.length, 1, 'N must be 1 when the footprint is square');
  close(w[0].size[0], e); close(w[0].size[1], e);
  close(w[0].offset[0], 0.3 - e/2); close(w[0].offset[1], 0.4 - e/2);
});

test('isotropic coverageFootprint is bit-identical to coverage()', () => {
  const font = wrapFont(loadOt('fonts/Tinos-Regular.ttf'));
  const cmds = font.outlineEm(font.glyphIdForChar('e'));
  const curves = preprocessGlyph(cmds);
  const bbox = bboxOfCommands(cmds);
  const sign = detectSign(curves, bbox);
  const e = 1/64, { ddx, ddy } = iso(e);
  for (let j=0;j<12;j++) for (let i=0;i<12;i++) {
    const cx = bbox.minX + (i+0.5)/12*(bbox.maxX-bbox.minX);
    const cy = bbox.minY + (j+0.5)/12*(bbox.maxY-bbox.minY);
    const a = coverage(curves, [e,e], [cx-e/2, cy-e/2], sign);
    const b = coverageFootprint(curves, [cx,cy], ddx, ddy, sign, 8);
    assert.equal(a, b, `mismatch at (${cx},${cy})`);
  }
});

test('windows tile the parallelogram exactly', () => {
  const ddx = [0.05, 0.012], ddy = [-0.004, 0.011];
  const maxN = 8;
  const wins = footprintWindows([0.5, 0.5], ddx, ddy, maxN);
  const ratio = anisotropy(ddx, ddy);
  assert.equal(wins.length, Math.min(maxN, Math.ceil(ratio - 1e-3)));
  for (const w of wins) { close(w.size[0], wins[0].size[0]); close(w.size[1], wins[0].size[1]); }

  const N = wins.length;
  const ctr = wins.map(w => [w.offset[0] + w.size[0]/2, w.offset[1] + w.size[1]/2]);
  const major = Math.hypot(...ddx) >= Math.hypot(...ddy) ? ddx : ddy;
  for (let i=0;i<N;i++) {
    const k = (i + 0.5)/N - 0.5;
    close(ctr[i][0], 0.5 + major[0]*k);
    close(ctr[i][1], 0.5 + major[1]*k);
  }
  // the N sub-parallelograms partition the pixel parallelogram: areas sum to it
  const area = Math.abs(ddx[0]*ddy[1] - ddx[1]*ddy[0]);
  const minor = major === ddx ? ddy : ddx;
  const sub = Math.abs((major[0]/N)*minor[1] - (major[1]/N)*minor[0]);
  close(N*sub, area);
});

test('maxN caps the window count', () => {
  const ddx = [0.2, 0], ddy = [0, 0.004];    // ratio 50
  assert.equal(footprintWindows([0,0], ddx, ddy, 4).length, 4);
  assert.equal(footprintWindows([0,0], ddx, ddy, 1).length, 1);
});

// Ground truth for a *sheared* footprint: sample the true parallelogram, not a
// square. This is the claim under test — multi-window tracks the parallelogram,
// a single AABB window over-blurs because it samples ink the pixel never covers.
function parallelogramCoverage(edges, c, ddx, ddy, N=16) {
  let hit = 0;
  for (let j=0;j<N;j++) for (let i=0;i<N;i++) {
    const u = (i+0.5)/N - 0.5, v = (j+0.5)/N - 0.5;
    const px = c[0] + u*ddx[0] + v*ddy[0];
    const py = c[1] + u*ddx[1] + v*ddy[1];
    if (windingInside(edges, px, py)) hit++;
  }
  return hit/(N*N);
}

test('multi-window beats a single AABB window on an anisotropic footprint', () => {
  const font = wrapFont(loadOt('fonts/Tinos-Regular.ttf'));
  const cmds = font.outlineEm(font.glyphIdForChar('e'));
  const curves = preprocessGlyph(cmds);
  const bbox = bboxOfCommands(cmds);
  const sign = detectSign(curves, bbox);
  const edges = flattenCommands(cmds, 24);

  // A pixel seen at a glancing angle: stretched ~8x along a diagonal.
  const a = 1/48, th = Math.PI/5;
  const ddx = [8*a*Math.cos(th), 8*a*Math.sin(th)];
  const ddy = [-a*Math.sin(th), a*Math.cos(th)];
  assert.ok(anisotropy(ddx, ddy) > 7.5, 'test fixture must actually be anisotropic');

  let e1 = 0, e8 = 0, n = 0;
  for (let j=0;j<16;j++) for (let i=0;i<16;i++) {
    const cx = bbox.minX + (i+0.5)/16*(bbox.maxX-bbox.minX);
    const cy = bbox.minY + (j+0.5)/16*(bbox.maxY-bbox.minY);
    const truth = parallelogramCoverage(edges, [cx,cy], ddx, ddy, 16);
    e1 += Math.abs(coverageFootprint(curves, [cx,cy], ddx, ddy, sign, 1) - truth);
    e8 += Math.abs(coverageFootprint(curves, [cx,cy], ddx, ddy, sign, 8) - truth);
    n++;
  }
  const mae1 = e1/n, mae8 = e8/n;
  console.log(`   anisotropic MAE: single AABB window ${mae1.toFixed(4)} -> 8 windows ${mae8.toFixed(4)}`);
  assert.ok(mae8 < mae1 * 0.6, `multi-window ${mae8} vs single ${mae1} — expected a clear win`);
  assert.ok(mae8 < 0.06, `multi-window MAE ${mae8} too high`);
});
