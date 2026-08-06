import { sumSweep } from './sweeper-core.js';

// GLSL needs a constant loop bound; keep this and MAX_WINDOWS in the sweeper
// fragment shader in step.
export const MAX_WINDOWS = 8;

const sat = (x) => Math.min(1, Math.max(0, x));

// A pixel's em-space footprint is the parallelogram centred at the fragment and
// spanned by ddx = dFdx(localEm), ddy = dFdy(localEm). Under an orthographic 2D
// view that parallelogram is a square and its AABB is exact; under perspective
// it shears, and the AABB can be several times its area — every bit of that
// excess is ink the pixel never actually covers, which is the blur you see at
// glancing angles.
//
// So slice the parallelogram into N slabs across its long axis and hand each
// slab's AABB to the sweep separately. The slabs tile the parallelogram exactly,
// and at N=1 the window is |ddx|+|ddy| — precisely fwidth, so the isotropic case
// costs nothing and behaves exactly as before.
//
// This is the paper's §4.4 footprint assembly. The sweep primitive itself
// (sweeper-core.js) is untouched: it still only ever answers "coverage of one
// rectangular window".
export function anisotropy(ddx, ddy) {
  const lx = Math.hypot(ddx[0], ddx[1]), ly = Math.hypot(ddy[0], ddy[1]);
  const lo = Math.min(lx, ly);
  return lo > 0 ? Math.max(lx, ly) / lo : 1;
}

export function footprintWindows(center, ddx, ddy, maxN = 1) {
  const lx = Math.hypot(ddx[0], ddx[1]), ly = Math.hypot(ddy[0], ddy[1]);
  const major = lx >= ly ? ddx : ddy;
  const minor = lx >= ly ? ddy : ddx;
  const cap = Math.max(1, Math.min(MAX_WINDOWS, Math.floor(maxN)));
  // -1e-3 keeps an isotropic footprint on exactly one window despite float noise.
  const N = Math.max(1, Math.min(cap, Math.ceil(anisotropy(ddx, ddy) - 1e-3)));
  const w = Math.abs(major[0])/N + Math.abs(minor[0]);
  const h = Math.abs(major[1])/N + Math.abs(minor[1]);
  const out = [];
  for (let i=0;i<N;i++) {
    const k = (i + 0.5)/N - 0.5;
    out.push({
      size: [w, h],
      offset: [center[0] + major[0]*k - 0.5*w, center[1] + major[1]*k - 0.5*h],
    });
  }
  return out;
}

// Clamping per window rather than after the sum preserves the existing overlap
// semantics: a self-intersecting contour saturates its own window instead of
// dragging the mean past 1.
export function coverageFootprint(curves, center, ddx, ddy, sign, maxN = 1) {
  const wins = footprintWindows(center, ddx, ddy, maxN);
  let acc = 0;
  for (const w of wins)
    acc += sat(sign * sumSweep(curves, w.size, w.offset) / (w.size[0]*w.size[1]));
  return acc / wins.length;
}
