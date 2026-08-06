import { evalQuad } from './geom.js';

export function evaluateBezier(p0, p1, p2, t) { return evalQuad(p0, p1, p2, t); }

// Port of intersect_monotonic (reference/out.txt:671), solving qa*t^2+qb*t+qc=0
// for where the curve crosses `target` on one axis.
//
// The reference divides by qa, so it guards small qa with a linear shortcut
// under an "adjustable" 1e-3 threshold. That threshold is absolute, and we sweep
// in em space where a pixel is only 1/ppem em wide — a shortcut taken at
// qa=2.4e-4 misplaces the crossing by up to qa/4 em, which is ~3% of a pixel at
// 500px and grows with size. Summed over a band that should be exactly one pixel
// tall, that leaves faint horizontal seams along near-linear curves.
//
// The algebraically equal form t = 2qc / (-qb - s*sqrt(d)) needs no threshold:
// monotonicity gives s = sign(c2-c0) = sign(qb), so the denominator's terms add
// instead of cancelling, and at qa=0 it reduces exactly to the linear formula.
export function intersectMonotonic(qa, c0, c1, c2, target) {
  const qb = 2*c1 - 2*c0;
  const qc = c0 - target;
  const d = qb*qb - 4*qa*qc;
  // No real crossing: the curve lies wholly one side of target, so report a t
  // off the matching end (callers clamp to [0,1]).
  if (d < 0) return (target < c0) === (c2 > c0) ? -1 : 2;
  const den = -qb - Math.sign(c2 - c0) * Math.sqrt(d);
  return den === 0 ? 0 : 2*qc / den;   // den==0 only when c0==c1==target
}

const sat = (x) => Math.min(1, Math.max(0, x));

// Port of scanline_sweep (reference/out.txt:730). size,offset,p0,p1,p2 are [x,y].
export function scanlineSweep(size, offset, P0, P1, P2) {
  let p0=[...P0], p1=[...P1], p2=[...P2];
  if (Math.max(p0[1],p2[1]) <= offset[1] || Math.min(p0[1],p2[1]) >= offset[1]+size[1]) return 0;
  const delta = [p2[0]-p0[0], p2[1]-p0[1]];
  p0=[p0[0]-offset[0],p0[1]-offset[1]]; p1=[p1[0]-offset[0],p1[1]-offset[1]]; p2=[p2[0]-offset[0],p2[1]-offset[1]];

  if (p0[0]===p1[0] && p0[0]===p2[0]) {           // vertical fast path
    if (p0[0] >= size[0]) return 0;
    const top = Math.min(Math.max(p0[1],p2[1]), size[1]);
    const bottom = Math.max(Math.min(p0[1],p2[1]), 0);
    const h = top - bottom;
    const b = Math.min(size[0], size[0]-p0[0]);
    return Math.sign(delta[1]) * b * h;
  }

  let qa = p0[1] + p2[1] - 2*p1[1];
  const bt = intersectMonotonic(qa, p0[1], p1[1], p2[1], 0);
  const tt = intersectMonotonic(qa, p0[1], p1[1], p2[1], size[1]);
  const vMinT = delta[1] > 0 ? bt : tt;
  const vMaxT = delta[1] > 0 ? tt : bt;
  const vMin = evaluateBezier(p0,p1,p2, sat(vMinT));
  const vMax = evaluateBezier(p0,p1,p2, sat(vMaxT));

  if (Math.max(vMin[0],vMax[0]) <= 0) return (vMax[1]-vMin[1]) * size[0];
  if (Math.min(vMin[0],vMax[0]) >= size[0]) return 0;

  qa = p0[0] + p2[0] - 2*p1[0];
  let hMinT, hMaxT;
  const hc = delta[0] > 0 ? [p0[0],p2[0],0,0] : [p2[0],p0[0],size[0],1];
  if (hc[0] >= hc[2]) hMinT = hc[3];
  else if (hc[1] <= hc[2]) hMinT = 1 - hc[3];
  else hMinT = intersectMonotonic(qa, p0[0], p1[0], p2[0], hc[2]);
  hc[2] = size[0] - hc[2];
  if (hc[0] >= hc[2]) hMaxT = hc[3];
  else if (hc[1] <= hc[2]) hMaxT = 1 - hc[3];
  else hMaxT = intersectMonotonic(qa, p0[0], p1[0], p2[0], hc[2]);

  const minT = sat(Math.max(vMinT, hMinT));
  const maxT = sat(Math.min(vMaxT, hMaxT));
  const q0 = vMinT >= hMinT ? vMin : evaluateBezier(p0,p1,p2,minT);
  const q1 = vMaxT <= hMaxT ? vMax : evaluateBezier(p0,p1,p2,maxT);

  let cvg = 0;
  if (minT > 0 && delta[0] > 0) {
    const h = delta[1] > 0 ? q0[1]-Math.max(0,p0[1]) : Math.min(size[1],p0[1])-q0[1];
    cvg = Math.sign(delta[1]) * h * size[0];
  }
  if (maxT < 1 && delta[0] < 0) {
    const h = delta[1] > 0 ? Math.min(size[1],p2[1])-q1[1] : q1[1]-Math.max(0,p2[1]);
    cvg += Math.sign(delta[1]) * h * size[0];
  }
  const h = q1[1] - q0[1];
  const b = size[0] - 0.5*(q0[0]+q1[0]);
  cvg += b * h;
  return cvg;
}

export function sumSweep(curves, size, offset) {
  let s = 0;
  for (const c of curves) s += scanlineSweep(size, offset, c.p0, c.p1, c.p2);
  return s;
}

export function coverage(curves, size, offset, sign) {
  return sat(sign * sumSweep(curves, size, offset) / (size[0]*size[1]));
}

// Grid-probe the bbox with a tiny window; the probe with the largest |raw sum|
// is confidently interior, so its sign is the interior-positive orientation.
export function detectSign(curves, bbox) {
  const w = Math.min(bbox.maxX-bbox.minX, bbox.maxY-bbox.minY) * 0.01 + 1e-4;
  const NX = 7, NY = 7;
  let bestAbs = 0, bestSign = 1;
  for (let j=0;j<NY;j++) for (let i=0;i<NX;i++) {
    const cx = bbox.minX + (i+0.5)/NX*(bbox.maxX-bbox.minX);
    const cy = bbox.minY + (j+0.5)/NY*(bbox.maxY-bbox.minY);
    const s = sumSweep(curves, [w,w], [cx-w/2, cy-w/2]);
    if (Math.abs(s) > bestAbs) { bestAbs = Math.abs(s); bestSign = s>=0 ? 1 : -1; }
  }
  return bestSign;
}
