import { evalQuad, evalCubic } from './geom.js';

const EPS = 1e-7;
const lerp = (a, b, t) => [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t];

export function isMonotonic(q) {
  for (const ax of [0,1]) {
    const a=q.p0[ax], b=q.p1[ax], c=q.p2[ax];
    const up = (a<=b+EPS && b<=c+EPS), dn = (a>=b-EPS && b>=c-EPS);
    if (!up && !dn) return false;
  }
  return true;
}

// Parameter where the derivative of one axis is zero, if strictly inside (0,1).
function criticalT(a, b, c) {
  const denom = a - 2*b + c;
  if (Math.abs(denom) < EPS) return null;
  const t = (a - b) / denom;
  return (t > EPS && t < 1 - EPS) ? t : null;
}

function quadDeriv(q, t) {
  return [
    2*(1-t)*(q.p1[0]-q.p0[0]) + 2*t*(q.p2[0]-q.p1[0]),
    2*(1-t)*(q.p1[1]-q.p0[1]) + 2*t*(q.p2[1]-q.p1[1]),
  ];
}

// Control point of a sub-quad = intersection of the tangent lines at its ends.
function intersectTangents(A, dA, C, dC) {
  const cross = dA[0]*dC[1] - dA[1]*dC[0];
  if (Math.abs(cross) < 1e-9) return [(A[0]+C[0])/2, (A[1]+C[1])/2];
  const s = ((C[0]-A[0])*dC[1] - (C[1]-A[1])*dC[0]) / cross;
  return [A[0] + s*dA[0], A[1] + s*dA[1]];
}

function segmentOfQuad(q, t0, t1) {
  const A = evalQuad(q.p0, q.p1, q.p2, t0);
  const C = evalQuad(q.p0, q.p1, q.p2, t1);
  const B = intersectTangents(A, quadDeriv(q, t0), C, quadDeriv(q, t1));
  return { p0: A, p1: B, p2: C };
}

function resplit(q, tsGlobal) {
  const cuts = [0, ...tsGlobal, 1];
  const out = [];
  for (let i=0;i<cuts.length-1;i++) out.push(segmentOfQuad(q, cuts[i], cuts[i+1]));
  return out;
}

export function splitMonotonic(q) {
  const ts = [];
  for (const ax of [0,1]) {
    const t = criticalT(q.p0[ax], q.p1[ax], q.p2[ax]);
    if (t !== null) ts.push(t);
  }
  ts.sort((a,b)=>a-b);
  return ts.length ? resplit(q, ts) : [q];
}

export function cubicToQuads(p0, c1, c2, p3, eps = 1e-3, depth = 0) {
  const ctrl = [(3*c1[0]-p0[0]+3*c2[0]-p3[0])/4, (3*c1[1]-p0[1]+3*c2[1]-p3[1])/4];
  const cubMid = evalCubic(p0,c1,c2,p3,0.5), quadMid = evalQuad(p0,ctrl,p3,0.5);
  const err = Math.hypot(cubMid[0]-quadMid[0], cubMid[1]-quadMid[1]);
  if (err <= eps || depth >= 6) return [{ p0, p1: ctrl, p2: p3 }];
  const m01=lerp(p0,c1,.5), m12=lerp(c1,c2,.5), m23=lerp(c2,p3,.5);
  const m012=lerp(m01,m12,.5), m123=lerp(m12,m23,.5), mid=lerp(m012,m123,.5);
  return [
    ...cubicToQuads(p0,m01,m012,mid, eps, depth+1),
    ...cubicToQuads(mid,m123,m23,p3, eps, depth+1),
  ];
}

export function preprocessGlyph(commands, eps = 1e-3) {
  const raw = [];
  let start = null, cur = null;
  const pushLine = (a, b) => {
    if (Math.abs(a[1]-b[1]) < EPS) return;                       // drop horizontal
    raw.push({ p0:a, p1:[(a[0]+b[0])/2,(a[1]+b[1])/2], p2:b });  // promote to quad
  };
  for (const c of commands) {
    if (c.type === 'M') { start=[c.x,c.y]; cur=start; }
    else if (c.type === 'L') { pushLine(cur,[c.x,c.y]); cur=[c.x,c.y]; }
    else if (c.type === 'Q') { raw.push({ p0:cur, p1:[c.x1,c.y1], p2:[c.x,c.y] }); cur=[c.x,c.y]; }
    else if (c.type === 'C') { for (const q of cubicToQuads(cur,[c.x1,c.y1],[c.x2,c.y2],[c.x,c.y],eps)) raw.push(q); cur=[c.x,c.y]; }
    else if (c.type === 'Z') { if (start) pushLine(cur,start); cur=start; }
  }
  const out = [];
  for (const q of raw)
    for (const m of splitMonotonic(q))
      if (Math.abs(m.p0[1]-m.p2[1]) >= EPS) out.push(m);       // drop degenerate horizontal pieces
  return out;
}
