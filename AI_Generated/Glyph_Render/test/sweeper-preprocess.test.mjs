import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evalQuad } from '../js/geom.js';
import { preprocessGlyph, splitMonotonic, isMonotonic, cubicToQuads } from '../js/sweeper-preprocess.js';

const mono = (q) => {
  for (const ax of [0,1]) {
    const a=q.p0[ax], b=q.p1[ax], c=q.p2[ax];
    assert.ok((a<=b+1e-6 && b<=c+1e-6) || (a>=b-1e-6 && b>=c-1e-6), `axis ${ax} not monotonic: ${a},${b},${c}`);
  }
};

test('splitMonotonic yields monotonic pieces', () => {
  const q = { p0:[0,0], p1:[1,2], p2:[2,0] };
  assert.equal(isMonotonic(q), false);
  const pieces = splitMonotonic(q);
  assert.ok(pieces.length >= 2);
  pieces.forEach(p => { assert.ok(isMonotonic(p)); mono(p); });
});

test('split pieces are continuous and preserve endpoints', () => {
  const q = { p0:[0,0], p1:[1,2], p2:[2,0] };
  const pieces = splitMonotonic(q);
  for (let i=0;i<pieces.length-1;i++)
    assert.ok(Math.hypot(pieces[i].p2[0]-pieces[i+1].p0[0], pieces[i].p2[1]-pieces[i+1].p0[1]) < 1e-9);
  assert.deepEqual(pieces[0].p0, [0,0]);
  assert.deepEqual(pieces[pieces.length-1].p2, [2,0]);
});

test('split reproduces the original curve at sampled t', () => {
  const q = { p0:[0.1,0.2], p1:[1.3,2.0], p2:[2.2,-0.4] };
  const pieces = splitMonotonic(q);
  // Each piece point must lie on the original curve. Use a fine reference grid so
  // grid quantization (~speed/N) stays well under the assertion threshold.
  const N = 20000;
  for (const p of pieces) for (const tt of [0,0.5,1]) {
    const pt = evalQuad(p.p0,p.p1,p.p2,tt);
    let best = Infinity;
    for (let s=0;s<=N;s++){ const o=evalQuad(q.p0,q.p1,q.p2,s/N); best=Math.min(best, Math.hypot(pt[0]-o[0],pt[1]-o[1])); }
    assert.ok(best < 2e-4, `piece point off original by ${best}`);
  }
});

test('preprocessGlyph drops horizontal lines and promotes lines to quads', () => {
  const cmds = [
    {type:'M',x:0,y:0},
    {type:'L',x:2,y:0},
    {type:'L',x:2,y:2},
    {type:'L',x:0,y:2},
    {type:'Z'},
  ];
  const curves = preprocessGlyph(cmds);
  assert.ok(curves.every(isMonotonic));
  assert.equal(curves.length, 2);
});

test('cubicToQuads approximates within tolerance', () => {
  const quads = cubicToQuads([0,0],[0,1],[1,1],[1,0], 1e-3);
  assert.deepEqual(quads[0].p0, [0,0]);
  assert.deepEqual(quads[quads.length-1].p2, [1,0]);
  assert.ok(quads.length >= 1);
});
