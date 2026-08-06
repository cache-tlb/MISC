import { test } from 'node:test';
import assert from 'node:assert/strict';
import { identity, multiply, transform, ortho2d, perspective, lookAt } from '../js/mat4.js';

const close = (a, b, eps=1e-6) => assert.ok(Math.abs(a-b) < eps, `${a} != ${b}`);

test('multiply has identity as its unit', () => {
  const m = perspective(1.0, 1.5, 0.1, 10);
  const l = multiply(identity(), m), r = multiply(m, identity());
  for (let i=0;i<16;i++) { close(l[i], m[i]); close(r[i], m[i]); }
});

test('multiply is associative and agrees with sequential transform', () => {
  const a = perspective(1.0, 1.5, 0.1, 10);
  const b = lookAt([0,0,3],[0,0,0],[0,1,0]);
  const c = ortho2d(100, [10,20], [640,480]);
  const lhs = transform(multiply(multiply(a,b),c), [0.3,-0.2,0,1]);
  const rhs = transform(a, transform(b, transform(c, [0.3,-0.2,0,1])));
  for (let i=0;i<4;i++) close(lhs[i], rhs[i]);
});

test('lookAt puts the eye at the origin looking down -z', () => {
  const v = transform(lookAt([0,0,5],[0,0,0],[0,1,0]), [0,0,5,1]);
  close(v[0],0); close(v[1],0); close(v[2],0);
  const front = transform(lookAt([0,0,5],[0,0,0],[0,1,0]), [0,0,0,1]);
  close(front[2], -5);                     // target sits 5 in front of the camera
});

test('perspective maps the near plane centre to ndc z = -1', () => {
  const p = perspective(Math.PI/3, 1, 0.1, 100);
  const v = transform(p, [0,0,-0.1,1]);
  close(v[3], 0.1);
  close(v[2]/v[3], -1);
});

// This is the regression lock for "2D must stay pixel-identical": the matrix has
// to reproduce the inline formula the shaders used before u_mvp existed.
test('ortho2d reproduces the original 2D vertex mapping', () => {
  const cases = [
    { p: 120,  o: [40, 300],   b: [1280, 720] },
    { p: 37.5, o: [-12, 55],   b: [800, 600] },
    { p: 960,  o: [512, 1024], b: [2048, 1536] },
  ];
  const ems = [[0,0],[1,0],[0,1],[-0.2,1.2],[3.7,-0.9]];
  for (const { p, o, b } of cases) {
    const m = ortho2d(p, o, b);
    for (const [ex, ey] of ems) {
      const dev = [o[0] + ex*p, o[1] - ey*p];
      const want = [dev[0]/b[0]*2-1, 1 - dev[1]/b[1]*2];
      const got = transform(m, [ex, ey, 0, 1]);
      close(got[3], 1);
      close(got[0], want[0]); close(got[1], want[1]);
    }
  }
});
