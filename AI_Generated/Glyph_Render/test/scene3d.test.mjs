import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ortho2d, transform } from '../js/mat4.js';
import { anisotropy } from '../js/sweeper-footprint.js';
import { PRESETS, clampPitch, clampDist, orbitEye, mvp3d, emPerPixelAxes } from '../js/scene3d.js';

const close = (a,b,eps=1e-6) => assert.ok(Math.abs(a-b) < eps, `${a} != ${b}`);

test('orbitEye: yaw=pitch=0 puts the camera on +z facing the plane', () => {
  const e = orbitEye({ yaw:0, pitch:0, dist:3 });
  close(e[0],0); close(e[1],0); close(e[2],3);
});

test('orbitEye preserves distance at any angle', () => {
  for (const [yaw,pitch] of [[0,0],[45,20],[80,12],[-130,-70]])
    close(Math.hypot(...orbitEye({ yaw, pitch, dist:2.5 })), 2.5);
});

test('clamps keep the camera out of the degenerate poses', () => {
  assert.equal(clampPitch(120), 85);
  assert.equal(clampPitch(-120), -85);
  assert.equal(clampDist(0), 0.3);
  assert.equal(clampDist(999), 20);
});

// Presets are framing multiples, not world units, so they must hold for any text
// extent: a wide block and a tall block both have to land inside the frame.
test('the framing distance fits the text block for any extent', () => {
  const aspect = 1280/720;
  for (const b of [{minX:-6,maxX:6,minY:-0.5,maxY:0.5},      // one long line
                   {minX:-0.4,maxX:0.4,minY:-4,maxY:4},      // narrow column
                   {minX:-1,maxX:1,minY:-1,maxY:1}]) {
    const m = mvp3d({ ...PRESETS.front, fov:50 }, b, aspect);
    for (const [ex,ey] of [[b.minX,b.minY],[b.maxX,b.minY],[b.minX,b.maxY],[b.maxX,b.maxY]]) {
      const v = transform(m, [ex, ey, 0, 1]);
      assert.ok(v[3] > 0, 'corner must be in front of the camera');
      assert.ok(Math.abs(v[0]/v[3]) <= 1 && Math.abs(v[1]/v[3]) <= 1,
        `corner (${ex},${ey}) fell outside the frame at ndc (${(v[0]/v[3]).toFixed(2)},${(v[1]/v[3]).toFixed(2)})`);
    }
  }
});

// The Jacobian probe is what the info readout reports and what the GPU's
// dFdx/dFdy compute; on the 2D matrix it has to return the very axes the old
// scalar uniform implied.
test('emPerPixelAxes on ortho2d recovers the orthographic footprint axes', () => {
  const pxPerEm = 120, backing = [1280, 720];
  const m = ortho2d(pxPerEm, [40, 400], backing);
  const { ddx, ddy } = emPerPixelAxes(m, [0.4, 0.6], backing);
  close(ddx[0], 1/pxPerEm); close(ddx[1], 0);
  close(ddy[0], 0);         close(ddy[1], -1/pxPerEm);
  close(anisotropy(ddx, ddy), 1);
});

test('a face-on 3D camera is near-isotropic, a grazing one is not', () => {
  const backing = [1280, 720], aspect = backing[0]/backing[1];
  const bounds = { minX:-2, maxX:2, minY:-0.6, maxY:0.6 };
  const probe = (cam) => {
    const ax = emPerPixelAxes(mvp3d({ ...cam, fov:50 }, bounds, aspect), [0,0], backing);
    return anisotropy(ax.ddx, ax.ddy);
  };
  const rFront = probe(PRESETS.front), rGraze = probe(PRESETS.grazing);
  assert.ok(rFront < 2.0, `front-on anisotropy ${rFront} should be near 1 (aspect aside)`);
  assert.ok(rGraze > 4.0, `grazing preset anisotropy ${rGraze} must justify multi-window`);
  console.log(`   anisotropy at screen centre: front ${rFront.toFixed(2)}, grazing ${rGraze.toFixed(2)}`);
});
