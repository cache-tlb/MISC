import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateSDF, generateMSDF } from '../js/sdf-generator.js';

const square = [
  {type:'M',x:0.2,y:0.2},{type:'L',x:0.8,y:0.2},{type:'L',x:0.8,y:0.8},{type:'L',x:0.2,y:0.8},{type:'Z'}
];
const median = (r,g,b) => Math.max(Math.min(r,g),Math.min(Math.max(r,g),b));

test('SDF sign: center >0.5 (inside), corner texel <0.5 (outside)', () => {
  const { size, data } = generateSDF(square, { res: 32, pad: 4 });
  const c = Math.floor(size/2);
  const center = data[c*size + c] / 255;
  const cornerTexel = data[0] / 255;
  assert.ok(center > 0.5, `center ${center}`);
  assert.ok(cornerTexel < 0.5, `corner ${cornerTexel}`);
});

test('MSDF median sign matches SDF sign', () => {
  const s = generateSDF(square, { res: 32, pad: 4 });
  const m = generateMSDF(square, { res: 32, pad: 4 });
  assert.equal(s.size, m.size);
  let agree = 0, total = 0;
  for (let i=0;i<s.size*s.size;i++){
    const sInside = s.data[i]/255 > 0.5;
    const r=m.data[i*3]/255, g=m.data[i*3+1]/255, b=m.data[i*3+2]/255;
    const mInside = median(r,g,b) > 0.5;
    if (sInside === mInside) agree++;
    total++;
  }
  assert.ok(agree/total > 0.97, `agreement ${agree/total}`);
});

const texelAt = (tile, u, v) => [Math.round((u-tile.minX)*tile.scale), Math.round((v-tile.minY)*tile.scale)];
const spreadAt = (m, tx, ty) => {
  const i = (ty*m.size+tx)*3, r=m.data[i]/255, g=m.data[i+1]/255, b=m.data[i+2]/255;
  return Math.max(r,g,b) - Math.min(r,g,b);
};

test('MSDF carries multi-edge info: channels diverge at corners, agree in flat regions', () => {
  // The sharp-corner benefit comes from each channel storing a different edge's
  // distance. Deep interior/exterior => channels agree (like SDF); near a corner
  // => channels diverge, which single-channel SDF cannot represent.
  const m = generateMSDF(square, { res: 32, pad: 4 });

  // deep interior at glyph center: channels agree, median inside
  const [ix,iy] = texelAt(m.tile, 0.5, 0.5);
  const ii = (iy*m.size+ix)*3;
  const medIn = median(m.data[ii]/255, m.data[ii+1]/255, m.data[ii+2]/255);
  assert.ok(spreadAt(m, ix, iy) < 0.05, `interior spread ${spreadAt(m,ix,iy)} should be ~0`);
  assert.ok(medIn > 0.5, `interior median ${medIn} should be inside`);

  // near the (0.8,0.8) corner: at least one nearby texel shows large channel spread
  const [cx,cy] = texelAt(m.tile, 0.8, 0.8);
  let maxSpread = 0;
  for (let dy=-2;dy<=2;dy++) for (let dx=-2;dx<=2;dx++) maxSpread = Math.max(maxSpread, spreadAt(m, cx+dx, cy+dy));
  assert.ok(maxSpread > 0.2, `corner channel spread ${maxSpread} should be large (multi-channel)`);
});
