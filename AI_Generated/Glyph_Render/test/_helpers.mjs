// Shared Node test loader. The vendored opentype build (browser dist) exposes
// `parse` but not `loadSync`, and the package is ESM ("type":"module"), so we
// require the CommonJS copy and parse a file buffer into an ArrayBuffer.
import { createRequire } from 'module';
import { readFileSync } from 'fs';

const require = createRequire(import.meta.url);
export const opentype = require('../vendor/opentype.cjs');

export function loadOt(path) {
  const b = readFileSync(path);
  return opentype.parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

// Ground-truth rasterizer for coverage tests: non-zero winding, point sampled.
// `edges` come from geom.js flattenCommands().
export function windingInside(edges, px, py) {
  let w = 0;
  for (const e of edges) {
    const [ax,ay]=e.a, [bx,by]=e.b;
    if ((ay <= py) !== (by <= py)) {
      const tx = ax + (py-ay)/(by-ay)*(bx-ax);
      if (tx > px) w += (by > ay) ? 1 : -1;
    }
  }
  return w !== 0;
}

// Coverage of the axis-aligned square window of side `w` centred at (cx,cy),
// by NxN stratified point samples.
export function supersampledCoverage(edges, cx, cy, w, N=4) {
  let hit = 0;
  for (let j=0;j<N;j++) for (let i=0;i<N;i++){
    const px = cx - w/2 + (i+0.5)/N*w, py = cy - w/2 + (j+0.5)/N*w;
    if (windingInside(edges, px, py)) hit++;
  }
  return hit/(N*N);
}
