// Column-major 4x4, laid out for uniformMatrix4fv(loc, false, m): m[col*4 + row].

export function identity() {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}

// a·b — so multiply(proj, view) applied to a column vector means proj·(view·v).
export function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) {
    let s = 0;
    for (let k=0;k<4;k++) s += a[k*4+r] * b[c*4+k];
    o[c*4+r] = s;
  }
  return o;
}

export function transform(m, v) {
  const o = [0,0,0,0];
  for (let r=0;r<4;r++)
    o[r] = m[r]*v[0] + m[4+r]*v[1] + m[8+r]*v[2] + m[12+r]*v[3];
  return o;
}

// The demo's original 2D mapping, as a matrix:
//   dev  = originDev + (em.x*pxPerEm, -em.y*pxPerEm)
//   clip = (dev.x/bw*2 - 1, 1 - dev.y/bh*2)
// The em y-flip and the device y-flip cancel, so m11 comes out positive.
export function ortho2d(pxPerEm, originDev, backing) {
  const [ox, oy] = originDev, [bw, bh] = backing;
  const m = identity();
  m[0]  = 2*pxPerEm/bw;
  m[5]  = 2*pxPerEm/bh;
  m[12] = 2*ox/bw - 1;
  m[13] = 1 - 2*oy/bh;
  return m;
}

export function perspective(fovYRad, aspect, near, far) {
  const f = 1/Math.tan(fovYRad/2), nf = 1/(near-far);
  const m = new Float32Array(16);
  m[0] = f/aspect; m[5] = f;
  m[10] = (far+near)*nf; m[11] = -1; m[14] = 2*far*near*nf;
  return m;
}

const sub   = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot   = (a,b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm  = (a) => { const l = Math.hypot(a[0],a[1],a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };

export function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));        // camera looks down -z
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x,eye), -dot(y,eye), -dot(z,eye), 1,
  ]);
}
