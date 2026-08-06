import { identity, multiply, perspective, lookAt, transform } from './mat4.js';

const DEG = Math.PI/180;

// yaw/pitch in degrees. yaw=pitch=0 is face-on.
//
// `dist` is NOT world units — it is a multiple of the distance that just frames
// the text block. Absolute distances would need retuning for every text length
// and font size; a framing multiple gives presets that hold for any content, and
// leaves the wheel a well-behaved 1.0-is-tight range to dolly through.
//
// `grazing` is tuned so the anisotropy ratio at screen centre lands in 5-10:
// enough to make the single-window AABB visibly blur, not so much that the text
// becomes unreadable.
export const PRESETS = {
  front:   { yaw: 0,  pitch: 0,  dist: 1.15 },
  angled:  { yaw: 45, pitch: 20, dist: 1.30 },
  grazing: { yaw: 80, pitch: 12, dist: 1.35 },
};

export const clampPitch = (p) => Math.min(85, Math.max(-85, p));
export const clampDist  = (d) => Math.min(20, Math.max(0.3, d));

export function orbitEye({ yaw, pitch, dist }) {
  const y = yaw*DEG, p = pitch*DEG;
  return [dist*Math.sin(y)*Math.cos(p), dist*Math.sin(p), dist*Math.cos(y)*Math.cos(p)];
}

// em -> plane-local world: centre the text block on the origin, scale to size.
export function planeModel(centerEm, worldPerEm) {
  const m = identity();
  m[0] = worldPerEm; m[5] = worldPerEm; m[10] = worldPerEm;
  m[12] = -centerEm[0]*worldPerEm;
  m[13] = -centerEm[1]*worldPerEm;
  return m;
}

// Distance at which the text block exactly fills the frame (em maps 1:1 to world,
// so this depends only on the block's em extent, the fov and the aspect).
export function fitDistance(bounds, fovDeg, aspect) {
  const hh = Math.max(1e-4, (bounds.maxY - bounds.minY)/2);
  const hw = Math.max(1e-4, (bounds.maxX - bounds.minX)/2);
  const t = Math.tan(fovDeg*DEG/2);
  return Math.max(hh/t, hw/(t*aspect));
}

export function mvp3d(cam, bounds, aspect) {
  const centerEm = [(bounds.minX+bounds.maxX)/2, (bounds.minY+bounds.maxY)/2];
  const d = Math.max(1e-3, cam.dist * fitDistance(bounds, cam.fov, aspect));
  const proj = perspective(cam.fov*DEG, aspect, d*0.01, d*100);
  const view = lookAt(orbitEye({ yaw: cam.yaw, pitch: cam.pitch, dist: d }), [0,0,0], [0,1,0]);
  return multiply(multiply(proj, view), planeModel(centerEm, 1));
}

function projectToPx(mvp, em, backing) {
  const v = transform(mvp, [em[0], em[1], 0, 1]);
  if (!(Math.abs(v[3]) > 1e-9)) return null;
  return [ (v[0]/v[3]*0.5 + 0.5)*backing[0], (0.5 - v[1]/v[3]*0.5)*backing[1] ];
}

// CPU mirror of the GPU's dFdx/dFdy(localEm): finite-difference the em->pixel
// map, then invert it to get the em displacement per one screen pixel. Used by
// the info readout to report the anisotropy the shader is actually seeing.
export function emPerPixelAxes(mvp, em, backing, h = 1e-3) {
  const p0 = projectToPx(mvp, em, backing);
  const px = projectToPx(mvp, [em[0]+h, em[1]], backing);
  const py = projectToPx(mvp, [em[0], em[1]+h], backing);
  if (!p0 || !px || !py) return null;
  const a = (px[0]-p0[0])/h, c = (px[1]-p0[1])/h;   // d(screen)/d(em.x)
  const b = (py[0]-p0[0])/h, d = (py[1]-p0[1])/h;   // d(screen)/d(em.y)
  const det = a*d - b*c;
  if (Math.abs(det) < 1e-12) return null;
  return { ddx: [ d/det, -c/det], ddy: [-b/det,  a/det] };   // columns of J^-1
}
