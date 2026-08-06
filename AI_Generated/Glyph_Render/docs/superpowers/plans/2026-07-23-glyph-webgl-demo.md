# Glyph Rendering — WebGL2 Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the interactive single-page WebGL2 demo that renders text three ways — Scanline Sweeper, single-channel SDF, MSDF — and compares them via a draggable split-screen wipe, consuming the Plan 1 core library.

**Architecture:** `index.html` hosts a canvas + control panel and loads ES modules. `js/gl-utils.js` wraps WebGL2 boilerplate. `js/sweeper-renderer.js` packs monotonic curves into an `RGBA32F` data texture and runs the ported sweep shader; `js/sdf-renderer.js` packs SDF/MSDF tiles into an atlas and samples them. Each renderer issues exactly one `drawArraysInstanced` call for all glyphs. `js/main.js` wires UI, rebuilds textures only when the character set changes, and drives the split-wipe with `gl.scissor`.

**Tech Stack:** WebGL2, GLSL ES 3.00, ES modules, Plan 1 core (`font-loader`, `layout`, `sweeper-preprocess`, `sdf-generator`), vendored opentype.js.

## Global Constraints

- WebGL2 only. Use `drawArraysInstanced`, `texelFetch`, `RGBA32F` sampling (NEAREST). (spec §3, §7.2)
- All geometry em-space, y-up; y-flip happens in the vertex shader. Sweeper fill sign is baked at pack time (reverse curves when `detectSign < 0`), so the shader treats interior as positive. (spec §5.1)
- Interaction (pan/zoom/font-size/gamma/spread/slider) changes **uniforms only** — never rebuilds buffers/textures. Rebuild only on text/font/SDF-resolution change. (spec §6)
- Each algorithm renders in exactly **one** `drawArraysInstanced` call. Split-wipe uses `gl.scissor`, both sides under the same view transform. (spec §9)
- SDF sign: interior positive; decode `dEm=(median-0.5)*2*spreadEm`; AA half-pixel = `0.5*emPerPixel`. `emPerPixel` shared with the sweeper so both edges have the same transition scale. (spec §8.2, §8.6)
- No third-party runtime JS beyond vendored opentype.js.
- Verification is browser-visual (served over `http://`, not `file://`, so ES module imports work).

**View transform (shared by both renderers).** Uniforms:
- `u_pxPerEm` (float) = `fontSizePx * zoom * devicePixelRatio` — device px per em.
- `u_originDev` (vec2) = device-px position of em-origin `(0,0)` (top-left based) = `(marginCssX + panCssX)*dpr, (marginCssY + panCssY)*dpr`.
- `u_backing` (vec2) = canvas backing store size in device px.
- `u_emPerPixel` (float) = `1.0 / u_pxPerEm` — the sweeper window size and the SDF AA scale.

Vertex mapping (both shaders):
```glsl
vec2 worldEm = localEm + a_emOrigin;
vec2 dev = u_originDev + vec2(worldEm.x*u_pxPerEm, -worldEm.y*u_pxPerEm); // y-up -> y-down
gl_Position = vec4(dev.x/u_backing.x*2.0-1.0, 1.0 - dev.y/u_backing.y*2.0, 0.0, 1.0);
```

---

## File Structure

```
index.html                UI markup, control panel, canvas, module bootstrap, CSS
js/gl-utils.js            createGL2, compile, program, createFloatTexture, createByteTexture, quad
js/sweeper-renderer.js    SweeperRenderer: build() packs curve texture + instances; render()
js/sdf-renderer.js        SdfRenderer: build() packs SDF/MSDF atlas + instances; render(); setMode()
js/main.js                app state, UI wiring, rebuild triggers, render loop, split-wipe, interaction
```

Consumes Plan 1: `js/font-loader.js`, `js/layout.js`, `js/sweeper-preprocess.js`, `js/sweeper-core.js` (`detectSign`), `js/sdf-generator.js`, `js/geom.js` (`bboxOfCommands`).

---

### Task 9: `js/gl-utils.js` + WebGL2 bootstrap smoke test

**Files:**
- Create: `js/gl-utils.js`, `index.html`

**Interfaces:**
- Produces:
  - `createGL2(canvas) -> WebGL2RenderingContext` (throws if unavailable)
  - `compile(gl, type, src) -> WebGLShader`
  - `createProgram(gl, vsSrc, fsSrc) -> WebGLProgram` (logs + throws on link error)
  - `createFloatTexture(gl, w, h, float32data) -> WebGLTexture` (RGBA32F, NEAREST)
  - `createByteTexture(gl, w, h, uint8data, {channels, filter}) -> WebGLTexture` (R8/RGB8/RGBA8)
  - `makeQuadBuffer(gl) -> WebGLBuffer` (4 verts triangle strip `(0,0)(1,0)(0,1)(1,1)`)
  - `resizeCanvasToDisplay(canvas) -> {changed, w, h, dpr}`

- [ ] **Step 1: Implement `js/gl-utils.js`**

```javascript
export function createGL2(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false, alpha: true });
  if (!gl) throw new Error('WebGL2 not available in this browser.');
  return gl;
}

export function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
    throw new Error('Shader compile error:\n' + gl.getShaderInfoLog(sh) + '\n---\n' + src);
  return sh;
}

export function createProgram(gl, vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error('Program link error:\n' + gl.getProgramInfoLog(p));
  return p;
}

export function createFloatTexture(gl, w, h, data) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
  return tex;
}

export function createByteTexture(gl, w, h, data, { channels = 4, filter = gl.LINEAR } = {}) {
  const map = { 1: [gl.R8, gl.RED], 3: [gl.RGB8, gl.RGB], 4: [gl.RGBA8, gl.RGBA] };
  const [internal, format] = map[channels];
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, gl.UNSIGNED_BYTE, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
  return tex;
}

export function makeQuadBuffer(gl) {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,0, 1,0, 0,1, 1,1]), gl.STATIC_DRAW);
  return buf;
}

export function resizeCanvasToDisplay(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  const changed = canvas.width !== w || canvas.height !== h;
  if (changed) { canvas.width = w; canvas.height = h; }
  return { changed, w, h, dpr };
}
```

- [ ] **Step 2: Create `index.html` with a WebGL2 smoke bootstrap**

Minimal first version — clears to a color and draws one quad to confirm the pipeline (replaced/extended in Task 12).

```html
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Scanline Sweeper vs SDF/MSDF</title>
<style>
  html,body{margin:0;height:100%;background:#111;color:#eee;font:14px system-ui,sans-serif}
  #stage{position:absolute;inset:0}
  #gl{width:100%;height:100%;display:block}
</style>
</head>
<body>
  <div id="stage"><canvas id="gl"></canvas></div>
  <script type="module">
    import { createGL2, createProgram, makeQuadBuffer, resizeCanvasToDisplay } from './js/gl-utils.js';
    const canvas = document.getElementById('gl');
    resizeCanvasToDisplay(canvas);
    const gl = createGL2(canvas);
    const prog = createProgram(gl,
      `#version 300 es
       in vec2 a_corner; void main(){ gl_Position = vec4(a_corner*1.6-0.8, 0.0, 1.0); }`,
      `#version 300 es
       precision highp float; out vec4 o; void main(){ o = vec4(0.2,0.7,1.0,1.0); }`);
    const quad = makeQuadBuffer(gl);
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    const loc = gl.getAttribLocation(prog, 'a_corner');
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.viewport(0,0,canvas.width,canvas.height);
    gl.clearColor(0.07,0.07,0.07,1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog); gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    console.log('WebGL2 smoke ok');
  </script>
</body>
</html>
```

- [ ] **Step 3: Serve and verify in a browser**

Run (from repo root): `python -m http.server 8080`
Open `http://localhost:8080/` in Chrome/Edge/Firefox.
Expected: dark canvas with a light-blue square in the center; console prints `WebGL2 smoke ok` and no errors. (Must be `http://`, not opening the file directly — ES modules require a server.)

- [ ] **Step 4: Commit**

```bash
git add js/gl-utils.js index.html
git commit -m "feat(demo): gl-utils helpers + WebGL2 bootstrap smoke"
```

---

### Task 10: `js/sweeper-renderer.js`

**Files:**
- Create: `js/sweeper-renderer.js`
- Modify: `index.html` (swap smoke for a sweeper render during this task's verification)

**Interfaces:**
- Consumes: `Font` (`outlineEm`), `preprocessGlyph`, `detectSign`, `bboxOfCommands`, gl-utils.
- Produces:
  - `class SweeperRenderer { constructor(gl); build(font, instances); render(view); stats }`
  - `instances`: `Instance[]` from `layoutText` (`{glyphId, ox, oy}`).
  - `view`: `{ pxPerEm, originDev:[x,y], backing:[w,h], emPerPixel, gamma, color:[r,g,b,a] }`.
  - `stats`: `{ curveCount, texW, texH, instanceCount }`.
  - Sign baked at pack time: if `detectSign(curves,bbox) < 0`, swap `p0<->p2` for each curve before packing (flips swept-area sign → interior positive).
  - Instance layout (Float32, stride 8): `[ox, oy, quadMinX, quadMinY, quadMaxX, quadMaxY, curveStartTexel, curveCount]`, quad = glyph bbox ± `PAD_EM`.

- [ ] **Step 1: Implement `js/sweeper-renderer.js`**

```javascript
import { preprocessGlyph } from './sweeper-preprocess.js';
import { detectSign } from './sweeper-core.js';
import { bboxOfCommands } from './geom.js';
import { createProgram, createFloatTexture, makeQuadBuffer } from './gl-utils.js';

const PAD_EM = 0.05;
const TEX_W = 1024;

const VS = `#version 300 es
in vec2 a_corner; in vec2 a_emOrigin; in vec2 a_quadMin; in vec2 a_quadMax;
in float a_curveStart; in float a_curveCount;
uniform float u_pxPerEm; uniform vec2 u_originDev; uniform vec2 u_backing;
out vec2 v_localEm; flat out float v_curveStart; flat out float v_curveCount;
void main(){
  vec2 localEm = mix(a_quadMin, a_quadMax, a_corner);
  v_localEm = localEm; v_curveStart = a_curveStart; v_curveCount = a_curveCount;
  vec2 worldEm = localEm + a_emOrigin;
  vec2 dev = u_originDev + vec2(worldEm.x*u_pxPerEm, -worldEm.y*u_pxPerEm);
  gl_Position = vec4(dev.x/u_backing.x*2.0-1.0, 1.0 - dev.y/u_backing.y*2.0, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D u_curveTex; uniform int u_curveTexW;
uniform float u_emPerPixel; uniform float u_gamma; uniform vec4 u_color;
in vec2 v_localEm; flat in float v_curveStart; flat in float v_curveCount;
out vec4 fragColor;
vec4 fetch(int idx){ return texelFetch(u_curveTex, ivec2(idx % u_curveTexW, idx / u_curveTexW), 0); }
float im(float qa,float c0,float c1,float c2,float target){
  if (abs(qa)<1e-3) return (target-c0)/(c2-c0);
  float qb=2.0*c1-2.0*c0, qc=c0-target, d=qb*qb-4.0*qa*qc;
  float sd=d<0.0?0.0:sqrt(d), inv2a=0.5/qa;
  return -qb*inv2a + sign(c2-c0)*sd*inv2a;
}
vec2 eb(vec2 p0,vec2 p1,vec2 p2,float t){ return mix(mix(p0,p1,t), mix(p1,p2,t), t); }
float sweep(vec2 size, vec2 offset, vec2 P0, vec2 P1, vec2 P2){
  if (max(P0.y,P2.y)<=offset.y || min(P0.y,P2.y)>=offset.y+size.y) return 0.0;
  vec2 delta=P2-P0; vec2 p0=P0-offset, p1=P1-offset, p2=P2-offset;
  if (p0.x==p1.x && p0.x==p2.x){
    if (p0.x>=size.x) return 0.0;
    float top=min(max(p0.y,p2.y),size.y), bottom=max(min(p0.y,p2.y),0.0);
    return sign(delta.y)*min(size.x,size.x-p0.x)*(top-bottom);
  }
  float qa=p0.y+p2.y-2.0*p1.y;
  float bt=im(qa,p0.y,p1.y,p2.y,0.0), tt=im(qa,p0.y,p1.y,p2.y,size.y);
  float vMinT=delta.y>0.0?bt:tt, vMaxT=delta.y>0.0?tt:bt;
  vec2 vMin=eb(p0,p1,p2,clamp(vMinT,0.0,1.0)), vMax=eb(p0,p1,p2,clamp(vMaxT,0.0,1.0));
  if (max(vMin.x,vMax.x)<=0.0) return (vMax.y-vMin.y)*size.x;
  if (min(vMin.x,vMax.x)>=size.x) return 0.0;
  qa=p0.x+p2.x-2.0*p1.x; float hMinT,hMaxT;
  vec4 hc = delta.x>0.0 ? vec4(p0.x,p2.x,0.0,0.0) : vec4(p2.x,p0.x,size.x,1.0);
  if (hc.x>=hc.z) hMinT=hc.w; else if (hc.y<=hc.z) hMinT=1.0-hc.w; else hMinT=im(qa,p0.x,p1.x,p2.x,hc.z);
  hc.z=size.x-hc.z;
  if (hc.x>=hc.z) hMaxT=hc.w; else if (hc.y<=hc.z) hMaxT=1.0-hc.w; else hMaxT=im(qa,p0.x,p1.x,p2.x,hc.z);
  float minT=clamp(max(vMinT,hMinT),0.0,1.0), maxT=clamp(min(vMaxT,hMaxT),0.0,1.0);
  vec2 q0 = vMinT>=hMinT ? vMin : eb(p0,p1,p2,minT);
  vec2 q1 = vMaxT<=hMaxT ? vMax : eb(p0,p1,p2,maxT);
  float cvg=0.0;
  if (minT>0.0 && delta.x>0.0){ float h=delta.y>0.0? q0.y-max(0.0,p0.y):min(size.y,p0.y)-q0.y; cvg=sign(delta.y)*h*size.x; }
  if (maxT<1.0 && delta.x<0.0){ float h=delta.y>0.0? min(size.y,p2.y)-q1.y:q1.y-max(0.0,p2.y); cvg+=sign(delta.y)*h*size.x; }
  cvg += (size.x-0.5*(q0.x+q1.x))*(q1.y-q0.y);
  return cvg;
}
void main(){
  vec2 size=vec2(u_emPerPixel); vec2 offset=v_localEm-0.5*size;
  int start=int(v_curveStart+0.5), count=int(v_curveCount+0.5);
  float area=0.0;
  for (int i=0;i<4096;i++){ if (i>=count) break;
    vec4 t0=fetch(start+i*2); vec4 t1=fetch(start+i*2+1);
    area += sweep(size, offset, t0.xy, t0.zw, t1.xy); }
  float cov=clamp(area/(size.x*size.y),0.0,1.0);
  cov=pow(cov, 1.0/u_gamma);
  fragColor=vec4(u_color.rgb, u_color.a*cov);
}`;

export class SweeperRenderer {
  constructor(gl) {
    this.gl = gl;
    this.prog = createProgram(gl, VS, FS);
    this.quad = makeQuadBuffer(gl);
    this.vao = gl.createVertexArray();
    this.instBuf = gl.createBuffer();
    this.u = {};
    for (const n of ['u_pxPerEm','u_originDev','u_backing','u_curveTex','u_curveTexW','u_emPerPixel','u_gamma','u_color'])
      this.u[n] = gl.getUniformLocation(this.prog, n);
    this.stats = { curveCount:0, texW:TEX_W, texH:0, instanceCount:0 };
  }

  build(font, instances) {
    const gl = this.gl;
    // preprocess unique glyphs -> curves (sign baked), record meta
    const meta = new Map(); // gid -> {start, count, bbox}
    const packedCurves = []; // flat: per curve p0x,p0y,p1x,p1y,p2x,p2y
    for (const gid of new Set(instances.map(i => i.glyphId))) {
      const cmds = font.outlineEm(gid);
      let curves = preprocessGlyph(cmds);
      const bbox = bboxOfCommands(cmds);
      if (curves.length && detectSign(curves, bbox) < 0)
        curves = curves.map(c => ({ p0: c.p2, p1: c.p1, p2: c.p0 }));
      const start = packedCurves.length / 6;
      for (const c of curves) packedCurves.push(c.p0[0],c.p0[1], c.p1[0],c.p1[1], c.p2[0],c.p2[1]);
      meta.set(gid, { start, count: curves.length, bbox });
    }
    // curve texture: 2 texels/curve, RGBA32F
    const totalCurves = packedCurves.length / 6;
    const texels = totalCurves * 2;
    const texH = Math.max(1, Math.ceil(texels / TEX_W));
    const texData = new Float32Array(TEX_W * texH * 4);
    for (let c = 0; c < totalCurves; c++) {
      const s = c*6, t = c*2*4;
      texData[t+0]=packedCurves[s+0]; texData[t+1]=packedCurves[s+1];
      texData[t+2]=packedCurves[s+2]; texData[t+3]=packedCurves[s+3];
      texData[t+4]=packedCurves[s+4]; texData[t+5]=packedCurves[s+5];
      texData[t+6]=0; texData[t+7]=0;
    }
    if (this.curveTex) gl.deleteTexture(this.curveTex);
    this.curveTex = createFloatTexture(gl, TEX_W, texH, texData);

    // instance buffer
    const inst = new Float32Array(instances.length * 8);
    let n = 0;
    for (const it of instances) {
      const m = meta.get(it.glyphId); if (!m) continue;
      const b = m.bbox;
      inst[n*8+0]=it.ox; inst[n*8+1]=it.oy;
      inst[n*8+2]=b.minX-PAD_EM; inst[n*8+3]=b.minY-PAD_EM;
      inst[n*8+4]=b.maxX+PAD_EM; inst[n*8+5]=b.maxY+PAD_EM;
      inst[n*8+6]=m.start*2;     inst[n*8+7]=m.count;
      n++;
    }
    this.instanceCount = n;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, inst.subarray(0, n*8), gl.DYNAMIC_DRAW);

    this._setupVao();
    this.stats = { curveCount: totalCurves, texW: TEX_W, texH, instanceCount: n };
  }

  _setupVao() {
    const gl = this.gl, p = this.prog;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    const cLoc = gl.getAttribLocation(p, 'a_corner');
    gl.enableVertexAttribArray(cLoc); gl.vertexAttribPointer(cLoc, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(cLoc, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    const stride = 8*4;
    const set = (name, size, off) => {
      const l = gl.getAttribLocation(p, name);
      gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, stride, off);
      gl.vertexAttribDivisor(l, 1);
    };
    set('a_emOrigin', 2, 0);
    set('a_quadMin', 2, 8);
    set('a_quadMax', 2, 16);
    set('a_curveStart', 1, 24);
    set('a_curveCount', 1, 28);
    gl.bindVertexArray(null);
  }

  render(view) {
    const gl = this.gl;
    if (!this.instanceCount) return;
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.curveTex);
    gl.uniform1i(this.u.u_curveTex, 0);
    gl.uniform1i(this.u.u_curveTexW, TEX_W);
    gl.uniform1f(this.u.u_pxPerEm, view.pxPerEm);
    gl.uniform2f(this.u.u_originDev, view.originDev[0], view.originDev[1]);
    gl.uniform2f(this.u.u_backing, view.backing[0], view.backing[1]);
    gl.uniform1f(this.u.u_emPerPixel, view.emPerPixel);
    gl.uniform1f(this.u.u_gamma, view.gamma);
    gl.uniform4fv(this.u.u_color, view.color);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.instanceCount);
    gl.bindVertexArray(null);
  }
}
```

- [ ] **Step 2: Temporarily wire the sweeper into `index.html` for verification**

Replace the `<script type="module">` body with:

```javascript
import { createGL2, resizeCanvasToDisplay } from './js/gl-utils.js';
import { createRequire } from './js/_noop.js'; // not used in browser; see note
```

Note: in the browser we can't `require`. Use this browser bootstrap instead (full replacement of the script block):

```javascript
import { createGL2, resizeCanvasToDisplay } from './js/gl-utils.js';
import { wrapFont } from './js/font-loader.js';
import { layoutText } from './js/layout.js';
import { SweeperRenderer } from './js/sweeper-renderer.js';

const canvas = document.getElementById('gl');
const gl = createGL2(canvas);
const buf = await (await fetch('./fonts/Tinos-Regular.ttf')).arrayBuffer();
const font = wrapFont(window.opentype.parse(buf));
const instances = layoutText(font, 'Ravage 42');
const sweeper = new SweeperRenderer(gl);
sweeper.build(font, instances);

function frame() {
  const { w, h, dpr } = resizeCanvasToDisplay(canvas);
  gl.viewport(0,0,w,h);
  gl.clearColor(0.07,0.07,0.07,1); gl.clear(gl.COLOR_BUFFER_BIT);
  const fontSizePx = 160, zoom = 1;
  const pxPerEm = fontSizePx*zoom*dpr;
  const view = {
    pxPerEm, originDev:[40*dpr, h*0.6], backing:[w,h],
    emPerPixel:1/pxPerEm, gamma:1.0, color:[0.95,0.95,0.95,1.0]
  };
  sweeper.render(view);
  requestAnimationFrame(frame);
}
frame();
```

Add opentype as a global for the browser (before the module script) in `<head>`:

```html
<script src="./vendor/opentype.min.js"></script>
```

- [ ] **Step 3: Serve and verify**

Run: `python -m http.server 8080` → open `http://localhost:8080/`.
Expected: the text "Ravage 42" in crisp white on dark. Zoom the browser / raise `fontSizePx` in code → edges stay razor-sharp (no blur, no stair-steps). Curves (a, g, e) are smooth; counters (holes in a, e, g, 4) are correctly empty. If glyphs render as filled blocks or inverted (holes filled), the sign bake is wrong — verify `detectSign` probe; if a glyph's interior is hollow, its curves were reversed incorrectly (re-check the `< 0` branch).

- [ ] **Step 4: Commit**

```bash
git add js/sweeper-renderer.js index.html
git commit -m "feat(demo): scanline sweeper renderer (curve data texture + instanced sweep shader)"
```

---

### Task 11: `js/sdf-renderer.js`

**Files:**
- Create: `js/sdf-renderer.js`
- Modify: `index.html` (verification: render SDF/MSDF next to nothing yet — full canvas)

**Interfaces:**
- Consumes: `Font`, `generateSDF`, `generateMSDF`, gl-utils.
- Produces:
  - `class SdfRenderer { constructor(gl); build(font, instances, {res, mode}); setMode(mode); render(view); stats }`
  - `mode`: `'sdf' | 'msdf'`. `build` generates tiles for the current mode, packs a shelf atlas, builds instances. `setMode` rebuilds the atlas for the new mode (cached per mode+res via a key; rebuild if key changed).
  - Instance layout (Float32, stride 11): `[ox, oy, tileMinX, tileMinY, tileMaxX, tileMaxY, u0, v0, u1, v1, spreadEm]`.
  - `view` adds nothing new (uses shared view + its own `mode`/`spreadEm` per-instance).
  - `stats`: `{ atlasW, atlasH, tiles, mode, res }`.

- [ ] **Step 1: Implement `js/sdf-renderer.js`**

```javascript
import { generateSDF, generateMSDF } from './sdf-generator.js';
import { createProgram, createByteTexture, makeQuadBuffer } from './gl-utils.js';

const ATLAS_W = 2048;

const VS = `#version 300 es
in vec2 a_corner; in vec2 a_emOrigin; in vec2 a_tileMin; in vec2 a_tileMax;
in vec4 a_rect; in float a_spread;
uniform float u_pxPerEm; uniform vec2 u_originDev; uniform vec2 u_backing;
out vec2 v_uv; flat out float v_spread;
void main(){
  vec2 localEm = mix(a_tileMin, a_tileMax, a_corner);
  v_uv = mix(a_rect.xy, a_rect.zw, a_corner);  // corner.y up == v up (row0 = em bottom)
  v_spread = a_spread;
  vec2 worldEm = localEm + a_emOrigin;
  vec2 dev = u_originDev + vec2(worldEm.x*u_pxPerEm, -worldEm.y*u_pxPerEm);
  gl_Position = vec4(dev.x/u_backing.x*2.0-1.0, 1.0 - dev.y/u_backing.y*2.0, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D u_atlas; uniform int u_msdf; uniform float u_emPerPixel; uniform vec4 u_color;
in vec2 v_uv; flat in float v_spread; out vec4 fragColor;
void main(){
  vec3 s = texture(u_atlas, v_uv).rgb;
  float m = u_msdf==1 ? max(min(s.r,s.g), min(max(s.r,s.g), s.b)) : s.r;
  float dEm = (m-0.5)*2.0*v_spread;
  float aa = 0.5*u_emPerPixel;
  float cov = clamp(0.5 + dEm/(2.0*aa), 0.0, 1.0);
  fragColor = vec4(u_color.rgb, u_color.a*cov);
}`;

export class SdfRenderer {
  constructor(gl) {
    this.gl = gl;
    this.prog = createProgram(gl, VS, FS);
    this.quad = makeQuadBuffer(gl);
    this.vao = gl.createVertexArray();
    this.instBuf = gl.createBuffer();
    this.u = {};
    for (const n of ['u_pxPerEm','u_originDev','u_backing','u_atlas','u_msdf','u_emPerPixel','u_color'])
      this.u[n] = gl.getUniformLocation(this.prog, n);
    this._font = null; this._instances = null; this._key = '';
    this.stats = { atlasW:0, atlasH:0, tiles:0, mode:'sdf', res:32 };
  }

  build(font, instances, { res = 32, mode = 'sdf' } = {}) {
    this._font = font; this._instances = instances; this._res = res; this.mode = mode;
    this._rebuild();
  }
  setMode(mode) { if (mode !== this.mode) { this.mode = mode; this._rebuild(); } }
  setRes(res) { if (res !== this._res) { this._res = res; this._rebuild(); } }

  _rebuild() {
    const gl = this.gl, key = `${this.mode}|${this._res}|${this._instances.length}`;
    const gen = this.mode === 'msdf' ? generateMSDF : generateSDF;
    // generate per unique glyph
    const tiles = new Map(); // gid -> {size, data, spreadEm}
    for (const gid of new Set(this._instances.map(i => i.glyphId))) {
      const t = gen(this._font.outlineEm(gid), { res: this._res, pad: 4 });
      tiles.set(gid, t);
    }
    // shelf-pack into one atlas (RGBA8; SDF uses R, MSDF uses RGB)
    let x=0, y=0, rowH=0, atlasH=0;
    const place = new Map();
    for (const [gid, t] of tiles) {
      if (x + t.size > ATLAS_W) { x=0; y+=rowH; rowH=0; }
      place.set(gid, { x, y, size:t.size });
      x += t.size; rowH = Math.max(rowH, t.size); atlasH = y + rowH;
    }
    atlasH = Math.max(1, atlasH);
    const atlas = new Uint8Array(ATLAS_W * atlasH * 4);
    for (const [gid, t] of tiles) {
      const p = place.get(gid);
      for (let ty=0; ty<t.size; ty++) for (let tx=0; tx<t.size; tx++){
        const di = ((p.y+ty)*ATLAS_W + (p.x+tx))*4;
        if (this.mode === 'msdf') { const si=(ty*t.size+tx)*3; atlas[di]=t.data[si]; atlas[di+1]=t.data[si+1]; atlas[di+2]=t.data[si+2]; atlas[di+3]=255; }
        else { const v=t.data[ty*t.size+tx]; atlas[di]=v; atlas[di+1]=v; atlas[di+2]=v; atlas[di+3]=255; }
      }
    }
    if (this.atlasTex) gl.deleteTexture(this.atlasTex);
    this.atlasTex = createByteTexture(gl, ATLAS_W, atlasH, atlas, { channels:4, filter: gl.LINEAR });

    // instances
    const inst = new Float32Array(this._instances.length * 11);
    let n=0;
    for (const it of this._instances) {
      const t = tiles.get(it.glyphId), p = place.get(it.glyphId); if (!t) continue;
      const emSpan = t.size / t.tile.scale; // tile em width==height
      const u0=p.x/ATLAS_W, v0=p.y/atlasH, u1=(p.x+t.size)/ATLAS_W, v1=(p.y+t.size)/atlasH;
      const o=n*11;
      inst[o+0]=it.ox; inst[o+1]=it.oy;
      inst[o+2]=t.tile.minX; inst[o+3]=t.tile.minY;
      inst[o+4]=t.tile.minX+emSpan; inst[o+5]=t.tile.minY+emSpan;
      inst[o+6]=u0; inst[o+7]=v0; inst[o+8]=u1; inst[o+9]=v1;
      inst[o+10]=t.spreadEm;
      n++;
    }
    this.instanceCount=n;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, inst.subarray(0,n*11), gl.DYNAMIC_DRAW);
    this._setupVao();
    this._key=key;
    this.stats = { atlasW:ATLAS_W, atlasH, tiles:tiles.size, mode:this.mode, res:this._res };
  }

  _setupVao() {
    const gl=this.gl, p=this.prog;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    const cLoc=gl.getAttribLocation(p,'a_corner');
    gl.enableVertexAttribArray(cLoc); gl.vertexAttribPointer(cLoc,2,gl.FLOAT,false,0,0); gl.vertexAttribDivisor(cLoc,0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    const stride=11*4;
    const set=(name,size,off)=>{ const l=gl.getAttribLocation(p,name); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l,size,gl.FLOAT,false,stride,off); gl.vertexAttribDivisor(l,1); };
    set('a_emOrigin',2,0); set('a_tileMin',2,8); set('a_tileMax',2,16); set('a_rect',4,24); set('a_spread',1,40);
    gl.bindVertexArray(null);
  }

  render(view) {
    const gl=this.gl; if (!this.instanceCount) return;
    gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.atlasTex); gl.uniform1i(this.u.u_atlas,0);
    gl.uniform1i(this.u.u_msdf, this.mode==='msdf'?1:0);
    gl.uniform1f(this.u.u_pxPerEm, view.pxPerEm);
    gl.uniform2f(this.u.u_originDev, view.originDev[0], view.originDev[1]);
    gl.uniform2f(this.u.u_backing, view.backing[0], view.backing[1]);
    gl.uniform1f(this.u.u_emPerPixel, view.emPerPixel);
    gl.uniform4fv(this.u.u_color, view.color);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.instanceCount);
    gl.bindVertexArray(null);
  }
}
```

- [ ] **Step 2: Verify by swapping the renderer in `index.html`**

Temporarily replace `SweeperRenderer` usage with:

```javascript
import { SdfRenderer } from './js/sdf-renderer.js';
const sdf = new SdfRenderer(gl);
sdf.build(font, instances, { res: 32, mode: 'msdf' });
// in frame(): sdf.render(view);
```

- [ ] **Step 3: Serve and verify**

Run: `python -m http.server 8080` → `http://localhost:8080/`.
Expected: "Ravage 42" via MSDF — sharp corners on the '4', 'R', 'g'. Switch `mode:'sdf'` → corners visibly rounder, edges softer, especially at `res:16`. Lower `res` shows SDF degrading (blobby) while MSDF holds corners. If MSDF shows color fringing on edges, the atlas is being LINEAR-filtered correctly but the median decode is off — verify the `u_msdf` branch and that RGB were packed (not RGBA-swizzled).

- [ ] **Step 4: Commit**

```bash
git add js/sdf-renderer.js index.html
git commit -m "feat(demo): SDF + MSDF renderer (shelf atlas, median decode, AA)"
```

---

### Task 12: `js/main.js` + full UI + split-wipe + interaction

**Files:**
- Create: `js/main.js`
- Rewrite: `index.html` (final UI; bootstrap now just imports `./js/main.js`)

**Interfaces:**
- Consumes: both renderers, `wrapFont`, `layoutText`, gl-utils.
- Produces the app: control panel, split-wipe, pan/zoom, rebuild-on-change contract, info readout.
- State object: `{ font, fontName, text, fontSizePx, zoom, panCss:[x,y], gamma, sdfRes, sdfMode, sliderX(0..1), color }`.
- Rebuild triggers: text/font → `buildAll()`; sdfRes → `sdf.setRes()`; sdfMode → `sdf.setMode()`. Everything else → `requestRender()` (uniforms only).

- [ ] **Step 1: Rewrite `index.html` with the final UI**

```html
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Scanline Sweeper vs SDF/MSDF · 字形渲染对比</title>
<script src="./vendor/opentype.min.js"></script>
<style>
  :root{--panel:#1b1d22;--ink:#e9e9ec;--muted:#9aa0aa;--accent:#4aa3ff}
  html,body{margin:0;height:100%;background:#0e0f12;color:var(--ink);font:13px/1.5 system-ui,"Microsoft YaHei",sans-serif}
  #app{display:grid;grid-template-columns:300px 1fr;height:100%}
  #panel{background:var(--panel);padding:14px;overflow:auto;box-shadow:2px 0 8px #0007;z-index:2}
  #panel h1{font-size:14px;margin:0 0 10px}
  #panel label{display:block;margin:12px 0 4px;color:var(--muted)}
  #panel select,#panel input[type=range],#panel textarea,#panel input[type=file]{width:100%;box-sizing:border-box}
  #panel textarea{height:90px;background:#0e0f12;color:var(--ink);border:1px solid #333;border-radius:6px;padding:6px;resize:vertical}
  .row{display:flex;gap:8px;align-items:center}
  .seg{display:flex;border:1px solid #333;border-radius:6px;overflow:hidden}
  .seg button{flex:1;background:#0e0f12;color:var(--ink);border:0;padding:6px;cursor:pointer}
  .seg button.on{background:var(--accent);color:#012}
  #stage{position:relative;overflow:hidden}
  #gl{width:100%;height:100%;display:block;cursor:grab}
  #gl.drag{cursor:grabbing}
  #divider{position:absolute;top:0;bottom:0;width:2px;background:var(--accent);transform:translateX(-1px);cursor:ew-resize;z-index:3}
  #divider::after{content:"⇄";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--accent);color:#012;border-radius:10px;padding:2px 6px;font-size:12px}
  #labels{position:absolute;top:8px;left:0;right:0;display:flex;justify-content:space-between;padding:0 12px;pointer-events:none;color:var(--muted);z-index:3}
  #info{margin-top:14px;color:var(--muted);font-size:12px;white-space:pre-line;border-top:1px solid #2a2d33;padding-top:10px}
  .val{color:var(--ink)}
</style>
</head>
<body>
<div id="app">
  <div id="panel">
    <h1>字形渲染对比 · Scanline Sweeper vs SDF</h1>

    <label>字体 Font</label>
    <select id="font">
      <option value="fonts/Tinos-Regular.ttf">Tinos (≈Times New Roman)</option>
      <option value="fonts/Geist-Regular.ttf">Geist (Sans)</option>
      <option value="fonts/DancingScript.ttf">Dancing Script (脚本)</option>
      <option value="fonts/NotoSansSC-Subset.ttf">Noto Sans SC (中文子集)</option>
    </select>
    <label>上传字体 Upload .ttf/.otf</label>
    <input id="fontFile" type="file" accept=".ttf,.otf"/>

    <label>右侧 SDF 模式</label>
    <div class="seg" id="mode"><button data-m="sdf" class="on">Single SDF</button><button data-m="msdf">MSDF</button></div>

    <label>SDF 分辨率 <span class="val" id="resVal">32</span> px/em</label>
    <select id="res"><option>16</option><option>24</option><option selected>32</option><option>48</option><option>64</option></select>

    <label>字号 <span class="val" id="sizeVal">120</span> px</label>
    <input id="size" type="range" min="8" max="400" value="120"/>

    <label>Gamma (Sweeper) <span class="val" id="gammaVal">1.0</span></label>
    <input id="gamma" type="range" min="0.5" max="2.5" step="0.05" value="1.0"/>

    <label>文本 Text（多行 / CJK）</label>
    <textarea id="text">Ravage 42
Sphinx of black quartz
汉字 字形 渲染</textarea>

    <div class="row" style="margin-top:10px"><button id="reset">重置视图 Reset view</button></div>
    <div id="info"></div>
  </div>

  <div id="stage">
    <canvas id="gl"></canvas>
    <div id="labels"><span>◀ Scanline Sweeper</span><span id="rlabel">SDF ▶</span></div>
    <div id="divider"></div>
  </div>
</div>
<script type="module">import './js/main.js';</script>
</body>
</html>
```

- [ ] **Step 2: Implement `js/main.js`**

```javascript
import { createGL2, resizeCanvasToDisplay } from './gl-utils.js';
import { wrapFont } from './font-loader.js';
import { layoutText, uniqueGlyphIds } from './layout.js';
import { SweeperRenderer } from './sweeper-renderer.js';
import { SdfRenderer } from './sdf-renderer.js';

const $ = (id) => document.getElementById(id);
const canvas = $('gl'), stage = $('stage'), divider = $('divider');
const gl = createGL2(canvas);

const state = {
  font: null, fontName: '', text: $('text').value,
  fontSizePx: 120, zoom: 1, panCss: [40, 0], gamma: 1.0,
  sdfRes: 32, sdfMode: 'sdf', sliderX: 0.5, color: [0.96,0.96,0.97,1.0],
};
let sweeper, sdf, needRender = true;

async function loadFontFromUrl(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  state.font = wrapFont(window.opentype.parse(buf));
  state.fontName = url.split('/').pop();
  buildAll();
}
function loadFontFromFile(file) {
  const r = new FileReader();
  r.onload = () => { state.font = wrapFont(window.opentype.parse(r.result)); state.fontName = file.name; buildAll(); };
  r.readAsArrayBuffer(file);
}

function buildAll() {
  if (!state.font) return;
  const instances = layoutText(state.font, state.text);
  const t0 = performance.now();
  if (!sweeper) sweeper = new SweeperRenderer(gl);
  if (!sdf) sdf = new SdfRenderer(gl);
  sweeper.build(state.font, instances);
  sdf.build(state.font, instances, { res: state.sdfRes, mode: state.sdfMode });
  state.buildMs = (performance.now() - t0).toFixed(1);
  state.uniqueCount = uniqueGlyphIds(instances).length;
  updateInfo(); requestRender();
}

function view(dpr, w, h) {
  const pxPerEm = state.fontSizePx * state.zoom * dpr;
  // place first-line baseline ~60% down, left margin + pan
  const originDev = [ (state.panCss[0]) * dpr, (h*0.60/dpr + state.panCss[1]) * dpr ];
  return { pxPerEm, originDev, backing:[w,h], emPerPixel:1/pxPerEm, gamma:state.gamma, color:state.color };
}

function render() {
  const { w, h, dpr } = resizeCanvasToDisplay(canvas);
  gl.viewport(0,0,w,h);
  gl.disable(gl.SCISSOR_TEST);
  gl.clearColor(0.055,0.06,0.07,1); gl.clear(gl.COLOR_BUFFER_BIT);
  const v = view(dpr, w, h);
  const splitPx = Math.round(state.sliderX * w);
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(0, 0, splitPx, h);                 // left = sweeper
  if (sweeper) sweeper.render(v);
  gl.scissor(splitPx, 0, w - splitPx, h);       // right = sdf/msdf
  if (sdf) sdf.render(v);
  gl.disable(gl.SCISSOR_TEST);
  divider.style.left = (state.sliderX * 100) + '%';
  needRender = false;
}
function requestRender(){ if (!needRender){ needRender = true; requestAnimationFrame(render); } }

function updateInfo() {
  const s = sweeper?.stats, a = sdf?.stats;
  const bytesCurve = s ? s.texW*s.texH*16 : 0;
  const bytesAtlas = a ? a.atlasW*a.atlasH*4 : 0;
  $('info').textContent =
`字体 ${state.fontName}
唯一字形 ${state.uniqueCount ?? 0}
Sweeper 曲线 ${s?.curveCount ?? 0} · 纹理 ${s?.texW}×${s?.texH} (${(bytesCurve/1024).toFixed(0)} KiB)
SDF 图集 ${a?.atlasW}×${a?.atlasH} · ${a?.tiles} tiles (${(bytesAtlas/1024).toFixed(0)} KiB) · ${a?.mode}@${a?.res}
重建耗时 ${state.buildMs} ms`;
}

// ---------- UI wiring ----------
$('font').onchange = (e) => loadFontFromUrl(e.target.value);
$('fontFile').onchange = (e) => { if (e.target.files[0]) loadFontFromFile(e.target.files[0]); };
$('mode').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('mode').querySelectorAll('button').forEach(x=>x.classList.remove('on')); b.classList.add('on');
  state.sdfMode = b.dataset.m; $('rlabel').textContent = (state.sdfMode==='msdf'?'MSDF':'SDF')+' ▶';
  sdf?.setMode(state.sdfMode); updateInfo(); requestRender();
});
$('res').onchange = (e) => { state.sdfRes = +e.target.value; $('resVal').textContent = e.target.value; sdf?.setRes(state.sdfRes); updateInfo(); requestRender(); };
$('size').oninput = (e) => { state.fontSizePx = +e.target.value; $('sizeVal').textContent = e.target.value; requestRender(); };
$('gamma').oninput = (e) => { state.gamma = +e.target.value; $('gammaVal').textContent = tot(state.gamma); requestRender(); };
$('text').oninput = (e) => { state.text = e.target.value; buildAll(); };
$('reset').onclick = () => { state.zoom = 1; state.panCss = [40,0]; state.fontSizePx = 120; $('size').value=120; $('sizeVal').textContent='120'; requestRender(); };
const tot = (x)=>Number(x).toFixed(2);

// pan (drag on canvas, but not when starting on the divider)
let dragging=false, last=[0,0];
canvas.addEventListener('pointerdown', e=>{ dragging=true; last=[e.clientX,e.clientY]; canvas.classList.add('drag'); canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e=>{ if(!dragging) return; state.panCss=[state.panCss[0]+(e.clientX-last[0]), state.panCss[1]+(e.clientY-last[1])]; last=[e.clientX,e.clientY]; requestRender(); });
canvas.addEventListener('pointerup', e=>{ dragging=false; canvas.classList.remove('drag'); });
// zoom around cursor
canvas.addEventListener('wheel', e=>{ e.preventDefault();
  const r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const f=Math.exp(-e.deltaY*0.0015); const nz=Math.min(40,Math.max(0.1,state.zoom*f));
  // keep cursor point stable: pan adjust
  const k=nz/state.zoom; state.panCss=[mx-(mx-state.panCss[0])*k, my-(my-state.panCss[1])*k];
  state.zoom=nz; requestRender();
}, {passive:false});
// divider drag
divider.addEventListener('pointerdown', e=>{ e.stopPropagation(); const move=ev=>{ const r=stage.getBoundingClientRect(); state.sliderX=Math.min(1,Math.max(0,(ev.clientX-r.left)/r.width)); requestRender(); }; const up=()=>{ window.removeEventListener('pointermove',move); window.removeEventListener('pointerup',up); }; window.addEventListener('pointermove',move); window.addEventListener('pointerup',up); });
window.addEventListener('resize', requestRender);

// ---------- boot ----------
$('rlabel').textContent = 'SDF ▶';
loadFontFromUrl($('font').value);
```

- [ ] **Step 3: Serve and verify end-to-end**

Run: `python -m http.server 8080` → `http://localhost:8080/`.
Verify:
1. Text renders; left of the divider = Sweeper, right = SDF; glyphs align exactly across the seam.
2. Drag the divider ⇄ — the same glyphs are Sweeper (left) vs SDF (right).
3. Switch **MSDF** — right side corners sharpen; label updates.
4. Set **res=16** + **MSDF** vs **Single SDF** — SDF corners round/blobby, MSDF stays sharp, Sweeper (left) is crisp regardless.
5. **Zoom** (wheel) way in on a corner — Sweeper stays razor-sharp; SDF/MSDF eventually show atlas-resolution limits. Confirm the info panel's "重建耗时" does **not** change while panning/zooming/resizing (uniforms-only), only when editing text/font/res.
6. Type **CJK** (汉字) with Noto Sans SC subset selected — renders on both sides.
7. Upload a font — re-renders.

- [ ] **Step 4: Fix the `tot()` hoisting** (used before definition in `$('gamma')`)

Move `const tot = (x)=>Number(x).toFixed(2);` above the UI wiring block (right after `let sweeper, sdf...`). Re-verify gamma label updates.

- [ ] **Step 5: Commit**

```bash
git add js/main.js index.html
git commit -m "feat(demo): main app — UI, split-wipe, pan/zoom, rebuild contract, info readout"
```

---

### Task 13: README + final polish

**Files:**
- Create: `README.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Write `README.md`**

```markdown
# Scanline Sweeper vs SDF/MSDF — 字形渲染对比 Demo

Interactive WebGL2 demo comparing three glyph renderers on identical text via a
draggable split-screen wipe:

- **Scanline Sweeper** — analytic per-pixel coverage from monotonic quadratic
  Béziers in a data texture (implements *The Scanline Sweeper*, `reference/`).
- **Single-channel SDF** (Green 2007) and **MSDF** (Chlumský 2015) baselines.

## Run
No build step. Serve over HTTP (ES modules need a server):

    python -m http.server 8080
    # open http://localhost:8080/

## Controls
Font (built-in or upload) · SDF mode (Single/MSDF) · SDF resolution · font size ·
gamma · multiline text (Latin/CJK) · drag to pan · wheel to zoom · drag the ⇄ divider.

## Structure
`js/` core (font-loader, layout, sweeper-preprocess, sweeper-core, sdf-generator)
is dependency-light and unit-tested (`npm test`). `js/*-renderer.js` + `main.js`
are the WebGL2 layer. `vendor/opentype.min.js` parses fonts. See `docs/superpowers/`.

## Tests
    npm test        # Node core tests
    node test/dump.mjs fonts/Tinos-Regular.ttf A 48   # dump SDF/MSDF/curves to out/

## Limitations
No hinting; contour overlaps clamped; simple layout (advances+kerning, no shaping);
bundled CJK is a Simplified-Chinese subset (upload for JP/KR). See design spec.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with run/controls/structure/limitations"
```

---

## Self-Review (completed)

- **Spec coverage:** WebGL2 + instancing + texelFetch (T9,T10) · curve data texture (T10, spec §7.2) · sweep shader ported from §8 (T10) · SDF+MSDF atlas + median decode + AA (T11, spec §8.6) · one draw call per algorithm + scissor split-wipe (T10,T11,T12, spec §9) · rebuild-only-on-character-change contract (T12, spec §6) · full UI: font/upload, res, mode, size, gamma, pan/zoom, multiline CJK, info readout (T12, spec §10) · limitations documented (T13, spec §11).
- **Placeholder scan:** the Task 10 Step 2 shows a wrong first attempt (`_noop`) immediately corrected by the full browser bootstrap that follows — the bootstrap block is the one to use; no residual TODOs.
- **Type consistency:** `view` object shape identical across `SweeperRenderer.render`, `SdfRenderer.render`, and `main.view()`. `Instance{glyphId,ox,oy}` from Plan 1 `layoutText` consumed by both `build()`s. `generateSDF/MSDF` return `{size,data,spreadEm,tile:{minX,minY,scale,size,spreadEm}}` consumed by `SdfRenderer._rebuild`. `stats` fields match `updateInfo`. Fix in T12S4 removes the one `tot()` temporal-dead-zone bug.
- **Cross-plan:** Sweeper GLSL (T10) mirrors Plan 1 `sweeper-core.js`; both must stay in sync — noted in both plans.
```
