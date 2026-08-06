import { preprocessGlyph } from './sweeper-preprocess.js';
import { detectSign } from './sweeper-core.js';
import { bboxOfCommands } from './geom.js';
import { createProgram, createFloatTexture, makeQuadBuffer } from './gl-utils.js';

const PAD_EM = 0.05;
const TEX_W = 1024;

const VS = `#version 300 es
in vec2 a_corner; in vec2 a_emOrigin; in vec2 a_quadMin; in vec2 a_quadMax;
in float a_curveStart; in float a_curveCount;
uniform mat4 u_mvp;
out vec2 v_localEm; flat out float v_curveStart; flat out float v_curveCount;
void main(){
  vec2 localEm = mix(a_quadMin, a_quadMax, a_corner);
  v_localEm = localEm; v_curveStart = a_curveStart; v_curveCount = a_curveCount;
  gl_Position = u_mvp * vec4(localEm + a_emOrigin, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D u_curveTex; uniform int u_curveTexW;
uniform int u_maxWindows; uniform float u_gamma; uniform vec4 u_color;
in vec2 v_localEm; flat in float v_curveStart; flat in float v_curveCount;
out vec4 fragColor;
vec4 fetch(int idx){ return texelFetch(u_curveTex, ivec2(idx % u_curveTexW, idx / u_curveTexW), 0); }
// Mirrors intersectMonotonic in sweeper-core.js — keep the two in sync. The
// threshold-free root form; see that file for why the reference's |qa|<1e-3
// linear shortcut seams near-linear curves in em space.
float im(float qa,float c0,float c1,float c2,float target){
  float qb=2.0*c1-2.0*c0, qc=c0-target, d=qb*qb-4.0*qa*qc;
  if (d<0.0) return ((target<c0)==(c2>c0)) ? -1.0 : 2.0;
  float den=-qb-sign(c2-c0)*sqrt(d);
  return den==0.0 ? 0.0 : 2.0*qc/den;
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
// Mirrors footprintWindows in sweeper-footprint.js — keep the two in sync.
// The pixel's em-space footprint is the parallelogram spanned by dFdx/dFdy of
// v_localEm; since that varying is perspective-correct, the two derivatives
// carry the projection for free — no camera parameters reach this shader.
// fwidth would give that parallelogram's AABB, which is exact head-on but
// several times too large at glancing angles. So slice it into N slabs across
// the long axis and sweep each slab's AABB. N==1 is exactly fwidth again.
const int MAX_WINDOWS = 8;
void main(){
  vec2 ddx = dFdx(v_localEm), ddy = dFdy(v_localEm);
  float lx = length(ddx), ly = length(ddy);
  vec2 major = lx >= ly ? ddx : ddy;
  vec2 minor = lx >= ly ? ddy : ddx;
  float lo = min(lx, ly);
  float ratio = lo > 0.0 ? max(lx, ly)/lo : 1.0;
  int N = clamp(int(ceil(ratio - 1e-3)), 1, clamp(u_maxWindows, 1, MAX_WINDOWS));
  vec2 size = abs(major)/float(N) + abs(minor);
  float invArea = 1.0/(size.x*size.y);

  int start=int(v_curveStart+0.5), count=int(v_curveCount+0.5);
  float cov=0.0;
  for (int w=0; w<MAX_WINDOWS; w++){ if (w>=N) break;
    vec2 offset = v_localEm + major*((float(w)+0.5)/float(N) - 0.5) - 0.5*size;
    float area=0.0;
    for (int i=0;i<4096;i++){ if (i>=count) break;
      vec4 t0=fetch(start+i*2); vec4 t1=fetch(start+i*2+1);
      area += sweep(size, offset, t0.xy, t0.zw, t1.xy); }
    cov += clamp(area*invArea,0.0,1.0); }   // clamp per window, then average
  cov /= float(N);
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
    for (const n of ['u_mvp','u_curveTex','u_curveTexW','u_maxWindows','u_gamma','u_color'])
      this.u[n] = gl.getUniformLocation(this.prog, n);
    this.instanceCount = 0;
    this.stats = { curveCount:0, texW:TEX_W, texH:0, instanceCount:0 };
  }

  build(font, instances) {
    const gl = this.gl;
    const meta = new Map();
    const packedCurves = [];
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
    const totalCurves = packedCurves.length / 6;
    const texels = totalCurves * 2;
    const texH = Math.max(1, Math.ceil(texels / TEX_W));
    const texData = new Float32Array(TEX_W * texH * 4);
    for (let c = 0; c < totalCurves; c++) {
      const s = c*6, t = c*2*4;
      texData[t+0]=packedCurves[s+0]; texData[t+1]=packedCurves[s+1];
      texData[t+2]=packedCurves[s+2]; texData[t+3]=packedCurves[s+3];
      texData[t+4]=packedCurves[s+4]; texData[t+5]=packedCurves[s+5];
    }
    if (this.curveTex) gl.deleteTexture(this.curveTex);
    this.curveTex = createFloatTexture(gl, TEX_W, texH, texData);

    const inst = new Float32Array(instances.length * 8);
    let n = 0;
    for (const it of instances) {
      const m = meta.get(it.glyphId); if (!m || m.count === 0) continue;
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
    gl.uniformMatrix4fv(this.u.u_mvp, false, view.mvp);
    gl.uniform1i(this.u.u_maxWindows, view.maxWindows ?? 1);
    gl.uniform1f(this.u.u_gamma, view.gamma);
    gl.uniform4fv(this.u.u_color, view.color);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.instanceCount);
    gl.bindVertexArray(null);
  }
}
