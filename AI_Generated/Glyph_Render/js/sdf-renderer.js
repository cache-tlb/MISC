import { generateSDF, generateMSDF } from './sdf-generator.js';
import { createProgram, createByteTexture, makeQuadBuffer } from './gl-utils.js';

const ATLAS_W = 2048;

const VS = `#version 300 es
in vec2 a_corner; in vec2 a_emOrigin; in vec2 a_tileMin; in vec2 a_tileMax;
in vec4 a_rect; in float a_spread;
uniform mat4 u_mvp;
out vec2 v_uv; out vec2 v_localEm; flat out float v_spread;
void main(){
  vec2 localEm = mix(a_tileMin, a_tileMax, a_corner);
  v_uv = mix(a_rect.xy, a_rect.zw, a_corner);  // corner.y up == v up (row0 = em bottom)
  v_localEm = localEm;
  v_spread = a_spread;
  gl_Position = u_mvp * vec4(localEm + a_emOrigin, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D u_atlas; uniform int u_msdf; uniform vec4 u_color;
in vec2 v_uv; in vec2 v_localEm; flat in float v_spread; out vec4 fragColor;
void main(){
  vec3 s = texture(u_atlas, v_uv).rgb;
  float m = u_msdf==1 ? max(min(s.r,s.g), min(max(s.r,s.g), s.b)) : s.r;
  float dEm = (m-0.5)*2.0*v_spread;
  // Per-pixel em footprint, same source as the sweeper. Under the 2D matrix
  // fwidth(v_localEm) is (emPerPixel, emPerPixel), so this is numerically the
  // old u_emPerPixel. A distance field carries one isotropic distance per texel,
  // so there is nothing here to decompose anisotropically — that asymmetry is
  // exactly what the grazing-angle comparison is meant to show.
  float aa = 0.5*max(fwidth(v_localEm).x, fwidth(v_localEm).y);
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
    for (const n of ['u_mvp','u_atlas','u_msdf','u_color'])
      this.u[n] = gl.getUniformLocation(this.prog, n);
    this._font = null; this._instances = null; this._res = 32; this.mode = 'sdf';
    this.instanceCount = 0;
    this.stats = { atlasW:0, atlasH:0, tiles:0, mode:'sdf', res:32 };
  }

  build(font, instances, { res = 32, mode = 'sdf' } = {}) {
    this._font = font; this._instances = instances; this._res = res; this.mode = mode;
    this._rebuild();
  }
  setMode(mode) { if (mode !== this.mode) { this.mode = mode; this._rebuild(); } }
  setRes(res) { if (res !== this._res) { this._res = res; this._rebuild(); } }

  _rebuild() {
    const gl = this.gl;
    const gen = this.mode === 'msdf' ? generateMSDF : generateSDF;
    const tiles = new Map();
    for (const gid of new Set(this._instances.map(i => i.glyphId)))
      tiles.set(gid, gen(this._font.outlineEm(gid), { res: this._res, pad: 4 }));

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

    const inst = new Float32Array(this._instances.length * 11);
    let n=0;
    for (const it of this._instances) {
      const t = tiles.get(it.glyphId), p = place.get(it.glyphId); if (!t) continue;
      const emSpan = t.size / t.tile.scale;
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
    gl.uniformMatrix4fv(this.u.u_mvp, false, view.mvp);
    gl.uniform4fv(this.u.u_color, view.color);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.instanceCount);
    gl.bindVertexArray(null);
  }
}
