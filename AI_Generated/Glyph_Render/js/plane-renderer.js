import { createProgram, makeQuadBuffer } from './gl-utils.js';

const MARGIN_EM = 0.6;
const CELL_EM = 0.25;

const VS = `#version 300 es
in vec2 a_corner;
uniform mat4 u_mvp; uniform vec4 u_extentEm;
out vec2 v_em;
void main(){
  vec2 em = mix(u_extentEm.xy, u_extentEm.zw, a_corner);
  v_em = em;
  gl_Position = u_mvp * vec4(em, 0.0, 1.0);
}`;

// Grid lines antialiased by their own screen-space footprint — the same fwidth
// trick the glyph shaders use, and what keeps the grid from shimmering as it
// recedes toward the horizon at grazing angles.
const FS = `#version 300 es
precision highp float;
uniform float u_cell;
in vec2 v_em; out vec4 fragColor;
void main(){
  vec2 g = v_em/u_cell;
  vec2 d = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), vec2(1e-6));
  float line = 1.0 - min(min(d.x, d.y), 1.0);
  fragColor = vec4(vec3(0.30,0.36,0.46), line*0.5 + 0.05);
}`;

export class PlaneRenderer {
  constructor(gl) {
    this.gl = gl;
    this.prog = createProgram(gl, VS, FS);
    this.quad = makeQuadBuffer(gl);
    this.vao = gl.createVertexArray();
    this.u = {};
    for (const n of ['u_mvp','u_extentEm','u_cell']) this.u[n] = gl.getUniformLocation(this.prog, n);
    this.extent = [0,0,1,1];
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    const l = gl.getAttribLocation(this.prog, 'a_corner');
    gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  setExtent(b) {
    this.extent = [b.minX-MARGIN_EM, b.minY-MARGIN_EM, b.maxX+MARGIN_EM, b.maxY+MARGIN_EM];
  }

  render(view) {
    const gl = this.gl;
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this.u.u_mvp, false, view.mvp);
    gl.uniform4fv(this.u.u_extentEm, this.extent);
    gl.uniform1f(this.u.u_cell, CELL_EM);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }
}
