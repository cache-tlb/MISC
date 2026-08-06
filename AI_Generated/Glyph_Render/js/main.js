import { createGL2, resizeCanvasToDisplay } from './gl-utils.js';
import { wrapFont } from './font-loader.js';
import { layoutText, uniqueGlyphIds } from './layout.js';
import { SweeperRenderer } from './sweeper-renderer.js';
import { SdfRenderer } from './sdf-renderer.js';

const $ = (id) => document.getElementById(id);
const canvas = $('gl'), stage = $('stage'), divider = $('divider');
const gl = createGL2(canvas);
const fmt2 = (x) => Number(x).toFixed(2);

const state = {
  font: null, fontName: '', text: $('text').value,
  fontSizePx: 120, zoom: 1, panCss: [40, null], gamma: 1.0,
  sdfRes: 32, sdfMode: 'sdf', sliderX: 0.5, color: [0.96,0.96,0.97,1.0],
  buildMs: '0', uniqueCount: 0,
};
let sweeper, sdf, needRender = false;

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
  if (state.panCss[1] === null) state.panCss[1] = (h * 0.6) / dpr;   // initial baseline (CSS px)
  const pxPerEm = state.fontSizePx * state.zoom * dpr;
  const originDev = [state.panCss[0] * dpr, state.panCss[1] * dpr];
  return { pxPerEm, originDev, backing:[w,h], emPerPixel:1/pxPerEm, gamma:state.gamma, color:state.color };
}

function render() {
  needRender = false;
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
}
function requestRender(){ if (!needRender){ needRender = true; requestAnimationFrame(render); } }

function updateInfo() {
  const s = sweeper?.stats, a = sdf?.stats;
  const bytesCurve = s ? s.texW*s.texH*16 : 0;
  const bytesAtlas = a ? a.atlasW*a.atlasH*4 : 0;
  $('info').textContent =
`字体 ${state.fontName}
唯一字形 ${state.uniqueCount}
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
$('gamma').oninput = (e) => { state.gamma = +e.target.value; $('gammaVal').textContent = fmt2(state.gamma); requestRender(); };
$('text').oninput = (e) => { state.text = e.target.value; buildAll(); };
$('reset').onclick = () => { state.zoom = 1; state.panCss = [40, null]; state.fontSizePx = 120; $('size').value=120; $('sizeVal').textContent='120'; requestRender(); };

// pan (drag on canvas)
let dragging=false, last=[0,0];
canvas.addEventListener('pointerdown', e=>{ dragging=true; last=[e.clientX,e.clientY]; canvas.classList.add('drag'); canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e=>{ if(!dragging) return; state.panCss=[state.panCss[0]+(e.clientX-last[0]), state.panCss[1]+(e.clientY-last[1])]; last=[e.clientX,e.clientY]; requestRender(); });
canvas.addEventListener('pointerup', ()=>{ dragging=false; canvas.classList.remove('drag'); });
// zoom around cursor
canvas.addEventListener('wheel', e=>{ e.preventDefault();
  const r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const f=Math.exp(-e.deltaY*0.0015), nz=Math.min(60,Math.max(0.1,state.zoom*f)), k=nz/state.zoom;
  if (state.panCss[1]===null) return;
  state.panCss=[mx-(mx-state.panCss[0])*k, my-(my-state.panCss[1])*k];
  state.zoom=nz; requestRender();
}, {passive:false});
// divider drag
divider.addEventListener('pointerdown', e=>{ e.stopPropagation();
  const move=ev=>{ const r=stage.getBoundingClientRect(); state.sliderX=Math.min(1,Math.max(0,(ev.clientX-r.left)/r.width)); requestRender(); };
  const up=()=>{ window.removeEventListener('pointermove',move); window.removeEventListener('pointerup',up); };
  window.addEventListener('pointermove',move); window.addEventListener('pointerup',up);
});
window.addEventListener('resize', requestRender);

// ---------- optional deep-link overrides (shareable states; also drive tests) ----------
const q = new URLSearchParams(location.search);
if (q.has('mode')) {
  state.sdfMode = q.get('mode')==='msdf' ? 'msdf' : 'sdf';
  $('mode').querySelectorAll('button').forEach(x=>x.classList.toggle('on', x.dataset.m===state.sdfMode));
}
if (q.has('res'))   { state.sdfRes=+q.get('res'); $('res').value=state.sdfRes; $('resVal').textContent=state.sdfRes; }
if (q.has('size'))  { state.fontSizePx=+q.get('size'); $('size').value=state.fontSizePx; $('sizeVal').textContent=state.fontSizePx; }
if (q.has('zoom'))  state.zoom=+q.get('zoom');
if (q.has('slider')) state.sliderX=Math.min(1,Math.max(0,+q.get('slider')));
if (q.has('text'))  { state.text=q.get('text'); $('text').value=state.text; }
const bootFont = q.has('font') ? 'fonts/'+q.get('font') : $('font').value;
if (q.has('font')) $('font').value = bootFont;

// ---------- boot ----------
$('rlabel').textContent = (state.sdfMode==='msdf'?'MSDF':'SDF') + ' ▶';
loadFontFromUrl(bootFont);
