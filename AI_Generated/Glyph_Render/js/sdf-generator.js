import { evalQuad, evalCubic, bboxOfCommands } from './geom.js';

// ---- contour parsing into directed segments (lines/curves) ----
function parseContours(commands) {
  const contours = [];
  let segs = null, start = null, cur = null;
  const flush = () => { if (segs && segs.length) contours.push(segs); };
  for (const c of commands) {
    if (c.type==='M'){ flush(); segs=[]; start=[c.x,c.y]; cur=start; }
    else if (c.type==='L'){ segs.push({type:'L',pts:[cur,[c.x,c.y]]}); cur=[c.x,c.y]; }
    else if (c.type==='Q'){ segs.push({type:'Q',pts:[cur,[c.x1,c.y1],[c.x,c.y]]}); cur=[c.x,c.y]; }
    else if (c.type==='C'){ segs.push({type:'C',pts:[cur,[c.x1,c.y1],[c.x2,c.y2],[c.x,c.y]]}); cur=[c.x,c.y]; }
    else if (c.type==='Z'){ if (start && (cur[0]!==start[0]||cur[1]!==start[1])) segs.push({type:'L',pts:[cur,start]}); cur=start; }
  }
  flush();
  return contours;
}

const sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const norm=(v)=>{const l=Math.hypot(v[0],v[1])||1;return [v[0]/l,v[1]/l];};
function segDir(seg, atEnd){
  const p=seg.pts;
  if (seg.type==='L') return norm(sub(p[1],p[0]));
  if (seg.type==='Q') return atEnd ? norm(sub(p[2],p[1])) : norm(sub(p[1],p[0]));
  return atEnd ? norm(sub(p[3],p[2])) : norm(sub(p[1],p[0]));
}

// ---- flatten one segment into line edges carrying its color ----
function flattenSeg(seg, color, segsPerCurve, out) {
  const p=seg.pts;
  const push=(a,b)=>{ if (a[0]!==b[0]||a[1]!==b[1]) out.push({a,b,color}); };
  if (seg.type==='L') push(p[0],p[1]);
  else if (seg.type==='Q'){ let prev=p[0]; for(let i=1;i<=segsPerCurve;i++){const q=evalQuad(p[0],p[1],p[2],i/segsPerCurve);push(prev,q);prev=q;} }
  else { let prev=p[0]; for(let i=1;i<=segsPerCurve;i++){const q=evalCubic(p[0],p[1],p[2],p[3],i/segsPerCurve);push(prev,q);prev=q;} }
}

// ---- msdfgen-style simple edge coloring ----
const RED=1,GREEN=2,YELLOW=3,BLUE=4,MAGENTA=5,CYAN=6,WHITE=7,BLACK=0;
function switchColor(state, banned=BLACK) {
  let color = state.color;
  const combined = color & banned;
  if (combined===RED||combined===GREEN||combined===BLUE){ state.color = combined ^ WHITE; return; }
  if (color===BLACK||color===WHITE){ const start=[CYAN,MAGENTA,YELLOW]; state.color=start[state.seed%3]; state.seed=Math.floor(state.seed/3); return; }
  const shifted = color << (1 + (state.seed & 1));
  state.color = (shifted | (shifted >> 3)) & WHITE;
  state.seed = state.seed >> 1;
}
function isCorner(aDir, bDir, crossThreshold) {
  const dot = aDir[0]*bDir[0]+aDir[1]*bDir[1];
  const cross = aDir[0]*bDir[1]-aDir[1]*bDir[0];
  return dot <= 0 || Math.abs(cross) > crossThreshold;
}
export function colorEdges(contours, angleThreshold) {
  const crossThreshold = Math.sin(angleThreshold);
  const state = { color: WHITE, seed: 0 };
  for (const segs of contours) {
    if (!segs.length) continue;
    const corners = [];
    let prevDir = segDir(segs[segs.length-1], true);
    for (let i=0;i<segs.length;i++){ const d=segDir(segs[i], false); if (isCorner(prevDir,d,crossThreshold)) corners.push(i); prevDir=segDir(segs[i], true); }
    if (corners.length===0){ for (const s of segs) s.color=WHITE; continue; }
    const m=segs.length, cornerCount=corners.length, startIdx=corners[0];
    state.color=WHITE; switchColor(state); const initial=state.color; let spline=0;
    for (let i=0;i<m;i++){
      const index=(startIdx+i)%m;
      if (spline+1<cornerCount && corners[spline+1]===index){ spline++; switchColor(state, spline===cornerCount-1?initial:BLACK); }
      segs[index].color = state.color;
    }
  }
}

// ---- distances ----
function edgeData(e, px, py) {           // returns {trueSigned, pseudo}
  const ax=e.a[0],ay=e.a[1],bx=e.b[0],by=e.b[1];
  const ex=bx-ax, ey=by-ay; const len2=ex*ex+ey*ey||1e-12;
  const aqx=px-ax, aqy=py-ay;
  const t=(aqx*ex+aqy*ey)/len2;
  const invLen=1/Math.sqrt(len2); const dx=ex*invLen, dy=ey*invLen;
  const cross = dx*aqy - dy*aqx;         // signed perp to infinite line (pseudo)
  const tc=Math.min(1,Math.max(0,t)); const nx=ax+tc*ex, ny=ay+tc*ey;
  const dTrue=Math.hypot(px-nx,py-ny);
  const sign=cross>=0?1:-1;
  return { trueSigned: sign*dTrue, pseudo: cross };
}

// non-zero winding inside test (orientation-independent) for SDF sign
function windingInside(edges, px, py) {
  let w=0;
  for (const e of edges){ const [ax,ay]=e.a,[bx,by]=e.b;
    if ((ay<=py)!==(by<=py)){ const tx=ax+(py-ay)/(by-ay)*(bx-ax); if (tx>px) w += (by>ay)?1:-1; } }
  return w!==0;
}

// Orient so the interior is to the left of directed edges (cross-sign inside-positive).
function orientInsidePositive(edges) {
  let area = 0;
  for (const e of edges) area += e.a[0]*e.b[1] - e.b[0]*e.a[1];
  if (area < 0) for (const e of edges) { const t=e.a; e.a=e.b; e.b=t; }
}

function tileMapping(commands, res, pad) {
  const bb = bboxOfCommands(commands);
  const w = bb.maxX-bb.minX, h = bb.maxY-bb.minY;
  const span = Math.max(w, h) || 1;
  const scale = res / span;               // texels per em (content area)
  const size = res + 2*pad;
  const minX = bb.minX - pad/scale - (Math.max(0, (h-w))/2);   // center content in square tile
  const minY = bb.minY - pad/scale - (Math.max(0, (w-h))/2);
  const spreadEm = pad/scale;
  return { size, scale, minX, minY, spreadEm };
}
const emAt = (tile, tx, ty) => [ tile.minX + (tx+0.5)/tile.scale, tile.minY + (ty+0.5)/tile.scale ];
const encode = (dEm, spreadEm) => Math.round(255 * Math.min(1, Math.max(0, 0.5 + dEm/(2*spreadEm))));

export function generateSDF(commands, opts={}) {
  const res=opts.res??32, pad=opts.pad??4;
  const tile=tileMapping(commands,res,pad); const {size,spreadEm}=tile;
  const edges=[]; for (const segs of parseContours(commands)) for (const s of segs) flattenSeg(s, WHITE, 16, edges);
  const data=new Uint8Array(size*size);
  for (let ty=0;ty<size;ty++) for (let tx=0;tx<size;tx++){
    const [px,py]=emAt(tile,tx,ty);
    let best=Infinity; for (const e of edges){ const d=edgeData(e,px,py); if (Math.abs(d.trueSigned)<best) best=Math.abs(d.trueSigned); }
    const inside=windingInside(edges,px,py);
    const dEm=(inside?1:-1)*best;
    data[ty*size+tx]=encode(dEm,spreadEm);
  }
  return { size, data, spreadEm, tile };
}

const median3 = (a,b,c) => Math.max(Math.min(a,b), Math.min(Math.max(a,b), c));

export function generateMSDF(commands, opts={}) {
  const res=opts.res??32, pad=opts.pad??4, angle=(opts.angleDeg??3)*Math.PI/180;
  const tile=tileMapping(commands,res,pad); const {size,spreadEm}=tile;
  const contours=parseContours(commands);
  colorEdges(contours, angle);
  const edges=[]; for (const segs of contours) for (const s of segs) flattenSeg(s, s.color??WHITE, 16, edges);
  orientInsidePositive(edges);
  const data=new Uint8Array(size*size*3);
  for (let ty=0;ty<size;ty++) for (let tx=0;tx<size;tx++){
    const [px,py]=emAt(tile,tx,ty);
    const bestTrue=[Infinity,Infinity,Infinity], pseudo=[0,0,0];
    let bestAll=Infinity;                                  // nearest edge overall (true SDF magnitude)
    for (const e of edges){ const d=edgeData(e,px,py);
      if (Math.abs(d.trueSigned)<bestAll) bestAll=Math.abs(d.trueSigned);
      for (let ch=0;ch<3;ch++){ if (e.color & (1<<ch)){ if (Math.abs(d.trueSigned)<bestTrue[ch]){ bestTrue[ch]=Math.abs(d.trueSigned); pseudo[ch]=d.pseudo; } } } }
    const i=(ty*size+tx)*3;
    // Error correction: if the median's inside/outside sign disagrees with the true
    // winding sign, this texel is a spurious MSDF edge -> replace with true SDF.
    const inside = windingInside(edges, px, py);
    const medSigned = median3(pseudo[0], pseudo[1], pseudo[2]);
    if ((medSigned > 0) !== inside) {
      const v = encode((inside?1:-1)*bestAll, spreadEm);
      data[i]=data[i+1]=data[i+2]=v;
    } else {
      for (let ch=0;ch<3;ch++) data[i+ch]=encode(pseudo[ch], spreadEm);
    }
  }
  return { size, data, spreadEm, tile };
}
