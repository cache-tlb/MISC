// Visual sanity dump for the core (pre-WebGL). Writes out/sdf.bmp, out/msdf.bmp,
// out/curves.svg for a glyph so SDF/MSDF/monotonic-curve output can be eyeballed.
//   node tools/dump.mjs <fontPath> <char> [res]
import { createRequire } from 'module';
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
const opentype = require('../vendor/opentype.cjs');
import { wrapFont } from '../js/font-loader.js';
import { preprocessGlyph } from '../js/sweeper-preprocess.js';
import { generateSDF, generateMSDF } from '../js/sdf-generator.js';

function loadOt(path) { const b = readFileSync(path); return opentype.parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }

function bmp24(width, height, rgb /* Uint8Array w*h*3, row 0 = top */) {
  const rowSize = Math.ceil(width*3/4)*4, pixArr = rowSize*height, fileSize = 54+pixArr;
  const b = Buffer.alloc(fileSize);
  b.write('BM',0); b.writeUInt32LE(fileSize,2); b.writeUInt32LE(54,10);
  b.writeUInt32LE(40,14); b.writeInt32LE(width,18); b.writeInt32LE(height,22);
  b.writeUInt16LE(1,26); b.writeUInt16LE(24,28); b.writeUInt32LE(pixArr,34);
  for (let y=0;y<height;y++){ const srcY=height-1-y; // BMP rows are bottom-up
    for (let x=0;x<width;x++){ const s=(srcY*width+x)*3, d=54+y*rowSize+x*3;
      b[d]=rgb[s+2]; b[d+1]=rgb[s+1]; b[d+2]=rgb[s]; } }
  return b;
}
const grayToRgb = (size, data) => { const o=new Uint8Array(size*size*3); for (let i=0;i<size*size;i++) o[i*3]=o[i*3+1]=o[i*3+2]=data[i]; return o; };

const [,, fontPath='fonts/Tinos-Regular.ttf', char='A', resStr='48'] = process.argv;
const res = parseInt(resStr,10);
const font = wrapFont(loadOt(fontPath));
const gid = font.glyphIdForChar(char);
if (!gid) { console.error(`glyph for '${char}' not found in ${fontPath}`); process.exit(1); }
const cmds = font.outlineEm(gid);
mkdirSync('out', { recursive: true });

const sdf = generateSDF(cmds, { res });
writeFileSync('out/sdf.bmp', bmp24(sdf.size, sdf.size, grayToRgb(sdf.size, sdf.data)));
const msdf = generateMSDF(cmds, { res });
writeFileSync('out/msdf.bmp', bmp24(msdf.size, msdf.size, msdf.data));

const curves = preprocessGlyph(cmds);
const svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.2 -0.9 1.4 1.4" width="400" height="400">',
  '<rect x="-0.2" y="-0.9" width="1.4" height="1.4" fill="#111"/>',
  '<g transform="scale(1,-1)">',
  ...curves.map(c=>`<path d="M${c.p0[0]},${c.p0[1]} Q${c.p1[0]},${c.p1[1]} ${c.p2[0]},${c.p2[1]}" fill="none" stroke="#39f" stroke-width="0.006"/>`),
  '</g></svg>'].join('\n');
writeFileSync('out/curves.svg', svg);
console.log(`wrote out/sdf.bmp out/msdf.bmp out/curves.svg — glyph '${char}', ${curves.length} monotonic curves, atlas ${sdf.size}px`);
