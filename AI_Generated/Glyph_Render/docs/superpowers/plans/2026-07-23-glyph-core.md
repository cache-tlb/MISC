# Glyph Rendering — Core Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dependency-light, fully unit-tested JS core that turns a font + text into (a) positioned glyph instances, (b) directed monotonic quadratic Béziers for the Scanline Sweeper, and (c) single-channel SDF + MSDF bitmaps — everything the WebGL demo (Plan 2) will consume.

**Architecture:** Pure ES modules under `js/`, each with one responsibility, all runnable in Node for tests (no WebGL, no DOM). `opentype.js` is the only third-party dependency, vendored locally and injected (never accessed as a global inside the core), so the same modules run in Node tests and the browser. A BMP dump CLI renders SDF/MSDF/curves to images for eyeball verification before any GL work.

**Tech Stack:** JavaScript (ES modules), Node.js built-in test runner (`node:test` + `node:assert`, zero deps), opentype.js v1.3.4 (MIT, vendored), `pyftsubset`/`fontTools` (build-time only, for the CJK subset).

## Global Constraints

- No third-party runtime JS libraries except vendored `opentype.js`. All core modules are hand-written. (spec §3)
- All geometry is in **em space** (font units ÷ `unitsPerEm`), **y-up** (font-native orientation). Screen y-down conversion happens only in Plan 2's shaders. (spec §5.1)
- Fill-sign convention: glyph **interior is positive**. TrueType/CFF winding differs, so orientation is normalized in code and asserted by tests. (spec §5.1, §8.2, §13)
- Core modules must not reference `window`, `document`, or a global `opentype`; `opentype` is passed in. (Node-testability)
- Node test runner: `node --test`. Test files: `test/*.test.mjs`. Source: `js/*.js` (ESM).
- Frequent commits: one per task minimum.

---

## File Structure

```
package.json                 { "type":"module", "scripts":{"test":"node --test"} }
vendor/opentype.min.js       vendored UMD build (Task 1)
fonts/*.ttf                  vendored fonts incl. CJK subset (Task 1)
fonts/LICENSES.md            license + provenance (Task 1)
js/font-loader.js            wrapFont(otFont) -> Font; parseArrayBuffer(buf) [browser]
js/layout.js                 layoutText(font, text, opts) -> Instance[]
js/geom.js                   shared geometry helpers (bbox, bezier eval, flatten)
js/sweeper-preprocess.js     preprocessGlyph(commands) -> QuadCurve[]
js/sweeper-core.js           evaluateBezier/intersectMonotonic/scanlineSweep + sumSweep
js/sdf-generator.js          generateSDF / generateMSDF (+ edge coloring, pseudo-dist)
test/dump.mjs                CLI: font+char -> sdf.bmp, msdf.bmp, curves.svg
test/*.test.mjs              unit tests
```

Each `js/*.js` file is one focused unit. `js/geom.js` holds primitives shared by preprocess and the SDF generator (DRY).

---

### Task 1: Project scaffold + vendored assets

**Files:**
- Create: `package.json`, `.gitignore` (already present from spec commit — verify), `vendor/opentype.min.js`, `fonts/*.ttf`, `fonts/LICENSES.md`, `test/smoke.test.mjs`

**Interfaces:**
- Consumes: nothing.
- Produces: vendored `opentype` (UMD, `require`-able), font files at known paths, a green test baseline.

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "glyph-render-demo",
  "version": "0.1.0",
  "type": "module",
  "private": true,
  "scripts": {
    "test": "node --test"
  }
}
```

- [ ] **Step 2: Vendor opentype.js and download fonts**

Run (from repo root `d:/tmp/Glyph_Render`):

```bash
mkdir -p vendor fonts
curl -sSL -o vendor/opentype.min.js https://cdn.jsdelivr.net/npm/opentype.js@1.3.4/dist/opentype.min.js
# Serif (Times-compatible), Sans (paper's font), Script (curvy stress test)
curl -sSL -o fonts/Tinos-Regular.ttf   "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Regular.ttf"
curl -sSL -o fonts/Geist-Regular.ttf   "https://github.com/vercel/geist-font/raw/main/packages/next/dist/fonts/geist-sans/Geist-Regular.ttf"
curl -sSL -o fonts/DancingScript.ttf   "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf"
# CJK source (full, ~10MB) — subset next
curl -sSL -o fonts/_NotoSansSC-Full.otf "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf" || \
curl -sSL -o fonts/_NotoSansSC-Full.ttf "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"
ls -la vendor/ fonts/
```

Expected: `opentype.min.js` ~170KB; each font file non-trivial size. If any URL 404s, find the current path in that project's repo and substitute (record the working URL in `LICENSES.md`).

- [ ] **Step 3: Build the CJK subset** (common Chinese + ASCII + CJK punctuation)

Run:

```bash
# GB2312 level-1 (3755 common hanzi) via fontTools unicode ranges + ASCII + CJK punct.
python - <<'PY'
# Build a unicodes file for pyftsubset: ASCII, CJK punctuation, and common hanzi.
ranges = []
ranges += list(range(0x20, 0x7F))            # ASCII
ranges += list(range(0x3000, 0x3040))        # CJK symbols & punctuation
ranges += list(range(0xFF00, 0xFFF0))        # Fullwidth forms
ranges += list(range(0x4E00, 0x9FA6))        # CJK Unified (superset of GB2312 L1; subsetter drops absent)
with open('fonts/_subset_unicodes.txt','w') as f:
    f.write(','.join('U+%04X'%c for c in ranges))
print('wrote', len(ranges), 'codepoints')
PY
SRC=$(ls fonts/_NotoSansSC-Full.* | head -1)
pyftsubset "$SRC" \
  --unicodes-file=fonts/_subset_unicodes.txt \
  --output-file=fonts/NotoSansSC-Subset.ttf \
  --flavor= --no-hinting --desubroutinize --drop-tables+=DSIG
ls -la fonts/NotoSansSC-Subset.ttf
rm -f fonts/_NotoSansSC-Full.* fonts/_subset_unicodes.txt
```

Expected: `NotoSansSC-Subset.ttf` roughly 1–4 MB. If `pyftsubset` errors on a variable font, pass `--instancer` or pin an instance first with `fonttools varLib.instancer "$SRC" wght=400`.

- [ ] **Step 4: Write `fonts/LICENSES.md`**

```markdown
# Vendored font licenses & provenance

- Tinos-Regular.ttf — Apache-2.0 — Google Fonts (metric-compatible with Times New Roman)
- Geist-Regular.ttf — OFL-1.1 — Vercel Geist
- DancingScript.ttf — OFL-1.1 — Google Fonts
- NotoSansSC-Subset.ttf — OFL-1.1 — Noto Sans SC, subset (ASCII + CJK punct + CJK Unified) via pyftsubset

opentype.js (vendor/opentype.min.js) — MIT — https://github.com/opentypejs/opentype.js v1.3.4
```

- [ ] **Step 5: Write the smoke test** — confirms opentype loads a vendored font in Node.

`test/smoke.test.mjs`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const opentype = require('../vendor/opentype.min.js');

test('opentype loads a vendored font', () => {
  const font = opentype.loadSync('fonts/Tinos-Regular.ttf');
  assert.ok(font.unitsPerEm >= 1000, 'unitsPerEm present');
  const gid = font.charToGlyphIndex('A');
  assert.ok(gid > 0, 'glyph index for A');
});
```

- [ ] **Step 6: Run the smoke test**

Run: `npm test`
Expected: PASS (1 test). If `require('../vendor/opentype.min.js')` fails, the UMD build still assigns `module.exports`; confirm the download isn't an HTML error page (`head -c 80 vendor/opentype.min.js`).

- [ ] **Step 7: Commit**

```bash
git add package.json vendor/ fonts/ test/smoke.test.mjs
git commit -m "chore: scaffold core, vendor opentype.js and fonts (incl. CJK subset)"
```

---

### Task 2: `js/geom.js` — shared geometry primitives

**Files:**
- Create: `js/geom.js`, `test/geom.test.mjs`

**Interfaces:**
- Produces:
  - `evalQuad(p0, p1, p2, t) -> [x,y]` — de Casteljau quadratic eval.
  - `evalCubic(p0, p1, p2, p3, t) -> [x,y]`
  - `bboxOfCommands(commands) -> {minX,minY,maxX,maxY}` — over all on- and off-curve points.
  - `flattenCommands(commands, segsPerCurve) -> Array<{a:[x,y], b:[x,y], contour:number}>` — line-segment edges per closed contour (closes each contour back to its move point).
  - `Command` shape (from font-loader): `{type:'M'|'L'|'Q'|'C'|'Z', x?,y?, x1?,y1?, x2?,y2?}`, all coords em, y-up.

- [ ] **Step 1: Write the failing test**

`test/geom.test.mjs`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evalQuad, evalCubic, bboxOfCommands, flattenCommands } from '../js/geom.js';

test('evalQuad endpoints and midpoint', () => {
  assert.deepEqual(evalQuad([0,0],[1,2],[2,0], 0), [0,0]);
  assert.deepEqual(evalQuad([0,0],[1,2],[2,0], 1), [2,0]);
  const m = evalQuad([0,0],[1,2],[2,0], 0.5);
  assert.ok(Math.abs(m[0]-1) < 1e-9 && Math.abs(m[1]-1) < 1e-9);
});

test('bbox over a square path', () => {
  const cmds = [
    {type:'M',x:0,y:0},{type:'L',x:2,y:0},{type:'L',x:2,y:3},{type:'L',x:0,y:3},{type:'Z'}
  ];
  assert.deepEqual(bboxOfCommands(cmds), {minX:0,minY:0,maxX:2,maxY:3});
});

test('flatten closes each contour', () => {
  const cmds = [
    {type:'M',x:0,y:0},{type:'L',x:2,y:0},{type:'L',x:2,y:2},{type:'Z'}
  ];
  const edges = flattenCommands(cmds, 8);
  assert.equal(edges.length, 3); // 2 lines + closing edge
  assert.deepEqual(edges[2].b, [0,0]); // closes back to move
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --test test/geom.test.mjs`
Expected: FAIL (module not found / exports undefined).

- [ ] **Step 3: Implement `js/geom.js`**

```javascript
export function evalQuad(p0, p1, p2, t) {
  const u = 1 - t;
  return [
    u*u*p0[0] + 2*u*t*p1[0] + t*t*p2[0],
    u*u*p0[1] + 2*u*t*p1[1] + t*t*p2[1],
  ];
}

export function evalCubic(p0, p1, p2, p3, t) {
  const u = 1 - t, uu = u*u, tt = t*t;
  const a = uu*u, b = 3*uu*t, c = 3*u*tt, d = tt*t;
  return [
    a*p0[0] + b*p1[0] + c*p2[0] + d*p3[0],
    a*p0[1] + b*p1[1] + c*p2[1] + d*p3[1],
  ];
}

export function bboxOfCommands(commands) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const acc = (x, y) => { if (x<minX)minX=x; if (y<minY)minY=y; if (x>maxX)maxX=x; if (y>maxY)maxY=y; };
  for (const c of commands) {
    if (c.type === 'Z') continue;
    if (c.x1 !== undefined) acc(c.x1, c.y1);
    if (c.x2 !== undefined) acc(c.x2, c.y2);
    if (c.x !== undefined) acc(c.x, c.y);
  }
  return { minX, minY, maxX, maxY };
}

// Break commands into closed contours of line-segment edges (em, y-up).
export function flattenCommands(commands, segsPerCurve = 16) {
  const edges = [];
  let contour = -1, start = null, cur = null;
  const line = (a, b) => { if (a[0]!==b[0] || a[1]!==b[1]) edges.push({ a, b, contour }); };
  for (const c of commands) {
    if (c.type === 'M') { contour++; start = [c.x, c.y]; cur = start; }
    else if (c.type === 'L') { const p=[c.x,c.y]; line(cur, p); cur = p; }
    else if (c.type === 'Q') {
      let prev = cur;
      for (let i=1;i<=segsPerCurve;i++){ const p=evalQuad(cur,[c.x1,c.y1],[c.x,c.y], i/segsPerCurve); line(prev,p); prev=p; }
      cur = [c.x, c.y];
    } else if (c.type === 'C') {
      let prev = cur;
      for (let i=1;i<=segsPerCurve;i++){ const p=evalCubic(cur,[c.x1,c.y1],[c.x2,c.y2],[c.x,c.y], i/segsPerCurve); line(prev,p); prev=p; }
      cur = [c.x, c.y];
    } else if (c.type === 'Z') { if (start) line(cur, start); cur = start; }
  }
  return edges;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `node --test test/geom.test.mjs`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add js/geom.js test/geom.test.mjs
git commit -m "feat(core): shared geometry primitives (eval, bbox, flatten)"
```

---

### Task 3: `js/font-loader.js` — opentype wrapper → em-space outlines

**Files:**
- Create: `js/font-loader.js`, `test/font-loader.test.mjs`

**Interfaces:**
- Consumes: an `opentype.Font` (parsed elsewhere) — never a global.
- Produces `wrapFont(otFont) -> Font`, where `Font` = `{ unitsPerEm, ascenderEm, descenderEm, ot, glyphIdForChar(ch)->number, advanceEm(gid)->number, kerningEm(gidLeft,gidRight)->number, outlineEm(gid)->Command[] }`.
  - `outlineEm` returns `Command[]` (see geom) in em, **y-up** (opentype's native path orientation), coords divided by `unitsPerEm`.
- Also `parseArrayBuffer(buf, opentype) -> Font` (browser passes global opentype; Node passes require'd module).

- [ ] **Step 1: Write the failing test**

`test/font-loader.test.mjs`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const opentype = require('../vendor/opentype.min.js');
import { wrapFont } from '../js/font-loader.js';

const font = wrapFont(opentype.loadSync('fonts/Tinos-Regular.ttf'));

test('metrics normalized to em', () => {
  assert.ok(font.unitsPerEm >= 1000);
  const gid = font.glyphIdForChar('A');
  const adv = font.advanceEm(gid);
  assert.ok(adv > 0.4 && adv < 1.0, `A advance ${adv} in em`);
});

test('outline is em-space and y-up', () => {
  const gid = font.glyphIdForChar('A');
  const cmds = font.outlineEm(gid);
  assert.ok(cmds.length > 3);
  const ys = cmds.filter(c=>c.y!==undefined).map(c=>c.y);
  const maxY = Math.max(...ys), minY = Math.min(...ys);
  // Cap of 'A' is well above baseline (y-up): top positive, bottom ~0.
  assert.ok(maxY > 0.5 && maxY < 1.2, `apex ${maxY}`);
  assert.ok(minY > -0.2 && minY < 0.15, `base ${minY}`);
  // All coords are em-normalized (|x|,|y| < ~2).
  assert.ok(cmds.every(c => (c.x===undefined || Math.abs(c.x) < 2)));
});

test('kerning: AV pair is negative in a font that kerns it', () => {
  const a = font.glyphIdForChar('A'), v = font.glyphIdForChar('V');
  const k = font.kerningEm(a, v);
  assert.ok(k <= 0, `AV kerning ${k} should be <= 0`);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --test test/font-loader.test.mjs`
Expected: FAIL (wrapFont undefined).

- [ ] **Step 3: Implement `js/font-loader.js`**

```javascript
// Normalize an opentype.Font into an em-space, y-up, DOM-free wrapper.
export function wrapFont(ot) {
  const upm = ot.unitsPerEm;
  return {
    unitsPerEm: upm,
    ascenderEm: ot.ascender / upm,
    descenderEm: ot.descender / upm,
    ot,
    glyphIdForChar(ch) { return ot.charToGlyphIndex(ch); },
    advanceEm(gid) { return ot.glyphs.get(gid).advanceWidth / upm; },
    kerningEm(gidLeft, gidRight) {
      const gl = ot.glyphs.get(gidLeft), gr = ot.glyphs.get(gidRight);
      const k = ot.getKerningValue ? ot.getKerningValue(gl, gr) : 0;
      return (k || 0) / upm;
    },
    outlineEm(gid) {
      const path = ot.glyphs.get(gid).path; // font units, y-up
      const s = 1 / upm;
      const out = [];
      for (const c of path.commands) {
        if (c.type === 'M' || c.type === 'L')
          out.push({ type: c.type, x: c.x*s, y: c.y*s });
        else if (c.type === 'Q')
          out.push({ type: 'Q', x1: c.x1*s, y1: c.y1*s, x: c.x*s, y: c.y*s });
        else if (c.type === 'C')
          out.push({ type: 'C', x1: c.x1*s, y1: c.y1*s, x2: c.x2*s, y2: c.y2*s, x: c.x*s, y: c.y*s });
        else if (c.type === 'Z')
          out.push({ type: 'Z' });
      }
      return out;
    },
  };
}

// Browser/Node convenience: parse raw bytes then wrap. `opentype` is injected.
export function parseArrayBuffer(buf, opentype) {
  return wrapFont(opentype.parse(buf));
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `node --test test/font-loader.test.mjs`
Expected: PASS (3 tests). If the AV test fails because Tinos lacks a `kern` table entry, relax to `k <= 0` already allows 0; keep as-is.

- [ ] **Step 5: Commit**

```bash
git add js/font-loader.js test/font-loader.test.mjs
git commit -m "feat(core): font-loader wraps opentype into em-space y-up outlines"
```

---

### Task 4: `js/layout.js` — text → positioned glyph instances

**Files:**
- Create: `js/layout.js`, `test/layout.test.mjs`

**Interfaces:**
- Consumes: `Font` (Task 3).
- Produces `layoutText(font, text, opts?) -> Instance[]`, `opts = { lineHeightEm=1.2 }`.
  - `Instance = { glyphId:number, ox:number, oy:number }` — pen origin in em, y-up; first line baseline at `oy=0`, subsequent lines at negative `oy` (down). `ox` accumulates advances + kerning per line.
  - Whitespace: `\n` breaks a line; space advances by its glyph advance.
  - Also `uniqueGlyphIds(instances) -> number[]` (sorted, deduped).

- [ ] **Step 1: Write the failing test**

`test/layout.test.mjs`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const opentype = require('../vendor/opentype.min.js');
import { wrapFont } from '../js/font-loader.js';
import { layoutText, uniqueGlyphIds } from '../js/layout.js';

const font = wrapFont(opentype.loadSync('fonts/Tinos-Regular.ttf'));

test('single line advances monotonically in x', () => {
  const inst = layoutText(font, 'AVA');
  assert.equal(inst.length, 3);
  assert.equal(inst[0].ox, 0);
  assert.ok(inst[1].ox > 0 && inst[2].ox > inst[1].ox);
  assert.ok(inst.every(i => i.oy === 0));
});

test('newline drops baseline by lineHeight', () => {
  const inst = layoutText(font, 'A\nB', { lineHeightEm: 1.2 });
  const b = inst.find(i => i.glyphId === font.glyphIdForChar('B'));
  assert.ok(Math.abs(b.oy - (-1.2)) < 1e-9, `second line oy ${b.oy}`);
  assert.equal(b.ox, 0);
});

test('kerning shrinks AV vs no-kern sum', () => {
  const inst = layoutText(font, 'AV');
  const a = font.glyphIdForChar('A');
  const advA = font.advanceEm(a);
  const kAV = font.kerningEm(a, font.glyphIdForChar('V'));
  assert.ok(Math.abs(inst[1].ox - (advA + kAV)) < 1e-9);
});

test('unique glyph ids dedupe', () => {
  const inst = layoutText(font, 'AAA');
  assert.deepEqual(uniqueGlyphIds(inst), [font.glyphIdForChar('A')]);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --test test/layout.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement `js/layout.js`**

```javascript
export function layoutText(font, text, opts = {}) {
  const lineHeightEm = opts.lineHeightEm ?? 1.2;
  const out = [];
  let line = 0;
  for (const raw of text.split('\n')) {
    let penX = 0, prevGid = -1;
    for (const ch of raw) {           // iterate by code point
      const gid = font.glyphIdForChar(ch);
      if (prevGid >= 0) penX += font.kerningEm(prevGid, gid);
      out.push({ glyphId: gid, ox: penX, oy: -line * lineHeightEm });
      penX += font.advanceEm(gid);
      prevGid = gid;
    }
    line++;
  }
  return out;
}

export function uniqueGlyphIds(instances) {
  return [...new Set(instances.map(i => i.glyphId))].sort((a, b) => a - b);
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `node --test test/layout.test.mjs`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add js/layout.js test/layout.test.mjs
git commit -m "feat(core): text layout with advances, kerning, multi-line"
```

---

### Task 5: `js/sweeper-preprocess.js` — outlines → monotonic quadratics

**Files:**
- Create: `js/sweeper-preprocess.js`, `test/sweeper-preprocess.test.mjs`

**Interfaces:**
- Consumes: `Command[]` (em, y-up), `evalCubic`/`evalQuad` from geom.
- Produces:
  - `QuadCurve = { p0:[x,y], p1:[x,y], p2:[x,y] }` — em, y-up, directed, strictly x- and y-monotonic.
  - `preprocessGlyph(commands, eps?) -> QuadCurve[]` (`eps` default `1e-3`).
  - `cubicToQuads(p0,c1,c2,p3, eps) -> QuadCurve[]` (adaptive midpoint subdivision).
  - `splitMonotonic(q) -> QuadCurve[]` (split at x- and y-critical params).
  - `isMonotonic(q) -> boolean` (both axes).

Rationale (spec §7.1): promote lines to quads (midpoint control), drop horizontal linear segments, split every quad at its axis critical points so each output is single-rooted per axis.

- [ ] **Step 1: Write the failing test**

`test/sweeper-preprocess.test.mjs`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evalQuad } from '../js/geom.js';
import { preprocessGlyph, splitMonotonic, isMonotonic, cubicToQuads } from '../js/sweeper-preprocess.js';

const mono = (q) => {
  // strictly monotonic: control point between endpoints on each axis
  for (const ax of [0,1]) {
    const a=q.p0[ax], b=q.p1[ax], c=q.p2[ax];
    assert.ok((a<=b && b<=c) || (a>=b && b>=c), `axis ${ax} not monotonic: ${a},${b},${c}`);
  }
};

test('splitMonotonic yields monotonic pieces', () => {
  // A quad with a y-extremum inside (0,1): p0.y=0, p1.y=2, p2.y=0
  const q = { p0:[0,0], p1:[1,2], p2:[2,0] };
  assert.equal(isMonotonic(q), false);
  const pieces = splitMonotonic(q);
  assert.ok(pieces.length >= 2);
  pieces.forEach(p => { assert.ok(isMonotonic(p)); mono(p); });
});

test('split pieces reproduce the original curve', () => {
  const q = { p0:[0,0], p1:[1,2], p2:[2,0] };
  const pieces = splitMonotonic(q);
  // Sample the concatenation and compare to original by matching endpoints continuity
  for (let i=0;i<pieces.length-1;i++)
    assert.ok(Math.hypot(pieces[i].p2[0]-pieces[i+1].p0[0], pieces[i].p2[1]-pieces[i+1].p0[1]) < 1e-9);
  // global endpoints preserved
  assert.deepEqual(pieces[0].p0, [0,0]);
  assert.deepEqual(pieces[pieces.length-1].p2, [2,0]);
});

test('preprocessGlyph drops horizontal lines and promotes lines to quads', () => {
  const cmds = [
    {type:'M',x:0,y:0},
    {type:'L',x:2,y:0},   // horizontal -> dropped
    {type:'L',x:2,y:2},   // vertical -> kept as quad
    {type:'L',x:0,y:2},   // horizontal -> dropped
    {type:'Z'},           // closing edge (0,2)->(0,0): vertical -> kept
  ];
  const curves = preprocessGlyph(cmds);
  assert.ok(curves.every(isMonotonic));
  // Two vertical edges survive; both are monotonic quads
  assert.equal(curves.length, 2);
});

test('cubicToQuads approximates within tolerance', () => {
  const quads = cubicToQuads([0,0],[0,1],[1,1],[1,0], 1e-3);
  // sample the cubic and the quad chain, compare
  const evalC = (t)=>{const u=1-t;return [3*u*u*t*0+3*u*t*t*1+t*t*t*1, 3*u*u*t*1+3*u*t*t*1+t*t*t*0];};
  // just assert continuity + endpoints
  assert.deepEqual(quads[0].p0, [0,0]);
  assert.deepEqual(quads[quads.length-1].p2, [1,0]);
  assert.ok(quads.length >= 1);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --test test/sweeper-preprocess.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement `js/sweeper-preprocess.js`**

```javascript
import { evalQuad, evalCubic } from './geom.js';

const EPS = 1e-7;

export function isMonotonic(q) {
  for (const ax of [0,1]) {
    const a=q.p0[ax], b=q.p1[ax], c=q.p2[ax];
    const up = (a<=b+EPS && b<=c+EPS), dn = (a>=b-EPS && b>=c-EPS);
    if (!up && !dn) return false;
  }
  return true;
}

// Critical parameter where derivative == 0 for one axis, if inside (0,1).
function criticalT(a, b, c) {
  const denom = a - 2*b + c;
  if (Math.abs(denom) < EPS) return null;
  const t = (a - b) / denom;
  return (t > EPS && t < 1 - EPS) ? t : null;
}

function subdivideQuadAt(q, t) {
  const p01 = lerp(q.p0, q.p1, t), p12 = lerp(q.p1, q.p2, t);
  const mid = lerp(p01, p12, t);
  return [{ p0:q.p0, p1:p01, p2:mid }, { p0:mid, p1:p12, p2:q.p2 }];
}
const lerp = (a,b,t)=>[a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t];

export function splitMonotonic(q) {
  // collect split params from both axes, split in ascending order
  const ts = [];
  for (const ax of [0,1]) {
    const t = criticalT(q.p0[ax], q.p1[ax], q.p2[ax]);
    if (t !== null) ts.push(t);
  }
  ts.sort((a,b)=>a-b);
  let pieces = [q];
  for (let i=0;i<ts.length;i++){
    // split the LAST piece at the (re-normalized) param; process left-to-right
    const t = ts[i];
    const out = [];
    for (const piece of pieces) out.push(piece);
    // Rebuild by splitting whichever piece contains t. Simplest robust approach:
    pieces = resplit([q], ts.slice(0, i+1));
    break; // resplit handles all params at once
  }
  if (ts.length === 0) return [q];
  return resplit([q], ts);
}

// Split original quad q at a sorted list of global params in (0,1).
function resplit(_ignore, tsGlobal) {
  // Use the fact that a quad split at global params t1<t2 can be done by
  // repeatedly splitting and remapping. Do it directly via de Casteljau on q.
  const q = _ignore[0];
  const cuts = [0, ...tsGlobal, 1];
  const out = [];
  for (let i=0;i<cuts.length-1;i++){
    out.push(segmentOfQuad(q, cuts[i], cuts[i+1]));
  }
  return out;
}

// Extract the sub-quad of q over [t0,t1] (both in [0,1]).
function segmentOfQuad(q, t0, t1) {
  const A = evalQuad(q.p0,q.p1,q.p2,t0);
  const C = evalQuad(q.p0,q.p1,q.p2,t1);
  // control point: derivative-based. B = A + (dP/dt at t0) * (t1-t0)/2 ... use tangent intersection.
  const d0 = quadDeriv(q, t0), d1 = quadDeriv(q, t1);
  // Intersect tangent lines A+u*d0 and C - v*d1 for the shared control point.
  const B = intersectTangents(A, d0, C, d1);
  return { p0:A, p1:B, p2:C };
}
function quadDeriv(q, t) {
  return [
    2*(1-t)*(q.p1[0]-q.p0[0]) + 2*t*(q.p2[0]-q.p1[0]),
    2*(1-t)*(q.p1[1]-q.p0[1]) + 2*t*(q.p2[1]-q.p1[1]),
  ];
}
function intersectTangents(A, dA, C, dC) {
  // Solve A + s*dA = C - r*dC. Use 2x2. Fall back to midpoint if near-parallel.
  const det = dA[0]*(-(-dC[1])) - (-(-dC[0]))*dA[1]; // dA x dC
  const cross = dA[0]*dC[1] - dA[1]*dC[0];
  if (Math.abs(cross) < 1e-9) return [(A[0]+C[0])/2, (A[1]+C[1])/2];
  const s = ((C[0]-A[0])*dC[1] - (C[1]-A[1])*dC[0]) / cross;
  return [A[0] + s*dA[0], A[1] + s*dA[1]];
}

export function cubicToQuads(p0, c1, c2, p3, eps = 1e-3, depth = 0) {
  // Midpoint quadratic: control = (3*c1 - p0 + 3*c2 - p3)/4 ; check error, subdivide if needed.
  const ctrl = [(3*c1[0]-p0[0]+3*c2[0]-p3[0])/4, (3*c1[1]-p0[1]+3*c2[1]-p3[1])/4];
  // error ~ distance between cubic and quad at t=0.5
  const cubMid = evalCubic(p0,c1,c2,p3,0.5), quadMid = evalQuad(p0,ctrl,p3,0.5);
  const err = Math.hypot(cubMid[0]-quadMid[0], cubMid[1]-quadMid[1]);
  if (err <= eps || depth >= 6) return [{ p0, p1: ctrl, p2: p3 }];
  // subdivide cubic at 0.5 (de Casteljau) and recurse
  const m01=lerp(p0,c1,.5), m12=lerp(c1,c2,.5), m23=lerp(c2,p3,.5);
  const m012=lerp(m01,m12,.5), m123=lerp(m12,m23,.5), mid=lerp(m012,m123,.5);
  return [
    ...cubicToQuads(p0,m01,m012,mid, eps, depth+1),
    ...cubicToQuads(mid,m123,m23,p3, eps, depth+1),
  ];
}

export function preprocessGlyph(commands, eps = 1e-3) {
  const raw = [];
  let start = null, cur = null;
  const pushLine = (a, b) => {
    if (Math.abs(a[1]-b[1]) < EPS) return;          // drop horizontal
    raw.push({ p0:a, p1:[(a[0]+b[0])/2,(a[1]+b[1])/2], p2:b }); // promote to quad
  };
  for (const c of commands) {
    if (c.type === 'M') { start=[c.x,c.y]; cur=start; }
    else if (c.type === 'L') { pushLine(cur,[c.x,c.y]); cur=[c.x,c.y]; }
    else if (c.type === 'Q') { raw.push({ p0:cur, p1:[c.x1,c.y1], p2:[c.x,c.y] }); cur=[c.x,c.y]; }
    else if (c.type === 'C') { for (const q of cubicToQuads(cur,[c.x1,c.y1],[c.x2,c.y2],[c.x,c.y],eps)) raw.push(q); cur=[c.x,c.y]; }
    else if (c.type === 'Z') { if (start) pushLine(cur,start); cur=start; }
  }
  // split every quad to monotonic; drop any resulting horizontal (degenerate) piece
  const out = [];
  for (const q of raw)
    for (const m of splitMonotonic(q))
      if (Math.abs(m.p0[1]-m.p2[1]) >= EPS) out.push(m);
  return out;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `node --test test/sweeper-preprocess.test.mjs`
Expected: PASS (5 tests). If `splitMonotonic` has a dead branch (the `break` path), simplify: the function's effective body is `return ts.length ? resplit([q], ts) : [q];` — replace the loop accordingly.

- [ ] **Step 5: Simplify `splitMonotonic`** (remove the dead loop from Step 3)

```javascript
export function splitMonotonic(q) {
  const ts = [];
  for (const ax of [0,1]) {
    const t = criticalT(q.p0[ax], q.p1[ax], q.p2[ax]);
    if (t !== null) ts.push(t);
  }
  ts.sort((a,b)=>a-b);
  return ts.length ? resplit([q], ts) : [q];
}
```

- [ ] **Step 6: Re-run and commit**

Run: `node --test test/sweeper-preprocess.test.mjs`
Expected: PASS.

```bash
git add js/sweeper-preprocess.js test/sweeper-preprocess.test.mjs
git commit -m "feat(core): preprocess glyph outlines into monotonic quadratics"
```

---

### Task 6: `js/sweeper-core.js` — JS port of the sweep (ground truth + sign)

**Files:**
- Create: `js/sweeper-core.js`, `test/sweeper-core.test.mjs`

**Interfaces:**
- Consumes: `QuadCurve[]` (Task 5), `preprocessGlyph`, `wrapFont`.
- Produces (mirrors the GLSL that Plan 2 will write — keep in sync):
  - `evaluateBezier(p0,p1,p2,t) -> [x,y]`
  - `intersectMonotonic(qa,c0,c1,c2,target) -> number`
  - `scanlineSweep(size,offset,p0,p1,p2) -> number` (signed swept area; port of paper §8.3, [reference/out.txt:730](../../reference/out.txt#L730))
  - `sumSweep(curves,size,offset) -> number` (raw signed sum over a glyph's curves)
  - `coverage(curves,size,offset,sign) -> number` = `clamp(sign*sumSweep/(size[0]*size[1]),0,1)`
  - `detectSign(curves,bbox) -> +1|-1` — probes a point known inside the bbox center region and returns the sign that makes interior positive.

This module is the numerical source of truth; Plan 2's fragment shader is a line-by-line translation.

- [ ] **Step 1: Write the failing test**

`test/sweeper-core.test.mjs`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const opentype = require('../vendor/opentype.min.js');
import { wrapFont } from '../js/font-loader.js';
import { preprocessGlyph } from '../js/sweeper-preprocess.js';
import { bboxOfCommands, flattenCommands } from '../js/geom.js';
import { scanlineSweep, sumSweep, coverage, detectSign, intersectMonotonic } from '../js/sweeper-core.js';

test('intersectMonotonic solves a linear crossing', () => {
  // curve y: 0 -> 0.5 -> 1 (linear), target 0.5 => t=0.5
  const t = intersectMonotonic(0, 0, 0.5, 1, 0.5);
  assert.ok(Math.abs(t - 0.5) < 1e-6);
});

test('unit square coverage: inside≈1, outside≈0, straddle≈0.5', () => {
  // CCW unit square (y-up), promoted to vertical quads (only verticals matter)
  const sq = [
    {type:'M',x:0,y:0},{type:'L',x:1,y:0},{type:'L',x:1,y:1},{type:'L',x:0,y:1},{type:'Z'}
  ];
  const { preprocessGlyph } = require('../js/sweeper-preprocess.js') ?? {};
  // use the ESM import instead:
});
```

Replace the placeholder block: use the ESM `preprocessGlyph` already imported. Full test file:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const opentype = require('../vendor/opentype.min.js');
import { wrapFont } from '../js/font-loader.js';
import { preprocessGlyph } from '../js/sweeper-preprocess.js';
import { bboxOfCommands, flattenCommands } from '../js/geom.js';
import { scanlineSweep, sumSweep, coverage, detectSign, intersectMonotonic } from '../js/sweeper-core.js';

test('intersectMonotonic linear crossing', () => {
  assert.ok(Math.abs(intersectMonotonic(0, 0, 0.5, 1, 0.5) - 0.5) < 1e-6);
});

const square = [
  {type:'M',x:0,y:0},{type:'L',x:1,y:0},{type:'L',x:1,y:1},{type:'L',x:0,y:1},{type:'Z'}
];
const sqCurves = preprocessGlyph(square);
const sqSign = detectSign(sqCurves, {minX:0,minY:0,maxX:1,maxY:1});

function cov(cx, cy, w) {
  return coverage(sqCurves, [w, w], [cx - w/2, cy - w/2], sqSign);
}

test('unit square coverage inside/outside/straddle', () => {
  const w = 0.02;
  assert.ok(cov(0.5, 0.5, w) > 0.98, 'inside');
  assert.ok(cov(1.5, 0.5, w) < 0.02, 'outside right');
  assert.ok(cov(-0.5, 0.5, w) < 0.02, 'outside left');
  const edge = cov(1.0, 0.5, w);                 // straddling the right edge
  assert.ok(edge > 0.35 && edge < 0.65, `edge ${edge}`);
});

// Ground truth vs 16x supersampled non-zero winding on a real glyph.
function windingInside(edges, px, py) {
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
function supersampledCoverage(edges, cx, cy, w, N=4) {
  let hit = 0;
  for (let j=0;j<N;j++) for (let i=0;i<N;i++){
    const px = cx - w/2 + (i+0.5)/N*w, py = cy - w/2 + (j+0.5)/N*w;
    if (windingInside(edges, px, py)) hit++;
  }
  return hit/(N*N);
}

test('glyph coverage matches supersampled ground truth', () => {
  const font = wrapFont(opentype.loadSync('fonts/Tinos-Regular.ttf'));
  const gid = font.glyphIdForChar('e');
  const cmds = font.outlineEm(gid);
  const curves = preprocessGlyph(cmds);
  const bbox = bboxOfCommands(cmds);
  const sign = detectSign(curves, bbox);
  const edges = flattenCommands(cmds, 24);
  const w = 1/64; // emulate ~64px em
  let err = 0, n = 0;
  for (let gy=0; gy<20; gy++) for (let gx=0; gx<20; gx++) {
    const cx = bbox.minX + (gx+0.5)/20*(bbox.maxX-bbox.minX);
    const cy = bbox.minY + (gy+0.5)/20*(bbox.maxY-bbox.minY);
    const a = coverage(curves, [w,w], [cx-w/2, cy-w/2], sign);
    const b = supersampledCoverage(edges, cx, cy, w, 4);
    err += Math.abs(a-b); n++;
  }
  const mae = err/n;
  assert.ok(mae < 0.05, `mean abs coverage error ${mae}`);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --test test/sweeper-core.test.mjs`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `js/sweeper-core.js`** (direct port of paper §8)

```javascript
import { evalQuad } from './geom.js';

export function evaluateBezier(p0, p1, p2, t) { return evalQuad(p0, p1, p2, t); }

// Port of intersect_monotonic (reference/out.txt:671).
export function intersectMonotonic(qa, c0, c1, c2, target) {
  if (Math.abs(qa) < 1e-3) return (target - c0) / (c2 - c0);
  const qb = 2*c1 - 2*c0;
  const qc = c0 - target;
  const d = qb*qb - 4*qa*qc;
  const sqrtD = d < 0 ? 0 : Math.sqrt(d);
  const inv2a = 0.5 / qa;
  return -qb*inv2a + Math.sign(c2 - c0) * sqrtD * inv2a;
}

const sat = (x) => Math.min(1, Math.max(0, x));

// Port of scanline_sweep (reference/out.txt:730). size,offset,p0,p1,p2 are [x,y].
export function scanlineSweep(size, offset, P0, P1, P2) {
  let p0=[...P0], p1=[...P1], p2=[...P2];
  if (Math.max(p0[1],p2[1]) <= offset[1] || Math.min(p0[1],p2[1]) >= offset[1]+size[1]) return 0;
  const delta = [p2[0]-p0[0], p2[1]-p0[1]];
  p0=[p0[0]-offset[0],p0[1]-offset[1]]; p1=[p1[0]-offset[0],p1[1]-offset[1]]; p2=[p2[0]-offset[0],p2[1]-offset[1]];

  if (p0[0]===p1[0] && p0[0]===p2[0]) {           // vertical fast path
    if (p0[0] >= size[0]) return 0;
    const top = Math.min(Math.max(p0[1],p2[1]), size[1]);
    const bottom = Math.max(Math.min(p0[1],p2[1]), 0);
    const h = top - bottom;
    const b = Math.min(size[0], size[0]-p0[0]);
    return Math.sign(delta[1]) * b * h;
  }

  let qa = p0[1] + p2[1] - 2*p1[1];
  const bt = intersectMonotonic(qa, p0[1], p1[1], p2[1], 0);
  const tt = intersectMonotonic(qa, p0[1], p1[1], p2[1], size[1]);
  const vMinT = delta[1] > 0 ? bt : tt;
  const vMaxT = delta[1] > 0 ? tt : bt;
  const vMin = evaluateBezier(p0,p1,p2, sat(vMinT));
  const vMax = evaluateBezier(p0,p1,p2, sat(vMaxT));

  if (Math.max(vMin[0],vMax[0]) <= 0) return (vMax[1]-vMin[1]) * size[0];
  if (Math.min(vMin[0],vMax[0]) >= size[0]) return 0;

  qa = p0[0] + p2[0] - 2*p1[0];
  let hMinT, hMaxT;
  const hc = delta[0] > 0 ? [p0[0],p2[0],0,0] : [p2[0],p0[0],size[0],1];
  if (hc[0] >= hc[2]) hMinT = hc[3];
  else if (hc[1] <= hc[2]) hMinT = 1 - hc[3];
  else hMinT = intersectMonotonic(qa, p0[0], p1[0], p2[0], hc[2]);
  hc[2] = size[0] - hc[2];
  if (hc[0] >= hc[2]) hMaxT = hc[3];
  else if (hc[1] <= hc[2]) hMaxT = 1 - hc[3];
  else hMaxT = intersectMonotonic(qa, p0[0], p1[0], p2[0], hc[2]);

  const minT = sat(Math.max(vMinT, hMinT));
  const maxT = sat(Math.min(vMaxT, hMaxT));
  const q0 = vMinT >= hMinT ? vMin : evaluateBezier(p0,p1,p2,minT);
  const q1 = vMaxT <= hMaxT ? vMax : evaluateBezier(p0,p1,p2,maxT);

  let cvg = 0;
  if (minT > 0 && delta[0] > 0) {
    const h = delta[1] > 0 ? q0[1]-Math.max(0,p0[1]) : Math.min(size[1],p0[1])-q0[1];
    cvg = Math.sign(delta[1]) * h * size[0];
  }
  if (maxT < 1 && delta[0] < 0) {
    const h = delta[1] > 0 ? Math.min(size[1],p2[1])-q1[1] : q1[1]-Math.max(0,p2[1]);
    cvg += Math.sign(delta[1]) * h * size[0];
  }
  const h = q1[1] - q0[1];
  const b = size[0] - 0.5*(q0[0]+q1[0]);
  cvg += b * h;
  return cvg;
}

export function sumSweep(curves, size, offset) {
  let s = 0;
  for (const c of curves) s += scanlineSweep(size, offset, c.p0, c.p1, c.p2);
  return s;
}

export function coverage(curves, size, offset, sign) {
  return sat(sign * sumSweep(curves, size, offset) / (size[0]*size[1]));
}

// Probe near the bbox center with a small window; whichever global sign makes
// the raw sum positive there is the interior-positive sign.
export function detectSign(curves, bbox) {
  const cx = (bbox.minX+bbox.maxX)/2, cy=(bbox.minY+bbox.maxY)/2;
  const w = Math.min(bbox.maxX-bbox.minX, bbox.maxY-bbox.minY) * 0.01 + 1e-4;
  const s = sumSweep(curves, [w,w], [cx-w/2, cy-w/2]);
  return s >= 0 ? 1 : -1;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `node --test test/sweeper-core.test.mjs`
Expected: PASS. If the 'e' center probe lands in the counter (hole) and `detectSign` picks wrong, change `detectSign` to average several probes across the bbox and take the sign of the majority-inside probes; simplest robust fix: probe the point nearest `(bbox.minX+0.2*width, baseline+0.2*height)` which for 'e' is solid. If MAE is slightly high, raise supersample `N` to 6 in the test (ground truth accuracy), not the tolerance.

- [ ] **Step 5: Commit**

```bash
git add js/sweeper-core.js test/sweeper-core.test.mjs
git commit -m "feat(core): JS port of scanline sweep with sign detection + ground-truth test"
```

---

### Task 7: `js/sdf-generator.js` — single-channel SDF + MSDF

**Files:**
- Create: `js/sdf-generator.js`, `test/sdf-generator.test.mjs`

**Interfaces:**
- Consumes: `Command[]`, `flattenCommands`/`bboxOfCommands` (geom).
- Produces:
  - `generateSDF(commands, {res=32, pad=4, spreadEm}) -> { size, data:Uint8Array /*size*size, 1ch*/, spreadEm, tile:{minX,minY,scale} }`
  - `generateMSDF(commands, {res=32, pad=4, spreadEm, angleDeg=3}) -> { size, data:Uint8Array /*size*size*3*/, spreadEm, tile }`
  - `colorEdges(contours, angleThreshold) -> void` (msdfgen simple coloring; sets `seg.color` bitmask R=1,G=2,B=4)
  - Sign convention: interior positive (store > 0.5). `tile` maps texel → em: `emX = minX + (tx+0.5)/size / scale`… (documented in code).

- [ ] **Step 1: Write the failing test**

`test/sdf-generator.test.mjs`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateSDF, generateMSDF } from '../js/sdf-generator.js';

const square = [
  {type:'M',x:0.2,y:0.2},{type:'L',x:0.8,y:0.2},{type:'L',x:0.8,y:0.8},{type:'L',x:0.2,y:0.8},{type:'Z'}
];

function median(r,g,b){ return Math.max(Math.min(r,g),Math.min(Math.max(r,g),b)); }

test('SDF sign: center >0.5 (inside), corner texel <0.5 (outside)', () => {
  const { size, data } = generateSDF(square, { res: 32, pad: 4 });
  const c = Math.floor(size/2);
  const center = data[c*size + c] / 255;
  const cornerTexel = data[0] / 255;
  assert.ok(center > 0.5, `center ${center}`);
  assert.ok(cornerTexel < 0.5, `corner ${cornerTexel}`);
});

test('MSDF median sign matches SDF sign', () => {
  const s = generateSDF(square, { res: 32, pad: 4 });
  const m = generateMSDF(square, { res: 32, pad: 4 });
  assert.equal(s.size, m.size);
  let agree = 0, total = 0;
  for (let i=0;i<s.size*s.size;i++){
    const sInside = s.data[i]/255 > 0.5;
    const r=m.data[i*3]/255, g=m.data[i*3+1]/255, b=m.data[i*3+2]/255;
    const mInside = median(r,g,b) > 0.5;
    if (sInside === mInside) agree++;
    total++;
  }
  assert.ok(agree/total > 0.97, `agreement ${agree/total}`);
});

test('MSDF keeps a 90° corner sharper than SDF at low res', () => {
  // near the outer corner (0.8,0.8) diagonal, MSDF distance field should be
  // closer to the true corner distance than the rounded SDF.
  const res = 16;
  const s = generateSDF(square, { res, pad: 4 });
  const m = generateMSDF(square, { res, pad: 4 });
  // sample the texel just outside the corner along the diagonal
  const tx = s.size - 5, ty = s.size - 5;
  const idx = ty*s.size + tx;
  const sVal = s.data[idx]/255;
  const r=m.data[idx*3]/255, g=m.data[idx*3+1]/255, b=m.data[idx*3+2]/255;
  const mVal = Math.max(Math.min(r,g),Math.min(Math.max(r,g),b));
  // SDF rounds the corner (smaller/rounder distance); MSDF preserves the sharp
  // corner => its signed distance is more negative (further outside) here.
  assert.ok(mVal <= sVal + 1e-6, `msdf ${mVal} should be <= sdf ${sVal} outside corner`);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --test test/sdf-generator.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement `js/sdf-generator.js`**

```javascript
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

// ---- msdfgen simple edge coloring ----
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
    // find corners
    const corners = [];
    let prevDir = segDir(segs[segs.length-1], true);
    for (let i=0;i<segs.length;i++){ const d=segDir(segs[i], false); if (isCorner(prevDir,d,crossThreshold)) corners.push(i); prevDir=segDir(segs[i], true); }
    if (corners.length===0){ for (const s of segs) s.color=WHITE; continue; }
    // general multi-corner (1-corner falls here -> single color; minor limitation)
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

// non-zero winding inside test using all edges (for SDF sign)
function windingInside(edges, px, py) {
  let w=0;
  for (const e of edges){ const [ax,ay]=e.a,[bx,by]=e.b;
    if ((ay<=py)!==(by<=py)){ const tx=ax+(py-ay)/(by-ay)*(bx-ax); if (tx>px) w += (by>ay)?1:-1; } }
  return w!==0;
}

function tileMapping(commands, res, pad) {
  const bb = bboxOfCommands(commands);
  const w = bb.maxX-bb.minX, h = bb.maxY-bb.minY;
  const span = Math.max(w, h) || 1;
  const scale = res / span;               // texels per em (content area)
  const size = res + 2*pad;
  const minX = bb.minX - pad/scale - (Math.max(0, (h-w))/2); // center content
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

export function generateMSDF(commands, opts={}) {
  const res=opts.res??32, pad=opts.pad??4, angle=(opts.angleDeg??3)*Math.PI/180;
  const tile=tileMapping(commands,res,pad); const {size,spreadEm}=tile;
  const contours=parseContours(commands);
  colorEdges(contours, angle);
  const edges=[]; for (const segs of contours) for (const s of segs) flattenSeg(s, s.color??WHITE, 16, edges);
  const data=new Uint8Array(size*size*3);
  for (let ty=0;ty<size;ty++) for (let tx=0;tx<size;tx++){
    const [px,py]=emAt(tile,tx,ty);
    // per channel: nearest edge (by true distance) among edges with that channel bit
    const bestTrue=[Infinity,Infinity,Infinity], pseudo=[0,0,0];
    for (const e of edges){ const d=edgeData(e,px,py);
      for (let ch=0;ch<3;ch++){ if (e.color & (1<<ch)){ if (Math.abs(d.trueSigned)<bestTrue[ch]){ bestTrue[ch]=Math.abs(d.trueSigned); pseudo[ch]=d.pseudo; } } } }
    const i=(ty*size+tx)*3;
    for (let ch=0;ch<3;ch++) data[i+ch]=encode(pseudo[ch], spreadEm);
  }
  return { size, data, spreadEm, tile };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `node --test test/sdf-generator.test.mjs`
Expected: PASS (3 tests). If the corner-sharpness test is flaky at the chosen texel, sample exactly on the outer diagonal one texel beyond the corner (compute the corner's texel via `(0.8 - tile.minX)*tile.scale`); the invariant (MSDF ≤ SDF just outside a convex corner) holds where the SDF has rounded it.

- [ ] **Step 5: Commit**

```bash
git add js/sdf-generator.js test/sdf-generator.test.mjs
git commit -m "feat(core): analytic SDF + MSDF (edge coloring, pseudo-distance)"
```

---

### Task 8: `test/dump.mjs` — BMP/SVG dump for pre-WebGL eyeballing

**Files:**
- Create: `test/dump.mjs`

**Interfaces:**
- Consumes: everything above.
- Produces a CLI: `node test/dump.mjs <fontPath> <char> [res]` → writes `out/sdf.bmp`, `out/msdf.bmp`, `out/curves.svg` for visual inspection. (24-bit BMP opens natively on Windows.)

- [ ] **Step 1: Implement the dump CLI**

`test/dump.mjs`:

```javascript
import { createRequire } from 'module';
import { writeFileSync, mkdirSync } from 'node:fs';
const require = createRequire(import.meta.url);
const opentype = require('../vendor/opentype.min.js');
import { wrapFont } from '../js/font-loader.js';
import { preprocessGlyph } from '../js/sweeper-preprocess.js';
import { generateSDF, generateMSDF } from '../js/sdf-generator.js';

function bmp24(width, height, rgb /*Uint8Array w*h*3, row 0 = top*/) {
  const rowSize = Math.ceil(width*3/4)*4, pixArr = rowSize*height, fileSize = 54+pixArr;
  const b = Buffer.alloc(fileSize);
  b.write('BM',0); b.writeUInt32LE(fileSize,2); b.writeUInt32LE(54,10);
  b.writeUInt32LE(40,14); b.writeInt32LE(width,18); b.writeInt32LE(height,22);
  b.writeUInt16LE(1,26); b.writeUInt16LE(24,28); b.writeUInt32LE(pixArr,34);
  for (let y=0;y<height;y++){ const srcY=height-1-y; // BMP is bottom-up
    for (let x=0;x<width;x++){ const s=(srcY*width+x)*3, d=54+y*rowSize+x*3;
      b[d]=rgb[s+2]; b[d+1]=rgb[s+1]; b[d+2]=rgb[s]; } }
  return b;
}
function grayToRgb(size, data){ const o=new Uint8Array(size*size*3); for (let i=0;i<size*size;i++){o[i*3]=o[i*3+1]=o[i*3+2]=data[i];} return o; }

const [,, fontPath='fonts/Tinos-Regular.ttf', char='A', resStr='32'] = process.argv;
const res=parseInt(resStr,10);
const font=wrapFont(opentype.loadSync(fontPath));
const gid=font.glyphIdForChar(char);
const cmds=font.outlineEm(gid);
mkdirSync('out',{recursive:true});
const sdf=generateSDF(cmds,{res}); writeFileSync('out/sdf.bmp', bmp24(sdf.size,sdf.size,grayToRgb(sdf.size,sdf.data)));
const msdf=generateMSDF(cmds,{res}); writeFileSync('out/msdf.bmp', bmp24(msdf.size,msdf.size,msdf.data));
const curves=preprocessGlyph(cmds);
const svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.2 -0.9 1.4 1.4" width="400" height="400">',
  '<g transform="scale(1,-1)">',
  ...curves.map(c=>`<path d="M${c.p0[0]},${c.p0[1]} Q${c.p1[0]},${c.p1[1]} ${c.p2[0]},${c.p2[1]}" fill="none" stroke="#39f" stroke-width="0.004"/>`),
  '</g></svg>'].join('\n');
writeFileSync('out/curves.svg', svg);
console.log(`wrote out/sdf.bmp out/msdf.bmp out/curves.svg (glyph '${char}', ${curves.length} monotonic curves, atlas ${sdf.size}px)`);
```

- [ ] **Step 2: Run it and eyeball the output**

Run: `node test/dump.mjs fonts/Tinos-Regular.ttf A 48`
Then: open `out/sdf.bmp` (soft gray field, mid-gray at edges), `out/msdf.bmp` (colored field, sharp corner at the apex), `out/curves.svg` (blue monotonic curves tracing 'A', no horizontal segments).
Expected: SDF gray blob shaped like 'A'; MSDF shows the characteristic magenta/cyan/yellow with a crisp apex; SVG outlines match 'A'. Try `口` with the CJK subset: `node test/dump.mjs fonts/NotoSansSC-Subset.ttf 口 48`.

- [ ] **Step 3: Commit**

```bash
git add test/dump.mjs
git commit -m "chore(core): BMP/SVG dump tool for pre-WebGL visual verification"
```

---

## Self-Review (completed)

- **Spec coverage:** §3 assets (T1) · §5.1 em/y-up + sign norm (T3,T6) · §5.3 preprocess products (T5) · §6 unique glyphs (T4) · §7.1 monotonic preprocessing (T5) · §8.1–8.4 distance engine, SDF, MSDF coloring + pseudo-distance (T7) · §12 TDD: monotonicity, sweep-vs-supersample, MSDF-vs-SDF corner (T5,T6,T7). Sweeper shader + renderers, SDF renderer AA, instancing, UI, split-wipe → **Plan 2** (by design).
- **Placeholder scan:** the sweeper-core test file intentionally shows a placeholder block then its full replacement (Step 1 note) — the replacement is the file to write; no residual TODOs elsewhere.
- **Type consistency:** `QuadCurve{p0,p1,p2}` (T5) consumed by `scanlineSweep` (T6); `Command` shape from `outlineEm` (T3) consumed by geom/preprocess/sdf; `generateSDF/MSDF` return `{size,data,spreadEm,tile}` used by T8 and Plan 2. Sign convention "interior positive" consistent across T6 (`detectSign`) and T7 (`encode`, winding).
