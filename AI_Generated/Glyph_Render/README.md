# Scanline Sweeper vs SDF/MSDF — 字形渲染对比 Demo

Interactive WebGL2 demo comparing three glyph renderers on identical text via a
draggable split-screen wipe (left = Sweeper, right = SDF/MSDF):

- **Scanline Sweeper** — analytic, per-pixel coverage integrated from monotonic
  quadratic Béziers stored in a data texture. Resolution-independent; stays crisp
  at any magnification. Implements *The Scanline Sweeper* (`reference/scanline_sweeper_preprint.pdf`).
- **Single-channel SDF** (Green 2007) and **MSDF** (Chlumský 2015) — baked
  distance-field atlases, sampled with hardware filtering. Classic baselines that
  round corners / wobble as the atlas resolution drops.

The comparison is the point: raise the font size or drop the SDF resolution and the
SDF side degrades while the Sweeper side stays sharp.

## Run

No build step. Serve over HTTP (ES modules need a server, not `file://`):

```bash
python -m http.server 8080
# open http://localhost:8080/
```

Any static server works.

## Controls

- **Font** — built-in Tinos (≈Times New Roman), Geist, Dancing Script, Noto Sans SC
  (Chinese subset), or **Upload** any `.ttf`/`.otf`.
- **SDF mode** — Single SDF or MSDF (which variant fills the right half).
- **SDF resolution** — 16–64 px/em atlas tiles.
- **Font size**, **Gamma** (Sweeper), **multiline text** (Latin or CJK).
- **Drag** to pan, **wheel** to zoom (around the cursor), drag the **⇄ divider** to wipe.
- Live readout: unique glyphs, sweeper curve count + texture size, SDF atlas size, rebuild time.

### Deep links

State can be encoded in the URL, e.g.
`?font=NotoSansSC-Subset.ttf&text=永字&mode=msdf&res=16&size=300&slider=0.5&zoom=2`.

## Structure

```
index.html               UI, canvas, control panel
vendor/opentype.min.js   font parsing (browser <script> global); opentype.cjs for Node
fonts/                   vendored open fonts incl. CJK subset (see fonts/LICENSES.md)
js/geom.js               bezier eval, bbox, flatten
js/font-loader.js        opentype -> em-space y-up outlines
js/layout.js             text -> positioned glyph instances (advances, kerning, multi-line)
js/sweeper-preprocess.js outlines -> directed monotonic quadratics
js/sweeper-core.js       JS port of the sweep (ground-truth / mirrors the GLSL)
js/sweeper-renderer.js   curve data texture + instanced sweep shader (one draw call)
js/sdf-generator.js      analytic SDF + MSDF (edge coloring, pseudo-distance, error correction)
js/sdf-renderer.js       SDF/MSDF atlas + instanced sampling shader (one draw call)
js/main.js               app: UI, split-wipe, pan/zoom, rebuild-on-change contract
tools/dump.mjs           dump a glyph's SDF/MSDF/curves to out/ for eyeballing
test/                    Node unit tests (no GL)
docs/superpowers/        design spec + implementation plans
```

The `js/` core (font-loader, layout, preprocess, sweeper-core, sdf-generator) is
dependency-light and unit-tested in Node; `js/*-renderer.js` + `main.js` are the
WebGL2 layer. Textures are rebuilt only when the character set changes (text / font /
SDF-resolution); pan, zoom, size, gamma and the wipe are uniform-only.

## Tests

```bash
npm test                                          # Node core tests (24)
node tools/dump.mjs fonts/Tinos-Regular.ttf A 48  # -> out/sdf.bmp, msdf.bmp, curves.svg
```

Key checks: monotonic subdivision is exact; the sweep matches a 36× supersampled
winding rasterizer on a real glyph (MAE < 0.05); MSDF sign agrees with SDF on real
fonts; MSDF carries multi-edge information at corners.

## Limitations

No font hinting (small-size stems don't snap to the pixel grid — paper §5.1); contour
overlaps are clamped rather than per-contour averaged (§4.3); simple layout (advances +
kerning, no complex shaping / ligatures); no curve acceleration structure (each fragment
loops the glyph's curves — paper §4.1); the bundled CJK font is a Simplified-Chinese
subset (GB2312 levels 1&2) — upload a fuller font for other CJK.

## Requires

A WebGL2 browser (Chrome, Edge, Firefox, Safari 15+).
