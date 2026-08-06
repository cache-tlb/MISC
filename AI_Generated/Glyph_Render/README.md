# Scanline Sweeper vs SDF/MSDF — 字形渲染对比 Demo

Interactive WebGL2 demo comparing three glyph renderers on identical text via a
draggable split-screen wipe (left = Sweeper, right = SDF/MSDF):

- **Scanline Sweeper** — analytic, per-pixel coverage integrated from monotonic
  quadratic Béziers stored in a data texture. Resolution-independent; stays crisp
  at any magnification. Implements *The Scanline Sweeper* (`reference/scanline_sweeper_preprint.pdf`).
  Works in **3D** too: glyphs lie on a plane in space, and each pixel's em-space
  footprint comes from the screen-space derivatives of the glyph coordinate,
  decomposed into N axis-aligned sweep windows (paper §4.4).
- **Single-channel SDF** (Green 2007) and **MSDF** (Chlumský 2015) — baked
  distance-field atlases, sampled with hardware filtering. Classic baselines that
  round corners / wobble as the atlas resolution drops.

The comparison is the point: raise the font size or drop the SDF resolution and the
SDF side degrades while the Sweeper side stays sharp. Switch to the 3D view and pick
the grazing camera preset for the sharpest version of that contrast — the Sweeper can
tighten its filter footprint anisotropically, a baked distance field cannot.

## Run

No build step. Serve over HTTP (ES modules need a server, not `file://`):

```bash
python -m http.server 8080
# open http://localhost:8080/
```

Any static server works.

## Controls

- **View** — **2D 正交** (default) or **3D 平面**: the text lies on a grid plane in
  space under a perspective camera.
- **Camera** (3D only) — 正面 / 45° / 掠射 presets, plus an FOV slider.
- **Anisotropy window cap N** — 1–8. The shader uses `N = ⌈ρ⌉` capped by this,
  where ρ is the footprint's anisotropy ratio. At N=1 the window is plain
  `fwidth`; raise it at a grazing angle and the Sweeper half visibly sharpens.
  In 2D, ρ=1, so this costs nothing whatever it is set to.
- **Font** — built-in Tinos (≈Times New Roman), Geist, Dancing Script, Noto Sans SC
  (Chinese subset), or **Upload** any `.ttf`/`.otf`.
- **SDF mode** — Single SDF or MSDF (which variant fills the right half).
- **SDF resolution** — 16–64 px/em atlas tiles.
- **Font size** (in 3D it drives the camera in, raising screen pixels per em),
  **Gamma** (Sweeper), **multiline text** (Latin or CJK).
- **Drag** — pans in 2D, orbits in 3D. **Wheel** — zooms around the cursor in 2D,
  dollies in 3D. Drag the **⇄ divider** to wipe.
- Live readout: unique glyphs, sweeper curve count + texture size, SDF atlas size,
  view mode, the live anisotropy ratio ρ and the resulting window count, rebuild time.

### Deep links

State can be encoded in the URL, e.g.
`?font=NotoSansSC-Subset.ttf&text=永字&mode=msdf&res=16&size=300&slider=0.5&zoom=2`.

3D adds `view=3d`, `preset=front|angled|grazing`, `yaw`, `pitch`, `dist`, `fov`, `aniso` —
e.g. the grazing-angle stress test:
`?view=3d&preset=grazing&aniso=8&size=260&text=Sphinx`

## Structure

```
index.html               UI, canvas, control panel
vendor/opentype.min.js   font parsing (browser <script> global); opentype.cjs for Node
fonts/                   vendored open fonts incl. CJK subset (see fonts/LICENSES.md)
js/geom.js               bezier eval, bbox, flatten
js/font-loader.js        opentype -> em-space y-up outlines
js/layout.js             text -> positioned glyph instances (advances, kerning, multi-line)
js/mat4.js               column-major mat4; ortho2d reproduces the old 2D mapping exactly
js/scene3d.js            orbit camera -> MVP, plane model, em/pixel Jacobian probe
js/sweeper-preprocess.js outlines -> directed monotonic quadratics
js/sweeper-core.js       JS port of the sweep (ground-truth / mirrors the GLSL)
js/sweeper-footprint.js  footprint assembly: pixel parallelogram -> N sweep windows
js/sweeper-renderer.js   curve data texture + instanced sweep shader (one draw call)
js/sdf-generator.js      analytic SDF + MSDF (edge coloring, pseudo-distance, error correction)
js/sdf-renderer.js       SDF/MSDF atlas + instanced sampling shader (one draw call)
js/plane-renderer.js     grid backdrop plane for the 3D view
js/main.js               app: UI, split-wipe, 2D pan/zoom + 3D orbit, rebuild-on-change contract
tools/dump.mjs           dump a glyph's SDF/MSDF/curves to out/ for eyeballing
test/                    Node unit tests (no GL)
docs/superpowers/        design spec + implementation plans
```

The `js/` core (font-loader, layout, mat4, scene3d, preprocess, sweeper-core,
sweeper-footprint, sdf-generator) is dependency-light and unit-tested in Node;
`js/*-renderer.js` + `main.js` are the WebGL2 layer. Textures are rebuilt only when the
character set changes (text / font / SDF-resolution); pan, zoom, size, gamma, the wipe,
**the camera and the window cap N** are uniform-only.

`sweeper-core.js` is untouched by the 3D work: it still answers only "coverage of one
rectangular window", exactly as the paper frames it, and `sweeper-footprint.js` sits
above it deciding what those windows are.

## Tests

```bash
npm test                                          # Node core tests (44)
node tools/dump.mjs fonts/Tinos-Regular.ttf A 48  # -> out/sdf.bmp, msdf.bmp, curves.svg
```

Key checks: monotonic subdivision is exact; the sweep matches a supersampled winding
rasterizer on a real glyph (MAE < 0.05); MSDF sign agrees with SDF on real fonts; MSDF
carries multi-edge information at corners. For 3D: `ortho2d` reproduces the pre-matrix
2D vertex mapping and the isotropic footprint path is bit-identical to the old
single-window `coverage()` (together these lock "2D is unchanged"); the N windows tile
the pixel parallelogram exactly; and against a supersampled **parallelogram** ground
truth at anisotropy ≈8, mean coverage error drops from **0.090** (single AABB window)
to **0.0097** (8 windows).

## Limitations

No font hinting (small-size stems don't snap to the pixel grid — paper §5.1); contour
overlaps are clamped rather than per-contour averaged (§4.3); simple layout (advances +
kerning, no complex shaping / ligatures); no curve acceleration structure (each fragment
loops the glyph's curves — paper §4.1); the bundled CJK font is a Simplified-Chinese
subset (GB2312 levels 1&2) — upload a fuller font for other CJK.

3D specifics: the shear-transform route (paper §4.4, 2×2 QR + shear on control points)
is **not** implemented — it loses curve monotonicity and would need in-shader
re-subdivision; the multi-window scheme is a piecewise box approximation, not true
anisotropic filtering, so blur remains past the N=8 cap; the footprint is treated as a
parallelogram (the weak-perspective approximation, paper footnote 1) rather than the
trapezoid strict perspective gives; depth testing is off, so 3D supports a single
coplanar plane, not mutually occluding geometry.

## Requires

A WebGL2 browser (Chrome, Edge, Firefox, Safari 15+).
