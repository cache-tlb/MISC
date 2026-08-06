# Participating Media Renderer — Design

Date: 2026-07-31
Status: approved

A single-page WebGL2 demo that renders a VDB cloud as a participating medium
inside a lit, shadowed room, following the concepts in
<https://alextardif.com/ParticipatingMedia.html> adapted from offline delta
tracking to a realtime ray march.

## Goals

- Load `assets/bunny_cloud.vdb` in the browser, with our own OpenVDB reader.
- Render it as a light-scattering, light-absorbing medium that interacts both
  ways with the opaque scene: the medium shadows surfaces, and the opaque
  geometry shadows the medium.
- Interactive frame rates at 1080p on a mid-range GPU.
- No external JavaScript libraries at all.

## Non-goals

- Path tracing / delta tracking. The reference article is offline; we take its
  physical model and its warnings about darkening, not its sampling strategy.
- Spectral rendering. RGB throughout.
- Emissive or heterogeneous-albedo media. One user-chosen medium colour.
- Anything resembling a build step. Plain `<script>` tags.

## Asset facts

Established by inspecting the files, not assumed.

### `assets/bunny_cloud.vdb`

| Property | Value |
|---|---|
| File format version | 223 (library 12.1, written by Houdini) |
| Grids | 1, named `density`, type `Tree_float_5_4_3` |
| Compression flags | `6` = `COMPRESS_ACTIVE_MASK (2) \| COMPRESS_BLOSC (4)` |
| Half float | no — values are 4-byte floats |
| Active bbox | `[-300, -47, -208] .. [276, 524, 229]` → 577 × 572 × 438 |
| Active voxels | 19 210 271 |
| Grid data offset | 174 |
| Leaf buffer offset | 4 574 480 |
| End offset | 76 328 146 |
| Class | `fog volume` |

Tree shape `5_4_3` means `Root → Internal<32³> → Internal<16³> → Leaf<8³>`.

Topology occupies bytes 174..4 574 480 and leaf voxel data occupies
4 574 480..76 328 146. Leaf data is ~72 MB compressed vs ~77 MB raw, which is
the expected poor ratio for LZ4 on float data and confirms the layout.

The file carries `file_delayed_load` metadata. We ignore it: it exists to let
OpenVDB seek within a memory-mapped file, and the buffers are still written
sequentially, which is how we read them.

### `assets/teapot/teapot.obj`

- 7850 positions, 8334 texcoords, 8028 faces.
- Faces are **mixed**: 7676 quads and 352 triangles, with `v/vt` indices only.
  They triangulate to 15704 triangles / 47112 indices.
- **No vertex normals in the file** — we must generate them.
- Bbox `(-71.893, 0, -47.928) .. (82.294, 79.007, 47.928)`, i.e. 154 × 79 × 96,
  already Y-up and already resting on `y = 0`, but off-centre in X (centre
  x ≈ 5.2), so we recentre on X and Z at load time.
- `default.png` is a 128 × 128 1-bit indexed grey checker. Browsers decode PNG
  natively, so no image library is needed.

## Coordinate system

Right handed, matching OpenGL and `camera.js`: **+x right, +y up, +z toward the
viewer**, camera looks down −z. Yaw rotates about +y by the right-hand rule, so
positive yaw turns left; pitch is positive looking up.

## File layout

| File | Responsibility | New? |
|---|---|---|
| `index.html` | canvas, UI panel markup, script tags | new |
| `src/m4.js` | matrix math | given, **unchanged** |
| `src/webgl-utils.js` | program/buffer helpers | given, **must not be modified** |
| `src/camera.js` | UE-style fly camera | given, **unchanged** |
| `src/gl-helpers.js` | textures, 3D textures, FBOs, fullscreen quad | new |
| `src/blosc.js` | Blosc container + LZ4 + byte unshuffle | new |
| `src/vdb-loader.js` | OpenVDB reader → dense downsampled grid | new |
| `src/obj-loader.js` | OBJ parse, triangulate, generate normals | new |
| `src/shaders.js` | all GLSL ES 3.00 sources | new |
| `src/scene.js` | room geometry, scene state, defaults | new |
| `src/renderer.js` | the four passes | new |
| `src/ui.js` | control panel wiring | new |
| `src/main.js` | bootstrap, asset load, frame loop | new |

`m4.js` already exposes everything needed (`perspective`, `orthographic`,
`lookAt`, `inverse`, `multiply`, `compose`, `transformPoint`,
`transformDirection`, …). No additions are required. If that turns out to be
wrong mid-implementation, additions go at the bottom of `m4.js` in its existing
style and get added to its export table.

`webgl-utils.js` provides `createProgramInfo`, `createBufferInfoFromArrays`,
`setBuffersAndAttributes`, `setUniforms`, `drawBufferInfo`,
`resizeCanvasToDisplaySize`. It has **no** texture or framebuffer helpers, which
is why `gl-helpers.js` exists.

> **Revised during implementation.** `createProgramInfo` turned out to be
> unusable here. Its uniform setter factory only handles `SAMPLER_2D` and
> `SAMPLER_CUBE` and throws `unknown type: 0x8b5f` on `SAMPLER_3D` — which
> every pass in this demo needs. Since that file must not be modified,
> `gl-helpers.js` gained `createProgram` and `setUniforms`: linking is still
> delegated to `webgl-utils`' `createProgramFromSources`, and attributes still
> use its `createAttributeSetters`, but uniform introspection and binding are
> ours. Two related constraints: array keys are mapped to attribute names with
> an `a_` prefix, so shaders declare `a_position`/`a_normal`/`a_texcoord`; and
> `drawBufferInfo` hardcodes `UNSIGNED_SHORT`, so the renderer issues its own
> `drawElements` with the index type the mesh actually uses.

Every new module follows the existing UMD-ish IIFE pattern from `camera.js` and
attaches one global.

## VDB loading

### `src/blosc.js`

Interface: `blosc.decompress(bytes, offset) -> Uint8Array`.

Blosc 1.x container: 16-byte header (version, versionlz, flags, typesize,
`nbytes`, `blocksize`, `cbytes`), then an `int32` offset table of
`ceil(nbytes / blocksize)` entries, then the blocks. Flags bit 0 is byte
shuffle, bit 1 is memcpy-ed, bits 5-7 are the compressor id. Within a block,
data may be split into `typesize` sub-streams, each prefixed by an `int32`
compressed length, where a length equal to the sub-block size means stored
uncompressed.

OpenVDB always compresses with LZ4 and byte shuffle at `typesize = sizeof(T)`,
so LZ4 block decompression plus a byte unshuffle is all we implement. Any other
compressor id throws with a clear message rather than returning garbage.

### `src/vdb-loader.js`

Interface:

```js
vdbLoader.load(arrayBuffer, {maxResolution: 256}) -> {
  data,          // Uint8Array, density normalised to 0..255
  dims,          // [nx, ny, nz]
  maxDensity,    // float, multiply the sampled 0..1 back up by this
  worldSize,     // [sx, sy, sz] extent implied by the VDB transform
  name,          // grid name
}
```

Read order, mirroring `openvdb::io::Archive`:

1. Header: 8-byte magic, `uint32` version, library major/minor, `uint8`
   `hasGridOffsets`, 36-char UUID. Version 223 stores compression **per grid**,
   so there is no file-level compression word.
2. File metadata map.
3. `int32` grid count, then per grid: unique name, type string, instance parent,
   and three `uint64` offsets (`gridPos`, `blockPos`, `endPos`).
4. Seek to `gridPos`: `uint32` compression flags, grid metadata map, transform.
5. `Tree::readTopology`: `uint32` buffer count, then the root node — background
   value, tile count, child count, then each child's `vec3i` origin followed by
   the child node, and each tile's origin, value and active flag.
6. `InternalNode::readTopology`: child mask, value mask, then (file version
   ≥ 222) compressed tile values via `readCompressedValues`, then recurse into
   set child bits.
7. `LeafNode::readTopology`: value mask only.
8. `Tree::readBuffers`: per leaf, `readCompressedValues` into the 512-value
   voxel buffer.

`readCompressedValues` for our flags:

- `uint8` metadata flag, one of `NO_MASK_OR_INACTIVE_VALS(0)`,
  `NO_MASK_AND_MINUS_BG(1)`, `NO_MASK_AND_ONE_INACTIVE_VAL(2)`,
  `MASK_AND_NO_INACTIVE_VALS(3)`, `MASK_AND_ONE_INACTIVE_VAL(4)`,
  `MASK_AND_TWO_INACTIVE_VALS(5)`, `NO_MASK_AND_ALL_VALS(6)`.
- Inactive values are read as dictated by that flag; a selection mask follows
  for flags 4 and 5.
- Unless the flag is `NO_MASK_AND_ALL_VALS`, only the values whose value-mask
  bit is set are stored; read that many and scatter them back through the mask,
  filling the rest with the appropriate inactive value.
- The payload itself is `int64` compressed byte count then a Blosc block; a
  count `<= 0` means the following `-count` bytes are stored uncompressed.

**`maxResolution` is a cap, not a target.** The accumulation scatters each
source voxel into exactly one target cell, so a target grid *larger* than the
source leaves regular planes of cells that receive nothing at all. For
`smoke.vdb` (source 111 × 222 × 112) the original code stretched that to
128 × 256 × 129 and left **34.7% of cells empty** in a periodic 3D lattice —
each of those divided its sum by a zero count, produced `NaN`, and became a
silent `0` on the way into the `Uint8Array`. On screen it read as a grid
running through the volume. The scale factor is now clamped with
`Math.max(1, longest / maxResolution)`, so small grids stay 1:1, and
`axisCounts` clamps to at least 1 so a zero divisor is impossible rather than
merely unlikely.

**Downsampling.** The dense grid would be 144 M voxels, so we never build it.
The target grid is sized so the longest source axis maps to `maxResolution` and
the others stay proportional — 577 × 572 × 438 at 256 gives 256 × 254 × 194
(12.6 M cells). Each leaf's active voxels are box-filter accumulated directly
into a `Float32Array` sum and divided by the constant source-cells-per-target-cell
volume, which conserves total density. The result is normalised by the observed
maximum into `Uint8Array` and uploaded as an `R8` `TEXTURE_3D` (12.6 MB); the
shader multiplies the sampled `0..1` back by `maxDensity`.

**Half float grids and multi-grid files** (added after the initial build, when
the assets were swapped). A grid whose type ends in `_HalfFloat`, or whose
metadata sets `is_saved_as_half_float`, stores its leaf and internal-node tile
buffers as 16-bit values. Two details matter and neither is guessable:

- Inactive values stay **full width** even in a half grid — OpenVDB reads them
  with `sizeof(ValueType)`, outside the `HalfReader` path that narrows the bulk
  buffer.
- `HalfReader::read` returns immediately on an empty buffer, **without** reading
  the compressed-size header that the full-width path always reads. Internal
  nodes with no active tiles are the common case, so getting this backwards
  desynchronises a half grid almost at once.

Conversion goes through a 65536-entry lookup table built once, since a grid can
hold millions of values.

Files may contain several grids — a simulation cache typically carries
`density`, `temperature` and a `vec3s` velocity field. Grid descriptors are
**not** contiguous: each is immediately followed by its own data, so the next
one starts at the previous grid's `endPos`. The loader walks them all, keeps
only scalar `Tree_float_*` trees, and prefers the one named `density`;
`options.gridName` overrides.

**The leaf walk is checked against `endPos`.** After reading every leaf buffer
the cursor must land exactly on the grid's recorded end offset. A mismatch
means a value width or block header was misread, which would otherwise surface
only as a plausible-looking but wrong cloud.

**Verification before any rendering work.** `blosc.js` and `vdb-loader.js` are
developed and run under Node against three invariants taken from the file's own
metadata:

1. Decoded active voxel count equals `19 210 271`.
2. Decoded active bbox equals `[-300,-47,-208] .. [276,524,229]`.
3. Box filtering never produces a value above the source maximum density.

> **Corrected during implementation.** Invariant 3 originally read "all decoded
> densities lie in `[0, 1]`, it is a fog volume". That is false for this file:
> the values span `0 .. 2.7922983` with a mean of `0.3044`, and 4.5% of them
> exceed `1.0`. Houdini fog volumes are not normalised. The replacement
> invariant is the one that actually constrains the filter — an average over a
> cell cannot exceed the largest value in that cell.

These are hard gates. Rendering work does not start until they pass.

## OBJ loading

`objLoader.parse(text) -> {position, normal, texcoord, indices}` shaped for
`webglUtils.createBufferInfoFromArrays`.

- Handles `v`, `vt`, `f`, ignores everything else.
- Faces may have any vertex count; triangulate as a fan.
- Index forms `v`, `v/vt`, `v//vn`, `v/vt/vn` are all accepted; negative indices
  are resolved relative to the current count.
- Vertices are deduplicated on the `v/vt/vn` triple.
- Since the teapot has no normals, normals are **area-weighted accumulated per
  face and normalised** — this smooths the teapot correctly. If a file does
  supply normals we use them.
- Positions are recentred on X and Z about their bbox centre, with the bbox
  minimum Y moved to 0, so UI position `(0, 0, 0)` puts the teapot standing on
  the floor at the origin.

## Scene

- **Floor**: 100 × 100 square centred at the origin on `y = 0`, normal +y.
- **Walls**: four, at `x = ±50` and `z = ±50`, 50 units tall, facing inward.
  The top is open so the light can reach in.
- **Checkerboard**: procedural in the fragment shader from world-space
  coordinates, 10-unit cells (10 across the room), greys `0.32` and `0.82`.
  Box filtered analytically over the pixel footprint, after Inigo Quilez, so
  grazing angles fade toward the average grey instead of aliasing.

> **Corrected during implementation.** The first version integrated the wrong
> wave. The box filter needs the antiderivative of a **square** wave of period
> two, `2·|fract(x/2) − 0.5|`, whose slope is exactly ±1. What was written,
> `x/2 + h·(1 − 2|h|)`, is the antiderivative of a *triangle* wave ramping
> 0 → 1, so the pattern came out as a product of soft ramps — visually blurred
> blobs rather than squares — and spanned only half the intended range
> (`0.005 … 0.500` instead of `0 … 1`), which squeezed the albedo into a narrow
> mid-grey band. The difference is also taken low minus high, `(p − w/2) −
> (p + w/2)`, since that ordering carries the sign. Filter width is
> `max(|dFdx|, |dFdy|)` rather than `fwidth`: `fwidth` sums the two
> derivatives, overstating the footprint and blurring the pattern too early.
- **Teapot**: loaded model, default uniform scale `0.3` (→ 46 × 24 × 29,
  standing on the floor at the origin). Albedo is a warm off-white
  `(0.85, 0.80, 0.72)` modulated by `default.png`.
- **Medium**: the VDB grid, placed by a model matrix from the UI. Its unit-cube
  local space is stretched to the grid's aspect ratio (577 : 572 : 438 →
  1 : 0.991 : 0.759) before the UI transform, so uniform scale keeps it
  undistorted. Defaults: position `(0, 38, 0)`, rotation `(0, 0, 0)`, uniform
  scale `32` → a 32 × 31.7 × 24.3 volume spanning `y` 22.1 to 53.9, clear of
  the teapot's 23.7 top and above the 50-unit walls at its crown.

### Default scene values

| Parameter | Default | Range |
|---|---|---|
| Light yaw | 30° | −180°..180° |
| Light pitch | −50° | −89°..89° |
| Light intensity | 3.0 | 0..10 |
| Ambient intensity | 0.35 | 0..2 |
| Teapot position | (0, 0, 0) | ±40 per axis |
| Teapot rotation | (0, 0, 0) | −180°..180° per axis |
| Teapot scale | 0.3 | 0.05..1.0 |
| Medium position | (0, 38, 0) | ±50 per axis |
| Medium rotation | (0, 0, 0) | −180°..180° per axis |
| Medium scale | 32 | 5..80 |
| Medium density multiplier | tuned at implementation | 0..3 |
| Medium colour | near-white, slightly cool | colour picker |

Light **direction** — the direction light travels — is derived from yaw/pitch
with the same formula `camera.js` uses for forward, so the two agree:
`dir = (−sin(yaw)·cos(pitch), sin(pitch), −cos(yaw)·cos(pitch))`. Pitch −50°
therefore points downward. Light colour is a fixed white; the brief asks only
for direction and intensity.

## Rendering

Four passes per frame.

### Pass 0 — Shadow map

- 2048 × 2048 `DEPTH_COMPONENT24` texture, no colour attachment.
- Light direction comes from UI yaw/pitch. The orthographic frustum is refit
  each frame: take the scene AABB (room + teapot + medium), transform its eight
  corners into light space, and take the min/max as the ortho bounds.
- **Opaque only** — floor, walls, teapot. The medium is never in the shadow map;
  its shadowing is computed analytically during shading.
- Sampled with 3 × 3 PCF and a slope-scaled depth bias.

### Pass 1 — Opaque

Target: `RGBA16F` colour + `DEPTH_COMPONENT24` **texture** (not a renderbuffer,
because the volume pass reads it).

Directional light:

```
shadow = PCF(shadowMap, worldPos)
if shadow > 0:
    TrLight = transmittance from worldPos toward the light through the medium
              (~12 ray-march steps)
    Lo += albedo / PI * lightColor * lightIntensity * TrLight * shadow * max(dot(N, L), 0)
```

Ambient — **the decision the brief delegated**: hemispheric sky/ground radiance,
occluded by the medium.

```
skyVis  = transmittance from worldPos straight up (+Y) through the medium (~8 steps)
Lo     += ambientIntensity * mix(groundColor, skyColor, N.y * 0.5 + 0.5) * albedo * skyVis
```

Rationale: it is one consistent physical story — ambient is uniform sky
radiance arriving over the hemisphere, and the medium blocks part of it. It
makes the cloud lay a soft ambient shadow on the floor, so the hard directional
shadow does not read as floating unsupported. Sky and ground colours are fixed
constants (a cool sky, a warm-neutral ground); only the intensity is exposed, as
the brief specifies.

> **Revised during implementation.** `skyVis` was originally a single ray
> straight up. Rendered, that stamps the medium's exact footprint onto the
> floor and walls as a hard-edged, pitch-black column, because every point
> below the cloud samples the same worst-case direction. Ambient is an integral
> over the hemisphere, so `skyVisibility` now averages **four** marches — one
> vertical and three tilted 35° apart — and blends the result with
> `mix(1.0, avg, 0.8)`, since the sampled cone covers only part of the
> hemisphere and the medium should never remove *all* the ambient. The
> occlusion then reads as a soft shadow instead of a stamped silhouette.

### Pass 2 — Volumetric

Target: `RGBA16F` at **half the canvas drawing-buffer size in each dimension**
(one quarter the pixels).

The transmittance march used here, the `TrLight` march in pass 1, and the
`skyVis` march in pass 1 are the same GLSL function with different step counts,
shared by `#include`-style string concatenation in `shaders.js`. All three need
the medium's inverse model matrix as a uniform.

Per pixel:

1. Reconstruct the world-space ray from the inverse view-projection matrix.
2. Transform the ray into the medium's local space with the inverse model
   matrix and intersect the unit AABB, giving the OBB intersection in world
   space for free.
3. Clamp `tmax` by the scene depth read from pass 1, so the medium is correctly
   occluded by geometry.
4. March ~64 steps with a per-pixel jittered start offset to trade banding for
   noise.

Per step:

- `D  = texture(density, localPos).r * maxDensity`
- `σt = D * densityScale`, `σs = σt * mediumColor`

  There is no separate extinction constant: the UI density multiplier *is* the
  extinction scale. `mediumColor` is the **single-scattering albedo** (0..1 per
  channel), so `σs ≤ σt` always holds and the medium can never scatter more
  than it extinguishes. Its default is near-white and slightly cool.
  The density multiplier's default is chosen during implementation, once the
  decoded mean and max density are known, to give an optical depth through the
  thick part of the cloud of roughly 4–6 — dense, but not flatly opaque.
- Light visibility is **both** a secondary march toward the light for `TrL`
  (~12 steps, exponentially growing) **and** `PCF(shadowMap, p)` — the shadow
  map term is what makes the teapot cast a real shadow *into* the cloud.
- Henyey-Greenstein phase function for the view/light angle, eccentricity
  `g = 0.3` (mildly forward-scattering, cloud-like). Not exposed in the UI.
- Energy-conserving analytic integration over the step, rather than a
  rectangle rule:

  ```
  S    = σs * phase * lightColor * lightIntensity * TrL * shadowVis
  Sint = (S - S * exp(-σt * dt)) / σt
  L   += T * Sint
  T   *= exp(-σt * dt)
  ```

- Ambient in-scatter `σs * ambientIntensity * skyColor` is added per step so the
  side of the cloud facing away from the light is not black.
- **Multiple scattering**: three Wrenninge-style octaves with
  `a = b = c = 0.5^i` scaling scattering, extinction and phase eccentricity
  respectively, reusing the already-sampled `D` and `TrL`. This is the direct
  answer to the article's warning about unnatural darkening — single scattering
  alone kills the cloud core.

Output: `vec4(inScatter, transmittance)`.

### Pass 3 — Composite

A fullscreen quad into the default framebuffer.

- Depth-aware (bilateral) upsample of the half-res volume buffer: weight the
  four contributing texels by the similarity of their depth to the full-res
  depth, so the volume does not bleed across silhouettes.
- `final = opaqueColor * T + inScatter`
- ACES-style tonemap, then gamma 2.2 to sRGB.

## Interaction

`camera.js` is used **as is** — it already implements exactly the requested
UE-style scheme: right mouse held to engage, `w`/`s` forward/back, `a`/`d`
strafe, `e`/`q` up/down, mouse for yaw/pitch, wheel for speed, shift to boost.
No modification is expected. Pointer lock is requested when available.

## UI

Built with **lil-gui 0.20.0**, vendored to `vendor/lil-gui.umd.js` and loaded
from disk rather than a CDN. It is the only third-party JavaScript in the
project and is used solely by `src/ui.js`; nothing in the rendering path
depends on it.

> **Revised after the initial build**, replacing a hand-rolled panel of
> `<input type="range">` rows. Three things were worth knowing:
>
> - lil-gui binds by holding a **reference** to the object, and for vectors and
>   colours that reference is the array itself. Reset therefore copies fresh
>   values *in place* rather than replacing `state.teapot` and friends, or the
>   controllers would keep writing into orphaned arrays.
> - Angles live in radians but are edited in degrees, via one-property accessor
>   proxies. The getter rounds, because the radian round trip turns 30° into
>   `29.999999999999996` and lil-gui would render every digit.
> - lil-gui's default `autoPlace` pins a `position: fixed`, `z-index: 1001`
>   layer with `max-height: 100%` over the canvas, and its opaque background
>   stopped the WebGL content compositing beneath it — the scene visibly
>   stopped at the panel's left edge for the panel's full height, even though
>   the GL viewport and drawing buffer were the full canvas. Hosting the GUI in
>   our own `#gui` container, sized to its content, avoids it.

| Group | Controls |
|---|---|
| Directional light | yaw, pitch, intensity |
| Ambient | intensity |
| Teapot | **visible**, position x/y/z, rotation x/y/z, uniform scale |
| Medium | **visible**, position x/y/z, rotation x/y/z, uniform scale, density multiplier, colour |
| — | FPS readout, reset button |

Controls write into the scene state object that `renderer.js` reads each frame;
nothing is rebuilt on change except the teapot and medium model matrices.

## Error handling

- No WebGL2 context, or a missing `EXT_color_buffer_float` → replace the canvas
  with a readable message naming what is missing.
- Asset fetch failure → on-screen message with the URL and status, plus the
  reminder that the page must be served over HTTP, not opened as `file://`.
- Unsupported Blosc compressor or an unexpected VDB grid type → throw with the
  offending value in the message. No silent fallback to garbage data.
- A loading overlay covers the canvas until the VDB is decoded, since that is a
  76 MB fetch plus a few seconds of parsing.

## Testing

This is a visual demo, so there is no meaningful unit-test surface for the
rendering itself. Verification is:

1. **VDB decode invariants under Node**, as listed above. These are exact and
   are the gate for the whole loader.
2. **OBJ parse check under Node**: index count `47112` from 7676 quads plus
   352 triangles, bbox matching the values measured above, all generated
   normals unit length, and shared vertices averaging their adjacent face
   normals rather than faceting.
3. **Visual verification in the browser** against a checklist: shadows land
   where the light direction implies; the teapot shadows the cloud; the cloud
   shadows the floor; the cloud core is not black; no banding at default
   quality; no halo at silhouettes; frame time acceptable at 1080p.

## Operational note

The page fetches assets, so it must be served over HTTP:

```bash
python -m http.server 8000
```

Opening `index.html` as `file://` will fail on the asset fetches.
