# Vendored font licenses & provenance

All fonts are open-licensed and were fetched via the Google Fonts CSS API
(`https://fonts.googleapis.com/css2?family=...` → gstatic `.ttf`), except the
CJK subset which was built from the Google Fonts variable source.

- **Tinos-Regular.ttf** — Apache-2.0 — Tinos (metric-compatible with Times New Roman), Google Fonts.
- **Geist-Regular.ttf** — OFL-1.1 — Geist (Vercel), Google Fonts.
- **DancingScript.ttf** — OFL-1.1 — Dancing Script (connected script; curve stress test), Google Fonts.
- **NotoSansSC-Subset.ttf** — OFL-1.1 — Noto Sans SC, weight 400 instance, subset to
  ASCII + CJK punctuation + fullwidth forms + GB2312 levels 1&2 (~6763 hanzi) + demo chars.
  Source: `ofl/notosanssc/NotoSansSC[wght].ttf` (google/fonts) → `fontTools.varLib.instancer wght=400` → `pyftsubset`.

opentype.js (`vendor/opentype.min.js` for the browser, `vendor/opentype.cjs` for Node)
— MIT — https://github.com/opentypejs/opentype.js v1.3.4.
