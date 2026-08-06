# vendor

Third-party code, kept here so the boundary against our own source in `src/`
stays obvious. Nothing in `src/` that touches rendering may depend on anything
in this directory.

## lil-gui.umd.js

- Version 0.20.0
- Source: <https://cdn.jsdelivr.net/npm/lil-gui@0.20.0/dist/lil-gui.umd.js>
- Home: <https://lil-gui.georgealways.com>
- Licence: MIT, © George Michael Brower

Used only by `src/ui.js` for the control panel. The UMD build injects its own
stylesheet, so no separate CSS file is needed. It is loaded from disk, not from
a CDN, and exposes the global `lil`.
