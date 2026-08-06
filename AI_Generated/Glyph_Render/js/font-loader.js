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
