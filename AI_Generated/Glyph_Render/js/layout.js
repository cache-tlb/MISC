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
