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

// em-space extent of a laid-out text block: pen advances horizontally, the
// font's own ascent/descent vertically. Used to centre the block on the 3D
// plane and to size the backdrop grid.
export function textBoundsEm(font, instances) {
  if (!instances.length) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const it of instances) {
    minX = Math.min(minX, it.ox);
    maxX = Math.max(maxX, it.ox + font.advanceEm(it.glyphId));
    minY = Math.min(minY, it.oy + font.descenderEm);
    maxY = Math.max(maxY, it.oy + font.ascenderEm);
  }
  return { minX, minY, maxX, maxY };
}
