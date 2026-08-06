import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadOt } from './_helpers.mjs';

test('opentype loads a vendored font', () => {
  const font = loadOt('fonts/Tinos-Regular.ttf');
  assert.ok(font.unitsPerEm >= 1000, 'unitsPerEm present');
  const gid = font.charToGlyphIndex('A');
  assert.ok(gid > 0, 'glyph index for A');
});

test('CJK subset resolves common hanzi incl. 渲染', () => {
  const font = loadOt('fonts/NotoSansSC-Subset.ttf');
  for (const ch of '汉字形渲染') assert.ok(font.charToGlyphIndex(ch) > 0, `has ${ch}`);
});
