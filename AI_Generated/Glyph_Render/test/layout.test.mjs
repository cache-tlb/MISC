import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadOt } from './_helpers.mjs';
import { wrapFont } from '../js/font-loader.js';
import { layoutText, uniqueGlyphIds } from '../js/layout.js';

const font = wrapFont(loadOt('fonts/Tinos-Regular.ttf'));

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
