import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadOt } from './_helpers.mjs';
import { wrapFont } from '../js/font-loader.js';

const font = wrapFont(loadOt('fonts/Tinos-Regular.ttf'));

test('metrics normalized to em', () => {
  assert.ok(font.unitsPerEm >= 1000);
  const gid = font.glyphIdForChar('A');
  const adv = font.advanceEm(gid);
  assert.ok(adv > 0.4 && adv < 1.0, `A advance ${adv} in em`);
});

test('outline is em-space and y-up', () => {
  const gid = font.glyphIdForChar('A');
  const cmds = font.outlineEm(gid);
  assert.ok(cmds.length > 3);
  const ys = cmds.filter(c=>c.y!==undefined).map(c=>c.y);
  const maxY = Math.max(...ys), minY = Math.min(...ys);
  assert.ok(maxY > 0.5 && maxY < 1.2, `apex ${maxY}`);
  assert.ok(minY > -0.2 && minY < 0.15, `base ${minY}`);
  assert.ok(cmds.every(c => (c.x===undefined || Math.abs(c.x) < 2)));
});

test('kerning: AV pair is <= 0', () => {
  const a = font.glyphIdForChar('A'), v = font.glyphIdForChar('V');
  const k = font.kerningEm(a, v);
  assert.ok(k <= 0, `AV kerning ${k} should be <= 0`);
});
