// Shared Node test loader. The vendored opentype build (browser dist) exposes
// `parse` but not `loadSync`, and the package is ESM ("type":"module"), so we
// require the CommonJS copy and parse a file buffer into an ArrayBuffer.
import { createRequire } from 'module';
import { readFileSync } from 'fs';

const require = createRequire(import.meta.url);
export const opentype = require('../vendor/opentype.cjs');

export function loadOt(path) {
  const b = readFileSync(path);
  return opentype.parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
