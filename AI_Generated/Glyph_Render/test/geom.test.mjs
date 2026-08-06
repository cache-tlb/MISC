import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evalQuad, evalCubic, bboxOfCommands, flattenCommands } from '../js/geom.js';

test('evalQuad endpoints and midpoint', () => {
  assert.deepEqual(evalQuad([0,0],[1,2],[2,0], 0), [0,0]);
  assert.deepEqual(evalQuad([0,0],[1,2],[2,0], 1), [2,0]);
  const m = evalQuad([0,0],[1,2],[2,0], 0.5);
  assert.ok(Math.abs(m[0]-1) < 1e-9 && Math.abs(m[1]-1) < 1e-9);
});

test('bbox over a square path', () => {
  const cmds = [
    {type:'M',x:0,y:0},{type:'L',x:2,y:0},{type:'L',x:2,y:3},{type:'L',x:0,y:3},{type:'Z'}
  ];
  assert.deepEqual(bboxOfCommands(cmds), {minX:0,minY:0,maxX:2,maxY:3});
});

test('flatten closes each contour', () => {
  const cmds = [
    {type:'M',x:0,y:0},{type:'L',x:2,y:0},{type:'L',x:2,y:2},{type:'Z'}
  ];
  const edges = flattenCommands(cmds, 8);
  assert.equal(edges.length, 3); // 2 lines + closing edge
  assert.deepEqual(edges[2].b, [0,0]); // closes back to move
});
