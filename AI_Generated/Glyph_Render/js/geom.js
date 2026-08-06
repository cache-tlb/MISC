export function evalQuad(p0, p1, p2, t) {
  const u = 1 - t;
  return [
    u*u*p0[0] + 2*u*t*p1[0] + t*t*p2[0],
    u*u*p0[1] + 2*u*t*p1[1] + t*t*p2[1],
  ];
}

export function evalCubic(p0, p1, p2, p3, t) {
  const u = 1 - t, uu = u*u, tt = t*t;
  const a = uu*u, b = 3*uu*t, c = 3*u*tt, d = tt*t;
  return [
    a*p0[0] + b*p1[0] + c*p2[0] + d*p3[0],
    a*p0[1] + b*p1[1] + c*p2[1] + d*p3[1],
  ];
}

export function bboxOfCommands(commands) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const acc = (x, y) => { if (x<minX)minX=x; if (y<minY)minY=y; if (x>maxX)maxX=x; if (y>maxY)maxY=y; };
  for (const c of commands) {
    if (c.type === 'Z') continue;
    if (c.x1 !== undefined) acc(c.x1, c.y1);
    if (c.x2 !== undefined) acc(c.x2, c.y2);
    if (c.x !== undefined) acc(c.x, c.y);
  }
  return { minX, minY, maxX, maxY };
}

// Break commands into closed contours of line-segment edges (em, y-up).
export function flattenCommands(commands, segsPerCurve = 16) {
  const edges = [];
  let contour = -1, start = null, cur = null;
  const line = (a, b) => { if (a[0]!==b[0] || a[1]!==b[1]) edges.push({ a, b, contour }); };
  for (const c of commands) {
    if (c.type === 'M') { contour++; start = [c.x, c.y]; cur = start; }
    else if (c.type === 'L') { const p=[c.x,c.y]; line(cur, p); cur = p; }
    else if (c.type === 'Q') {
      let prev = cur;
      for (let i=1;i<=segsPerCurve;i++){ const p=evalQuad(cur,[c.x1,c.y1],[c.x,c.y], i/segsPerCurve); line(prev,p); prev=p; }
      cur = [c.x, c.y];
    } else if (c.type === 'C') {
      let prev = cur;
      for (let i=1;i<=segsPerCurve;i++){ const p=evalCubic(cur,[c.x1,c.y1],[c.x2,c.y2],[c.x,c.y], i/segsPerCurve); line(prev,p); prev=p; }
      cur = [c.x, c.y];
    } else if (c.type === 'Z') { if (start) line(cur, start); cur = start; }
  }
  return edges;
}
