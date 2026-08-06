/*
 * Every GLSL ES 3.00 source in the demo.
 *
 * Shared code is kept in chunks that get concatenated into the shaders that
 * need them, so the transmittance march used by the opaque pass and the one
 * used by the volume pass are literally the same code.
 *
 * @module shaders
 */
(function(root, factory) {  // eslint-disable-line
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define([], factory);
  } else {
    // Browser globals
    root.shaders = factory();
  }
}(this, function() {
  "use strict";

  var VERSION = '#version 300 es\n';

  var PRECISION = [
    'precision highp float;',
    'precision highp int;',
    'precision highp sampler2D;',
    'precision highp sampler3D;',
    '',
    'const float PI = 3.14159265359;',
    'const vec3 LIGHT_COLOR = vec3(1.0);',
    'const vec3 SKY_COLOR = vec3(0.42, 0.55, 0.78);',
    'const vec3 GROUND_COLOR = vec3(0.33, 0.29, 0.25);',
    '',
  ].join('\n');

  // ------------------------------------------------------------------
  // Shared chunks
  // ------------------------------------------------------------------

  /*
   * The medium: box intersection, density sampling and transmittance.
   *
   * Local space is the unit cube centred on the origin, so the density lookup
   * is localPos + 0.5. Ray directions are transformed into local space but
   * deliberately left un-normalised, which keeps the ray parameter t in world
   * units so optical depth comes out right.
   */
  var mediumChunk = [
    'uniform sampler3D u_density;',
    'uniform mat4 u_mediumInverse;',
    'uniform float u_maxDensity;',
    'uniform float u_densityScale;',
    'uniform vec3 u_mediumColor;',
    '',
    'bool boxIntersect(vec3 ro, vec3 rd, out float t0, out float t1) {',
    '  vec3 invD = 1.0 / rd;',
    '  vec3 a = (vec3(-0.5) - ro) * invD;',
    '  vec3 b = (vec3(0.5) - ro) * invD;',
    '  vec3 lo = min(a, b);',
    '  vec3 hi = max(a, b);',
    '  t0 = max(max(lo.x, lo.y), lo.z);',
    '  t1 = min(min(hi.x, hi.y), hi.z);',
    '  return t1 > max(t0, 0.0);',
    '}',
    '',
    'float sampleDensity(vec3 localPos) {',
    '  return texture(u_density, localPos + 0.5).r * u_maxDensity;',
    '}',
    '',
    '// Transmittance from a world position along a unit world direction.',
    'float transmittanceThrough(vec3 worldPos, vec3 dir, int steps) {',
    '  if (u_densityScale <= 0.0) {',
    '    return 1.0;',
    '  }',
    '  vec3 ro = (u_mediumInverse * vec4(worldPos, 1.0)).xyz;',
    '  vec3 rd = (u_mediumInverse * vec4(dir, 0.0)).xyz;',
    '  float t0, t1;',
    '  if (!boxIntersect(ro, rd, t0, t1)) {',
    '    return 1.0;',
    '  }',
    '  t0 = max(t0, 0.0);',
    '  float dt = (t1 - t0) / float(steps);',
    '  float t = t0 + dt * 0.5;',
    '  float tau = 0.0;',
    '  for (int i = 0; i < 32; i++) {',
    '    if (i >= steps) { break; }',
    '    tau += sampleDensity(ro + rd * t);',
    '    t += dt;',
    '  }',
    '  return exp(-tau * dt * u_densityScale);',
    '}',
    '',
  ].join('\n');

  /*
   * The shadow map. Sampled as a plain depth texture with manual 3x3 PCF, so
   * the volume pass can call the same function; hardware comparison sampling
   * would need a different sampler type there.
   */
  var shadowChunk = [
    'uniform sampler2D u_shadowMap;',
    'uniform mat4 u_lightViewProjection;',
    'uniform vec2 u_shadowTexel;',
    'uniform vec2 u_shadowBias;',
    'uniform vec3 u_lightDirection;',
    '',
    'float shadowFactor(vec3 worldPos, vec3 N) {',
    '  vec4 lightClip = u_lightViewProjection * vec4(worldPos, 1.0);',
    '  vec3 proj = lightClip.xyz / lightClip.w * 0.5 + 0.5;',
    '  // Outside the map counts as lit, never as shadowed.',
    '  if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 ||',
    '      proj.y < 0.0 || proj.y > 1.0) {',
    '    return 1.0;',
    '  }',
    '  float slope = 1.0 - abs(dot(N, u_lightDirection));',
    '  float bias = max(u_shadowBias.x, u_shadowBias.y * slope);',
    '  float sum = 0.0;',
    '  for (int y = -1; y <= 1; y++) {',
    '    for (int x = -1; x <= 1; x++) {',
    '      vec2 uv = proj.xy + vec2(float(x), float(y)) * u_shadowTexel;',
    '      float depth = texture(u_shadowMap, uv).r;',
    '      sum += (proj.z - bias <= depth) ? 1.0 : 0.0;',
    '    }',
    '  }',
    '  return sum / 9.0;',
    '}',
    '',
  ].join('\n');

  /* An analytically anti-aliased checkerboard, after Inigo Quilez. */
  var checkerChunk = [
    '// Antiderivative of a square wave of period 2: a triangle wave whose',
    '// slope is exactly +/-1. Integrating anything smoother here turns the',
    '// checker into a product of soft ramps, which reads as blurred blobs',
    '// rather than squares and only spans half the intended contrast.',
    'vec2 checkerIntegral(vec2 x) {',
    '  return 2.0 * abs(fract(x * 0.5) - 0.5);',
    '}',
    '',
    '// Box filtered over the pixel footprint, so grazing angles fade to the',
    '// average grey instead of aliasing into noise. The two samples are',
    '// subtracted low minus high, which is what carries the sign.',
    'float checkerBox(vec2 p) {',
    '  // max of the derivatives, not fwidth: fwidth sums |dFdx| and |dFdy|,',
    '  // overstating the footprint and blurring the pattern early.',
    '  vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 0.001;',
    '  vec2 i = (checkerIntegral(p - 0.5 * w) - checkerIntegral(p + 0.5 * w)) / w;',
    '  return 0.5 - 0.5 * i.x * i.y;',
    '}',
    '',
  ].join('\n');

  var tonemapChunk = [
    'vec3 acesFilm(vec3 x) {',
    '  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),',
    '      0.0, 1.0);',
    '}',
    '',
  ].join('\n');

  // ------------------------------------------------------------------
  // Depth only pass, for the shadow map
  // ------------------------------------------------------------------

  var depthVS = VERSION + [
    'in vec4 a_position;',
    'uniform mat4 u_lightViewProjection;',
    'uniform mat4 u_model;',
    'void main() {',
    '  gl_Position = u_lightViewProjection * u_model * a_position;',
    '}',
  ].join('\n');

  var depthFS = VERSION + PRECISION + [
    'void main() {}',
  ].join('\n');

  // ------------------------------------------------------------------
  // Opaque pass
  // ------------------------------------------------------------------

  var opaqueVS = VERSION + [
    'in vec4 a_position;',
    'in vec3 a_normal;',
    'in vec2 a_texcoord;',
    '',
    'uniform mat4 u_viewProjection;',
    'uniform mat4 u_model;',
    'uniform mat4 u_normalMatrix;',
    '',
    'out vec3 v_worldPosition;',
    'out vec3 v_normal;',
    'out vec2 v_texcoord;',
    '',
    'void main() {',
    '  vec4 world = u_model * a_position;',
    '  v_worldPosition = world.xyz;',
    '  v_normal = mat3(u_normalMatrix) * a_normal;',
    '  v_texcoord = a_texcoord;',
    '  gl_Position = u_viewProjection * world;',
    '}',
  ].join('\n');

  var opaqueFS = VERSION + PRECISION + mediumChunk + shadowChunk +
      checkerChunk + [
    'in vec3 v_worldPosition;',
    'in vec3 v_normal;',
    'in vec2 v_texcoord;',
    '',
    'uniform float u_lightIntensity;',
    'uniform float u_ambientIntensity;',
    'uniform vec3 u_albedo;',
    'uniform int u_useChecker;',
    'uniform int u_useTexture;',
    'uniform sampler2D u_texture;',
    '',
    'out vec4 outColor;',
    '',
    '// How much of the sky reaches this point through the medium.',
    '//',
    '// A single straight-up ray is not enough: it stamps the medium\'s exact',
    '// footprint onto the floor as a hard edged black column, because every',
    '// point below the cloud samples the same worst case direction. Ambient is',
    '// an integral over the hemisphere, so averaging a small cone of',
    '// directions is both closer to the truth and what makes the occlusion',
    '// read as a soft shadow. The cone still covers only part of the',
    '// hemisphere, hence the final blend: the medium can darken the ambient',
    '// but never remove all of it.',
    'float skyVisibility(vec3 worldPos) {',
    '  vec3 d0 = vec3(0.0, 1.0, 0.0);',
    '  vec3 d1 = vec3(0.5736, 0.8192, 0.0);',
    '  vec3 d2 = vec3(-0.2868, 0.8192, 0.4967);',
    '  vec3 d3 = vec3(-0.2868, 0.8192, -0.4967);',
    '  float v = transmittanceThrough(worldPos, d0, 8) +',
    '            transmittanceThrough(worldPos, d1, 8) +',
    '            transmittanceThrough(worldPos, d2, 8) +',
    '            transmittanceThrough(worldPos, d3, 8);',
    '  return mix(1.0, v * 0.25, 0.8);',
    '}',
    '',
    '// Checker on the two axes perpendicular to the dominant normal axis, so',
    '// walls pattern in their own plane instead of smearing.',
    'vec2 checkerPlane(vec3 p, vec3 n) {',
    '  vec3 a = abs(n);',
    '  if (a.y >= a.x && a.y >= a.z) { return p.xz; }',
    '  if (a.x >= a.z) { return p.zy; }',
    '  return p.xy;',
    '}',
    '',
    'void main() {',
    '  vec3 N = normalize(v_normal);',
    '  if (!gl_FrontFacing) { N = -N; }',
    '',
    '  vec3 albedo = u_albedo;',
    '  if (u_useChecker == 1) {',
    '    float c = checkerBox(checkerPlane(v_worldPosition, N) / 10.0);',
    '    albedo = vec3(mix(0.32, 0.82, c));',
    '  }',
    '  if (u_useTexture == 1) {',
    '    albedo *= texture(u_texture, v_texcoord).rgb;',
    '  }',
    '',
    '  vec3 L = -u_lightDirection;',
    '  float ndotl = max(dot(N, L), 0.0);',
    '',
    '  // Direct light: the shadow map first, because a fragment already in',
    '  // shadow does not need the medium marched at all.',
    '  vec3 direct = vec3(0.0);',
    '  float shadow = shadowFactor(v_worldPosition, N);',
    '  if (shadow > 0.0 && ndotl > 0.0) {',
    '    float trL = transmittanceThrough(v_worldPosition, L, 12);',
    '    direct = albedo / PI * LIGHT_COLOR * u_lightIntensity * trL * shadow * ndotl;',
    '  }',
    '',
    '  // Ambient: hemispheric sky and ground radiance, occluded by whatever',
    '  // of the medium sits between this point and the sky.',
    '  float skyVis = skyVisibility(v_worldPosition);',
    '  vec3 hemi = mix(GROUND_COLOR, SKY_COLOR, N.y * 0.5 + 0.5);',
    '  vec3 ambient = u_ambientIntensity * hemi * albedo * skyVis;',
    '',
    '  outColor = vec4(direct + ambient, 1.0);',
    '}',
  ].join('\n');

  // ------------------------------------------------------------------
  // Fullscreen passes
  // ------------------------------------------------------------------

  var fullscreenVS = VERSION + [
    'in vec2 a_position;',
    'out vec2 v_uv;',
    'void main() {',
    '  v_uv = a_position * 0.5 + 0.5;',
    '  gl_Position = vec4(a_position, 0.0, 1.0);',
    '}',
  ].join('\n');

  /*
   * The volumetric march, at half resolution.
   *
   * Output is vec4(inScatter, transmittance): the light this ray picked up
   * inside the medium, and how much of whatever is behind it survives.
   */
  var volumeFS = VERSION + PRECISION + mediumChunk + shadowChunk + [
    'in vec2 v_uv;',
    '',
    'uniform mat4 u_inverseViewProjection;',
    'uniform vec3 u_cameraPosition;',
    'uniform sampler2D u_sceneDepth;',
    'uniform float u_lightIntensity;',
    'uniform float u_ambientIntensity;',
    'uniform float u_time;',
    'uniform int u_steps;',
    'uniform int u_lightSteps;',
    'uniform int u_octaves;',
    '',
    'out vec4 outColor;',
    '',
    'const float PHASE_G = 0.3;',
    '',
    'float henyeyGreenstein(float cosTheta, float g) {',
    '  float g2 = g * g;',
    '  float d = 1.0 + g2 - 2.0 * g * cosTheta;',
    '  return (1.0 - g2) / (4.0 * PI * max(d, 1e-4) * sqrt(max(d, 1e-4)));',
    '}',
    '',
    'float hash(vec2 p) {',
    '  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);',
    '}',
    '',
    'void main() {',
    '  vec2 ndc = v_uv * 2.0 - 1.0;',
    '',
    '  // World ray from the inverse view projection.',
    '  vec4 far = u_inverseViewProjection * vec4(ndc, 1.0, 1.0);',
    '  vec3 farPoint = far.xyz / far.w;',
    '  vec3 rayDir = normalize(farPoint - u_cameraPosition);',
    '',
    '  // How far the opaque scene lets us march.',
    '  float sceneDepth = texture(u_sceneDepth, v_uv).r;',
    '  float sceneDist = 1e20;',
    '  if (sceneDepth < 1.0) {',
    '    vec4 sp = u_inverseViewProjection * vec4(ndc, sceneDepth * 2.0 - 1.0, 1.0);',
    '    sceneDist = length(sp.xyz / sp.w - u_cameraPosition);',
    '  }',
    '',
    '  vec3 ro = (u_mediumInverse * vec4(u_cameraPosition, 1.0)).xyz;',
    '  vec3 rd = (u_mediumInverse * vec4(rayDir, 0.0)).xyz;',
    '',
    '  float t0, t1;',
    '  if (!boxIntersect(ro, rd, t0, t1) || u_densityScale <= 0.0) {',
    '    outColor = vec4(0.0, 0.0, 0.0, 1.0);',
    '    return;',
    '  }',
    '  t0 = max(t0, 0.0);',
    '  t1 = min(t1, sceneDist);',
    '  if (t1 <= t0) {',
    '    outColor = vec4(0.0, 0.0, 0.0, 1.0);',
    '    return;',
    '  }',
    '',
    '  float dt = (t1 - t0) / float(u_steps);',
    '  // Jitter the start so the step pattern shows up as noise, not bands.',
    '  float jitter = hash(gl_FragCoord.xy + fract(u_time) * 133.7);',
    '  float t = t0 + dt * jitter;',
    '',
    '  float cosTheta = dot(rayDir, -u_lightDirection);',
    '',
    '  vec3 scattered = vec3(0.0);',
    '  float transmittance = 1.0;',
    '',
    '  for (int i = 0; i < 128; i++) {',
    '    if (i >= u_steps || t > t1) { break; }',
    '',
    '    vec3 localPos = ro + rd * t;',
    '    float density = sampleDensity(localPos);',
    '',
    '    if (density > 0.001) {',
    '      vec3 worldPos = u_cameraPosition + rayDir * t;',
    '      float sigmaT = density * u_densityScale;',
    '      vec3 sigmaS = sigmaT * u_mediumColor;',
    '',
    '      float trL = transmittanceThrough(worldPos, -u_lightDirection, u_lightSteps);',
    '      float vis = shadowFactor(worldPos, vec3(0.0));',
    '',
    '      // Multiple scattering as octaves: each successive one scatters',
    '      // less, is absorbed less, and is more isotropic. Without them the',
    '      // core of a cloud this thick goes dead black.',
    '      vec3 S = vec3(0.0);',
    '      for (int o = 0; o < 4; o++) {',
    '        if (o >= u_octaves) { break; }',
    '        float a = pow(0.5, float(o));',
    '        float phase = henyeyGreenstein(cosTheta, PHASE_G * a);',
    '        S += a * sigmaS * phase * LIGHT_COLOR * u_lightIntensity *',
    '             pow(trL, a) * vis;',
    '      }',
    '      // Ambient in-scatter, so the unlit side is not black either.',
    '      S += sigmaS * u_ambientIntensity * SKY_COLOR;',
    '',
    '      // Energy conserving integration across the step. A plain',
    '      // rectangle rule loses energy badly at these step counts.',
    '      float stepT = exp(-sigmaT * dt);',
    '      vec3 integrated = (S - S * stepT) / max(sigmaT, 1e-6);',
    '      scattered += transmittance * integrated;',
    '      transmittance *= stepT;',
    '',
    '      if (transmittance < 0.003) { break; }',
    '    }',
    '',
    '    t += dt;',
    '  }',
    '',
    '  outColor = vec4(scattered, transmittance);',
    '}',
  ].join('\n');

  /*
   * Composite: depth aware upsample of the half resolution volume, then
   * combine with the opaque colour, tonemap and encode.
   */
  var compositeFS = VERSION + PRECISION + tonemapChunk + [
    'in vec2 v_uv;',
    '',
    'uniform sampler2D u_opaque;',
    'uniform sampler2D u_volume;',
    'uniform sampler2D u_sceneDepth;',
    'uniform vec2 u_halfResolution;',
    'uniform vec2 u_clipPlanes;',
    '',
    'out vec4 outColor;',
    '',
    'float linearDepth(float d) {',
    '  float near = u_clipPlanes.x;',
    '  float far = u_clipPlanes.y;',
    '  float z = d * 2.0 - 1.0;',
    '  return (2.0 * near * far) / (far + near - z * (far - near));',
    '}',
    '',
    '// Bilateral upsample. Weighting the four half resolution taps by how',
    '// close their depth is to this pixel keeps the volume from bleeding',
    '// across silhouettes, which is what a plain bilinear filter does.',
    'vec4 upsampleVolume(float centerDepth) {',
    '  vec2 texel = 1.0 / u_halfResolution;',
    '  vec2 coord = v_uv * u_halfResolution - 0.5;',
    '  vec2 base = floor(coord);',
    '  vec2 f = coord - base;',
    '',
    '  vec4 sum = vec4(0.0);',
    '  float weightSum = 0.0;',
    '',
    '  for (int j = 0; j < 2; j++) {',
    '    for (int i = 0; i < 2; i++) {',
    '      vec2 tc = (base + vec2(float(i), float(j)) + 0.5) * texel;',
    '      float bilinear = (i == 0 ? 1.0 - f.x : f.x) *',
    '                       (j == 0 ? 1.0 - f.y : f.y);',
    '      float d = linearDepth(texture(u_sceneDepth, tc).r);',
    '      // Tolerance scales with distance: a fixed world space threshold',
    '      // would reject almost every tap on a floor seen at a grazing',
    '      // angle, where neighbouring pixels are metres apart in depth.',
    '      float sigma = max(0.5, centerDepth * 0.04);',
    '      float w = bilinear * exp(-abs(d - centerDepth) / sigma);',
    '      sum += texture(u_volume, tc) * w;',
    '      weightSum += w;',
    '    }',
    '  }',
    '',
    '  if (weightSum < 1e-5) {',
    '    return texture(u_volume, v_uv);',
    '  }',
    '  return sum / weightSum;',
    '}',
    '',
    'void main() {',
    '  float centerDepth = linearDepth(texture(u_sceneDepth, v_uv).r);',
    '  vec4 volume = upsampleVolume(centerDepth);',
    '  vec3 opaque = texture(u_opaque, v_uv).rgb;',
    '',
    '  vec3 color = opaque * volume.a + volume.rgb;',
    '  color = acesFilm(color);',
    '  outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);',
    '}',
  ].join('\n');

  return {
    depthVS: depthVS,
    depthFS: depthFS,
    opaqueVS: opaqueVS,
    opaqueFS: opaqueFS,
    fullscreenVS: fullscreenVS,
    volumeFS: volumeFS,
    compositeFS: compositeFS,
  };

}));
