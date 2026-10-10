/**
 * The map engine: draws the release's map data (pipeline/release/mapdata.py, "v2") with WebGL2.
 *
 * Painted like the homepage's still images (design/home-mock/make_map.py), at every zoom, in three steps:
 *  1. light: every study's density (256 px tiles, nothing sampled) as coloured light, and the studies themselves
 *     as stars (the importance quadtree: most cited first), each into its own high-precision buffer;
 *  2. glow: that light blurred at two sizes fixed on screen (~3.5 and ~14 css px: a lens, not the data);
 *  3. a film curve per theme, the same formulas as make_map.py: dark = long exposure, the densest cores burn
 *     toward white; light = backlit glass, stars as fine ink grains.
 * Exposure adapts to what is on screen (the 99.7th percentile of studies per pixel, as make_map.py), slowly.
 *
 * The caller owns the camera (atlas.ts) and calls draw() when something changed.
 */

export interface Cam { S: number; ox: number; oy: number; z: number }   // map box size (css px), world (0,0) on screen, zoom
export interface Palette { dark: boolean; bg: number[]; fields: Record<string, number[]> }
export interface Star { x: number; y: number; info: number; sub: number; key: string; i: number }

const TILE = 256, SIDE = 258;

/** A density tile. The raw planes go to the GPU as they are; one GPU pass ("bake") turns them into a filterable
 *  light texture (rgb = colour x studies, a = studies), again whenever the palette changes. The CPU keeps only
 *  the raw counts (exposure, picking): nothing per pixel is computed in JavaScript. */
type DTile = { key: string; z: number; x: number; y: number; cntTex: WebGLTexture; subTex: WebGLTexture; purTex: WebGLTexture;
               tex: WebGLTexture; cnt: Uint16Array; sub: Uint8Array; born: number; baked: string };
type PTile = { key: string; z: number; x: number; y: number; n: number; vao: WebGLVertexArrayObject; buf: WebGLBuffer;
               raw: Uint16Array; born: number; shown: number };
type Target = { tex: WebGLTexture; fb: WebGLFramebuffer; w: number; h: number };

/** The body of a fetched .bin. The servers send the tiles with Content-Encoding: gzip, so the browser inflates them
 *  in its network process, off the thread that draws; a body that still starts with the gzip magic (a server
 *  that does not say so) is inflated here instead. */
async function gunzipBody(r: Response): Promise<ArrayBuffer> {
  const b = await r.arrayBuffer(), h = new Uint8Array(b, 0, Math.min(2, b.byteLength));
  if (h[0] !== 0x1f || h[1] !== 0x8b) return b;
  return await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

function shader(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const p = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    gl.attachShader(p, s);
  }
  gl.bindAttribLocation(p, 0, "a");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  const u: Record<string, WebGLUniformLocation> = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i)!; u[a.name.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, a.name)!; }
  return { p, u };
}

// cubic B-spline filtering from 4 bilinear taps (GPU Gems 2, ch. 20): smooth, no texel blocks when magnified
const CUBIC = `
vec4 cubic(sampler2D t, vec2 uv, vec2 size) {
  vec2 px = uv * size - 0.5, f = fract(px); px -= f;
  vec2 f2 = f * f, f3 = f2 * f;
  vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
  vec2 s0 = w0 + w1, s1 = w2 + w3;
  vec2 c0 = (px - 0.5 + w1 / s0) / size, c1 = (px + 1.5 + w3 / s1) / size;
  float gx = s0.x / (s0.x + s1.x), gy = s0.y / (s0.y + s1.y);
  return mix(mix(texture(t, vec2(c1.x, c1.y)), texture(t, vec2(c0.x, c1.y)), gx),
             mix(texture(t, vec2(c1.x, c0.y)), texture(t, vec2(c0.x, c0.y)), gx), gy);
}`;

// ------------------------------------------------------------------ 0. bake: raw tile planes -> light texture
const BAKE_FS = `#version 300 es
precision highp float; precision highp usampler2D;
uniform usampler2D u_cnt; uniform usampler2D u_sub; uniform sampler2D u_pur; uniform sampler2D u_fcol;
uniform vec3 u_grey; uniform float u_logq; uniform float u_lin;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float c = float(texelFetch(u_cnt, p, 0).r);
  if (c <= 0.0) { o = vec4(0.0); return; }
  float n = exp2(c / u_logq) - 1.0;                        // studies in this pixel
  vec4 f = texelFetch(u_fcol, ivec2(int(texelFetch(u_sub, p, 0).r), 0), 0);   // its main field's colour
  float pur = texelFetch(u_pur, p, 0).r;
  // as make_map.py: one colour per field, paler where fields mix; only-unclassified places muted (x 0.3)
  vec3 col = f.a > 0.5 ? mix(u_grey, f.rgb, 0.35 + 0.65 * pur) : u_grey * 0.3;
  // float: light adds up under filtering; stored / 64 so the densest pixels (100,000+ studies with the "mode"
  // placement) stay inside half-float range (65,504) instead of overflowing to Inf, which the blur turned into a black square
  o = u_lin > 0.5 ? vec4(col * n, n) / 64.0
                  : vec4(col, log2(1.0 + n) / 16.0);       // 8-bit fallback: colour and log studies
}`;

// ------------------------------------------------------------------ 1. light: density tiles
const DENSITY_VS = `#version 300 es
in vec2 a; uniform vec4 u_rect; uniform vec4 u_uv; uniform vec2 u_view;
out vec2 v_uv;
void main() {
  vec2 px = u_rect.xy + a * u_rect.zw;                  // device px, y down
  v_uv = u_uv.xy + a * u_uv.zw;
  gl_Position = vec4(px / u_view * 2.0 - 1.0, 0.0, 1.0); gl_Position.y = -gl_Position.y;
}`;
const DENSITY_FS = `#version 300 es
precision highp float;
in vec2 v_uv; uniform sampler2D u_t;
uniform float u_area;      // device px per texel, squared
uniform float u_k;         // exposure: light per (study per device px)
uniform float u_w;         // weight (cross-fade between a tile and its parent)
uniform float u_cubic;     // 1 when a texel is several pixels wide
uniform float u_scale;     // light buffer scale (8-bit fallback)
uniform float u_lin;       // baked as linear light (float) or colour + log studies (8-bit)
uniform float u_knee;      // exposure level of the 99.7th percentile: brighter than this is compressed
out vec4 o;
${CUBIC}
void main() {
  vec4 t = u_cubic > 0.5 ? cubic(u_t, v_uv, vec2(258.0)) : texture(u_t, v_uv);
  float n = u_lin > 0.5 ? t.a * 64.0 : exp2(t.a * 16.0) - 1.0;      // studies in the texel
  if (n <= 1e-6) discard;
  vec3 col = u_lin > 0.5 ? t.rgb / t.a : t.rgb;
  // light: linear up to the 99.7th percentile of the screen (calibrated against the homepage image), square root
  // beyond, so the densest clusters glow instead of burning a white halo through the blur
  float x = n / u_area * u_k;
  float y = x <= u_knee ? x : u_knee * sqrt(x / u_knee);
  o = vec4(col * y * u_w * u_scale, 1.0);
}`;

// ------------------------------------------------------------------ 1. light: stars
const STAR_VS = `#version 300 es
in vec2 a_xy; in uint a_info; in uint a_sub;
uniform vec4 u_tile;       // world x, y, size of the tile
uniform vec3 u_cam;        // device px: world origin x, y; world size (S * z * dpr)
uniform vec2 u_view; uniform float u_size; uniform float u_dpr;
uniform sampler2D u_pal;
out vec3 v_col; out float v_r; out float v_w; out float v_ps;
void main() {
  vec2 w = u_tile.xy + a_xy * u_tile.z;
  vec2 px = u_cam.xy + w * u_cam.z;
  float c = float((a_info >> 12u) & 7u);                // citation class 0..7
  float r = (0.45 + 0.24 * c) * u_size * u_dpr;          // radius, device px: the cited are bigger
  float rr = max(r, 0.75);
  v_ps = ceil(rr * 2.0 + 2.0);
  gl_PointSize = v_ps;
  v_r = rr;
  v_w = (0.55 + 0.2 * c) * (r * r) / (rr * rr);          // tiny stars: not smaller than a pixel, same total light
  v_col = texelFetch(u_pal, ivec2(int(a_sub), 0), 0).rgb;
  gl_Position = vec4(px / u_view * 2.0 - 1.0, 0.0, 1.0); gl_Position.y = -gl_Position.y;
}`;
const STAR_FS = `#version 300 es
precision highp float;
in vec3 v_col; in float v_r; in float v_w; in float v_ps;
uniform float u_gain; uniform float u_scale;
out vec4 o;
void main() {
  float d = length(gl_PointCoord - 0.5) * v_ps;
  float f = 1.0 - smoothstep(v_r - 0.75, v_r + 0.75, d);  // an antialiased disc
  if (f <= 0.002) discard;
  o = vec4(v_col * f * v_w * u_gain * u_scale, 1.0);
}`;

// ------------------------------------------------------------------ 2. glow
const FULL_VS = `#version 300 es
in vec2 a; out vec2 v;
void main() { v = a; gl_Position = vec4(a * 2.0 - 1.0, 0.0, 1.0); }`;
const DOWN_FS = `#version 300 es
precision highp float;
in vec2 v; uniform sampler2D u_a; uniform sampler2D u_b; uniform float u_two; uniform vec2 u_texel;
out vec4 o;
vec4 tap(vec2 p) { vec4 x = texture(u_a, p); if (u_two > 0.5) x += texture(u_b, p); return x; }
void main() {   // 4 bilinear taps over a 4x4 block: a tent filter, no shimmer when the map moves
  o = 0.25 * (tap(v + u_texel * vec2(-1.0, -1.0)) + tap(v + u_texel * vec2(1.0, -1.0))
            + tap(v + u_texel * vec2(-1.0, 1.0)) + tap(v + u_texel * vec2(1.0, 1.0)));
}`;
const BLUR_FS = `#version 300 es
precision highp float;
in vec2 v; uniform sampler2D u_a; uniform vec2 u_dir;     // one texel along x or y
out vec4 o;
void main() {   // 9-tap Gaussian (sigma ~1.6 texels) in 5 bilinear taps
  o = texture(u_a, v) * 0.2270270
    + (texture(u_a, v + u_dir * 1.3846154) + texture(u_a, v - u_dir * 1.3846154)) * 0.3162162
    + (texture(u_a, v + u_dir * 3.2307692) + texture(u_a, v - u_dir * 3.2307692)) * 0.0702703;
}`;

// ------------------------------------------------------------------ 3. film curve (make_map.py)
const COMPOSE_FS = `#version 300 es
precision highp float;
in vec2 v;
uniform sampler2D u_haze; uniform sampler2D u_stars; uniform sampler2D u_mid; uniform sampler2D u_big;
uniform vec2 u_midSize; uniform vec2 u_bigSize; uniform float u_dark; uniform vec3 u_bg; uniform float u_scale;
uniform float u_wSharp; uniform float u_wMid; uniform float u_wBig; uniform float u_grain;
out vec4 o;
${CUBIC}
float mx(vec3 c) { return max(c.r, max(c.g, c.b)); }
vec3 sat(vec3 c, float k) { float g = (c.r + c.g + c.b) / 3.0; return max(vec3(0.0), g + (c - g) * k); }
void main() {
  float inv = 1.0 / u_scale;
  vec3 haze = texture(u_haze, v).rgb * inv, stars = texture(u_stars, v).rgb * inv;
  vec3 mid = cubic(u_mid, v, u_midSize).rgb * inv, big = cubic(u_big, v, u_bigSize).rgb * inv;
  vec3 sharp = haze + stars;
  vec3 rgb;
  if (u_dark > 0.5) {
    // long exposure: sharp light + glow, a film-like curve; only the densest cores burn toward white
    vec3 L = sat(sharp * u_wSharp + mid * u_wMid + big * u_wBig, 1.35);
    float lum = mx(L);
    vec3 e = 1.0 - exp(-L * 1.6);
    float white = clamp((1.0 - exp(-lum * 0.9)) - 0.55, 0.0, 1.0) / 0.45;
    e = e * (1.0 - white * 0.6) + white * 0.6;
    float al = clamp(mx(e), 0.0, 1.0);
    rgb = e + u_bg * (1.0 - al);
  } else {
    // backlit glass: the light as vivid colour on cream; the densest cores turn into pale warm light
    vec3 L = sharp * (u_wSharp * 0.97) + mid * (u_wMid * 0.91) + big * (u_wBig * 0.87);
    vec3 e = 1.0 - exp(-L * 1.8);
    float lum = mx(e);
    vec3 hue = sat(e / max(lum, 1e-6), 1.6); hue /= max(mx(hue), 1e-6);
    float core = pow(clamp((lum - 0.62) / 0.38, 0.0, 1.0), 1.5);
    vec3 hazeRgb = hue * 0.86 * (1.0 - core) + vec3(1.0, 0.985, 0.94) * core;
    float hazeA = clamp(pow(lum, 0.9) * 0.85 + core * 0.15, 0.0, 1.0);
    // the stars as fine grains of ink inside the glow
    vec3 G = stars * u_grain + haze * 0.35;
    float gl = mx(G);
    vec3 gh = sat(G / max(gl, 1e-6), 1.5); gh /= max(mx(gh), 1e-6);
    float ga = clamp(1.0 - exp(-gl * 4.0), 0.0, 1.0) * 0.75 * (1.0 - core * 0.85);
    float al = ga + hazeA * (1.0 - ga);
    vec3 col = (gh * 0.5 * ga + hazeRgb * hazeA * (1.0 - ga)) / max(al, 1e-6);
    rgb = mix(u_bg, col, al);
  }
  o = vec4(rgb, 1.0);
}`;

export class MapGL {
  gl: WebGL2RenderingContext;
  base = "";
  meta: any = null;
  dtiles = new Map<string, DTile>();
  ptiles = new Map<string, PTile>();
  dHave = new Set<string>();
  pHave = new Map<string, number>();
  loading = new Map<string, Promise<void>>();
  pal: Palette = { dark: true, bg: [0.07, 0.07, 0.07], fields: {} };
  palKey = "";
  palTex: WebGLTexture;
  progs: Record<string, { p: WebGLProgram; u: Record<string, WebGLUniformLocation> }>;
  quad: WebGLVertexArrayObject;
  k = 0; kTarget = 0; lastStats = 0;
  onChange: () => void = () => {};
  dLevel = 0; pLevel = 0;
  /** float buffers when the device can render to them; else 8 bits with the light scaled down */
  hdr: boolean;
  scale: number;
  targets: Record<string, Target> = {};
  /** The last frame was final: every visible tile loaded and faded in, exposure settled. */
  complete = false;
  targetKey = "";

  constructor(public canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: true, preserveDrawingBuffer: false });
    if (!gl) throw new Error("no WebGL2");
    this.gl = gl;
    this.hdr = !!gl.getExtension("EXT_color_buffer_float") || !!gl.getExtension("EXT_color_buffer_half_float");
    this.scale = this.hdr ? 1 : 0.125;
    this.progs = {
      density: shader(gl, DENSITY_VS, DENSITY_FS), stars: shader(gl, STAR_VS, STAR_FS), bake: shader(gl, FULL_VS, BAKE_FS),
      down: shader(gl, FULL_VS, DOWN_FS), blur: shader(gl, FULL_VS, BLUR_FS), compose: shader(gl, FULL_VS, COMPOSE_FS),
    };
    this.quad = gl.createVertexArray()!;
    gl.bindVertexArray(this.quad);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.palTex = gl.createTexture()!;
    this.fcolTex = gl.createTexture()!;
    this.bakeFb = gl.createFramebuffer()!;
  }
  fcolTex: WebGLTexture;
  bakeFb: WebGLFramebuffer;

  async load(base: string) {
    this.base = base;
    const r = await fetch(`${base}/map.json`);
    if (!r.ok) throw new Error("no map data");
    this.meta = await r.json();
    // which tiles exist: one bit per tile and level (mapdata.py compact()); older map.json: lists
    const bits = (z: number, b64: string, add: (k: string) => void) => {
      const raw = atob(b64), side = 1 << z;
      for (let i = 0; i < raw.length; i++) {
        const v = raw.charCodeAt(i);
        if (v) for (let j = 0; j < 8; j++) if (v & (1 << j)) { const t = i * 8 + j; add(`${z}/${t % side}/${Math.floor(t / side)}`); }
      }
    };
    const d = this.meta.density, q = this.meta.points;
    if (d.mask) for (const [z, m] of Object.entries(d.mask)) bits(+z, m as string, (k) => this.dHave.add(k));
    else for (const [z, list] of Object.entries(d.tiles)) for (const [x, y] of list as number[][]) this.dHave.add(`${z}/${x}/${y}`);
    if (q.mask) for (const [z, m] of Object.entries(q.mask)) bits(+z, m as string, (k) => this.pHave.set(k, 1));
    else for (const [k, n] of Object.entries(q.tiles)) this.pHave.set(k, n as number);
    this.palKey = "";
  }

  setPalette(p: Palette) { this.pal = p; this.palKey = ""; this.onChange(); }

  private fieldRGB(): number[][] {
    return this.meta.fields.map((f: any, i: number) => i === 0 ? null : this.pal.fields[f.name] ?? null);
  }

  /** Star colours: the field's colour, nudged a little per subfield so neighbouring specialties read apart. */
  private palette() {
    const key = `${this.pal.dark}:${Object.keys(this.pal.fields).length}`;
    if (key === this.palKey || !this.meta) return;
    this.palKey = key;
    const fields = this.fieldRGB(), grey = this.grey();
    const seen: Record<number, number> = {};
    const data = new Uint8Array(256 * 4);
    this.meta.subfields.forEach((s: any, i: number) => {
      const n = (seen[s.field] = (seen[s.field] ?? -1) + 1);
      const c = fields[s.field] ?? grey, t = ((n * 0.618) % 1) - 0.5, amt = 0.07;
      const v = c.map((x: number, j: number) => Math.min(1, Math.max(0, x + t * amt * (j === 1 ? -1 : 1))));
      data.set([v[0] * 255, v[1] * 255, v[2] * 255, 255], i * 4);
    });
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    // the field's own colour per subfield code (alpha 0 = unclassified), for the bake
    const fc = new Uint8Array(256 * 4);
    this.meta.subfields.forEach((s: any, i: number) => {
      const c = fields[s.field];
      if (c) fc.set([c[0] * 255, c[1] * 255, c[2] * 255, 255], i * 4);
    });
    gl.bindTexture(gl.TEXTURE_2D, this.fcolTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, fc);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }
  private grey() { return this.pal.dark ? [0.62, 0.6, 0.57] : [0.5, 0.49, 0.47]; }

  /** Raw planes -> light texture, on the GPU (one 258 x 258 pass). */
  private bake(t: DTile) {
    const gl = this.gl, P = this.progs.bake;
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    if (!t.baked) {   // first bake: allocate
      if (this.hdr) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, SIDE, SIDE, 0, gl.RGBA, gl.HALF_FLOAT, null);
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, SIDE, SIDE, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      this.texParams(gl.LINEAR);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bakeFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
    gl.viewport(0, 0, SIDE, SIDE);
    gl.disable(gl.BLEND);
    gl.useProgram(P.p);
    const units: [string, WebGLTexture][] = [["u_cnt", t.cntTex], ["u_sub", t.subTex], ["u_pur", t.purTex], ["u_fcol", this.fcolTex]];
    units.forEach(([name, tex], i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(P.u[name], i); });
    const g = this.grey();
    gl.uniform3f(P.u.u_grey, g[0], g[1], g[2]);
    gl.uniform1f(P.u.u_logq, this.meta.density.log_scale);
    gl.uniform1f(P.u.u_lin, this.hdr ? 1 : 0);
    gl.bindVertexArray(this.quad);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.activeTexture(gl.TEXTURE0);
    t.baked = this.palKey;
  }

  private loadDensity(key: string) {
    if (this.loading.has(key)) return;
    const p = (async () => {
      try {
        const r = await fetch(`${this.base}/d/${key}.bin`);
        if (!r.ok) return;
        const buf = await gunzipBody(r);
        const N = SIDE * SIDE, cnt = new Uint16Array(buf, 0, N), sub = new Uint8Array(buf, 2 * N, N), pur = new Uint8Array(buf, 3 * N, N);
        const gl = this.gl;
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        const mk = (internal: number, format: number, type: number, data: ArrayBufferView) => {
          const t = gl.createTexture()!;
          gl.bindTexture(gl.TEXTURE_2D, t);
          gl.texImage2D(gl.TEXTURE_2D, 0, internal, SIDE, SIDE, 0, format, type, data);
          this.texParams(gl.NEAREST);
          return t;
        };
        const cntTex = mk(gl.R16UI, gl.RED_INTEGER, gl.UNSIGNED_SHORT, cnt);
        const subTex = mk(gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE, sub);
        const purTex = mk(gl.R8, gl.RED, gl.UNSIGNED_BYTE, pur);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        const [z, x, y] = key.split("/").map(Number);
        this.dtiles.set(key, { key, z, x, y, cntTex, subTex, purTex, tex: gl.createTexture()!, cnt, sub, born: 0, baked: "" });
        this.evict();
        this.onChange();
      } catch { /* a missing tile: its parent keeps showing */ }
      finally { this.loading.delete(key); }
    })();
    this.loading.set(key, p);
  }

  /** Usable this frame: loaded and baked with the current palette (at most a few bakes per frame). */
  private ready(t: DTile | undefined, now: number): DTile | undefined {
    if (!t) return undefined;
    if (t.baked !== this.palKey) {
      if (this.bakesLeft <= 0) return t.baked ? t : undefined;   // an old palette's bake is better than nothing
      this.bakesLeft--;
      const first = !t.baked;
      this.bake(t);
      if (first) t.born = now;
    }
    return t;
  }
  private bakesLeft = 0;
  /** set by the caller: the map is moving (glide, fling, drag, pinch, wheel) */
  moving = false;
  /** dev: switch passes off to time them (tools/check scripts) */
  dbg = { haze: true, stars: true, glow: true, forceRs: 0, finish: false };
  wall: number[] = [];
  lastRs = 1;
  /** render pixels while moving; adapted to the GPU from its timer */
  moveBudget = 2.0e6;
  /** called with the time between two frames while moving: aim at 60 fps, spend what is left on pixels */
  private slow = 0; private fast = 0;
  adaptFrame(dt: number) {
    const max = 16e6, min = 0.6e6;
    if (dt > 21) { this.fast = 0; if (++this.slow >= 3) { this.moveBudget = Math.max(min, this.moveBudget * 0.8); this.slow = 0; } }
    else if (dt < 17.5) { this.slow = 0; if (++this.fast >= 20) { this.moveBudget = Math.min(max, this.moveBudget * 1.12); this.fast = 0; } }
  }
  private lastMovingFrame = false;
  BAKES = 3;

  private texParams(filter: number) {
    const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private loadPoints(key: string) {
    if (this.loading.has("p" + key)) return;
    const p = (async () => {
      try {
        const r = await fetch(`${this.base}/p/${key}.bin`);
        if (!r.ok) return;
        const raw = new Uint16Array(await gunzipBody(r));
        const gl = this.gl, vao = gl.createVertexArray()!, buf = gl.createBuffer()!;
        gl.bindVertexArray(vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, raw, gl.STATIC_DRAW);
        const P = this.progs.stars.p;
        const l0 = gl.getAttribLocation(P, "a_xy"), l1 = gl.getAttribLocation(P, "a_info"), l2 = gl.getAttribLocation(P, "a_sub");
        gl.enableVertexAttribArray(l0); gl.vertexAttribPointer(l0, 2, gl.UNSIGNED_SHORT, true, 8, 0);
        gl.enableVertexAttribArray(l1); gl.vertexAttribIPointer(l1, 1, gl.UNSIGNED_SHORT, 8, 4);
        gl.enableVertexAttribArray(l2); gl.vertexAttribIPointer(l2, 1, gl.UNSIGNED_SHORT, 8, 6);
        gl.bindVertexArray(null);
        const [z, x, y] = key.split("/").map(Number);
        this.ptiles.set(key, { key, z, x, y, n: raw.length / 4, vao, buf, raw, born: performance.now(), shown: 0 });
        this.evict();
        this.onChange();
      } catch { /* */ }
      finally { this.loading.delete("p" + key); }
    })();
    this.loading.set("p" + key, p);
  }

  private wantD = new Set<string>();
  private wantP = new Set<string>();
  private evict() {
    if (this.dtiles.size > 260) for (const [k, t] of this.dtiles) {
      if (this.wantD.has(k) || t.z <= 2) continue;
      for (const x of [t.tex, t.cntTex, t.subTex, t.purTex]) this.gl.deleteTexture(x);
      this.dtiles.delete(k);
      if (this.dtiles.size <= 200) break;
    }
    if (this.ptiles.size > 260) for (const [k, t] of this.ptiles) {
      if (this.wantP.has(k) || t.z <= 1) continue;
      this.gl.deleteVertexArray(t.vao); this.gl.deleteBuffer(t.buf); this.ptiles.delete(k);
      if (this.ptiles.size <= 200) break;
    }
  }

  // ---------------------------------------------------------------- render targets
  private target(name: string, w: number, h: number): Target {
    const gl = this.gl, old = this.targets[name];
    if (old && old.w === w && old.h === h) return old;
    if (old) { gl.deleteTexture(old.tex); gl.deleteFramebuffer(old.fb); }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (this.hdr) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    this.texParams(gl.LINEAR);
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (this.hdr && gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      // float targets advertised but not usable: 8 bits from now on
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteTexture(tex); gl.deleteFramebuffer(fb);
      this.hdr = false; this.scale = 0.125;
      for (const k of Object.keys(this.targets)) delete this.targets[k];
      return this.target(name, w, h);
    }
    return (this.targets[name] = { tex, fb, w, h });
  }
  private blackT: Target | null = null;
  private black(): Target {
    if (!this.blackT) {
      const gl = this.gl, tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
      this.blackT = { tex, fb: null as any, w: 1, h: 1 };
    }
    return this.blackT;
  }
  private into(t: Target | null, W?: number, H?: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t ? t.fb : null);
    gl.viewport(0, 0, t ? t.w : W!, t ? t.h : H!);
  }

  /** The world rectangle on screen (css px) -> visible world bounds. */
  private bounds(c: Cam, w: number, h: number) {
    const span = c.S * c.z;
    return { x0: -c.ox / span, y0: -c.oy / span, x1: (w - c.ox) / span, y1: (h - c.oy) / span };
  }

  /** Density level: a texel about 1.5 device pixels wide (as sharp as the data allows). */
  densityLevel(c: Cam, dpr: number) {
    return Math.max(0, Math.min(this.meta.density.levels - 1, Math.ceil(Math.log2(c.S * c.z * dpr / (TILE * 1.5)))));
  }
  /** Point levels drawn: those whose tiles are at least 256 css px wide on screen. */
  pointLevel(c: Cam) {
    return Math.max(0, Math.min(this.meta.points.levels - 1, Math.floor(Math.log2(c.S * c.z / 256))));
  }
  /** Stars per css px^2 at most: many fine grains far out (they make the texture), fewer and bigger close in.
   *  A tile draws the first N of its studies (most cited first), N = its share of this budget. */
  starDensity(z: number) { return 1 / (9 * Math.pow(Math.max(1, z), 0.75)); }

  /** timing of the last frames (dev checks): CPU ms of draw(), and GPU ms when the timer extension exists */
  perf = { cpu: [] as number[], gpu: [] as number[], q: [] as WebGLQuery[], phases: [] as Record<string, number>[] };
  private ph: Record<string, number> = {}; private phT = 0;
  private mark(name: string) { const t = performance.now(); this.ph[name] = (this.ph[name] ?? 0) + t - this.phT; this.phT = t; }
  timer: any = null;
  draw(c: Cam, w: number, h: number) {
    const t0 = performance.now();
    if (this.timer === null) this.timer = this.gl.getExtension("EXT_disjoint_timer_query_webgl2") || false;
    let q: WebGLQuery | null = null;
    if (this.timer) {
      const gl = this.gl;
      while (this.perf.q.length && gl.getQueryParameter(this.perf.q[0], gl.QUERY_RESULT_AVAILABLE)) {
        const x = this.perf.q.shift()!, ms = gl.getQueryParameter(x, gl.QUERY_RESULT) / 1e6;
        this.perf.gpu.push(ms); gl.deleteQuery(x);
      }
      if (this.perf.q.length < 8) { q = gl.createQuery()!; gl.beginQuery(this.timer.TIME_ELAPSED_EXT, q); }
    }
    this.ph = {}; this.phT = performance.now();
    this.lastMovingFrame = this.moving;
    const tw = performance.now();
    this.drawFrame(c, w, h);
    if (this.dbg.finish) { const px = new Uint8Array(4); this.gl.readPixels(0, 0, 1, 1, this.gl.RGBA, this.gl.UNSIGNED_BYTE, px); this.wall.push(performance.now() - tw); }
    this.perf.phases.push(this.ph); if (this.perf.phases.length > 600) this.perf.phases.splice(0, 300);
    if (q) { this.gl.endQuery(this.timer.TIME_ELAPSED_EXT); this.perf.q.push(q); }
    this.perf.cpu.push(performance.now() - t0);
    if (this.perf.cpu.length > 600) this.perf.cpu.splice(0, 300);
    if (this.perf.gpu.length > 600) this.perf.gpu.splice(0, 300);
  }

  private drawFrame(c: Cam, w: number, h: number) {
    const gl = this.gl, dpr = Math.min(devicePixelRatio || 1, 2);
    const W0 = Math.round(w * dpr), H0 = Math.round(h * dpr);
    // dynamic resolution: while the map moves, the whole picture is painted into fewer pixels and the browser's
    // compositor stretches the canvas (the eye cannot see detail in motion; weak GPUs keep 60 fps); the moment it
    // stops, one full-resolution frame. The budget follows the frame rate reached (adaptFrame): fast GPUs never drop.
    const rs = this.dbg.forceRs || (this.moving ? Math.min(1, Math.sqrt(this.moveBudget / (W0 * H0))) : 1);
    const W = Math.max(1, Math.round(W0 * rs)), H = Math.max(1, Math.round(H0 * rs)), r = dpr * (W / W0);
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    const Wr = W, Hr = H;
    const tag = rs < 1 ? "lo" : "hi";
    this.lastRs = rs;
    const bg = this.pal.bg;
    if (!this.meta) { this.into(null, W, H); gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT); return; }
    this.palette();
    const b = this.bounds(c, w, h), now = performance.now(), sc = this.scale;
    let busy = false, pending = false;
    this.bakesLeft = this.BAKES;
    this.mark("setup");
    // dark theme: the stars add into the same light buffer (two full-screen passes fewer); light theme needs them apart
    const merged = this.pal.dark;
    const haze = this.target(`haze:${tag}`, Wr, Hr), stars = merged ? haze : this.target(`stars:${tag}`, Wr, Hr);
    const lv = (n: number) => this.target(`d${n}:${tag}`, Math.max(1, Math.ceil(Wr / 2 ** n)), Math.max(1, Math.ceil(Hr / 2 ** n)));
    const midL = Math.max(1, Math.min(4, Math.round(Math.log2(3.7 * r / 1.7))));
    const bigL = Math.max(midL + 1, Math.min(5, Math.round(Math.log2(13.5 * r / 1.7))));

    // ------------------------------------------------------------- 1a. density -> haze
    const L = this.densityLevel(c, dpr), n = 1 << L;
    this.dLevel = L;
    this.wantD.clear();
    const draws: { t: DTile; sx: number; sy: number; ss: number; uv: number[]; w: number }[] = [];
    const missing: [number, string][] = [];
    const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    for (let ty = Math.max(0, Math.floor(b.y0 * n)); ty <= Math.min(n - 1, Math.floor(b.y1 * n)); ty++)
      for (let tx = Math.max(0, Math.floor(b.x0 * n)); tx <= Math.min(n - 1, Math.floor(b.x1 * n)); tx++) {
        const key = `${L}/${tx}/${ty}`;
        if (!this.dHave.has(key)) continue;
        this.wantD.add(key);
        const t = this.ready(this.dtiles.get(key), now);
        const wx = tx / n, wy = ty / n, ws = 1 / n;
        const fade = t ? Math.min(1, (now - t.born) / 300) : 0;
        if (fade < 1) {   // the best loaded ancestor fills in until this tile arrives (light adds up: weights sum to 1)
          for (let l = L - 1; l >= 0; l--) {
            const m = 1 << (L - l), a = this.ready(this.dtiles.get(`${l}/${tx >> (L - l)}/${ty >> (L - l)}`), now);
            if (!a) continue;
            draws.push({ t: a, sx: wx, sy: wy, ss: ws, uv: [(tx % m) / m, (ty % m) / m, 1 / m], w: 1 - fade });
            break;
          }
          busy = true; pending = true;
        }
        if (t) draws.push({ t, sx: wx, sy: wy, ss: ws, uv: [0, 0, 1], w: fade });
        else if (!this.dtiles.has(key)) missing.push([Math.hypot(wx + ws / 2 - cx, wy + ws / 2 - cy), key]);
        else { busy = true; pending = true; }   // loaded, waiting for its bake
      }
    missing.sort((p, q) => p[0] - q[0]);
    for (const [, k] of missing.slice(0, 16)) this.loadDensity(k);
    for (let l = Math.max(0, L - 3); l < L; l++) {   // coarse cover ready for fast moves
      const m = 1 << l;
      for (let ty = Math.max(0, Math.floor(b.y0 * m)); ty <= Math.min(m - 1, Math.floor(b.y1 * m)); ty++)
        for (let tx = Math.max(0, Math.floor(b.x0 * m)); tx <= Math.min(m - 1, Math.floor(b.x1 * m)); tx++) {
          const k = `${l}/${tx}/${ty}`;
          if (this.dHave.has(k)) { this.wantD.add(k); if (!this.dtiles.has(k)) this.loadDensity(k); else this.ready(this.dtiles.get(k), now); }
        }
    }
    if (missing.length || this.loading.size) busy = true;
    if (missing.length) pending = true;
    this.mark("tiles+bakes");

    if (now - this.lastStats > 180) { this.lastStats = now; this.kTarget = this.exposure(c, dpr, b, L) || this.kTarget; }
    if (!this.k) this.k = this.kTarget;
    if (!this.k) pending = true;
    else if (this.kTarget) {
      this.k *= Math.exp(Math.log(this.kTarget / this.k) * 0.1);
      if (Math.abs(Math.log(this.kTarget / this.k)) > 0.01) { busy = true; pending = true; }
    }

    this.mark("exposure");
    const span = c.S * c.z * r, spanDev = c.S * c.z * dpr;   // render px, device px (brightness is per device px)
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    this.into(haze); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    const P = this.progs.density;
    gl.useProgram(P.p);
    gl.bindVertexArray(this.quad);
    gl.uniform2f(P.u.u_view, Wr, Hr);
    // very close in, the stars lead and the haze steps back
    const hazeW = Math.max(0.5, Math.min(1, 1 - (Math.log2(c.z) - 5) / 3));
    gl.uniform1f(P.u.u_k, this.k * hazeW); gl.uniform1f(P.u.u_scale, sc);
    gl.uniform1i(P.u.u_t, 0); gl.uniform1f(P.u.u_lin, this.hdr ? 1 : 0);
    gl.uniform1f(P.u.u_knee, this.kneeLevel());
    for (const d of draws) {
      if (d.w <= 0 || !this.dbg.haze) continue;
      const texelPx = spanDev / ((1 << d.t.z) * TILE);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, d.t.tex);
      gl.uniform1f(P.u.u_area, texelPx * texelPx);
      gl.uniform1f(P.u.u_w, d.w);
      gl.uniform1f(P.u.u_cubic, texelPx * (r / dpr) > 2.5 ? 1 : 0);
      gl.uniform4f(P.u.u_rect, c.ox * r + d.sx * span, c.oy * r + d.sy * span, d.ss * span, d.ss * span);
      const u0 = (1 + d.uv[0] * TILE) / SIDE, v0 = (1 + d.uv[1] * TILE) / SIDE, us = d.uv[2] * TILE / SIDE;
      gl.uniform4f(P.u.u_uv, u0, v0, us, us);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    this.mark("haze");
    // ------------------------------------------------------------- 1b. stars
    const Lp = this.pointLevel(c);
    this.pLevel = Lp;
    this.wantP.clear();
    this.into(stars); if (!merged) gl.clear(gl.COLOR_BUFFER_BIT);
    // overlapping stars keep the brightest instead of adding up: a cluster of thousands of studies is a bright
    // patch, not a white-hot blob (and in the light theme, not a black stain of ink)
    gl.blendEquation(gl.MAX);
    const S = this.progs.stars;
    gl.useProgram(S.p);
    gl.uniform2f(S.u.u_view, Wr, Hr);
    gl.uniform3f(S.u.u_cam, c.ox * r, c.oy * r, span);
    gl.uniform1f(S.u.u_size, Math.min(2.4, 0.8 + Math.max(0, Math.log2(c.z)) * 0.2));
    gl.uniform1f(S.u.u_dpr, r); gl.uniform1f(S.u.u_scale, sc);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.palTex); gl.uniform1i(S.u.u_pal, 0);
    const budget = this.starDensity(c.z);
    const pm: [number, string][] = [];
    for (let l = 0; l <= Lp; l++) {
      const m = 1 << l;
      for (let ty = Math.max(0, Math.floor(b.y0 * m)); ty <= Math.min(m - 1, Math.floor(b.y1 * m)); ty++)
        for (let tx = Math.max(0, Math.floor(b.x0 * m)); tx <= Math.min(m - 1, Math.floor(b.x1 * m)); tx++) {
          const k = `${l}/${tx}/${ty}`;
          if (!this.pHave.has(k)) continue;
          this.wantP.add(k);
          const t = this.ptiles.get(k);
          if (!t) { pm.push([l * 10 + Math.hypot((tx + .5) / m - cx, (ty + .5) / m - cy), k]); continue; }
          if (!this.dbg.stars) continue;
          const fade = Math.min(1, (now - t.born) / 400);
          if (fade < 1) { busy = true; pending = true; }
          const side = c.S * c.z / m;
          t.shown = Math.min(t.n, Math.ceil(budget * side * side / (Lp + 1)));
          gl.uniform4f(S.u.u_tile, tx / m, ty / m, 1 / m, 0);
          gl.uniform1f(S.u.u_gain, fade * this.starGain(c.z));
          gl.bindVertexArray(t.vao);
          gl.drawArrays(gl.POINTS, 0, t.shown);
        }
    }
    pm.sort((p, q) => p[0] - q[0]);
    for (const [, k] of pm.slice(0, 10)) this.loadPoints(k);
    if (pm.length) { busy = true; pending = true; }
    // idle: fetch the next finer level around the middle of the screen, where zooming in usually goes
    if (!pending && this.loading.size < 3 && L + 1 < this.meta.density.levels) {
      const m = 2 * n, qx = (b.x1 - b.x0) / 4, qy = (b.y1 - b.y0) / 4;
      let left = 16;
      for (let ty = Math.max(0, Math.floor((cy - qy) * m)); ty <= Math.min(m - 1, Math.floor((cy + qy) * m)) && left; ty++)
        for (let tx = Math.max(0, Math.floor((cx - qx) * m)); tx <= Math.min(m - 1, Math.floor((cx + qx) * m)) && left; tx++) {
          const k = `${L + 1}/${tx}/${ty}`;
          if (this.dHave.has(k) && !this.dtiles.has(k)) { this.wantD.add(k); this.loadDensity(k); left--; }
          else if (this.dtiles.has(k)) this.wantD.add(k);
        }
    }
    this.complete = !pending;

    this.mark("stars");
    // ------------------------------------------------------------- 2. glow: down chain, blur at two sizes
    gl.blendEquation(gl.FUNC_ADD);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.quad);
    const D = this.progs.down;
    gl.useProgram(D.p);
    gl.uniform1i(D.u.u_a, 0); gl.uniform1i(D.u.u_b, 1);
    let src: Target = haze;
    for (let i = 1; i <= bigL; i++) {
      const dst = lv(i);
      this.into(dst);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, stars.tex);
      gl.uniform1f(D.u.u_two, i === 1 && !merged ? 1 : 0);
      gl.uniform2f(D.u.u_texel, 1 / src.w, 1 / src.h);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      src = dst;
    }
    const B = this.progs.blur;
    gl.useProgram(B.p);
    gl.uniform1i(B.u.u_a, 0);
    const blur = (i: number) => {
      const a = lv(i), tmp = this.target(`t${i}`, a.w, a.h);
      this.into(tmp); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, a.tex);
      gl.uniform2f(B.u.u_dir, 1 / a.w, 0); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.into(a); gl.bindTexture(gl.TEXTURE_2D, tmp.tex);
      gl.uniform2f(B.u.u_dir, 0, 1 / a.h); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return a;
    };
    const mid = blur(midL), big = blur(bigL);

    this.mark("glow");
    // ------------------------------------------------------------- 3. film curve -> screen
    const C = this.progs.compose;
    this.into(null, W, H);
    gl.useProgram(C.p);
    const tex = [haze, merged ? this.black() : stars, mid, big];
    ["u_haze", "u_stars", "u_mid", "u_big"].forEach((name, i) => {
      gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, tex[i].tex); gl.uniform1i(C.u[name], i);
    });
    gl.uniform2f(C.u.u_midSize, mid.w, mid.h); gl.uniform2f(C.u.u_bigSize, big.w, big.h);
    gl.uniform1f(C.u.u_dark, this.pal.dark ? 1 : 0); gl.uniform3f(C.u.u_bg, bg[0], bg[1], bg[2]);
    gl.uniform1f(C.u.u_scale, sc);
    // make_map.py: sharp x1.6 + fine blur x1.4 (both "sharp" here), glow 6 px x2.2, glow 22 px x3.0
    // light theme: each star is one grain of ink whatever its light (undo the star gain)
    gl.uniform1f(C.u.u_grain, this.grain(c.z) / Math.max(1e-6, this.starGain(c.z)));
    gl.uniform1f(C.u.u_wSharp, 3.0); gl.uniform1f(C.u.u_wMid, 2.2); gl.uniform1f(C.u.u_wBig, 3.0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE0);
    this.mark("compose");
    if (busy) this.onChange();
  }

  /** Exposure, as make_map.py: light = studies per pixel / the 99.7th percentile of what is on screen.
   *  A histogram of the raw (log-scaled) counts of a sample of the visible pixels: no sorting, no per-sample pow. */
  private hist = new Uint32Array(4096);
  private exposure(c: Cam, dpr: number, b: { x0: number; y0: number; x1: number; y1: number }, L: number) {
    const n = 1 << L, span = c.S * c.z * dpr, texelPx = span / (n * TILE), area = texelPx * texelPx;
    const h = this.hist; h.fill(0);
    let total = 0;
    for (const t of this.dtiles.values()) {
      if (t.z !== L || !t.baked) continue;
      const wx = t.x / n, wy = t.y / n;
      if (wx > b.x1 || wy > b.y1 || wx + 1 / n < b.x0 || wy + 1 / n < b.y0) continue;
      // the visible texel range of this tile, sampled on a grid of at most ~48 x 48
      const ix0 = Math.max(0, Math.floor((b.x0 - wx) * n * TILE)), ix1 = Math.min(TILE - 1, Math.floor((b.x1 - wx) * n * TILE));
      const iy0 = Math.max(0, Math.floor((b.y0 - wy) * n * TILE)), iy1 = Math.min(TILE - 1, Math.floor((b.y1 - wy) * n * TILE));
      const sx = Math.max(1, ((ix1 - ix0) / 48) | 0), sy = Math.max(1, ((iy1 - iy0) / 48) | 0);
      const cnt = t.cnt;
      for (let y = iy0; y <= iy1; y += sy) for (let x = ix0, o = (y + 1) * SIDE + 1; x <= ix1; x += sx) {
        const v = cnt[o + x];
        if (v) { h[v >> 4]++; total++; }
      }
    }
    if (total < 50) return 0;
    let want = Math.floor(total * 0.003), i = 4095;
    for (; i > 0 && want >= h[i]; i--) want -= h[i];
    const ref = (Math.pow(2, ((i << 4) + 8) / this.meta.density.log_scale) - 1) / area;
    // EXPOSURE: matched to the homepage image at the whole-map view (measured on 1x and 2x screens), so the
    // swap from the image to the live map is not a jump; light needs more, low-density screens a little less
    const e = this.kneeLevel();
    // capped: where studies are few, a lone study stays a faint glow (its star carries it), not a bright patch
    return Math.min(e / ref, 60);
  }
  EXPOSURE: number | null = null;
  /** the light the 99.7th percentile gets (exposure() sets k so that ref x k = this); matched to the homepage images
   *  (make_map.py) on 1x and 2x screens, 2026-10-10, with stars blended by MAX */
  kneeLevel() { const hi = Math.min(devicePixelRatio || 1, 2) >= 1.5; return this.EXPOSURE ?? (this.pal.dark ? (hi ? 2 : 1) : (hi ? 3 : 2.2)); }
  /** Star light: faint far out (the density carries the picture, as on the homepage image), stronger closer in,
   *  where single studies are what there is to see. */
  STARS = 0.003;
  GRAIN = 0.06;
  starGain(z: number) { return Math.min(0.2, this.STARS * Math.pow(Math.max(1, z), 0.95)); }
  /** light theme: ink per star; fine dust far out, real dots close in */
  grain(z: number) { return Math.min(0.28, this.GRAIN * Math.pow(Math.max(1, z), 0.6)); }

  /** The star under a screen point (css px), among the stars drawn. */
  nearest(c: Cam, sx: number, sy: number, reach = 14): Star | null {
    const span = c.S * c.z;
    let best: Star | null = null, bd = reach * reach;
    for (const t of this.ptiles.values()) {
      if (!this.wantP.has(t.key)) continue;
      const m = 1 << t.z, ox = t.x / m, oy = t.y / m, s = 1 / m;
      const x0 = c.ox + ox * span, y0 = c.oy + oy * span, ss = s * span;
      if (sx < x0 - reach || sy < y0 - reach || sx > x0 + ss + reach || sy > y0 + ss + reach) continue;
      const r = t.raw;
      for (let i = 0, j = 0; i < t.shown; i++, j += 4) {
        const px = x0 + (r[j] / 65535) * ss, py = y0 + (r[j + 1] / 65535) * ss;
        const d = (px - sx) ** 2 + (py - sy) ** 2;
        if (d < bd) { bd = d; best = { x: ox + (r[j] / 65535) * s, y: oy + (r[j + 1] / 65535) * s, info: r[j + 2], sub: r[j + 3], key: t.key, i }; }
      }
    }
    return best;
  }

  /** The field of the place under a world point, if loaded. */
  fieldAt(x: number, y: number): number | null {
    const L = this.dLevel, n = 1 << L, t = this.dtiles.get(`${L}/${Math.floor(x * n)}/${Math.floor(y * n)}`);
    if (!t) return null;
    const px = 1 + Math.floor((x * n - t.x) * TILE), py = 1 + Math.floor((y * n - t.y) * TILE);
    return t.cnt[py * SIDE + px] ? this.meta.subfields[t.sub[py * SIDE + px]]?.field ?? null : null;
  }
}
