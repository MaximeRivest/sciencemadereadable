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

type DTile = { key: string; z: number; x: number; y: number; tex: WebGLTexture; col: WebGLTexture;
               D: Float32Array; sub: Uint8Array; pur: Uint8Array; born: number; colFor: string };
type PTile = { key: string; z: number; x: number; y: number; n: number; vao: WebGLVertexArrayObject; buf: WebGLBuffer;
               raw: Uint16Array; born: number; shown: number };
type Target = { tex: WebGLTexture; fb: WebGLFramebuffer; w: number; h: number };

async function gunzip(b: ArrayBuffer): Promise<ArrayBuffer> {
  const s = new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(s).arrayBuffer();
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
in vec2 v_uv; uniform sampler2D u_d; uniform sampler2D u_c;
uniform float u_area;      // device px per texel, squared
uniform float u_k;         // exposure: light per (study per device px)
uniform float u_w;         // weight (cross-fade between a tile and its parent)
uniform float u_cubic;     // 1 when a texel is several pixels wide
uniform float u_scale;     // light buffer scale (8-bit fallback)
out vec4 o;
${CUBIC}
void main() {
  vec2 sz = vec2(258.0);
  float n = u_cubic > 0.5 ? cubic(u_d, v_uv, sz).r : texture(u_d, v_uv).r;   // studies per texel
  vec4 c = u_cubic > 0.5 ? cubic(u_c, v_uv, sz) : texture(u_c, v_uv);
  if (n <= 1e-5 || c.a <= 1e-3) discard;
  vec3 col = c.rgb / c.a;
  o = vec4(col * (n / u_area) * u_k * u_w * u_scale, 1.0);
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
      density: shader(gl, DENSITY_VS, DENSITY_FS), stars: shader(gl, STAR_VS, STAR_FS),
      down: shader(gl, FULL_VS, DOWN_FS), blur: shader(gl, FULL_VS, BLUR_FS), compose: shader(gl, FULL_VS, COMPOSE_FS),
    };
    this.quad = gl.createVertexArray()!;
    gl.bindVertexArray(this.quad);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.palTex = gl.createTexture()!;
  }

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
    for (const t of this.dtiles.values()) t.colFor = "";
  }
  private grey() { return this.pal.dark ? [0.62, 0.6, 0.57] : [0.5, 0.49, 0.47]; }

  /** A tile's colour: its dominant field's colour (as make_map.py: one colour per field), paler where fields
   *  mix; places with only unclassified studies are muted (x 0.3, as make_map.py). */
  private colourTexture(t: DTile) {
    if (t.colFor === this.palKey) return;
    const fields = this.fieldRGB(), grey = this.grey(), subs = this.meta.subfields;
    const px = new Uint8Array(SIDE * SIDE * 4);
    for (let i = 0; i < SIDE * SIDE; i++) {
      if (!t.D[i]) continue;
      const f = subs[t.sub[i]]?.field ?? 0, c = fields[f];
      if (!c) { px.set([grey[0] * 0.3 * 255, grey[1] * 0.3 * 255, grey[2] * 0.3 * 255, 255], i * 4); continue; }
      const p = t.pur[i] / 255, m = 0.35 + 0.65 * p;
      px.set([(grey[0] + (c[0] - grey[0]) * m) * 255, (grey[1] + (c[1] - grey[1]) * m) * 255,
              (grey[2] + (c[2] - grey[2]) * m) * 255, 255], i * 4);
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, t.col);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, SIDE, SIDE, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    t.colFor = this.palKey;
  }

  private loadDensity(key: string) {
    if (this.loading.has(key)) return;
    const p = (async () => {
      try {
        const r = await fetch(`${this.base}/d/${key}.bin`);
        if (!r.ok) return;
        const buf = await gunzip(await r.arrayBuffer());
        const N = SIDE * SIDE, cnt = new Uint16Array(buf, 0, N), sub = new Uint8Array(buf, 2 * N, N), pur = new Uint8Array(buf, 3 * N, N);
        const D = new Float32Array(N), scale = this.meta.density.log_scale;
        for (let i = 0; i < N; i++) if (cnt[i]) D[i] = Math.pow(2, cnt[i] / scale) - 1;
        const gl = this.gl;
        const tex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, SIDE, SIDE, 0, gl.RED, gl.FLOAT, D);
        this.texParams(gl.LINEAR);
        const col = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, col); this.texParams(gl.LINEAR);
        const [z, x, y] = key.split("/").map(Number);
        this.dtiles.set(key, { key, z, x, y, tex, col, D, sub: new Uint8Array(sub), pur: new Uint8Array(pur), born: performance.now(), colFor: "" });
        this.evict();
        this.onChange();
      } catch { /* a missing tile: its parent keeps showing */ }
      finally { this.loading.delete(key); }
    })();
    this.loading.set(key, p);
  }

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
        const raw = new Uint16Array(await gunzip(await r.arrayBuffer()));
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
      this.gl.deleteTexture(t.tex); this.gl.deleteTexture(t.col); this.dtiles.delete(k);
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

  draw(c: Cam, w: number, h: number) {
    const gl = this.gl, dpr = Math.min(devicePixelRatio || 1, 2);
    const W = Math.round(w * dpr), H = Math.round(h * dpr);
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    const bg = this.pal.bg;
    if (!this.meta) { this.into(null, W, H); gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT); return; }
    this.palette();
    const b = this.bounds(c, w, h), now = performance.now(), sc = this.scale;
    let busy = false, pending = false;
    const haze = this.target("haze", W, H), stars = this.target("stars", W, H);
    const lv = (n: number) => this.target(`d${n}`, Math.max(1, Math.ceil(W / 2 ** n)), Math.max(1, Math.ceil(H / 2 ** n)));
    const midL = Math.max(1, Math.min(4, Math.round(Math.log2(3.7 * dpr / 1.7))));
    const bigL = Math.max(midL + 1, Math.min(5, Math.round(Math.log2(13.5 * dpr / 1.7))));

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
        const t = this.dtiles.get(key);
        const wx = tx / n, wy = ty / n, ws = 1 / n;
        const fade = t ? Math.min(1, (now - t.born) / 300) : 0;
        if (fade < 1) {   // the best loaded ancestor fills in until this tile arrives (light adds up: weights sum to 1)
          for (let l = L - 1; l >= 0; l--) {
            const m = 1 << (L - l), a = this.dtiles.get(`${l}/${tx >> (L - l)}/${ty >> (L - l)}`);
            if (!a) continue;
            draws.push({ t: a, sx: wx, sy: wy, ss: ws, uv: [(tx % m) / m, (ty % m) / m, 1 / m], w: 1 - fade });
            break;
          }
          busy = true; pending = true;
        }
        if (t) draws.push({ t, sx: wx, sy: wy, ss: ws, uv: [0, 0, 1], w: fade });
        else missing.push([Math.hypot(wx + ws / 2 - cx, wy + ws / 2 - cy), key]);
      }
    missing.sort((p, q) => p[0] - q[0]);
    for (const [, k] of missing.slice(0, 16)) this.loadDensity(k);
    for (let l = Math.max(0, L - 3); l < L; l++) {   // coarse cover ready for fast moves
      const m = 1 << l;
      for (let ty = Math.max(0, Math.floor(b.y0 * m)); ty <= Math.min(m - 1, Math.floor(b.y1 * m)); ty++)
        for (let tx = Math.max(0, Math.floor(b.x0 * m)); tx <= Math.min(m - 1, Math.floor(b.x1 * m)); tx++) {
          const k = `${l}/${tx}/${ty}`;
          if (this.dHave.has(k)) { this.wantD.add(k); if (!this.dtiles.has(k)) this.loadDensity(k); }
        }
    }
    if (missing.length || this.loading.size) busy = true;
    if (missing.length) pending = true;

    if (now - this.lastStats > 180) { this.lastStats = now; this.kTarget = this.exposure(c, dpr, b, L) || this.kTarget; }
    if (!this.k) this.k = this.kTarget;
    if (!this.k) pending = true;
    else if (this.kTarget) {
      this.k *= Math.exp(Math.log(this.kTarget / this.k) * 0.1);
      if (Math.abs(Math.log(this.kTarget / this.k)) > 0.01) { busy = true; pending = true; }
    }

    const span = c.S * c.z * dpr;
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    this.into(haze); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    const P = this.progs.density;
    gl.useProgram(P.p);
    gl.bindVertexArray(this.quad);
    gl.uniform2f(P.u.u_view, W, H);
    // very close in, the stars lead and the haze steps back
    const hazeW = Math.max(0.5, Math.min(1, 1 - (Math.log2(c.z) - 5) / 3));
    gl.uniform1f(P.u.u_k, this.k * hazeW); gl.uniform1f(P.u.u_scale, sc);
    gl.uniform1i(P.u.u_d, 0); gl.uniform1i(P.u.u_c, 1);
    for (const d of draws) {
      if (d.w <= 0) continue;
      this.colourTexture(d.t);
      const texelPx = span / ((1 << d.t.z) * TILE);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, d.t.tex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, d.t.col);
      gl.uniform1f(P.u.u_area, texelPx * texelPx);
      gl.uniform1f(P.u.u_w, d.w);
      gl.uniform1f(P.u.u_cubic, texelPx > 2.5 ? 1 : 0);
      gl.uniform4f(P.u.u_rect, c.ox * dpr + d.sx * span, c.oy * dpr + d.sy * span, d.ss * span, d.ss * span);
      const u0 = (1 + d.uv[0] * TILE) / SIDE, v0 = (1 + d.uv[1] * TILE) / SIDE, us = d.uv[2] * TILE / SIDE;
      gl.uniform4f(P.u.u_uv, u0, v0, us, us);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // ------------------------------------------------------------- 1b. stars
    const Lp = this.pointLevel(c);
    this.pLevel = Lp;
    this.wantP.clear();
    this.into(stars); gl.clear(gl.COLOR_BUFFER_BIT);
    const S = this.progs.stars;
    gl.useProgram(S.p);
    gl.uniform2f(S.u.u_view, W, H);
    gl.uniform3f(S.u.u_cam, c.ox * dpr, c.oy * dpr, span);
    gl.uniform1f(S.u.u_size, Math.min(2.4, 0.8 + Math.max(0, Math.log2(c.z)) * 0.2));
    gl.uniform1f(S.u.u_dpr, dpr); gl.uniform1f(S.u.u_scale, sc);
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

    // ------------------------------------------------------------- 2. glow: down chain, blur at two sizes
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
      gl.uniform1f(D.u.u_two, i === 1 ? 1 : 0);
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

    // ------------------------------------------------------------- 3. film curve -> screen
    const C = this.progs.compose;
    this.into(null, W, H);
    gl.useProgram(C.p);
    const tex = [haze, stars, mid, big];
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
    if (busy) this.onChange();
  }

  /** Exposure, as make_map.py: light = studies per pixel / the 99.7th percentile of what is on screen. */
  private exposure(c: Cam, dpr: number, b: { x0: number; y0: number; x1: number; y1: number }, L: number) {
    const n = 1 << L, span = c.S * c.z * dpr, texelPx = span / (n * TILE), area = texelPx * texelPx;
    const vals: number[] = [];
    for (const t of this.dtiles.values()) {
      if (t.z !== L) continue;
      const wx = t.x / n, wy = t.y / n;
      if (wx > b.x1 || wy > b.y1 || wx + 1 / n < b.x0 || wy + 1 / n < b.y0) continue;
      for (let i = 0; i < 2000; i++) {
        const px = 1 + ((i * 97) % 256), py = 1 + ((i * 61 + (i >> 8) * 13) % 256);
        const x = wx + (px - 1) / (n * TILE), y = wy + (py - 1) / (n * TILE);
        if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
        const d = t.D[py * SIDE + px];
        if (d > 0) vals.push(d / area);
      }
    }
    if (vals.length < 50) return 0;
    vals.sort((p, q) => p - q);
    const ref = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.997))];
    // capped: where studies are few, a lone study stays a faint glow (its star carries it), not a bright patch
    // EXPOSURE: matched to the homepage image at the whole-map view (measured on 1x and 2x screens), so the
    // swap from the image to the live map is not a jump; light needs more, low-density screens a little less
    const e = this.EXPOSURE ?? (this.pal.dark ? 0.5 : 0.7) * Math.pow(dpr / 2, 0.45);
    return Math.min(e / ref, 60);
  }
  EXPOSURE: number | null = null;
  /** Star light: faint far out (the density carries the picture, as on the homepage image), stronger closer in,
   *  where single studies are what there is to see. */
  STARS = 0.003;
  GRAIN = 0.04;
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
    return t.D[py * SIDE + px] ? this.meta.subfields[t.sub[py * SIDE + px]]?.field ?? null : null;
  }
}
