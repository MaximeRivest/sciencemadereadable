/**
 * The map engine: draws the release's map data (pipeline/release/mapdata.py, "v2") with WebGL2.
 *
 *  - density: every study counted (nothing sampled), a pyramid of 256 px tiles with a 1 px apron, so
 *    hardware filtering has no seams. Brightness is the number of studies per screen pixel through a
 *    camera-like exposure that adapts to what is on screen; colour is the field (tinted by subfield),
 *    paler where fields mix.
 *  - stars: individual studies from the importance quadtree (most cited first), faded in by zoom and by
 *    citations: the classics shine from afar, every study is there once zoomed in.
 *
 * The caller owns the camera (atlas.ts) and calls draw() once per animation frame.
 */

export interface Cam { S: number; ox: number; oy: number; z: number }   // map box size (css px), world (0,0) on screen, zoom
export interface Palette { dark: boolean; bg: number[]; fields: Record<string, number[]> }
export interface Star { x: number; y: number; info: number; sub: number; key: string; i: number }

const TILE = 256, SIDE = 258;

type DTile = { key: string; z: number; x: number; y: number; tex: WebGLTexture; col: WebGLTexture;
               D: Float32Array; sub: Uint8Array; pur: Uint8Array; born: number; colFor: string };
type PTile = { key: string; z: number; x: number; y: number; n: number; vao: WebGLVertexArrayObject; buf: WebGLBuffer;
               raw: Uint16Array; born: number; shown: number };

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
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  const u: Record<string, WebGLUniformLocation> = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i)!; u[a.name] = gl.getUniformLocation(p, a.name)!; }
  return { p, u };
}

const DENSITY_VS = `#version 300 es
in vec2 a; uniform vec4 u_rect; uniform vec4 u_uv; uniform vec2 u_view;
out vec2 v_uv;
void main() {
  vec2 px = u_rect.xy + a * u_rect.zw;                  // device px
  v_uv = u_uv.xy + a * u_uv.zw;
  gl_Position = vec4(px / u_view * 2.0 - 1.0, 0.0, 1.0); gl_Position.y = -gl_Position.y;
}`;
const DENSITY_FS = `#version 300 es
precision highp float;
in vec2 v_uv; uniform sampler2D u_d; uniform sampler2D u_c;
// cubic B-spline filtering from 4 bilinear taps (GPU Gems 2, ch. 20): smooth, no texel blocks when close
vec4 cubic(sampler2D t, vec2 uv) {
  vec2 size = vec2(258.0), px = uv * size - 0.5, f = fract(px); px -= f;
  vec2 f2 = f * f, f3 = f2 * f;
  vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
  vec2 s0 = w0 + w1, s1 = w2 + w3;
  vec2 c0 = (px - 0.5 + w1 / s0) / size, c1 = (px + 1.5 + w3 / s1) / size;
  return mix(mix(texture(t, vec2(c1.x, c1.y)), texture(t, vec2(c0.x, c1.y)), s0.x / (s0.x + s1.x)),
             mix(texture(t, vec2(c1.x, c0.y)), texture(t, vec2(c0.x, c0.y)), s0.x / (s0.x + s1.x)), s0.y / (s0.y + s1.y));
}
uniform float u_area;      // device px per texel, squared
uniform float u_k; uniform float u_gamma; uniform float u_alpha; uniform float u_dark; uniform vec3 u_bg;
out vec4 o;
void main() {
  float n = cubic(u_d, v_uv).r;                         // studies per texel (filtered)
  vec4 c = cubic(u_c, v_uv);
  if (n <= 1e-4 || c.a <= 1e-3) discard;
  vec3 col = c.rgb / c.a;
  float d = n / u_area;                                 // studies per device px
  float b = 1.0 - exp(-pow(d * u_k, u_gamma));
  vec3 rgb;
  if (u_dark > 0.5) {
    rgb = u_bg + col * b * 1.15;
    rgb += vec3(smoothstep(0.62, 1.0, b)) * 0.55;      // dense cores burn white, like a long exposure
  } else {
    vec3 ink = col * (1.0 - 0.35 * smoothstep(0.5, 1.0, b));
    rgb = mix(u_bg, ink, b * 0.92);
  }
  o = vec4(rgb, u_alpha);
}`;

const STAR_VS = `#version 300 es
in vec2 a_xy; in uint a_info; in uint a_sub;
uniform vec4 u_tile;       // world x, y, size of the tile
uniform vec3 u_cam;        // device px: world origin x, y; world size (S * z * dpr)
uniform vec2 u_view; uniform float u_size; uniform float u_dpr;
uniform sampler2D u_pal;
out vec4 v_col; out float v_r;
void main() {
  vec2 w = u_tile.xy + a_xy * u_tile.z;
  vec2 px = u_cam.xy + w * u_cam.z;
  float c = float((a_info >> 12u) & 7u);                // citation class 0..7
  float r = (0.45 + 0.24 * c) * u_size * u_dpr;          // radius, device px: the cited are bigger
  gl_PointSize = r * 2.0 + 2.0;
  v_r = r;
  vec3 col = texelFetch(u_pal, ivec2(int(a_sub), 0), 0).rgb;
  v_col = vec4(col, 0.13 + 0.12 * c);
  gl_Position = vec4(px / u_view * 2.0 - 1.0, 0.0, 1.0); gl_Position.y = -gl_Position.y;
}`;
const STAR_FS = `#version 300 es
precision highp float;
in vec4 v_col; in float v_r; uniform float u_alpha; uniform float u_dark;
out vec4 o;
void main() {
  float d = length(gl_PointCoord - 0.5) * (v_r * 2.0 + 2.0);
  float core = 1.0 - smoothstep(v_r - 0.7, v_r + 0.7, d);
  float a = core * v_col.a * u_alpha;
  if (a <= 0.003) discard;
  vec3 c = u_dark > 0.5 ? mix(v_col.rgb, vec3(1.0), 0.25 * core) : v_col.rgb * 0.75;
  o = vec4(c * a, a);
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
  subCol: number[][] = [];
  palTex: WebGLTexture;
  dprog; sprog; quad: WebGLVertexArrayObject;
  k = 0; kTarget = 0; lastStats = 0;
  onChange: () => void = () => {};
  dLevel = 0; pLevel = 0;
  starsOn = 0;

  constructor(public canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: true, preserveDrawingBuffer: false });
    if (!gl) throw new Error("no WebGL2");
    this.gl = gl;
    this.dprog = shader(gl, DENSITY_VS, DENSITY_FS);
    this.sprog = shader(gl, STAR_VS, STAR_FS);
    this.quad = gl.createVertexArray()!;
    gl.bindVertexArray(this.quad);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.dprog.p, "a");
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.palTex = gl.createTexture()!;
  }

  async load(base: string) {
    this.base = base;
    const r = await fetch(`${base}/map.json`);
    if (!r.ok) throw new Error("no map data");
    this.meta = await r.json();
    for (const [z, list] of Object.entries(this.meta.density.tiles)) for (const [x, y] of list as number[][]) this.dHave.add(`${z}/${x}/${y}`);
    for (const [k, n] of Object.entries(this.meta.points.tiles)) this.pHave.set(k, n as number);
    this.palKey = "";
  }

  setPalette(p: Palette) { this.pal = p; this.palKey = ""; this.onChange(); }

  /** Subfield colours: the field's colour, nudged per subfield so neighbouring specialties read apart. */
  private palette() {
    const key = `${this.pal.dark}:${Object.keys(this.pal.fields).length}`;
    if (key === this.palKey || !this.meta) return;
    this.palKey = key;
    const fields = this.meta.fields.map((f: any) => this.pal.fields[f.name] ?? [0.55, 0.57, 0.6]);
    const seen: Record<number, number> = {};
    this.subCol = this.meta.subfields.map((s: any) => {
      const n = (seen[s.field] = (seen[s.field] ?? -1) + 1);
      const c = fields[s.field] ?? [0.55, 0.57, 0.6];
      const t = ((n * 0.618) % 1) - 0.5;               // golden-ratio spread in [-0.5, 0.5)
      const amt = this.pal.dark ? 0.16 : 0.12;
      return c.map((v: number, i: number) => Math.min(1, Math.max(0, v + t * amt * (i === 1 ? -1 : 1) + (this.pal.dark ? 0 : 0))));
    });
    const gl = this.gl, data = new Uint8Array(256 * 4);
    this.subCol.forEach((c, i) => { data[i * 4] = c[0] * 255; data[i * 4 + 1] = c[1] * 255; data[i * 4 + 2] = c[2] * 255; data[i * 4 + 3] = 255; });
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    for (const t of this.dtiles.values()) t.colFor = "";
  }

  private colourTexture(t: DTile) {
    if (t.colFor === this.palKey) return;
    const px = new Uint8Array(SIDE * SIDE * 4), grey = this.pal.dark ? [0.62, 0.64, 0.68] : [0.42, 0.43, 0.45];
    for (let i = 0; i < SIDE * SIDE; i++) {
      if (!t.D[i]) continue;
      const c = this.subCol[t.sub[i]] ?? grey, p = t.pur[i] / 255;
      const m = this.pal.dark ? 0.3 + 0.7 * p * p : 0.5 + 0.5 * p;   // mixed places are paler
      px[i * 4] = (grey[0] + (c[0] - grey[0]) * m) * 255; px[i * 4 + 1] = (grey[1] + (c[1] - grey[1]) * m) * 255;
      px[i * 4 + 2] = (grey[2] + (c[2] - grey[2]) * m) * 255; px[i * 4 + 3] = 255;
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, t.col);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, SIDE, SIDE, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
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
        for (const t of [tex]) { gl.bindTexture(gl.TEXTURE_2D, t); this.texParams(); }
        const col = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, col); this.texParams();
        const [z, x, y] = key.split("/").map(Number);
        const t: DTile = { key, z, x, y, tex, col, D, sub: new Uint8Array(sub), pur: new Uint8Array(pur), born: performance.now(), colFor: "" };
        this.dtiles.set(key, t);
        this.evict();
        this.onChange();
      } catch { /* a missing tile: its parent keeps showing */ }
      finally { this.loading.delete(key); }
    })();
    this.loading.set(key, p);
  }

  private texParams() {
    const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
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
        const P = this.sprog.p;
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
    if (this.dtiles.size > 220) for (const [k, t] of this.dtiles) {
      if (this.wantD.has(k) || t.z <= 2) continue;
      this.gl.deleteTexture(t.tex); this.gl.deleteTexture(t.col); this.dtiles.delete(k);
      if (this.dtiles.size <= 170) break;
    }
    if (this.ptiles.size > 260) for (const [k, t] of this.ptiles) {
      if (this.wantP.has(k) || t.z <= 1) continue;
      this.gl.deleteVertexArray(t.vao); this.gl.deleteBuffer(t.buf); this.ptiles.delete(k);
      if (this.ptiles.size <= 200) break;
    }
  }

  /** The world rectangle on screen (css px) -> visible world bounds. */
  private bounds(c: Cam, w: number, h: number) {
    const span = c.S * c.z;
    return { x0: -c.ox / span, y0: -c.oy / span, x1: (w - c.ox) / span, y1: (h - c.oy) / span };
  }

  /** Density level: a texel about this many device pixels wide (softer when far in: the stars carry the detail there). */
  densityLevel(c: Cam, dpr: number) {
    const texel = 1.25 * Math.max(1, c.z / 5) ** 0.55;
    return Math.max(0, Math.min(this.meta.density.levels - 1, Math.ceil(Math.log2(c.S * c.z * dpr / (TILE * texel)))));
  }
  /** Point levels drawn: those whose tiles are at least 256 css px wide on screen. */
  pointLevel(c: Cam) {
    return Math.max(0, Math.min(this.meta.points.levels - 1, Math.floor(Math.log2(c.S * c.z / 256))));
  }
  /** Stars per css px^2 at most. A tile draws the first N of its studies (they are sorted most cited
   *  first), N = its share of this budget: sparse places show every study, dense places their classics,
   *  and far enough in, every study everywhere. */
  starDensity = 1 / 150;

  draw(c: Cam, w: number, h: number, alpha = 1) {
    const gl = this.gl, dpr = Math.min(devicePixelRatio || 1, 2);
    const W = Math.round(w * dpr), H = Math.round(h * dpr);
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    gl.viewport(0, 0, W, H);
    const bg = this.pal.bg;
    gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.meta) return;
    this.palette();
    const b = this.bounds(c, w, h), now = performance.now();
    let busy = false;

    // ------------------------------------------------------------- density
    const L = this.densityLevel(c, dpr), n = 1 << L;
    this.dLevel = L;
    this.wantD.clear();
    const tx0 = Math.max(0, Math.floor(b.x0 * n)), tx1 = Math.min(n - 1, Math.floor(b.x1 * n));
    const ty0 = Math.max(0, Math.floor(b.y0 * n)), ty1 = Math.min(n - 1, Math.floor(b.y1 * n));
    const draws: { t: DTile; sx: number; sy: number; ss: number; uv: number[]; a: number }[] = [];
    const missing: [number, string][] = [];
    const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
      const key = `${L}/${tx}/${ty}`;
      if (!this.dHave.has(key)) continue;
      this.wantD.add(key);
      const t = this.dtiles.get(key);
      const wx = tx / n, wy = ty / n, ws = 1 / n;
      // the best loaded ancestor fills in until this tile arrives (and while it fades in)
      const fade = t ? Math.min(1, (now - t.born) / 260) : 0;
      if (fade < 1) {
        for (let l = L - 1; l >= 0; l--) {
          const m = 1 << (L - l), a = this.dtiles.get(`${l}/${tx >> (L - l)}/${ty >> (L - l)}`);
          if (!a) continue;
          const fx = (tx % m) / m, fy = (ty % m) / m;
          draws.push({ t: a, sx: wx, sy: wy, ss: ws, uv: [fx, fy, 1 / m], a: 1 });
          break;
        }
      }
      if (t) draws.push({ t, sx: wx, sy: wy, ss: ws, uv: [0, 0, 1], a: fade }); else missing.push([Math.hypot(wx + ws / 2 - cx, wy + ws / 2 - cy), key]);
      if (fade < 1) busy = true;
    }
    missing.sort((p, q) => p[0] - q[0]);
    for (const [, k] of missing.slice(0, 12)) this.loadDensity(k);
    if (L > 0) for (let l = Math.max(0, L - 3); l < L; l++) {   // keep coarse cover ready for fast moves
      const m = 1 << l;
      for (let ty = Math.max(0, Math.floor(b.y0 * m)); ty <= Math.min(m - 1, Math.floor(b.y1 * m)); ty++)
        for (let tx = Math.max(0, Math.floor(b.x0 * m)); tx <= Math.min(m - 1, Math.floor(b.x1 * m)); tx++) {
          const k = `${l}/${tx}/${ty}`;
          if (this.dHave.has(k)) { this.wantD.add(k); if (!this.dtiles.has(k)) this.loadDensity(k); }
        }
    }
    if (missing.length || this.loading.size) busy = true;

    // exposure: the brightest few percent of what is on screen sit near full brightness
    if (now - this.lastStats > 180) { this.lastStats = now; this.kTarget = this.exposure(c, dpr, b, L) || this.kTarget; }
    if (!this.k) this.k = this.kTarget;
    else if (this.kTarget) {
      const step = Math.exp(Math.log(this.kTarget / this.k) * 0.12);
      this.k *= step;
      if (Math.abs(Math.log(this.kTarget / this.k)) > 0.01) busy = true;
    }

    const P = this.dprog;
    gl.useProgram(P.p);
    gl.bindVertexArray(this.quad);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform2f(P.u.u_view, W, H);
    gl.uniform1f(P.u.u_k, this.k); gl.uniform1f(P.u.u_gamma, this.pal.dark ? 0.66 : 0.72);
    gl.uniform1f(P.u.u_dark, this.pal.dark ? 1 : 0); gl.uniform3f(P.u.u_bg, bg[0], bg[1], bg[2]);
    gl.uniform1i(P.u.u_d, 0); gl.uniform1i(P.u.u_c, 1);
    const span = c.S * c.z * dpr;
    const haze = Math.max(0.45, Math.min(1, 1 - (Math.log2(c.z) - 6) / 3));   // very close: the stars lead
    for (const d of draws) {
      this.colourTexture(d.t);
      const tn = 1 << d.t.z, texelPx = span / (tn * TILE);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, d.t.tex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, d.t.col);
      gl.uniform1f(P.u.u_area, texelPx * texelPx);
      gl.uniform1f(P.u.u_alpha, d.a * alpha * haze);
      gl.uniform4f(P.u.u_rect, c.ox * dpr + d.sx * span, c.oy * dpr + d.sy * span, d.ss * span, d.ss * span);
      // texel centres: content texels 1..256 of 258
      const u0 = (1 + d.uv[0] * TILE) / SIDE, v0 = (1 + d.uv[1] * TILE) / SIDE, us = d.uv[2] * TILE / SIDE;
      gl.uniform4f(P.u.u_uv, u0, v0, us, us);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // ------------------------------------------------------------- stars
    const Lp = this.pointLevel(c);
    this.pLevel = Lp;
    this.wantP.clear();
    const S = this.sprog;
    gl.useProgram(S.p);
    if (this.pal.dark) gl.blendFunc(gl.ONE, gl.ONE); else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform2f(S.u.u_view, W, H);
    gl.uniform3f(S.u.u_cam, c.ox * dpr, c.oy * dpr, span);
    gl.uniform1f(S.u.u_size, Math.min(2.4, 0.8 + Math.max(0, Math.log2(c.z)) * 0.2));
    gl.uniform1f(S.u.u_dpr, dpr);
    gl.uniform1f(S.u.u_dark, this.pal.dark ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.palTex); gl.uniform1i(S.u.u_pal, 0);
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
          if (fade < 1) busy = true;
          const side = c.S * c.z / m;
          t.shown = Math.min(t.n, Math.ceil(this.starDensity * side * side / (Lp + 1)));
          gl.uniform4f(S.u.u_tile, tx / m, ty / m, 1 / m, 0);
          // far out the haze carries the picture; the stars come forward as you zoom in
          const near = Math.min(1, 0.35 + 0.65 * Math.log2(Math.max(1, c.z)) / 3);
          gl.uniform1f(S.u.u_alpha, fade * alpha * near * (this.pal.dark ? 1 : 0.85));
          gl.bindVertexArray(t.vao);
          gl.drawArrays(gl.POINTS, 0, t.shown);
        }
    }
    pm.sort((p, q) => p[0] - q[0]);
    for (const [, k] of pm.slice(0, 10)) this.loadPoints(k);
    if (pm.length) busy = true;
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    if (busy) this.onChange();
  }

  /** Exposure from the density on screen: k such that the 96th percentile of studies per device pixel is bright. */
  private exposure(c: Cam, dpr: number, b: { x0: number; y0: number; x1: number; y1: number }, L: number) {
    const n = 1 << L, span = c.S * c.z * dpr, texelPx = span / (n * TILE), area = texelPx * texelPx;
    const vals: number[] = [];
    for (const t of this.dtiles.values()) {
      if (t.z !== L) continue;
      const wx = t.x / n, wy = t.y / n;
      if (wx > b.x1 || wy > b.y1 || wx + 1 / n < b.x0 || wy + 1 / n < b.y0) continue;
      for (let i = 0; i < 1400; i++) {
        const px = 1 + ((i * 97) % 256), py = 1 + ((i * 61 + (i >> 8) * 13) % 256);
        const x = wx + (px - 1) / (n * TILE), y = wy + (py - 1) / (n * TILE);
        if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
        const d = t.D[py * SIDE + px];
        if (d > 0) vals.push(d / area);
      }
    }
    if (vals.length < 50) return 0;
    vals.sort((p, q) => p - q);
    const hi = vals[Math.floor(vals.length * 0.975)];
    // capped: where studies are few, a lone study stays a faint glow (its star carries it), not a bright patch
    return Math.min(1.3 / hi, 40);
  }

  /** The star under a screen point (css px), among the stars drawn. */
  nearest(c: Cam, sx: number, sy: number, reach = 14): Star | null {
    const span = c.S * c.z;
    let best: Star | null = null, bd = reach * reach;
    for (const t of this.ptiles.values()) {
      if (!this.wantP.has(t.key)) continue;
      const m = 1 << t.z, ox = t.x / m, oy = t.y / m, s = 1 / m;
      // skip tiles far from the point
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

  /** Approximate studies under a screen point (for "how many studies here"). */
  fieldAt(x: number, y: number): number | null {
    const L = this.dLevel, n = 1 << L, t = this.dtiles.get(`${L}/${Math.floor(x * n)}/${Math.floor(y * n)}`);
    if (!t) return null;
    const px = 1 + Math.floor((x * n - t.x) * TILE), py = 1 + Math.floor((y * n - t.y) * TILE);
    return t.D[py * SIDE + px] ? this.meta.subfields[t.sub[py * SIDE + px]]?.field ?? null : null;
  }
}
