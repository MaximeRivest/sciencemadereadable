// Bundle the web app: node tools/build.mjs [--watch]
// web/app.js is what every page loads; code only some readers need (rewriting with their own Claude or OpenAI key:
// lm15, functai, about 0.8 MB) is split into web/chunks/ and loaded when first used.
import * as esbuild from "esbuild";
import { rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
const shim = fileURLToPath(new URL("./lm15_browser_shim.js", import.meta.url));
const lm15Browser = {
  name: "lm15-browser",
  setup(b) { b.onResolve({ filter: /^@lm15\/lm15$/ }, () => ({ path: shim })); },
};
rmSync(new URL("../web/chunks", import.meta.url), { recursive: true, force: true });   // no stale chunks
const ctx = await esbuild.context({
  entryPoints: { app: "web/src/app.ts" }, bundle: true, format: "esm", platform: "browser", target: "es2022",
  outdir: "web", splitting: true, chunkNames: "chunks/[name]-[hash]", conditions: ["functai-source"], sourcemap: true, minify: !process.env.NOMINIFY,
  plugins: [lm15Browser], logLevel: "warning", nodePaths: ["node_modules"],
});
if (process.argv.includes("--watch")) await ctx.watch(); else { await ctx.rebuild(); await ctx.dispose(); }
