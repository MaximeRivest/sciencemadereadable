// Bundle the web app: node tools/build.mjs [--watch]
import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
const shim = fileURLToPath(new URL("./lm15_browser_shim.js", import.meta.url));
const lm15Browser = {
  name: "lm15-browser",
  setup(b) { b.onResolve({ filter: /^@lm15\/lm15$/ }, () => ({ path: shim })); },
};
const ctx = await esbuild.context({
  entryPoints: ["web/src/app.ts"], bundle: true, format: "esm", platform: "browser", target: "es2022",
  outfile: "web/app.js", conditions: ["functai-source"], sourcemap: true, minify: true,
  plugins: [lm15Browser], logLevel: "warning", nodePaths: ["node_modules"],
});
if (process.argv.includes("--watch")) await ctx.watch(); else { await ctx.rebuild(); await ctx.dispose(); }
