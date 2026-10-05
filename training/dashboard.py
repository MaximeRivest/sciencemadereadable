"""Training dashboard: a web page with the progress of every training run.

    python3 training/dashboard.py            # serves http://0.0.0.0:8790

From the laptop (Tailscale): https://lambda.tail69222b.ts.net:18790
Reads runs/*/config.json, status.json and log.jsonl; never writes anything.
Standard library only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
SERVICES = ["podman-inktype-vllm", "chattering-semantic", "kokoro-tts", "parakeet-server"]


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_data(folder: Path) -> dict:
    train, validation, messages = [], [], []
    try:
        lines = (folder / "log.jsonl").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        kind = e.get("kind")
        if kind in ("train", "validation"):          # a restart repeats steps: forget the abandoned attempt
            if kind == "train":
                train[:] = [t for t in train if t["step"] < e["step"]]
                validation[:] = [v for v in validation if v["step"] < e["step"]]
            else:
                validation[:] = [v for v in validation if v["step"] < e["step"]]
        if kind == "train":
            train.append({k: e.get(k) for k in ("step", "loss", "loss_opening", "loss_section", "lr", "grad_norm",
                                                 "tokens_per_second", "seen_answer_tokens", "max_memory_gb", "at")})
        elif kind == "validation":
            validation.append(e)
        elif kind == "message":
            messages.append(e)
    status = read_json(folder / "status.json") or {}
    if status.get("state") == "training" or status.get("state") in ("validating", "loading"):
        age = time.time() - (folder / "status.json").stat().st_mtime
        status["seconds_since_update"] = round(age)
    bench = {f.stem: read_json(f) for f in sorted((folder / "benchmark").glob("*.json"))}
    return {"name": folder.name, "config": read_json(folder / "config.json") or {}, "status": status, "benchmark": bench,
            "train": train, "validation": validation, "messages": messages[-15:]}


def gpus() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                              "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        i, util, used, total, temp, power = [x.strip() for x in line.split(",")]
        rows.append({"index": int(i), "util": float(util), "used_gb": float(used) / 1024,
                     "total_gb": float(total) / 1024, "temp": float(temp), "power": float(power)})
    return rows


def services() -> dict:
    out = {}
    for s in SERVICES:
        try:
            out[s] = subprocess.run(["systemctl", "is-active", s], capture_output=True, text=True,
                                    timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            out[s] = "?"
    return out


def api() -> dict:
    runs = [run_data(f) for f in sorted(RUNS.glob("*")) if f.is_dir() and not f.name.startswith("smoke")]
    return {"now": time.time(), "runs": runs, "references": read_json(RUNS / "references.json") or {}, "plan": read_json(RUNS / "plan.json") or [],
            "gpus": gpus(), "services": services()}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api"):
            body, kind = json.dumps(api()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training · Scholar's Reading List</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root { --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e4e2dc; --accent:#2f6fde; }
  @media (prefers-color-scheme: dark) { :root { --bg:#151514; --fg:#ecebe6; --muted:#9a9890; --card:#1f1f1d; --line:#33322f; } }
  body { margin:0; font:15px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  main { max-width:1200px; margin:0 auto; padding:20px; }
  h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:0 0 10px; }
  .muted { color:var(--muted); } .grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; margin:8px 0; }
  .bar > div { height:100%; background:var(--accent); }
  .kv { display:grid; grid-template-columns:auto 1fr; gap:2px 12px; font-size:14px; }
  .kv span:nth-child(odd) { color:var(--muted); }
  .state { font-weight:600; } .warn { color:#c0392b; font-weight:600; }
  .chart { position:relative; height:300px; }
  .wide { grid-column:1/-1; } .note { font-size:13px; color:var(--muted); margin-top:6px; }
  table { border-collapse:collapse; font-size:14px; width:100%; } td,th { padding:3px 8px; text-align:left; border-bottom:1px solid var(--line); }
  .chip { display:inline-flex; align-items:center; gap:6px; margin:3px 6px 3px 0; padding:3px 9px; border:1px solid var(--line); border-radius:14px; font-size:13px; cursor:pointer; user-select:none; }
  .chip.off { opacity:.4; text-decoration:line-through; }
  .chip input { display:none; }
  .fbtn { font-size:12px; margin-right:6px; padding:2px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); cursor:pointer; }
  .log { font:12px ui-monospace, monospace; max-height:160px; overflow:auto; white-space:pre-wrap; }
</style></head><body><main>
<h1>Training the student model</h1>
<div class="muted" id="updated">loading…</div>
<div class="card" id="filters" style="margin-top:14px"></div>
<div class="grid" id="cards" style="margin-top:14px"></div>
<div class="grid" style="margin-top:14px">
  <div class="card wide"><h2>Benchmark v0.3 · Opus judge, 0–10 · 3 test papers (15 parts)</h2>
    <div id="bench"></div>
    <div class="note">Faithful: every fact kept, nothing wrong added. Structure: headings, order, authors' voice. Numbers: share of the original's numbers found in the rewrite (automatic). Opus is the teacher; "Qwen 4B, untrained" is where the 4B started (with the full teacher prompt). Each checkpoint is rewritten on the CPU (llama.cpp, 8-bit), so a 4B checkpoint takes about an hour.</div></div>
  <div class="card wide"><h2>Validation loss (lower is better) · held-out papers</h2>
    <div class="chart"><canvas id="val"></canvas></div>
    <div class="note">Measured on 195 conversations from 40 papers no model trains on. X axis: how many answer words-pieces (tokens) the model has learned from, so runs of different sizes line up. Dotted lines: the opening step only.</div></div>
  <div class="card wide"><h2>Training loss · smoothed over 20 steps</h2>
    <div class="chart"><canvas id="train"></canvas></div>
    <div class="note">Noisy by nature (each step sees different papers); the validation loss above is the one to trust.</div></div>
  <div class="card"><h2>Model size curve</h2><div class="chart"><canvas id="size"></canvas></div>
    <div class="note">Best validation loss of each run against its size (billions of parameters).</div></div>
  <div class="card"><h2>GPUs and paused services</h2><div id="gpus"></div></div>
</div>
<script>
const COLORS = ["#2f6fde","#e67e22","#27ae60","#8e44ad","#c0392b","#16a085"];
const charts = {};
const fmtH = h => h == null ? "–" : h < 1 ? Math.round(h*60) + " min" : h.toFixed(1) + " h";
const fmtM = n => n >= 1e9 ? (n/1e9).toFixed(1)+"B" : n >= 1e6 ? (n/1e6).toFixed(0)+"M" : n;
function smooth(xs, k) { const out=[]; let s=0; const q=[]; for (const x of xs) { q.push(x); s+=x; if (q.length>k) s-=q.shift(); out.push(s/q.length); } return out; }
function chart(id, type, datasets, xTitle, yTitle, extra={}) {
  const opts = { animation:false, maintainAspectRatio:false, parsing:false, interaction:{mode:"nearest", intersect:false},
    scales:{ x:{type:"linear", title:{display:true,text:xTitle}}, y:{title:{display:true,text:yTitle}} },
    plugins:{ legend:{position:"bottom"} }, ...extra };
  if (charts[id]) { charts[id].data.datasets = datasets; charts[id].update(); return; }
  charts[id] = new Chart(document.getElementById(id), { type, data:{datasets}, options:opts });
}
// What is shown, saved in this browser (survives the refresh and reloading the page).
const F = Object.assign({hidden: {}, opening: true, refs: true, cards: true},
                        JSON.parse(localStorage.getItem("srl-dashboard-filters") || "{}"));
const saveF = () => localStorage.setItem("srl-dashboard-filters", JSON.stringify(F));
const shown = name => !F.hidden[name];
let last = null;

function renderFilters(d, plan) {
  const names = [...new Set([...plan.map(p => p.name), ...d.runs.map(r => r.name)])];
  const label = n => (plan.find(p => p.name === n) || {}).model || n;
  const state = n => ((d.runs.find(r => r.name === n) || {}).status || {}).state || "waiting";
  const chips = names.map(n => {
    const k = plan.findIndex(p => p.name === n);
    return `<label class="chip ${shown(n) ? "" : "off"}"><input type="checkbox" data-run="${n}" ${shown(n) ? "checked" : ""}>
      <span style="color:${COLORS[(k < 0 ? 0 : k) % COLORS.length]}">●</span>${label(n)} <span class="muted">· ${String(state(n)).split(" ")[0]}</span></label>`;
  }).join("");
  const opt = (key, text) => `<label class="chip ${F[key] ? "" : "off"}"><input type="checkbox" data-opt="${key}" ${F[key] ? "checked" : ""}>${text}</label>`;
  document.getElementById("filters").innerHTML = `<h2>Show</h2><div>${chips}</div>
    <div style="margin-top:6px">${opt("cards", "run cards")}${opt("opening", "opening-step curves (dotted)")}${opt("refs", "reference rows in the benchmark")}</div>
    <div style="margin-top:8px"><button class="fbtn" data-act="all">show all runs</button><button class="fbtn" data-act="active">only active runs</button><button class="fbtn" data-act="none">hide all runs</button></div>`;
  document.querySelectorAll("#filters [data-run]").forEach(el => el.onchange = () => { F.hidden[el.dataset.run] = !el.checked; saveF(); draw(last); });
  document.querySelectorAll("#filters [data-opt]").forEach(el => el.onchange = () => { F[el.dataset.opt] = el.checked; saveF(); draw(last); });
  document.querySelectorAll("#filters [data-act]").forEach(el => el.onclick = () => {
    names.forEach(n => {
      const st = String(state(n));
      F.hidden[n] = el.dataset.act === "none" || (el.dataset.act === "active" && !/^(training|validating|loading)/.test(st));
    });
    saveF(); draw(last);
  });
}

async function refresh() {
  let d; try { d = await (await fetch("api")).json(); } catch (e) { document.getElementById("updated").textContent = "cannot reach lambda: " + e; return; }
  document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString() + " · refreshes every 30 s";
  last = d; draw(d);
}

function draw(d) {
  if (!d) return;
  const cards = [];
  const runsByName = Object.fromEntries(d.runs.map(r => [r.name, r]));
  const plan = d.plan.length ? d.plan : d.runs.map(r => ({name:r.name, model:r.config.model, gpu:r.config.cuda_visible_devices}));
  renderFilters(d, plan);
  plan.forEach((p, k) => {
    if (!shown(p.name) || !F.cards) return;
    const r = runsByName[p.name]; const s = r ? r.status : {}; const c = r ? r.config : {};
    const pct = s.total_steps ? 100*s.step/s.total_steps : 0;
    const lastVal = r && r.validation.length ? r.validation[r.validation.length-1] : null;
    const firstVal = r && r.validation.length ? r.validation[0] : null;
    let state = r ? (s.state || "starting") : "waiting its turn";
    let stale = s.seconds_since_update > 900 ? `<div class="warn">no news for ${Math.round(s.seconds_since_update/60)} min: stopped or stuck?</div>` : "";
    cards.push(`<div class="card"><h2><span style="color:${COLORS[k % COLORS.length]}">●</span> ${p.model} <span class="muted">· GPU ${p.gpu}</span></h2>
      <div class="state">${state}</div>${stale}
      <div class="bar"><div style="width:${pct.toFixed(1)}%"></div></div>
      <div class="kv">
        <span>step</span><span>${s.step ?? 0} / ${s.total_steps ?? "–"} (${pct.toFixed(1)}%)</span>
        <span>time left</span><span>${s.state === "finished" ? "done" : (s.state === "training" || s.state === "validating") ? fmtH(s.eta_hours) : "–"}</span>
        <span>trained for</span><span>${fmtH(s.train_hours)}</span>
        <span>speed</span><span>${s.seconds_per_step ? s.seconds_per_step + " s per step" : "–"}</span>
        <span>validation loss</span><span>${lastVal ? lastVal.all.toFixed(4) + (firstVal && firstVal !== lastVal ? " (started at " + firstVal.all.toFixed(4) + ")" : "") : "–"}</span>
        <span>GPU memory peak</span><span>${s.max_memory_gb ? s.max_memory_gb + " GB" : "–"}</span>
        <span>model</span><span>${c.parameters ? fmtM(c.parameters) + " parameters, " + fmtM(c.trainable_parameters) + (c.method === "full" ? " trained (all weights)" : " trained (LoRA)") : "–"}</span>
      </div>
      ${r && r.messages.length ? `<div class="log">${r.messages.slice(-6).map(m => m.at.slice(11,16) + "  " + m.text).join("\n")}</div>` : ""}
    </div>`);
  });
  document.getElementById("cards").innerHTML = cards.join("");
  document.getElementById("cards").style.display = F.cards ? "" : "none";

  const val = [], train = [], size = [];
  d.runs.forEach((r, i) => {
    if (!shown(r.name)) return;
    const k = plan.findIndex(p => p.name === r.name);
    const color = COLORS[(k < 0 ? i : k) % COLORS.length], label = (k >= 0 ? plan[k].model : r.config.model) || r.name;
    val.push({label, borderColor:color, backgroundColor:color, data:r.validation.map(v => ({x:v.seen_answer_tokens/1e6, y:v.all})), pointRadius:3});
    if (F.opening && r.validation.some(v => v.opening != null))
      val.push({label: label + " · opening", borderColor:color, borderDash:[4,4], pointRadius:0, data:r.validation.filter(v => v.opening != null).map(v => ({x:v.seen_answer_tokens/1e6, y:v.opening}))});
    const ys = smooth(r.train.map(t => t.loss), 20);
    train.push({label, borderColor:color, pointRadius:0, borderWidth:1.5, data:r.train.map((t, k) => ({x:t.seen_answer_tokens/1e6, y:ys[k]}))});
    const trained = r.validation.filter(v => v.step > 0);
    if (trained.length && r.config.parameters)
      size.push({label, backgroundColor:color, pointRadius:6, data:[{x:r.config.parameters/1e9, y:Math.min(...trained.map(v => v.all))}]});
  });
  chart("val", "line", val, "answer tokens learned (millions)", "loss");
  chart("train", "line", train, "answer tokens learned (millions)", "loss");
  chart("size", "scatter", size, "parameters (billions)", "best validation loss");

  const rows = []; let refs = null;
  d.runs.forEach(r => Object.entries(r.benchmark || {}).forEach(([step, b]) => {
    if (!b) return; refs = refs || b.candidates;
    if (!shown(r.name)) return;
    const name = Object.keys(b.candidates)[0], c = b.candidates[name];
    const k = plan.findIndex(p => p.name === r.name);
    const extra = step.replace(/^step-\d+-?/, "");
    rows.push({label: (k >= 0 ? plan[k].model : r.name) + " · step " + parseInt(step.replace("step-", "")) + (extra ? " + " + extra : ""), c, order: (r.config.parameters || 0) * 1e6 + parseInt(step.replace("step-", "")) || 0});
  }));
  rows.sort((a, b) => a.order - b.order);
  if (refs && F.refs) {
    const extra = d.references || {};
    rows.unshift({label: "Opus · v8 prompts (current, the teacher)", c: refs.opus, ref: true});
    ["Astra · v8 prompts (current)", "Luna · v8 prompts (current)", "Opus · v1 prompts", "Astra · v1 prompts", "Luna · v1 prompts"]
      .forEach((k, i) => extra[k] && rows.splice(1 + i, 0, {label: k, c: extra[k], ref: true}));
    rows.push({label: "Qwen 4B, untrained", c: refs.qwen4b, ref: true});
    if (extra["Qwen 0.8B, untrained"]) rows.push({label: "Qwen 0.8B, untrained (" + extra["Qwen 0.8B, untrained"].note + ")", c: extra["Qwen 0.8B, untrained"], ref: true});
  }
  const f = v => v == null ? "–" : v.toFixed(1);
  document.getElementById("bench").innerHTML = rows.length ? `<table><tr><th></th><th>faithful</th><th>understandable</th><th>pleasant</th><th>structure</th><th>mean</th><th>numbers kept</th><th>serious issues</th></tr>` +
    rows.map(r => `<tr style="${r.ref ? "color:var(--muted)" : ""}"><td>${r.label}</td><td>${f(r.c.faithful_and_exact)}</td><td>${f(r.c.understandable)}</td><td>${f(r.c.pleasant_to_read)}</td><td>${f(r.c.structure_and_voice)}</td><td><b>${f(r.c.mean)}</b></td><td>${r.c.mean == null ? "–" : Math.round(100 * r.c.numbers_kept) + "%"}</td><td>${r.c.major_issues ?? "–"}</td></tr>`).join("") + "</table>" : "no checkpoint scored yet";

  const g = d.gpus.map(g => `<tr><td>GPU ${g.index}</td><td>${g.util}% busy</td><td>${g.used_gb.toFixed(1)} / ${g.total_gb.toFixed(0)} GB</td><td>${g.temp}°C</td><td>${Math.round(g.power)} W</td></tr>`).join("");
  const sv = Object.entries(d.services).map(([k, v]) => `<tr><td>${k}</td><td>${v === "active" ? "running" : (v === "inactive" || v === "failed") ? "paused" : v}</td></tr>`).join("");
  document.getElementById("gpus").innerHTML = `<table>${g}</table><h2 style="margin-top:14px">Services that need the GPUs</h2><table>${sv}</table>
    <div class="note">Bring them back when training is over: training/gpu_services.sh start</div>`;
}
refresh(); setInterval(refresh, 30000);
</script></main></body></html>
"""


if __name__ == "__main__":
    print(f"training dashboard on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
