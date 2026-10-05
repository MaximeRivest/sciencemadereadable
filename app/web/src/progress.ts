/**
 * The progress panel: where the rewrite is (in line, reading, writing, ready), a bar that moves
 * smoothly, the time left, and one step per part. The amount left is estimated from the original:
 * a rewrite runs about 1.2 times the original's length.
 */
const STEPS: [string, string[]][] = [
  ["Summary", ["title", "abstract", "introduction_first", "conclusion"]],
  ["Introduction", ["introduction_rest"]], ["Methods", ["methods"]], ["Results", ["results"]], ["Discussion", ["discussion"]],
];

export class ProgressPanel {
  private box: HTMLElement;
  private expected: Record<string, number> = {};
  private started = 0;
  private firstText = 0;
  private target = 0;           // the bar's fraction; CSS animates the movement (also in a background tab)
  private eta = 0;
  private timer = 0;
  private label = "";
  private sub = "";
  private steps: Record<string, "wait" | "active" | "done"> = {};

  constructor(box: HTMLElement) {
    this.box = box;
    box.innerHTML = `<div class="pg-line"><span class="pg-label"></span><span class="pg-eta"></span></div>
      <div class="pg-bar"><div class="pg-fill"></div></div>
      <div class="pg-foot"><span class="pg-steps"></span><span class="pg-sub"></span></div>`;
  }

  start(sections: Record<string, string>) {
    this.expected = Object.fromEntries(Object.entries(sections).filter(([, v]) => v).map(([k, v]) => [k, Math.max(200, v.length * 1.2)]));
    this.started = performance.now();
    this.firstText = 0;
    this.target = 0;
    this.eta = 0;
    this.steps = Object.fromEntries(STEPS.filter(([, ks]) => ks.some((k) => this.expected[k])).map(([n]) => [n, "wait"]));
    this.box.hidden = false;
    this.box.classList.remove("ready", "failed");
    this.say("Sending your paper…", "");
    if (!this.timer) this.timer = window.setInterval(this.tick, 250);
  }

  queued(ahead: number, online: boolean) {
    this.say(online ? (ahead ? `Waiting in line · ${ahead} paper${ahead === 1 ? "" : "s"} ahead of yours` : "Next in line…")
                    : "Waiting for our GPU to come back online…", "");
    this.started = performance.now();
  }

  /** What has been written so far. */
  update(parts: Record<string, string>, done: Iterable<string>, sharing = 0) {
    const fin = new Set(done);
    let written = 0, total = 0;
    for (const [k, e] of Object.entries(this.expected)) {
      total += e;
      written += fin.has(k) ? e : Math.min(e * 0.97, (parts[k] ?? "").length);
    }
    const anyText = Object.values(parts).some((t) => t && t.trim());
    const now = performance.now();
    if (anyText && !this.firstText) this.firstText = now;
    this.target = anyText ? Math.max(0.06, Math.min(0.97, written / total)) : this.target;
    if (anyText && now - this.firstText > 3000) {
      const rate = written / ((now - this.firstText) / 1000);
      const left = Math.max(0, (total - written) / Math.max(rate, 1));
      this.eta = this.eta ? this.eta * 0.7 + left * 0.3 : left;
    }
    for (const [name, ks] of STEPS) {
      if (!(name in this.steps)) continue;
      const present = ks.filter((k) => this.expected[k]);
      this.steps[name] = present.every((k) => fin.has(k)) ? "done" : present.some((k) => (parts[k] ?? "").trim()) ? "active" : "wait";
    }
    const openingDone = this.steps.Summary === "done";
    const active = Object.values(this.steps).filter((s) => s === "active").length;
    const left = Object.values(this.steps).filter((s) => s !== "done").length;
    const label = !anyText ? "Reading the paper…" : !openingDone ? "Writing the summary…"
      : left <= 1 ? "Writing the last section…" : active > 1 ? `Writing ${active} sections at once…` : "Writing the sections…";
    this.say(label, sharing > 0 ? `sharing the GPU with ${sharing} other reader${sharing === 1 ? "" : "s"}` : "");
  }

  finish(seconds: number) {
    this.target = 1;
    this.eta = 0;
    for (const k of Object.keys(this.steps)) this.steps[k] = "done";
    this.box.classList.add("ready");
    this.say(`Ready ✓ · written in ${Math.round(seconds)} s`, "");
    this.paint();
    this.stop();
    setTimeout(() => { if (this.box.classList.contains("ready")) this.box.hidden = true; }, 6000);
  }

  fail(message: string) {
    this.stop();
    this.box.classList.add("failed");
    this.say(message, "");
  }

  hide() { this.box.hidden = true; this.stop(); }

  private stop() { if (this.timer) { clearInterval(this.timer); this.timer = 0; } }

  private say(label: string, sub: string) {
    this.label = label;
    this.sub = sub;
    this.paint();
  }

  private paint() {
    const q = (s: string) => this.box.querySelector(s) as HTMLElement;
    q(".pg-label").textContent = this.label;
    q(".pg-sub").textContent = this.sub;
    const reading = !this.firstText && !this.box.classList.contains("ready");
    q(".pg-eta").textContent = this.eta > 0 && this.eta <= 20 && !this.box.classList.contains("ready") ? "almost done"
      : this.eta > 20 ? `about ${this.eta > 90 ? `${Math.round(this.eta / 60)} min` : `${Math.round(this.eta / 10) * 10} s`} left`
      : !this.box.classList.contains("ready") ? `${Math.round((performance.now() - this.started) / 1000)} s` : "";
    q(".pg-fill").style.width = `${(this.target * 100).toFixed(1)}%`;
    this.box.classList.toggle("reading", reading);
    q(".pg-steps").innerHTML = Object.entries(this.steps).map(([n, s]) => `<span class="pg-step ${s}">${n}</span>`).join("");
  }

  private tick = () => {
    // while the paper is being read (no text yet), creep toward 6% so it never looks stuck
    if (!this.firstText && !this.box.classList.contains("ready")) {
      const t = (performance.now() - this.started) / 1000;
      this.target = Math.max(this.target, 0.06 * (1 - Math.exp(-t / 8)));
    }
    this.paint();
  };
}
