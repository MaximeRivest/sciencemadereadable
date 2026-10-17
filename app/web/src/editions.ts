/** Private reading editions: capabilities stay in this browser, never in URLs. */
import { API } from "./jobs.ts";

interface LocalJob {
  id: string;
  token: string;
  title: string;
  expires: number;
}
interface View {
  id: string;
  kind: string;
  state: string;
  stage: string;
  error: string;
  ahead: number;
  expires: number;
  retries: number;
  audio: boolean;
}
const KEY = "smr-private-reading-v1";
const esc = (s: unknown) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
let generation = 0;
let memory: LocalJob[] = [];
function load(): LocalJob[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]")
      .filter(
        (x: any) =>
          /^[a-f0-9]{32}$/.test(x.id) &&
          typeof x.token === "string" &&
          x.token.length < 100,
      )
      .slice(0, 30);
  } catch {
    return memory;
  }
}
function remember(jobs: LocalJob[]) {
  memory = jobs;
  try {
    localStorage.setItem(KEY, JSON.stringify(jobs));
  } catch {
    /* current tab keeps access; offer the recovery file */
  }
}
function add(job: LocalJob) {
  remember([job, ...load().filter((j) => j.id !== job.id)].slice(0, 30));
}
async function request(path: string, job?: LocalJob, init: RequestInit = {}) {
  const response = await fetch(API + "/api/editions" + path, {
    ...init,
    credentials: "include",
    headers: {
      ...(job ? { Authorization: `Bearer ${job.token}` } : {}),
      ...init.headers,
    },
    signal: init.signal ?? AbortSignal.timeout(30000),
  });
  let data: any;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "The reading service did not return a usable answer. Please try again.",
    );
  }
  if (!response.ok)
    throw new Error(data.error ?? `Reading service: HTTP ${response.status}`);
  return data;
}
function download(name: string, data: string, type: string) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
const ROLES = [
  "title",
  "heading",
  "paragraph",
  "table",
  "caption",
  "equation",
  "metadata",
  "references",
  "furniture",
  "other",
  "uncertain",
];
const SECTIONS = [
  "title",
  "abstract",
  "introduction",
  "methods",
  "results",
  "discussion",
  "conclusion",
  "results_discussion",
  "other",
  "uncertain",
];
const labels: Record<string, string> = {
  uploading: "Uploading",
  queued: "In the queue",
  extracting: "Reading the original",
  classifying: "Identifying sections",
  review: "Check the sections",
  waiting_model: "Waiting for the model",
  rewriting: "Rewriting",
  exporting: "Preparing reading formats",
  audio: "Making the audio",
  done: "Ready",
  failed: "Needs attention",
  cancelled: "Cancelled",
};

export function mount(root: HTMLElement, doi = "", model = "our-9b") {
  const gen = ++generation;
  const alive = () =>
    gen === generation && document.body.dataset.view === "reading";
  root.innerHTML = `<header class="edition-heading"><p class="edition-kicker">YOUR READING DESK</p><h1>Read it your way.</h1>
    <p>Plain-language research, on screen, on an e-paper tablet, or in your headphones.</p></header>
    <div class="edition-layout"><section class="edition-card" aria-labelledby="edition-new-title"><h2 id="edition-new-title">${doi ? "Make editions of this saved paper" : "Start with a research PDF"}</h2>
      ${
        doi
          ? `<p class="edition-doi">${esc(doi)}</p><p>Uses the selected rewrite already saved on our server, without rewriting it again.</p><p class="quiet small">Requires a matching archived source PDF within the current 20-page / 25 MiB limit.</p>`
          : `<label class="edition-drop" for="edition-file"><strong>Choose a PDF</strong><span>English research papers · up to 200 pages and 25 MiB · we rewrite the main text (up to about 25,000 words), not references or appendices</span><input id="edition-file" type="file" accept="application/pdf,.pdf"></label>
         <p>You’ll check the extracted sections before we rewrite it. Unclear sections are flagged, not silently guessed.</p>
         <label class="edition-check"><input id="edition-rights" type="checkbox"> I have permission to upload and process this paper.</label>
         <label class="edition-check"><input id="edition-consent" type="checkbox"> I agree to send its extracted text to TypeSafe for section labelling. Don’t upload confidential material without permission.</label>`
      }
      <label class="edition-check"><input id="edition-audio" type="checkbox" checked> Include Kokoro audio and chapter downloads</label>
      <p class="quiet small">Web reader, EPUB and PDF are included. Original figures are kept; uncertain crop boundaries use clearly labelled source-page panels.</p>
      <button id="edition-submit" class="primary">${doi ? "Prepare reading editions" : "Upload and check sections"}</button>
      <p id="edition-message" class="edition-message" role="status"></p>
    </section><aside class="edition-card edition-privacy"><h2>Private by default</h2><ul><li>Uploads are not added to the public library.</li><li>Files expire after <strong>48 hours</strong>. Download what you want to keep.</li><li>You can delete your files sooner.</li><li>Access keys stay in this browser. Save a recovery key to open a job on another device.</li></ul>
      <p class="small quiet">Rewriting and OCR can make mistakes. Keep the original nearby, especially for numbers and formulas. Current limit: two unfinished jobs and eight requests per day per connection.</p>
      <label class="edition-restore">Restore a private access key<input id="edition-restore" type="file" accept="application/json,.json"></label>
    </aside></div><section class="edition-jobs" aria-labelledby="edition-jobs-title"><h2 id="edition-jobs-title">Your recent jobs</h2><div id="edition-jobs"></div></section>
    <section id="edition-review" class="edition-review" hidden></section>
    <dialog id="edition-page-dialog" aria-label="Original PDF page"><button class="edition-close" type="button">Close source page</button><img alt="Original PDF page"/></dialog>`;
  const $ = <T extends HTMLElement = HTMLElement>(s: string) =>
    root.querySelector(s) as T;
  const message = (text: string, error = false) => {
    if (!alive()) return;
    $("#edition-message").textContent = text;
    $("#edition-message").classList.toggle("error", error);
  };
  const jobsRoot = $("#edition-jobs");
  const views = new Map<string, View>();
  let submitting = false,
    lastHTML = "",
    pageURL = "";
  const pageDialog = $("#edition-page-dialog") as HTMLDialogElement;
  pageDialog.querySelector("button")!.onclick = () => pageDialog.close();
  pageDialog.addEventListener("close", () => {
    if (pageURL) URL.revokeObjectURL(pageURL);
    pageURL = "";
  });

  async function showPage(job: LocalJob, page: number) {
    try {
      const r = await fetch(
        `${API}/api/editions/jobs/${job.id}/pages/${page}`,
        {
          headers: { Authorization: `Bearer ${job.token}` },
          credentials: "include",
        },
      );
      if (!r.ok) throw new Error("The original page could not be loaded.");
      if (pageURL) URL.revokeObjectURL(pageURL);
      const blob = await r.blob();
      if (!alive()) return;
      pageURL = URL.createObjectURL(blob);
      const img = pageDialog.querySelector("img")!;
      img.src = pageURL;
      img.alt = `Original PDF page ${page}`;
      if (!pageDialog.open) pageDialog.showModal();
    } catch (e: any) {
      message(e.message, true);
    }
  }

  async function review(job: LocalJob) {
    const box = $("#edition-review");
    box.hidden = false;
    box.innerHTML = '<p role="status">Loading extracted sections…</p>';
    try {
      const data = await request(`/jobs/${job.id}/review`, job);
      if (!alive()) return;
      const flagged = new Set(data.review.map((r: any) => r.block));
      box.innerHTML = `<h2 tabindex="-1">Check the extracted sections</h2><p>Compare uncertain blocks with their source page. A label identifies where a passage belongs; it does not change its text. References and publication details remain available separately.</p>
        <p class="edition-review-summary">${data.ledger.length} blocks · ${flagged.size} need explicit checking${data.missing_required.length ? " · Missing: " + esc(data.missing_required.join(", ")) : ""}</p>
        <div class="edition-section-summary">${Object.entries(data.sections)
          .map(
            ([k, v]) =>
              `<span><strong>${esc(k.replaceAll("_", " "))}</strong> ${String(v).split(/\s+/).filter(Boolean).length} words</span>`,
          )
          .join("")}</div>
        <div class="edition-blocks"></div><label class="edition-check"><input id="edition-confirm-all" type="checkbox"> I checked the section mapping and want to continue with this extracted text.</label>
        <button id="edition-confirm" class="primary">Confirm sections and rewrite</button><p id="edition-review-message" role="status"></p>`;
      const blocks = box.querySelector(".edition-blocks")!;
      for (const block of data.ledger) {
        const row = document.createElement("details");
        row.className = "edition-block";
        row.dataset.id = block.id;
        if (flagged.has(block.id)) row.open = true;
        row.innerHTML = `<summary>${esc(block.id)} · page ${block.page} · ${esc(block.role)} → ${esc(block.destination || "kept separately")}${flagged.has(block.id) ? " · CHECK" : ""}</summary>
          <button class="source-page" type="button">View original page ${block.page}</button><pre></pre><div class="edition-labels">
          <label>Block type<select class="block-role">${ROLES.map((v) => `<option value="${v}" ${v === block.role ? "selected" : ""}>${v}</option>`).join("")}</select></label>
          <label>Section<select class="block-section">${SECTIONS.map((v) => `<option value="${v}" ${v === block.section ? "selected" : ""}>${v.replaceAll("_", " ")}</option>`).join("")}</select></label>
          <label>Paragraph across pages<select class="block-boundary"><option value="separate">Separate paragraph</option><option value="join" ${block.boundary === "join" ? "selected" : ""}>Continues previous paragraph</option></select></label></div>
          <label class="edition-check"><input class="block-reviewed" type="checkbox"> I checked this block’s labels against the source.</label>`;
        row.querySelector("pre")!.textContent = block.text;
        (row.querySelector(".source-page") as HTMLButtonElement).onclick = () =>
          showPage(job, block.page);
        for (const select of row.querySelectorAll("select"))
          select.onchange = () => {
            (row.querySelector(".block-reviewed") as HTMLInputElement).checked =
              true;
          };
        blocks.append(row);
      }
      (box.querySelector("h2") as HTMLElement).focus({ preventScroll: true });
      box.scrollIntoView({ behavior: "instant" });
      (box.querySelector("#edition-confirm") as HTMLButtonElement).onclick =
        async () => {
          const msg = box.querySelector("#edition-review-message")!;
          if (
            !(box.querySelector("#edition-confirm-all") as HTMLInputElement)
              .checked
          ) {
            msg.textContent = "Confirm that you checked the mapping first.";
            return;
          }
          const changes: Record<string, any> = {};
          for (const row of box.querySelectorAll<HTMLElement>(
            ".edition-block",
          )) {
            const checked = (
              row.querySelector(".block-reviewed") as HTMLInputElement
            ).checked;
            if (flagged.has(row.dataset.id!) && !checked) {
              msg.textContent = `Check ${row.dataset.id} before continuing.`;
              (row as HTMLDetailsElement).open = true;
              row.scrollIntoView();
              return;
            }
            if (checked)
              changes[row.dataset.id!] = {
                role: (row.querySelector(".block-role") as HTMLSelectElement)
                  .value,
                section: (
                  row.querySelector(".block-section") as HTMLSelectElement
                ).value,
                boundary: (
                  row.querySelector(".block-boundary") as HTMLSelectElement
                ).value,
              };
          }
          const button = box.querySelector(
            "#edition-confirm",
          ) as HTMLButtonElement;
          button.disabled = true;
          try {
            await request(`/jobs/${job.id}/confirm`, job, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ confirmed: true, changes }),
            });
            box.hidden = true;
            message(
              "Sections confirmed. Your rewrite is in the private queue.",
            );
            await poll();
          } catch (e: any) {
            msg.textContent = e.message;
          } finally {
            button.disabled = false;
          }
        };
    } catch (e: any) {
      box.textContent = e.message;
    }
  }

  function draw() {
    const jobs = load();
    const content = jobs.length
      ? jobs
          .map((j) => {
            const v = views.get(j.id);
            return `<article class="edition-job" data-job="${j.id}"><div><h3>${esc(j.title)}</h3><p class="edition-job-state">${esc(v ? (labels[v.state] ?? v.state) : "Checking status…")}</p><p>${esc(v?.error || v?.stage || "")}${v?.state === "queued" && v.ahead ? ` · ${v.ahead} ahead` : ""}</p><p class="quiet small">Expires ${esc(new Date(j.expires * 1000).toLocaleString())}</p></div><div class="edition-job-actions">
      ${v?.state === "done" ? '<button data-action="open" class="primary">Open reader & downloads</button>' : ""}${v?.state === "review" ? '<button data-action="review" class="primary">Check sections</button>' : ""}
      ${v && (v.state === "failed" || (v.state === "done" && v.error)) && v.retries < 2 ? '<button data-action="retry">' + (v.state === "done" ? "Retry audio" : "Retry") + "</button>" : ""}${v && !["done", "failed", "cancelled", "unavailable"].includes(v.state) ? '<button data-action="cancel">Cancel processing</button>' : ""}
      <button data-action="key">Save access key</button><button data-action="delete">Delete files</button></div></article>`;
          })
          .join("")
      : '<p class="quiet">No private jobs saved in this browser yet.</p>';
    if (content === lastHTML) return;
    lastHTML = content;
    jobsRoot.innerHTML = content;
    for (const button of jobsRoot.querySelectorAll<HTMLButtonElement>(
      "button[data-action]",
    ))
      button.onclick = async () => {
        const id = button.closest<HTMLElement>("[data-job]")!.dataset.job!;
        const job = load().find((j) => j.id === id)!;
        if (button.dataset.action === "review") return review(job);
        if (button.dataset.action === "key") {
          download(
            `private-reading-${id.slice(0, 8)}.json`,
            JSON.stringify(job, null, 2),
            "application/json",
          );
          message(
            "Keep that access key private. Anyone with it can open or delete this job until it expires.",
          );
          return;
        }
        if (button.dataset.action === "delete") {
          if (
            !confirm(
              "Delete this job’s uploaded PDF, rewrite, audio and downloads? This cannot be undone.",
            )
          )
            return;
          try {
            await request(`/jobs/${id}`, job, { method: "DELETE" });
          } catch (e: any) {
            if (!/unavailable|expired/i.test(e.message)) {
              message(e.message, true);
              return;
            }
          }
          remember(load().filter((j) => j.id !== id));
          $("#edition-review").hidden = true;
          draw();
          return;
        }
        button.disabled = true;
        try {
          if (button.dataset.action === "open") {
            const tab = window.open("about:blank", "_blank");
            if (tab) tab.opener = null;
            try {
              const data = await request(`/jobs/${id}/open`, job, {
                method: "POST",
              });
              const url = API + data.url;
              if (tab) tab.location.href = url;
              else location.href = url;
            } catch (e) {
              tab?.close();
              throw e;
            }
          } else {
            await request(`/jobs/${id}/${button.dataset.action}`, job, {
              method: "POST",
            });
            await poll();
          }
        } catch (e: any) {
          message(e.message, true);
        } finally {
          button.disabled = false;
        }
      };
  }

  async function poll() {
    if (!alive()) return;
    await Promise.all(
      load().map(async (j) => {
        try {
          views.set(j.id, await request(`/jobs/${j.id}`, j));
        } catch (e: any) {
          views.set(j.id, {
            id: j.id,
            kind: "pdf",
            state: "unavailable",
            stage: e.message,
            error: "",
            ahead: 0,
            expires: j.expires,
            retries: 0,
            audio: false,
          });
        }
      }),
    );
    if (alive()) draw();
  }
  async function loop() {
    await poll();
    if (alive()) setTimeout(loop, 4000);
  }
  draw();
  void loop();
  ($("#edition-submit") as HTMLButtonElement).onclick = async () => {
    if (submitting) return;
    const audio = ($("#edition-audio") as HTMLInputElement).checked;
    let file: File | undefined;
    if (!doi) {
      file = ($("#edition-file") as HTMLInputElement).files?.[0];
      if (!file) {
        message("Choose a research PDF first.", true);
        return;
      }
      if (file.size > 25 * 1024 * 1024) {
        message("The PDF must be under 25 MiB.", true);
        return;
      }
      if (
        !($("#edition-rights") as HTMLInputElement).checked ||
        !($("#edition-consent") as HTMLInputElement).checked
      ) {
        message(
          "Confirm permission and TypeSafe processing consent first.",
          true,
        );
        return;
      }
    }
    const submitButton = $("#edition-submit") as HTMLButtonElement;
    submitting = true;
    submitButton.disabled = true;
    message(
      doi ? "Preparing your private reading job…" : "Uploading your PDF…",
    );
    try {
      const data = doi
        ? await request("/papers", undefined, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ doi, model, audio }),
          })
        : await request("/jobs?audio=" + audio, undefined, {
            method: "POST",
            headers: {
              "Content-Type": "application/pdf",
              "X-SMR-Consent": "typesafe",
              "X-SMR-Rights": "permitted",
            },
            body: file,
            signal: AbortSignal.timeout(120000),
          });
      add({ ...data, title: doi || file!.name });
      message(
        "Saved in your private queue. You can leave this page and return using this browser.",
      );
      await poll();
    } catch (e: any) {
      message(
        /fetch|network/i.test(e.message)
          ? "The reading service could not be reached. Please try again shortly."
          : e.message,
        true,
      );
    } finally {
      submitting = false;
      submitButton.disabled = false;
    }
  };
  ($("#edition-restore") as HTMLInputElement).onchange = async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try {
      if (file.size > 5000) throw new Error("That access file is too large.");
      const job = JSON.parse(await file.text());
      if (
        !/^[a-f0-9]{32}$/.test(job.id) ||
        typeof job.token !== "string" ||
        job.token.length > 100
      )
        throw new Error("That is not a reading access key.");
      const v = await request(`/jobs/${job.id}`, job);
      add({
        ...job,
        title: String(job.title || "Restored reading job").slice(0, 300),
        expires: v.expires,
      });
      message("Private job restored on this device.");
      await poll();
    } catch (e: any) {
      message(e.message, true);
    }
  };
}
