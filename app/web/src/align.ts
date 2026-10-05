/**
 * Which rewrite paragraphs go with which original paragraph (training/side_by_side.py):
 * each original paragraph gets a run of consecutive rewrite paragraphs (a rewrite may split a
 * paragraph, never reorders), chosen by dynamic programming to share the most numbers and words.
 */
const STOP = new Set(("the a an of and or in on at to for from by with is are was were be been this that these those we our it its as " +
  "than which who not no but if so such into over under between among also can may").split(" "));

function bag(text: string): Set<string> {
  const nums = text.match(/\d+(?:[.,]\d+)?/g) ?? [];
  const words = (text.toLowerCase().match(/[a-z]{4,}/g) ?? []).filter((w) => !STOP.has(w));
  return new Set([...nums, ...words]);
}

export function paragraphs(text: string): string[] {
  return (text ?? "").split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
}

export function align(orig: string[], rewrite: string[]): string[][] {
  const n = orig.length, m = rewrite.length;
  const bags = orig.map(bag);
  const sim = (i: number, k: number, j: number) => {
    if (j === k) return 0;
    const A = bags[i], B = bag(rewrite.slice(k, j).join(" "));
    if (!A.size || !B.size) return 0.05;
    let inter = 0;
    for (const x of A) if (B.has(x)) inter++;
    return inter / (A.size + B.size - inter);
  };
  const NEG = -1e9;
  const best = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(NEG));
  const back = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  best[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 0; j <= m; j++) {
      for (let k = Math.max(0, j - 6); k <= j; k++) {
        if (best[i - 1][k] === NEG) continue;
        const v = best[i - 1][k] + sim(i - 1, k, j) - (j === k ? 0.02 : 0);
        if (v > best[i][j]) { best[i][j] = v; back[i][j] = k; }
      }
    }
  }
  const groups: string[][] = [];
  let j = m;
  for (let i = n; i > 0; i--) {
    const k = back[i][j];
    groups.push(rewrite.slice(k, j));
    j = k;
  }
  groups.reverse();
  if (j > 0 && groups.length) groups[0] = [...rewrite.slice(0, j), ...groups[0]];
  return groups;
}
