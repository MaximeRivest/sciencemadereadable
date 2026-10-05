/**
 * The string values of a JSON object that is still being written: the big models answer with
 * {"title": "...", "abstract": "...", ...} (or {"text": "..."}), streamed a few characters at a time.
 * Returns, for each wanted key whose value has started, its decoded text so far. Tolerates a cut
 * anywhere, including inside an escape sequence (that part waits for the next piece).
 * Display only: the final, checked answer comes from the finished prediction.
 */
export function partialStrings(json: string, keys: readonly string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of keys) {
    const m = new RegExp(`"${key}"\\s*:\\s*"`).exec(json);
    if (!m) continue;
    let i = m.index + m[0].length;
    let s = "";
    while (i < json.length) {
      const c = json[i];
      if (c === '"') break;
      if (c !== "\\") { s += c; i++; continue; }
      const e = json[i + 1];
      if (e === undefined) break;                       // cut after the backslash
      if (e === "u") {
        const hex = json.slice(i + 2, i + 6);
        if (hex.length < 4) break;                      // cut inside \uXXXX
        s += String.fromCharCode(parseInt(hex, 16));
        i += 6;
        continue;
      }
      s += ({ n: "\n", t: "\t", r: "\r", b: "\b", f: "\f", '"': '"', "\\": "\\", "/": "/" } as Record<string, string>)[e] ?? e;
      i += 2;
    }
    out[key] = s;
  }
  return out;
}
