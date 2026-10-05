// lm15's browser build, plus a stand-in for the one terminal-only export functai imports
// (accounts.ts: TerminalUI, used only by interactive logins, which a web page never runs).
export * from "@lm15/lm15/browser";
export class TerminalUI {
  constructor() { throw new Error("TerminalUI is not available in a browser"); }
}
