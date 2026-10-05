/**
 * The nine writers, with the settings their scored rewrites used and what we measured
 * (3 held-out papers; training/speed, training/head_to_head).
 * Prices: list prices per million tokens, October 2026.
 */
export type Provider = "anthropic" | "openai" | "ours";

export interface ModelInfo {
  id: string;
  name: string;
  provider: Provider;
  lm: string;                 // the provider's model name (ours: the name on our server)
  effort?: "off" | "low";     // reasoning, as in the scored runs
  maxTokens?: number;
  inPrice?: number;           // USD per million input tokens
  outPrice?: number;
  score: number;              // average judge score, 0-10
  winRate: number;            // head-to-head win rate (Opus judge), %
  seconds: number;            // one paper, measured
  usdPerPaper: number;        // a typical 7,100-word paper
  note?: string;
}

export const MODELS: ModelInfo[] = [
  { id: "opus", name: "Claude Opus 5.5", provider: "anthropic", lm: "claude-opus-5-5", effort: "off", maxTokens: 64000,
    inPrice: 4, outPrice: 20, score: 8.45, winRate: 88, seconds: 83, usdPerPaper: 0.97 },
  { id: "sonnet", name: "Claude Sonnet 5.5", provider: "anthropic", lm: "claude-sonnet-5-5", effort: "off", maxTokens: 64000,
    inPrice: 2, outPrice: 10, score: 8.07, winRate: 67, seconds: 60, usdPerPaper: 0.44 },
  { id: "astra", name: "GPT-6 Astra", provider: "openai", lm: "gpt-6-astra", effort: "low",
    inPrice: 10, outPrice: 50, score: 8.15, winRate: 75, seconds: 301, usdPerPaper: 1.21, note: "slow: 2 to 8 minutes" },
  { id: "sol", name: "GPT-6 Sol", provider: "openai", lm: "gpt-6-sol", effort: "off",
    inPrice: 2, outPrice: 10, score: 7.87, winRate: 59, seconds: 71, usdPerPaper: 0.21 },
  { id: "terra", name: "GPT-5.6 Terra", provider: "openai", lm: "gpt-5.6-terra", effort: "off",
    inPrice: 2, outPrice: 12, score: 7.45, winRate: 35, seconds: 72, usdPerPaper: 0.25 },
  { id: "luna", name: "GPT-6 Luna", provider: "openai", lm: "gpt-6-luna", effort: "off",
    inPrice: 0.1, outPrice: 0.5, score: 7.5, winRate: 29, seconds: 82, usdPerPaper: 0.011 },
  { id: "our-9b", name: "Our 9B", provider: "ours", lm: "our-9b",
    score: 7.78, winRate: 55, seconds: 15, usdPerPaper: 0.005, note: "Qwen 3.5 9B, fine-tuned" },
  { id: "our-4b", name: "Our 4B", provider: "ours", lm: "our-4b",
    score: 7.2, winRate: 40, seconds: 9, usdPerPaper: 0.004, note: "Qwen 3.5 4B, fine-tuned" },
  { id: "our-0.8b", name: "Our 0.8B", provider: "ours", lm: "our-0.8b",
    score: 5.73, winRate: 2, seconds: 9, usdPerPaper: 0.001, note: "Qwen 3.5 0.8B, fine-tuned; often inaccurate" },
];
