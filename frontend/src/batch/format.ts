import type { ReviewSentiment } from "./aggregate";

export const SENTIMENT_LABELS: Record<ReviewSentiment, string> = {
  positive: "Positive",
  negative: "Negative",
  neutral: "Neutral",
  conflict: "Conflict",
};

export function percent(share: number): string {
  return `${Math.round(share * 100)}%`;
}
