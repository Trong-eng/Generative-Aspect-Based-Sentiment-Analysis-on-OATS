const KNOWN_SENTIMENTS: Record<string, { symbol: string; tone: string }> = {
  positive: { symbol: "+", tone: "positive" },
  negative: { symbol: "−", tone: "negative" },
  neutral: { symbol: "○", tone: "neutral" },
  conflict: { symbol: "±", tone: "conflict" },
};

/** Sentiment is conveyed by the text and a symbol, with color only as reinforcement. */
export function SentimentBadge({ sentiment }: { sentiment: string }) {
  const known = KNOWN_SENTIMENTS[sentiment.trim().toLowerCase()];
  const tone = known?.tone ?? "unknown";
  return (
    <span className={`sentiment sentiment-${tone}`} data-sentiment={tone}>
      <span className="sentiment-symbol" aria-hidden="true">
        {known?.symbol ?? "?"}
      </span>
      {sentiment}
    </span>
  );
}
