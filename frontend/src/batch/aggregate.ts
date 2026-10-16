import type { Quadruple } from "../api";

export const REVIEW_SENTIMENTS = ["positive", "negative", "neutral", "conflict"] as const;
export type ReviewSentiment = (typeof REVIEW_SENTIMENTS)[number];

/** `category` keeps full OATS aspects ("rooms cleanliness"); `entity` groups by the first word ("rooms"). */
export type GroupBy = "category" | "entity";

export interface AnalyzedRow {
  reviewId: string;
  text: string;
  quadruples: Quadruple[];
}

export interface OpinionCount {
  opinion: string;
  /** Number of reviews using this opinion for the aspect. */
  reviews: number;
}

export interface AspectReview {
  reviewId: string;
  text: string;
  sentiment: ReviewSentiment;
  quadruples: Quadruple[];
}

export interface AspectStat {
  aspect: string;
  /** Reviews that mention the aspect at least once (the denominator of every percentage). */
  reviews: number;
  counts: Record<ReviewSentiment, number>;
  /** counts / reviews, 0..1. */
  shares: Record<ReviewSentiment, number>;
  opinions: Record<"positive" | "negative", OpinionCount[]>;
  examples: AspectReview[];
}

export interface DashboardData {
  reviewCount: number;
  reviewsWithAspects: number;
  aspects: AspectStat[];
  /** Quadruples whose sentiment is not one of the four OATS labels; excluded from counts. */
  unknownSentimentQuadruples: number;
}

function normalizeSentiment(value: string): ReviewSentiment | null {
  const lowered = value.trim().toLowerCase();
  return (REVIEW_SENTIMENTS as readonly string[]).includes(lowered) ? (lowered as ReviewSentiment) : null;
}

/**
 * Collapses every mention of one aspect inside one review into a single review-level label:
 * - any `conflict`, or both positive and negative → conflict
 * - otherwise positive or negative if present (a polar mention outweighs a neutral one)
 * - otherwise neutral
 */
export function reviewLabel(sentiments: ReviewSentiment[]): ReviewSentiment | null {
  if (sentiments.length === 0) return null;
  const set = new Set(sentiments);
  if (set.has("conflict") || (set.has("positive") && set.has("negative"))) return "conflict";
  if (set.has("negative")) return "negative";
  if (set.has("positive")) return "positive";
  return "neutral";
}

export function aspectKey(aspect: string, groupBy: GroupBy): string {
  const trimmed = aspect.trim().replace(/\s+/g, " ");
  return groupBy === "entity" ? trimmed.split(" ")[0]! : trimmed;
}

function isImplicit(value: string): boolean {
  return value.trim().toUpperCase() === "NULL";
}

function topOpinions(counter: Map<string, Set<string>>): OpinionCount[] {
  return [...counter.entries()]
    .map(([opinion, reviews]) => ({ opinion, reviews: reviews.size }))
    .sort((a, b) => b.reviews - a.reviews || a.opinion.localeCompare(b.opinion));
}

const emptyCounts = (): Record<ReviewSentiment, number> => ({ positive: 0, negative: 0, neutral: 0, conflict: 0 });

/** Aggregates quadruples into per-aspect sentiment shares, counting each review at most once per aspect. */
export function aggregateByReview(rows: AnalyzedRow[], groupBy: GroupBy): DashboardData {
  const reviews = new Map<string, { text: string[]; quadruples: Quadruple[] }>();
  for (const row of rows) {
    const review = reviews.get(row.reviewId) ?? { text: [], quadruples: [] };
    review.text.push(row.text);
    review.quadruples.push(...row.quadruples);
    reviews.set(row.reviewId, review);
  }

  let unknownSentimentQuadruples = 0;
  let reviewsWithAspects = 0;
  const byAspect = new Map<
    string,
    { counts: Record<ReviewSentiment, number>; examples: AspectReview[]; opinions: Record<"positive" | "negative", Map<string, Set<string>>> }
  >();

  for (const [reviewId, review] of reviews) {
    const perAspect = new Map<string, { sentiments: ReviewSentiment[]; quadruples: Quadruple[] }>();
    for (const quad of review.quadruples) {
      const sentiment = normalizeSentiment(quad.sentiment);
      if (!sentiment) {
        unknownSentimentQuadruples++;
        continue;
      }
      if (!quad.aspect.trim()) continue;
      const key = aspectKey(quad.aspect, groupBy);
      const entry = perAspect.get(key) ?? { sentiments: [], quadruples: [] };
      entry.sentiments.push(sentiment);
      entry.quadruples.push(quad);
      perAspect.set(key, entry);
    }
    if (perAspect.size > 0) reviewsWithAspects++;

    for (const [key, entry] of perAspect) {
      const label = reviewLabel(entry.sentiments)!;
      const stat = byAspect.get(key) ?? {
        counts: emptyCounts(),
        examples: [] as AspectReview[],
        opinions: { positive: new Map(), negative: new Map() },
      };
      stat.counts[label]++;
      stat.examples.push({ reviewId, text: review.text.join(" "), sentiment: label, quadruples: entry.quadruples });
      for (const quad of entry.quadruples) {
        const polarity = normalizeSentiment(quad.sentiment);
        if ((polarity === "positive" || polarity === "negative") && !isImplicit(quad.opinion)) {
          const opinion = quad.opinion.trim().toLowerCase();
          const seen = stat.opinions[polarity].get(opinion) ?? new Set<string>();
          seen.add(reviewId);
          stat.opinions[polarity].set(opinion, seen);
        }
      }
      byAspect.set(key, stat);
    }
  }

  const aspects: AspectStat[] = [...byAspect.entries()].map(([aspect, stat]) => {
    const total = REVIEW_SENTIMENTS.reduce((sum, sentiment) => sum + stat.counts[sentiment], 0);
    const shares = emptyCounts();
    for (const sentiment of REVIEW_SENTIMENTS) shares[sentiment] = total ? stat.counts[sentiment] / total : 0;
    return {
      aspect,
      reviews: total,
      counts: stat.counts,
      shares,
      opinions: { positive: topOpinions(stat.opinions.positive), negative: topOpinions(stat.opinions.negative) },
      examples: stat.examples,
    };
  });

  return { reviewCount: reviews.size, reviewsWithAspects, aspects, unknownSentimentQuadruples };
}

export type SortKey = "negative_share" | "negative_count" | "reviews" | "aspect";

export function sortAspects(aspects: AspectStat[], sortKey: SortKey): AspectStat[] {
  const sorted = [...aspects];
  const byName = (a: AspectStat, b: AspectStat) => a.aspect.localeCompare(b.aspect);
  switch (sortKey) {
    case "negative_share":
      return sorted.sort((a, b) => b.shares.negative - a.shares.negative || b.reviews - a.reviews || byName(a, b));
    case "negative_count":
      return sorted.sort((a, b) => b.counts.negative - a.counts.negative || b.reviews - a.reviews || byName(a, b));
    case "reviews":
      return sorted.sort((a, b) => b.reviews - a.reviews || byName(a, b));
    case "aspect":
      return sorted.sort(byName);
  }
}

function csvCell(value: string | number): string {
  // Leading = + - @ would be evaluated as a formula by spreadsheet apps.
  const text = typeof value === "string" && /^[=+\-@]/.test(value) ? `'${value}` : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Aggregated table as CSV (counts and percentages), for spreadsheets or reports. */
export function aspectsToCsv(aspects: AspectStat[]): string {
  const header = [
    "aspect",
    "reviews",
    ...REVIEW_SENTIMENTS,
    ...REVIEW_SENTIMENTS.map((sentiment) => `${sentiment}_pct`),
    "top_negative_opinions",
  ];
  const lines = aspects.map((stat) =>
    [
      stat.aspect,
      stat.reviews,
      ...REVIEW_SENTIMENTS.map((sentiment) => stat.counts[sentiment]),
      ...REVIEW_SENTIMENTS.map((sentiment) => (stat.shares[sentiment] * 100).toFixed(1)),
      stat.opinions.negative
        .slice(0, 5)
        .map((item) => `${item.opinion} (${item.reviews})`)
        .join("; "),
    ]
      .map(csvCell)
      .join(","),
  );
  return [header.join(","), ...lines].join("\r\n") + "\r\n";
}
