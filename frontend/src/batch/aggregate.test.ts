import { describe, expect, it } from "vitest";
import type { Quadruple } from "../api";
import { aggregateByReview, aspectKey, aspectsToCsv, reviewLabel, sortAspects, type AnalyzedRow } from "./aggregate";

function q(aspect: string, sentiment: string, opinion = "x", target = "t"): Quadruple {
  return { target, aspect, opinion, sentiment };
}

function row(reviewId: string, ...quadruples: Quadruple[]): AnalyzedRow {
  return { reviewId, text: `text of ${reviewId}`, quadruples };
}

describe("reviewLabel", () => {
  it.each([
    [["negative", "negative"], "negative"],
    [["positive"], "positive"],
    [["neutral", "neutral"], "neutral"],
    [["positive", "neutral"], "positive"],
    [["negative", "neutral"], "negative"],
    [["positive", "negative"], "conflict"],
    [["conflict"], "conflict"],
    [["positive", "conflict"], "conflict"],
  ] as const)("%j → %s", (sentiments, expected) => {
    expect(reviewLabel([...sentiments])).toBe(expected);
  });

  it("returns null for no mentions", () => {
    expect(reviewLabel([])).toBeNull();
  });
});

describe("aggregateByReview", () => {
  it("counts each review once per aspect, not once per mention", () => {
    const data = aggregateByReview(
      [
        row("r1", q("rooms cleanliness", "negative", "dirty"), q("rooms cleanliness", "negative", "stained")),
        row("r2", q("rooms cleanliness", "positive", "spotless")),
        row("r3", q("service general", "positive")),
      ],
      "category",
    );
    const rooms = data.aspects.find((stat) => stat.aspect === "rooms cleanliness")!;
    expect(rooms.reviews).toBe(2);
    expect(rooms.counts).toEqual({ positive: 1, negative: 1, neutral: 0, conflict: 0 });
    expect(rooms.shares.negative).toBe(0.5);
    expect(data.reviewCount).toBe(3);
    expect(data.reviewsWithAspects).toBe(3);
  });

  it("combines rows with the same review id before labelling", () => {
    const data = aggregateByReview(
      [row("r1", q("rooms cleanliness", "positive")), row("r1", q("rooms cleanliness", "negative"))],
      "category",
    );
    expect(data.reviewCount).toBe(1);
    expect(data.aspects[0]).toMatchObject({ reviews: 1, counts: { conflict: 1 } });
    expect(data.aspects[0]!.examples[0]!.text).toBe("text of r1 text of r1");
  });

  it("shows model conflict labels separately and keeps shares summing to 1", () => {
    const data = aggregateByReview(
      [
        row("r1", q("course quality", "conflict")),
        row("r2", q("course quality", "neutral")),
        row("r3", q("course quality", "negative")),
        row("r4", q("course quality", "positive")),
      ],
      "category",
    );
    const stat = data.aspects[0]!;
    expect(stat.counts).toEqual({ positive: 1, negative: 1, neutral: 1, conflict: 1 });
    expect(Object.values(stat.shares).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
  });

  it("groups by entity when requested", () => {
    expect(aspectKey("  rooms   cleanliness ", "entity")).toBe("rooms");
    const data = aggregateByReview(
      [row("r1", q("rooms cleanliness", "negative"), q("rooms comfort", "positive"))],
      "entity",
    );
    expect(data.aspects).toHaveLength(1);
    expect(data.aspects[0]).toMatchObject({ aspect: "rooms", reviews: 1, counts: { conflict: 1 } });
  });

  it("ranks opinions by the number of reviews and ignores implicit opinions", () => {
    const data = aggregateByReview(
      [
        row("r1", q("rooms cleanliness", "negative", "Dirty"), q("rooms cleanliness", "negative", "dirty")),
        row("r2", q("rooms cleanliness", "negative", "dirty")),
        row("r3", q("rooms cleanliness", "negative", "stained")),
        row("r4", q("rooms cleanliness", "negative", "NULL")),
      ],
      "category",
    );
    expect(data.aspects[0]!.opinions.negative).toEqual([
      { opinion: "dirty", reviews: 2 },
      { opinion: "stained", reviews: 1 },
    ]);
  });

  it("ignores unknown sentiment labels and empty aspects but counts them", () => {
    const data = aggregateByReview([row("r1", q("rooms general", "mixed"), q(" ", "negative"))], "category");
    expect(data.aspects).toEqual([]);
    expect(data.unknownSentimentQuadruples).toBe(1);
    expect(data.reviewsWithAspects).toBe(0);
  });
});

describe("sortAspects and aspectsToCsv", () => {
  const data = aggregateByReview(
    [
      row("r1", q("a", "negative"), q("b", "negative"), q("c", "positive")),
      row("r2", q("a", "positive"), q("b", "negative"), q("c", "positive")),
      row("r3", q("b", "positive"), q("c", "negative")),
    ],
    "category",
  );

  it("sorts by negative share, negative count, volume or name", () => {
    expect(sortAspects(data.aspects, "negative_share").map((stat) => stat.aspect)).toEqual(["b", "a", "c"]);
    expect(sortAspects(data.aspects, "negative_count").map((stat) => stat.aspect)).toEqual(["b", "c", "a"]);
    expect(sortAspects(data.aspects, "reviews").map((stat) => stat.aspect)).toEqual(["b", "c", "a"]);
    expect(sortAspects(data.aspects, "aspect").map((stat) => stat.aspect)).toEqual(["a", "b", "c"]);
  });

  it("exports counts and percentages, escaping cells and neutralising formulas", () => {
    const tricky = aggregateByReview([row("r1", q('=cmd,"x"', "negative", "bad, really"))], "category");
    const csv = aspectsToCsv([...sortAspects(data.aspects, "aspect").slice(0, 1), ...tricky.aspects]);
    const lines = csv.trimEnd().split("\r\n");
    expect(lines[0]).toBe(
      "aspect,reviews,positive,negative,neutral,conflict,positive_pct,negative_pct,neutral_pct,conflict_pct,top_negative_opinions",
    );
    expect(lines[1]).toBe("a,2,1,1,0,0,50.0,50.0,0.0,0.0,x (1)");
    expect(lines[2]).toBe(`"'=cmd,""x""",1,0,1,0,0,0.0,100.0,0.0,0.0,"bad, really (1)"`);
  });
});
