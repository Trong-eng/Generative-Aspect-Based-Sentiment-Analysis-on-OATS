import { describe, expect, it } from "vitest";
import sampleCsv from "../../public/samples/hotels_reviews_sample.csv?raw";
import { aggregateByReview, sortAspects } from "../batch/aggregate";
import { parseReviewFile } from "../batch/parseReviewFile";
import { createMockPredictionService } from "./mockService";
import { mockExtract } from "./mockLexicon";

describe("mockExtract", () => {
  it("maps entity + opinion words to OATS categories", () => {
    expect(mockExtract("Great location, but the room was dirty.", "hotels")).toEqual([
      { target: "location", aspect: "location general", opinion: "Great", sentiment: "positive" },
      { target: "room", aspect: "rooms cleanliness", opinion: "dirty", sentiment: "negative" },
    ]);
  });

  it("flips polarity after a negation and keeps the negation in the opinion", () => {
    expect(mockExtract("The bathroom was not clean.", "hotels")).toEqual([
      { target: "bathroom", aspect: "rooms cleanliness", opinion: "not clean", sentiment: "negative" },
    ]);
  });

  it("uses NULL target and the domain's general aspect when no entity is named", () => {
    expect(mockExtract("Absolutely terrible.", "coursera")).toEqual([
      { target: "NULL", aspect: "course general", opinion: "terrible", sentiment: "negative" },
    ]);
  });

  it("does not match a word inside a longer word", () => {
    expect(mockExtract("The receptionist was unhelpful.", "hotels")).toEqual([
      { target: "receptionist", aspect: "service general", opinion: "unhelpful", sentiment: "negative" },
    ]);
  });
});

describe("bundled hotels sample", () => {
  it("surfaces rooms cleanliness as the most negative aspect in mock mode", async () => {
    const file = parseReviewFile("sample.csv", sampleCsv);
    const service = createMockPredictionService({ delayMs: 0 });
    const { results } = await service.predictBatch({
      domain: "hotels",
      model_type: "paraphrase",
      items: file.rows.map((row) => ({ id: row.rowId, text: row.text })),
    });
    const byId = new Map(results.map((result) => [result.id, result.quadruples]));
    const data = aggregateByReview(
      file.rows.map((row) => ({ reviewId: row.reviewId, text: row.text, quadruples: byId.get(row.rowId)! })),
      "category",
    );
    const [weakest] = sortAspects(
      data.aspects.filter((stat) => stat.reviews >= 3),
      "negative_share",
    );
    expect(file.reviewCount).toBe(30);
    expect(weakest?.aspect).toBe("rooms cleanliness");
    expect(weakest?.opinions.negative[0]?.opinion).toBe("dirty");
  });
});
