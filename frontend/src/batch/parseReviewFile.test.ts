import { describe, expect, it } from "vitest";
import { FileParseError, MAX_ROWS, parseCsv, parseReviewFile } from "./parseReviewFile";

const quad = { target: "room", aspect: "rooms cleanliness", opinion: "dirty", sentiment: "negative" };

function parseError(run: () => unknown): string {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(FileParseError);
    return (error as Error).message;
  }
  throw new Error("expected a FileParseError");
}

describe("parseCsv", () => {
  it("handles quotes, doubled quotes, embedded commas/newlines and CRLF", () => {
    const records = parseCsv('id,text\r\n1,"Clean, quiet"\r\n2,"He said ""wow""\nthen left"\r\n\r\n3,plain');
    expect(records.map((record) => record.cells)).toEqual([
      ["id", "text"],
      ["1", "Clean, quiet"],
      ["2", 'He said "wow"\nthen left'],
      ["3", "plain"],
    ]);
    expect(records.map((record) => record.line)).toEqual([1, 2, 3, 6]);
  });

  it("rejects an unclosed quote", () => {
    expect(parseError(() => parseCsv('text\n"never closed'))).toMatch(/unclosed quote/);
  });
});

describe("parseReviewFile", () => {
  it("reads one review per line from .txt and skips blank lines", () => {
    const parsed = parseReviewFile("reviews.txt", "Great room.\n\n  \nDirty bathroom.\r\n");
    expect(parsed.format).toBe("txt");
    expect(parsed.rows.map((row) => row.text)).toEqual(["Great room.", "Dirty bathroom."]);
    expect(parsed.reviewCount).toBe(2);
    expect(parsed.precomputed).toBe(false);
  });

  it("reads a CSV text column, strips a BOM, and groups rows by review_id", () => {
    const parsed = parseReviewFile(
      "reviews.CSV",
      "﻿Review_ID,Rating,Text\nr1,2,The room was dirty.\nr1,2,Staff were rude.\nr2,5,Lovely stay.\n",
    );
    expect(parsed.rows).toHaveLength(3);
    expect(parsed.reviewCount).toBe(2);
    expect(parsed.rows[0]).toMatchObject({ reviewId: "r1", text: "The room was dirty." });
    expect(new Set(parsed.rows.map((row) => row.rowId)).size).toBe(3);
  });

  it("requires a text column in CSV", () => {
    expect(parseError(() => parseReviewFile("x.csv", "id,body\n1,hello"))).toMatch(/text column/);
  });

  it("keeps good JSONL rows and reports bad ones", () => {
    const parsed = parseReviewFile(
      "x.jsonl",
      ['{"text": "Clean room", "id": 7}', "not json", '{"text": ""}', '{"other": 1}', '"Nice staff"'].join("\n"),
    );
    expect(parsed.rows.map((row) => [row.reviewId, row.text])).toEqual([
      ["7", "Clean room"],
      ["row-5", "Nice staff"],
    ]);
    expect(parsed.issues).toEqual([
      { line: 2, message: "Invalid JSON." },
      { line: 3, message: "Missing or empty text." },
      { line: 4, message: "Missing or empty text." },
    ]);
  });

  it("reads a JSON array and rejects other JSON roots", () => {
    expect(parseReviewFile("x.json", '["A", {"text": "B"}]').rows).toHaveLength(2);
    expect(parseError(() => parseReviewFile("x.json", '{"text": "A"}'))).toMatch(/list of reviews/);
    expect(parseError(() => parseReviewFile("x.json", "{"))).toMatch(/not valid JSON/);
  });

  it("uses predictions already in the file (capstone.predict output, object or tuple quads)", () => {
    const parsed = parseReviewFile(
      "preds.jsonl",
      [
        JSON.stringify({ text: "Dirty room.", quadruples: [quad], raw_generation: "x" }),
        JSON.stringify({ text: "Nothing.", quadruples: [] }),
        JSON.stringify({ text: "Tuple.", quadruples: [["room", "rooms cleanliness", "dirty", "negative"]] }),
      ].join("\n"),
    );
    expect(parsed.precomputed).toBe(true);
    expect(parsed.rows.map((row) => row.quadruples)).toEqual([[quad], [], [quad]]);
  });

  it("prefers predicted over gold quadruples in capstone.evaluate predictions.jsonl", () => {
    const row = {
      review_id: "r9",
      text: "Dirty room.",
      gold_quadruples: [{ ...quad, opinion: "GOLD" }],
      normalized_quadruples: [quad],
    };
    const parsed = parseReviewFile("predictions.jsonl", JSON.stringify(row));
    expect(parsed.rows[0]).toMatchObject({ reviewId: "r9", quadruples: [quad] });
  });

  it("rejects files that mix rows with and without predictions", () => {
    const content = [JSON.stringify({ text: "A", quadruples: [] }), JSON.stringify({ text: "B" })].join("\n");
    expect(parseError(() => parseReviewFile("x.jsonl", content))).toMatch(/Only 1 of 2 rows include quadruples/);
  });

  it("skips rows whose quadruples are malformed", () => {
    const content = [
      JSON.stringify({ text: "A", quadruples: [quad] }),
      JSON.stringify({ text: "B", quadruples: [{ target: "x" }] }),
    ].join("\n");
    const parsed = parseReviewFile("x.jsonl", content);
    expect(parsed.rows).toHaveLength(1);
    expect(parsed.issues[0]).toMatchObject({ line: 2 });
  });

  it("rejects unsupported, empty and oversized files", () => {
    expect(parseError(() => parseReviewFile("reviews.xlsx", "x"))).toMatch(/Unsupported file type/);
    expect(parseError(() => parseReviewFile("reviews.txt", "\n  \n"))).toMatch(/No reviews/);
    const tooMany = Array.from({ length: MAX_ROWS + 1 }, (_, index) => `Review ${index}`).join("\n");
    expect(parseError(() => parseReviewFile("reviews.txt", tooMany))).toMatch(/limit is 5000/);
  });
});
