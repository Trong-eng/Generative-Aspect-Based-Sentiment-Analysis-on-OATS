import { describe, expect, it } from "vitest";
import { PredictionError, type PredictRequest } from "./types";
import { parseErrorBody, parsePredictResponse } from "./validate";

const request: PredictRequest = { text: "The room was spotless.", domain: "hotels", model_type: "paraphrase" };
const quad = { target: "room", aspect: "room cleanliness", opinion: "spotless", sentiment: "positive" };

function expectMalformed(body: unknown) {
  try {
    parsePredictResponse(body, request);
  } catch (error) {
    expect(error).toBeInstanceOf(PredictionError);
    expect((error as PredictionError).kind).toBe("malformed");
    return;
  }
  throw new Error("expected parsePredictResponse to throw");
}

describe("parsePredictResponse", () => {
  it("accepts the contract example and keeps values unchanged", () => {
    const body = {
      ...request,
      quadruples: [quad],
      raw_generation: "room cleanliness is great because room is spotless",
      invalid_predictions: [],
    };
    expect(parsePredictResponse(body, request)).toEqual(body);
  });

  it("tolerates missing optional diagnostics", () => {
    const parsed = parsePredictResponse({ ...request, quadruples: [] }, request);
    expect(parsed.quadruples).toEqual([]);
    expect(parsed).not.toHaveProperty("raw_generation");
    expect(parsed).not.toHaveProperty("invalid_predictions");
  });

  it("drops mistyped optional diagnostics instead of failing", () => {
    const parsed = parsePredictResponse(
      { ...request, quadruples: [quad], raw_generation: 42, invalid_predictions: [{ raw: "x", reason: "y" }, "junk"] },
      request,
    );
    expect(parsed).not.toHaveProperty("raw_generation");
    expect(parsed.invalid_predictions).toEqual([{ raw: "x", reason: "y" }]);
  });

  it("falls back to request identity when echo fields are missing", () => {
    const parsed = parsePredictResponse({ quadruples: [quad] }, request);
    expect(parsed).toMatchObject(request);
  });

  it.each([
    ["null body", null],
    ["array body", [quad]],
    ["missing quadruples", { ...request }],
    ["quadruples not a list", { ...request, quadruples: "room" }],
    ["quadruple not an object", { ...request, quadruples: ["room"] }],
    ["quadruple missing a field", { ...request, quadruples: [{ target: "room", aspect: "a", opinion: "b" }] }],
    ["quadruple with non-string field", { ...request, quadruples: [{ ...quad, sentiment: 1 }] }],
  ])("rejects %s as malformed", (_label, body) => {
    expectMalformed(body);
  });
});

describe("parseErrorBody", () => {
  it("reads the contract error shape", () => {
    expect(parseErrorBody({ error: { code: "MODEL_NOT_READY", message: "Checkpoint chưa sẵn sàng." } })).toEqual({
      code: "MODEL_NOT_READY",
      message: "Checkpoint chưa sẵn sàng.",
    });
  });

  it("ignores non-contract bodies such as tracebacks", () => {
    expect(parseErrorBody({ detail: "Traceback (most recent call last): ..." })).toEqual({});
    expect(parseErrorBody("<html>500</html>")).toEqual({});
    expect(parseErrorBody(undefined)).toEqual({});
  });

  it("caps very long messages", () => {
    const { message } = parseErrorBody({ error: { code: "X", message: "a".repeat(1000) } });
    expect(message).toHaveLength(300);
  });
});
