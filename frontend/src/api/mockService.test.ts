import { describe, expect, it } from "vitest";
import { createPredictionService, isMockEnabled } from ".";
import { MOCK_FIXTURES } from "./fixtures";
import { createMockPredictionService } from "./mockService";
import { DOMAINS, MODEL_TYPES, PredictionError } from "./types";
import { parsePredictResponse } from "./validate";

const mock = createMockPredictionService({ delayMs: 0 });

describe("createMockPredictionService", () => {
  it("returns fixtures that depend on domain and model and satisfy the contract", async () => {
    const seen = new Set<string>();
    for (const domain of DOMAINS) {
      for (const model_type of MODEL_TYPES) {
        const request = { text: "Some sentence.", domain, model_type };
        const response = await mock.predict(request);
        expect(response).toMatchObject(request);
        expect(response.quadruples).toEqual(MOCK_FIXTURES[domain][model_type].quadruples);
        // The mock must pass the same validation as real API responses.
        expect(parsePredictResponse(JSON.parse(JSON.stringify(response)), request)).toEqual(response);
        seen.add(JSON.stringify(response.quadruples));
      }
    }
    expect(seen.size).toBe(DOMAINS.length * MODEL_TYPES.length);
  });

  it("simulates empty and error outcomes with text markers", async () => {
    const empty = await mock.predict({ text: "Nothing here #empty", domain: "hotels", model_type: "gas" });
    expect(empty.quadruples).toEqual([]);

    await expect(mock.predict({ text: "#error", domain: "hotels", model_type: "gas" })).rejects.toMatchObject({
      kind: "server",
      code: "MODEL_NOT_READY",
    });
  });

  it("honours cancellation", async () => {
    const slow = createMockPredictionService({ delayMs: 10_000 });
    const controller = new AbortController();
    const pending = slow.predict({ text: "x", domain: "hotels", model_type: "gas" }, { signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toBeInstanceOf(PredictionError);
    await expect(pending).rejects.toMatchObject({ kind: "aborted" });
  });
});

describe("createPredictionService", () => {
  it("defaults to mock mode and switches to the API only when disabled explicitly", () => {
    expect(isMockEnabled({})).toBe(true);
    expect(isMockEnabled({ VITE_USE_MOCK_API: "true" })).toBe(true);
    expect(isMockEnabled({ VITE_USE_MOCK_API: "false" })).toBe(false);
    expect(isMockEnabled({ VITE_USE_MOCK_API: "0" })).toBe(false);

    expect(createPredictionService({}).mode).toBe("mock");
    expect(createPredictionService({ VITE_USE_MOCK_API: "false", VITE_API_BASE_URL: "https://x.test" }).mode).toBe(
      "api",
    );
  });
});
