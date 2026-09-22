import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { PredictOptions, PredictRequest, PredictResponse, PredictionService } from "../api";
import { usePrediction } from "./usePrediction";

const first: PredictRequest = { text: "First.", domain: "hotels", model_type: "gas" };
const second: PredictRequest = { text: "Second.", domain: "coursera", model_type: "paraphrase" };

function controllableService() {
  const calls: { signal?: AbortSignal; resolve: (value: PredictResponse) => void }[] = [];
  const service: PredictionService = {
    mode: "api",
    predict: vi.fn(
      (_request: PredictRequest, options: PredictOptions = {}) =>
        new Promise<PredictResponse>((resolve) => calls.push({ signal: options.signal, resolve })),
    ),
    predictBatch: vi.fn(),
  };
  return { service, calls };
}

describe("usePrediction", () => {
  it("aborts the previous request and ignores its late response", async () => {
    const { service, calls } = controllableService();
    const { result } = renderHook(() => usePrediction(service));

    act(() => void result.current.run(first));
    act(() => void result.current.run(second));
    expect(calls[0]!.signal?.aborted).toBe(true);
    expect(calls[1]!.signal?.aborted).toBe(false);

    await act(async () => calls[1]!.resolve({ ...second, quadruples: [] }));
    await act(async () =>
      calls[0]!.resolve({
        ...first,
        quadruples: [{ target: "room", aspect: "rooms general", opinion: "old", sentiment: "negative" }],
      }),
    );

    expect(result.current.state).toMatchObject({ status: "success", request: second, response: { text: "Second." } });
  });
});
