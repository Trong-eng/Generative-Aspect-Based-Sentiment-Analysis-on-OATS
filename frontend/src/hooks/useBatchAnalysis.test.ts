import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  PredictionError,
  type BatchPredictRequest,
  type BatchPredictResponse,
  type PredictOptions,
  type PredictionService,
} from "../api";
import { parseReviewFile } from "../batch/parseReviewFile";
import { BATCH_CHUNK_SIZE, useBatchAnalysis } from "./useBatchAnalysis";

const quad = { target: "room", aspect: "rooms cleanliness", opinion: "dirty", sentiment: "negative" };

function fileOf(count: number) {
  return parseReviewFile("reviews.txt", Array.from({ length: count }, (_, index) => `Review ${index + 1}`).join("\n"));
}

function echoResponse(request: BatchPredictRequest): BatchPredictResponse {
  return { results: request.items.map((item) => ({ id: item.id, quadruples: [quad] })) };
}

function serviceWith(predictBatch: PredictionService["predictBatch"]) {
  const spy = vi.fn(predictBatch);
  const service: PredictionService = { mode: "api", predict: vi.fn(), predictBatch: spy };
  return { service, spy };
}

describe("useBatchAnalysis", () => {
  it("sends the file in chunks and reports progress until done", async () => {
    const { service, spy } = serviceWith(async (request) => echoResponse(request));
    const { result } = renderHook(() => useBatchAnalysis(service));
    const file = fileOf(BATCH_CHUNK_SIZE * 2 + 5);

    await act(async () => result.current.start({ file, domain: "hotels", modelType: "gas" }));

    expect(spy).toHaveBeenCalledTimes(3);
    expect(spy.mock.calls.map(([request]) => request.items.length)).toEqual([BATCH_CHUNK_SIZE, BATCH_CHUNK_SIZE, 5]);
    expect(spy.mock.calls[0]![0]).toMatchObject({ domain: "hotels", model_type: "gas" });
    expect(result.current.state.status).toBe("done");
    if (result.current.state.status === "done") {
      expect(result.current.state.rows).toHaveLength(45);
      expect(result.current.state.rows[0]).toEqual({ reviewId: "row-1", text: "Review 1", quadruples: [quad] });
    }
  });

  it("stops on an error and resumes with only the remaining rows", async () => {
    let calls = 0;
    const { service, spy } = serviceWith(async (request) => {
      calls++;
      if (calls === 2) throw new PredictionError("network", "offline");
      return echoResponse(request);
    });
    const { result } = renderHook(() => useBatchAnalysis(service));
    const file = fileOf(BATCH_CHUNK_SIZE * 2 + 5);

    await act(async () => result.current.start({ file, domain: "hotels", modelType: "gas" }));
    expect(result.current.state).toMatchObject({ status: "stopped", done: BATCH_CHUNK_SIZE, total: 45 });

    await act(async () => result.current.resume());
    expect(result.current.state.status).toBe("done");
    const resumedIds = spy.mock.calls.slice(2).flatMap(([request]) => request.items.map((item) => item.id));
    expect(resumedIds).toHaveLength(25);
    expect(resumedIds[0]).toBe(`row-${BATCH_CHUNK_SIZE + 1}`);
  });

  it("uses predictions from the file without calling the service", async () => {
    const { service, spy } = serviceWith(async (request) => echoResponse(request));
    const { result } = renderHook(() => useBatchAnalysis(service));
    const file = parseReviewFile("p.jsonl", JSON.stringify({ text: "Dirty room.", quadruples: [quad] }));

    await act(async () => result.current.start({ file, domain: "hotels", modelType: "gas" }));

    expect(spy).not.toHaveBeenCalled();
    expect(result.current.state).toMatchObject({ status: "done", rows: [{ quadruples: [quad] }] });
  });

  it("cancels the in-flight chunk", async () => {
    let signal: AbortSignal | undefined;
    const { service } = serviceWith(
      (_request: BatchPredictRequest, options: PredictOptions = {}) =>
        new Promise((_resolve, reject) => {
          signal = options.signal;
          signal?.addEventListener("abort", () => reject(new PredictionError("aborted", "cancelled")));
        }),
    );
    const { result } = renderHook(() => useBatchAnalysis(service));

    act(() => result.current.start({ file: fileOf(3), domain: "hotels", modelType: "gas" }));
    expect(result.current.state.status).toBe("running");
    await act(async () => result.current.cancel());

    expect(signal?.aborted).toBe(true);
    expect(result.current.state).toMatchObject({ status: "stopped", done: 0, error: { kind: "aborted" } });
  });
});
