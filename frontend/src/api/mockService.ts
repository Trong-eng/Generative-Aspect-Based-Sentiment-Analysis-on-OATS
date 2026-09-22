import { MOCK_FIXTURES, MOCK_TRIGGERS } from "./fixtures";
import { mockExtract } from "./mockLexicon";
import type { PredictOptions, PredictionService } from "./service";
import {
  PredictionError,
  type BatchPredictRequest,
  type BatchPredictResponse,
  type PredictRequest,
  type PredictResponse,
} from "./types";

export interface MockServiceOptions {
  delayMs: number;
}

function wait(delayMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const aborted = () => {
      clearTimeout(timer);
      reject(new PredictionError("aborted", "The request was cancelled."));
    };
    if (signal?.aborted) {
      aborted();
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", aborted);
      resolve();
    }, delayMs);
    signal?.addEventListener("abort", aborted, { once: true });
  });
}

function modelNotReady(): PredictionError {
  return new PredictionError("server", "Checkpoint cho domain/model này chưa sẵn sàng.", {
    code: "MODEL_NOT_READY",
    status: 503,
  });
}

/** In-browser stand-in for the prediction API that returns the same response types as the HTTP service. */
export function createMockPredictionService(options: MockServiceOptions): PredictionService {
  async function predict(request: PredictRequest, { signal }: PredictOptions = {}): Promise<PredictResponse> {
    await wait(options.delayMs, signal);
    const lowered = request.text.toLowerCase();
    if (lowered.includes(MOCK_TRIGGERS.error)) {
      throw modelNotReady();
    }
    const fixture = MOCK_FIXTURES[request.domain][request.model_type];
    return {
      text: request.text,
      domain: request.domain,
      model_type: request.model_type,
      quadruples: lowered.includes(MOCK_TRIGGERS.empty) ? [] : fixture.quadruples.map((quad) => ({ ...quad })),
      raw_generation: lowered.includes(MOCK_TRIGGERS.empty) ? "" : fixture.raw_generation,
      invalid_predictions: [...(fixture.invalid_predictions ?? [])],
    };
  }

  /**
   * Batch mock: keyword rules over each text (see mockLexicon.ts), so the dashboard reflects
   * the uploaded reviews. The model type is echoed only; both models share the same rules.
   */
  async function predictBatch(
    request: BatchPredictRequest,
    { signal }: PredictOptions = {},
  ): Promise<BatchPredictResponse> {
    await wait(Math.min(options.delayMs, 150), signal);
    if (request.items.some((item) => item.text.toLowerCase().includes(MOCK_TRIGGERS.error))) {
      throw modelNotReady();
    }
    return {
      results: request.items.map((item) => ({
        id: item.id,
        quadruples: mockExtract(item.text, request.domain),
        invalid_predictions: [],
      })),
    };
  }

  return { mode: "mock", predict, predictBatch };
}
