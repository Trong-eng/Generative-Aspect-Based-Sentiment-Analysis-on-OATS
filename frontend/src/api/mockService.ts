import { MOCK_FIXTURES, MOCK_TRIGGERS } from "./fixtures";
import type { PredictOptions, PredictionService } from "./service";
import { PredictionError, type PredictRequest, type PredictResponse } from "./types";

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

/** In-browser stand-in for `POST /predict` that returns the same response type as the HTTP service. */
export function createMockPredictionService(options: MockServiceOptions): PredictionService {
  async function predict(request: PredictRequest, { signal }: PredictOptions = {}): Promise<PredictResponse> {
    await wait(options.delayMs, signal);
    const lowered = request.text.toLowerCase();
    if (lowered.includes(MOCK_TRIGGERS.error)) {
      throw new PredictionError("server", "Checkpoint cho domain/model này chưa sẵn sàng.", {
        code: "MODEL_NOT_READY",
        status: 503,
      });
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

  return { mode: "mock", predict };
}
