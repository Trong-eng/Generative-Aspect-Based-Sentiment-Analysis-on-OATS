import type { PredictRequest, PredictResponse } from "./types";

export interface PredictOptions {
  /** Aborting cancels the request; the service then rejects with kind `aborted`. */
  signal?: AbortSignal;
}

/**
 * Boundary between the UI and whatever produces predictions. The HTTP and mock
 * services both implement it, so components never know which one is active.
 */
export interface PredictionService {
  readonly mode: "mock" | "api";
  predict(request: PredictRequest, options?: PredictOptions): Promise<PredictResponse>;
}
