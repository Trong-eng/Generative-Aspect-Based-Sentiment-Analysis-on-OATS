import { useCallback, useEffect, useRef, useState } from "react";
import { PredictionError, type PredictRequest, type PredictResponse, type PredictionService } from "../api";

export type PredictionState =
  | { status: "idle" }
  | { status: "loading"; request: PredictRequest }
  | { status: "success"; request: PredictRequest; response: PredictResponse }
  | { status: "error"; request: PredictRequest; error: PredictionError };

function toPredictionError(error: unknown): PredictionError {
  if (error instanceof PredictionError) {
    return error;
  }
  // Never surface arbitrary exception text (it may carry internals); use a generic message.
  return new PredictionError("malformed", "Something went wrong while reading the prediction.");
}

/**
 * Runs predictions one at a time. A new request (or unmount) aborts the previous one,
 * and results from superseded requests are ignored, so a slow old response can never
 * overwrite a newer one.
 */
export function usePrediction(service: PredictionService) {
  const [state, setState] = useState<PredictionState>({ status: "idle" });
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const run = useCallback(
    async (request: PredictRequest) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const requestId = ++requestIdRef.current;
      setState({ status: "loading", request });
      try {
        const response = await service.predict(request, { signal: controller.signal });
        if (requestId === requestIdRef.current) {
          setState({ status: "success", request, response });
        }
      } catch (error) {
        if (requestId === requestIdRef.current) {
          setState({ status: "error", request, error: toPredictionError(error) });
        }
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
      }
    },
    [service],
  );

  const retry = useCallback(() => {
    if (state.status === "error") {
      void run(state.request);
    }
  }, [run, state]);

  return { state, run, retry };
}
