import type { PredictOptions, PredictionService } from "./service";
import { PredictionError, type PredictRequest, type PredictResponse } from "./types";
import { parseErrorBody, parsePredictResponse } from "./validate";

export interface HttpServiceOptions {
  /** Backend origin/prefix; empty means a relative `/predict` on the current origin. */
  baseUrl: string;
  timeoutMs: number;
  fetchImpl?: typeof fetch;
}

export function predictUrl(baseUrl: string): string {
  return `${baseUrl.trim().replace(/\/+$/, "")}/predict`;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}

export function createHttpPredictionService(options: HttpServiceOptions): PredictionService {
  const fetchImpl = options.fetchImpl ?? ((...args: Parameters<typeof fetch>) => fetch(...args));
  const url = predictUrl(options.baseUrl);

  async function predict(request: PredictRequest, { signal }: PredictOptions = {}): Promise<PredictResponse> {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, options.timeoutMs);
    const forwardAbort = () => controller.abort();
    if (signal?.aborted) {
      controller.abort();
    } else {
      signal?.addEventListener("abort", forwardAbort, { once: true });
    }
    const cancelled = () =>
      timedOut
        ? new PredictionError("timeout", "The server took too long to respond.")
        : new PredictionError("aborted", "The request was cancelled.");

    try {
      let response: Response;
      try {
        response = await fetchImpl(url, {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify({ text: request.text, domain: request.domain, model_type: request.model_type }),
          signal: controller.signal,
        });
      } catch {
        if (controller.signal.aborted) {
          throw cancelled();
        }
        throw new PredictionError("network", "Could not reach the prediction server.");
      }

      const body = await readJson(response);
      if (controller.signal.aborted) {
        throw cancelled();
      }
      if (!response.ok) {
        const { code, message } = parseErrorBody(body);
        throw new PredictionError("server", message ?? `The server returned an error (HTTP ${response.status}).`, {
          code,
          status: response.status,
        });
      }
      if (body === undefined) {
        throw new PredictionError("malformed", "Unexpected response from the server (body is not valid JSON).");
      }
      return parsePredictResponse(body, request);
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", forwardAbort);
    }
  }

  return { mode: "api", predict };
}
