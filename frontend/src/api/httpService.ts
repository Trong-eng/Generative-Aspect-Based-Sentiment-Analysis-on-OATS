import type { PredictOptions, PredictionService } from "./service";
import { PredictionError, type BatchPredictRequest, type PredictRequest } from "./types";
import { parseBatchResponse, parseErrorBody, parsePredictResponse } from "./validate";

export interface HttpServiceOptions {
  /** Backend origin/prefix; empty means a relative `/predict` on the current origin. */
  baseUrl: string;
  timeoutMs: number;
  /** Timeout for one batch call; batches are slower than single predictions. Defaults to 4× `timeoutMs`. */
  batchTimeoutMs?: number;
  fetchImpl?: typeof fetch;
}

export function predictUrl(baseUrl: string, path = "/predict"): string {
  return `${baseUrl.trim().replace(/\/+$/, "")}${path}`;
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

  /** POSTs JSON with timeout + caller cancellation and maps every failure to a PredictionError. */
  async function postJson(url: string, payload: unknown, timeoutMs: number, signal?: AbortSignal): Promise<unknown> {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
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
          body: JSON.stringify(payload),
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
      return body;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", forwardAbort);
    }
  }

  return {
    mode: "api",
    async predict(request: PredictRequest, { signal }: PredictOptions = {}) {
      const payload = { text: request.text, domain: request.domain, model_type: request.model_type };
      const body = await postJson(predictUrl(options.baseUrl), payload, options.timeoutMs, signal);
      return parsePredictResponse(body, request);
    },
    async predictBatch(request: BatchPredictRequest, { signal }: PredictOptions = {}) {
      const payload = {
        domain: request.domain,
        model_type: request.model_type,
        items: request.items.map((item) => ({ id: item.id, text: item.text })),
      };
      const timeoutMs = options.batchTimeoutMs ?? options.timeoutMs * 4;
      const body = await postJson(predictUrl(options.baseUrl, "/predict/batch"), payload, timeoutMs, signal);
      return parseBatchResponse(body, request);
    },
  };
}
