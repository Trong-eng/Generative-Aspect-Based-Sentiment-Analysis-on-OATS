import { afterEach, describe, expect, it, vi } from "vitest";
import { createHttpPredictionService, predictUrl } from "./httpService";
import { PredictionError, type PredictRequest } from "./types";

const request: PredictRequest = { text: "The room was spotless.", domain: "hotels", model_type: "paraphrase" };
const successBody = {
  ...request,
  quadruples: [{ target: "room", aspect: "room cleanliness", opinion: "spotless", sentiment: "positive" }],
  invalid_predictions: [],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** A fetch that never settles until its signal aborts, like a hung server. */
function hangingFetch(): typeof fetch {
  return vi.fn((_url: RequestInfo | URL, init?: RequestInit) => {
    return new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });
  }) as unknown as typeof fetch;
}

async function rejection(promise: Promise<unknown>): Promise<PredictionError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(PredictionError);
    return error as PredictionError;
  }
  throw new Error("expected the promise to reject");
}

afterEach(() => {
  vi.useRealTimers();
});

describe("predictUrl", () => {
  it("builds relative and absolute URLs without double slashes", () => {
    expect(predictUrl("")).toBe("/predict");
    expect(predictUrl("https://api.example.test/")).toBe("https://api.example.test/predict");
    expect(predictUrl("https://api.example.test/v1")).toBe("https://api.example.test/v1/predict");
  });
});

describe("createHttpPredictionService", () => {
  it("POSTs text, domain and model_type as JSON to /predict", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(successBody));
    const service = createHttpPredictionService({ baseUrl: "https://api.example.test", timeoutMs: 1000, fetchImpl });

    const response = await service.predict({ text: "Great course.", domain: "coursera", model_type: "gas" });

    expect(service.mode).toBe("api");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://api.example.test/predict");
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(JSON.parse(init.body as string)).toEqual({ text: "Great course.", domain: "coursera", model_type: "gas" });
    expect(response.quadruples).toEqual(successBody.quadruples);
  });

  it("maps a contract error body to a server error with code and message", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({ error: { code: "MODEL_NOT_READY", message: "Checkpoint cho domain/model này chưa sẵn sàng." } }, 503),
    );
    const service = createHttpPredictionService({ baseUrl: "", timeoutMs: 1000, fetchImpl });

    const error = await rejection(service.predict(request));

    expect(error.kind).toBe("server");
    expect(error.code).toBe("MODEL_NOT_READY");
    expect(error.status).toBe(503);
    expect(error.message).toBe("Checkpoint cho domain/model này chưa sẵn sàng.");
  });

  it("does not leak non-contract server bodies such as stack traces", async () => {
    const fetchImpl = vi.fn(
      async () => new Response("Traceback (most recent call last):\n  File app.py", { status: 500 }),
    );
    const service = createHttpPredictionService({ baseUrl: "", timeoutMs: 1000, fetchImpl });

    const error = await rejection(service.predict(request));

    expect(error.kind).toBe("server");
    expect(error.message).toBe("The server returned an error (HTTP 500).");
    expect(error.message).not.toContain("Traceback");
  });

  it("reports network failures", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const service = createHttpPredictionService({ baseUrl: "", timeoutMs: 1000, fetchImpl });

    expect((await rejection(service.predict(request))).kind).toBe("network");
  });

  it("times out a hung request", async () => {
    vi.useFakeTimers();
    const service = createHttpPredictionService({ baseUrl: "", timeoutMs: 5000, fetchImpl: hangingFetch() });

    const pending = rejection(service.predict(request));
    await vi.advanceTimersByTimeAsync(5000);

    expect((await pending).kind).toBe("timeout");
  });

  it("rejects as aborted when the caller cancels", async () => {
    const controller = new AbortController();
    const service = createHttpPredictionService({ baseUrl: "", timeoutMs: 5000, fetchImpl: hangingFetch() });

    const pending = rejection(service.predict(request, { signal: controller.signal }));
    controller.abort();

    expect((await pending).kind).toBe("aborted");
  });

  it("rejects success responses that are not JSON or break the schema", async () => {
    const notJson = createHttpPredictionService({
      baseUrl: "",
      timeoutMs: 1000,
      fetchImpl: vi.fn(async () => new Response("<html>ok</html>", { status: 200 })),
    });
    const wrongShape = createHttpPredictionService({
      baseUrl: "",
      timeoutMs: 1000,
      fetchImpl: vi.fn(async () => jsonResponse({ ...request, quadruples: [{ target: 1 }] })),
    });

    expect((await rejection(notJson.predict(request))).kind).toBe("malformed");
    expect((await rejection(wrongShape.predict(request))).kind).toBe("malformed");
  });
});
