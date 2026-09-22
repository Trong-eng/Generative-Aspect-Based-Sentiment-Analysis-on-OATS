import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { App } from "./App";
import {
  PredictionError,
  type PredictOptions,
  type PredictRequest,
  type PredictResponse,
  type PredictionService,
  type Quadruple,
} from "./api";
import { createHttpPredictionService } from "./api/httpService";
import { createMockPredictionService } from "./api/mockService";

const roomQuad: Quadruple = { target: "room", aspect: "room cleanliness", opinion: "spotless", sentiment: "positive" };
const staffQuad: Quadruple = { target: "staff", aspect: "service general", opinion: "rude", sentiment: "negative" };

function respond(request: PredictRequest, quadruples: Quadruple[], extra: Partial<PredictResponse> = {}): PredictResponse {
  return { ...request, quadruples, ...extra };
}

/** A service whose calls stay pending until the test settles them. */
function controllableService() {
  const calls: {
    request: PredictRequest;
    signal?: AbortSignal;
    resolve: (value: PredictResponse) => void;
    reject: (error: unknown) => void;
  }[] = [];
  const predict = vi.fn((request: PredictRequest, options: PredictOptions = {}) => {
    return new Promise<PredictResponse>((resolve, reject) => {
      calls.push({ request, signal: options.signal, resolve, reject });
    });
  });
  const service: PredictionService = { mode: "api", predict, predictBatch: vi.fn() };
  return { service, predict, calls };
}

function staticService(result: (request: PredictRequest) => PredictResponse | Promise<PredictResponse>): PredictionService {
  return { mode: "api", predict: vi.fn(async (request: PredictRequest) => result(request)), predictBatch: vi.fn() };
}

async function analyze(text: string, options: { domain?: string; model?: string } = {}) {
  const user = userEvent.setup();
  const input = screen.getByLabelText("Sentence");
  await user.clear(input);
  if (text) {
    await user.type(input, text);
  }
  if (options.domain) {
    await user.selectOptions(panel().getByLabelText("Domain"), options.domain);
  }
  if (options.model) {
    await user.selectOptions(panel().getByLabelText("Model"), options.model);
  }
  await user.click(screen.getByRole("button", { name: "Analyze" }));
  return user;
}

/** The single-review tab; the hidden batch tab has its own Domain/Model fields. */
function panel() {
  return within(screen.getByRole("tabpanel", { name: "Single review" }));
}

function cards() {
  return within(screen.getByRole("list", { name: "Predicted quadruples" })).getAllByRole("listitem");
}

describe("App", () => {
  it("offers exactly three domains and two models with labelled controls", () => {
    render(<App service={createMockPredictionService({ delayMs: 0 })} />);
    const domainOptions = within(panel().getByLabelText("Domain")).getAllByRole("option");
    const modelOptions = within(panel().getByLabelText("Model")).getAllByRole("option");
    expect(domainOptions.map((option) => option.textContent)).toEqual(["Amazon_FF", "Coursera", "Hotels"]);
    expect(domainOptions.map((option) => (option as HTMLOptionElement).value)).toEqual(["amazon_ff", "coursera", "hotels"]);
    expect(modelOptions.map((option) => option.textContent)).toEqual(["GAS-T5", "Paraphrase-T5"]);
    expect(modelOptions.map((option) => (option as HTMLOptionElement).value)).toEqual(["gas", "paraphrase"]);
    expect(screen.getByText(/mock mode/i)).toBeInTheDocument();
  });

  it.each([
    ["empty", ""],
    ["whitespace only", "   "],
  ])("does not send a request for %s input", async (_label, text) => {
    const { service, predict } = controllableService();
    render(<App service={service} />);

    await analyze(text);

    expect(predict).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/empty or only whitespace/i);
    expect(screen.getByLabelText("Sentence")).toHaveAttribute("aria-invalid", "true");
  });

  it("sends the text with the selected domain and model", async () => {
    const { service, predict } = controllableService();
    render(<App service={service} />);

    await analyze("Great course.", { domain: "coursera", model: "gas" });

    expect(predict).toHaveBeenCalledTimes(1);
    expect(predict.mock.calls[0]?.[0]).toEqual({ text: "Great course.", domain: "coursera", model_type: "gas" });
  });

  it("shows loading, disables submit and ignores repeated submits", async () => {
    const { service, predict, calls } = controllableService();
    render(<App service={service} />);

    const user = await analyze("The room was spotless.");

    const button = screen.getByRole("button", { name: /analyzing/i });
    expect(button).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(/analyzing with paraphrase-t5 · hotels/i);
    await user.click(button);
    await user.type(screen.getByLabelText("Sentence"), "{Control>}{Enter}{/Control}");
    expect(predict).toHaveBeenCalledTimes(1);

    calls[0]!.resolve(respond(calls[0]!.request, [roomQuad]));
    expect(await screen.findByRole("button", { name: "Analyze" })).toBeEnabled();
  });

  it("renders a single quadruple with all four fields unchanged", async () => {
    render(<App service={staticService((request) => respond(request, [roomQuad]))} />);

    await analyze("The room was spotless.");

    const [card] = cards();
    expect(cards()).toHaveLength(1);
    const scoped = within(card!);
    expect(scoped.getByText("Target").nextElementSibling).toHaveTextContent("room");
    expect(scoped.getByText("Aspect").nextElementSibling).toHaveTextContent("room cleanliness");
    expect(scoped.getByText("Opinion").nextElementSibling).toHaveTextContent("spotless");
    expect(scoped.getByText("Sentiment").nextElementSibling).toHaveTextContent("positive");
    expect(screen.getByRole("status")).toHaveTextContent("1 quadruple(s) found");
  });

  it("renders multiple quadruples, with sentiment shown as text plus a symbol", async () => {
    render(<App service={staticService((request) => respond(request, [roomQuad, staffQuad]))} />);

    await analyze("Clean room, rude staff.");

    expect(cards()).toHaveLength(2);
    const negative = within(cards()[1]!).getByText("negative");
    expect(negative).toHaveAttribute("data-sentiment", "negative");
    expect(negative.querySelector(".sentiment-symbol")).toHaveTextContent("−");
  });

  it("shows an empty state when the response has no quadruples", async () => {
    render(<App service={staticService((request) => respond(request, []))} />);

    await analyze("I stayed here in May.");

    expect(await screen.findByRole("heading", { name: "No quadruples found" })).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Predicted quadruples" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("No quadruples found");
  });

  it("renders NULL target/opinion as implicit while keeping the raw value", async () => {
    const quad = { target: "NULL", aspect: "course general", opinion: "NULL", sentiment: "positive" };
    render(<App service={staticService((request) => respond(request, [quad]))} />);

    await analyze("Strongly recommended.");

    const implicit = within(cards()[0]!).getAllByText(/Implicit \/ Không nêu rõ/);
    expect(implicit).toHaveLength(2);
    expect(implicit[0]).toHaveAttribute("title", "Raw value: NULL");
    expect(implicit[0]).toHaveTextContent("(NULL)");
  });

  it("keeps diagnostics collapsed and tolerates their absence", async () => {
    const withDiagnostics = staticService((request) =>
      respond(request, [roomQuad], {
        raw_generation: "room cleanliness is great because room is spotless",
        invalid_predictions: [{ raw: "quiz is ok because", reason: "malformed_generation" }],
      }),
    );
    const { unmount } = render(<App service={withDiagnostics} />);
    await analyze("The room was spotless.");
    const details = screen.getByText("Diagnostics").closest("details")!;
    expect(details).not.toHaveAttribute("open");
    expect(within(details).getByText(/malformed_generation/)).toBeInTheDocument();
    unmount();

    render(<App service={staticService((request) => respond(request, [roomQuad]))} />);
    await analyze("The room was spotless.");
    expect(cards()).toHaveLength(1);
    expect(screen.queryByText("Diagnostics")).not.toBeInTheDocument();
  });

  it.each([
    ["timeout", new PredictionError("timeout", "The server took too long to respond."), "The request timed out"],
    ["network", new PredictionError("network", "Could not reach the prediction server."), "Could not connect"],
    [
      "server",
      new PredictionError("server", "Checkpoint cho domain/model này chưa sẵn sàng.", { code: "MODEL_NOT_READY" }),
      "The server could not complete the prediction",
    ],
  ])("shows a readable %s error and retries the same request", async (_kind, error, title) => {
    const { service, predict, calls } = controllableService();
    render(<App service={service} />);
    const user = await analyze("The room was spotless.", { domain: "amazon_ff", model: "gas" });

    calls[0]!.reject(error);

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByRole("heading", { name: title })).toBeInTheDocument();
    expect(alert).toHaveTextContent(error.message);
    if (error.code) {
      expect(alert).toHaveTextContent(`Error code: ${error.code}`);
    }

    await user.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(predict).toHaveBeenCalledTimes(2);
    expect(predict.mock.calls[1]?.[0]).toEqual(predict.mock.calls[0]?.[0]);
    calls[1]!.resolve(respond(calls[1]!.request, [roomQuad]));
    expect(await screen.findByRole("list", { name: "Predicted quadruples" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("does not crash or leak details on an unexpected exception", async () => {
    const service: PredictionService = {
      mode: "api",
      predict: vi.fn(async () => {
        throw new Error("KeyError at /srv/app/model.py line 42");
      }),
      predictBatch: vi.fn(),
    };
    render(<App service={service} />);

    await analyze("The room was spotless.");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Received an unexpected response");
    expect(alert).not.toHaveTextContent("model.py");
  });

  it("does not crash on a malformed API response", async () => {
    const fetchImpl = vi.fn(
      async () => new Response(JSON.stringify({ quadruples: [{ target: "room" }] }), { status: 200 }),
    );
    render(<App service={createHttpPredictionService({ baseUrl: "", timeoutMs: 1000, fetchImpl })} />);

    await analyze("The room was spotless.");

    expect(await screen.findByRole("alert")).toHaveTextContent("Received an unexpected response");
    expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
  });

  it("aborts the in-flight request on unmount", async () => {
    const { service, calls } = controllableService();
    const { unmount } = render(<App service={service} />);
    await analyze("First sentence.");
    expect(calls[0]!.signal?.aborted).toBe(false);
    unmount();
    expect(calls[0]!.signal?.aborted).toBe(true);
  });

  it.each([
    ["mock service", () => createMockPredictionService({ delayMs: 0 })],
    [
      "HTTP service",
      () =>
        createHttpPredictionService({
          baseUrl: "https://api.example.test",
          timeoutMs: 1000,
          fetchImpl: vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
            const body = JSON.parse(String(init?.body)) as PredictRequest;
            return new Response(JSON.stringify({ ...body, quadruples: [roomQuad] }), { status: 200 });
          }),
        }),
    ],
  ])("renders results from the %s through the same components", async (_label, makeService) => {
    render(<App service={makeService()} />);

    await analyze("The room was spotless.", { domain: "hotels", model: "paraphrase" });

    expect(cards()).toHaveLength(1);
    expect(within(cards()[0]!).getByText("room cleanliness")).toBeInTheDocument();
  });
});
