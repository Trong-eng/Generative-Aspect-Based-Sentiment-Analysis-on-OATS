import { DOMAIN_LABELS, MODEL_LABELS, type Domain, type ModelType, type PredictRequest, type PredictResponse } from "../api";
import type { PredictionState } from "../hooks/usePrediction";
import { QuadrupleCard } from "./QuadrupleCard";

function domainLabel(value: string): string {
  return DOMAIN_LABELS[value as Domain] ?? value;
}

function modelLabel(value: string): string {
  return MODEL_LABELS[value as ModelType] ?? value;
}

function describe(request: PredictRequest | PredictResponse): string {
  return `${modelLabel(request.model_type)} · ${domainLabel(request.domain)}`;
}

const ERROR_TITLES = {
  timeout: "The request timed out",
  network: "Could not connect",
  server: "The server could not complete the prediction",
  malformed: "Received an unexpected response",
  aborted: "The request was cancelled",
} as const;

function Diagnostics({ response }: { response: PredictResponse }) {
  const invalid = response.invalid_predictions ?? [];
  if (response.raw_generation === undefined && invalid.length === 0) {
    return null;
  }
  return (
    <details className="diagnostics">
      <summary>Diagnostics</summary>
      {response.raw_generation !== undefined && (
        <>
          <h3>Raw generation</h3>
          <pre>{response.raw_generation || "(empty)"}</pre>
        </>
      )}
      <h3>Invalid predictions ({invalid.length})</h3>
      {invalid.length > 0 ? (
        <ul className="invalid-list">
          {invalid.map((item, index) => (
            <li key={index}>
              <pre>{JSON.stringify(item)}</pre>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">None.</p>
      )}
    </details>
  );
}

export function ResultsPanel({ state, onRetry }: { state: PredictionState; onRetry: () => void }) {
  return (
    <section className="panel results" aria-labelledby="results-heading">
      <h2 id="results-heading">Results</h2>

      {/* Polite live region announces progress and outcomes; errors additionally use role=alert. */}
      <p className="visually-hidden" aria-live="polite" role="status">
        {state.status === "loading" && `Analyzing with ${describe(state.request)}.`}
        {state.status === "success" &&
          (state.response.quadruples.length === 0
            ? "Analysis finished. No quadruples found."
            : `Analysis finished. ${state.response.quadruples.length} quadruple(s) found.`)}
      </p>

      {state.status === "idle" && (
        <p className="muted">Enter a sentence, choose a domain and model, then press Analyze.</p>
      )}

      {state.status === "loading" && (
        <div className="loading" aria-hidden="true">
          <span className="spinner" /> Analyzing with {describe(state.request)}…
        </div>
      )}

      {state.status === "error" && (
        <div className="error-box" role="alert">
          <h3>{ERROR_TITLES[state.error.kind]}</h3>
          <p>{state.error.message}</p>
          {state.error.code && <p className="muted">Error code: {state.error.code}</p>}
          <button type="button" className="secondary" onClick={onRetry}>
            Try again
          </button>
        </div>
      )}

      {state.status === "success" && (
        <>
          <p className="result-meta">
            <span className="tag">{describe(state.response)}</span>
            <span className="result-text">“{state.response.text}”</span>
          </p>
          {state.response.quadruples.length === 0 ? (
            <div className="empty-state">
              <h3>No quadruples found</h3>
              <p>The model did not extract any aspect–opinion pair from this text.</p>
            </div>
          ) : (
            <>
              <p className="muted">{state.response.quadruples.length} quadruple(s)</p>
              <ul className="quad-list" aria-label="Predicted quadruples">
                {state.response.quadruples.map((quadruple, index) => (
                  <QuadrupleCard key={index} quadruple={quadruple} index={index} />
                ))}
              </ul>
            </>
          )}
          <Diagnostics response={state.response} />
        </>
      )}
    </section>
  );
}
