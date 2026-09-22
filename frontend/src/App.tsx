import type { PredictionService } from "./api";
import { PredictForm } from "./components/PredictForm";
import { ResultsPanel } from "./components/ResultsPanel";
import { usePrediction } from "./hooks/usePrediction";

export function App({ service }: { service: PredictionService }) {
  const { state, run, retry } = usePrediction(service);

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>OATS ASQP Analyzer</h1>
          <p className="muted">Extract (target, aspect, opinion, sentiment) quadruples from a review sentence.</p>
        </div>
        {service.mode === "mock" ? (
          <span className="mode-badge mode-mock">Mock mode — sample outputs, not model predictions</span>
        ) : (
          <span className="mode-badge mode-api">API mode</span>
        )}
      </header>
      <main className="layout">
        <PredictForm busy={state.status === "loading"} onSubmit={(request) => void run(request)} />
        <ResultsPanel state={state} onRetry={retry} />
      </main>
      <footer className="app-footer muted">
        Research prototype. Model quality has not been evaluated here; no accuracy figures are shown.
      </footer>
    </div>
  );
}
