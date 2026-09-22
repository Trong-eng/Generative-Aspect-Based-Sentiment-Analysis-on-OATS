import { useRef, useState, type KeyboardEvent } from "react";
import type { PredictionService } from "./api";
import { BatchTab } from "./components/BatchTab";
import { PredictForm } from "./components/PredictForm";
import { ResultsPanel } from "./components/ResultsPanel";
import { usePrediction } from "./hooks/usePrediction";

const TABS = [
  { id: "single", label: "Single review" },
  { id: "batch", label: "Review file & dashboard" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function SingleReviewTab({ service }: { service: PredictionService }) {
  const { state, run, retry } = usePrediction(service);
  return (
    <div className="layout">
      <PredictForm busy={state.status === "loading"} onSubmit={(request) => void run(request)} />
      <ResultsPanel state={state} onRetry={retry} />
    </div>
  );
}

export function App({ service }: { service: PredictionService }) {
  const [active, setActive] = useState<TabId>("single");
  const tabRefs = useRef<Record<TabId, HTMLButtonElement | null>>({ single: null, batch: null });

  function handleTabKey(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const moves: Record<string, number> = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: TABS.length - 1 };
    if (!(event.key in moves)) return;
    event.preventDefault();
    const next = TABS[(moves[event.key]! + TABS.length) % TABS.length]!;
    setActive(next.id);
    tabRefs.current[next.id]?.focus();
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>OATS ASQP Analyzer</h1>
          <p className="muted">Extract (target, aspect, opinion, sentiment) quadruples from reviews.</p>
        </div>
        {service.mode === "mock" ? (
          <span className="mode-badge mode-mock">Mock mode — sample outputs, not model predictions</span>
        ) : (
          <span className="mode-badge mode-api">API mode</span>
        )}
      </header>

      <div className="tabs" role="tablist" aria-label="Analysis mode">
        {TABS.map((tab, index) => (
          <button
            key={tab.id}
            ref={(element) => {
              tabRefs.current[tab.id] = element;
            }}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={active === tab.id}
            aria-controls={`panel-${tab.id}`}
            tabIndex={active === tab.id ? 0 : -1}
            className="tab"
            onClick={() => setActive(tab.id)}
            onKeyDown={(event) => handleTabKey(event, index)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Both panels stay mounted so switching tabs keeps each tab's input and results. */}
      <main>
        <div role="tabpanel" id="panel-single" aria-labelledby="tab-single" hidden={active !== "single"}>
          <SingleReviewTab service={service} />
        </div>
        <div role="tabpanel" id="panel-batch" aria-labelledby="tab-batch" hidden={active !== "batch"}>
          <BatchTab service={service} />
        </div>
      </main>

      <footer className="app-footer muted">
        Research prototype. Model quality has not been evaluated here; no accuracy figures are shown.
      </footer>
    </div>
  );
}
