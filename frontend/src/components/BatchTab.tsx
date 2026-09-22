import { useId, useState, type FormEvent } from "react";
import type { Domain, ModelType, PredictionService } from "../api";
import { FileParseError, MAX_FILE_BYTES, MAX_ROWS, parseReviewFile, type ParsedReviewFile } from "../batch/parseReviewFile";
import { useBatchAnalysis } from "../hooks/useBatchAnalysis";
import { AspectDashboard } from "./AspectDashboard";
import { DomainModelFields } from "./DomainModelFields";

const SAMPLE_FILE = `${import.meta.env.BASE_URL}samples/hotels_reviews_sample.csv`;

async function readFileText(file: File): Promise<string> {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

export function BatchTab({ service }: { service: PredictionService }) {
  const { state, start, resume, cancel, reset } = useBatchAnalysis(service);
  const [domain, setDomain] = useState<Domain>("hotels");
  const [modelType, setModelType] = useState<ModelType>("paraphrase");
  const [parsed, setParsed] = useState<{ name: string; file: ParsedReviewFile } | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const ids = { file: useId(), hint: useId(), error: useId() };
  const running = state.status === "running";

  async function handleFile(file: File | undefined) {
    reset();
    setParsed(null);
    setFileError(null);
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      setFileError(`The file is larger than ${MAX_FILE_BYTES / 1024 / 1024} MB.`);
      return;
    }
    try {
      setParsed({ name: file.name, file: parseReviewFile(file.name, await readFileText(file)) });
    } catch (error) {
      setFileError(error instanceof FileParseError ? error.message : "The file could not be read.");
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (running) return;
    if (!parsed) {
      setFileError(fileError ?? "Choose a file with reviews first.");
      return;
    }
    start({ file: parsed.file, domain, modelType });
  }

  return (
    <div className="batch-layout">
      <form className="panel predict-form" onSubmit={handleSubmit} noValidate aria-label="Analyze a file of reviews">
        <div className="field">
          <label htmlFor={ids.file}>Reviews file</label>
          <input
            id={ids.file}
            type="file"
            accept=".csv,.txt,.jsonl,.json"
            disabled={running}
            aria-describedby={fileError ? `${ids.hint} ${ids.error}` : ids.hint}
            aria-invalid={fileError ? true : undefined}
            onChange={(event) => void handleFile(event.target.files?.[0])}
          />
          <div id={ids.hint} className="hint">
            <p>
              One review per row, up to {MAX_ROWS.toLocaleString()} rows. <strong>.csv</strong> with a{" "}
              <code>text</code> column, <strong>.txt</strong> with one review per line, or <strong>.jsonl</strong>/
              <strong>.json</strong> with <code>text</code>. An optional <code>review_id</code> groups several rows
              (e.g. sentences) into one review.
            </p>
            <p>
              Rows that already contain <code>quadruples</code> (e.g. <code>capstone.predict</code> output) are used
              as-is without calling the model. <a href={SAMPLE_FILE} download>Download a sample file</a>.
            </p>
          </div>
          {fileError && (
            <p id={ids.error} className="field-error" role="alert">
              {fileError}
            </p>
          )}
        </div>

        {parsed && (
          <p className="file-summary" data-testid="file-summary">
            <strong>{parsed.name}</strong>: {parsed.file.rows.length} row(s), {parsed.file.reviewCount} review(s)
            {parsed.file.precomputed && " — predictions included in the file"}
            {parsed.file.issues.length > 0 && `, ${parsed.file.issues.length} row(s) skipped`}
          </p>
        )}
        {parsed && parsed.file.issues.length > 0 && (
          <details className="issues">
            <summary>Skipped rows</summary>
            <ul>
              {parsed.file.issues.slice(0, 20).map((issue) => (
                <li key={issue.line}>
                  Line {issue.line}: {issue.message}
                </li>
              ))}
              {parsed.file.issues.length > 20 && <li>…and {parsed.file.issues.length - 20} more</li>}
            </ul>
          </details>
        )}

        <DomainModelFields
          domain={domain}
          modelType={modelType}
          onDomainChange={setDomain}
          onModelTypeChange={setModelType}
          disabled={running}
        />
        {parsed?.file.precomputed && (
          <p className="hint">Domain labels the report only; the model selection is ignored for pre-computed files.</p>
        )}

        <div className="button-row">
          <button type="submit" className="primary" disabled={running} aria-busy={running}>
            {running ? "Analyzing…" : "Analyze file"}
          </button>
          {running && (
            <button type="button" className="secondary" onClick={cancel}>
              Cancel
            </button>
          )}
        </div>
      </form>

      <section className="panel results" aria-labelledby="dashboard-heading">
        <h2 id="dashboard-heading">Aspect dashboard</h2>
        <p className="visually-hidden" aria-live="polite" role="status">
          {state.status === "running" && `Analyzing reviews: ${state.done} of ${state.total} rows done.`}
          {state.status === "done" && "Analysis finished. Dashboard updated."}
        </p>

        {state.status === "idle" && (
          <p className="muted">
            Upload a file of reviews to see, for each aspect, how many reviews are positive, negative, neutral or
            conflicting — and which aspects draw the most negative reviews.
          </p>
        )}

        {state.status === "running" && (
          <div className="progress-block">
            <progress
              max={state.total}
              value={state.done}
              aria-label="Analysis progress"
              aria-valuetext={`${state.done} of ${state.total} rows`}
            />
            <p className="muted">
              {state.done} / {state.total} rows analyzed…
            </p>
          </div>
        )}

        {state.status === "stopped" && (
          <div className="error-box" role="alert">
            <h3>{state.error.kind === "aborted" ? "Analysis cancelled" : "Analysis stopped"}</h3>
            {state.error.kind !== "aborted" && <p>{state.error.message}</p>}
            {state.error.code && <p className="muted">Error code: {state.error.code}</p>}
            <p className="muted">
              {state.done} of {state.total} rows were analyzed before stopping.
            </p>
            <div className="button-row">
              <button type="button" className="secondary" onClick={resume}>
                Resume
              </button>
              <button type="button" className="secondary" onClick={() => start(state.job)}>
                Start over
              </button>
            </div>
          </div>
        )}

        {state.status === "done" && (
          <AspectDashboard
            rows={state.rows}
            domain={state.job.domain}
            modelType={state.job.modelType}
            source={state.job.file.precomputed ? "file" : service.mode}
          />
        )}
      </section>
    </div>
  );
}
