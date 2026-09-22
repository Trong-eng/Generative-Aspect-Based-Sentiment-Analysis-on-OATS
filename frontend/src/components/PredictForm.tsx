import { useId, useState, type FormEvent, type KeyboardEvent } from "react";
import type { Domain, ModelType, PredictRequest } from "../api";
import { DomainModelFields } from "./DomainModelFields";

interface PredictFormProps {
  busy: boolean;
  onSubmit: (request: PredictRequest) => void;
}

export function PredictForm({ busy, onSubmit }: PredictFormProps) {
  const [text, setText] = useState("");
  const [domain, setDomain] = useState<Domain>("hotels");
  const [modelType, setModelType] = useState<ModelType>("paraphrase");
  const [validationError, setValidationError] = useState<string | null>(null);
  const ids = { text: useId(), hint: useId(), error: useId() };

  function submit() {
    if (busy) {
      return;
    }
    if (!text.trim()) {
      setValidationError("Enter a sentence to analyze — the input is empty or only whitespace.");
      return;
    }
    setValidationError(null);
    // The text is sent as typed; sentence splitting/preprocessing is the backend's job.
    onSubmit({ text, domain, model_type: modelType });
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      submit();
    }
  }

  const describedBy = validationError ? `${ids.hint} ${ids.error}` : ids.hint;

  return (
    <form className="panel predict-form" onSubmit={handleSubmit} noValidate aria-label="Analyze a sentence">
      <div className="field">
        <label htmlFor={ids.text}>Sentence</label>
        <textarea
          id={ids.text}
          name="text"
          rows={4}
          value={text}
          placeholder="The room was spotless."
          aria-invalid={validationError ? true : undefined}
          aria-describedby={describedBy}
          onChange={(event) => {
            setText(event.target.value);
            if (validationError && event.target.value.trim()) {
              setValidationError(null);
            }
          }}
          onKeyDown={handleKeyDown}
        />
        <p id={ids.hint} className="hint">
          The models were trained on single sentences. Text is sent unchanged; multi-sentence reviews are not
          officially supported yet. Press Ctrl+Enter to submit.
        </p>
        {validationError && (
          <p id={ids.error} className="field-error" role="alert">
            {validationError}
          </p>
        )}
      </div>

      <DomainModelFields
        domain={domain}
        modelType={modelType}
        onDomainChange={setDomain}
        onModelTypeChange={setModelType}
      />

      <button type="submit" className="primary" disabled={busy} aria-busy={busy}>
        {busy ? "Analyzing…" : "Analyze"}
      </button>
    </form>
  );
}
