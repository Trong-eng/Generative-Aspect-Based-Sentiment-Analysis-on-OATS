import { useId } from "react";
import { DOMAINS, DOMAIN_LABELS, MODEL_LABELS, MODEL_TYPES, type Domain, type ModelType } from "../api";

interface DomainModelFieldsProps {
  domain: Domain;
  modelType: ModelType;
  onDomainChange: (domain: Domain) => void;
  onModelTypeChange: (modelType: ModelType) => void;
  disabled?: boolean;
}

export function DomainModelFields({ domain, modelType, onDomainChange, onModelTypeChange, disabled }: DomainModelFieldsProps) {
  const ids = { domain: useId(), model: useId() };
  return (
    <div className="field-row">
      <div className="field">
        <label htmlFor={ids.domain}>Domain</label>
        <select
          id={ids.domain}
          name="domain"
          value={domain}
          disabled={disabled}
          onChange={(event) => onDomainChange(event.target.value as Domain)}
        >
          {DOMAINS.map((value) => (
            <option key={value} value={value}>
              {DOMAIN_LABELS[value]}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor={ids.model}>Model</label>
        <select
          id={ids.model}
          name="model_type"
          value={modelType}
          disabled={disabled}
          onChange={(event) => onModelTypeChange(event.target.value as ModelType)}
        >
          {MODEL_TYPES.map((value) => (
            <option key={value} value={value}>
              {MODEL_LABELS[value]}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
