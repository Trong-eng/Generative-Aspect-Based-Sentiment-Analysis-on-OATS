import type { Quadruple } from "../api";
import { SentimentBadge } from "./SentimentBadge";

export const IMPLICIT_LABEL = "Implicit / Không nêu rõ";

export function isImplicit(value: string): boolean {
  return value.trim().toUpperCase() === "NULL";
}

/** Shows an implicit (NULL) element in friendly words while keeping the raw value inspectable. */
function ElementValue({ value }: { value: string }) {
  if (isImplicit(value)) {
    return (
      <span className="implicit" title={`Raw value: ${value}`}>
        {IMPLICIT_LABEL}
        <span className="raw-value"> ({value})</span>
      </span>
    );
  }
  return <span>{value}</span>;
}

export function QuadrupleCard({ quadruple, index }: { quadruple: Quadruple; index: number }) {
  return (
    <li className="quad-card">
      <h3 className="visually-hidden">Quadruple {index + 1}</h3>
      <dl>
        <div>
          <dt>Target</dt>
          <dd>
            <ElementValue value={quadruple.target} />
          </dd>
        </div>
        <div>
          <dt>Aspect</dt>
          <dd>{quadruple.aspect}</dd>
        </div>
        <div>
          <dt>Opinion</dt>
          <dd>
            <ElementValue value={quadruple.opinion} />
          </dd>
        </div>
        <div>
          <dt>Sentiment</dt>
          <dd>
            <SentimentBadge sentiment={quadruple.sentiment} />
          </dd>
        </div>
      </dl>
    </li>
  );
}
