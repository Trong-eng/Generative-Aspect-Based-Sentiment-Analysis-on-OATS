import { useEffect, useId, useRef, useState } from "react";
import { REVIEW_SENTIMENTS, type AspectStat, type OpinionCount, type ReviewSentiment } from "../batch/aggregate";
import { percent, SENTIMENT_LABELS } from "../batch/format";
import { IMPLICIT_LABEL, isImplicit } from "./QuadrupleCard";
import { SentimentBadge } from "./SentimentBadge";

const MAX_EXAMPLES = 10;

function OpinionList({ title, items }: { title: string; items: OpinionCount[] }) {
  return (
    <div>
      <h4>{title}</h4>
      {items.length === 0 ? (
        <p className="muted">None.</p>
      ) : (
        <ol className="opinion-list">
          {items.slice(0, 8).map((item) => (
            <li key={item.opinion}>
              “{item.opinion}” <span className="muted">— {item.reviews} review(s)</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function AspectDetail({ stat, onClose }: { stat: AspectStat; onClose: () => void }) {
  const [filter, setFilter] = useState<ReviewSentiment | "all">(stat.counts.negative > 0 ? "negative" : "all");
  const headingRef = useRef<HTMLHeadingElement>(null);
  const filterId = useId();

  // Remounted per aspect (keyed by the parent), so this runs whenever a new aspect opens.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  const examples = stat.examples.filter((example) => filter === "all" || example.sentiment === filter);

  return (
    <section className="aspect-detail" aria-labelledby="aspect-detail-heading">
      <div className="detail-header">
        <h3 id="aspect-detail-heading" ref={headingRef} tabIndex={-1}>
          {stat.aspect}
        </h3>
        <button type="button" className="secondary" onClick={onClose}>
          Close
        </button>
      </div>
      <p>
        {stat.reviews} review(s) mention this aspect:{" "}
        {REVIEW_SENTIMENTS.map((sentiment) => `${SENTIMENT_LABELS[sentiment]} ${percent(stat.shares[sentiment])}`).join(
          " · ",
        )}
      </p>

      <div className="opinion-columns">
        <OpinionList title="Top negative opinions" items={stat.opinions.negative} />
        <OpinionList title="Top positive opinions" items={stat.opinions.positive} />
      </div>

      <div className="field example-filter">
        <label htmlFor={filterId}>Show reviews</label>
        <select id={filterId} value={filter} onChange={(event) => setFilter(event.target.value as ReviewSentiment | "all")}>
          <option value="all">All ({stat.reviews})</option>
          {REVIEW_SENTIMENTS.map((sentiment) => (
            <option key={sentiment} value={sentiment}>
              {SENTIMENT_LABELS[sentiment]} ({stat.counts[sentiment]})
            </option>
          ))}
        </select>
      </div>

      {examples.length === 0 ? (
        <p className="muted">No reviews in this group.</p>
      ) : (
        <ul className="example-list" aria-label={`Reviews about ${stat.aspect}`}>
          {examples.slice(0, MAX_EXAMPLES).map((example) => (
            <li key={example.reviewId}>
              <p className="example-text">{example.text}</p>
              <p className="example-quads">
                <SentimentBadge sentiment={example.sentiment} />
                {example.quadruples.map((quad, index) => (
                  <span key={index} className="example-quad">
                    {isImplicit(quad.target) ? IMPLICIT_LABEL : quad.target} →{" "}
                    {isImplicit(quad.opinion) ? IMPLICIT_LABEL : `“${quad.opinion}”`} ({quad.sentiment})
                  </span>
                ))}
              </p>
            </li>
          ))}
        </ul>
      )}
      {examples.length > MAX_EXAMPLES && (
        <p className="muted">
          Showing {MAX_EXAMPLES} of {examples.length} reviews.
        </p>
      )}
    </section>
  );
}
