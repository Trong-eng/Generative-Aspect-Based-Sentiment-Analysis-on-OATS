import { useId, useMemo, useState } from "react";
import { DOMAIN_LABELS, MODEL_LABELS, type Domain, type ModelType } from "../api";
import {
  aggregateByReview,
  aspectsToCsv,
  REVIEW_SENTIMENTS,
  sortAspects,
  type AnalyzedRow,
  type AspectStat,
  type GroupBy,
  type SortKey,
} from "../batch/aggregate";
import { percent, SENTIMENT_LABELS } from "../batch/format";
import { AspectDetail } from "./AspectDetail";

interface AspectDashboardProps {
  rows: AnalyzedRow[];
  domain: Domain;
  modelType: ModelType;
  /** Where the quadruples came from: the mock rules, the API model, or the uploaded file. */
  source: "mock" | "api" | "file";
}

function downloadCsv(content: string, fileName: string) {
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

function SourceNote({ source, domain, modelType }: Pick<AspectDashboardProps, "source" | "domain" | "modelType">) {
  if (source === "mock") {
    return (
      <p className="notice notice-mock">
        Mock mode: aspects come from simple keyword rules, not the ASQP model. Use it to try the dashboard only.
      </p>
    );
  }
  if (source === "file") {
    return <p className="notice">Using the predictions included in the uploaded file ({DOMAIN_LABELS[domain]}).</p>;
  }
  return (
    <p className="notice">
      Estimated by {MODEL_LABELS[modelType]} on {DOMAIN_LABELS[domain]}. Model accuracy has not been evaluated in this
      app, so treat the percentages as estimates.
    </p>
  );
}

function DistributionBar({ stat }: { stat: AspectStat }) {
  return (
    <div className="dist-bar" aria-hidden="true">
      {REVIEW_SENTIMENTS.map((sentiment) =>
        stat.counts[sentiment] > 0 ? (
          <span key={sentiment} className={`dist-${sentiment}`} style={{ width: `${stat.shares[sentiment] * 100}%` }} />
        ) : null,
      )}
    </div>
  );
}

export function AspectDashboard({ rows, domain, modelType, source }: AspectDashboardProps) {
  const [groupBy, setGroupBy] = useState<GroupBy>("category");
  const [sortKey, setSortKey] = useState<SortKey>("negative_share");
  const [minReviewsInput, setMinReviewsInput] = useState("3");
  const minReviews = Math.max(1, Math.floor(Number(minReviewsInput)) || 1);
  const [selected, setSelected] = useState<string | null>(null);
  const ids = { group: useId(), sort: useId(), min: useId() };

  const data = useMemo(() => aggregateByReview(rows, groupBy), [rows, groupBy]);
  const visible = useMemo(
    () => sortAspects(data.aspects.filter((stat) => stat.reviews >= minReviews), sortKey),
    [data, minReviews, sortKey],
  );
  const hidden = data.aspects.length - visible.length;
  const weakest = sortAspects(
    visible.filter((stat) => stat.counts.negative > 0),
    "negative_share",
  )[0];
  const selectedStat = data.aspects.find((stat) => stat.aspect === selected) ?? null;

  return (
    <div className="dashboard">
      <SourceNote source={source} domain={domain} modelType={modelType} />

      <dl className="summary-grid">
        <div>
          <dt>Reviews</dt>
          <dd>{data.reviewCount}</dd>
        </div>
        <div>
          <dt>Reviews with an aspect</dt>
          <dd>{data.reviewsWithAspects}</dd>
        </div>
        <div>
          <dt>Aspects shown</dt>
          <dd>{visible.length}</dd>
        </div>
      </dl>

      {weakest ? (
        <div className="callout" data-testid="weakest-aspect">
          <h3>Most negative aspect: {weakest.aspect}</h3>
          <p>
            {percent(weakest.shares.negative)} of the {weakest.reviews} reviews that mention it are negative (
            {weakest.counts.negative} review(s)).
            {weakest.opinions.negative.length > 0 && (
              <>
                {" "}
                Most common complaints:{" "}
                {weakest.opinions.negative
                  .slice(0, 3)
                  .map((item) => `“${item.opinion}” (${item.reviews})`)
                  .join(", ")}
                .
              </>
            )}
          </p>
          <button type="button" className="secondary" onClick={() => setSelected(weakest.aspect)}>
            See reviews for {weakest.aspect}
          </button>
        </div>
      ) : (
        data.aspects.length > 0 && <p className="muted">No aspect with negative reviews among those shown.</p>
      )}

      <div className="controls">
        <div className="field">
          <label htmlFor={ids.group}>Group by</label>
          <select
            id={ids.group}
            value={groupBy}
            onChange={(event) => {
              setGroupBy(event.target.value as GroupBy);
              setSelected(null);
            }}
          >
            <option value="category">Aspect category (e.g. rooms cleanliness)</option>
            <option value="entity">Entity (e.g. rooms)</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor={ids.sort}>Sort by</label>
          <select id={ids.sort} value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)}>
            <option value="negative_share">% negative</option>
            <option value="negative_count">Negative reviews</option>
            <option value="reviews">Reviews mentioning</option>
            <option value="aspect">Name</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor={ids.min}>Minimum reviews</label>
          <input
            id={ids.min}
            type="number"
            min={1}
            value={minReviewsInput}
            onChange={(event) => setMinReviewsInput(event.target.value)}
          />
        </div>
        <button
          type="button"
          className="secondary"
          disabled={visible.length === 0}
          onClick={() => downloadCsv(aspectsToCsv(visible), `aspect_sentiment_${domain}.csv`)}
        >
          Export CSV
        </button>
      </div>

      {data.aspects.length === 0 ? (
        <div className="empty-state">
          <h3>No aspects found</h3>
          <p>No aspect–sentiment pair was extracted from these reviews.</p>
        </div>
      ) : (
        <>
          {hidden > 0 && (
            <p className="muted">
              {hidden} aspect(s) with fewer than {minReviews} reviews are hidden to avoid conclusions from tiny samples.
            </p>
          )}
          <div className="table-wrap">
            <table className="aspect-table">
              <caption className="visually-hidden">Review sentiment per aspect</caption>
              <thead>
                <tr>
                  <th scope="col">Aspect</th>
                  <th scope="col">Reviews</th>
                  {REVIEW_SENTIMENTS.map((sentiment) => (
                    <th scope="col" key={sentiment}>
                      {SENTIMENT_LABELS[sentiment]}
                    </th>
                  ))}
                  <th scope="col">Distribution</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((stat) => (
                  <tr key={stat.aspect} className={stat.aspect === selected ? "selected" : undefined}>
                    <th scope="row">
                      <button
                        type="button"
                        className="link-button"
                        aria-expanded={stat.aspect === selected}
                        onClick={() => setSelected(stat.aspect === selected ? null : stat.aspect)}
                      >
                        {stat.aspect}
                      </button>
                    </th>
                    <td>{stat.reviews}</td>
                    {REVIEW_SENTIMENTS.map((sentiment) => (
                      <td key={sentiment} className={`cell-${sentiment}`}>
                        {percent(stat.shares[sentiment])} <span className="muted">({stat.counts[sentiment]})</span>
                      </td>
                    ))}
                    <td>
                      <DistributionBar stat={stat} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details className="method">
            <summary>How are reviews counted?</summary>
            <ul>
              <li>Each review counts at most once per aspect; percentages use the reviews that mention the aspect.</li>
              <li>
                A review that is both positive and negative about an aspect, or that the model labels{" "}
                <code>conflict</code>, counts as <strong>Conflict</strong>.
              </li>
              <li>A positive or negative mention outweighs a neutral mention of the same aspect in the same review.</li>
              <li>Rows with the same <code>review_id</code> are combined into one review.</li>
              {data.unknownSentimentQuadruples > 0 && (
                <li>{data.unknownSentimentQuadruples} quadruple(s) with an unknown sentiment label were ignored.</li>
              )}
            </ul>
          </details>
        </>
      )}

      {selectedStat && (
        <AspectDetail key={selectedStat.aspect} stat={selectedStat} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
