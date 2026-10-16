import { useCallback, useEffect, useRef, useState } from "react";
import { PredictionError, type Domain, type ModelType, type PredictionService, type Quadruple } from "../api";
import type { AnalyzedRow } from "../batch/aggregate";
import type { ParsedReviewFile } from "../batch/parseReviewFile";

/** Rows per `/predict/batch` call. Small enough for a responsive progress bar and modest request bodies. */
export const BATCH_CHUNK_SIZE = 20;

export interface BatchJob {
  file: ParsedReviewFile;
  domain: Domain;
  modelType: ModelType;
}

export type BatchState =
  | { status: "idle" }
  | { status: "running"; job: BatchJob; done: number; total: number }
  | { status: "stopped"; job: BatchJob; done: number; total: number; error: PredictionError }
  | { status: "done"; job: BatchJob; rows: AnalyzedRow[] };

function asPredictionError(error: unknown): PredictionError {
  if (error instanceof PredictionError) return error;
  // The UI shows a generic message; keep the real cause in the console for debugging.
  console.error("Unexpected error during batch analysis", error);
  return new PredictionError("malformed", "Something went wrong while reading the predictions.");
}

/**
 * Sends a parsed file to `predictBatch` chunk by chunk. Finished chunks are kept, so after
 * an error or a cancel `resume()` continues with the remaining rows instead of starting over.
 */
export function useBatchAnalysis(service: PredictionService) {
  const [state, setState] = useState<BatchState>({ status: "idle" });
  const resultsRef = useRef(new Map<string, Quadruple[]>());
  const controllerRef = useRef<AbortController | null>(null);
  const runIdRef = useRef(0);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const process = useCallback(
    async (job: BatchJob) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const runId = ++runIdRef.current;
      const results = resultsRef.current;
      const { rows } = job.file;
      const total = rows.length;
      const pending = rows.filter((row) => !results.has(row.rowId));

      const finish = () =>
        setState({
          status: "done",
          job,
          rows: rows.map((row) => ({ reviewId: row.reviewId, text: row.text, quadruples: results.get(row.rowId) ?? [] })),
        });

      if (job.file.precomputed) {
        for (const row of rows) results.set(row.rowId, row.quadruples ?? []);
        finish();
        return;
      }

      setState({ status: "running", job, done: total - pending.length, total });
      try {
        for (let start = 0; start < pending.length; start += BATCH_CHUNK_SIZE) {
          const chunk = pending.slice(start, start + BATCH_CHUNK_SIZE);
          const response = await service.predictBatch(
            {
              domain: job.domain,
              model_type: job.modelType,
              items: chunk.map((row) => ({ id: row.rowId, text: row.text })),
            },
            { signal: controller.signal },
          );
          if (runId !== runIdRef.current) return;
          for (const result of response.results) results.set(result.id, result.quadruples);
          setState({ status: "running", job, done: total - pending.length + start + chunk.length, total });
        }
        if (runId === runIdRef.current) finish();
      } catch (error) {
        if (runId === runIdRef.current) {
          setState({ status: "stopped", job, done: results.size, total, error: asPredictionError(error) });
        }
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
      }
    },
    [service],
  );

  const start = useCallback(
    (job: BatchJob) => {
      resultsRef.current = new Map();
      void process(job);
    },
    [process],
  );

  const resume = useCallback(() => {
    if (state.status === "stopped") void process(state.job);
  }, [process, state]);

  const cancel = useCallback(() => controllerRef.current?.abort(), []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    runIdRef.current++;
    resultsRef.current = new Map();
    setState({ status: "idle" });
  }, []);

  return { state, start, resume, cancel, reset };
}
