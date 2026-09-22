/** Wire types for the planned `POST /predict` contract (see frontend/README.md). */

export const DOMAINS = ["amazon_ff", "coursera", "hotels"] as const;
export type Domain = (typeof DOMAINS)[number];

export const MODEL_TYPES = ["gas", "paraphrase"] as const;
export type ModelType = (typeof MODEL_TYPES)[number];

export const DOMAIN_LABELS: Record<Domain, string> = {
  amazon_ff: "Amazon_FF",
  coursera: "Coursera",
  hotels: "Hotels",
};

export const MODEL_LABELS: Record<ModelType, string> = {
  gas: "GAS-T5",
  paraphrase: "Paraphrase-T5",
};

/** One ASQP quadruple exactly as the backend returns it; values are never rewritten. */
export interface Quadruple {
  target: string;
  aspect: string;
  opinion: string;
  sentiment: string;
}

/** A generated fragment the backend parser rejected, e.g. `{raw, reason}` from capstone/models. */
export type InvalidPrediction = Record<string, unknown>;

export interface PredictRequest {
  text: string;
  domain: Domain;
  model_type: ModelType;
}

export interface PredictResponse {
  text: string;
  domain: string;
  model_type: string;
  quadruples: Quadruple[];
  /** Optional diagnostics; absent when the backend omits them. */
  raw_generation?: string;
  invalid_predictions?: InvalidPrediction[];
}

export type PredictionErrorKind = "timeout" | "network" | "server" | "malformed" | "aborted";

/** The only error type services throw, so the UI never sees raw exceptions or stack traces. */
export class PredictionError extends Error {
  readonly kind: PredictionErrorKind;
  readonly code?: string;
  readonly status?: number;

  constructor(kind: PredictionErrorKind, message: string, options: { code?: string; status?: number } = {}) {
    super(message);
    this.name = "PredictionError";
    this.kind = kind;
    this.code = options.code;
    this.status = options.status;
  }
}
