import {
  PredictionError,
  type BatchPredictRequest,
  type BatchPredictResponse,
  type BatchResult,
  type InvalidPrediction,
  type PredictRequest,
  type PredictResponse,
  type Quadruple,
} from "./types";

const QUADRUPLE_FIELDS = ["target", "aspect", "opinion", "sentiment"] as const;
const MAX_SERVER_MESSAGE_LENGTH = 300;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function malformed(detail: string): PredictionError {
  return new PredictionError("malformed", `Unexpected response from the server (${detail}).`);
}

function parseQuadruple(value: unknown, index: number): Quadruple {
  if (!isRecord(value)) {
    throw malformed(`quadruple ${index} is not an object`);
  }
  for (const field of QUADRUPLE_FIELDS) {
    if (typeof value[field] !== "string") {
      throw malformed(`quadruple ${index} has no string '${field}'`);
    }
  }
  return {
    target: value.target as string,
    aspect: value.aspect as string,
    opinion: value.opinion as string,
    sentiment: value.sentiment as string,
  };
}

/**
 * Validates a success body. Required: `quadruples` as a list of four-string records.
 * Optional diagnostics are kept only when well-typed so a sloppy backend cannot crash the UI.
 */
export function parsePredictResponse(body: unknown, request: PredictRequest): PredictResponse {
  if (!isRecord(body)) {
    throw malformed("body is not a JSON object");
  }
  if (!Array.isArray(body.quadruples)) {
    throw malformed("'quadruples' is missing or not a list");
  }
  const response: PredictResponse = {
    text: typeof body.text === "string" ? body.text : request.text,
    domain: typeof body.domain === "string" ? body.domain : request.domain,
    model_type: typeof body.model_type === "string" ? body.model_type : request.model_type,
    quadruples: body.quadruples.map(parseQuadruple),
  };
  if (typeof body.raw_generation === "string") {
    response.raw_generation = body.raw_generation;
  }
  if (Array.isArray(body.invalid_predictions)) {
    response.invalid_predictions = body.invalid_predictions.filter(isRecord) as InvalidPrediction[];
  }
  return response;
}

/**
 * Validates a batch body: every requested id must come back exactly once with valid
 * quadruples. Unknown ids are rejected so results can never attach to the wrong review.
 */
export function parseBatchResponse(body: unknown, request: BatchPredictRequest): BatchPredictResponse {
  if (!isRecord(body) || !Array.isArray(body.results)) {
    throw malformed("'results' is missing or not a list");
  }
  const expected = new Set(request.items.map((item) => item.id));
  const seen = new Set<string>();
  const results: BatchResult[] = body.results.map((value, index) => {
    if (!isRecord(value) || typeof value.id !== "string") {
      throw malformed(`result ${index} has no string 'id'`);
    }
    if (!expected.has(value.id) || seen.has(value.id)) {
      throw malformed(`result ${index} has an unknown or duplicate id`);
    }
    seen.add(value.id);
    if (!Array.isArray(value.quadruples)) {
      throw malformed(`result ${index} has no 'quadruples' list`);
    }
    const result: BatchResult = { id: value.id, quadruples: value.quadruples.map(parseQuadruple) };
    if (Array.isArray(value.invalid_predictions)) {
      result.invalid_predictions = value.invalid_predictions.filter(isRecord) as InvalidPrediction[];
    }
    return result;
  });
  if (seen.size !== expected.size) {
    throw malformed(`${expected.size - seen.size} requested item(s) are missing from 'results'`);
  }
  return { results };
}

/**
 * Reads `{error: {code, message}}` from an error body. Anything else (HTML error pages,
 * stack traces) is discarded so internal details never reach the screen.
 */
export function parseErrorBody(body: unknown): { code?: string; message?: string } {
  if (!isRecord(body) || !isRecord(body.error)) {
    return {};
  }
  const { code, message } = body.error;
  return {
    code: typeof code === "string" && code.trim() ? code.trim() : undefined,
    message:
      typeof message === "string" && message.trim()
        ? message.trim().slice(0, MAX_SERVER_MESSAGE_LENGTH)
        : undefined,
  };
}
