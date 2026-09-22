import { createHttpPredictionService } from "./httpService";
import { createMockPredictionService } from "./mockService";
import type { PredictionService } from "./service";

export interface AppEnv {
  VITE_USE_MOCK_API?: string;
  VITE_API_BASE_URL?: string;
  VITE_API_TIMEOUT_MS?: string;
  VITE_MOCK_DELAY_MS?: string;
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MOCK_DELAY_MS = 600;

function positiveNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return value !== undefined && value.trim() !== "" && Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

/** Mock mode is on unless VITE_USE_MOCK_API is explicitly "false" (or "0"). */
export function isMockEnabled(env: AppEnv): boolean {
  const flag = env.VITE_USE_MOCK_API?.trim().toLowerCase();
  return flag !== "false" && flag !== "0";
}

/** Picks the service from configuration; components only ever see `PredictionService`. */
export function createPredictionService(env: AppEnv): PredictionService {
  if (isMockEnabled(env)) {
    return createMockPredictionService({ delayMs: positiveNumber(env.VITE_MOCK_DELAY_MS, DEFAULT_MOCK_DELAY_MS) });
  }
  return createHttpPredictionService({
    baseUrl: env.VITE_API_BASE_URL ?? "",
    timeoutMs: positiveNumber(env.VITE_API_TIMEOUT_MS, DEFAULT_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS,
  });
}

export type { PredictionService, PredictOptions } from "./service";
export * from "./types";
