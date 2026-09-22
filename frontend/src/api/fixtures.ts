import type { Domain, ModelType, PredictResponse } from "./types";

type Fixture = Omit<PredictResponse, "text" | "domain" | "model_type">;

/**
 * Illustrative mock outputs keyed by domain and model so the UI wiring can be checked
 * without a checkpoint. Categories follow OATS naming; none of this is a real model output.
 */
export const MOCK_FIXTURES: Record<Domain, Record<ModelType, Fixture>> = {
  amazon_ff: {
    gas: {
      quadruples: [
        { target: "soup", aspect: "food quality", opinion: "tastes fresh and homemade", sentiment: "positive" },
        { target: "NULL", aspect: "food prices", opinion: "overpriced", sentiment: "negative" },
      ],
      raw_generation:
        '[["soup","food quality","tastes fresh and homemade","positive"],["NULL","food prices","overpriced","negative"]]',
      invalid_predictions: [],
    },
    paraphrase: {
      quadruples: [
        { target: "soup", aspect: "food quality", opinion: "tastes fresh and homemade", sentiment: "positive" },
      ],
      raw_generation: "food quality is great because soup is tastes fresh and homemade",
      invalid_predictions: [],
    },
  },
  coursera: {
    gas: {
      quadruples: [
        { target: "course", aspect: "course quality", opinion: "NULL", sentiment: "positive" },
        { target: "lectures", aspect: "lecture delivery", opinion: "too fast", sentiment: "negative" },
      ],
      raw_generation:
        '[["course","course quality","NULL","positive"],["lectures","lecture delivery","too fast","negative"]]',
      invalid_predictions: [],
    },
    paraphrase: {
      quadruples: [
        { target: "NULL", aspect: "course general", opinion: "strongly recommend", sentiment: "positive" },
        { target: "assignments", aspect: "assignment difficulty", opinion: "challenging", sentiment: "conflict" },
      ],
      raw_generation:
        "course general is great because it is strongly recommend [SSEP] assignment difficulty is mixed because assignments is challenging [SSEP] quiz is ok because",
      invalid_predictions: [{ raw: "quiz is ok because", reason: "malformed_generation" }],
    },
  },
  hotels: {
    gas: {
      quadruples: [
        { target: "location", aspect: "location general", opinion: "Great", sentiment: "positive" },
        { target: "staff", aspect: "service general", opinion: "unhelpful", sentiment: "negative" },
        { target: "breakfast", aspect: "food quality", opinion: "average", sentiment: "neutral" },
      ],
      raw_generation:
        '[["location","location general","Great","positive"],["staff","service general","unhelpful","negative"],["breakfast","food quality","average","neutral"]]',
      invalid_predictions: [],
    },
    paraphrase: {
      quadruples: [
        { target: "room", aspect: "room cleanliness", opinion: "spotless", sentiment: "positive" },
      ],
      raw_generation: "room cleanliness is great because room is spotless",
      invalid_predictions: [],
    },
  },
};

/** Text markers that make the mock simulate other outcomes, for manual QA of UI states. */
export const MOCK_TRIGGERS = {
  empty: "#empty",
  error: "#error",
} as const;
