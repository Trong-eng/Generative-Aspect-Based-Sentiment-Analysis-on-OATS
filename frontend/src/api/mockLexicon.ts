import type { Domain, Quadruple } from "./types";

/**
 * Keyword-rule stand-in for the model, used only by the mock service so the batch
 * dashboard shows text-dependent results without a checkpoint. It is deliberately
 * simple and is NOT an estimate of how the trained models behave.
 */

type Polarity = "positive" | "negative" | "neutral";

interface OpinionRule {
  phrase: string;
  polarity: Polarity;
  /** Preferred OATS attribute, e.g. "cleanliness" → "rooms cleanliness". */
  attribute?: string;
}

interface EntityRule {
  /** OATS entity prefix, e.g. "rooms" in "rooms cleanliness". */
  entity: string;
  pattern: RegExp;
}

interface DomainRules {
  general: string;
  categories: ReadonlySet<string>;
  entities: EntityRule[];
}

const OPINIONS: OpinionRule[] = [
  { phrase: "clean", polarity: "positive", attribute: "cleanliness" },
  { phrase: "spotless", polarity: "positive", attribute: "cleanliness" },
  { phrase: "dirty", polarity: "negative", attribute: "cleanliness" },
  { phrase: "filthy", polarity: "negative", attribute: "cleanliness" },
  { phrase: "stained", polarity: "negative", attribute: "cleanliness" },
  { phrase: "dusty", polarity: "negative", attribute: "cleanliness" },
  { phrase: "smelly", polarity: "negative", attribute: "cleanliness" },
  { phrase: "mouldy", polarity: "negative", attribute: "cleanliness" },
  { phrase: "moldy", polarity: "negative", attribute: "cleanliness" },
  { phrase: "comfortable", polarity: "positive", attribute: "comfort" },
  { phrase: "comfy", polarity: "positive", attribute: "comfort" },
  { phrase: "quiet", polarity: "positive", attribute: "comfort" },
  { phrase: "uncomfortable", polarity: "negative", attribute: "comfort" },
  { phrase: "noisy", polarity: "negative", attribute: "comfort" },
  { phrase: "good value", polarity: "positive", attribute: "prices" },
  { phrase: "affordable", polarity: "positive", attribute: "prices" },
  { phrase: "cheap", polarity: "positive", attribute: "prices" },
  { phrase: "overpriced", polarity: "negative", attribute: "prices" },
  { phrase: "expensive", polarity: "negative", attribute: "prices" },
  { phrase: "pricey", polarity: "negative", attribute: "prices" },
  { phrase: "spacious", polarity: "positive", attribute: "design_features" },
  { phrase: "modern", polarity: "positive", attribute: "design_features" },
  { phrase: "cramped", polarity: "negative", attribute: "design_features" },
  { phrase: "outdated", polarity: "negative", attribute: "design_features" },
  { phrase: "tiny", polarity: "negative", attribute: "design_features" },
  { phrase: "fresh", polarity: "positive", attribute: "quality" },
  { phrase: "delicious", polarity: "positive", attribute: "quality" },
  { phrase: "tasty", polarity: "positive", attribute: "quality" },
  { phrase: "stale", polarity: "negative", attribute: "quality" },
  { phrase: "bland", polarity: "negative", attribute: "quality" },
  { phrase: "engaging", polarity: "positive", attribute: "quality" },
  { phrase: "clear", polarity: "positive", attribute: "quality" },
  { phrase: "useful", polarity: "positive", attribute: "quality" },
  { phrase: "confusing", polarity: "negative", attribute: "quality" },
  { phrase: "boring", polarity: "negative", attribute: "quality" },
  { phrase: "useless", polarity: "negative", attribute: "quality" },
  { phrase: "broken", polarity: "negative", attribute: "quality" },
  { phrase: "detailed", polarity: "positive", attribute: "comprehensiveness" },
  { phrase: "thorough", polarity: "positive", attribute: "comprehensiveness" },
  { phrase: "shallow", polarity: "negative", attribute: "comprehensiveness" },
  { phrase: "superficial", polarity: "negative", attribute: "comprehensiveness" },
  { phrase: "manageable", polarity: "positive", attribute: "workload" },
  { phrase: "overwhelming", polarity: "negative", attribute: "workload" },
  { phrase: "too much work", polarity: "negative", attribute: "workload" },
  { phrase: "on time", polarity: "positive", attribute: "delivery" },
  { phrase: "fast", polarity: "positive", attribute: "delivery" },
  { phrase: "quick", polarity: "positive", attribute: "delivery" },
  { phrase: "late", polarity: "negative", attribute: "delivery" },
  { phrase: "slow", polarity: "negative", attribute: "delivery" },
  { phrase: "damaged", polarity: "negative", attribute: "quality" },
  { phrase: "crushed", polarity: "negative", attribute: "quality" },
  { phrase: "practical", polarity: "positive", attribute: "relatability" },
  { phrase: "relevant", polarity: "positive", attribute: "relatability" },
  { phrase: "irrelevant", polarity: "negative", attribute: "relatability" },
  { phrase: "unresponsive", polarity: "negative", attribute: "response" },
  { phrase: "responsive", polarity: "positive", attribute: "response" },
  { phrase: "excellent", polarity: "positive" },
  { phrase: "fantastic", polarity: "positive" },
  { phrase: "amazing", polarity: "positive" },
  { phrase: "great", polarity: "positive" },
  { phrase: "friendly", polarity: "positive" },
  { phrase: "helpful", polarity: "positive" },
  { phrase: "lovely", polarity: "positive" },
  { phrase: "perfect", polarity: "positive" },
  { phrase: "good", polarity: "positive" },
  { phrase: "nice", polarity: "positive" },
  { phrase: "unhelpful", polarity: "negative" },
  { phrase: "rude", polarity: "negative" },
  { phrase: "terrible", polarity: "negative" },
  { phrase: "awful", polarity: "negative" },
  { phrase: "horrible", polarity: "negative" },
  { phrase: "disappointing", polarity: "negative" },
  { phrase: "poor", polarity: "negative" },
  { phrase: "bad", polarity: "negative" },
  { phrase: "average", polarity: "neutral" },
  { phrase: "okay", polarity: "neutral" },
  { phrase: "ok", polarity: "neutral" },
  { phrase: "decent", polarity: "neutral" },
];

function words(...values: string[]): RegExp {
  return new RegExp(`\\b(?:${values.join("|")})\\b`, "gi");
}

/** Subsets of the OATS aspect inventories (see OATS-ABSA-Dataset quads files). */
const RULES: Record<Domain, DomainRules> = {
  hotels: {
    general: "hotel general",
    categories: new Set([
      "hotel general", "hotel prices", "hotel cleanliness", "hotel comfort", "hotel design_features", "hotel quality",
      "service general", "location general",
      "rooms general", "rooms cleanliness", "rooms comfort", "rooms design_features", "rooms prices", "rooms quality",
      "food_drinks quality", "food_drinks prices", "food_drinks style_options",
      "facilities general", "facilities cleanliness", "facilities comfort", "facilities design_features",
      "facilities prices", "facilities quality",
      "room_amenities general", "room_amenities cleanliness", "room_amenities comfort",
      "room_amenities design_features", "room_amenities prices", "room_amenities quality",
    ]),
    entities: [
      { entity: "room_amenities", pattern: words("towels?", "tv", "air ?con(?:ditioning)?", "minibar", "kettle", "hair ?dryer") },
      { entity: "rooms", pattern: words("rooms?", "bathrooms?", "beds?", "sheets", "showers?", "carpets?") },
      { entity: "service", pattern: words("staff", "reception(?:ist)?", "service", "manager", "housekeeping") },
      { entity: "location", pattern: words("location", "area", "neighbou?rhood") },
      { entity: "food_drinks", pattern: words("breakfast", "food", "restaurant", "coffee", "bar", "dinner", "buffet") },
      { entity: "facilities", pattern: words("pool", "gym", "spa", "parking", "elevators?", "lifts?", "wifi", "wi-fi") },
      { entity: "hotel", pattern: words("hotel", "property", "lobby", "place") },
    ],
  },
  amazon_ff: {
    general: "food general",
    categories: new Set([
      "food general", "food quality", "food style_options", "food prices",
      "shipment quality", "shipment delivery", "shipment prices", "amazon availability", "amazon prices",
    ]),
    entities: [
      { entity: "shipment", pattern: words("shipping", "delivery", "package", "packaging", "box", "arrived") },
      { entity: "amazon", pattern: words("amazon", "seller", "stock") },
      { entity: "food", pattern: words("taste", "flavou?r", "soup", "snacks?", "cereal", "product", "tea", "chips", "food", "sauce") },
    ],
  },
  coursera: {
    general: "course general",
    categories: new Set([
      "course general", "course quality", "course comprehensiveness", "course relatability", "course value",
      "course workload", "faculty general", "faculty comprehensiveness", "faculty response", "faculty relatability",
      "presentation quality", "presentation comprehensiveness", "presentation relatability", "presentation workload",
      "assignments quality", "assignments comprehensiveness", "assignments relatability", "assignments workload",
      "material quality", "material comprehensiveness", "material relatability", "material quantity", "grades general",
    ]),
    entities: [
      { entity: "faculty", pattern: words("instructors?", "professors?", "teachers?", "lecturers?") },
      { entity: "presentation", pattern: words("lectures?", "videos?", "slides") },
      { entity: "assignments", pattern: words("assignments?", "quiz(?:zes)?", "homework", "projects?", "exercises?") },
      { entity: "material", pattern: words("materials?", "readings?", "notes") },
      { entity: "grades", pattern: words("grading", "grades?", "peer reviews?") },
      { entity: "course", pattern: words("course", "class", "program(?:me)?") },
    ],
  },
};

const NEGATION = /(?:\b(?:not|never|no|hardly)|n't)\s+(?:\w+\s+)?$/i;

interface Match {
  start: number;
  end: number;
  text: string;
}

function findAll(pattern: RegExp, text: string): Match[] {
  const found: Match[] = [];
  for (const match of text.matchAll(new RegExp(pattern.source, "gi"))) {
    found.push({ start: match.index, end: match.index + match[0].length, text: match[0] });
  }
  return found;
}

function categoryFor(rules: DomainRules, entity: string | null, attribute: string | undefined): string {
  const candidates = entity
    ? [attribute && `${entity} ${attribute}`, `${entity} general`, `${entity} quality`]
    : [attribute && `${rules.general.split(" ")[0]} ${attribute}`];
  return candidates.find((candidate): candidate is string => !!candidate && rules.categories.has(candidate)) ?? rules.general;
}

function flip(polarity: Polarity): Polarity {
  return polarity === "positive" ? "negative" : polarity === "negative" ? "positive" : polarity;
}

/** Extracts rule-based quadruples clause by clause, pairing each opinion with the nearest entity. */
export function mockExtract(text: string, domain: Domain): Quadruple[] {
  const rules = RULES[domain];
  const quadruples: Quadruple[] = [];
  const seen = new Set<string>();
  for (const clause of text.split(/[.!?;,]+|\bbut\b|\bhowever\b|\bwhile\b/i)) {
    const entities = rules.entities.flatMap((rule) =>
      findAll(rule.pattern, clause).map((match) => ({ ...match, entity: rule.entity })),
    );
    const taken: Match[] = [];
    for (const opinion of OPINIONS) {
      for (const match of findAll(new RegExp(`\\b${opinion.phrase}\\b`), clause)) {
        if (taken.some((other) => match.start < other.end && other.start < match.end)) {
          continue;
        }
        taken.push(match);
        const negation = NEGATION.exec(clause.slice(0, match.start));
        const polarity = negation ? flip(opinion.polarity) : opinion.polarity;
        const opinionText = negation ? `${negation[0]}${match.text}`.trim() : match.text;
        const nearest = entities
          .map((entity) => ({ entity, distance: Math.abs(entity.start - match.start) }))
          .sort((a, b) => a.distance - b.distance)[0]?.entity;
        const quad: Quadruple = {
          target: nearest ? nearest.text : "NULL",
          aspect: categoryFor(rules, nearest?.entity ?? null, opinion.attribute),
          opinion: opinionText,
          sentiment: polarity,
        };
        const key = JSON.stringify(quad).toLowerCase();
        if (!seen.has(key)) {
          seen.add(key);
          quadruples.push(quad);
        }
      }
    }
  }
  return quadruples;
}
