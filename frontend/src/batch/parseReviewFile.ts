import type { Quadruple } from "../api";

export const MAX_FILE_BYTES = 10 * 1024 * 1024;
export const MAX_ROWS = 5000;

export type ReviewFileFormat = "txt" | "csv" | "jsonl" | "json";

/** One text unit from the file. Rows sharing `reviewId` belong to the same review. */
export interface ReviewRow {
  /** Unique per row; sent to the API as the batch item id. */
  rowId: string;
  reviewId: string;
  text: string;
  /** Present when the file already carries predictions (e.g. capstone.predict output). */
  quadruples?: Quadruple[];
}

export interface FileIssue {
  line: number;
  message: string;
}

export interface ParsedReviewFile {
  format: ReviewFileFormat;
  rows: ReviewRow[];
  reviewCount: number;
  /** True when every row carries quadruples, so no model call is needed. */
  precomputed: boolean;
  /** Skipped rows; the rest of the file is still used. */
  issues: FileIssue[];
}

export class FileParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FileParseError";
  }
}

const TEXT_COLUMNS = ["text", "review", "review_text", "content", "comment", "sentence"];
const ID_COLUMNS = ["review_id", "id"];
const QUAD_KEYS = ["quadruples", "normalized_quadruples", "predicted_quadruples"];
const QUAD_FIELDS = ["target", "aspect", "opinion", "sentiment"] as const;

interface RawRow {
  line: number;
  text: unknown;
  reviewId?: unknown;
  quadruples?: unknown;
  hasQuadruples: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toQuadruples(value: unknown): Quadruple[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const quads: Quadruple[] = [];
  for (const item of value) {
    if (isRecord(item) && QUAD_FIELDS.every((field) => typeof item[field] === "string")) {
      quads.push({
        target: item.target as string,
        aspect: item.aspect as string,
        opinion: item.opinion as string,
        sentiment: item.sentiment as string,
      });
    } else if (Array.isArray(item) && item.length === 4 && item.every((field) => typeof field === "string")) {
      // capstone's GAS tuple order: [target, aspect, opinion, sentiment]
      const [target, aspect, opinion, sentiment] = item as string[];
      quads.push({ target: target!, aspect: aspect!, opinion: opinion!, sentiment: sentiment! });
    } else {
      return null;
    }
  }
  return quads;
}

function fromObject(value: unknown, line: number): RawRow {
  if (typeof value === "string") {
    return { line, text: value, hasQuadruples: false };
  }
  if (!isRecord(value)) {
    return { line, text: undefined, hasQuadruples: false };
  }
  const quadKey = QUAD_KEYS.find((key) => key in value);
  return {
    line,
    text: value.text,
    reviewId: ID_COLUMNS.map((key) => value[key]).find((id) => id !== undefined && id !== null && id !== ""),
    quadruples: quadKey ? value[quadKey] : undefined,
    hasQuadruples: quadKey !== undefined,
  };
}

/** Minimal RFC 4180 CSV reader: quoted fields, doubled quotes, embedded commas/newlines. */
export function parseCsv(content: string): { cells: string[]; line: number }[] {
  const records: { cells: string[]; line: number }[] = [];
  let cells: string[] = [];
  let field = "";
  let inQuotes = false;
  let line = 1;
  let recordLine = 1;
  for (let index = 0; index < content.length; index++) {
    const char = content[index]!;
    if (inQuotes) {
      if (char === '"') {
        if (content[index + 1] === '"') {
          field += '"';
          index++;
        } else {
          inQuotes = false;
        }
      } else {
        if (char === "\n") line++;
        field += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      cells.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && content[index + 1] === "\n") index++;
      cells.push(field);
      records.push({ cells, line: recordLine });
      cells = [];
      field = "";
      line++;
      recordLine = line;
    } else {
      field += char;
    }
  }
  if (inQuotes) {
    throw new FileParseError(`CSV has an unclosed quote starting on line ${recordLine}.`);
  }
  if (field !== "" || cells.length > 0) {
    cells.push(field);
    records.push({ cells, line: recordLine });
  }
  return records.filter((record) => record.cells.some((cell) => cell.trim() !== ""));
}

function rawRowsFor(format: ReviewFileFormat, content: string, issues: FileIssue[]): RawRow[] {
  switch (format) {
    case "txt":
      return content
        .split(/\r?\n/)
        .map((text, index) => ({ line: index + 1, text, hasQuadruples: false }))
        .filter((row) => row.text.trim() !== "");
    case "jsonl":
      return content.split(/\r?\n/).flatMap((text, index) => {
        if (!text.trim()) return [];
        try {
          return [fromObject(JSON.parse(text), index + 1)];
        } catch {
          issues.push({ line: index + 1, message: "Invalid JSON." });
          return [];
        }
      });
    case "json": {
      let value: unknown;
      try {
        value = JSON.parse(content);
      } catch {
        throw new FileParseError("The JSON file is not valid JSON.");
      }
      if (!Array.isArray(value)) {
        throw new FileParseError("A .json file must contain a list of reviews (strings or objects with 'text').");
      }
      return value.map((item, index) => fromObject(item, index + 1));
    }
    case "csv": {
      const [header, ...records] = parseCsv(content);
      if (!header) {
        return [];
      }
      const names = header.cells.map((cell) => cell.trim().toLowerCase());
      const textIndex = names.findIndex((name) => TEXT_COLUMNS.includes(name));
      if (textIndex < 0) {
        throw new FileParseError(`CSV needs a header with a text column (one of: ${TEXT_COLUMNS.join(", ")}).`);
      }
      const idIndex = names.findIndex((name) => ID_COLUMNS.includes(name));
      return records.map((record) => ({
        line: record.line,
        text: record.cells[textIndex],
        reviewId: idIndex >= 0 ? record.cells[idIndex]?.trim() || undefined : undefined,
        hasQuadruples: false,
      }));
    }
  }
}

export function detectFormat(fileName: string): ReviewFileFormat {
  const extension = fileName.toLowerCase().split(".").pop();
  if (extension === "txt" || extension === "csv" || extension === "jsonl" || extension === "json") {
    return extension;
  }
  throw new FileParseError("Unsupported file type. Use .csv, .txt, .jsonl or .json.");
}

/** Parses file content into review rows; bad rows become issues, fatal problems throw FileParseError. */
export function parseReviewFile(fileName: string, content: string): ParsedReviewFile {
  const format = detectFormat(fileName);
  const issues: FileIssue[] = [];
  const raw = rawRowsFor(format, content.charCodeAt(0) === 0xfeff ? content.slice(1) : content, issues);

  const withQuads = raw.filter((row) => row.hasQuadruples).length;
  if (withQuads > 0 && withQuads < raw.length) {
    throw new FileParseError(
      `Only ${withQuads} of ${raw.length} rows include quadruples. Use a file where every row has predictions, or none do.`,
    );
  }
  const precomputed = raw.length > 0 && withQuads === raw.length;

  const rows: ReviewRow[] = [];
  for (const row of raw) {
    if (typeof row.text !== "string" || !row.text.trim()) {
      issues.push({ line: row.line, message: "Missing or empty text." });
      continue;
    }
    let quadruples: Quadruple[] | undefined;
    if (precomputed) {
      const parsed = toQuadruples(row.quadruples);
      if (!parsed) {
        issues.push({ line: row.line, message: "Quadruples are not a list of {target, aspect, opinion, sentiment}." });
        continue;
      }
      quadruples = parsed;
    }
    const rowId = `row-${row.line}`;
    const reviewId =
      typeof row.reviewId === "string" || typeof row.reviewId === "number" ? String(row.reviewId) : rowId;
    rows.push({ rowId, reviewId, text: row.text.trim(), ...(quadruples ? { quadruples } : {}) });
  }

  if (rows.length === 0) {
    throw new FileParseError("No reviews found in the file.");
  }
  if (rows.length > MAX_ROWS) {
    throw new FileParseError(`The file has ${rows.length} rows; the limit is ${MAX_ROWS}. Split it into smaller files.`);
  }
  return { format, rows, reviewCount: new Set(rows.map((row) => row.reviewId)).size, precomputed, issues };
}
