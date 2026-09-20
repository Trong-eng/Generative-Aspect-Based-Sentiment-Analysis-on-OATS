"""Rebuild structured error cases from saved evaluation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from capstone.evaluation import analyze_errors


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_error_analysis(predictions_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(predictions_path).expanduser()
    rows = _read_jsonl(source)
    cases = [
        {
            **row,
            "predicted_quadruples": row.get("normalized_quadruples", row.get("parsed_quadruples", [])),
        }
        for row in rows
    ]
    errors = analyze_errors(cases)
    destination = Path(output_path).expanduser() if output_path else source.parent / "error_cases.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for error in errors:
            stream.write(json.dumps(error, ensure_ascii=False) + "\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write error analysis from predictions.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    destination = write_error_analysis(args.predictions, args.output)
    print(json.dumps({"error_cases": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

