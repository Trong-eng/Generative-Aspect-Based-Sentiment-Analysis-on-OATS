"""Small, config-driven PyTorch trainer for both T5 ASQP formulations."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from capstone.config import load_config, to_serializable_config
from capstone.data.serialization import validate_prepared_metadata
from capstone.evaluation import evaluate_records
from capstone.logging_utils import configure_logging
from capstone.models.base import ParsedGeneration
from capstone.models.gas import parse_gas_generation, serialize_gas
from capstone.models.paraphrase import parse_paraphrase_generation, serialize_paraphrase
from capstone.training.seed import seed_everything
from capstone.types import Quadruple


def _require_training_dependencies():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Training requires torch and transformers from requirements.txt") from exc
    return torch, AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def _quads(record: dict[str, Any]) -> list[Quadruple]:
    values = record.get("quadruples", [])
    if not isinstance(values, list):
        raise ValueError("Each processed record must have a quadruples list")
    return [Quadruple.from_dict(value) if hasattr(Quadruple, "from_dict") else Quadruple(**value) for value in values]


def _record_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        field: record[field]
        for field in ("review_id", "sentence_id", "source_line_number")
        if record.get(field) is not None
    }


def _serializer_and_parser(model_type: str) -> tuple[Callable, Callable]:
    if model_type == "gas":
        return serialize_gas, parse_gas_generation
    if model_type == "paraphrase":
        return serialize_paraphrase, parse_paraphrase_generation
    raise ValueError(f"Unsupported model type: {model_type}")


class _TextDataset:
    def __init__(self, records, tokenizer, serializer, max_input_length, max_output_length):
        self.items = []
        for record in records:
            text = str(record.get("text", ""))
            labels = serializer(_quads(record))
            source_tokens = tokenizer(text, max_length=max_input_length, truncation=True)
            target_tokens = tokenizer(text_target=labels, max_length=max_output_length, truncation=True)
            self.items.append({
                "input_ids": source_tokens["input_ids"],
                "attention_mask": source_tokens["attention_mask"],
                "labels": target_tokens["input_ids"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _collate(batch, tokenizer, torch):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("The configured tokenizer has no pad_token_id")
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(item["labels"], dtype=torch.long) for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _parse(model_type: str, generation: str, known_aspects: set[str]) -> ParsedGeneration:
    _, parser = _serializer_and_parser(model_type)
    return parser(generation, known_aspects=known_aspects)


def _checkpoint_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint-epoch-{epoch:03d}"


def _save_checkpoint(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def train_model(config_path: str | Path) -> Path:
    """Train the configured model and return the selected checkpoint path."""
    config = load_config(config_path)
    errors = []
    training = config.get("training", {})
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate", "optimizer", "scheduler"):
        if key not in training:
            errors.append(f"training.{key} is required")
    for key in ("max_input_length", "max_output_length"):
        if key not in config.get("sequence", {}):
            errors.append(f"sequence.{key} is required")
    for key in ("num_beams", "max_new_tokens", "early_stopping"):
        if key not in config.get("generation", {}):
            errors.append(f"generation.{key} is required")
    if errors:
        raise ValueError("Invalid training configuration:\n- " + "\n- ".join(errors))
    if str(training["optimizer"]).casefold() != "adamw":
        raise ValueError("Only the AdamW optimizer is implemented")

    torch, AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler = _require_training_dependencies()
    seed = seed_everything(config["experiment"]["seed"])
    model_type = config["model"]["type"]
    serializer, parser = _serializer_and_parser(model_type)
    domain = config["experiment"]["domain"]
    processed_root = Path(config["paths"]["processed_data"]) / domain
    prepared_metadata = validate_prepared_metadata(
        processed_root,
        domain=domain,
        split_mode=config["experiment"]["split_mode"],
        sentiment_mode=config["dataset"]["sentiment_mode"],
        dataset_version=config.get("dataset", {}).get("version", "local_oats_xml"),
        seed=config["experiment"]["seed"],
        cleaning_configuration=config.get("cleaning"),
    )
    train_records = _read_jsonl(processed_root / "train.jsonl")
    dev_records = _read_jsonl(processed_root / "dev.jsonl")
    if not train_records:
        raise ValueError(f"Training split is empty: {processed_root / 'train.jsonl'}")
    if not dev_records:
        raise ValueError(f"Validation split is empty: {processed_root / 'dev.jsonl'}")

    output_dir = Path(config["paths"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment output already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        output_dir / "logs" / "run.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
    )
    logger.info("Starting configured %s model training for %s", model_type, domain)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["model_name"]).to(device)
    max_input_length = int(config["sequence"]["max_input_length"])
    max_output_length = int(config["sequence"]["max_output_length"])
    train_dataset = _TextDataset(train_records, tokenizer, serializer, max_input_length, max_output_length)
    dev_dataset = _TextDataset(dev_records, tokenizer, serializer, max_input_length, max_output_length)
    collate = lambda batch: _collate(batch, tokenizer, torch)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        generator=loader_generator,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * int(training["epochs"])
    warmup_steps = int(total_updates * float(training.get("warmup_ratio", 0.0)))
    scheduler = get_scheduler(
        name=str(training["scheduler"]).casefold(),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    clip_norm = float(training.get("max_grad_norm", 1.0))
    best_f1 = -1.0
    best_checkpoint: Path | None = None
    history: list[dict[str, Any]] = []
    log_file = output_dir / "logs" / "training_log.jsonl"
    history_file = output_dir / "training_history.json"
    started = time.perf_counter()
    aspect_inventory = prepared_metadata.get("aspect_inventory")
    if (
        not isinstance(aspect_inventory, list)
        or not aspect_inventory
        or any(not isinstance(aspect, str) or not aspect.strip() for aspect in aspect_inventory)
    ):
        raise ValueError(
            f"Prepared metadata has no valid domain aspect inventory: {processed_root / 'metadata.json'}"
        )
    known_aspects = set(aspect_inventory)

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batch_count = 0
        update_count = 0
        accumulated_group_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}")
            total_loss += float(loss.detach().cpu())
            accumulated_group_loss += float(loss.detach().cpu())
            batch_count += 1
            group_start = ((batch_index - 1) // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            (loss / group_size).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
                _append_jsonl(log_file, {
                    "epoch": epoch,
                    "step": (epoch - 1) * updates_per_epoch + update_count,
                    "train_loss": accumulated_group_loss / group_size,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "elapsed_time": time.perf_counter() - started,
                })
                accumulated_group_loss = 0.0

        model.eval()
        validation_loss = 0.0
        validation_batches = 0
        cases: list[dict[str, Any]] = []
        with torch.no_grad():
            record_offset = 0
            for batch in dev_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                validation_loss += float(outputs.loss.detach().cpu())
                validation_batches += 1
                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=int(config["generation"]["num_beams"]),
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    early_stopping=config["generation"]["early_stopping"],
                )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                batch_size = len(decoded)
                source_records = dev_records[record_offset:record_offset + batch_size]
                record_offset += batch_size
                for record, raw_generation in zip(source_records, decoded):
                    parsed = parser(raw_generation, known_aspects=known_aspects)
                    cases.append({
                        **_record_identity(record),
                        "gold_quadruples": _quads(record),
                        "predicted_quadruples": parsed.quadruples,
                        "text": record.get("text", ""),
                    })
        metrics, _, _ = evaluate_records(cases)
        epoch_history = {
            "epoch": epoch,
            "step": (epoch - 1) * updates_per_epoch + update_count,
            "train_loss": total_loss / max(batch_count, 1),
            "validation_loss": validation_loss / max(validation_batches, 1),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "validation_f1": float(metrics["f1"]),
            "elapsed_time": time.perf_counter() - started,
        }
        history.append(epoch_history)
        _json_dump(history_file, history)
        logger.info(
            "Epoch %d/%d train_loss=%.6f validation_loss=%.6f validation_f1=%.6f",
            epoch, int(training["epochs"]), epoch_history["train_loss"],
            epoch_history["validation_loss"], epoch_history["validation_f1"],
        )
        if metrics["f1"] > best_f1:
            best_f1 = float(metrics["f1"])
            best_checkpoint = _checkpoint_path(output_dir, epoch)
            _save_checkpoint(model, tokenizer, best_checkpoint)
            _json_dump(output_dir / "best_checkpoint.json", {
                "selection_metric": "validation_exact_match_f1",
                "validation_f1": best_f1,
                "epoch": epoch,
                "path": best_checkpoint.relative_to(output_dir).as_posix(),
            })

    experiment = config["experiment"]
    metadata = {
        "experiment_id": output_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(Path(config["_project_root"])),
        "domain": domain,
        "split_mode": experiment["split_mode"],
        "dataset_version": config.get("dataset", {}).get("version", "local_oats_xml"),
        "cleaning_configuration": config.get("cleaning", {}),
        "sentiment_mode": config["dataset"]["sentiment_mode"],
        "seed": seed,
        "model_type": model_type,
        "model_backbone": config["model"]["model_name"],
        "tokenizer": config["model"]["tokenizer_name"],
        "hyperparameters": training,
        "checkpoint": best_checkpoint.relative_to(output_dir).as_posix() if best_checkpoint else None,
        "evaluation_split": "dev",
        "train_samples": len(train_records),
        "validation_samples": len(dev_records),
        "split_fingerprints": prepared_metadata.get("split_fingerprints", {}),
        "source_fingerprints": prepared_metadata.get("source_fingerprints", {}),
        "aspect_inventory": sorted(known_aspects),
    }
    _json_dump(output_dir / "metadata.json", metadata)
    yaml = __import__("yaml")
    _json_dump(output_dir / "training_summary.json", {
        "epochs_completed": len(history),
        "best_validation_f1": best_f1,
        "best_checkpoint": best_checkpoint.relative_to(output_dir).as_posix() if best_checkpoint else None,
    })
    snapshot_path = output_dir / "config.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(to_serializable_config(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("Training finished; selected checkpoint: %s", best_checkpoint)
    if best_checkpoint is None:
        raise RuntimeError("Training completed without producing a checkpoint")
    return best_checkpoint


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train a GAS-T5 or Paraphrase-T5 OATS ASQP model")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args(argv)
    checkpoint = train_model(args.config)
    print(json.dumps({"checkpoint": str(checkpoint)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
