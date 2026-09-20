import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from capstone.models.base import ParsedGeneration
from capstone.training import trainer


class _FakeTensor:
    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def backward(self):
        return None

    def __truediv__(self, _value):
        return self

    def __float__(self):
        return 0.25


class _FakeLoader:
    def __init__(self, dataset, *, batch_size, collate_fn, **_kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        items = [self.dataset[index] for index in range(len(self.dataset))]
        for start in range(0, len(items), self.batch_size):
            yield self.collate_fn(items[start : start + self.batch_size])


class _FakeLoss:
    def zero_grad(self, **_kwargs):
        return None

    def step(self):
        return None


def test_training_records_logging_ontology_and_relative_checkpoint_without_real_model_work(
    tmp_path, monkeypatch
):
    processed_root = tmp_path / "processed"
    domain_root = processed_root / "hotels"
    domain_root.mkdir(parents=True)
    row = json.dumps({
        "review_id": "r1",
        "sentence_id": "r1:0",
        "text": "A fixture sentence.",
        "quadruples": [],
    }) + "\n"
    (domain_root / "train.jsonl").write_text(row, encoding="utf-8")
    (domain_root / "dev.jsonl").write_text(row, encoding="utf-8")
    inventory = ["hotel service", "room cleanliness"]
    prepared_metadata = {
        "aspect_inventory": inventory,
        "split_fingerprints": {name: "a" * 64 for name in ("train", "dev", "test")},
        "source_fingerprints": {"raw_xml": "b" * 64},
    }
    output_dir = tmp_path / "experiment"
    config = {
        "experiment": {"name": "mocked", "domain": "hotels", "split_mode": "clean_split", "seed": 7},
        "model": {"type": "gas", "model_name": "fixture", "tokenizer_name": "fixture"},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "optimizer": "AdamW",
            "scheduler": "linear",
        },
        "sequence": {"max_input_length": 16, "max_output_length": 16},
        "generation": {"num_beams": 1, "max_new_tokens": 8, "early_stopping": False},
        "dataset": {"sentiment_mode": "paper_compatible", "version": "fixture"},
        "cleaning": {"remove_unannotated_reviews": False},
        "logging": {"level": "DEBUG"},
        "paths": {
            "raw_data": tmp_path / "raw",
            "processed_data": processed_root,
            "output_dir": output_dir,
        },
        "_project_root": tmp_path,
    }

    class FakeModel:
        def to(self, _device):
            return self

        def parameters(self):
            return []

        def train(self):
            return None

        def eval(self):
            return None

        def __call__(self, **_batch):
            return SimpleNamespace(loss=_FakeTensor())

        def generate(self, **_kwargs):
            return ["fixture generation"]

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.fixture").write_text("mock", encoding="utf-8")

    class FakeTokenizer:
        def batch_decode(self, _generated, skip_special_tokens):
            assert skip_special_tokens is True
            return ["[]"]

        def save_pretrained(self, path):
            (Path(path) / "tokenizer.fixture").write_text("mock", encoding="utf-8")

    class FakeDataset:
        def __init__(self, records, *_args):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    logged = {}
    parser_aspects = []

    class FakeLogger:
        def info(self, *_args):
            return None

    class FakeScheduler:
        def step(self):
            return None

        def get_last_lr(self):
            return [0.001]

    class FakeOptimizer(_FakeLoss):
        pass

    class FakeGenerator:
        def manual_seed(self, _seed):
            return None

    fake_torch = SimpleNamespace(
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: False),
        Generator=FakeGenerator,
        utils=SimpleNamespace(data=SimpleNamespace(DataLoader=_FakeLoader)),
        optim=SimpleNamespace(AdamW=lambda *_args, **_kwargs: FakeOptimizer()),
        nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *_args: None)),
        no_grad=nullcontext,
        isfinite=lambda _value: True,
    )

    def parse(_generation, *, known_aspects):
        parser_aspects.append(set(known_aspects))
        return ParsedGeneration()

    monkeypatch.setattr(trainer, "load_config", lambda _path: config)
    monkeypatch.setattr(trainer, "_require_training_dependencies", lambda: (
        fake_torch,
        SimpleNamespace(from_pretrained=lambda *_args: FakeModel()),
        SimpleNamespace(from_pretrained=lambda *_args: FakeTokenizer()),
        lambda **_kwargs: FakeScheduler(),
    ))
    monkeypatch.setattr(trainer, "seed_everything", lambda seed: seed)
    monkeypatch.setattr(trainer, "validate_prepared_metadata", lambda *_args, **_kwargs: prepared_metadata)
    monkeypatch.setattr(trainer, "_TextDataset", FakeDataset)
    monkeypatch.setattr(
        trainer,
        "_collate",
        lambda _batch, _tokenizer, _torch: {
            "input_ids": _FakeTensor(),
            "attention_mask": _FakeTensor(),
            "labels": _FakeTensor(),
        },
    )
    monkeypatch.setattr(
        trainer,
        "_serializer_and_parser",
        lambda _model_type: (lambda _quads: "[]", parse),
    )
    def configure_logging(path, *, level):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logged["level"] = level
        return FakeLogger()

    monkeypatch.setattr(trainer, "configure_logging", configure_logging)
    monkeypatch.setattr(trainer, "_git_commit", lambda _root: None)

    checkpoint = trainer.train_model("unused.yaml")

    experiment_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    selection = json.loads((output_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint == output_dir / "checkpoints" / "checkpoint-epoch-001"
    assert logged["level"] == "DEBUG"
    assert parser_aspects == [set(inventory)]
    assert experiment_metadata["aspect_inventory"] == inventory
    assert experiment_metadata["checkpoint"] == "checkpoints/checkpoint-epoch-001"
    assert selection["path"] == "checkpoints/checkpoint-epoch-001"
