from capstone.audit_data import audit_processed
from capstone.data.serialization import write_jsonl
from capstone.types import SentenceRecord


def test_processed_audit_reports_sentence_counts_and_review_overlap(tmp_path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(
        train,
        [SentenceRecord("shared", "shared:0", "Train text.", [])],
    )
    write_jsonl(dev, [SentenceRecord("dev-only", "dev-only:0", "Dev text.", [])])
    write_jsonl(
        test,
        [SentenceRecord("shared", "shared:1", "Leaked test text.", [])],
    )

    result = audit_processed(tmp_path)

    assert result["splits"]["train"]["number_of_sentences"] == 1
    assert result["splits"]["test"]["number_of_reviews"] == 1
    assert result["review_overlap"]["train_test"] == ["shared"]
    assert result["no_review_leakage"] is False
