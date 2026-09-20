from pathlib import Path

import pytest

from capstone.data.parser import (
    DataFormatError,
    load_original_splits,
    parse_oats_xml,
    parse_original_split_file,
)
from capstone.data.paths import resolve_domain_paths


XML = """<Reviews>
  <Review rid="r1"><sentences>
    <sentence sid="r1:0"><text>A clean room.</text><Opinions>
      <Opinion category="ROOMS#CLEANLINESS" polarity="Positive" target="room"
        opinion_phrase="clean" target_from="8" target_to="11"
        opinion_phrase_from="2" opinion_phrase_to="6" />
      <Opinion category="ROOMS#DESIGN_FEATURES" polarity="Positive" target="room"
        opinion_phrase="clean" />
      <Opinion category="POLARITY#POSITIVE" polarity="Negative" target="NULL"
        opinion_phrase="bad" />
    </Opinions></sentence>
    <sentence sid="r1:1"><text>No opinion here.</text><Opinions /></sentence>
  </sentences></Review>
  <Review rid="r2"><sentences>
    <sentence sid="r2:0"><text>Useful course.</text><Opinions>
      <Opinion category="COURSE#QUALITY" polarity="Conflict" target="course"
        opinion_phrase="NULL" />
    </Opinions></sentence>
  </sentences></Review>
</Reviews>"""


def test_xml_parser_preserves_review_ids_null_values_and_empty_sentences(tmp_path: Path):
    xml_path = tmp_path / "tiny.xml"
    xml_path.write_text(XML, encoding="utf-8")

    parsed = parse_oats_xml(xml_path)

    assert [review.review_id for review in parsed.reviews] == ["r1", "r2"]
    assert [sentence.sentence_id for sentence in parsed.reviews[0].sentences] == [
        "r1:0",
        "r1:1",
    ]
    assert parsed.reviews[0].sentences[1].quadruples == []
    assert parsed.reviews[0].sentences[0].quadruples[0].aspect == "rooms cleanliness"
    assert parsed.reviews[0].sentences[0].quadruples[0].target == "room"
    assert parsed.reviews[0].sentences[0].quadruples[1].aspect == "rooms design_features"
    assert parsed.reviews[1].sentences[0].quadruples[0].opinion == "NULL"
    assert parsed.reviews[1].sentences[0].quadruples[0].sentiment == "conflict"


def test_xml_parser_reports_malformed_quadruples_without_aborting(tmp_path: Path):
    xml_path = tmp_path / "malformed.xml"
    xml_path.write_text(
        '<Reviews><Review rid="r"><sentences><sentence sid="s">'
        '<text>Text.</text><Opinions><Opinion category="ROOMS#GENERAL" '
        'polarity="Positive" target="room" /></Opinions></sentence>'
        '</sentences></Review></Reviews>',
        encoding="utf-8",
    )

    parsed = parse_oats_xml(xml_path)

    assert parsed.reviews[0].sentences[0].quadruples == []
    assert any(issue.reason == "malformed_quadruple" for issue in parsed.issues)


def test_original_txt_parser_uses_asqp_order_and_rejects_malformed_line(tmp_path: Path):
    split_path = tmp_path / "train.txt"
    split_path.write_text(
        "A clean room.####[['room', 'rooms cleanliness', 'positive', 'clean']]\n"
        "No opinion here.####[]\n",
        encoding="utf-8",
    )

    rows = parse_original_split_file(split_path)

    assert rows[0].quadruples[0].target == "room"
    assert rows[0].quadruples[0].aspect == "rooms cleanliness"
    assert rows[0].quadruples[0].opinion == "clean"
    assert rows[0].quadruples[0].sentiment == "positive"
    assert rows[1].quadruples == []

    split_path.write_text("broken line\n", encoding="utf-8")
    with pytest.raises(DataFormatError):
        parse_original_split_file(split_path)


def test_original_split_alignment_attaches_txt_rows_to_xml_review_ids(tmp_path: Path):
    xml_path = tmp_path / "tiny.xml"
    xml_path.write_text(XML, encoding="utf-8")
    train_path = tmp_path / "train.txt"
    dev_path = tmp_path / "dev.txt"
    test_path = tmp_path / "test.txt"
    train_path.write_text(
        "A clean room.####[['room', 'rooms cleanliness', 'positive', 'clean']]\n"
        "No opinion here.####[]\n",
        encoding="utf-8",
    )
    dev_path.write_text(
        "Useful course.####[['course', 'course quality', 'conflict', 'NULL']]\n",
        encoding="utf-8",
    )
    test_path.write_text("", encoding="utf-8")

    parsed = parse_oats_xml(xml_path)
    splits = load_original_splits(
        parsed.reviews,
        {"train": train_path, "dev": dev_path, "test": test_path},
    )

    assert [review.review_id for review in splits["train"]] == ["r1"]
    assert [sentence.sentence_id for sentence in splits["train"][0].sentences] == [
        "r1:0",
        "r1:1",
    ]
    assert splits["dev"][0].review_id == "r2"


def test_original_split_alignment_does_not_assume_splits_follow_xml_order(tmp_path: Path):
    xml_path = tmp_path / "tiny.xml"
    xml_path.write_text(XML, encoding="utf-8")
    split_files = {
        "train": tmp_path / "train.txt",
        "dev": tmp_path / "dev.txt",
        "test": tmp_path / "test.txt",
    }
    split_files["train"].write_text(
        "Useful course.####[['course', 'course quality', 'conflict', 'NULL']]\n",
        encoding="utf-8",
    )
    split_files["dev"].write_text("A clean room.####[]\n", encoding="utf-8")
    split_files["test"].write_text("", encoding="utf-8")

    splits = load_original_splits(parse_oats_xml(xml_path).reviews, split_files)

    assert splits["train"][0].review_id == "r2"
    assert splits["dev"][0].review_id == "r1"


def test_original_split_alignment_rejects_unresolvable_duplicate_text(tmp_path: Path):
    xml_path = tmp_path / "duplicate.xml"
    xml_path.write_text(
        "<Reviews>"
        "<Review rid='r1'><sentences><sentence sid='r1:0'><text>Same text.</text>"
        "<Opinions /></sentence></sentences></Review>"
        "<Review rid='r2'><sentences><sentence sid='r2:0'><text>Same text.</text>"
        "<Opinions /></sentence></sentences></Review>"
        "</Reviews>",
        encoding="utf-8",
    )
    split_files = {
        "train": tmp_path / "train.txt",
        "dev": tmp_path / "dev.txt",
        "test": tmp_path / "test.txt",
    }
    split_files["train"].write_text("Same text.####[]\n", encoding="utf-8")
    split_files["dev"].write_text("", encoding="utf-8")
    split_files["test"].write_text("", encoding="utf-8")

    with pytest.raises(DataFormatError, match="matches multiple"):
        load_original_splits(parse_oats_xml(xml_path).reviews, split_files)


def test_hotels_resolver_prefers_corrected_xml_when_present(tmp_path: Path):
    hotels_quads = tmp_path / "hotels" / "quads"
    hotels_quads.mkdir(parents=True)
    original = hotels_quads / "Hotels.xml"
    corrected = hotels_quads / "Hotels_corr.xml"
    original.write_text("<Reviews />", encoding="utf-8")
    corrected.write_text("<Reviews />", encoding="utf-8")

    paths = resolve_domain_paths("hotels", tmp_path)

    assert paths.xml_path == corrected


@pytest.mark.parametrize(
    ("domain", "xml_filename"),
    [
        ("amazon_ff", "Amazon_FF.xml"),
        ("coursera", "Coursera.xml"),
        ("hotels", "Hotels.xml"),
    ],
)
def test_domain_resolver_maps_all_supported_domains(domain, xml_filename, tmp_path: Path):
    quads = tmp_path / domain / "quads"
    quads.mkdir(parents=True)
    expected_xml = quads / xml_filename
    expected_xml.write_text("<Reviews />", encoding="utf-8")

    paths = resolve_domain_paths(domain, tmp_path, prefer_corrected_hotels=False)

    assert paths.domain == domain
    assert paths.xml_path == expected_xml
    assert paths.split_files == {
        name: quads / f"{name}.txt" for name in ("train", "dev", "test")
    }
