"""Resolve immutable raw OATS inputs for each supported domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DOMAINS = ("amazon_ff", "coursera", "hotels")
SPLIT_NAMES = ("train", "dev", "test")
XML_FILENAMES = {
    "amazon_ff": "Amazon_FF.xml",
    "coursera": "Coursera.xml",
    "hotels": "Hotels.xml",
}


@dataclass(frozen=True)
class DomainPaths:
    domain: str
    domain_dir: Path
    xml_path: Path
    split_files: dict[str, Path]


def default_raw_root() -> Path:
    """Return the repository's bundled OATS ``data`` directory."""

    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "OATS-ABSA-Dataset" / "data"


def resolve_domain_paths(
    domain: str,
    raw_root: str | Path | None = None,
    *,
    prefer_corrected_hotels: bool = True,
) -> DomainPaths:
    domain = domain.lower()
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain {domain!r}; choose from {', '.join(DOMAINS)}")

    root = Path(raw_root) if raw_root is not None else default_raw_root()
    domain_dir = root / domain / "quads"
    original_xml = domain_dir / XML_FILENAMES[domain]
    corrected_xml = domain_dir / "Hotels_corr.xml"
    xml_path = (
        corrected_xml
        if domain == "hotels" and prefer_corrected_hotels and corrected_xml.is_file()
        else original_xml
    )
    split_files = {name: domain_dir / f"{name}.txt" for name in SPLIT_NAMES}
    return DomainPaths(domain, domain_dir, xml_path, split_files)
