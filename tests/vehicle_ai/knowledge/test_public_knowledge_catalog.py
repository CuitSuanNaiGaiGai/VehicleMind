"""The committed corpus is small, sourced, and scoped for retrieval."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "config/knowledge/source_catalog.yaml"


def test_public_catalog_has_20_valid_sourced_entries() -> None:
    sources = KnowledgeCatalog().load(MANIFEST)

    assert len(sources) >= 20
    assert len({source.source_id for source in sources}) == len(sources)
    assert all(source.text.strip() for source in sources)
    assert all(
        source.section and source.version and source.license_note for source in sources
    )
    assert all(urlsplit(source.source_uri).scheme == "https" for source in sources)


def test_public_catalog_has_distinct_common_and_demo_sources() -> None:
    sources = KnowledgeCatalog().load(MANIFEST)
    common = {
        source.source_id for source in sources if source.profile == "vehicle_common"
    }
    demo = {
        source.source_id for source in sources if source.profile == "vehiclemind_demo"
    }

    assert common == {f"K{number:03d}" for number in range(1, 11)}
    assert demo == {f"K{number:03d}" for number in range(11, 21)}
