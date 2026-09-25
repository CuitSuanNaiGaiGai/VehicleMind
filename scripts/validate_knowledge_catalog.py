"""Validate the public A3 source corpus and print its reproducible identity."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "config/knowledge/source_catalog.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 A3 知识来源、范围和内容哈希")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    sources = KnowledgeCatalog().load(args.manifest)
    if len(sources) < 20:
        raise ValueError("A3 public catalog requires at least 20 sources")
    counts = {
        profile: sum(source.profile == profile for source in sources)
        for profile in ("vehicle_common", "vehiclemind_demo")
    }
    if min(counts.values()) < 8:
        raise ValueError("A3 requires at least 8 sources in each profile class")
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(
        f"sources={len(sources)} common={counts['vehicle_common']} "
        f"demo={counts['vehiclemind_demo']} manifest_sha256={manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
