#!/usr/bin/env python3
"""Export effective mega blocks, subgroups, and underlying raw CSV features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.weighted_blocked_pca import (  # noqa: E402
    GUARDED_ARCHETYPE_BINS,
    GUARDED_POSITION_COMPONENTS,
    GUARDED_USAGE_BINS,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "weighted_blocked_pca_kmeans_k24_58_42_balanced20pct_recombined"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def raw_sources(model_feature: str) -> tuple[tuple[str, ...], str | None]:
    if model_feature == "stable_ast_tov_ratio":
        return (
            ("Stable Assists Per 75", "Stable TOV Per 75"),
            "ratio: Stable Assists Per 75 / Stable TOV Per 75",
        )
    if model_feature == "Guarded Position Spread":
        return (
            tuple(GUARDED_POSITION_COMPONENTS),
            "entropy percentile derived from raw guarded-position composition",
        )
    prefix = "__guard_position_composition_"
    if model_feature.startswith(prefix):
        index = int(model_feature.removeprefix(prefix))
        return (
            (GUARDED_POSITION_COMPONENTS[index],),
            "row-normalized raw guarded-position share",
        )
    prefix = "__guard_archetype_composition_"
    if model_feature.startswith(prefix):
        index = int(model_feature.removeprefix(prefix))
        name, members = list(GUARDED_ARCHETYPE_BINS.items())[index]
        return tuple(members), f"row-normalized {name} aggregate"
    prefix = "__guard_usage_composition_"
    if model_feature.startswith(prefix):
        index = int(model_feature.removeprefix(prefix))
        name, members = list(GUARDED_USAGE_BINS.items())[index]
        return tuple(members), f"row-normalized {name} aggregate"
    return (model_feature,), None


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.input_dir / "hierarchy_weight_manifest.json").read_text(encoding="utf-8")
    )
    effective_blocks = manifest["effective_blocks"]
    subgroups = manifest["subgroups"]
    unique_raw_features: set[str] = set()
    for subgroup in subgroups:
        for feature in subgroup["features"]:
            sources, _ = raw_sources(feature)
            unique_raw_features.update(sources)

    output = args.output or args.input_dir / "model_mega_blocks_subgroups_raw_features.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    active_domain = manifest.get("active_domain", "All")
    domain_weight_lines = (
        ["Offense weight: 0.580000", "Defense weight: 0.420000"]
        if active_domain == "All"
        else [f"Standalone {active_domain} model weight: 1.000000"]
    )
    lines = [
        "WEIGHTED BLOCKED-PCA MODEL FEATURE HIERARCHY",
        "=" * 80,
        "",
        f"Effective mega blocks: {len(effective_blocks)}",
        f"Subgroups: {len(subgroups)}",
        f"Model input features: {manifest['feature_count']}",
        f"Unique underlying raw CSV features: {len(unique_raw_features)}",
        f"Active clustering domain: {active_domain}",
        *domain_weight_lines,
        "",
        "An indented MODEL INPUT line appears only when the model feature is engineered.",
        "The RAW FEATURES beneath it are the actual source columns used to construct it.",
    ]

    for block_number, block in enumerate(effective_blocks, start=1):
        block_name = block["name"]
        block_subgroups = [row for row in subgroups if row["effective_block"] == block_name]
        lines.extend(
            [
                "",
                "=" * 80,
                f"MEGA BLOCK {block_number}: {block_name}",
                f"DOMAIN: {block['domain']}",
                f"TOTAL MODEL WEIGHT: {block['final_weight']:.6f}",
                "=" * 80,
            ]
        )
        for subgroup_number, subgroup in enumerate(block_subgroups, start=1):
            lines.extend(
                [
                    "",
                    f"  SUBGROUP {subgroup_number}: {subgroup['subgroup']}",
                    f"  SUBGROUP WEIGHT: {subgroup['subgroup_weight']:.6f}",
                    f"  RETAINED PCA COMPONENTS: {subgroup['retained_component_count']}",
                    "  RAW FEATURES:",
                ]
            )
            for model_feature in subgroup["features"]:
                sources, transformation = raw_sources(model_feature)
                if transformation is None:
                    lines.append(f"    - {sources[0]}")
                else:
                    lines.append(f"    MODEL INPUT: {model_feature}")
                    lines.append(f"      TRANSFORMATION: {transformation}")
                    for source in sources:
                        lines.append(f"      - {source}")

    lines.extend(["", "=" * 80, "UNIQUE RAW FEATURE LIST", "=" * 80])
    for feature in sorted(unique_raw_features, key=str.casefold):
        lines.append(f"- {feature}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
