#!/usr/bin/env python3
"""Build the slim, self-contained dataset the v4 similarity engine reads.

The engine in ``backend/similarity_engine.py`` is a port of ``sim.ipynb`` and
expects the BBall Index feature table. That table lives outside this repo and
carries 346 columns plus two impact side-loads. This script merges what the
engine actually touches into one checked-in file:

    backend/data/similarity_model_dataset.csv

Row scope is the FULL BBall Index population (every player-season, 2015-16 on).
The engine fits on all of it -- within-season z-scores, subgroup PCA, peer
percentiles and the continuity learner are all population statistics, so
trimming rows here would silently change the model. Which rows may be RETURNED
as comps is a separate question, carried by the ``in_site_roster`` flag.

Usage:
    python3 scripts/build_similarity_dataset.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DATA_DIR = REPO_ROOT / "backend" / "data"

DEFAULT_SOURCE = Path("/Users/harsha/Desktop/testing_galaxy/bballindex_complete_dataset_with_pbp_midrange.csv")
DEFAULT_IMPACT = [
    Path("/Users/harsha/Desktop/PickPocketProjectOfficial/bballindex_full_features.csv"),
    # The canonical ``bballindex_missing_players_scraped.csv`` was deleted. This dated
    # backup carries identical O-LEBRON values for all 74 player-seasons that the
    # full-features file does not cover; the three surviving backups agree exactly.
    Path("/Users/harsha/Desktop/PickPocketProjectOfficial/bballindex_missing_players_scraped_backup_before_shot_location_features_20260609_165132.csv"),
]
DEFAULT_SITE_DATASET = BACKEND_DATA_DIR / "fullseasonfeatures_16_17_25_26.csv"
DEFAULT_ENGINE = REPO_ROOT / "backend" / "similarity_engine.py"

KEY_COLUMNS = ["player", "season", "team", "position", "age"]
# Pulled from the impact side-loads when absent from the main table.
IMPACT_COLUMNS = ["O-LEBRON", "Height", "Weight", "Minutes"]

NAME_SUFFIXES = ("iii", "iv", "ii", "jr", "sr", "v")


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z]", "", str(value).lower())


def base_name(value: object) -> str:
    """Suffix-insensitive name key: 'Marcus Morris Sr.' -> 'marcusmorris'."""
    normalized = normalize_name(value)
    for suffix in NAME_SUFFIXES:  # longest first, so 'iii' is not eaten by 'ii'
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 3:
            return normalized[: -len(suffix)]
    return normalized


def engine_referenced_columns(engine_path: Path, available: set[str]) -> set[str]:
    """Every string literal in the engine source that names a real dataset column.

    The engine's block hierarchy, identity baskets and role-gate inputs are all
    written as literal column names, so this recovers what it reads without
    importing the module (which would require the dataset to already exist).
    """
    source = engine_path.read_text(encoding="utf-8")
    literals = set(re.findall(r"'([^'\n]{2,80})'", source))
    literals |= set(re.findall(r'"([^"\n]{2,80})"', source))
    return {literal for literal in literals if literal in available}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the slim similarity-model dataset.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--impact", nargs="*", default=[str(path) for path in DEFAULT_IMPACT])
    parser.add_argument("--site-dataset", default=str(DEFAULT_SITE_DATASET))
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE))
    parser.add_argument("--output", default=str(BACKEND_DATA_DIR / "similarity_model_dataset.csv"))
    args = parser.parse_args()

    source_path = Path(args.source).expanduser()
    if not source_path.exists():
        sys.exit(f"Source dataset not found: {source_path}")

    frame = pd.read_csv(source_path, low_memory=False).reset_index(drop=True)
    row_count = len(frame)
    print(f"Source: {source_path.name}  ({row_count} rows x {frame.shape[1]} cols)")

    # --- impact side-load: only the columns the main table lacks ------------
    missing_impact = [column for column in IMPACT_COLUMNS if column not in frame.columns]
    if missing_impact:
        parts = []
        for impact_path in args.impact:
            path = Path(impact_path).expanduser()
            if not path.exists():
                print(f"  ! impact file missing, skipping: {path.name}")
                continue
            impact_frame = pd.read_csv(path, low_memory=False)
            usable = [column for column in missing_impact if column in impact_frame.columns]
            if usable:
                parts.append(impact_frame[["player", "season"] + usable])
        if not parts:
            sys.exit(f"No impact source supplied columns: {missing_impact}")
        merged_impact = pd.concat(parts, ignore_index=True).drop_duplicates(["player", "season"])
        frame = frame.merge(merged_impact, on=["player", "season"], how="left")
        if len(frame) != row_count:
            sys.exit("Impact join changed the row count")
        for column in missing_impact:
            filled = int(frame[column].notna().sum())
            print(f"  impact {column!r}: {filled}/{row_count} filled")

    # --- column selection ---------------------------------------------------
    available = set(frame.columns)
    engine_path = Path(args.engine).expanduser()
    if engine_path.exists():
        referenced = engine_referenced_columns(engine_path, available)
    else:
        print(f"  ! engine not found at {engine_path}; keeping every column")
        referenced = available

    keep = set(KEY_COLUMNS) | set(IMPACT_COLUMNS) | referenced
    keep &= available
    ordered = [column for column in frame.columns if column in keep]

    # --- site roster flag ---------------------------------------------------
    site_path = Path(args.site_dataset).expanduser()
    site = pd.read_csv(site_path, low_memory=False)
    site_keys = set(zip(site["Player Name"].map(base_name), site["Season"].astype(str)))
    roster_flag = [
        (base_name(player), str(season)) in site_keys
        for player, season in zip(frame["player"], frame["season"])
    ]

    slim = frame[ordered].copy()
    slim["in_site_roster"] = roster_flag

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(output_path, index=False)

    matched = int(slim["in_site_roster"].sum())
    print(f"\nWrote {output_path}")
    print(f"  rows      {len(slim)}  (fitting population)")
    print(f"  columns   {len(slim.columns)}  (from {frame.shape[1]})")
    print(f"  roster    {matched}/{len(site)} site player-seasons matched")
    print(f"  size      {output_path.stat().st_size / 1e6:.1f} MB")

    unmatched = sorted(
        site_keys - {(base_name(p), str(s)) for p, s in zip(frame["player"], frame["season"])}
    )
    if unmatched:
        print(f"  ! {len(unmatched)} site player-seasons absent from BBall Index:")
        for name, season in unmatched:
            print(f"      {name} {season}")
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(source_path),
                "impact_sources": [str(p) for p in args.impact],
                "rows": len(slim),
                "columns": len(slim.columns),
                "site_roster_matched": matched,
                "site_roster_total": int(len(site)),
                "unmatched_site_player_seasons": [list(item) for item in unmatched],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  manifest  {manifest_path.name}")


if __name__ == "__main__":
    main()
