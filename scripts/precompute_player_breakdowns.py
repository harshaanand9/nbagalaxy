#!/usr/bin/env python3
"""Precompute skill-breakdown and 3PT-breakdown payloads for every active guard.

Run from the project root:
    python3 scripts/precompute_player_breakdowns.py

Outputs:
    backend/data/player_skill_breakdowns.json
    backend/data/player_three_pt_breakdowns.json
    backend/data/player_breakdown_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app  # noqa: E402
from app import (  # noqa: E402
    BACKEND_DATA_DIR,
    DEFAULT_DATASET_PATH,
    PLAYER_COMPS_DATASET_PATH,
    EUCLIDEAN_KMEANS_LOCKED_K,
    LOWER_IS_BETTER_PERCENTILE_FEATURES,
    PERCENTILE_AND_BADGE_EXCLUDED_NAMES,
    SKILL_BREAKDOWN_EXCLUDED_FEATURES,
    SKILL_BREAKDOWN_GROUP_ORDER,
    THREE_PT_BREAKDOWN_GROUP_ORDER,
    THREE_PT_BREAKDOWN_LOWER_IS_BETTER_BY_GROUP,
    build_locked_euclidean_feature_signature,
    build_skill_breakdown_score_frames,
    get_cluster_title,
    get_locked_euclidean_kmeans_feature_columns,
    get_player_headshot_payload,
    prepare_cluster_runtime,
)


def make_json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and sets into plain JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(make_json_safe(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = make_json_safe(payload)
    path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")


def build_manifest(dataset_path: Path, breakdown_kind: str) -> Dict[str, Any]:
    dataset_mtime_ns = dataset_path.stat().st_mtime_ns if dataset_path.exists() else None
    return {
        "breakdown_kind": breakdown_kind,
        "dataset_path": str(dataset_path),
        "dataset_mtime_ns": dataset_mtime_ns,
        "algorithm": "kmeans",
        "distance_metric": "euclidean",
        "k": int(EUCLIDEAN_KMEANS_LOCKED_K),
        "feature_signature": build_locked_euclidean_feature_signature(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "precompute_script": "scripts/precompute_player_breakdowns.py",
    }


def build_precomputed_breakdowns_for_kind(runtime: Dict[str, Any], dataset_path: Path, breakdown_kind: str) -> Dict[str, Any]:
    guards = runtime["guards"].copy().reset_index(drop=True)
    labels = np.asarray(runtime["labels"], dtype=int)
    if guards.empty:
        raise ValueError("No active guard rows are available for breakdown precomputation.")

    component_score_frame, subsection_frames, used_group_features, missing_features, required_features = build_skill_breakdown_score_frames(
        guards=guards,
        breakdown_kind=breakdown_kind,
    )
    if missing_features:
        raise ValueError(f"Missing {breakdown_kind} feature columns: {missing_features}")

    group_order = THREE_PT_BREAKDOWN_GROUP_ORDER if breakdown_kind == "three_pt_breakdown" else SKILL_BREAKDOWN_GROUP_ORDER
    full_guard_mask = np.ones(len(guards), dtype=bool)

    def component_scores_for_row(row_index: int) -> Dict[str, float]:
        return {group_name: float(component_score_frame.at[row_index, group_name]) for group_name in group_order}

    def subsection_scores_for_row(row_index: int) -> Dict[str, Dict[str, float]]:
        if breakdown_kind == "three_pt_breakdown":
            return {
                "3PT Shooting Talent": {
                    group_name: float(component_score_frame.at[row_index, group_name])
                    for group_name in group_order
                }
            }
        return {
            group_name: {
                subsection_name: float(subsection_frames[group_name].at[row_index, subsection_name])
                for subsection_name in subsection_frames[group_name].columns
            }
            for group_name in group_order
            if group_name in subsection_frames
        }

    def component_median_scores(row_mask: np.ndarray) -> Dict[str, float]:
        selected_rows = component_score_frame.loc[row_mask, group_order]
        if selected_rows.empty:
            return {group_name: 0.0 for group_name in group_order}
        medians = selected_rows.median(axis=0, skipna=True).fillna(0.0)
        return {group_name: float(medians[group_name]) for group_name in group_order}

    def subsection_median_scores(row_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
        if breakdown_kind == "three_pt_breakdown":
            selected_rows = component_score_frame.loc[row_mask, group_order]
            medians = selected_rows.median(axis=0, skipna=True).fillna(0.0)
            return {"3PT Shooting Talent": {group_name: float(medians[group_name]) for group_name in group_order}}

        payload: Dict[str, Dict[str, float]] = {}
        for group_name, subsection_frame in subsection_frames.items():
            selected_rows = subsection_frame.loc[row_mask]
            if selected_rows.empty:
                medians = pd.Series(0.0, index=subsection_frame.columns)
            else:
                medians = selected_rows.median(axis=0, skipna=True).fillna(0.0)
            payload[group_name] = {column_name: float(medians[column_name]) for column_name in subsection_frame.columns}
        return payload

    def guard_median_payload_for_season(season: str) -> Dict[str, Any]:
        season_guard_mask = guards["Season"].astype(str).eq(str(season)).to_numpy()
        return {
            "label": f"Median Player {season}",
            "season": str(season),
            "scores": component_median_scores(season_guard_mask),
            "subsections": subsection_median_scores(season_guard_mask),
        }

    cluster_median_by_number: Dict[int, Dict[str, Any]] = {}
    for cluster_number in sorted(set(int(value) for value in labels.tolist())):
        cluster_mask = labels == cluster_number
        cluster_title = get_cluster_title(cluster_number, "kmeans", "euclidean")
        cluster_median_by_number[cluster_number] = {
            "label": f"{cluster_title} Median",
            "cluster_number": int(cluster_number),
            "cluster_title": cluster_title,
            "scores": component_median_scores(cluster_mask),
            "subsections": subsection_median_scores(cluster_mask),
        }

    score_logic = "Same-season guard percentile scoring with LeBron James, Scottie Barnes, and Ben Simmons excluded from percentile peer pools."
    if breakdown_kind == "skill_breakdown":
        score_logic += " ThreePT, MidRange, and RimPressure use median subsection scores; Playmaking averages subsection scores; D-LEBRON uses the same-season guard percentile of D-LEBRON."

    players: Dict[str, Dict[str, Any]] = {}
    for row_index, player_row in guards.iterrows():
        player_cluster_number = int(labels[int(row_index)])
        player_name = str(player_row["Player Name"])
        player_key = str(player_row["player_key"])
        cluster_title = get_cluster_title(player_cluster_number, "kmeans", "euclidean")
        player_season = str(player_row["Season"])
        players[player_key] = {
            "breakdown_kind": breakdown_kind,
            "algorithm": "kmeans",
            "distance_metric": "euclidean",
            "k": int(EUCLIDEAN_KMEANS_LOCKED_K),
            "player_key": player_key,
            "cluster_number": player_cluster_number,
            "cluster_title": cluster_title,
            "axes": group_order,
            "feature_groups": used_group_features,
            "required_features": required_features,
            "excluded_features": sorted(SKILL_BREAKDOWN_EXCLUDED_FEATURES) if breakdown_kind == "skill_breakdown" else [],
            "excluded_percentile_players": sorted(PERCENTILE_AND_BADGE_EXCLUDED_NAMES),
            "lower_is_better_percentile_features": sorted(LOWER_IS_BETTER_PERCENTILE_FEATURES.union({"pct_3fga_wide_open", "avg_closest_defender_3FGA"})),
            "local_lower_is_better_by_group": (
                {
                    group_name: sorted(list(feature_names))
                    for group_name, feature_names in THREE_PT_BREAKDOWN_LOWER_IS_BETTER_BY_GROUP.items()
                }
                if breakdown_kind == "three_pt_breakdown"
                else {}
            ),
            "local_percentile_rules_by_group": {},
            "score_logic": score_logic,
            "precomputed": True,
            "cache_source": "precomputed_breakdown_json",
            "player": {
                "label": player_name,
                "player_name": player_name,
                "season": player_season,
                "team": str(player_row["teams_played"]),
                "position": str(player_row["position"]),
                **get_player_headshot_payload(player_name),
                "scores": component_scores_for_row(int(row_index)),
                "subsections": subsection_scores_for_row(int(row_index)),
            },
            "cluster_median": cluster_median_by_number[player_cluster_number],
            "guard_median": guard_median_payload_for_season(player_season),
        }

    return {
        "manifest": build_manifest(dataset_path, breakdown_kind),
        "player_count": len(players),
        "players": players,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute player skill/3PT breakdown payloads.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to the main fullseasonfeatures_16_17_25_26.csv file used for all non-defense skill-breakdown logic")
    parser.add_argument("--dlebron-dataset", default=PLAYER_COMPS_DATASET_PATH, help="Path to fullseasonfeatures_player_comps_real.csv; only D-LEBRON is side-loaded from here for the defensive skill breakdown")
    parser.add_argument("--output-dir", default=str(BACKEND_DATA_DIR), help="Directory to write breakdown JSON files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser()
    dlebron_dataset_path = Path(args.dlebron_dataset).expanduser()
    backend_app.PLAYER_COMPS_DATASET_PATH = str(dlebron_dataset_path)
    backend_app._DLEBRON_SOURCE_CACHE = {"path": None, "mtime_ns": None, "lookup": None}
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not dlebron_dataset_path.exists():
        raise FileNotFoundError(f"D-LEBRON dataset file not found: {dlebron_dataset_path}")

    runtime = prepare_cluster_runtime(
        str(dataset_path),
        "kmeans",
        "euclidean",
        EUCLIDEAN_KMEANS_LOCKED_K,
        get_locked_euclidean_kmeans_feature_columns(raw=False),
    )

    skill_payload = build_precomputed_breakdowns_for_kind(runtime, dataset_path, "skill_breakdown")
    three_pt_payload = build_precomputed_breakdowns_for_kind(runtime, dataset_path, "three_pt_breakdown")

    skill_output_path = output_dir / "player_skill_breakdowns.json"
    three_pt_output_path = output_dir / "player_three_pt_breakdowns.json"
    manifest_output_path = output_dir / "player_breakdown_manifest.json"

    write_json(skill_output_path, skill_payload)
    write_json(three_pt_output_path, three_pt_payload)
    write_json(
        manifest_output_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path),
            "dataset_mtime_ns": dataset_path.stat().st_mtime_ns,
            "dlebron_dataset_path": str(dlebron_dataset_path),
            "dlebron_dataset_mtime_ns": dlebron_dataset_path.stat().st_mtime_ns,
            "algorithm": "kmeans",
            "distance_metric": "euclidean",
            "k": int(EUCLIDEAN_KMEANS_LOCKED_K),
            "feature_signature": build_locked_euclidean_feature_signature(),
            "files": {
                "skill_breakdown": str(skill_output_path),
                "three_pt_breakdown": str(three_pt_output_path),
            },
            "player_count": int(skill_payload["player_count"]),
        },
    )

    print(f"Wrote {skill_payload['player_count']:,} skill-breakdown payloads to {skill_output_path}")
    print(f"Wrote {three_pt_payload['player_count']:,} 3PT-breakdown payloads to {three_pt_output_path}")
    print(f"Wrote manifest to {manifest_output_path}")


if __name__ == "__main__":
    main()
