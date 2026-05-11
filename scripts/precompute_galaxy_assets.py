#!/usr/bin/env python3
"""
Precompute the Galaxy similarity assets from the same locked KMeans truth
space used by the web app:

    per-game volume derivation where applicable -> season median imputation
    -> season guard z-score clipping -> no PCA -> equal block weighting
    -> cosine distance, with Luka Doncic comps using Euclidean distance.

By default, this preserves an existing galaxy_precomputed.json coordinate
embedding when the player keys still match, so similarity logic can be
refreshed without moving the visual Galaxy.

Outputs:
    backend/data/galaxy_precomputed.json
    backend/data/similar_players_precomputed_production.csv
    backend/data/archetype_edges.csv
    backend/data/cluster_medoids.csv
    backend/data/archetype_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app  # noqa: E402
from app import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    EUCLIDEAN_KMEANS_LOCKED_K,
    EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
    EUCLIDEAN_KMEANS_LOCKED_SPACE_TRANSFORM,
    BACKEND_DATA_DIR,
    build_galaxy_payload,
    build_locked_euclidean_feature_signature,
    build_locked_euclidean_kmeans_space,
    compute_galaxy_display_coordinates,
    get_locked_euclidean_kmeans_feature_columns,
    load_base_dataframe,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_display_coordinates(
    output_path: Path,
    guards: pd.DataFrame,
) -> tuple[Any, Any] | None:
    if not output_path.exists():
        return None

    try:
        payload = read_json(output_path)
    except Exception:
        return None

    coordinate_rows = payload.get("coordinates", [])
    if not isinstance(coordinate_rows, list):
        return None

    coordinate_by_key = {
        str(row.get("player_key")): row
        for row in coordinate_rows
        if isinstance(row, dict) and row.get("player_key") is not None
    }
    guard_keys = guards["player_key"].astype(str).tolist()
    if set(guard_keys) != set(coordinate_by_key.keys()):
        return None

    import numpy as np

    coordinates = np.asarray(
        [
            [
                float(coordinate_by_key[player_key].get("galaxy_x", 0.0)),
                float(coordinate_by_key[player_key].get("galaxy_y", 0.0)),
                float(coordinate_by_key[player_key].get("galaxy_z", 0.0)),
            ]
            for player_key in guard_keys
        ],
        dtype=float,
    )
    display_meta = payload.get("display_meta", {})
    if not isinstance(display_meta, dict):
        display_meta = {}
    display_meta = {**display_meta, "coordinates_preserved": True}
    return coordinates, display_meta


def build_similar_players_frame(galaxy_payload: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for edge in galaxy_payload.get("similarity_edges", []):
        block_scores = edge.get("block_scores", {})
        if not isinstance(block_scores, dict):
            block_scores = {}

        row: Dict[str, Any] = {
            "player_name": edge.get("source_player_name", ""),
            "season": edge.get("source_season", ""),
            "team": edge.get("source_team", ""),
            "position": edge.get("source_position", ""),
            "source_player_season_id": edge.get("source", ""),
            "related_player_name": edge.get("target_player_name", ""),
            "related_season": edge.get("target_season", ""),
            "related_team": edge.get("target_team", ""),
            "related_position": edge.get("target_position", ""),
            "related_player_season_id": edge.get("target", ""),
            "rank": edge.get("rank", 0),
            "cluster": edge.get("source_cluster", 0),
            "cluster_number": edge.get("source_cluster", 0),
            "source_cluster_number": edge.get("source_cluster", 0),
            "related_cluster": edge.get("target_cluster", 0),
            "related_cluster_number": edge.get("target_cluster", 0),
            "same_cluster": edge.get("same_cluster", False),
            "same_archetype": edge.get("same_cluster", False),
            "overall_distance": edge.get("truth_distance", 0.0),
            "overall_similarity_score": edge.get("similarity_score", 0.0),
            "similarity_distance_metric": edge.get("similarity_distance_metric", ""),
            "similarity_metric_used": edge.get("similarity_metric_used", ""),
            "strongest_similarity_blocks": edge.get("strongest_similarity_blocks", ""),
            "biggest_difference_blocks": edge.get("biggest_difference_blocks", ""),
            "pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
            "k": EUCLIDEAN_KMEANS_LOCKED_K,
            "feature_signature": build_locked_euclidean_feature_signature(),
        }

        for block_name, score_payload in block_scores.items():
            if not isinstance(score_payload, dict):
                continue
            column_prefix = str(block_name).lower()
            row[f"{column_prefix}_distance"] = score_payload.get("distance", 0.0)
            row[f"{column_prefix}_similarity_score"] = score_payload.get("similarity_score", 0.0)

        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute 3D galaxy assets for the locked KMeans website mode.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to fullseasonfeatures_16_17_25_26.csv")
    parser.add_argument("--dlebron-dataset", default=backend_app.PLAYER_COMPS_DATASET_PATH, help="Path to fullseasonfeatures_player_comps_real.csv for D-LEBRON and passing side-loads")
    parser.add_argument("--pullup-dataset", default=backend_app.PULLUP_DATASET_PATH, help="Path to fullseasonfeatures_13_14_25_26_pullup.csv for pull-up 2PA/game side-loads")
    parser.add_argument("--output-dir", default=str(BACKEND_DATA_DIR), help="Directory to write precomputed assets")
    parser.add_argument("--refresh-display-coordinates", action="store_true", help="Recompute the 3D display coordinates instead of preserving the existing embedding")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    galaxy_output_path = output_dir / "galaxy_precomputed.json"
    backend_app.PLAYER_COMPS_DATASET_PATH = str(Path(args.dlebron_dataset).expanduser().resolve())
    backend_app.PULLUP_DATASET_PATH = str(Path(args.pullup_dataset).expanduser().resolve())
    dataset_meta = load_base_dataframe(args.dataset)
    base_guards = dataset_meta["guards"].copy()

    guards, truth_matrix, raw_labels, metric_meta = build_locked_euclidean_kmeans_space(base_guards)
    labels = raw_labels + 1
    preserved_display_payload = None if args.refresh_display_coordinates else load_existing_display_coordinates(galaxy_output_path, guards)
    if preserved_display_payload is not None:
        display_coordinates, display_meta = preserved_display_payload
    else:
        display_coordinates, display_meta = compute_galaxy_display_coordinates(truth_matrix)

    galaxy_payload = build_galaxy_payload(
        guards=guards,
        X_metric=truth_matrix,
        labels=labels,
        display_coordinates=display_coordinates,
        display_meta=display_meta,
        algorithm="kmeans",
        distance_metric="euclidean",
        metric_meta=metric_meta,
        include_block_details=True,
    )

    coordinate_rows = []
    for row_index, row in guards.reset_index(drop=True).iterrows():
        coordinate_rows.append(
            {
                "player_key": row["player_key"],
                "player_name": row["Player Name"],
                "season": row["Season"],
                "teams_played": row["teams_played"],
                "position": row["position"],
                "cluster": int(labels[row_index]),
                "galaxy_x": float(display_coordinates[row_index, 0]),
                "galaxy_y": float(display_coordinates[row_index, 1]),
                "galaxy_z": float(display_coordinates[row_index, 2]),
            }
        )

    full_payload = {
        "truth_space": EUCLIDEAN_KMEANS_LOCKED_SPACE_TRANSFORM,
        "metric_meta": metric_meta,
        "display_meta": display_meta,
        "coordinates": coordinate_rows,
        "galaxy": galaxy_payload,
    }

    write_json(galaxy_output_path, full_payload)
    write_json(output_dir / "archetype_labels.json", galaxy_payload.get("archetype_labels", []))

    pd.DataFrame(galaxy_payload.get("cluster_edges", [])).to_csv(output_dir / "archetype_edges.csv", index=False)
    pd.DataFrame(galaxy_payload.get("cluster_medoids", [])).to_csv(output_dir / "cluster_medoids.csv", index=False)
    build_similar_players_frame(galaxy_payload).to_csv(output_dir / "similar_players_precomputed_production.csv", index=False)

    print(f"Wrote galaxy assets to: {output_dir}")
    print(f"Rows: {len(coordinate_rows)}")
    print(f"Similarity edges: {len(galaxy_payload.get('similarity_edges', []))}")
    print(f"Archetype edges: {len(galaxy_payload.get('cluster_edges', []))}")


if __name__ == "__main__":
    main()
