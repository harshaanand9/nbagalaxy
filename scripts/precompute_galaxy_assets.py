#!/usr/bin/env python3
"""
Precompute the 3D galaxy assets from the same locked Euclidean KMeans truth space
used by the web app:

    season median imputation -> season z-score clipping -> blockwise PCA at 0.90
    explained variance -> equal block weighting -> Euclidean distance.

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

from app import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    EUCLIDEAN_KMEANS_LOCKED_K,
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
            "strongest_similarity_blocks": edge.get("strongest_similarity_blocks", ""),
            "biggest_difference_blocks": edge.get("biggest_difference_blocks", ""),
            "pipeline": "blocked_pca_090_euclidean",
            "pca_variance_target": 0.90,
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
    parser.add_argument("--output-dir", default=str(BACKEND_DATA_DIR), help="Directory to write precomputed assets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_meta = load_base_dataframe(args.dataset)
    base_guards = dataset_meta["guards"].copy()

    guards, truth_matrix, raw_labels, metric_meta = build_locked_euclidean_kmeans_space(base_guards)
    labels = raw_labels + 1
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
        "truth_space": "blocked_pca_090_weighted_euclidean",
        "metric_meta": metric_meta,
        "display_meta": display_meta,
        "coordinates": coordinate_rows,
        "galaxy": galaxy_payload,
    }

    write_json(output_dir / "galaxy_precomputed.json", full_payload)
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
