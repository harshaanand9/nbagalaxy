#!/usr/bin/env python3
"""Precompute the default first-load payload consumed by the frontend.

This turns the expensive first-run API work into a static JSON asset:
config, locked Euclidean K-Means galaxy data, every player detail panel, and
every cluster report for the default galaxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app  # noqa: E402


DEFAULT_DATASET = BACKEND_DIR / "data" / "fullseasonfeatures_16_17_25_26.csv"
DEFAULT_DLEBRON_DATASET = BACKEND_DIR / "data" / "fullseasonfeatures_player_comps_real.csv"
DEFAULT_PULLUP_DATASET = Path("/Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_13_14_25_26_pullup.csv")
DEFAULT_OUTPUT = REPO_ROOT / "frontend" / "public" / "precomputed" / "default_bootstrap.json"
DEFAULT_FULL_OUTPUT = REPO_ROOT / "frontend" / "public" / "precomputed" / "default_bootstrap.full.json"


def first_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded.resolve()
    return paths[0].expanduser().resolve()


def write_compact_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute the default frontend bootstrap payload.")
    parser.add_argument(
        "--dataset",
        default=str(first_existing_path([Path(backend_app.DEFAULT_DATASET_PATH), DEFAULT_DATASET])),
        help="Path to fullseasonfeatures_16_17_25_26.csv",
    )
    parser.add_argument(
        "--dlebron-dataset",
        default=str(first_existing_path([Path(backend_app.PLAYER_COMPS_DATASET_PATH), DEFAULT_DLEBRON_DATASET])),
        help="Path to fullseasonfeatures_player_comps_real.csv",
    )
    parser.add_argument(
        "--pullup-dataset",
        default=str(first_existing_path([Path(backend_app.PULLUP_DATASET_PATH), DEFAULT_PULLUP_DATASET])),
        help="Path to fullseasonfeatures_13_14_25_26_pullup.csv",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Slim bootstrap written for deploy (config + cluster only).",
    )
    parser.add_argument(
        "--full-output",
        default=str(DEFAULT_FULL_OUTPUT),
        help="Full bootstrap written for local dev, including click-time caches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    dlebron_dataset_path = Path(args.dlebron_dataset).expanduser().resolve()
    pullup_dataset_path = Path(args.pullup_dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    full_output_path = Path(args.full_output).expanduser().resolve()

    backend_app.DEFAULT_DATASET_PATH = str(dataset_path)
    backend_app.PLAYER_COMPS_DATASET_PATH = str(dlebron_dataset_path)
    backend_app.PULLUP_DATASET_PATH = str(pullup_dataset_path)

    algorithm = "kmeans"
    distance_metric = "euclidean"
    k = backend_app.EUCLIDEAN_KMEANS_LOCKED_K
    features = backend_app.get_locked_euclidean_kmeans_feature_columns(raw=False)

    print("Building default cluster payload...")
    cluster_payload = backend_app.compute_cluster_payload(
        str(dataset_path),
        algorithm,
        distance_metric,
        k,
        features,
    )

    print("Building player detail panels...")
    player_details_by_key = {}
    for point in cluster_payload.get("points", []):
        player_key = str(point.get("player_key", ""))
        if not player_key:
            continue
        player_details_by_key[player_key] = backend_app.build_player_detail_payload(player_key)

    print("Building cluster reports...")
    cluster_reports_by_number = {}
    for cluster_number in range(1, int(cluster_payload.get("k", k)) + 1):
        cluster_reports_by_number[str(cluster_number)] = backend_app.compute_cluster_report_payload(
            str(dataset_path),
            algorithm,
            distance_metric,
            k,
            features,
            cluster_number,
        )

    payload = {
        "manifest": {
            "kind": "default_bootstrap",
            "app_version": backend_app.APP_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path),
            "dataset_mtime_ns": dataset_path.stat().st_mtime_ns,
            "algorithm": algorithm,
            "distance_metric": distance_metric,
            "k": int(k),
            "feature_signature": backend_app.build_locked_euclidean_feature_signature(),
            "player_count": len(cluster_payload.get("points", [])),
            "player_detail_count": len(player_details_by_key),
            "cluster_report_count": len(cluster_reports_by_number),
        },
        "config": backend_app.get_config(),
        "cluster": cluster_payload,
        "player_details_by_key": player_details_by_key,
        "cluster_reports_by_number": cluster_reports_by_number,
    }

    # The deployed bootstrap carries only what the galaxy needs to draw itself.
    # player_details_by_key and cluster_reports_by_number are click-time caches --
    # App.jsx reads them with `?? {}` and falls back to the API when they are
    # absent -- and together they push the file past GitHub's 100 MB limit, which
    # is what Vercel deploys from. The full payload stays local and gitignored so
    # `npm run dev` needs no backend running.
    CACHE_KEYS = ("player_details_by_key", "cluster_reports_by_number")
    slim_payload = {key: value for key, value in payload.items() if key not in CACHE_KEYS}
    slim_payload["manifest"] = {**payload["manifest"], "payload_scope": "slim"}
    payload["manifest"] = {**payload["manifest"], "payload_scope": "full"}

    write_compact_json(output_path, slim_payload)
    write_compact_json(full_output_path, payload)
    slim_mb = output_path.stat().st_size / 1048576
    full_mb = full_output_path.stat().st_size / 1048576
    print(f"Wrote slim bootstrap (deployed) to: {output_path}  [{slim_mb:.1f} MB]")
    print(f"Wrote full bootstrap (local dev)  to: {full_output_path}  [{full_mb:.1f} MB]")
    print(f"Players: {len(player_details_by_key)}")
    print(f"Cluster reports: {len(cluster_reports_by_number)}")


if __name__ == "__main__":
    main()
