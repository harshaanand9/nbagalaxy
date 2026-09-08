#!/usr/bin/env python3
"""Precompute per-player static assets so the site never waits on the backend.

The deployed frontend used to answer every click with an API call. That put a
free-tier backend in the hot path of the UI: cold starts stalled the panel, and
a backend running older data 404'd for players it had never heard of.

This writes what those endpoints return as flat files under
``frontend/public/precomputed``, served straight from the CDN:

    players/<slug>.json   detail panel + skill breakdown + 3PT breakdown
    comps/<slug>.json     the SIMILAR_PLAYERS view
    cluster_reports/<n>.json
    comparison_options.json

``<slug>`` is the player_key with every run of non-alphanumerics collapsed to
``_``; the script asserts the mapping is collision-free before writing anything.
App.jsx derives the same slug, so no index file is needed.

The detail panels and cluster reports are lifted from default_bootstrap.full.json
rather than recomputed -- precompute_default_bootstrap.py already built them.

    python3 scripts/precompute_static_player_assets.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app  # noqa: E402

PRECOMPUTED_DIR = REPO_ROOT / "frontend" / "public" / "precomputed"
FULL_BOOTSTRAP = PRECOMPUTED_DIR / "default_bootstrap.full.json"


def slugify(player_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", player_key).strip("_")


def write_json(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-bootstrap", default=str(FULL_BOOTSTRAP))
    parser.add_argument("--out-dir", default=str(PRECOMPUTED_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Only emit N players (smoke test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    full_path = Path(args.full_bootstrap).expanduser().resolve()
    if not full_path.exists():
        raise SystemExit(
            f"{full_path} not found. Run scripts/precompute_default_bootstrap.py first."
        )

    print(f"Reading {full_path.name} ...")
    full = json.loads(full_path.read_text())
    points = full["cluster"]["points"]
    details = full.get("player_details_by_key", {})
    reports = full.get("cluster_reports_by_number", {})

    slug_by_key = {p["player_key"]: slugify(p["player_key"]) for p in points}
    if len(set(slug_by_key.values())) != len(slug_by_key):
        raise SystemExit("slug collision -- refusing to write ambiguous filenames")
    print(f"{len(points)} players, {len(set(slug_by_key.values()))} unique slugs")

    k = backend_app.EUCLIDEAN_KMEANS_LOCKED_K
    features = backend_app.get_locked_euclidean_kmeans_feature_columns(raw=False)
    dataset = backend_app.DEFAULT_DATASET_PATH

    # Hoisted once. build_similar_players_response_from_galaxy rebuilds these per
    # call -- including a 3212-row DataFrame.iterrows() -- which costs ~3.4s each
    # and would turn this script into a three-hour job.
    print("Preparing cluster runtime ...")
    point_by_key = {str(p.get("player_key")): p for p in points}
    v4_payload = backend_app.load_similarity_v4_payload()
    v4_players = (v4_payload or {}).get("players", {})
    print(f"v4 similarity entries: {len(v4_players)}")

    todo = points[: args.limit] if args.limit else points
    counts = {"players": 0, "comps": 0, "skill_failed": 0, "threept_failed": 0, "comps_missing": 0}
    bytes_players = bytes_comps = 0
    started = time.time()

    for i, point in enumerate(todo, 1):
        key = str(point["player_key"])
        slug = slug_by_key[key]

        skill = three_pt = None
        try:
            skill = backend_app.build_skill_breakdown_payload(dataset, "kmeans", "euclidean", k, features, key)
        except Exception:
            counts["skill_failed"] += 1
        try:
            three_pt = backend_app.build_three_pt_breakdown_payload(dataset, "kmeans", "euclidean", k, features, key)
        except Exception:
            counts["threept_failed"] += 1

        bytes_players += write_json(
            out_dir / "players" / f"{slug}.json",
            {"player_key": key, "detail": details.get(key), "skill": skill, "three_pt": three_pt},
        )
        counts["players"] += 1

        entry = v4_players.get(key)
        if entry is not None:
            comps = backend_app.build_similar_players_response_v4(
                source_point=point, source_key=key, entry=entry,
                payload=v4_payload, point_by_key=point_by_key,
            )
            bytes_comps += write_json(out_dir / "comps" / f"{slug}.json", comps)
            counts["comps"] += 1
        else:
            counts["comps_missing"] += 1

        if i % 250 == 0 or i == len(todo):
            rate = i / max(time.time() - started, 1e-6)
            print(f"  {i}/{len(todo)}  ({rate:.0f}/s)")

    for number, report in reports.items():
        write_json(out_dir / "cluster_reports" / f"{number}.json", report)

    try:
        write_json(out_dir / "comparison_options.json", backend_app.build_player_comparison_options_payload())
    except Exception as exc:  # non-fatal: the comparison tool keeps its API path
        print(f"WARNING: comparison options not written ({type(exc).__name__}: {exc})")

    print()
    print(f"players/         {counts['players']:>5} files  {bytes_players/1048576:8.1f} MB")
    print(f"comps/           {counts['comps']:>5} files  {bytes_comps/1048576:8.1f} MB")
    print(f"cluster_reports/ {len(reports):>5} files")
    print(f"skill missing: {counts['skill_failed']}, 3PT missing: {counts['threept_failed']}, comps missing: {counts['comps_missing']}")
    print(f"done in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
