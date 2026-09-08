#!/usr/bin/env python3
"""Per-component similarity and presence-filtered shared concepts.

Adds two things to what the v4 model already publishes, without touching how it
ranks or scores anything:

  components  -- 3PT / MidRange / RimPressure / Playmaking / Playtypes / Defense
                 similarity for each comp, for the site's Advanced Mode.
  shared      -- the most-alike blocks, filtered to concepts BOTH players
                 actually do. `most_alike_block_ids` ranks by the block's
                 contribution to the pair distance, which is small both when two
                 players genuinely match and when neither does the thing at all,
                 so guard-vs-guard pairs surfaced "P&R Roll Man" and big-vs-big
                 pairs surfaced "Perimeter Defense Matchups". A block now has to
                 clear league average for BOTH players to be shown.

Both are derived from similarity_engine.block_decomposition, which is the
model's own per-block split of the pair distance. Nothing here feeds back into
comp selection, ranking or the off/def/overall scores.

Needs the engine, so run it with an interpreter that has matplotlib:

    python3 scripts/precompute_similarity_components.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import similarity_engine as E  # noqa: E402

V4_PATH = BACKEND_DIR / "data" / "similarity_v4.json"
OUT_PATH = BACKEND_DIR / "data" / "similarity_components.json"

# The six components the site reports, over the engine's 34 blocks. Touches,
# Ball Dominance, Dribbling Tendencies and Guarded By Data sit under Playtypes
# to match the locked clustering's feature groups, which put usage and shot-diet
# context there rather than under Playmaking.
COMPONENTS: Dict[str, List[str]] = {
    "ThreePT": [
        "3PT Shooting: Pull-Up + Self Created 3PT", "3PT Shooting: C+S + Off-Ball 3PT",
    ],
    "MidRange": ["Pull Up Midrange", "Non Paint Midrange", "Short Midrange", "Long Midrange"],
    "RimPressure": [
        "Rim", "Paint - Non Rim", "Drive Tendencies", "Free Throw Generation", "Offensive Rebounding",
    ],
    "Playmaking": ["Playmaking Volume", "Playmaking Efficiency"],
    "Playtypes": [
        "Cut", "Handoff", "Isolations", "Off-Ball Screens", "P&R Ball Handler", "P&R Roll Man",
        "Spot Up", "Transition", "Post Ups", "Touches", "Ball Dominance", "Dribbling Tendencies",
        "Guarded By Data",
    ],
    "Defense": [
        "Perimeter Defense", "Perimeter Defense Matchups", "Paint Defense", "Paint Defense Matchups",
        "Defensive Matchups", "Defensive Versatility", "Defensive Advanced / Impact Stats",
        "Defensive Rebounding",
    ],
}

DOMAINS = ("overall", "offense", "defense")
SHARED_TOP_N = 3
# Population median for the score transform, sampled rather than computed over all
# 3206^2 pairs. 200 x 400 = 80k pairs moves the median by <1% between seeds.
CALIBRATION_SOURCES = 200
CALIBRATION_TARGETS = 400


def build_presence() -> np.ndarray:
    """Signed within-season z per player per block: how much they actually do it.

    Averages the block's own raw features, standardized inside each season so era
    and pace do not decide presence. >= 0 means at or above the league average
    for that season, which is the bar a block must clear for both players.
    """
    df = E.df
    presence = np.full((len(df), len(E.BLOCK_NAMES)), np.nan)
    seasons = df["season"]
    for block_index, block_name in enumerate(E.BLOCK_NAMES):
        columns = [c for cols in E.BLOCKS[block_name].values() for c in cols if c in df.columns]
        if not columns:
            continue
        frame = df[columns]
        means = frame.groupby(seasons).transform("mean")
        stds = frame.groupby(seasons).transform("std").replace(0, np.nan)
        presence[:, block_index] = ((frame - means) / stds).mean(axis=1).values
    return presence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4", default=str(V4_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--limit", type=int, default=0, help="Only process N players (smoke test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v4 = json.loads(Path(args.v4).read_text())
    player_keys = v4["player_keys"]
    players = v4["players"]

    block_names = list(E.BLOCK_NAMES)
    block_index_by_name = {name: i for i, name in enumerate(block_names)}
    for component, members in COMPONENTS.items():
        missing = [m for m in members if m not in block_index_by_name]
        if missing:
            raise SystemExit(f"{component} names unknown blocks: {missing}")
    covered = [b for members in COMPONENTS.values() for b in members]
    if sorted(covered) != sorted(block_names):
        raise SystemExit("component map must partition all 34 blocks exactly once")
    component_ids = {c: [block_index_by_name[m] for m in members] for c, members in COMPONENTS.items()}

    print("Building presence matrix ...")
    presence = build_presence()

    # engine row for every key the payload knows about
    row_of: Dict[str, int] = {}
    for key in player_keys:
        name, season = key.split("||")[0], key.split("||")[1]
        try:
            row_of[key] = int(E._resolve(name, season))
        except Exception:
            continue
    print(f"resolved {len(row_of)}/{len(player_keys)} player rows")

    todo = list(players.items())[: args.limit] if args.limit else list(players.items())
    raw: Dict[str, Dict[str, List[dict]]] = {}
    started = time.time()

    for i, (source_key, entry) in enumerate(todo, 1):
        source_row = row_of.get(source_key)
        if source_row is None:
            continue
        comps = entry.get("comps") or {}
        wanted = {}
        for domain in DOMAINS:
            for record in comps.get(domain, []) or []:
                target_key = player_keys[record[0]]
                target_row = row_of.get(target_key)
                if target_row is not None:
                    wanted[record[0]] = target_row
        if not wanted:
            continue
        decomposition = E.block_decomposition(source_row, list(wanted.values()))

        per_target = {}
        for target_index, target_row in wanted.items():
            blocks = decomposition.get(target_row)
            if not blocks:
                continue
            distances = {}
            for component, ids in component_ids.items():
                contribution = sum(blocks[block_names[b]]["contribution"] for b in ids)
                weight = sum(blocks[block_names[b]]["pair_weight"] for b in ids)
                distances[component] = float(np.sqrt(contribution / weight)) if weight > 1e-12 else 0.0
            order = sorted(range(len(block_names)), key=lambda b: blocks[block_names[b]]["divergence"])
            shared = [
                b for b in order
                if np.isfinite(presence[source_row, b]) and np.isfinite(presence[target_row, b])
                and presence[source_row, b] >= 0 and presence[target_row, b] >= 0
            ][:SHARED_TOP_N]
            per_target[target_index] = {"d": distances, "shared": shared}

        for domain in DOMAINS:
            for record in comps.get(domain, []) or []:
                if record[0] in per_target:
                    raw.setdefault(source_key, {}).setdefault(domain, []).append(
                        {"target_index": record[0], **per_target[record[0]]}
                    )

        if i % 250 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  ({i/max(time.time()-started,1e-6):.0f}/s)")

    # Calibrate each component the way the engine calibrates overall similarity:
    # 100 / (1 + (d / median d)^2). percentile_similarity takes that median over
    # the FULL candidate field, not over the shortlist, so calibrating on the
    # stored comps -- which are all top-10 matches -- would put a typical comp at
    # 50 and leave these six unable to be read next to the domain scores. Sample
    # random player-vs-random-candidate pairs to recover the population median.
    print("\nCalibrating against random pairs ...")
    rng = np.random.default_rng(0)
    rows_all = np.array(sorted(set(row_of.values())), dtype=int)
    sample_sources = rng.choice(rows_all, size=min(CALIBRATION_SOURCES, rows_all.size), replace=False)
    pooled = {component: [] for component in COMPONENTS}
    for source_row in sample_sources:
        targets = rng.choice(rows_all, size=min(CALIBRATION_TARGETS, rows_all.size), replace=False)
        targets = [int(t) for t in targets if int(t) != int(source_row)]
        decomposition = E.block_decomposition(int(source_row), targets)
        for blocks in decomposition.values():
            for component, ids in component_ids.items():
                contribution = sum(blocks[block_names[b]]["contribution"] for b in ids)
                weight = sum(blocks[block_names[b]]["pair_weight"] for b in ids)
                if weight > 1e-12:
                    pooled[component].append(np.sqrt(contribution / weight))
    scales = {}
    for component in COMPONENTS:
        values = np.array(pooled[component], dtype=float)
        positive = values[np.isfinite(values) & (values > 0)]
        scales[component] = float(np.median(positive)) if positive.size else 1.0
        print(f"  {component:12} median d = {scales[component]:.4f}  (n={positive.size})")

    out = {"meta": {"components": {c: list(m) for c, m in COMPONENTS.items()},
                    "scales": scales, "shared_top_n": SHARED_TOP_N,
                    "presence_rule": "both players >= season-average z in the block"},
           "block_names": block_names,
           "players": {}}
    for source_key, domains in raw.items():
        out["players"][source_key] = {
            domain: [
                {"target_index": row["target_index"],
                 "components": {c: round(100.0 / (1.0 + (row["d"][c] / scales[c]) ** 2), 1) for c in COMPONENTS},
                 "shared_block_ids": row["shared"]}
                for row in rows
            ]
            for domain, rows in domains.items()
        }

    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    size = Path(args.out).stat().st_size / 1048576
    print(f"\nWrote {args.out} [{size:.1f} MB] for {len(out['players'])} players in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
