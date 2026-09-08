#!/usr/bin/env python3
"""Precompute player comparisons with the v4 similarity engine.

Runs ``backend/similarity_engine.py`` -- the port of sim.ipynb -- over every
player-season in the site's galaxy and writes:

    backend/data/similarity_v4.json          comps + attention, keyed by player_key
    backend/data/similarity_v4_comps.csv     flat comp table (inspection / analysis)

The model fits on the full BBall Index population; only the CANDIDATE pool is
restricted to the site's roster, so every comp returned is a player-season that
exists in the galaxy and can be clicked.

Attention tables are stored as arrays aligned to a shared label list rather than
as per-player objects, which keeps the payload around 10 MB instead of 100 MB.

Usage:
    python3 scripts/precompute_similarity_v4.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

TOP_N = 10
NAME_SUFFIXES = ("iii", "iv", "ii", "jr", "sr", "v")


def base_name(value: object) -> str:
    normalized = re.sub(r"[^a-z]", "", str(value).lower())
    for suffix in NAME_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 3:
            return normalized[: -len(suffix)]
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute v4 similarity assets.")
    parser.add_argument("--dataset", default=None, help="Site feature table (defaults to app's)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    import app as backend_app
    from app import (
        BACKEND_DATA_DIR,
        DEFAULT_DATASET_PATH,
        build_locked_euclidean_kmeans_space,
        load_base_dataframe,
    )

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else BACKEND_DATA_DIR
    dataset_path = args.dataset or DEFAULT_DATASET_PATH

    print("Loading site roster ...")
    dataset_meta = load_base_dataframe(dataset_path)
    guards, _matrix, raw_labels, _meta = build_locked_euclidean_kmeans_space(dataset_meta["guards"].copy())
    guards = guards.reset_index(drop=True)
    labels = raw_labels + 1
    print(f"  {len(guards)} player-seasons in the galaxy")

    print("Fitting the v4 similarity model (sim.ipynb port) ...")
    started = time.time()
    import similarity_engine as engine
    print(f"  fitted in {time.time() - started:.1f}s: "
          f"{engine.N} player-seasons, {engine.G} blocks, {engine.NSUB} subgroups")

    # --- map engine rows <-> site rows -------------------------------------
    engine_row_by_key = {
        (base_name(player), str(season)): index
        for index, (player, season) in enumerate(zip(engine.df["player"], engine.df["season"]))
    }
    site_key_by_engine_row: dict[int, str] = {}
    engine_row_by_site_row: dict[int, int] = {}
    unmatched: list[str] = []
    for site_row, row in guards.iterrows():
        lookup = (base_name(row["Player Name"]), str(row["Season"]))
        engine_row = engine_row_by_key.get(lookup)
        if engine_row is None:
            unmatched.append(f'{row["Player Name"]} {row["Season"]}')
            continue
        engine_row_by_site_row[int(site_row)] = int(engine_row)
        site_key_by_engine_row[int(engine_row)] = str(row["player_key"])

    print(f"  matched {len(engine_row_by_site_row)}/{len(guards)} to the BBall Index table")
    if unmatched:
        print(f"  ! no BBall Index features, so no comps for {len(unmatched)}: {', '.join(unmatched)}")

    # Candidates are exactly the galaxy rows we could match.
    candidate_mask = np.zeros(engine.N, dtype=bool)
    candidate_mask[list(site_key_by_engine_row.keys())] = True
    print(f"  candidate pool: {int(candidate_mask.sum())} player-seasons")

    cluster_by_key = {str(row["player_key"]): int(labels[index]) for index, row in guards.iterrows()}
    meta_by_key = {
        str(row["player_key"]): {
            "player_name": str(row["Player Name"]),
            "season": str(row["Season"]),
            "team": str(row["teams_played"]),
            "position": str(row["position"]),
        }
        for _, row in guards.iterrows()
    }

    # --- canonical attention label order (shared by every player) -----------
    block_counts = np.bincount(engine.SUB_BLOCK, minlength=engine.G)
    skillset_labels = []
    for index, (block, subgroup, _features) in enumerate(engine.SUBS):
        is_subgroup = block_counts[engine.SUB_BLOCK[index]] > 1
        skillset_labels.append({
            "skillset": subgroup if is_subgroup else block,
            "type": "subgroup" if is_subgroup else "block",
            "family": engine._report_family(block),
            "parent_block": block if is_subgroup else "",
            "domain": engine.BLOCK_DOMAIN[block],
        })
    family_order: list[tuple[str, str]] = []
    for label in skillset_labels:
        key = (label["family"], label["domain"])
        if key not in family_order:
            family_order.append(key)
    family_index = {key: position for position, key in enumerate(family_order)}
    skillset_family_index = np.array(
        [family_index[(label["family"], label["domain"])] for label in skillset_labels]
    )

    block_labels = list(engine.BLOCK_NAMES)
    block_index = {name: position for position, name in enumerate(block_labels)}
    player_key_order = [str(guards.at[row, "player_key"]) for row in sorted(engine_row_by_site_row)]
    player_key_index = {key: position for position, key in enumerate(player_key_order)}

    # Query-independent pieces, hoisted out of the per-player loop.
    final_sub_weights = engine._final_sub_weights()
    learned_sub = engine.BASE_W[:, engine.SUB_BLOCK] * engine.SUB_MULT

    players_payload: dict[str, dict] = {}
    flat_rows: list[dict] = []
    started = time.time()
    total = len(engine_row_by_site_row)

    for counter, (site_row, engine_row) in enumerate(sorted(engine_row_by_site_row.items()), start=1):
        source_key = str(guards.at[site_row, "player_key"])
        off_weight = engine.auto_off_weight(engine_row)

        views = engine.comps_for(
            engine_row,
            top_n=args.top_n,
            candidate_mask=candidate_mask,
            exclude_same_player=True,
        )

        # Paper's why_similar decomposition for every comp we are about to emit.
        decomposition = engine.block_decomposition(
            engine_row,
            {row["row"] for rows in views.values() for row in rows},
        )

        comps_payload: dict[str, list[list]] = {}
        for domain, rows in views.items():
            encoded = []
            for rank, row in enumerate(rows, start=1):
                target_key = site_key_by_engine_row.get(row["row"])
                if target_key is None:
                    continue
                blocks = decomposition.get(row["row"], {})
                ranked = sorted(blocks.items(), key=lambda item: item[1]["divergence"])
                alike = [block_index[name] for name, _ in ranked[:3]]
                different = [block_index[name] for name, _ in reversed(ranked[-3:])]
                # Positional record; the backend expands it to named fields.
                # [target, off_sim, def_sim, overall_sim, off_d, def_d, overall_d,
                #  [3 most-alike block ids], [3 most-different block ids]]
                encoded.append([
                    player_key_index[target_key],
                    round(row["off_similarity"], 1),
                    round(row["def_similarity"], 1),
                    round(row["overall_similarity"], 1),
                    round(row["off_distance"], 4),
                    round(row["def_distance"], 4),
                    round(row["overall_distance"], 4),
                    alike,
                    different,
                ])
                target_meta = meta_by_key[target_key]
                flat_rows.append({
                    "domain": domain,
                    "player_key": source_key,
                    "player_name": meta_by_key[source_key]["player_name"],
                    "season": meta_by_key[source_key]["season"],
                    "off_weight": round(off_weight, 4),
                    "rank": rank,
                    "related_player_key": target_key,
                    "related_player_name": target_meta["player_name"],
                    "related_season": target_meta["season"],
                    "off_similarity": round(row["off_similarity"], 1),
                    "def_similarity": round(row["def_similarity"], 1),
                    "overall_similarity": round(row["overall_similarity"], 1),
                    "off_distance": round(row["off_distance"], 4),
                    "def_distance": round(row["def_distance"], 4),
                    "overall_distance": round(row["overall_distance"], 4),
                    "most_alike_blocks": ", ".join(name for name, _ in ranked[:3]),
                    "most_different_blocks": ", ".join(name for name, _ in reversed(ranked[-3:])),
                })
            comps_payload[domain] = encoded

        # Attention, in canonical order, as the paper's two tables.
        balanced = engine._domain_normalized(final_sub_weights[engine_row], off_weight)
        skillset_attention = 100.0 * balanced
        skillset_learned = learned_sub[engine_row]
        family_attention = np.bincount(
            skillset_family_index, weights=skillset_attention, minlength=len(family_order)
        )
        family_learned = np.bincount(
            skillset_family_index, weights=skillset_learned, minlength=len(family_order)
        )

        players_payload[source_key] = {
            "off_weight": round(float(off_weight), 4),
            "def_weight": round(1.0 - float(off_weight), 4),
            "skillset_attention": [round(float(value), 3) for value in skillset_attention],
            "skillset_learned_w": [round(float(value), 4) for value in skillset_learned],
            "family_attention": [round(float(value), 3) for value in family_attention],
            "family_learned_w": [round(float(value), 4) for value in family_learned],
            "comps": comps_payload,
        }

        if counter % 400 == 0 or counter == total:
            elapsed = time.time() - started
            print(f"  {counter}/{total} players  ({elapsed:.0f}s elapsed, "
                  f"{elapsed / counter * (total - counter):.0f}s remaining)")

    payload = {
        "meta": {
            "engine": "similarity_engine.py (sim.ipynb v4 port)",
            "paper": "NBA Player-Season Similarity Algorithm",
            "fitting_population": int(engine.N),
            "candidate_pool": int(candidate_mask.sum()),
            "blocks": int(engine.G),
            "subgroups": int(engine.NSUB),
            "top_n": int(args.top_n),
            "similarity_transform": "100 / (1 + (d / median_positive_d) ** 2)",
            "comp_record_fields": [
                "target_index", "off_similarity", "def_similarity", "overall_similarity",
                "off_distance", "def_distance", "overall_distance",
                "most_alike_block_ids", "most_different_block_ids",
            ],
            "position_penalty_weight": float(engine.POSITION_PENALTY_WEIGHT),
            "profile_blend": float(engine.PROFILE_BLEND),
            "sharpening_strategy": str(engine.SHARPENING_STRATEGY),
            "offense_bias": float(engine.OFFENSE_BIAS),
            "min_domain_weight": float(engine.MIN_DOMAIN_WEIGHT),
            "players_without_features": unmatched,
        },
        "player_keys": player_key_order,
        "block_labels": block_labels,
        "family_labels": [{"family": family, "domain": domain} for family, domain in family_order],
        "skillset_labels": skillset_labels,
        "players": players_payload,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "similarity_v4.json"
    json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    csv_path = output_dir / "similarity_v4_comps.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print(f"\nWrote {json_path}  ({json_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {csv_path}  ({len(flat_rows)} rows)")


if __name__ == "__main__":
    main()
