"""Re-solve BADGE_TIER_THRESHOLDS so every badge lands on the same share of the league.

Badge scores are same-season percentiles against every player-season in the
dataset, so a tier threshold is directly a rarity statement. Rather than
hand-picking 26 sets of numbers, this script measures the score distribution of
the players who clear each badge's opportunity gate and solves for the cut that
puts the intended share of the league in each tier.

Run after any dataset refresh, then paste the printed table into
backend/badge_engine.py.

    python3 scripts/calibrate_badge_thresholds.py
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import app as A  # noqa: E402
import badge_engine as BE  # noqa: E402

# Cumulative share of the whole league that should hold each tier or better.
TIER_TARGETS = {"diamond": 0.006, "gold": 0.024, "silver": 0.060, "bronze": 0.120}


def collect_gate_passing_scores(players: pd.DataFrame) -> dict:
    """Run the badge engine and record every score whose opportunity gate passed."""
    recorded = collections.defaultdict(list)
    original_build = BE.build_badge_payload

    def recording_build(badge_id, tier, score, components, demotion_reasons=None):
        recorded[badge_id].append(float(score))
        return original_build(badge_id, tier or "bronze", score, components, demotion_reasons)

    BE.build_badge_payload = recording_build
    try:
        BE.compute_badges_for_guards(players)
    finally:
        BE.build_badge_payload = original_build
    return recorded


def solve_thresholds(recorded: dict, league_size: int) -> dict:
    solved = {}
    for badge_id in BE.BADGE_DEFINITIONS:
        scores = np.sort(np.array(recorded.get(badge_id, []), dtype=float))[::-1]
        if scores.size == 0:
            continue
        values = []
        for tier in ("diamond", "gold", "silver", "bronze"):
            wanted = int(round(TIER_TARGETS[tier] * league_size))
            values.append(float(scores[-1] if wanted >= scores.size else scores[wanted]))
        for index in range(1, len(values)):
            if values[index] >= values[index - 1]:
                values[index] = values[index - 1] - 0.5
        solved[badge_id] = tuple(round(max(0.0, min(100.0, value)), 1) for value in values)
    return solved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=A.DEFAULT_DATASET_PATH)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    players = pd.read_csv(args.dataset, low_memory=False)
    players = A.add_locked_similarity_derived_features(players).reset_index(drop=True)
    for feature_name in A.ALLOWED_FEATURES:
        if feature_name not in players.columns:
            players[feature_name] = 0.0
    players["player_key"] = players.apply(A.stable_player_key, axis=1)

    recorded = collect_gate_passing_scores(players)
    solved = solve_thresholds(recorded, len(players))

    print(f"# league size = {len(players)} player-seasons")
    print("BADGE_TIER_THRESHOLDS = {")
    for badge_id, thresholds in solved.items():
        gate_share = 100.0 * len(recorded[badge_id]) / len(players)
        print(f'    "{badge_id}": {thresholds},  # gate passes {gate_share:.1f}% of the league')
    print("}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(solved, indent=1))


if __name__ == "__main__":
    main()
