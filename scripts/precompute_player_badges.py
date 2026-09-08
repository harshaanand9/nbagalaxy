#!/usr/bin/env python3
"""Precompute player-season badge assignments for the cluster site.

Run from the project root:
    python3 scripts/precompute_player_badges.py \
      --dataset /Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_16_17_25_26.csv \
      --output backend/data/player_badges.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import (  # noqa: E402
    DLEBRON_FEATURE,
    META_COLS,
    PLAYER_COMPS_DATASET_PATH,
    ROWS_TO_REMOVE,
    attach_badge_rarity_columns,
    build_assignment_key,
    stable_player_key,
)
from badge_engine import BADGE_REQUIRED_FEATURES, build_badge_rows, compute_badges_for_guards  # noqa: E402


def attach_dlebron_from_player_comps(dataframe: pd.DataFrame, dlebron_dataset_path: Path) -> pd.DataFrame:
    """Side-load D-LEBRON without changing the main badge feature dataset."""
    if not dlebron_dataset_path.exists():
        raise FileNotFoundError(f"D-LEBRON dataset file not found: {dlebron_dataset_path}")

    source = pd.read_csv(dlebron_dataset_path)
    required_columns = {"Player Name", "Season", DLEBRON_FEATURE}
    missing_columns = sorted(required_columns.difference(source.columns))
    if missing_columns:
        raise ValueError(f"D-LEBRON dataset is missing columns: {missing_columns}")

    compact = source[["Player Name", "Season", DLEBRON_FEATURE]].copy()
    compact[DLEBRON_FEATURE] = pd.to_numeric(compact[DLEBRON_FEATURE], errors="coerce")
    compact = compact.dropna(subset=[DLEBRON_FEATURE])
    compact["_assignment_key"] = compact.apply(
        lambda row: build_assignment_key(row["Player Name"], row["Season"]),
        axis=1,
    )
    compact = compact.drop_duplicates(subset=["_assignment_key"], keep="last")
    lookup = compact.set_index("_assignment_key")[DLEBRON_FEATURE]

    output = dataframe.copy()
    keys = output.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)
    side_loaded = keys.map(lookup)
    if DLEBRON_FEATURE in output.columns:
        output[DLEBRON_FEATURE] = pd.to_numeric(output[DLEBRON_FEATURE], errors="coerce").fillna(side_loaded)
    else:
        output[DLEBRON_FEATURE] = side_loaded
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute badge assignments for every player-season.")
    parser.add_argument(
        "--dataset",
        default="/Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_16_17_25_26.csv",
        help="Path to the fullseasonfeatures CSV.",
    )
    parser.add_argument(
        "--dlebron-dataset",
        default=PLAYER_COMPS_DATASET_PATH,
        help="Path to fullseasonfeatures_player_comps_real.csv; only D-LEBRON is side-loaded from here for Defensive Lock-Down badges.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "backend" / "data" / "player_badges.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser()
    dlebron_dataset_path = Path(args.dlebron_dataset).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)
    dataframe = attach_dlebron_from_player_comps(dataframe, dlebron_dataset_path)
    missing_meta = [column for column in META_COLS if column not in dataframe.columns]
    if missing_meta:
        raise ValueError(f"Missing required metadata columns: {missing_meta}")

    missing_badge_features = [feature for feature in BADGE_REQUIRED_FEATURES if feature not in dataframe.columns]
    if missing_badge_features:
        raise ValueError(f"Missing badge feature columns: {missing_badge_features}")

    remove_df = pd.DataFrame(ROWS_TO_REMOVE, columns=["Player Name", "Season", "teams_played", "position"])
    dataframe = dataframe.merge(
        remove_df.assign(_remove_flag=1),
        on=["Player Name", "Season", "teams_played", "position"],
        how="left",
    )
    dataframe = dataframe[dataframe["_remove_flag"] != 1].drop(columns=["_remove_flag"]).copy()

    guards = dataframe.copy()
    if guards.empty:
        raise ValueError("No player-seasons found in the dataset.")
    guards["player_key"] = guards.apply(stable_player_key, axis=1)

    badges_by_player_key = compute_badges_for_guards(guards)
    badge_rows = build_badge_rows(guards, badges_by_player_key)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame = pd.DataFrame(badge_rows)
    if not output_frame.empty:
        output_frame = attach_badge_rarity_columns(output_frame, guards)
    output_frame.to_csv(output_path, index=False)

    if output_frame.empty:
        print(f"Wrote empty badge file: {output_path}")
        return

    print(f"Wrote {len(output_frame):,} badge rows to {output_path}")
    print("Badge counts by tier:")
    print(output_frame["badge_tier"].value_counts().to_string())
    print("\nBadge counts by badge:")
    print(output_frame.groupby(["badge_name", "badge_tier"]).size().sort_values(ascending=False).to_string())
    print("\nJames Harden badge rows:")
    harden_rows = output_frame[output_frame["Player Name"].str.casefold().eq("james harden")]
    print(len(harden_rows))


if __name__ == "__main__":
    main()
