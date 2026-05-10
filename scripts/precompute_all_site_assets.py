#!/usr/bin/env python3
"""Run all site precompute steps in the right order.

Run from the project root:
    python3 scripts/precompute_all_site_assets.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_DATASET = "/Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_16_17_25_26.csv"
DEFAULT_DLEBRON_DATASET = "/Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_player_comps_real.csv"


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute all cluster-site assets.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to fullseasonfeatures_16_17_25_26.csv")
    parser.add_argument("--dlebron-dataset", default=DEFAULT_DLEBRON_DATASET, help="Path to fullseasonfeatures_player_comps_real.csv; D-LEBRON is side-loaded for the defensive skill breakdown and Defensive Lock-Down badge")
    parser.add_argument("--skip-headshots", action="store_true", help="Skip player headshot map generation")
    parser.add_argument("--skip-badge-assets", action="store_true", help="Skip local badge image asset generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = str(Path(args.dataset).expanduser())
    dlebron_dataset = str(Path(args.dlebron_dataset).expanduser())

    run_step("Precompute galaxy assets", [sys.executable, str(SCRIPTS_DIR / "precompute_galaxy_assets.py"), "--dataset", dataset])
    run_step("Precompute player badges", [sys.executable, str(SCRIPTS_DIR / "precompute_player_badges.py"), "--dataset", dataset, "--dlebron-dataset", dlebron_dataset])
    run_step("Precompute skill and 3PT breakdowns", [sys.executable, str(SCRIPTS_DIR / "precompute_player_breakdowns.py"), "--dataset", dataset, "--dlebron-dataset", dlebron_dataset])

    if not args.skip_headshots:
        run_step("Build player headshots", [sys.executable, str(SCRIPTS_DIR / "build_player_headshots.py"), "--extra-csv", dataset])

    if not args.skip_badge_assets:
        run_step("Bootstrap badge source images", [sys.executable, str(SCRIPTS_DIR / "bootstrap_badge_sources.py")])
        run_step("Build badge image assets", [sys.executable, str(SCRIPTS_DIR / "build_badge_assets.py")])

    print("\nAll requested site assets are precomputed.")


if __name__ == "__main__":
    main()
