import hashlib
import json
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    from badge_engine import (
        BADGE_REQUIRED_FEATURES,
        build_badge_rows,
        compute_badges_for_guards,
    )
except ImportError:
    from backend.badge_engine import (
        BADGE_REQUIRED_FEATURES,
        build_badge_rows,
        compute_badges_for_guards,
    )


APP_VERSION = "0.22.0"
BACKEND_DIR = Path(__file__).resolve().parent
BACKEND_DATA_DIR = BACKEND_DIR / "data"
DEFAULT_DATASET_PATH = os.environ.get(
    "CLUSTER_DATASET_PATH",
    str(BACKEND_DATA_DIR / "fullseasonfeatures_16_17_25_26.csv"),
)
CACHE_DIR = Path(os.environ.get("CLUSTER_CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_PRODUCTION_OUTPUT_DIR = Path(
    "/Users/harsha/Desktop/PickPocketProjectOfficial/kmeans_blocked_pca_euclidean_outputs_13_14_25_26"
)


def first_existing_path(path_candidates: List[Path]) -> Path:
    for path_candidate in path_candidates:
        if path_candidate.exists():
            return path_candidate
    return path_candidates[0]

META_COLS = ["Player Name", "Season", "teams_played", "position"]
CONTEXT_COLS = ["GP", "MP", "OffPoss", "DefPoss"]

CURRENT_CSV_FEATURES = [
    "Avg2ptShotDistance",
    "Avg3ptShotDistance",
    "OffPoss",
    "DefPoss",
    "off_fouls_drawn_frequency",
    "zero_to_two_drib_3PA_Frequency",
    "zero_to_two_drib_3FGA_Accuracy",
    "zero_to_two_drib_3FGA_avg_drib",
    "three_to_seven_plus_drib_3FGA_Frequency",
    "three_to_seven_plus_drib_3FGA_Accuracy",
    "pct_three_to_seven_plus_drib_3FGA_seven_plus",
    "zero_to_two_drib_2FGA_Frequency",
    "zero_to_two_drib_2FGA_Accuracy",
    "zero_to_two_drib_2FGA_avg_drib",
    "three_to_seven_plus_drib_2FGA_Frequency",
    "three_to_seven_plus_drib_2FGA_Accuracy",
    "pct_three_to_seven_plus_drib_2FGA_seven_plus",
    "tight_very_tight_3fga_frequency",
    "tight_very_tight_3fga_accuracy",
    "open_3fga_frequency",
    "open_3fga_accuracy",
    "tight_very_tight_2fga_frequency",
    "tight_very_tight_2fga_accuracy",
    "open_2fga_frequency",
    "open_2fga_accuracy",
    "pull_up_3P_frequency",
    "pull_up_3P_accuracy",
    "pull_up_2P_frequency",
    "pull_up_2P_accuracy",
    "catch_shoot_3P_frequency",
    "catch_shoot_3P_accuracy",
    "MidRangeFrequency",
    "MidRangeAccuracy",
    "RestrictedArea_Frequency",
    "RestrictedArea_Accuracy",
    "Paint_Non_RA_Frequency",
    "Paint_Non_RA_Accuracy",
    "drive_fga_frequency",
    "drive_fg_pct",
    "drive_fta_frequency",
    "Pct_Paint_FGA_from_drives",
    "drib_tov_ratio",
    "touch_frequency",
    "potential_ast_tov_ratio",
    "assist_frequency",
    "pass_frequency",
    "Pass_shot_ratio",
    "potential_assist_frequency",
    "opp_players_fg_pct_difference",
    "contested_shot_frequency",
    "Opp_players_fga_per_75_poss",
    "dunks_per_75_poss",
    "Blocks_per_75",
    "Steals_per_75",
    "Deflections_per_75",
    "avg_drib_fg3a",
    "avg_drib_fg2a",
    "3P_Accuracy",
    "Wide_Open_3FGA_Frequency",
    "Wide_Open_3FG_PCT",
    "pct_3fga_wide_open",
    "3fga_frequency",
    "pts_from_3s_per_75",
    "pts_from_midrange_per_75",
    "pts_from_drives_per_75",
    "avg_drib_per_touch",
    "fta_per_75",
    "drive_fta_per_75",
    "pass_tov_ratio",
    "pct_fga_3FGA",
    "pct_fga_MR",
    "pct_fga_drive_fga",
    "potential_assist_FGA_ratio",
    "avg_closest_defender_3FGA",
    "avg_sec_per_touch",
    "ASSISTS_ON_OFF",
    "EFG_PCT_ON_OFF",
    "PACE_ON_OFF",
    "OPP_SHOT_QUALITY_ON_OFF",
    "OPP_EFG_PCT_ON_OFF",
    "OPP_SHOT_QUALITY_TEAM",
    "OPP_EFG_PCT_TEAM",
    "pct_2p_fg_assisted",
    "pct_3p_fg_assisted",
    "crafted_cdpm",
    "crafted_box_creation",
    "crafted_passer_rating",
    "D-LEBRON",
    "tight_very_tight_3fga_per_game",
    "open_3fga_per_game",
    "Wide_Open_3FGA_per_game",
    "pull_up_3PA_per_game",
    "catch_shoot_3PA_per_game",
    "tight_very_tight_2fga_per_game",
    "open_2fga_per_game",
    "pull_up_2PA_per_game",
    "restricted_area_fga_per_game",
    "paint_non_ra_fga_per_game",
    "drive_fga_per_game",
    "drives_per_game",
    "contested_shots_per_game",
    "off_fouls_drawn_per_game",
    "potential_assists_and_ft_assists",
]

EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES = {
    "ThreePT": [
        "tight_very_tight_3fga_per_game",
        "tight_very_tight_3fga_accuracy",
        "open_3fga_per_game",
        "open_3fga_accuracy",
        "Wide_Open_3FGA_per_game",
        "Wide_Open_3FG_PCT",
        "pull_up_3PA_per_game",
        "pull_up_3P_accuracy",
        "catch_shoot_3PA_per_game",
        "catch_shoot_3P_accuracy",
        "avg_drib_fg3a",
        "Avg3ptShotDistance",
        "pts_from_3s_per_75",
        "pct_fga_3FGA",
        "pct_3p_fg_assisted",
        "avg_closest_defender_3FGA",
    ],
    "MidRange": [
        "tight_very_tight_2fga_per_game",
        "tight_very_tight_2fga_accuracy",
        "open_2fga_per_game",
        "open_2fga_accuracy",
        "pull_up_2PA_per_game",
        "pull_up_2P_accuracy",
        "Avg2ptShotDistance",
        "pct_fga_MR",
        "pts_from_midrange_per_75",
        "pct_2p_fg_assisted",
    ],
    "RimPressure": [
        "restricted_area_fga_per_game",
        "RestrictedArea_Accuracy",
        "paint_non_ra_fga_per_game",
        "Paint_Non_RA_Accuracy",
        "drive_fga_per_game",
        "drive_fg_pct",
        "dunks_per_75_poss",
        "pts_from_drives_per_75",
        "pct_fga_drive_fga",
        "drive_fta_rate",
    ],
    "Playmaking": [
        "avg_drib_per_touch",
        "avg_sec_per_touch",
        "crafted_box_creation",
        "crafted_passer_rating",
        "pts_created_from_assists",
        "potential_assists_and_ft_assists",
        "ASSISTS_ON_OFF",
        "EFG_PCT_ON_OFF",
        "assists_tov_ratio",
        "potential_assist_tov_ratio",
        "THREE_PT_FG_PCT_ON_OFF",
        "PTS_PER_100_ON_OFF",
    ],
    "Defense": [
        "D-LEBRON",
        "Opp_players_fga_per_75_poss",
        "contested_shots_per_game",
        "off_fouls_drawn_per_game",
    ],
}
EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS = {
    "ThreePT": 0.200,
    "MidRange": 0.200,
    "RimPressure": 0.200,
    "Playmaking": 0.200,
    "Defense": 0.200,
}
EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER = ["ThreePT", "MidRange", "RimPressure", "Playmaking", "Defense"]


def build_locked_euclidean_feature_signature() -> str:
    signature_payload = {
        "euclidean_kmeans_locked_group_features": EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES,
        "euclidean_kmeans_locked_group_weights": EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS,
        "euclidean_kmeans_locked_group_order": EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER,
        "euclidean_kmeans_locked_k": EUCLIDEAN_KMEANS_LOCKED_K,
        "euclidean_kmeans_locked_clip_zscore": EUCLIDEAN_KMEANS_LOCKED_CLIP_ZSCORE,
        "euclidean_kmeans_locked_pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
        "euclidean_kmeans_locked_similarity_distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
        "euclidean_kmeans_locked_luka_override_metric": LUKA_DONCIC_SIMILARITY_DISTANCE_METRIC,
    }
    raw_signature = json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw_signature).hexdigest()
EUCLIDEAN_KMEANS_LOCKED_K = 12
EUCLIDEAN_KMEANS_LOCKED_PIPELINE = "per_game_raw52_equal_blocks_cosine_luka_euclidean"
EUCLIDEAN_KMEANS_LOCKED_SPACE_TRANSFORM = "season_median_imputed_guard_standardized_clipped_raw52_equal_blocks"
EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC = "cosine"
LUKA_DONCIC_SIMILARITY_DISTANCE_METRIC = "euclidean"
LUKA_DONCIC_SIMILARITY_OVERRIDE_KEY = "lukadoncic"
GALAXY_SIMILAR_PLAYER_COUNT = 5
GALAXY_CLUSTER_KNN_COUNT = 2
GALAXY_UMAP_N_NEIGHBORS = 35
GALAXY_UMAP_MIN_DIST = 0.15
GALAXY_RANDOM_STATE = 42

SKILL_BREAKDOWN_EXCLUDED_FEATURES = {
    "avg_drib_fg3a",
    "pct_fga_3FGA",
    "pct_3p_fg_assisted",
    "pct_3fga_wide_open",
    "pct_fga_MR",
    "Avg2ptShotDistance",
    "pct_2p_fg_assisted",
    "Pct_Paint_FGA_from_drives",
    "pct_fga_drive_fga",
    "avg_sec_per_touch",
}
SKILL_BREAKDOWN_GROUP_ORDER = ["ThreePT", "MidRange", "RimPressure", "Playmaking", "Defense"]
SKILL_BREAKDOWN_PLAYMAKING_FEATURES = [
    "assist_frequency",
    "drib_tov_ratio",
    "potential_assist_frequency",
    "ASSISTS_ON_OFF",
    "assists_tov_ratio",
    "EFG_PCT_ON_OFF",
    "pts_created_from_assists",
    "PTS_PER_100_ON_OFF",
    "pts_created_to_tov_ratio",
    "THREE_PT_FG_PCT_ON_OFF",
    "crafted_box_creation",
    "crafted_passer_rating",
]
SKILL_BREAKDOWN_DEFENSE_FEATURES = [
    "Blocks_per_75",
    "Steals_per_75",
    "Deflections_per_75",
    "off_fouls_drawn_frequency",
    "opp_players_fg_pct_difference",
    "contested_shot_frequency",
    "crafted_cdpm",
]
SKILL_BREAKDOWN_GROUP_FEATURES = {
    group_name: [
        feature_name
        for feature_name in EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES[group_name]
        if feature_name not in SKILL_BREAKDOWN_EXCLUDED_FEATURES
    ]
    for group_name in SKILL_BREAKDOWN_GROUP_ORDER
}
SKILL_BREAKDOWN_GROUP_FEATURES["Playmaking"] = SKILL_BREAKDOWN_PLAYMAKING_FEATURES
SKILL_BREAKDOWN_GROUP_FEATURES["Defense"] = SKILL_BREAKDOWN_DEFENSE_FEATURES

THREE_PT_BREAKDOWN_GROUP_ORDER = [
    "Deep Range Shooting",
    "Catch and Shooting",
    "Contested 3PT Shot Making",
    "Pull Up 3PT Shooting",
    "3PT Volume",
    "3PT Accuracy",
]
THREE_PT_BREAKDOWN_GROUP_FEATURES = {
    "Deep Range Shooting": ["Avg3ptShotDistance", "3fga_frequency", "3P_Accuracy"],
    "Catch and Shooting": ["catch_shoot_3P_frequency", "catch_shoot_3P_accuracy"],
    "Contested 3PT Shot Making": [
        "pct_3fga_wide_open",
        "avg_closest_defender_3FGA",
        "3P_Accuracy",
        "tight_very_tight_3fga_frequency",
        "3fga_frequency",
    ],
    "Pull Up 3PT Shooting": ["pull_up_3P_frequency", "pull_up_3P_accuracy"],
    "3PT Volume": ["3fga_frequency", "traditional_fg3a"],
    "3PT Accuracy": ["3P_Accuracy", "3fga_frequency"],
}
THREE_PT_BREAKDOWN_LOWER_IS_BETTER_BY_GROUP = {
    "Contested 3PT Shot Making": {"pct_3fga_wide_open", "avg_closest_defender_3FGA"},
}
SKILL_BREAKDOWN_LOCAL_PERCENTILE_RULES_BY_GROUP = {}
THREE_PT_BREAKDOWN_LOCAL_PERCENTILE_RULES_BY_GROUP = {}
EUCLIDEAN_KMEANS_LOCKED_CLIP_ZSCORE = 3.50
EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH = Path(
    os.environ.get(
        "EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH",
        str(
            first_existing_path(
                [
                    LOCAL_PRODUCTION_OUTPUT_DIR / "euclidean_kmeans_locked_assignments.csv",
                    BACKEND_DATA_DIR / "euclidean_kmeans_locked_assignments.csv",
                ]
            )
        ),
    )
)
GALAXY_PRECOMPUTED_PATHS = [
    LOCAL_PRODUCTION_OUTPUT_DIR / "galaxy_precomputed.json",
    BACKEND_DATA_DIR / "galaxy_precomputed.json",
    BACKEND_DIR.parent / "data" / "galaxy_precomputed.json",
    BACKEND_DIR.parent / "galaxy_precomputed.json",
]
_GALAXY_PRECOMPUTED_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "payload": None}

BREAKDOWN_PRECOMPUTED_PATHS = {
    "skill_breakdown": BACKEND_DATA_DIR / "player_skill_breakdowns.json",
    "three_pt_breakdown": BACKEND_DATA_DIR / "player_three_pt_breakdowns.json",
}
_BREAKDOWN_PRECOMPUTED_CACHE: Dict[str, Dict[str, object]] = {}

SIMILAR_PLAYERS_PATHS = [
    LOCAL_PRODUCTION_OUTPUT_DIR / "similar_players_precomputed_production.csv",
    BACKEND_DATA_DIR / "similar_players_precomputed_production.csv",
    BACKEND_DIR.parent / "data" / "similar_players_precomputed_production.csv",
    BACKEND_DIR.parent / "similar_players_precomputed_production.csv",
    LOCAL_PRODUCTION_OUTPUT_DIR / "similar_players.csv",
    BACKEND_DATA_DIR / "similar_players.csv",
    BACKEND_DIR.parent / "data" / "similar_players.csv",
    BACKEND_DIR.parent / "similar_players.csv",
]
_SIMILAR_PLAYERS_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "dataframe": None}
SIMILAR_PLAYERS_BLOCK_SCORE_COLUMNS = [
    "threept_similarity_score",
    "midrange_similarity_score",
    "rimpressure_similarity_score",
    "playmaking_similarity_score",
    "defense_similarity_score",
]
SIMILAR_PLAYERS_REQUIRED_DETAIL_COLUMNS = [
    "strongest_similarity_blocks",
    "biggest_difference_blocks",
    "threept_distance",
    "midrange_distance",
    "rimpressure_distance",
    "playmaking_distance",
    "defense_distance",
    *SIMILAR_PLAYERS_BLOCK_SCORE_COLUMNS,
]

HEADSHOT_MAP_PATH = Path(
    os.environ.get(
        "PLAYER_HEADSHOT_MAP_PATH",
        str(BACKEND_DATA_DIR / "player_headshots.csv"),
    )
)
_HEADSHOT_MAP_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "rows_by_key": None}
HEADSHOT_FALLBACK_URL = "/headshots/fallback.svg"


PLAYER_COMPS_DATASET_PATH = os.environ.get(
    "PLAYER_COMPS_DATASET_PATH",
    str(BACKEND_DATA_DIR / "fullseasonfeatures_player_comps_real.csv"),
)
PLAYER_COMPS_PACE_PATH = os.environ.get(
    "PLAYER_COMPS_PACE_PATH",
    str(BACKEND_DATA_DIR / "league_average_pace_2016_17_to_2025_26.csv"),
)
PULLUP_DATASET_PATH = os.environ.get(
    "PULLUP_DATASET_PATH",
    "/Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_13_14_25_26_pullup.csv",
)
PLAYER_COMPS_TARGET_PACE_MODE = os.environ.get("PLAYER_COMPS_TARGET_PACE_MODE", "latest")
DLEBRON_FEATURE = "D-LEBRON"
_DLEBRON_SOURCE_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "lookup": None}
_PULLUP_SOURCE_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "lookup": None}
_PLAYER_COMPS_PERCENTILE_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "payload": None}

PLAYER_COMPS_LOWER_IS_BETTER_FEATURES = {
    "drive_tov_rate",
    "traditional_tov",
    "opp_players_fg_pct_difference",
    "defense_dash_overall_d_fg_pct",
    "DEF_PTS_PER_100_ON_OFF",
    "DEF_THREE_PT_FG_PCT_ON_OFF",
}

PLAYER_COMPARISON_MODES = [
    {"value": "raw_stats", "label": "Raw Stats"},
    {"value": "pace_adjusted_raw_stats", "label": "Raw Stats — Pace Adjusted"},
    {"value": "raw_frequencies", "label": "Raw Frequencies"},
    {"value": "raw_per_75", "label": "Raw Per 75"},
    {"value": "same_season_percentile", "label": "Same-Season Percentile"},
    {"value": "all_seasons_percentile", "label": "All-Seasons Percentile"},
]

PLAYER_COMPARISON_CATEGORIES = [
    {"value": "traditional", "label": "Traditional Stats"},
    {"value": "three_pt", "label": "3PT Stats"},
    {"value": "midrange", "label": "MidRange Stats"},
    {"value": "rim_pressure", "label": "Rim Pressure Stats"},
    {"value": "playmaking", "label": "Playmaking Stats"},
    {"value": "defense", "label": "Defensive Stats"},
]

PLAYER_COMPARISON_FEATURES = {
    "traditional": [
        {"feature": "traditional_pts", "label": "PTS/G", "kind": "volume"},
        {"feature": "traditional_reb", "label": "TRB/G", "kind": "volume"},
        {"feature": "traditional_ast", "label": "AST/G", "kind": "volume"},
        {"feature": "traditional_tov", "label": "TOV/G", "kind": "volume"},
        {"feature": "traditional_stl", "label": "STL/G", "kind": "volume"},
        {"feature": "traditional_blk", "label": "BLK/G", "kind": "volume"},
        {"feature": "traditional_fg_pct", "label": "FG%", "kind": "percentage"},
        {"feature": "3P_Accuracy", "label": "3P%", "kind": "percentage", "aliases": ["traditional_fg3_pct", "traditional_fg3_pct", "fg3_pct", "FG3_PCT"]},
        {"feature": "traditional_ft_pct", "label": "FT%", "kind": "percentage"},
    ],
    "three_pt": [
        {"feature": "3P_Accuracy", "label": "3P%", "kind": "percentage", "aliases": ["traditional_fg3_pct", "fg3_pct", "FG3_PCT"]},
        {"feature": "traditional_fg3a", "label": "3PA/Game", "kind": "volume"},
        {"feature": "3fga_frequency", "label": "3PA Frequency", "kind": "frequency"},
        {"feature": "Avg3ptShotDistance", "label": "Avg 3PT Shot Distance", "kind": "distance"},
        {"feature": "pull_up_3P_frequency", "label": "Pull-Up 3PA Frequency", "kind": "frequency"},
        {"feature": "pull_up_3PA", "label": "Pull-Up 3PA/Game", "kind": "volume"},
        {"feature": "pull_up_3P_accuracy", "label": "Pull-Up 3P%", "kind": "percentage"},
        {"feature": "catch_shoot_3P_frequency", "label": "Catch-Shoot 3PA Frequency", "kind": "frequency"},
        {"feature": "catch_shoot_3PA", "label": "Catch-Shoot 3PA/Game", "kind": "volume"},
        {"feature": "catch_shoot_3P_accuracy", "label": "Catch-Shoot 3P%", "kind": "percentage"},
        {"feature": "zero_to_one_drib_3PA_frequency", "label": "0-1 Dribble 3PA Frequency", "kind": "frequency"},
        {"feature": "zero_to_one_drib_3PA_accuracy", "label": "0-1 Dribble 3P%", "kind": "percentage"},
        {"feature": "two_drib_3PA_frequency", "label": "2 Dribble 3PA Frequency", "kind": "frequency"},
        {"feature": "two_drib_3P_Accuracy", "label": "2 Dribble 3P%", "kind": "percentage", "aliases": ["two_drib_3PA_accuracy", "two_drib_3P_accuracy"]},
        {"feature": "three_to_six_drib_3FGA_frequency", "label": "3-6 Dribble 3PA Frequency", "kind": "frequency"},
        {"feature": "three_to_six_drib_3FGA_accuracy", "label": "3-6 Dribble 3P%", "kind": "percentage"},
        {"feature": "seven_plus_drib_3FGA_frequency", "label": "7+ Dribble 3PA Frequency", "kind": "frequency"},
        {"feature": "seven_plus_drib_3FGA_accuracy", "label": "7+ Dribble 3P%", "kind": "percentage"},
        {"feature": "pct_3p_fg_assisted", "label": "Assisted 3PM%", "kind": "percentage"},
        {"feature": "pct_3fga_wide_open", "label": "Wide-Open 3PA Share", "kind": "percentage"},
        {"feature": "shot_contest_0_2_fg3a", "label": "Very Tight 3PA/Game", "kind": "volume"},
        {"feature": "shot_contest_0_2_fg3a_frequency", "label": "Very Tight 3PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_0_2_fg3_pct", "label": "Very Tight 3P%", "kind": "percentage"},
        {"feature": "shot_contest_2_4_fg3a", "label": "Tight 3PA/Game", "kind": "volume"},
        {"feature": "shot_contest_2_4_fg3a_frequency", "label": "Tight 3PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_2_4_fg3_pct", "label": "Tight 3P%", "kind": "percentage"},
        {"feature": "shot_contest_4_6_fg3a", "label": "Semi-Contested 3PA/Game", "kind": "volume"},
        {"feature": "shot_contest_4_6_fg3a_frequency", "label": "Semi-Contested 3PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_4_6_fg3_pct", "label": "Semi-Contested 3P%", "kind": "percentage"},
        {"feature": "shot_contest_6_plus_fg3a", "label": "Wide-Open 3PA/Game", "kind": "volume"},
        {"feature": "shot_contest_6_plus_fg3a_frequency", "label": "Wide-Open 3PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_6_plus_fg3_pct", "label": "Wide-Open 3P%", "kind": "percentage"},
        {"feature": "pct_fga_3FGA", "label": "3PA Shot Diet Share", "kind": "percentage"},
    ],
    "midrange": [
        {"feature": "Avg2ptShotDistance", "label": "Avg 2PT Shot Distance", "kind": "distance"},
        {"feature": "pull_up_2PA", "label": "Pull-Up 2PA/Game", "kind": "volume"},
        {"feature": "shot_contest_0_2_fg2a", "label": "Very Tight 2PA/Game", "kind": "volume"},
        {"feature": "shot_contest_0_2_fg2a_frequency", "label": "Very Tight 2PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_0_2_fg2_pct", "label": "Very Tight 2P%", "kind": "percentage"},
        {"feature": "shot_contest_2_4_fg2a", "label": "Tight 2PA/Game", "kind": "volume"},
        {"feature": "shot_contest_2_4_fg2a_frequency", "label": "Tight 2PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_2_4_fg2_pct", "label": "Tight 2P%", "kind": "percentage"},
        {"feature": "shot_contest_4_6_fg2a", "label": "Semi-Contested 2PA/Game", "kind": "volume"},
        {"feature": "shot_contest_4_6_fg2a_frequency", "label": "Semi-Contested 2PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_4_6_fg2_pct", "label": "Semi-Contested 2P%", "kind": "percentage"},
        {"feature": "shot_contest_6_plus_fg2a", "label": "Wide-Open 2PA/Game", "kind": "volume"},
        {"feature": "shot_contest_6_plus_fg2a_frequency", "label": "Wide-Open 2PA Frequency", "kind": "frequency"},
        {"feature": "shot_contest_6_plus_fg2_pct", "label": "Wide-Open 2P%", "kind": "percentage"},
        {"feature": "pct_fga_MR", "label": "Midrange Shot Diet Share", "kind": "percentage"},
        {"feature": "MidRangeFrequency", "label": "Midrange Frequency", "kind": "frequency"},
        {"feature": "MidRangeAccuracy", "label": "Midrange FG%", "kind": "percentage"},
        {"feature": "by_zone_statistics_mid_range_fga", "label": "Midrange FGA/Game", "kind": "volume"},
        {"feature": "pts_from_midrange_per_75", "label": "Midrange Points/75", "kind": "per75"},
    ],
    "rim_pressure": [
        {"feature": "RestrictedArea_Frequency", "label": "Restricted Area Frequency", "kind": "frequency"},
        {"feature": "RestrictedArea_Accuracy", "label": "Restricted Area FG%", "kind": "percentage"},
        {"feature": "by_zone_statistics_restricted_area_fga", "label": "Restricted Area FGA/Game", "kind": "volume"},
        {"feature": "by_zone_statistics_in_the_paint_non_ra_fga", "label": "Paint Non-RA FGA/Game", "kind": "volume"},
        {"feature": "Paint_Non_RA_Frequency", "label": "Paint Non-RA Frequency", "kind": "frequency"},
        {"feature": "Paint_Non_RA_Accuracy", "label": "Paint Non-RA FG%", "kind": "percentage"},
        {"feature": "drives_drive_fga", "label": "Drive FGA/Game", "kind": "volume"},
        {"feature": "drive_fga_frequency", "label": "Drive FGA Frequency", "kind": "frequency"},
        {"feature": "drive_fg_pct", "label": "Drive FG%", "kind": "percentage"},
        {"feature": "drive_FTAr", "label": "Drive Free Throw Rate", "kind": "ratio"},
        {"feature": "drives_drives", "label": "Drives/Game", "kind": "volume"},
        {"feature": "drive_frequency", "label": "Drive Frequency", "kind": "frequency"},
        {"feature": "pct_drives_results_in_FGA", "label": "Drive FGA Result%", "kind": "percentage"},
        {"feature": "drive_tov_rate", "label": "Drive TOV Rate", "kind": "percentage"},
    ],
    "playmaking": [
        {"feature": "traditional_ast", "label": "AST/G", "kind": "volume"},
        {"feature": "assist_frequency", "label": "Assist Frequency", "kind": "frequency"},
        {"feature": "potential_assist_frequency", "label": "Potential Assist Frequency", "kind": "frequency"},
        {"feature": "passing_potential_ast", "label": "Potential AST/G", "kind": "volume"},
        {"feature": "traditional_tov", "label": "TOV/G", "kind": "volume"},
        {"feature": "assists_tov_ratio", "label": "AST/TOV", "kind": "ratio"},
        {"feature": "potential_assist_tov_ratio", "label": "Potential AST/TOV", "kind": "ratio"},
        {"feature": "passing_ast_points_created", "label": "AST Points Created/G", "kind": "volume"},
        {"feature": "passing_passes_made", "label": "Passes Made/G", "kind": "volume"},
        {"feature": "pass_tov_ratio", "label": "Pass/TOV", "kind": "ratio"},
        {"feature": "crafted_passer_rating", "label": "Crafted Passer Rating", "kind": "rating"},
        {"feature": "crafted_box_creation", "label": "Crafted Box Creation", "kind": "rating"},
        {"feature": "drib_tov_ratio", "label": "Dribble/TOV", "kind": "ratio"},
        {"feature": "passing_secondary_ast", "label": "Secondary AST/G", "kind": "volume"},
        {"feature": "secondary_ast_frequency", "label": "Secondary AST Frequency", "kind": "frequency"},
        {"feature": "pts_created_from_assists", "label": "Points Created From Assists", "kind": "volume"},
        {"feature": "THREE_PT_FG_PCT_ON_OFF", "label": "Team 3P% On/Off", "kind": "percentage"},
        {"feature": "PTS_PER_100_ON_OFF", "label": "Team PTS/100 On/Off", "kind": "rating"},
        {"feature": "drives_drive_ast", "label": "Drive AST/G", "kind": "volume"},
        {"feature": "drive_ast_frequency", "label": "Drive AST Frequency", "kind": "frequency"},
        {"feature": "drive_ast_tov_ratio", "label": "Drive AST/TOV", "kind": "ratio"},
        {"feature": "passing_ft_ast", "label": "FT AST/G", "kind": "volume"},
        {"feature": "passing_ft_ast_frequency", "label": "FT AST Frequency", "kind": "frequency"},
        {"feature": "pct_passes_assists", "label": "Passes Becoming Assists", "kind": "percentage"},
        {"feature": "hustle_screen_ast_pts", "label": "Screen AST Points/G", "kind": "volume"},
        {"feature": "hustle_screen_ast_pts_frequency", "label": "Screen AST Points Frequency", "kind": "frequency"},
    ],
    "defense": [
        {"feature": "opp_players_fg_pct_difference", "label": "Opponent FG% Difference", "kind": "percentage"},
        {"feature": "defense_dash_overall_d_fga", "label": "Defended FGA/G", "kind": "volume"},
        {"feature": "defense_dash_overall_d_fg_pct", "label": "Defended FG%", "kind": "percentage"},
        {"feature": "offensive_fouls_drawn", "label": "Offensive Fouls Drawn/G", "kind": "volume"},
        {"feature": "hustle_contested_shots", "label": "Contested Shots/G", "kind": "volume"},
        {"feature": "hustle_deflections", "label": "Deflections/G", "kind": "volume"},
        {"feature": "hustle_charges_drawn", "label": "Charges Drawn/G", "kind": "volume"},
        {"feature": "off_fouls_drawn_frequency", "label": "Offensive Fouls Drawn Frequency", "kind": "frequency"},
        {"feature": "contested_shot_frequency", "label": "Contested Shot Frequency", "kind": "frequency"},
        {"feature": "charge_drawn_frequency", "label": "Charge Drawn Frequency", "kind": "frequency"},
        {"feature": "deflection_frequency", "label": "Deflection Frequency", "kind": "frequency"},
        {"feature": "crafted_cdpm", "label": "Crafted CDPM", "kind": "rating"},
        {"feature": "D-LEBRON", "label": "D-LEBRON", "kind": "rating"},
        {"feature": "DEF_PTS_PER_100_ON_OFF", "label": "Opponent PTS/100 On/Off", "kind": "rating"},
        {"feature": "DEF_THREE_PT_FG_PCT_ON_OFF", "label": "Opponent 3P% On/Off", "kind": "percentage"},
        {"feature": "traditional_stl", "label": "STL/G", "kind": "volume"},
        {"feature": "traditional_blk", "label": "BLK/G", "kind": "volume"},
        {"feature": "steal_frequency", "label": "Steal Frequency", "kind": "frequency"},
        {"feature": "block_frequency", "label": "Block Frequency", "kind": "frequency"},
    ],
}

_PLAYER_COMPS_DATA_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "dataframe": None, "column_map": None}
_PLAYER_COMPS_PACE_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "pace_by_season": None, "target_pace": None}
_PLAYER_COMPS_ASSIGNMENT_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "lookup": None}
_PLAYER_COMPS_BADGE_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "by_player_key": None, "by_identity_key": None}
HEADSHOT_NAME_ALIASES = {
    "oguguaanunoby": "oganunoby",
    "ogananoby": "oganunoby",
    "pjwashingtonjr": "pjwashington",
    "tjmcconnell": "tjmconnell",
    "cjelleby": "cjelleby",
    "kjmartin": "kenyonmartinjr",
    "kenyonmartin": "kenyonmartinjr",
    "kenyonmartinjr": "kenyonmartinjr",
    "nicclaxton": "nicolasclaxton",
    "nicolasclaxton": "nicolasclaxton",
    "eneskanter": "enesfreedom",
    "enesfreedom": "enesfreedom",
    "jimmybutler": "jimmybutleriii",
    "jimmybutleriii": "jimmybutleriii",
    "jamaimashack": "jahmaimashack",
    "jahmaimashack": "jahmaimashack",
}

EUCLIDEAN_KMEANS_LOCKED_EXCLUDED_NAMES = {
    "ben simmons",
    "lebron james",
    "scottie barnes",
}
EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER = {
    1: "3PT-Reliant Sharpshooter",
    2: "Slashing Oriented, Scoring + Playmaking Threat",
    3: "3PT Oriented Lock-Down Secondary Handler",
    4: "Expanded Role Sharpshooter",
    5: "Playmaking and 3PT Spark Plug",
    6: "3-Level Scoring and Playmaking Machine",
    7: "Talented Pass-First Point",
    8: "Limited Playmaking Spark Plug",
    9: "Inside-the-Arc Scoring and Playmaking Machine",
    10: "Streaky Shooting Defensive Savant",
    11: "Slashing Oriented Lock-Down Point",
    12: "3-Level Scoring Machine",
}

EUCLIDEAN_KMEANS_CLUSTER_DESCRIPTION_BY_NUMBER = {
    1: 'These players are defined by an extremely three-point-heavy reliant profile, almost entirely built around off-ball shooting tendencies. The heat chart makes this evident: catch-and-shoot 3PA frequency, 3PA frequency, 3P accuracy, points from 3s per 75, percent of FGA from 3PA, and assisted 3PM rates all sit well above the mean. This archetype’s offensive value comes from spacing the floor, relocating off the ball, and punishing defenses through catch and shoot three point looks.\n\nThe key difference between this group and the Expanded Role Sharpshooter archetype is offensive versatility. The Expanded Role Sharpshooter also shares a top tier off-ball three-point skillset, but that group adds a meaningful midrange layer to its scoring profile as well. The 3PT-Reliant Sharpshooter, on the other hand, scores almost solely through off-ball three-point scenarios. The low average-dribbles-before-3PA value reinforces this idea even further: these are not players that regularly create shots off the dribble.\n\nOutside of three-point shooting, the offensive profile is very limited. The median midrange, rim-pressure, drive, and playmaking features are all well below the average guard value. This makes this archetype highly concentrated: these players can be extremely valuable when their shooting is actively bending the defense, but their impact is completely tied to how their shot is feeling that day.',
    2: 'These players are defined by their ability to consistently bend defenses through downhill pressure. The heat chart makes this evident: restricted area frequency, paint non-RA frequency, drive FGA frequency, points from drives per 75, and percent of FGA coming from drives are all well above the mean. This is not really a pull-up shot-making archetype or a high-volume perimeter creation group, the offensive value of this group instead mainly comes from getting to the rim, forcing rotations, and generating drive and kick opportunities.\n\nMost of the playmaking features also sit above average, especially the potential assist frequency and potential assist-to-FGA ratio, showing that these players are not purely slashers, but they also show a willingness to create for others. But these playmaking features are simply not strong enough to push this group into true primary offensive engine territory.\n\nThis playmaking gap shows up especially when comparing them to their more offensively complete counterpart in Cluster #10, the Midrange + Slashing Oriented, Scoring + Playmaking Threat archetype. That group has a noticeably stronger passing profile and usually holds the ball for longer periods of time. The largest separation between these two clusters though is the wide gap in mid-range self creation and production. Cluster #2 gets most of its scoring pressure from drives and paint touches, while the Midrange + Slashing Oriented, Scoring + Playmaking Threats add a much more nuanced and efficient mid-range bag to their skillset.',
    3: 'These players are defined by their exceptional defensive metrics and their offensive reliance on shooting threes. The heat chart captures this profile clearly: most defensive indicators, including blocks per 75, steals per 75, deflections per 75, and other defensive metrics, sit well above the mean. This creates an archetype built around defensive value first.\n\nScoring is a different story. Most scoring features fall well below the mean, with three-point shooting standing out as the one area closer to average. The combination of above-average percent of FGA from three, assisted three-point rate, and catch-and-shoot three-point frequency points to a player who relies heavily on off-ball perimeter opportunities. These players are not typically bending defenses with their shooting abilities though. Their elevated wide-open three-point frequency suggests that their perimeter value comes more from defenses daring them to shoot.',
    4: 'These players are defined by an off-ball-centered play style, built around catch-and-shoot opportunities and assisted field goal creation. The heat chart makes that clear: 3PT shooting, catch-and-shoot efficiency, and assisted 3PM rates all sit well above the mean, showing how effective this archetype is in off-ball perimeter scoring situations.\n\nAt first glance, this group may seem similar to the 3PT-Reliant Sharpshooter archetype, but there are a few important differences. The Expanded Role Sharpshooter is more offensively versatile. While the 3PT-Reliant Sharpshooter skillset is almost entirely dependent on perimeter shooting, this archetype adds a meaningful midrange layer as well.\n\nThis shows up clearly in the median profile, where every midrange feature sits above the average guard value. This makes the group less one-dimensional than their 3PT-Reliant Sharpshooter counterpart: they still provide high-level off-ball three point gravity, but they also tend to attack the midrange area through handoffs and pull-up opportunities off three point pump fakes. Other than off-ball three point scenarios and some midrange self creation, the offensive profile of this archetype is quite barren: the playmaking, rim-pressure metrics are all well below the mean.',
    5: 'These players are defined by a balanced secondary-creator skillset built around above-average three point shooting, capable playmaking, and a slightly positive defensive impact. The heat chart makes this clear: wide-open 3PA frequency, pull-up 3PA frequency, catch-and-shoot 3PA frequency, 3PA frequency, 3P accuracy, points from 3s per 75, and percent of FGA from 3PA all sit above the mean. This is not a dominant self-creation perimeter archetype, but these players provide their outside shooting value through a mixture of spot-up shooting, movement shooting, and occasional pull-up attempts.\n\nThe playmaking profile is also silently impressive. The median values for dribble-to-turnover ratio, potential assist-to-turnover ratio, pass-turnover ratio, pass-shot ratio, and potential assist-FGA ratio are all above average, showing that this group is comfortable creating quality shots for their teammates. These metrics are not high enough to put these players in primary offensive engine conversations though. But their playmaking skillset is still developed enough to be considered useful as a secondary creator.\n\nThe defensive metrics are not elite, but they are slightly positive overall. The above-average deflections, offensive fouls drawn, and the on/off metrics suggest a scrappy defensive style built more around active hands than athletic dominance. This archetype’s value comes from this well-rounded combination of adept three point shooting, functional secondary playmaking, and enough defensive activity to stay on the floor in competitive lineups.',
    6: 'These players are defined by their ability to create shots for themselves and others from all three levels while carrying the ball for a significant portion of the game. The heat chart makes this clear: the 3pt shooting, midrange, rim pressure and playmaking features all sit well above the mean.\n\nHaving absurdly low assisted FGM rates, these players are clearly creating their own shots instead of relying on teammates to set them up. The Tight/very tight 3PA frequency, Pull-up 3PA frequency, Average dribbles before 3PA, Tight/very tight 2PA frequency, Pull-up 2PA frequency all being well above the mean exemplify this superb self creation ability.\n\nThe playmaking features are also clearly above average, which separates this archetype from their pure scoring counterpart in the 3-Level Scoring Threat archetype. Overall, this archetype represents a complete on-ball offensive genius, built around self-created shooting, three-level scoring, and playmaking talent.',
    7: 'These players are defined almost entirely by their elite playmaking skillset. The heat chart makes this extremely clear: dribble-to-turnover ratio, potential assist-to-turnover ratio, assist frequency, pass-turnover ratio, pass-shot ratio, potential assist frequency, and potential assist-FGA ratio are all well above the mean. Compared to every other archetype, this cluster holds by far the most impressive passing profile in the model.\n\nThe scoring profile is much more limited though. The three point metrics are mostly below average, especially 3PA frequency, points from threes per 75, and percent of FGA from 3PA. This group also does not stand out as a dominant rim-pressure or self-created scoring archetype. Their offensive value instead comes purely from playmaking and generating quality looks for teammates.\n\nThis archetype has some similarities to the Rim-Oriented, Lock-Down Point archetype, but the differences are pretty clear. The Talented, Pass-First Point has a sizable playmaking advantage, with much stronger passing volume and efficiency indicators. However, the defensive gap between the two groups is just as large in the other direction. The Rim-Oriented, Lock-Down Point archetype is built around remarkable defensive metrics, while this cluster is only slightly above average defensively. Overall, this archetype represents the purest table-setting guards in the model: elite passers who can run an offense, but whose value depends much more on playmaking than scoring or defense.',
    8: 'This is the second most populous cluster, and it is also one of the weakest grouped together cluster. These players are mainly grouped together by their limited playmaking impact. The heat chart makes this clear, with every median playmaking feature sitting well below the average guard value.\n\nThe scoring profiles within this archetype are less homogeneous than most other clusters. Some of these players create their own shots, while others do more of their damage off the catch, and their scoring can come from different areas across all three levels. What ties them together though is this shared score-first mentality.\n\nIn other words, this archetype is made up of guards who can get buckets in different ways, but their unwillingness to create for their teammates to become complete offensive engines. Their value comes from individual scoring rather than from participating and creating in a broader offensive picture.',
    9: 'These players are defined by their ability to consistently create offense inside the arc. Unlike the 3-Level Scoring + Playmaking Threat archetype, this group is not built around self created three point ability. The heat chart makes that clear: 3PA frequency, points from 3s per 75, percent of FGA from 3PA, and assisted 3PM rates are all below the average guard value. This lack of consistent perimeter volume is the main thing separating this archetype from the more complete three-level creators.\n\nInside the arc though, this cluster is extremely impressive. This archetype holds by far the strongest midrange profile in the model. Pull-up 2PA frequency, percent of FGA from midrange, and points from midrange per 75 all sit dramatically above the mean, showing how comfortable these players are at creating shots from the midrange area. The rim penetration metrics are also consistently well above average. Paint non-RA frequency, drive FGA frequency, percent of paint FGA from drives, points from drives per 75, and percent of FGA from drives all point toward a strong downhill scoring profile.\n\nAlso holding impressive playmaking metrics, this archetype becomes more than just an inside-the-arc scorer. These players can pressure the paint, self-create from the midrange at an elite level while simultaneously creating high quality opportunities for their teammates, signs of a true primary creator. Their offensive profile may not stretch defenses as much from three, but inside the arc, this is one of the most complete creator archetypes in the model.',
    10: 'These players are almost exclusively characterized by their world-class defensive abilities. The heat chart illustrates this further: Blocks per 75, Deflections per 75, Contested shot frequency, Opponent FG% difference are all consistently above the mean. Also, considering that guard defense usually does not disrupt opposing offenses as much as rim-protecting bigs or versatile wings, the negative opponent shot quality on/off metric is even more impressive than what it appears.\n\nUnfortunately, these players share a completely antithetical story on offense. Portrayed by their substantially low scoring features, these players mostly score from assisted open three pointers or dunks. These players are well below average at creating shots for others as well.\n\nAll things considered, this archetype holds undoubtedly the finest defending guards in the NBA, but the lack of reliant offensive skill separates this group from other defensive-minded archetypes.',
    11: 'These players are embodied by their limited scoring skillsets paired with exemplary defensive and secondary playmaking skills. The heat chart elucidates this: among all the scoring metrics, only the restricted area, dunking, and assisted three point rates are above average. The rest of the scoring metrics are well below the average.\n\nThe main value of this archetype comes from their defensive skillset. This can be seen in the heat chart where all of the defensive metrics, including the on/off metrics, are all well above the mean. This archetype also shows a willingness to pass. Features such as Potential assist-FGA ratio, Pass-shot ratio and all the other playmaking features being above average signify this.\n\nThis partially developed playmaking ability is the only thing separating this archetype from its pure defensive counterpart in the Streaky Shooting Defensive Savant.',
    12: 'These players are defined by an isolation-heavy playstyle and the ability to take and make difficult, contested shots from all three levels. The heat chart makes that pretty clear: pull-up scoring, tight/very tight shot-making, and the broader three-level scoring features all sit well above the mean, with most of them clearing the +1.5 standard deviation range.\n\nThis archetype has its fair share of lower-percentile features though. The heat chart makes it apparent that there is an obvious inadequacy in playmaking. These players are generally not creating as much for their teammates; almost all of the median playmaking features being well below the mean value. A less obvious inefficiency within this archetype is the lack of consistent free-throw generation, where they also lag behind the true elite offensive guards.\n\nThis combination of an unconvincing passing skill-set paired with inconsistent free throw generation is the only thing separating this group from the 3-Level Scoring + Playmaking Threat archetype.',
}


DEFAULT_FEATURES = []
ADDITIONAL_ALLOWED_FEATURES = []
ALLOWED_FEATURES = list(dict.fromkeys(CURRENT_CSV_FEATURES + ADDITIONAL_ALLOWED_FEATURES + BADGE_REQUIRED_FEATURES))
LOWER_IS_BETTER_PERCENTILE_FEATURES = {
    "opp_players_fg_pct_difference",
    "OPP_SHOT_QUALITY_ON_OFF",
    "OPP_EFG_PCT_ON_OFF",
}

PERCENTILE_AND_BADGE_EXCLUDED_NAMES = {
    "lebron james",
    "scottie barnes",
    "ben simmons",
}

DEFENSE_SCORE_CAPPED_NAMES = {
    "james harden",
    "damian lillard",
    "stephen curry",
}

ROWS_TO_REMOVE = [
    ("Ben Simmons", "2019-20", "PHI", "PG"),
    ("Ben Simmons", "2022-23", "BRK", "PG"),
    ("Ben Simmons", "2018-19", "PHI", "PG"),
    ("Ben Simmons", "2020-21", "PHI", "PG"),
    ("Ben Simmons", "2024-25", "BRK,LAC", "PG"),
]

ALLOWED_ALGORITHMS = {"kmeans"}
DEFAULT_ALGORITHM = "kmeans"
ALLOWED_DISTANCE_METRICS = ("euclidean",)
DEFAULT_DISTANCE_METRIC = "euclidean"
DEFAULT_KMEANS_K = 9
KMEANS_N_INIT = 50
KMEANS_RANDOM_STATE = 42
KMEANS_MAX_ITER = 300
KMEANS_TOL = 1e-4
PCA_EXPLAINED_VAR_TARGET = 0.87
COSINE_EPS = 1e-12

CLUSTER_NAME_BY_NUMBER = {
    1: "Pass-First, Perimeter Oriented Point",
    2: "Perimeter Oriented, Spark Plug Point",
    3: "Defensive Oriented, Playmaking Point",
    4: "Slashing+Midrange Oriented Shot Creator",
    5: "Prototypical 3-and-D Off-Guard",
    6: "Defensive Minded Utility Off-Guard",
    7: "Perimeter Oriented, Pull-up Shooting Engine",
    8: "Midrange + Paint Oriented Engine",
    9: "3PT-Reliant Sharpshooter",
}


class ClusterRequest(BaseModel):
    algorithm: Literal["kmeans"] = DEFAULT_ALGORITHM
    distance_metric: Literal["euclidean"] = DEFAULT_DISTANCE_METRIC
    k: int = Field(ge=2, le=30)
    features: List[str]


class PlayerDetailRequest(BaseModel):
    player_key: str


class ClusterReportRequest(BaseModel):
    algorithm: Literal["kmeans"] = DEFAULT_ALGORITHM
    distance_metric: Literal["euclidean"] = DEFAULT_DISTANCE_METRIC
    k: int = Field(ge=2, le=30)
    features: List[str]
    cluster_number: int = Field(ge=1, le=30)


class PlayerSkillBreakdownRequest(BaseModel):
    algorithm: Literal["kmeans"] = DEFAULT_ALGORITHM
    distance_metric: Literal["euclidean"] = DEFAULT_DISTANCE_METRIC
    k: int = Field(default=DEFAULT_KMEANS_K, ge=2, le=30)
    features: List[str] = Field(default_factory=list)
    player_key: str
    cluster_number: Optional[int] = Field(default=None, ge=1, le=30)


class PlayerComparisonRequest(BaseModel):
    player_keys: List[str] = Field(min_length=1, max_length=2)
    category: str = "traditional"
    mode: str = "raw_stats"


app = FastAPI(title="NBA Guard Cluster Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nbagalaxy.vercel.app",
        "https://nbagalaxy.com",
        "https://www.nbagalaxy.com",
        "https://nbagalaxy-git-main-harshaanand9s-projects.vercel.app",
        "https://nbagalaxy-fskfdlub4-harshaanand9s-projects.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_DATASET_CACHE: Dict[str, Dict] = {}


def stable_player_key(row: pd.Series) -> str:
    return "||".join(
        [
            str(row["Player Name"]),
            str(row["Season"]),
            str(row["teams_played"]),
            str(row["position"]),
        ]
    )


def get_locked_euclidean_kmeans_feature_columns(raw: bool = False) -> List[str]:
    feature_columns: List[str] = []
    for group_name in EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER:
        feature_columns.extend(EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES[group_name])
    return feature_columns if raw else ordered_unique(feature_columns)


def normalize_player_name_for_assignment_key(player_name: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(player_name))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized


def normalize_player_name_for_headshot(player_name: object) -> str:
    normalized_key = normalize_player_name_for_assignment_key(player_name)
    return HEADSHOT_NAME_ALIASES.get(normalized_key, normalized_key)


def find_headshot_map_path() -> Optional[Path]:
    path_candidates = [
        HEADSHOT_MAP_PATH,
        BACKEND_DATA_DIR / "player_headshots.csv",
        BACKEND_DIR.parent / "data" / "player_headshots.csv",
        BACKEND_DIR.parent / "player_headshots.csv",
    ]
    for path_candidate in path_candidates:
        expanded_path = path_candidate.expanduser()
        if expanded_path.exists():
            return expanded_path
    return None


def load_headshot_rows_by_key() -> Dict[str, Dict[str, object]]:
    csv_path = find_headshot_map_path()
    if csv_path is None:
        return {}

    mtime_ns = csv_path.stat().st_mtime_ns
    cached_rows = _HEADSHOT_MAP_CACHE.get("rows_by_key")
    if (
        cached_rows is not None
        and _HEADSHOT_MAP_CACHE.get("path") == str(csv_path)
        and _HEADSHOT_MAP_CACHE.get("mtime_ns") == mtime_ns
    ):
        return cached_rows

    dataframe = pd.read_csv(csv_path)
    rows_by_key: Dict[str, Dict[str, object]] = {}
    for _, row in dataframe.iterrows():
        player_name = row.get("player_name", "")
        normalized_name = row.get("normalized_player_name", "")
        keys = {
            normalize_player_name_for_headshot(player_name),
            normalize_player_name_for_headshot(normalized_name),
        }
        for key in keys:
            if key:
                rows_by_key[key] = row.to_dict()

    _HEADSHOT_MAP_CACHE["path"] = str(csv_path)
    _HEADSHOT_MAP_CACHE["mtime_ns"] = mtime_ns
    _HEADSHOT_MAP_CACHE["rows_by_key"] = rows_by_key
    return rows_by_key


def get_player_headshot_payload(player_name: object) -> Dict[str, object]:
    normalized_key = normalize_player_name_for_headshot(player_name)
    row = load_headshot_rows_by_key().get(normalized_key)
    if row is None:
        return {
            "headshot_url": HEADSHOT_FALLBACK_URL,
            "headshot_cdn_url": None,
            "nba_person_id": None,
            "nba_full_name": None,
            "headshot_match_status": "unmatched",
        }

    local_headshot_path = safe_json_value(row.get("local_headshot_path"), None)
    headshot_url = str(local_headshot_path).strip() if local_headshot_path else ""
    if not headshot_url:
        headshot_url = HEADSHOT_FALLBACK_URL

    return {
        "headshot_url": headshot_url,
        "headshot_cdn_url": safe_json_value(row.get("headshot_url"), None),
        "nba_person_id": str(row.get("nba_person_id")) if safe_json_value(row.get("nba_person_id"), None) is not None else None,
        "nba_full_name": safe_json_value(row.get("nba_full_name"), None),
        "headshot_match_status": safe_json_value(row.get("match_status"), "matched"),
    }


def build_assignment_key(player_name: object, season: object) -> str:
    return f"{normalize_player_name_for_assignment_key(player_name)}||{str(season).strip()}"


def load_dlebron_lookup() -> Dict[str, float]:
    """Load D-LEBRON by normalized player-season from the player-comps CSV.

    This is intentionally a narrow side-load used only for the defensive skill
    breakdown. The main skill-breakdown dataset remains the normal full-season
    feature file so the other four skill blocks keep their existing columns and
    formulas.
    """
    path = Path(PLAYER_COMPS_DATASET_PATH).expanduser()
    if not path.exists():
        return {}

    mtime_ns = path.stat().st_mtime_ns
    cached_lookup = _DLEBRON_SOURCE_CACHE.get("lookup")
    if (
        cached_lookup is not None
        and _DLEBRON_SOURCE_CACHE.get("path") == str(path)
        and _DLEBRON_SOURCE_CACHE.get("mtime_ns") == mtime_ns
    ):
        return cached_lookup  # type: ignore[return-value]

    source = pd.read_csv(path)
    required_columns = {"Player Name", "Season", DLEBRON_FEATURE}
    if not required_columns.issubset(set(source.columns)):
        lookup: Dict[str, float] = {}
    else:
        compact = source[["Player Name", "Season", DLEBRON_FEATURE]].copy()
        compact[DLEBRON_FEATURE] = pd.to_numeric(compact[DLEBRON_FEATURE], errors="coerce")
        compact = compact.dropna(subset=[DLEBRON_FEATURE])
        compact["_assignment_key"] = compact.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)
        compact = compact.drop_duplicates(subset=["_assignment_key"], keep="last")
        lookup = {
            str(row["_assignment_key"]): float(row[DLEBRON_FEATURE])
            for _, row in compact.iterrows()
            if pd.notna(row[DLEBRON_FEATURE])
        }

    _DLEBRON_SOURCE_CACHE["path"] = str(path)
    _DLEBRON_SOURCE_CACHE["mtime_ns"] = mtime_ns
    _DLEBRON_SOURCE_CACHE["lookup"] = lookup
    return lookup


PLAYER_COMPS_SIMILARITY_SIDELOAD_FEATURES = [
    DLEBRON_FEATURE,
    "passing_potential_ast",
    "passing_ft_ast",
    "hustle_contested_shots",
    "offensive_fouls_drawn",
]

FREQUENCY_TO_PER_GAME_MAP = {
    "tight_very_tight_3fga_frequency": {"output": "tight_very_tight_3fga_per_game", "possession_base": "OffPoss"},
    "open_3fga_frequency": {"output": "open_3fga_per_game", "possession_base": "OffPoss"},
    "Wide_Open_3FGA_Frequency": {"output": "Wide_Open_3FGA_per_game", "possession_base": "OffPoss"},
    "pull_up_3P_frequency": {"output": "pull_up_3PA_per_game", "possession_base": "OffPoss"},
    "catch_shoot_3P_frequency": {"output": "catch_shoot_3PA_per_game", "possession_base": "OffPoss"},
    "tight_very_tight_2fga_frequency": {"output": "tight_very_tight_2fga_per_game", "possession_base": "OffPoss"},
    "open_2fga_frequency": {"output": "open_2fga_per_game", "possession_base": "OffPoss"},
    "pull_up_2P_frequency": {"output": "pull_up_2PA_per_game", "possession_base": "OffPoss"},
    "RestrictedArea_Frequency": {"output": "restricted_area_fga_per_game", "possession_base": "OffPoss"},
    "Paint_Non_RA_Frequency": {"output": "paint_non_ra_fga_per_game", "possession_base": "OffPoss"},
    "drive_fga_frequency": {"output": "drive_fga_per_game", "possession_base": "OffPoss"},
    "drive_frequency": {"output": "drives_per_game", "possession_base": "OffPoss"},
    "potential_assist_frequency": {"output": "potential_assists_per_game", "possession_base": "OffPoss"},
    "contested_shot_frequency": {"output": "contested_shots_per_game", "possession_base": "DefPoss"},
    "off_fouls_drawn_frequency": {"output": "off_fouls_drawn_per_game", "possession_base": "DefPoss"},
}


def safe_per_game_from_frequency(
    frequency_series: pd.Series,
    possession_series: pd.Series,
    gp_series: pd.Series,
) -> pd.Series:
    frequency_values = pd.to_numeric(frequency_series, errors="coerce")
    possession_values = pd.to_numeric(possession_series, errors="coerce")
    gp_values = pd.to_numeric(gp_series, errors="coerce")
    per_game_values = np.where(gp_values > 0, frequency_values * possession_values / gp_values, np.nan)
    return pd.Series(per_game_values, index=frequency_series.index, dtype=float)


def load_player_comps_side_feature_lookup(feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    path = Path(PLAYER_COMPS_DATASET_PATH).expanduser()
    if not path.exists():
        return {}

    try:
        source = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}

    required_columns = {"Player Name", "Season"}
    available_feature_names = [feature_name for feature_name in feature_names if feature_name in source.columns]
    if not required_columns.issubset(set(source.columns)) or not available_feature_names:
        return {}

    compact = source[["Player Name", "Season", *available_feature_names]].copy()
    compact["_assignment_key"] = compact.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)
    compact = compact.drop_duplicates(subset=["_assignment_key"], keep="last")

    lookup: Dict[str, Dict[str, float]] = {}
    for _, row in compact.iterrows():
        feature_payload: Dict[str, float] = {}
        for feature_name in available_feature_names:
            numeric_value = pd.to_numeric(pd.Series([row[feature_name]]), errors="coerce").iloc[0]
            if pd.notna(numeric_value):
                feature_payload[feature_name] = float(numeric_value)
        if feature_payload:
            lookup[str(row["_assignment_key"])] = feature_payload
    return lookup


def load_pullup_2pa_per_game_lookup() -> Dict[str, float]:
    path = Path(PULLUP_DATASET_PATH).expanduser()
    if not path.exists():
        return {}

    mtime_ns = path.stat().st_mtime_ns
    cached_lookup = _PULLUP_SOURCE_CACHE.get("lookup")
    if (
        cached_lookup is not None
        and _PULLUP_SOURCE_CACHE.get("path") == str(path)
        and _PULLUP_SOURCE_CACHE.get("mtime_ns") == mtime_ns
    ):
        return cached_lookup  # type: ignore[return-value]

    try:
        source = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}

    required_columns = {"Player Name", "Season", "pullup_pull_up_fga", "pullup_pull_up_fg3a"}
    if not required_columns.issubset(set(source.columns)):
        return {}

    compact = source[["Player Name", "Season", "pullup_pull_up_fga", "pullup_pull_up_fg3a"]].copy()
    compact["_assignment_key"] = compact.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)
    compact["pull_up_2PA_per_game"] = (
        pd.to_numeric(compact["pullup_pull_up_fga"], errors="coerce")
        - pd.to_numeric(compact["pullup_pull_up_fg3a"], errors="coerce")
    ).clip(lower=0.0)
    compact = compact.dropna(subset=["pull_up_2PA_per_game"])
    compact = compact.drop_duplicates(subset=["_assignment_key"], keep="last")
    lookup = {
        str(row["_assignment_key"]): float(row["pull_up_2PA_per_game"])
        for _, row in compact.iterrows()
        if pd.notna(row["pull_up_2PA_per_game"])
    }

    _PULLUP_SOURCE_CACHE["path"] = str(path)
    _PULLUP_SOURCE_CACHE["mtime_ns"] = mtime_ns
    _PULLUP_SOURCE_CACHE["lookup"] = lookup
    return lookup


def attach_pullup_2pa_per_game(guards: pd.DataFrame) -> pd.DataFrame:
    lookup = load_pullup_2pa_per_game_lookup()
    if not lookup:
        return guards

    output = guards.copy()
    if "pull_up_2PA_per_game" not in output.columns:
        output["pull_up_2PA_per_game"] = np.nan

    keys = output.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)
    pullup_values = keys.map(lookup)
    output["pull_up_2PA_per_game"] = pullup_values.fillna(pd.to_numeric(output["pull_up_2PA_per_game"], errors="coerce"))
    return output


def attach_player_comps_side_features(guards: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    lookup = load_player_comps_side_feature_lookup(feature_names)
    output = guards.copy()
    for feature_name in feature_names:
        if feature_name not in output.columns:
            output[feature_name] = np.nan

    if not lookup:
        return output

    keys = output.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)
    for feature_name in feature_names:
        side_values = keys.map(lambda key: lookup.get(str(key), {}).get(feature_name, np.nan))
        output[feature_name] = pd.to_numeric(output[feature_name], errors="coerce").fillna(side_values)
    return output


def add_locked_similarity_derived_features(guards: pd.DataFrame) -> pd.DataFrame:
    output = attach_player_comps_side_features(guards, PLAYER_COMPS_SIMILARITY_SIDELOAD_FEATURES)

    for _, spec in FREQUENCY_TO_PER_GAME_MAP.items():
        output_column = str(spec["output"])
        if output_column not in output.columns:
            output[output_column] = np.nan

    if "GP" in output.columns:
        for frequency_column, spec in FREQUENCY_TO_PER_GAME_MAP.items():
            output_column = str(spec["output"])
            possession_column = str(spec["possession_base"])
            if frequency_column not in output.columns or possession_column not in output.columns:
                continue
            output[output_column] = safe_per_game_from_frequency(
                frequency_series=output[frequency_column],
                possession_series=output[possession_column],
                gp_series=output["GP"],
            )

    output = attach_pullup_2pa_per_game(output)

    if "hustle_contested_shots" in output.columns and "contested_shots_per_game" in output.columns:
        output["contested_shots_per_game"] = pd.to_numeric(output["contested_shots_per_game"], errors="coerce").fillna(
            pd.to_numeric(output["hustle_contested_shots"], errors="coerce")
        )
    if "offensive_fouls_drawn" in output.columns and "off_fouls_drawn_per_game" in output.columns:
        output["off_fouls_drawn_per_game"] = pd.to_numeric(output["off_fouls_drawn_per_game"], errors="coerce").fillna(
            pd.to_numeric(output["offensive_fouls_drawn"], errors="coerce")
        )

    potential_assists = pd.Series(np.nan, index=output.index, dtype=float)
    if "passing_potential_ast" in output.columns:
        potential_assists = pd.to_numeric(output["passing_potential_ast"], errors="coerce")
    if potential_assists.isna().all() and "potential_assists_per_game" in output.columns:
        potential_assists = pd.to_numeric(output["potential_assists_per_game"], errors="coerce")

    ft_assists = pd.Series(np.nan, index=output.index, dtype=float)
    if "passing_ft_ast" in output.columns:
        ft_assists = pd.to_numeric(output["passing_ft_ast"], errors="coerce")

    combined_playmaking_volume = potential_assists.fillna(0.0) + ft_assists.fillna(0.0)
    combined_playmaking_volume = combined_playmaking_volume.where(potential_assists.notna() | ft_assists.notna(), np.nan)
    output["potential_assists_and_ft_assists"] = combined_playmaking_volume.astype(float)

    return output


def attach_dlebron_to_skill_breakdown_guards(guards: pd.DataFrame) -> pd.DataFrame:
    """Return guards with D-LEBRON populated from the player-comps CSV when needed."""
    return attach_player_comps_side_features(guards, [DLEBRON_FEATURE])


def attach_dlebron_to_guards(guards: pd.DataFrame) -> pd.DataFrame:
    return attach_dlebron_to_skill_breakdown_guards(guards)


def attach_dlebron_to_skill_breakdown_guards_legacy(guards: pd.DataFrame) -> pd.DataFrame:
    """Legacy narrow D-LEBRON lookup kept for compatibility with old call paths."""

    lookup = load_dlebron_lookup()
    if not lookup:
        return guards

    output = guards.copy()
    keys = output.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)
    dlebron_values = keys.map(lookup)
    if DLEBRON_FEATURE in output.columns:
        output[DLEBRON_FEATURE] = pd.to_numeric(output[DLEBRON_FEATURE], errors="coerce").fillna(dlebron_values)
    else:
        output[DLEBRON_FEATURE] = dlebron_values
    return output


def is_locked_euclidean_kmeans_request(algorithm: str, distance_metric: str) -> bool:
    return algorithm == "kmeans" and distance_metric == "euclidean"


def get_cluster_title(cluster_number: int, algorithm: Optional[str] = None, distance_metric: Optional[str] = None) -> str:
    if algorithm == "kmeans" and distance_metric == "euclidean":
        return EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER.get(int(cluster_number), f"Cluster {int(cluster_number)}")
    return CLUSTER_NAME_BY_NUMBER.get(int(cluster_number), f"Cluster {int(cluster_number)}")


def get_cluster_description(cluster_number: int, algorithm: Optional[str] = None, distance_metric: Optional[str] = None) -> str:
    if algorithm == "kmeans" and distance_metric == "euclidean":
        return EUCLIDEAN_KMEANS_CLUSTER_DESCRIPTION_BY_NUMBER.get(int(cluster_number), "")
    return ""


def find_similar_players_csv_path() -> Optional[Path]:
    for path in SIMILAR_PLAYERS_PATHS:
        expanded_path = path.expanduser()
        if expanded_path.exists():
            return expanded_path
    return None


def normalize_filter_value(value: object) -> str:
    return str(value).strip().lower()


def parse_optional_float(value: object) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = normalize_filter_value(value)
    return normalized in {"1", "true", "t", "yes", "y"}


def safe_json_value(value: object, fallback: object = None) -> object:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_similar_players_dataframe() -> tuple[pd.DataFrame, Path]:
    csv_path = find_similar_players_csv_path()
    if csv_path is None:
        searched_paths = " | ".join(str(path) for path in SIMILAR_PLAYERS_PATHS)
        raise FileNotFoundError(f"similar_players_precomputed_production.csv not found. Searched: {searched_paths}")

    mtime_ns = csv_path.stat().st_mtime_ns
    cached_dataframe = _SIMILAR_PLAYERS_CACHE.get("dataframe")
    if (
        cached_dataframe is not None
        and _SIMILAR_PLAYERS_CACHE.get("path") == str(csv_path)
        and _SIMILAR_PLAYERS_CACHE.get("mtime_ns") == mtime_ns
    ):
        return cached_dataframe.copy(), csv_path

    dataframe = pd.read_csv(csv_path)
    required_columns = {"player_name", "season", "related_player_name", "related_season", "rank"}
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"similar players CSV is missing required columns: {missing_columns}")

    _SIMILAR_PLAYERS_CACHE["path"] = str(csv_path)
    _SIMILAR_PLAYERS_CACHE["mtime_ns"] = mtime_ns
    _SIMILAR_PLAYERS_CACHE["dataframe"] = dataframe.copy()
    return dataframe, csv_path


def filter_similar_players_dataframe(
    dataframe: pd.DataFrame,
    player_name: str,
    season: str,
    pipeline: Optional[str],
    k: Optional[int],
    pca_variance_target: Optional[str],
) -> pd.DataFrame:
    candidate_rows = dataframe.loc[
        (dataframe["player_name"].map(normalize_filter_value) == normalize_filter_value(player_name))
        & (dataframe["season"].map(normalize_filter_value) == normalize_filter_value(season))
    ].copy()

    if candidate_rows.empty:
        return candidate_rows

    # Apply optional filters only when they keep at least one row. This preserves the endpoint
    # across slightly different pipeline labels or numeric formatting in precomputed CSVs.
    if pipeline and "pipeline" in candidate_rows.columns:
        filtered_rows = candidate_rows.loc[
            candidate_rows["pipeline"].map(normalize_filter_value) == normalize_filter_value(pipeline)
        ].copy()
        if not filtered_rows.empty:
            candidate_rows = filtered_rows

    if k is not None and "k" in candidate_rows.columns:
        numeric_k = pd.to_numeric(candidate_rows["k"], errors="coerce")
        filtered_rows = candidate_rows.loc[numeric_k == int(k)].copy()
        if not filtered_rows.empty:
            candidate_rows = filtered_rows

    if pca_variance_target and "pca_variance_target" in candidate_rows.columns:
        requested_float = parse_optional_float(pca_variance_target)
        pca_series = candidate_rows["pca_variance_target"]
        if requested_float is not None:
            numeric_pca = pd.to_numeric(pca_series, errors="coerce")
            filtered_rows = candidate_rows.loc[np.isclose(numeric_pca, requested_float, rtol=0.0, atol=1e-8)].copy()
        else:
            filtered_rows = candidate_rows.loc[
                pca_series.map(normalize_filter_value) == normalize_filter_value(pca_variance_target)
            ].copy()
        if not filtered_rows.empty:
            candidate_rows = filtered_rows

    return candidate_rows


def row_int(row: pd.Series, column_name: str, fallback: int = 0) -> int:
    if column_name not in row.index:
        return fallback
    value = pd.to_numeric(pd.Series([row[column_name]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return fallback
    return int(value)


def row_float(row: pd.Series, column_name: str, fallback: float = 0.0) -> float:
    if column_name not in row.index:
        return fallback
    value = pd.to_numeric(pd.Series([row[column_name]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return fallback
    return float(value)


def row_string(row: pd.Series, column_name: str, fallback: str = "") -> str:
    if column_name not in row.index:
        return fallback
    value = safe_json_value(row[column_name], fallback)
    return fallback if value is None else str(value)


def build_block_scores(row: pd.Series) -> Dict[str, Dict[str, float]]:
    return {
        "ThreePT": {
            "distance": row_float(row, "threept_distance"),
            "similarity_score": row_float(row, "threept_similarity_score"),
        },
        "MidRange": {
            "distance": row_float(row, "midrange_distance"),
            "similarity_score": row_float(row, "midrange_similarity_score"),
        },
        "RimPressure": {
            "distance": row_float(row, "rimpressure_distance"),
            "similarity_score": row_float(row, "rimpressure_similarity_score"),
        },
        "Playmaking": {
            "distance": row_float(row, "playmaking_distance"),
            "similarity_score": row_float(row, "playmaking_similarity_score"),
        },
        "Defense": {
            "distance": row_float(row, "defense_distance"),
            "similarity_score": row_float(row, "defense_similarity_score"),
        },
    }


def similar_players_rows_have_detail_payload(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return False

    missing_columns = [
        column_name
        for column_name in SIMILAR_PLAYERS_REQUIRED_DETAIL_COLUMNS
        if column_name not in rows.columns
    ]
    if missing_columns:
        return False

    score_frame = rows[SIMILAR_PLAYERS_BLOCK_SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if score_frame.isna().any().any():
        return False
    if float(score_frame.abs().sum().sum()) <= 0.0:
        return False

    for column_name in ["strongest_similarity_blocks", "biggest_difference_blocks"]:
        non_empty_mask = rows[column_name].map(lambda value: bool(str(safe_json_value(value, "")).strip()))
        if not bool(non_empty_mask.all()):
            return False

    return True


def ordered_unique(items: List[str]) -> List[str]:
    seen = set()
    ordered_items: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered_items.append(item)
    return ordered_items


def compute_season_percentiles(numeric_all: pd.DataFrame, seasons: pd.Series) -> pd.DataFrame:
    percentile_df = pd.DataFrame(index=numeric_all.index, columns=numeric_all.columns, dtype=float)

    for _, idx in seasons.groupby(seasons).groups.items():
        season_block = numeric_all.loc[idx]
        for feature in numeric_all.columns:
            rank_ascending = feature not in LOWER_IS_BETTER_PERCENTILE_FEATURES
            ranked = season_block[feature].rank(method="average", pct=True, ascending=rank_ascending) * 100.0
            percentile_df.loc[idx, feature] = ranked

    return percentile_df.fillna(0.0)


def compute_feature_percentile_by_season(feature_values: pd.Series, seasons: pd.Series, lower_is_better: bool = False) -> pd.Series:
    percentile_series = pd.Series(index=feature_values.index, dtype=float)

    for _, idx in seasons.groupby(seasons).groups.items():
        season_values = feature_values.loc[idx]
        ranked = season_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
        percentile_series.loc[idx] = ranked

    return percentile_series.fillna(0.0)


def compute_local_feature_percentile_by_season(
    feature_values: pd.Series,
    anchor_values: pd.Series,
    seasons: pd.Series,
    lower_is_better: bool = False,
    anchor_lower_is_better: bool = False,
    mode: str = "pm_10",
    min_peer_count: int = 5,
) -> pd.Series:
    local_percentile_series = pd.Series(index=feature_values.index, dtype=float)
    anchor_percentile_series = compute_feature_percentile_by_season(
        anchor_values,
        seasons,
        lower_is_better=anchor_lower_is_better,
    )

    for _, idx in seasons.groupby(seasons).groups.items():
        season_feature_values = feature_values.loc[idx]
        season_anchor_percentiles = anchor_percentile_series.loc[idx]
        season_valid_feature_values = season_feature_values.dropna()

        for row_index in idx:
            player_value = feature_values.loc[row_index]
            player_anchor_percentile = anchor_percentile_series.loc[row_index]
            if pd.isna(player_value) or pd.isna(player_anchor_percentile):
                local_percentile_series.loc[row_index] = 0.0
                continue

            if mode == "floor_to_100":
                lower_bound = max(0.0, float(player_anchor_percentile) - 10.0)
                peer_mask = season_anchor_percentiles >= lower_bound
            else:
                lower_bound = max(0.0, float(player_anchor_percentile) - 10.0)
                upper_bound = min(100.0, float(player_anchor_percentile) + 10.0)
                peer_mask = season_anchor_percentiles.between(lower_bound, upper_bound, inclusive="both")

            peer_values = season_feature_values.loc[peer_mask].dropna()
            if len(peer_values) < min_peer_count:
                peer_values = season_valid_feature_values
            if peer_values.empty:
                local_percentile_series.loc[row_index] = 0.0
                continue

            comparison_values = pd.concat([peer_values, pd.Series([float(player_value)], index=["__player__"])])
            ranked = comparison_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
            local_percentile_series.loc[row_index] = float(ranked.loc["__player__"])

    return local_percentile_series.fillna(0.0)


def build_component_score_frame(
    guards: pd.DataFrame,
    group_order: List[str],
    group_features: Dict[str, List[str]],
    lower_is_better_by_group: Optional[Dict[str, set]] = None,
    local_percentile_rules_by_group: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
) -> tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    lower_is_better_by_group = lower_is_better_by_group or {}
    local_percentile_rules_by_group = local_percentile_rules_by_group or {}
    all_features = ordered_unique([
        feature_name
        for features_in_group in group_features.values()
        for feature_name in features_in_group
    ])
    missing_features = [feature_name for feature_name in all_features if feature_name not in guards.columns]
    if missing_features:
        return pd.DataFrame(index=guards.index), {}, missing_features

    numeric_feature_frame = guards[all_features].apply(pd.to_numeric, errors="coerce")
    numeric_feature_frame = numeric_feature_frame.replace([np.inf, -np.inf], np.nan)

    component_score_frame = pd.DataFrame(index=guards.index)
    used_group_features: Dict[str, List[str]] = {}

    for group_name in group_order:
        group_percentiles = []
        available_group_features = [
            feature_name
            for feature_name in group_features[group_name]
            if feature_name in numeric_feature_frame.columns
        ]
        used_group_features[group_name] = available_group_features

        for feature_name in available_group_features:
            lower_is_better = (
                feature_name in LOWER_IS_BETTER_PERCENTILE_FEATURES
                or feature_name in lower_is_better_by_group.get(group_name, set())
            )
            local_rule = local_percentile_rules_by_group.get(group_name, {}).get(feature_name)
            if local_rule:
                anchor_feature_name = str(local_rule.get("anchor_feature", ""))
                if anchor_feature_name not in numeric_feature_frame.columns:
                    return pd.DataFrame(index=guards.index), {}, [anchor_feature_name]
                anchor_lower_is_better = (
                    anchor_feature_name in LOWER_IS_BETTER_PERCENTILE_FEATURES
                    or anchor_feature_name in lower_is_better_by_group.get(group_name, set())
                )
                feature_percentile = compute_local_feature_percentile_by_season(
                    numeric_feature_frame[feature_name],
                    numeric_feature_frame[anchor_feature_name],
                    guards["Season"],
                    lower_is_better=lower_is_better,
                    anchor_lower_is_better=anchor_lower_is_better,
                    mode=str(local_rule.get("mode", "pm_10")),
                )
            else:
                feature_percentile = compute_feature_percentile_by_season(
                    numeric_feature_frame[feature_name],
                    guards["Season"],
                    lower_is_better=lower_is_better,
                )
            group_percentiles.append(
                feature_percentile.where(numeric_feature_frame[feature_name].notna())
            )

        if group_percentiles:
            group_percentile_frame = pd.concat(group_percentiles, axis=1)
            component_score_frame[group_name] = group_percentile_frame.median(axis=1, skipna=True).fillna(0.0)
        else:
            component_score_frame[group_name] = 0.0

    return component_score_frame, used_group_features, []


def normalize_player_name_for_percentile_pool(player_name: object) -> str:
    text = "" if player_name is None else str(player_name)
    text = text.replace("ı", "i").replace("İ", "I").replace("ø", "o").replace("Ø", "O").replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)


def build_percentile_peer_mask(guards: pd.DataFrame) -> pd.Series:
    if "Player Name" not in guards.columns:
        return pd.Series(True, index=guards.index, dtype=bool)
    normalized_names = guards["Player Name"].map(normalize_player_name_for_percentile_pool)
    return ~normalized_names.isin(PERCENTILE_AND_BADGE_EXCLUDED_NAMES)


def _safe_float(value: object, fallback: float = 0.0) -> float:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return fallback if pd.isna(numeric_value) else float(numeric_value)


def _series_percentile_from_peer_pool(values: pd.Series, seasons: pd.Series, lower_is_better: bool = False, peer_mask: Optional[pd.Series] = None, min_peer_count: int = 5) -> pd.Series:
    percentile_series = pd.Series(0.0, index=values.index, dtype=float)
    if peer_mask is None:
        peer_mask = pd.Series(True, index=values.index, dtype=bool)
    peer_mask = peer_mask.reindex(values.index).fillna(False).astype(bool)
    for _, season_indices in seasons.groupby(seasons).groups.items():
        season_indices = pd.Index(season_indices)
        season_peer_indices = season_indices[peer_mask.loc[season_indices].to_numpy()]
        peer_values = values.loc[season_peer_indices].dropna()
        fallback_peer_values = values.loc[season_indices].dropna()
        if len(peer_values) < min_peer_count:
            peer_values = fallback_peer_values
        if peer_values.empty:
            continue
        ranked_peer_values = peer_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
        for row_index in season_indices:
            player_value = values.loc[row_index]
            if pd.isna(player_value):
                percentile_series.loc[row_index] = 0.0
            elif row_index in ranked_peer_values.index:
                percentile_series.loc[row_index] = float(ranked_peer_values.loc[row_index])
            else:
                comparison_values = pd.concat([peer_values, pd.Series([float(player_value)], index=["__player__"])])
                ranked = comparison_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
                percentile_series.loc[row_index] = float(ranked.loc["__player__"])
    return percentile_series.fillna(0.0)


def _local_percentile_from_peer_pool(values: pd.Series, anchor_percentiles: pd.Series, seasons: pd.Series, row_index: int, lower_is_better: bool = False, peer_mask: Optional[pd.Series] = None, mode: str = "floor_to_100", min_peer_count: int = 5) -> float:
    if row_index not in values.index:
        return 0.0
    player_value = values.loc[row_index]
    player_anchor_percentile = anchor_percentiles.loc[row_index]
    if pd.isna(player_value) or pd.isna(player_anchor_percentile):
        return 0.0
    if peer_mask is None:
        peer_mask = pd.Series(True, index=values.index, dtype=bool)
    peer_mask = peer_mask.reindex(values.index).fillna(False).astype(bool)
    season_mask = seasons.astype(str).eq(str(seasons.loc[row_index]))
    lower_bound = max(0.0, float(player_anchor_percentile) - 10.0)
    if mode == "pm_10":
        upper_bound = min(100.0, float(player_anchor_percentile) + 10.0)
        local_mask = anchor_percentiles.between(lower_bound, upper_bound, inclusive="both")
    else:
        local_mask = anchor_percentiles >= lower_bound
    peer_values = values.loc[season_mask & peer_mask & local_mask].dropna()
    fallback_values = values.loc[season_mask & peer_mask].dropna()
    if len(peer_values) < min_peer_count:
        peer_values = fallback_values
    if peer_values.empty:
        peer_values = values.loc[season_mask].dropna()
    if peer_values.empty:
        return 0.0
    if row_index in peer_values.index:
        ranked = peer_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
        return float(ranked.loc[row_index])
    comparison_values = pd.concat([peer_values, pd.Series([float(player_value)], index=["__player__"])])
    ranked = comparison_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
    return float(ranked.loc["__player__"])


def _median_percentile(values: List[float]) -> float:
    cleaned_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    return 0.0 if not cleaned_values else float(np.median(cleaned_values))


def _season_zscore_series_from_peer_pool(values: pd.Series, seasons: pd.Series, peer_mask: pd.Series, lower_is_better: bool = False) -> pd.Series:
    zscore_series = pd.Series(0.0, index=values.index, dtype=float)
    peer_mask = peer_mask.reindex(values.index).fillna(False).astype(bool)
    for _, season_indices in seasons.groupby(seasons).groups.items():
        season_indices = pd.Index(season_indices)
        peer_indices = season_indices[peer_mask.loc[season_indices].to_numpy()]
        peer_values = values.loc[peer_indices].dropna()
        if len(peer_values) < 2:
            peer_values = values.loc[season_indices].dropna()
        if peer_values.empty:
            continue
        center = float(peer_values.mean())
        scale = float(peer_values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        z = (values.loc[season_indices].astype(float) - center) / scale
        if lower_is_better:
            z = -z
        zscore_series.loc[season_indices] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return zscore_series.fillna(0.0)


def _compute_defensive_skill_components(guards: pd.DataFrame, numeric_frame: pd.DataFrame, percentile_frame: pd.DataFrame, peer_mask: pd.Series) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    defense_percentile = percentile_frame[DLEBRON_FEATURE]
    frame = pd.DataFrame(index=guards.index, dtype=float)
    frame[DLEBRON_FEATURE] = defense_percentile.fillna(0.0)
    frame["Defense"] = defense_percentile.fillna(0.0)
    feature_groups = {DLEBRON_FEATURE: [DLEBRON_FEATURE]}
    return frame, feature_groups


def build_skill_breakdown_score_frames(guards: pd.DataFrame, breakdown_kind: str) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, List[str]], List[str], List[str]]:
    three_pt_features = ["Avg3ptShotDistance", "3fga_frequency", "3P_Accuracy", "catch_shoot_3P_frequency", "catch_shoot_3P_accuracy", "avg_closest_defender_3FGA", "pct_3fga_wide_open", "tight_very_tight_3fga_frequency", "pull_up_3P_frequency", "pull_up_3P_accuracy", "traditional_fg3a"]
    if breakdown_kind == "three_pt_breakdown":
        required_features = ordered_unique(three_pt_features)
    else:
        guards = attach_dlebron_to_skill_breakdown_guards(guards)
        required_features = ordered_unique(three_pt_features + ["MidRangeFrequency", "MidRangeAccuracy", "by_zone_statistics_mid_range_fga", "tight_very_tight_2fga_frequency", "drives_drive_fga", "drives_drives", "pts_from_drives_per_75", "drive_fga_frequency", "drive_fg_pct", "assist_frequency", "potential_assist_frequency", "pts_created_from_assists", "potential_assist_tov_ratio", "assists_tov_ratio", "pts_created_to_tov_ratio", "crafted_box_creation", "crafted_passer_rating", DLEBRON_FEATURE])
    missing_features = [feature_name for feature_name in required_features if feature_name not in guards.columns]
    if missing_features:
        return pd.DataFrame(index=guards.index), {}, {}, missing_features, required_features
    numeric_frame = guards[required_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    seasons = guards["Season"]
    peer_mask = build_percentile_peer_mask(guards)
    percentile_frame = pd.DataFrame(index=guards.index, dtype=float)
    lower_extra = {"pct_3fga_wide_open", "avg_closest_defender_3FGA"}
    for feature_name in required_features:
        percentile_frame[feature_name] = _series_percentile_from_peer_pool(numeric_frame[feature_name], seasons, lower_is_better=feature_name in LOWER_IS_BETTER_PERCENTILE_FEATURES or feature_name in lower_extra, peer_mask=peer_mask)
    def pct(row_index: int, feature_name: str) -> float:
        return _safe_float(percentile_frame.at[row_index, feature_name])
    def local(row_index: int, target_feature: str, anchor_feature: str, lower_is_better: bool = False) -> float:
        return _local_percentile_from_peer_pool(numeric_frame[target_feature], percentile_frame[anchor_feature], seasons, row_index, lower_is_better=lower_is_better, peer_mask=peer_mask, mode="floor_to_100")
    threept_rows = []
    for row_index in guards.index:
        contested = _median_percentile([pct(row_index, "pct_3fga_wide_open"), pct(row_index, "avg_closest_defender_3FGA"), local(row_index, "3P_Accuracy", "avg_closest_defender_3FGA"), pct(row_index, "tight_very_tight_3fga_frequency"), pct(row_index, "3fga_frequency")])
        if pct(row_index, "3fga_frequency") < 50.0:
            contested = min(contested, 70.0)
        threept_rows.append({
            "Deep Range Shooting": _median_percentile([pct(row_index, "Avg3ptShotDistance"), pct(row_index, "3fga_frequency"), local(row_index, "3P_Accuracy", "Avg3ptShotDistance")]),
            "Catch and Shooting": _median_percentile([pct(row_index, "catch_shoot_3P_frequency"), local(row_index, "catch_shoot_3P_accuracy", "avg_closest_defender_3FGA")]),
            "Contested 3PT Shot Making": contested,
            "Pull Up 3PT Shooting": _median_percentile([pct(row_index, "pull_up_3P_frequency"), local(row_index, "pull_up_3P_accuracy", "pull_up_3P_frequency")]),
            "3PT Volume": _median_percentile([pct(row_index, "3fga_frequency"), pct(row_index, "traditional_fg3a")]),
            "3PT Accuracy": 0.70 * local(row_index, "3P_Accuracy", "avg_closest_defender_3FGA") + 0.30 * pct(row_index, "3fga_frequency"),
        })
    threept_frame = pd.DataFrame(threept_rows, index=guards.index).fillna(0.0)
    if breakdown_kind == "three_pt_breakdown":
        return threept_frame, {"3PT Shooting Talent": threept_frame}, THREE_PT_BREAKDOWN_GROUP_FEATURES, [], required_features
    component_score_frame = pd.DataFrame(index=guards.index, dtype=float)
    subsection_frames: Dict[str, pd.DataFrame] = {"ThreePT": threept_frame}
    component_score_frame["ThreePT"] = threept_frame.median(axis=1, skipna=True).fillna(0.0)
    mid_rows, rim_rows, play_rows = [], [], []
    for row_index in guards.index:
        mid_rows.append({"Volume Mid-Range Shooting": _median_percentile([pct(row_index, "MidRangeFrequency"), pct(row_index, "by_zone_statistics_mid_range_fga")]), "Mid-Range Efficiency": _median_percentile([pct(row_index, "MidRangeFrequency"), local(row_index, "MidRangeAccuracy", "tight_very_tight_2fga_frequency"), pct(row_index, "by_zone_statistics_mid_range_fga")])})
        rim_rows.append({"Driving Volume": _median_percentile([pct(row_index, "drives_drive_fga"), pct(row_index, "drives_drives"), pct(row_index, "pts_from_drives_per_75")]), "Driving Efficiency": _median_percentile([local(row_index, "drive_fg_pct", "drive_fga_frequency"), pct(row_index, "pts_from_drives_per_75")])})
        play_rows.append({"Passing Volume": _median_percentile([pct(row_index, "assist_frequency"), pct(row_index, "potential_assist_frequency"), pct(row_index, "pts_created_from_assists")]), "Passing Efficiency": _median_percentile([pct(row_index, "pts_created_from_assists"), pct(row_index, "assist_frequency"), pct(row_index, "potential_assist_frequency"), local(row_index, "potential_assist_tov_ratio", "potential_assist_frequency"), local(row_index, "assists_tov_ratio", "assist_frequency"), pct(row_index, "pts_created_to_tov_ratio")]), "Crafted Metrics": _median_percentile([pct(row_index, "crafted_box_creation"), pct(row_index, "crafted_passer_rating")])})
    mid_frame = pd.DataFrame(mid_rows, index=guards.index).fillna(0.0)
    rim_frame = pd.DataFrame(rim_rows, index=guards.index).fillna(0.0)
    play_frame = pd.DataFrame(play_rows, index=guards.index).fillna(0.0)
    defense_frame, defense_feature_groups = _compute_defensive_skill_components(guards, numeric_frame, percentile_frame, peer_mask)
    subsection_frames.update({"MidRange": mid_frame, "RimPressure": rim_frame, "Playmaking": play_frame, "Defense": defense_frame[[DLEBRON_FEATURE]]})
    component_score_frame["MidRange"] = mid_frame.median(axis=1, skipna=True).fillna(0.0)
    component_score_frame["RimPressure"] = rim_frame.median(axis=1, skipna=True).fillna(0.0)
    component_score_frame["Playmaking"] = play_frame.mean(axis=1, skipna=True).fillna(0.0)
    component_score_frame["Defense"] = defense_frame["Defense"].fillna(0.0)
    feature_groups = {
        "ThreePT": ordered_unique([feature for features in THREE_PT_BREAKDOWN_GROUP_FEATURES.values() for feature in features]),
        "MidRange": ["MidRangeFrequency", "MidRangeAccuracy", "by_zone_statistics_mid_range_fga", "tight_very_tight_2fga_frequency"],
        "RimPressure": ["drives_drive_fga", "drives_drives", "pts_from_drives_per_75", "drive_fga_frequency", "drive_fg_pct"],
        "Playmaking": ["assist_frequency", "potential_assist_frequency", "pts_created_from_assists", "potential_assist_tov_ratio", "assists_tov_ratio", "pts_created_to_tov_ratio", "crafted_box_creation", "crafted_passer_rating"],
        "Defense": ordered_unique([feature for features in defense_feature_groups.values() for feature in features]),
    }
    return component_score_frame[SKILL_BREAKDOWN_GROUP_ORDER], subsection_frames, feature_groups, [], required_features



def load_precomputed_badges_by_player_key(guards: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    """Load precomputed badge rows without blocking the initial cluster render.

    Initial scatter/galaxy loading should not recompute the entire badge engine.
    Run scripts/precompute_player_badges.py to refresh backend/data/player_badges.csv.
    """
    badge_path = BACKEND_DATA_DIR / "player_badges.csv"
    empty_badges = {str(player_key): [] for player_key in guards.get("player_key", pd.Series([], dtype=str)).astype(str)}
    if not badge_path.exists():
        return empty_badges

    try:
        badge_frame = pd.read_csv(badge_path)
    except Exception:
        return empty_badges

    required_columns = {"player_key", "badge_id", "badge_name", "badge_tier", "badge_score_percentile"}
    if not required_columns.issubset(set(badge_frame.columns)):
        return empty_badges
    badge_frame = attach_badge_rarity_columns(badge_frame, guards)

    badges_by_player_key: Dict[str, List[Dict[str, object]]] = dict(empty_badges)
    valid_player_keys = set(empty_badges.keys())
    for _, badge_row in badge_frame.iterrows():
        player_key = str(badge_row.get("player_key", ""))
        if player_key not in valid_player_keys:
            continue

        components = {}
        demotion_reasons = []
        component_json = badge_row.get("component_percentiles_json", "")
        demotion_json = badge_row.get("demotion_reasons_json", "")
        try:
            if isinstance(component_json, str) and component_json.strip():
                parsed_components = json.loads(component_json)
                if isinstance(parsed_components, dict):
                    components = parsed_components
        except Exception:
            components = {}
        try:
            if isinstance(demotion_json, str) and demotion_json.strip():
                parsed_reasons = json.loads(demotion_json)
                if isinstance(parsed_reasons, list):
                    demotion_reasons = parsed_reasons
        except Exception:
            demotion_reasons = []

        badge_payload = {
            "id": str(badge_row.get("badge_id", "")),
            "name": str(badge_row.get("badge_name", "")),
            "tier": str(badge_row.get("badge_tier", "")),
            "category": str(badge_row.get("badge_category", "")) if "badge_category" in badge_frame.columns else "",
            "score_percentile": _safe_float(badge_row.get("badge_score_percentile"), 0.0),
            "rarity_percent": _safe_float(badge_row.get("rarity_percent"), 0.0),
            "rarity_label": str(badge_row.get("rarity_label", "")),
            "components": components,
            "demotion_reasons": demotion_reasons,
        }
        badges_by_player_key.setdefault(player_key, []).append(badge_payload)

    # Preserve the same ordering the badge engine uses when category is present; otherwise keep CSV order.
    return badges_by_player_key


PLAYER_COMPS_PERCENTILE_EXCLUDE_COLUMNS = {
    "Player Name",
    "teams_played",
    "Season",
    "position",
    "GP",
    "MP",
    "OffPoss",
    "DefPoss",
    "player_key",
    "_assignment_key",
}

PLAYER_COMPS_FEATURE_LABEL_OVERRIDES = {
    "Player Name": "Player Name",
    "teams_played": "Teams Played",
    "Season": "Season",
    "GP": "Games Played",
    "MP": "Minutes Played",
    "OffPoss": "Offensive Possessions",
    "DefPoss": "Defensive Possessions",
    "position": "Position",

    "two_drib_3PA_frequency": "Two-Dribble 3FGA Frequency",
    "two_drib_3P_Accuracy": "Two-Dribble 3P%",
    "pull_up_3PA": "Pull-Up 3FGA",
    "catch_shoot_3PA": "Catch + Shoot 3FGA",

    "shot_contest_0_2_fg3a": "Very Tight 3FGA",
    "shot_contest_0_2_fg3a_frequency": "Very Tight 3FGA Frequency",
    "shot_contest_0_2_fg3_pct": "Very Tight 3P%",
    "shot_contest_2_4_fg3a": "Tight 3FGA",
    "shot_contest_2_4_fg3a_frequency": "Tight 3FGA Frequency",
    "shot_contest_2_4_fg3_pct": "Tight 3P%",
    "shot_contest_4_6_fg3a": "Semi-Contested 3FGA",
    "shot_contest_4_6_fg3a_frequency": "Semi-Contested 3FGA Frequency",
    "shot_contest_4_6_fg3_pct": "Semi-Contested 3P%",
    "shot_contest_6_plus_fg3a": "Wide Open 3FGA",
    "shot_contest_6_plus_fg3a_frequency": "Wide Open 3FGA Frequency",
    "shot_contest_6_plus_fg3_pct": "Wide Open 3P%",

    "3P_Accuracy": "3P%",
    "traditional_fg3a": "3FGA",
    "3fga_frequency": "3FGA Frequency",
    "Avg3ptShotDistance": "Average 3PT Shot Distance",
    "pull_up_3P_frequency": "Pull-Up 3PT Frequency",
    "pull_up_3P_accuracy": "Pull-Up 3P%",
    "catch_shoot_3P_frequency": "Catch + Shoot 3PT Frequency",
    "catch_shoot_3P_accuracy": "Catch + Shoot 3P%",
    "zero_to_one_drib_3PA_frequency": "0-1 Dribble 3FGA Frequency",
    "zero_to_one_drib_3PA_accuracy": "0-1 Dribble 3P%",
    "three_to_six_drib_3FGA_frequency": "3-6 Dribble 3FGA Frequency",
    "three_to_six_drib_3FGA_accuracy": "3-6 Dribble 3P%",
    "seven_plus_drib_3FGA_frequency": "7+ Dribble 3FGA Frequency",
    "seven_plus_drib_3FGA_accuracy": "7+ Dribble 3P%",
    "pct_3p_fg_assisted": "Pct of 3PM Assisted",
    "pct_3fga_wide_open": "Pct of 3FGA Wide Open",
    "pct_fga_3FGA": "Pct of FGA from Three",

    "pull_up_2PA": "Pull-Up 2FGA",
    "shot_contest_0_2_fg2a": "Very Tight 10ft+ 2FGA",
    "shot_contest_0_2_fg2a_frequency": "Very Tight 10ft+ 2FGA Frequency",
    "shot_contest_0_2_fg2_pct": "Very Tight 10ft+ 2P%",
    "shot_contest_2_4_fg2a": "Tight 10ft+ 2FGA",
    "shot_contest_2_4_fg2a_frequency": "Tight 10ft+ 2FGA Frequency",
    "shot_contest_2_4_fg2_pct": "Tight 10ft+ 2P%",
    "shot_contest_4_6_fg2a": "Semi-Contested 10ft+ 2FGA",
    "shot_contest_4_6_fg2a_frequency": "Semi-Contested 10ft+ 2FGA Frequency",
    "shot_contest_4_6_fg2_pct": "Semi-Contested 10ft+ 2P%",
    "shot_contest_6_plus_fg2a": "Wide Open 10ft+ 2FGA",
    "shot_contest_6_plus_fg2a_frequency": "Wide Open 10ft+ 2FGA Frequency",
    "shot_contest_6_plus_fg2_pct": "Wide Open 10ft+ 2P%",

    "by_zone_statistics_mid_range_fga": "Mid-Range FGA",
    "Avg2ptShotDistance": "Average 2PT Shot Distance",
    "pct_fga_MR": "Pct of FGA from Mid-Range",
    "MidRangeFrequency": "Mid-Range Frequency",
    "MidRangeAccuracy": "Mid-Range FG%",
    "pts_from_midrange_per_75": "Points from Mid-Range per 75",

    "drives_drives": "Drives",
    "drives_drive_fga": "Drive FGA",
    "drive_FTAr": "Drive Free Throw Rate",
    "pct_drives_results_in_FGA": "Pct of Drives Ending in FGA",
    "drive_tov_rate": "Drive Turnover Rate",
    "drive_frequency": "Drive Frequency",
    "by_zone_statistics_restricted_area_fga": "Restricted Area FGA",
    "by_zone_statistics_in_the_paint_non_ra_fga": "Paint Non-RA FGA",
    "RestrictedArea_Frequency": "Restricted Area Frequency",
    "RestrictedArea_Accuracy": "Restricted Area FG%",
    "Paint_Non_RA_Frequency": "Paint Non-RA Frequency",
    "Paint_Non_RA_Accuracy": "Paint Non-RA FG%",
    "drive_fga_frequency": "Drive FGA Frequency",
    "drive_fg_pct": "Drive FG%",

    "traditional_ast": "Assists",
    "passing_potential_ast": "Potential Assists",
    "passing_ast_points_created": "Assist Points Created",
    "passing_passes_made": "Passes Made",
    "passing_secondary_ast": "Secondary Assists",
    "passing_ft_ast": "Free Throw Assists",
    "pass_frequency": "Pass Frequency",
    "secondary_ast_frequency": "Secondary Assist Frequency",
    "passing_ft_ast_frequency": "Free Throw Assist Frequency",
    "drive_ast_frequency": "Drive Assist Frequency",
    "drive_ast_tov_ratio": "Drive Assist-to-Turnover Ratio",
    "hustle_screen_ast_pts": "Screen Assist Points",
    "hustle_screen_ast_pts_frequency": "Screen Assist Points Frequency",

    "defense_dash_overall_d_fga": "Opponent FGA Defended",
    "defense_dash_overall_d_fg_pct": "Opponent FG% Defended",
    "hustle_contested_shots": "Contested Shots",
    "hustle_deflections": "Deflections",
    "hustle_charges_drawn": "Charges Drawn",
    "offensive_fouls_drawn": "Offensive Fouls Drawn",
    "charge_drawn_frequency": "Charge Drawn Frequency",
    "deflection_frequency": "Deflection Frequency",
    "opp_players_fg_pct_difference": "Opponent FG% Difference",
    "off_fouls_drawn_frequency": "Offensive Fouls Drawn Frequency",
    "contested_shot_frequency": "Contested Shot Frequency",
    "crafted_cdpm": "Crafted CDPM",
    "DEF_PTS_PER_100_ON_OFF": "Defensive Points per 100 On/Off",
    "DEF_THREE_PT_FG_PCT_ON_OFF": "Defensive 3PT FG% On/Off",

    "traditional_pts": "Points",
    "traditional_reb": "Rebounds",
    "traditional_tov": "Turnovers",
    "traditional_stl": "Steals",
    "traditional_blk": "Blocks",
    "steal_frequency": "Steal Frequency",
    "block_frequency": "Block Frequency",
    "traditional_fg_pct": "FG%",
    "traditional_ft_pct": "FT%",

    "O-LEBRON": "O-LEBRON",
    "D-LEBRON": "D-LEBRON",
    "LEBRON": "LEBRON",
    "WAR": "WAR",

    "assist_frequency": "Assist Frequency",
    "potential_assist_frequency": "Potential Assist Frequency",
    "assists_tov_ratio": "Assist-to-Turnover Ratio",
    "potential_assist_tov_ratio": "Potential Assist-to-Turnover Ratio",
    "pass_tov_ratio": "Pass-to-Turnover Ratio",
    "crafted_passer_rating": "Crafted Passer Rating",
    "crafted_box_creation": "Crafted Box Creation",
    "drib_tov_ratio": "Dribble Turnover Ratio",
    "pts_created_from_assists": "Points Created from Assists",
    "THREE_PT_FG_PCT_ON_OFF": "3PT FG% On/Off",
    "PTS_PER_100_ON_OFF": "Points per 100 On/Off",
    "drives_drive_ast": "Drive Assists",
    "pct_passes_assists": "Pct Passes are Assists",
    "shot_contest_2_4_fg3a_frequency": "Tight 3FGA Frequency",
}


def humanize_player_comps_feature_label(feature_name: str) -> str:
    if feature_name in PLAYER_COMPS_FEATURE_LABEL_OVERRIDES:
        return PLAYER_COMPS_FEATURE_LABEL_OVERRIDES[feature_name]
    text = str(feature_name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").replace("/", " /")
    replacements = [
        ("fg3a", "3PA"),
        ("fg2a", "2PA"),
        ("fg3 pct", "3PT FG%"),
        ("fg2 pct", "2PT FG%"),
        ("3p accuracy", "3PT accuracy"),
        ("3pa", "3PA"),
        ("2pa", "2PA"),
        ("fga", "FGA"),
        ("fg pct", "FG%"),
        ("pct", "%"),
        ("ast", "assist"),
        ("tov", "turnover"),
        ("fta", "FTA"),
        ("ft", "free throw"),
        ("ra", "RA"),
        ("non ra", "non-RA"),
        ("on off", "on/off"),
        ("per 100", "per 100"),
        ("per 75", "per 75"),
    ]
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    for old_value, new_value in replacements:
        normalized = normalized.replace(old_value, new_value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    acronyms = {"3pt", "2pt", "fga", "fg%", "fta", "ra", "war", "lebron"}
    words = []
    for word in normalized.split(" "):
        compact = word.lower().strip()
        if compact in acronyms or any(ch.isdigit() for ch in compact) or "%" in compact or "/" in compact:
            words.append(word.upper() if compact in {"fga", "fta", "war", "lebron", "ra"} else word)
        elif word in {"on/off", "per"}:
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words).replace("3pt", "3PT").replace("2pt", "2PT")


def load_player_comps_percentile_source() -> Optional[Dict[str, object]]:
    path = Path(PLAYER_COMPS_DATASET_PATH).expanduser()
    if not path.exists():
        return None
    mtime_ns = path.stat().st_mtime_ns
    cached_payload = _PLAYER_COMPS_PERCENTILE_CACHE.get("payload")
    if (
        cached_payload is not None
        and _PLAYER_COMPS_PERCENTILE_CACHE.get("path") == str(path)
        and _PLAYER_COMPS_PERCENTILE_CACHE.get("mtime_ns") == mtime_ns
    ):
        return cached_payload  # type: ignore[return-value]

    source = pd.read_csv(path)
    required_columns = {"Player Name", "Season", "position"}
    if not required_columns.issubset(set(source.columns)):
        return None

    eligible = source[source["position"].isin(["PG", "SG"])].copy()
    excluded_names = {normalize_player_name_for_percentile_pool(name) for name in ["LeBron James", "Scottie Barnes", "Ben Simmons"]}
    eligible = eligible[~eligible["Player Name"].map(normalize_player_name_for_percentile_pool).isin(excluded_names)].copy()
    if eligible.empty:
        return None

    if "teams_played" not in eligible.columns:
        eligible["teams_played"] = ""
    eligible["player_key"] = eligible.apply(stable_player_key, axis=1)
    eligible["_assignment_key"] = eligible.apply(lambda row: build_assignment_key(row["Player Name"], row["Season"]), axis=1)

    numeric_columns = []
    for column in eligible.columns:
        if column in PLAYER_COMPS_PERCENTILE_EXCLUDE_COLUMNS:
            continue
        numeric_series = pd.to_numeric(eligible[column], errors="coerce")
        if numeric_series.notna().sum() < 2:
            continue
        numeric_columns.append(str(column))
    numeric_columns = ordered_unique(numeric_columns)
    if not numeric_columns:
        return None

    numeric_all = eligible[numeric_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    percentile_df = pd.DataFrame(index=eligible.index, columns=numeric_columns, dtype=float)
    for _, idx in eligible.groupby("Season").groups.items():
        season_block = numeric_all.loc[idx]
        for feature_name in numeric_columns:
            lower_is_better = feature_name in PLAYER_COMPS_LOWER_IS_BETTER_FEATURES
            ranked = season_block[feature_name].rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
            percentile_df.loc[idx, feature_name] = ranked
    percentile_df = percentile_df.fillna(0.0)

    payload = {
        "path": str(path),
        "mtime_ns": mtime_ns,
        "eligible": eligible,
        "numeric_all": numeric_all,
        "percentile_df": percentile_df,
        "features": numeric_columns,
        "by_player_key": {str(row["player_key"]): idx for idx, row in eligible.iterrows()},
        "by_assignment_key": {str(row["_assignment_key"]): idx for idx, row in eligible.iterrows()},
    }
    _PLAYER_COMPS_PERCENTILE_CACHE["path"] = str(path)
    _PLAYER_COMPS_PERCENTILE_CACHE["mtime_ns"] = mtime_ns
    _PLAYER_COMPS_PERCENTILE_CACHE["payload"] = payload
    return payload


def build_player_comps_feature_percentile_items(player_key: str, player_meta: Dict[str, object]) -> Optional[List[Dict[str, object]]]:
    source = load_player_comps_percentile_source()
    if not source:
        return None
    by_player_key = source.get("by_player_key", {})
    by_assignment_key = source.get("by_assignment_key", {})
    row_index = by_player_key.get(str(player_key)) if isinstance(by_player_key, dict) else None
    if row_index is None and isinstance(by_assignment_key, dict):
        assignment_key = build_assignment_key(player_meta.get("player_name", ""), player_meta.get("season", ""))
        row_index = by_assignment_key.get(assignment_key)
    if row_index is None:
        return None

    numeric_all: pd.DataFrame = source["numeric_all"]  # type: ignore[assignment]
    percentile_df: pd.DataFrame = source["percentile_df"]  # type: ignore[assignment]
    features: List[str] = source["features"]  # type: ignore[assignment]
    items: List[Dict[str, object]] = []
    for feature_name in features:
        value = numeric_all.at[row_index, feature_name]
        percentile = percentile_df.at[row_index, feature_name]
        if pd.isna(value) or pd.isna(percentile):
            continue
        items.append({
            "feature": feature_name,
            "label": humanize_player_comps_feature_label(feature_name),
            "value": float(value),
            "percentile": float(percentile),
            "lower_is_better": feature_name in PLAYER_COMPS_LOWER_IS_BETTER_FEATURES,
        })
    return items


def attach_badge_rarity_columns(badge_frame: pd.DataFrame, guards: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Attach cumulative same-season guard rarity labels to badge rows.

    Rarity is cumulative by tier: a gold badge counts as gold, silver, and bronze
    for the same badge skill. Therefore bronze rarity is the most common, silver
    means silver-or-better, gold means gold-or-better, and diamond means diamond.
    """
    if badge_frame.empty or not {"Season", "badge_id", "badge_tier"}.issubset(set(badge_frame.columns)):
        return badge_frame

    output = badge_frame.copy()
    excluded_names = {normalize_player_name_for_percentile_pool(name) for name in ["LeBron James", "Scottie Barnes", "Ben Simmons"]}

    if guards is not None and not guards.empty and {"Season", "Player Name", "position"}.issubset(set(guards.columns)):
        eligible = guards[guards["position"].isin(["PG", "SG"])].copy()
        eligible = eligible[~eligible["Player Name"].map(normalize_player_name_for_percentile_pool).isin(excluded_names)]
        if "player_key" not in eligible.columns:
            try:
                eligible["player_key"] = eligible.apply(stable_player_key, axis=1)
            except Exception:
                eligible["player_key"] = eligible.index.astype(str)
        denominators = eligible.drop_duplicates(["Season", "player_key"]).groupby("Season")["player_key"].size().to_dict()
    elif {"Season", "player_key"}.issubset(set(output.columns)):
        denominators = output.drop_duplicates(["Season", "player_key"]).groupby("Season")["player_key"].size().to_dict()
    else:
        denominators = output.groupby("Season").size().to_dict()

    tier_rank = {"diamond": 0, "gold": 1, "silver": 2, "bronze": 3}
    badge_players_by_season_skill_tier: Dict[tuple, set] = {}
    for _, badge_row in output.iterrows():
        season = badge_row.get("Season")
        badge_id = badge_row.get("badge_id")
        tier = str(badge_row.get("badge_tier", "")).strip().lower()
        rank = tier_rank.get(tier)
        if rank is None:
            continue
        player_identifier = str(badge_row.get("player_key") or badge_row.get("Player Name") or badge_row.name)
        for candidate_tier, candidate_rank in tier_rank.items():
            if rank <= candidate_rank:
                badge_players_by_season_skill_tier.setdefault((season, badge_id, candidate_tier), set()).add(player_identifier)

    rarity_values = []
    rarity_labels = []
    for _, row in output.iterrows():
        season_value = row.get("Season")
        season = str(season_value or "")
        badge_id = row.get("badge_id")
        tier = str(row.get("badge_tier", "")).strip().lower()
        denominator = float(denominators.get(season_value, denominators.get(season, 0)) or 0)
        cumulative_count = float(len(badge_players_by_season_skill_tier.get((season_value, badge_id, tier), set())))
        rarity_percent = (cumulative_count / denominator * 100.0) if denominator > 0 else 0.0
        rarity_values.append(rarity_percent)
        rarity_labels.append(f"{rarity_percent:.2f}% of {season} guards" if season else f"{rarity_percent:.2f}% of guards")

    output["rarity_percent"] = rarity_values
    output["rarity_label"] = rarity_labels
    return output


def load_base_dataframe(dataset_path: str) -> Dict:
    dataset_path = str(Path(dataset_path).expanduser())
    cache_key = dataset_path

    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "Set CLUSTER_DATASET_PATH or edit DEFAULT_DATASET_PATH in backend/app.py."
        )

    df = pd.read_csv(path)

    missing_required = [c for c in META_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    missing_features = [c for c in ALLOWED_FEATURES if c not in df.columns]
    if missing_features:
        # Badge/skill-breakdown features are optional for the initial scatter/galaxy render.
        # Fill absent optional columns instead of killing /api/cluster with a 500.
        df = pd.concat(
            [df, pd.DataFrame({missing_feature: np.nan for missing_feature in missing_features}, index=df.index)],
            axis=1,
        )

    remove_df = pd.DataFrame(
        ROWS_TO_REMOVE,
        columns=["Player Name", "Season", "teams_played", "position"],
    )

    df = df.merge(
        remove_df.assign(_remove_flag=1),
        on=["Player Name", "Season", "teams_played", "position"],
        how="left",
    )
    df = df[df["_remove_flag"] != 1].drop(columns=["_remove_flag"]).copy()
    df = add_locked_similarity_derived_features(df)

    guards = df[df["position"].isin(["PG", "SG"])].copy()
    if guards.empty:
        raise ValueError("No guards found with position equal to PG or SG.")

    guards["player_key"] = guards.apply(stable_player_key, axis=1)

    numeric_all = guards[ALLOWED_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric_all = numeric_all.replace([np.inf, -np.inf], np.nan)

    percentile_df = compute_season_percentiles(numeric_all, guards["Season"])
    badges_by_player_key = load_precomputed_badges_by_player_key(guards)

    stats_lookup = {}
    for i, row in guards.iterrows():
        headshot_payload = get_player_headshot_payload(row["Player Name"])
        stats_lookup[row["player_key"]] = {
            "meta": {
                "player_name": row["Player Name"],
                "season": row["Season"],
                "teams_played": row["teams_played"],
                "position": row["position"],
                **headshot_payload,
            },
            "stats": {
                feature: float(numeric_all.loc[i, feature]) if pd.notna(numeric_all.loc[i, feature]) else 0.0
                for feature in ALLOWED_FEATURES
            },
            "percentiles": {
                feature: float(percentile_df.loc[i, feature]) if pd.notna(percentile_df.loc[i, feature]) else 0.0
                for feature in ALLOWED_FEATURES
            },
            "badges": badges_by_player_key.get(row["player_key"], []),
        }

    payload = {
        "dataset_path": dataset_path,
        "dataset_mtime_ns": path.stat().st_mtime_ns,
        "guards": guards.reset_index(drop=True),
        "stats_lookup": stats_lookup,
        "badges_by_player_key": badges_by_player_key,
    }
    _DATASET_CACHE[cache_key] = payload
    return payload


def seasonwise_guard_standardize(feature_df: pd.DataFrame, seasons: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=feature_df.index, columns=feature_df.columns, dtype=float)

    global_medians = feature_df.median(numeric_only=True).fillna(0.0)

    for season in seasons.unique():
        idx = seasons[seasons == season].index
        block = feature_df.loc[idx].copy()
        block = block.fillna(block.median(numeric_only=True))
        block = block.fillna(global_medians)

        means = block.mean(axis=0, skipna=True)
        stds = block.std(axis=0, skipna=True, ddof=0).replace(0, np.nan)

        z = (block - means) / stds
        z = z.fillna(0.0)
        out.loc[idx] = z

    return out




def load_locked_euclidean_assignments() -> pd.DataFrame:
    if not EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Locked Euclidean KMeans assignments file not found: {EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH}"
        )

    assignments = pd.read_csv(EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH)
    required_columns = {"Player Name", "Season", "cluster_raw", "cluster", "cluster_label"}
    missing_columns = sorted(required_columns - set(assignments.columns))
    if missing_columns:
        raise ValueError(
            f"Locked Euclidean KMeans assignments file is missing columns: {missing_columns}"
        )

    assignments = assignments.copy()
    assignments["assignment_key"] = [
        build_assignment_key(player_name, season)
        for player_name, season in zip(assignments["Player Name"], assignments["Season"])
    ]

    duplicated_keys = assignments.loc[assignments["assignment_key"].duplicated(), "assignment_key"].unique().tolist()
    if duplicated_keys:
        raise ValueError(
            f"Locked Euclidean KMeans assignments contain duplicate player-season keys: {duplicated_keys[:10]}"
        )

    assignments["cluster_raw"] = assignments["cluster_raw"].astype(int)
    assignments["cluster"] = assignments["cluster"].astype(int)
    return assignments


def season_median_impute_for_features(
    dataframe: pd.DataFrame,
    feature_list: List[str],
    season_column_name: str,
) -> pd.DataFrame:
    imputed_dataframe = dataframe.copy()
    for feature_name in feature_list:
        season_median_series = imputed_dataframe.groupby(season_column_name, sort=False)[feature_name].transform("median")
        global_median_value = imputed_dataframe[feature_name].median()
        if pd.isna(global_median_value):
            global_median_value = 0.0
        imputed_dataframe[feature_name] = imputed_dataframe[feature_name].fillna(season_median_series)
        imputed_dataframe[feature_name] = imputed_dataframe[feature_name].fillna(float(global_median_value))
    return imputed_dataframe


def season_standardize_clip_for_locked_euclidean(
    dataframe: pd.DataFrame,
    feature_list: List[str],
    season_column_name: str,
    clip_zscore_value: float,
) -> pd.DataFrame:
    standardized_frame = pd.DataFrame(index=dataframe.index)

    for feature_name in feature_list:
        season_mean_series = dataframe.groupby(season_column_name, sort=False)[feature_name].transform("mean")
        season_std_series = dataframe.groupby(season_column_name, sort=False)[feature_name].transform("std")

        safe_std_series = season_std_series.replace(0.0, np.nan)
        standardized_series = (dataframe[feature_name] - season_mean_series) / safe_std_series
        standardized_series = standardized_series.replace([np.inf, -np.inf], np.nan)
        standardized_series = standardized_series.fillna(0.0)
        standardized_series = standardized_series.clip(-clip_zscore_value, clip_zscore_value)

        standardized_frame[feature_name] = standardized_series.astype(float)

    return standardized_frame



def build_locked_equal_block_weighted_raw_space(
    standardized_feature_frame: pd.DataFrame,
) -> tuple[np.ndarray, List[Dict[str, object]], Dict[str, Dict[str, int]]]:
    transformed_group_matrices: List[np.ndarray] = []
    block_summary_rows: List[Dict[str, object]] = []
    block_slice_map: Dict[str, Dict[str, int]] = {}
    start_component_index = 0

    for group_name in EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER:
        group_feature_list = EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES[group_name]
        group_weight = float(EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS[group_name])
        group_input_matrix = standardized_feature_frame[group_feature_list].to_numpy(dtype=float)
        feature_count = int(len(group_feature_list))
        per_coordinate_group_scale = float(np.sqrt(group_weight / feature_count))
        weighted_group_matrix = group_input_matrix * per_coordinate_group_scale

        transformed_group_matrices.append(weighted_group_matrix)

        end_component_index = start_component_index + feature_count
        block_slice_map[group_name] = {
            "start_component_index": int(start_component_index),
            "end_component_index": int(end_component_index),
            "feature_count": int(feature_count),
            "features": list(group_feature_list),
        }
        block_summary_rows.append(
            {
                "group": group_name,
                "feature_count": int(feature_count),
                "group_weight": float(group_weight),
                "per_coordinate_group_scale": float(per_coordinate_group_scale),
                "start_component_index": int(start_component_index),
                "end_component_index": int(end_component_index),
                "features": list(group_feature_list),
            }
        )
        start_component_index = end_component_index

    final_clustering_matrix = np.hstack(transformed_group_matrices)
    return final_clustering_matrix, block_summary_rows, block_slice_map

def build_locked_euclidean_kmeans_space(base_guards: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    base_guards = add_locked_similarity_derived_features(base_guards)
    locked_feature_columns = get_locked_euclidean_kmeans_feature_columns(raw=False)
    missing_feature_columns = [column_name for column_name in locked_feature_columns if column_name not in base_guards.columns]
    if missing_feature_columns:
        raise ValueError(
            "These required locked similarity columns are missing from the dataset: "
            f"{missing_feature_columns}"
        )

    assignments = load_locked_euclidean_assignments()

    locked_guards = base_guards.copy().reset_index(drop=True)
    locked_guards["assignment_key"] = [
        build_assignment_key(player_name, season)
        for player_name, season in zip(locked_guards["Player Name"], locked_guards["Season"])
    ]

    excluded_key_set = {normalize_player_name_for_assignment_key(name) for name in EUCLIDEAN_KMEANS_LOCKED_EXCLUDED_NAMES}
    locked_guards = locked_guards.loc[
        ~locked_guards["Player Name"].map(normalize_player_name_for_assignment_key).isin(excluded_key_set)
    ].copy()

    assignment_lookup = assignments[["assignment_key", "cluster_raw", "cluster", "cluster_label"]].copy()
    locked_guards = locked_guards.merge(assignment_lookup, on="assignment_key", how="left", validate="one_to_one")

    unmatched_guard_count = int(locked_guards["cluster_raw"].isna().sum())
    if unmatched_guard_count > 0:
        # Static locked mode intentionally uses the provided cluster list as the source of truth.
        # Player-seasons not in that list are dropped instead of being newly clustered live.
        locked_guards = locked_guards.loc[locked_guards["cluster_raw"].notna()].copy()

    if locked_guards.empty:
        raise ValueError(
            "No guard rows matched the locked Euclidean KMeans assignment list. "
            "Check player names, seasons, and the assignment file."
        )

    locked_guards = locked_guards.reset_index(drop=True)
    raw_labels = locked_guards["cluster_raw"].astype(int).to_numpy()

    numeric_feature_frame = locked_guards[locked_feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric_feature_frame = numeric_feature_frame.replace([np.inf, -np.inf], np.nan)
    numeric_feature_frame["Season"] = locked_guards["Season"].values

    imputed_feature_frame = season_median_impute_for_features(
        dataframe=numeric_feature_frame,
        feature_list=locked_feature_columns,
        season_column_name="Season",
    )

    standardized_feature_frame = season_standardize_clip_for_locked_euclidean(
        dataframe=imputed_feature_frame,
        feature_list=locked_feature_columns,
        season_column_name="Season",
        clip_zscore_value=EUCLIDEAN_KMEANS_LOCKED_CLIP_ZSCORE,
    )

    final_clustering_matrix, block_summary_rows, block_slice_map = build_locked_equal_block_weighted_raw_space(
        standardized_feature_frame=standardized_feature_frame,
    )

    matched_assignment_count = int(locked_guards.shape[0])
    assignment_key_set = set(locked_guards["assignment_key"].tolist())
    missing_dataset_assignment_count = int((~assignments["assignment_key"].isin(assignment_key_set)).sum())

    locked_guards = locked_guards.drop(columns=["assignment_key"])

    metric_meta = {
        "distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
        "similarity_distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
        "space_transform": EUCLIDEAN_KMEANS_LOCKED_SPACE_TRANSFORM,
        "pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
        "euclidean_kmeans_locked_mode": True,
        "euclidean_kmeans_locked_k": EUCLIDEAN_KMEANS_LOCKED_K,
        "euclidean_kmeans_locked_assignment_source": str(EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH),
        "euclidean_kmeans_locked_features": locked_feature_columns,
        "euclidean_kmeans_locked_feature_count": len(locked_feature_columns),
        "euclidean_kmeans_locked_group_features": EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES,
        "euclidean_kmeans_locked_group_weights": EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS,
        "euclidean_kmeans_locked_group_order": EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER,
        "euclidean_kmeans_locked_feature_signature": build_locked_euclidean_feature_signature(),
        "euclidean_kmeans_locked_clip_zscore": EUCLIDEAN_KMEANS_LOCKED_CLIP_ZSCORE,
        "euclidean_kmeans_locked_block_summary": block_summary_rows,
        "euclidean_kmeans_locked_block_slices": block_slice_map,
        "euclidean_kmeans_locked_excluded_names": sorted(EUCLIDEAN_KMEANS_LOCKED_EXCLUDED_NAMES),
        "euclidean_kmeans_locked_luka_euclidean_override_names": ["Luka Doncic"],
        "euclidean_kmeans_locked_luka_euclidean_override_metric": LUKA_DONCIC_SIMILARITY_DISTANCE_METRIC,
        "euclidean_kmeans_unmatched_guard_rows_dropped": unmatched_guard_count,
        "euclidean_kmeans_matched_assignment_count": matched_assignment_count,
        "euclidean_kmeans_assignment_rows_not_in_dataset": missing_dataset_assignment_count,
    }

    return locked_guards, final_clustering_matrix, raw_labels, metric_meta

def build_cache_key(
    dataset_meta: Dict,
    algorithm: str,
    distance_metric: str,
    k: int,
    features: List[str],
) -> str:
    feature_key = sorted(features)
    if is_locked_euclidean_kmeans_request(algorithm, distance_metric):
        k = EUCLIDEAN_KMEANS_LOCKED_K
        feature_key = ["__LOCKED_EUCLIDEAN_KMEANS_PRESET__"]

    key_obj = {
        "version": APP_VERSION,
        "dataset_path": dataset_meta["dataset_path"],
        "dataset_mtime_ns": dataset_meta["dataset_mtime_ns"],
        "algorithm": algorithm,
        "distance_metric": distance_metric,
        "k": k,
        "features": feature_key,
        "kmeans_n_init": KMEANS_N_INIT,
        "kmeans_random_state": KMEANS_RANDOM_STATE,
        "kmeans_max_iter": KMEANS_MAX_ITER,
        "kmeans_tol": KMEANS_TOL,
        "pca_target": PCA_EXPLAINED_VAR_TARGET,
        "euclidean_kmeans_locked_k": EUCLIDEAN_KMEANS_LOCKED_K,
        "euclidean_kmeans_locked_assignments_path": str(EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH),
        "euclidean_kmeans_locked_group_features": EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES,
        "euclidean_kmeans_locked_group_weights": EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS,
        "euclidean_kmeans_locked_group_order": EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER,
        "euclidean_kmeans_locked_pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
        "euclidean_kmeans_locked_similarity_distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
        "euclidean_kmeans_locked_luka_override_metric": LUKA_DONCIC_SIMILARITY_DISTANCE_METRIC,
        "galaxy_similar_player_count": GALAXY_SIMILAR_PLAYER_COUNT,
        "galaxy_cluster_knn_count": GALAXY_CLUSTER_KNN_COUNT,
        "galaxy_umap_n_neighbors": GALAXY_UMAP_N_NEIGHBORS,
        "galaxy_umap_min_dist": GALAXY_UMAP_MIN_DIST,
        "galaxy_random_state": GALAXY_RANDOM_STATE,
        "euclidean_kmeans_locked_features": get_locked_euclidean_kmeans_feature_columns(raw=True),
        "euclidean_kmeans_locked_clip_zscore": EUCLIDEAN_KMEANS_LOCKED_CLIP_ZSCORE,
        "euclidean_kmeans_cluster_name_by_number": EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER,
        "euclidean_kmeans_cluster_description_by_number": EUCLIDEAN_KMEANS_CLUSTER_DESCRIPTION_BY_NUMBER,
        "cluster_name_by_number": CLUSTER_NAME_BY_NUMBER,
        "cosine_eps": COSINE_EPS,
        "removed_rows": ROWS_TO_REMOVE,
        "lower_is_better_percentile_features": sorted(LOWER_IS_BETTER_PERCENTILE_FEATURES),
        "player_comparison_lower_is_better_features": sorted(PLAYER_COMPS_LOWER_IS_BETTER_FEATURES),
        "player_comparison_modes": PLAYER_COMPARISON_MODES,
        "player_comparison_categories": PLAYER_COMPARISON_CATEGORIES,
        "skill_breakdown_group_order": SKILL_BREAKDOWN_GROUP_ORDER,
        "skill_breakdown_group_features": SKILL_BREAKDOWN_GROUP_FEATURES,
        "skill_breakdown_excluded_features": sorted(SKILL_BREAKDOWN_EXCLUDED_FEATURES),
        "three_pt_breakdown_group_order": THREE_PT_BREAKDOWN_GROUP_ORDER,
        "three_pt_breakdown_group_features": THREE_PT_BREAKDOWN_GROUP_FEATURES,
        "three_pt_breakdown_lower_is_better_by_group": {
            group_name: sorted(list(feature_names))
            for group_name, feature_names in THREE_PT_BREAKDOWN_LOWER_IS_BETTER_BY_GROUP.items()
        },
    }
    raw = json.dumps(key_obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_file_for(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl"


def load_cluster_cache(key: str) -> Optional[Dict]:
    path = cache_file_for(key)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cluster_cache(key: str, payload: Dict) -> None:
    path = cache_file_for(key)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


def normalize_rows(X: np.ndarray, eps: float = COSINE_EPS) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def prepare_metric_space(X_pca: np.ndarray, pca: PCA, distance_metric: str) -> tuple[np.ndarray, Dict]:
    if distance_metric == "euclidean":
        return X_pca.copy(), {
            "distance_metric": "euclidean",
            "space_transform": "identity",
        }

    if distance_metric == "cosine":
        X_cos = normalize_rows(X_pca, eps=COSINE_EPS)
        return X_cos, {
            "distance_metric": "cosine",
            "space_transform": "row_l2_normalized",
            "cosine_eps": COSINE_EPS,
        }

    raise ValueError(f"Unsupported distance metric: {distance_metric}")


def recompute_centers(
    points_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    num_features = points_matrix.shape[1]
    centers_matrix = np.zeros((num_clusters, num_features), dtype=float)

    for cluster_index in range(num_clusters):
        cluster_mask = cluster_labels == cluster_index
        if not np.any(cluster_mask):
            raise ValueError(f"Assignment produced empty cluster {cluster_index}.")
        centers_matrix[cluster_index] = points_matrix[cluster_mask].mean(axis=0)

    return centers_matrix


def get_cluster_medoid_local_index(cluster_points_matrix: np.ndarray) -> int:
    pairwise_difference_tensor = (
        cluster_points_matrix[:, None, :] - cluster_points_matrix[None, :, :]
    )
    pairwise_distance_matrix_local = np.linalg.norm(pairwise_difference_tensor, axis=2)
    summed_distance_vector = pairwise_distance_matrix_local.sum(axis=1)
    medoid_local_index = int(np.argmin(summed_distance_vector))
    return medoid_local_index


def spherical_kmeans_plus_plus_init(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = X.shape[0]
    first_idx = int(rng.integers(0, n_samples))
    center_indices = [first_idx]
    centers = [X[first_idx].copy()]

    for _ in range(1, n_clusters):
        similarities = np.clip(X @ np.stack(centers, axis=0).T, -1.0, 1.0)
        if similarities.ndim == 1:
            similarities = similarities[:, None]
        cosine_distances = 1.0 - similarities.max(axis=1)
        cosine_distances = np.maximum(cosine_distances, 0.0)
        cosine_distances[center_indices] = 0.0

        weights = cosine_distances ** 2
        weight_sum = float(weights.sum())
        if weight_sum <= COSINE_EPS:
            remaining = np.setdiff1d(np.arange(n_samples), np.array(center_indices), assume_unique=False)
            if remaining.size == 0:
                break
            next_idx = int(rng.choice(remaining))
        else:
            probabilities = weights / weight_sum
            next_idx = int(rng.choice(n_samples, p=probabilities))

        center_indices.append(next_idx)
        centers.append(X[next_idx].copy())

    while len(centers) < n_clusters:
        next_idx = int(rng.integers(0, n_samples))
        centers.append(X[next_idx].copy())

    return normalize_rows(np.stack(centers, axis=0), eps=COSINE_EPS)


def spherical_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    tol: float,
    random_state: int,
):
    X_norm = normalize_rows(X, eps=COSINE_EPS)
    rng = np.random.default_rng(random_state)

    best_labels = None
    best_centers = None
    best_similarity_sum = -np.inf
    best_iteration_count = 0

    for _ in range(n_init):
        seed = int(rng.integers(0, 2**31 - 1))
        init_rng = np.random.default_rng(seed)
        centers = spherical_kmeans_plus_plus_init(X_norm, n_clusters=n_clusters, rng=init_rng)
        previous_labels = None
        iteration_count = 0

        for iteration in range(max_iter):
            similarities = np.clip(X_norm @ centers.T, -1.0, 1.0)
            labels = similarities.argmax(axis=1)
            max_similarities = similarities[np.arange(X_norm.shape[0]), labels]

            new_centers = np.zeros_like(centers)
            cluster_sums = np.bincount(labels, minlength=n_clusters)
            empty_clusters = []

            for cluster_idx in range(n_clusters):
                mask = labels == cluster_idx
                if mask.any():
                    new_centers[cluster_idx] = X_norm[mask].sum(axis=0)
                else:
                    empty_clusters.append(cluster_idx)

            if empty_clusters:
                order = np.argsort(max_similarities)
                replacement_cursor = 0
                for cluster_idx in empty_clusters:
                    while replacement_cursor < order.size:
                        candidate_idx = int(order[replacement_cursor])
                        replacement_cursor += 1
                        candidate_cluster = int(labels[candidate_idx])
                        if cluster_sums[candidate_cluster] > 1:
                            cluster_sums[candidate_cluster] -= 1
                            labels[candidate_idx] = cluster_idx
                            cluster_sums[cluster_idx] += 1
                            new_centers[cluster_idx] = X_norm[candidate_idx]
                            break
                    if not np.any(new_centers[cluster_idx]):
                        fallback_idx = int(init_rng.integers(0, X_norm.shape[0]))
                        new_centers[cluster_idx] = X_norm[fallback_idx]

            new_centers = normalize_rows(new_centers, eps=COSINE_EPS)
            iteration_count = iteration + 1

            if previous_labels is not None and np.array_equal(labels, previous_labels):
                centers = new_centers
                break

            center_shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers
            previous_labels = labels.copy()
            if center_shift < tol:
                break

        final_similarities = np.clip(X_norm @ centers.T, -1.0, 1.0)
        final_labels = final_similarities.argmax(axis=1)
        similarity_sum = float(final_similarities[np.arange(X_norm.shape[0]), final_labels].sum())

        if similarity_sum > best_similarity_sum:
            best_similarity_sum = similarity_sum
            best_labels = final_labels.copy()
            best_centers = centers.copy()
            best_iteration_count = iteration_count

    return best_centers, best_labels, best_iteration_count


def median_absolute_deviation(series: pd.Series | np.ndarray) -> float:
    numeric_values = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        return 0.0
    median_value = float(np.median(numeric_values))
    return float(np.median(np.abs(numeric_values - median_value)))


def pairwise_distance_matrix(X: np.ndarray, distance_metric: str) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    if distance_metric == "cosine":
        X_norm = normalize_rows(np.asarray(X, dtype=float))
        similarities = np.clip(X_norm @ X_norm.T, -1.0, 1.0)
        distances = 1.0 - similarities
        np.fill_diagonal(distances, 0.0)
        return distances

    deltas = X[:, None, :] - X[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, 0.0)
    return distances




def find_galaxy_precomputed_path() -> Optional[Path]:
    for path in GALAXY_PRECOMPUTED_PATHS:
        expanded_path = path.expanduser()
        if expanded_path.exists():
            return expanded_path
    return None


def load_galaxy_precomputed_payload() -> Optional[Dict[str, object]]:
    csv_path = find_galaxy_precomputed_path()
    if csv_path is None:
        return None

    mtime_ns = csv_path.stat().st_mtime_ns
    cached_payload = _GALAXY_PRECOMPUTED_CACHE.get("payload")
    if (
        cached_payload is not None
        and _GALAXY_PRECOMPUTED_CACHE.get("path") == str(csv_path)
        and _GALAXY_PRECOMPUTED_CACHE.get("mtime_ns") == mtime_ns
    ):
        return cached_payload  # type: ignore[return-value]

    with csv_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    _GALAXY_PRECOMPUTED_CACHE["path"] = str(csv_path)
    _GALAXY_PRECOMPUTED_CACHE["mtime_ns"] = mtime_ns
    _GALAXY_PRECOMPUTED_CACHE["payload"] = payload
    return payload


def get_matching_precomputed_galaxy_assets(guards: pd.DataFrame) -> Optional[tuple[np.ndarray, Dict[str, object], Dict[str, object]]]:
    payload = load_galaxy_precomputed_payload()
    if payload is None:
        return None

    coordinate_rows = payload.get("coordinates", [])
    if not isinstance(coordinate_rows, list):
        return None

    coordinate_by_key: Dict[str, Dict[str, object]] = {
        str(row.get("player_key")): row
        for row in coordinate_rows
        if isinstance(row, dict) and row.get("player_key") is not None
    }

    guard_keys = guards["player_key"].astype(str).tolist()
    if set(guard_keys) != set(coordinate_by_key.keys()):
        return None

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
    expected_signature = build_locked_euclidean_feature_signature()
    metric_meta = payload.get("metric_meta", {})
    if not isinstance(metric_meta, dict):
        return None

    precomputed_signature = str(metric_meta.get("euclidean_kmeans_locked_feature_signature", ""))
    if not precomputed_signature:
        legacy_signature_payload = {
            "euclidean_kmeans_locked_group_features": metric_meta.get("euclidean_kmeans_locked_group_features"),
            "euclidean_kmeans_locked_group_weights": metric_meta.get("euclidean_kmeans_locked_group_weights"),
            "euclidean_kmeans_locked_group_order": metric_meta.get("euclidean_kmeans_locked_group_order"),
            "euclidean_kmeans_locked_k": metric_meta.get("euclidean_kmeans_locked_k"),
            "euclidean_kmeans_locked_clip_zscore": metric_meta.get("euclidean_kmeans_locked_clip_zscore"),
            "euclidean_kmeans_locked_pipeline": metric_meta.get("pipeline"),
            "euclidean_kmeans_locked_similarity_distance_metric": metric_meta.get("similarity_distance_metric"),
            "euclidean_kmeans_locked_luka_override_metric": metric_meta.get("euclidean_kmeans_locked_luka_euclidean_override_metric"),
        }
        raw_legacy_signature = json.dumps(legacy_signature_payload, sort_keys=True).encode("utf-8")
        precomputed_signature = hashlib.sha256(raw_legacy_signature).hexdigest()

    if precomputed_signature != expected_signature:
        return None

    display_meta = {**display_meta, "precomputed": True}
    galaxy_payload = payload.get("galaxy", {})
    if not isinstance(galaxy_payload, dict):
        return None
    galaxy_payload = {**galaxy_payload, "precomputed": True}
    return coordinates, display_meta, galaxy_payload


def normalize_galaxy_coordinates(coordinates: np.ndarray) -> np.ndarray:
    if coordinates.size == 0:
        return coordinates
    centered = coordinates - np.nanmean(coordinates, axis=0, keepdims=True)
    scale = float(np.nanpercentile(np.linalg.norm(centered, axis=1), 95))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return centered / scale


def compute_galaxy_display_coordinates(X_metric: np.ndarray) -> tuple[np.ndarray, Dict[str, object]]:
    if X_metric.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), {"method": "empty"}

    # Never run live UMAP inside /api/cluster by default. UMAP can block the
    # initial page render for a long time; production-quality galaxy coordinates
    # should come from scripts/precompute_galaxy_assets.py. If no precomputed
    # galaxy file exists, fall back to fast 3D PCA so the site loads immediately.
    if os.environ.get("ENABLE_LIVE_GALAXY_UMAP", "0").strip().lower() in {"1", "true", "yes"} and X_metric.shape[0] >= 8:
        try:
            import umap  # type: ignore

            model = umap.UMAP(
                n_components=3,
                n_neighbors=min(GALAXY_UMAP_N_NEIGHBORS, max(2, X_metric.shape[0] - 1)),
                min_dist=GALAXY_UMAP_MIN_DIST,
                metric="euclidean",
                random_state=GALAXY_RANDOM_STATE,
            )
            coordinates = model.fit_transform(X_metric)
            return normalize_galaxy_coordinates(np.asarray(coordinates, dtype=float)), {
                "method": "umap_3d",
                "n_components": 3,
                "n_neighbors": int(min(GALAXY_UMAP_N_NEIGHBORS, max(2, X_metric.shape[0] - 1))),
                "min_dist": float(GALAXY_UMAP_MIN_DIST),
                "metric": "euclidean",
                "random_state": int(GALAXY_RANDOM_STATE),
                "live_umap_enabled": True,
            }
        except Exception as exc:
            fallback_reason = str(exc)
    else:
        fallback_reason = "Live UMAP disabled; using fast PCA fallback. Run scripts/precompute_galaxy_assets.py for cached UMAP galaxy coordinates."

    component_count = int(min(3, X_metric.shape[1], X_metric.shape[0]))
    if component_count <= 0:
        coordinates = np.zeros((X_metric.shape[0], 3), dtype=float)
    else:
        pca_model = PCA(n_components=component_count, svd_solver="full")
        reduced = pca_model.fit_transform(X_metric)
        coordinates = np.zeros((X_metric.shape[0], 3), dtype=float)
        coordinates[:, :component_count] = reduced

    return normalize_galaxy_coordinates(coordinates), {
        "method": "pca_3d_fallback",
        "n_components": int(component_count),
        "fallback_reason": fallback_reason,
    }


def convert_truth_distance_to_similarity_score(distance_value: float, sigma_value: float) -> float:
    if not np.isfinite(distance_value):
        return 0.0
    if not np.isfinite(sigma_value) or sigma_value <= 1e-12:
        return 100.0
    score = 100.0 * np.exp(-((distance_value ** 2) / (2.0 * (sigma_value ** 2))))
    return float(np.clip(score, 0.0, 100.0))


def get_similarity_candidate_indices(guards: pd.DataFrame, source_index: int) -> np.ndarray:
    player_name_array = guards["Player Name"].astype(str).to_numpy()
    normalized_name_array = np.asarray(
        [normalize_player_name_for_assignment_key(player_name) for player_name in player_name_array],
        dtype=object,
    )
    row_count = int(len(player_name_array))
    candidate_mask = np.ones(row_count, dtype=bool)
    candidate_mask[int(source_index)] = False
    candidate_mask &= normalized_name_array != normalized_name_array[int(source_index)]
    return np.where(candidate_mask)[0]


def safe_similarity_sigma(distance_values: np.ndarray) -> float:
    finite_distances = np.asarray(distance_values, dtype=float)
    finite_distances = finite_distances[np.isfinite(finite_distances)]
    if finite_distances.size == 0:
        return 1.0

    sigma_value = float(np.nanquantile(finite_distances, 0.10))
    if not np.isfinite(sigma_value) or sigma_value <= 1e-12:
        positive_distances = finite_distances[finite_distances > 1e-12]
        sigma_value = float(np.nanmedian(positive_distances)) if positive_distances.size else 1.0
    return sigma_value


def build_similarity_block_details(
    guards: pd.DataFrame,
    X_metric: np.ndarray,
    block_slices: Dict[str, Dict[str, int]],
    source_index: int,
    target_index: int,
    distance_metric: str = EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
) -> Dict[str, object]:
    candidate_indices = get_similarity_candidate_indices(guards, source_index)
    block_scores: Dict[str, Dict[str, float]] = {}

    for block_name in EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER:
        slice_meta = block_slices.get(block_name, {})
        start_index = int(slice_meta.get("start_component_index", 0))
        end_index = int(slice_meta.get("end_component_index", start_index))
        if end_index <= start_index:
            continue

        source_vector = X_metric[int(source_index), start_index:end_index]
        target_vector = X_metric[int(target_index), start_index:end_index]
        if distance_metric == "cosine":
            source_norm = normalize_rows(source_vector.reshape(1, -1))[0]
            target_norm = normalize_rows(target_vector.reshape(1, -1))[0]
            distance_value = float(1.0 - np.clip(np.dot(source_norm, target_norm), -1.0, 1.0))
        else:
            distance_value = float(np.linalg.norm(source_vector - target_vector))

        if candidate_indices.size:
            candidate_matrix = X_metric[candidate_indices, start_index:end_index]
            if distance_metric == "cosine":
                source_norm = normalize_rows(source_vector.reshape(1, -1))[0]
                candidate_norms = normalize_rows(candidate_matrix)
                candidate_distances = 1.0 - np.clip(candidate_norms @ source_norm, -1.0, 1.0)
            else:
                candidate_distances = np.linalg.norm(candidate_matrix - source_vector.reshape(1, -1), axis=1)
            sigma_value = safe_similarity_sigma(candidate_distances)
        else:
            sigma_value = 1.0

        block_scores[block_name] = {
            "distance": round(distance_value, 6),
            "similarity_score": round(convert_truth_distance_to_similarity_score(distance_value, sigma_value), 1),
        }

    similarity_ranked_blocks = sorted(
        block_scores.items(),
        key=lambda item: (-float(item[1]["similarity_score"]), float(item[1]["distance"]), item[0]),
    )
    difference_ranked_blocks = sorted(
        block_scores.items(),
        key=lambda item: (float(item[1]["distance"]), -float(item[1]["similarity_score"]), item[0]),
        reverse=True,
    )

    def format_block_summary(items: List[tuple[str, Dict[str, float]]]) -> str:
        return ", ".join(
            f"{block_name} ({float(score_payload['similarity_score']):.1f})"
            for block_name, score_payload in items[:2]
        )

    return {
        "block_scores": block_scores,
        "strongest_similarity_blocks": format_block_summary(similarity_ranked_blocks),
        "biggest_difference_blocks": format_block_summary(difference_ranked_blocks),
    }


def build_similarity_edges_for_galaxy(
    guards: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    top_n: int,
    X_metric: Optional[np.ndarray] = None,
    block_slices: Optional[Dict[str, Dict[str, int]]] = None,
    default_similarity_metric: str = EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
    euclidean_override_distance_matrix: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    player_key_array = guards["player_key"].astype(str).to_numpy()
    player_name_array = guards["Player Name"].astype(str).to_numpy()
    normalized_name_array = np.asarray(
        [normalize_player_name_for_assignment_key(player_name) for player_name in player_name_array],
        dtype=object,
    )
    season_array = guards["Season"].astype(str).to_numpy()
    team_array = guards["teams_played"].astype(str).to_numpy()
    position_array = guards["position"].astype(str).to_numpy()
    edges: List[Dict[str, object]] = []
    row_count = int(len(player_key_array))

    for source_index in range(row_count):
        source_uses_euclidean_override = (
            normalized_name_array[int(source_index)] == LUKA_DONCIC_SIMILARITY_OVERRIDE_KEY
            and euclidean_override_distance_matrix is not None
        )
        source_distance_metric = (
            LUKA_DONCIC_SIMILARITY_DISTANCE_METRIC
            if source_uses_euclidean_override
            else default_similarity_metric
        )
        source_distance_matrix = (
            euclidean_override_distance_matrix
            if source_uses_euclidean_override
            else distance_matrix
        )
        candidate_mask = normalized_name_array != normalized_name_array[int(source_index)]
        candidate_mask[int(source_index)] = False
        candidate_indices = np.where(candidate_mask)[0]
        if candidate_indices.size == 0:
            continue

        source_distances = source_distance_matrix[source_index, candidate_indices]
        order = np.argsort(source_distances, kind="stable")[:top_n]
        selected_indices = candidate_indices[order]
        sigma_value = safe_similarity_sigma(source_distances)

        for rank_index, target_index in enumerate(selected_indices, start=1):
            distance_value = float(source_distance_matrix[source_index, target_index])
            block_detail_payload: Dict[str, object] = {
                "block_scores": {},
                "strongest_similarity_blocks": "",
                "biggest_difference_blocks": "",
            }
            if X_metric is not None and block_slices:
                block_detail_payload = build_similarity_block_details(
                    guards=guards,
                    X_metric=np.asarray(X_metric, dtype=float),
                    block_slices=block_slices,
                    source_index=int(source_index),
                    target_index=int(target_index),
                    distance_metric=source_distance_metric,
                )

            edges.append(
                {
                    "source": str(player_key_array[source_index]),
                    "target": str(player_key_array[target_index]),
                    "source_player_name": str(player_name_array[source_index]),
                    "source_season": str(season_array[source_index]),
                    "source_team": str(team_array[source_index]),
                    "source_position": str(position_array[source_index]),
                    "target_player_name": str(player_name_array[target_index]),
                    "target_season": str(season_array[target_index]),
                    "target_team": str(team_array[target_index]),
                    "target_position": str(position_array[target_index]),
                    "rank": int(rank_index),
                    "truth_distance": round(distance_value, 6),
                    "similarity_distance_metric": source_distance_metric,
                    "similarity_metric_used": source_distance_metric,
                    "similarity_score": round(convert_truth_distance_to_similarity_score(distance_value, sigma_value), 1),
                    "same_cluster": bool(labels[source_index] == labels[target_index]),
                    "source_cluster": int(labels[source_index]),
                    "target_cluster": int(labels[target_index]),
                    **block_detail_payload,
                }
            )

    return edges

def build_minimum_spanning_tree_edge_pairs(local_distance_matrix: np.ndarray) -> List[tuple[int, int]]:
    local_count = int(local_distance_matrix.shape[0])
    if local_count <= 1:
        return []

    try:
        from scipy.sparse.csgraph import minimum_spanning_tree  # type: ignore

        mst = minimum_spanning_tree(local_distance_matrix)
        row_indices, col_indices = mst.nonzero()
        return [(int(row), int(col)) for row, col in zip(row_indices, col_indices)]
    except Exception:
        # Dense Prim fallback so the app still works if SciPy is not importable.
        selected = np.zeros(local_count, dtype=bool)
        selected[0] = True
        edge_pairs: List[tuple[int, int]] = []
        for _ in range(local_count - 1):
            best_source = -1
            best_target = -1
            best_distance = np.inf
            selected_indices = np.where(selected)[0]
            unselected_indices = np.where(~selected)[0]
            for source in selected_indices:
                candidate_distances = local_distance_matrix[source, unselected_indices]
                local_position = int(np.argmin(candidate_distances))
                candidate_target = int(unselected_indices[local_position])
                candidate_distance = float(candidate_distances[local_position])
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_source = int(source)
                    best_target = candidate_target
            if best_source < 0 or best_target < 0:
                break
            selected[best_target] = True
            edge_pairs.append((best_source, best_target))
        return edge_pairs


def build_cluster_constellation_edges_for_galaxy(
    guards: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    same_cluster_knn_count: int,
) -> List[Dict[str, object]]:
    player_key_array = guards["player_key"].astype(str).to_numpy()
    edges_by_key: Dict[tuple[int, int], Dict[str, object]] = {}

    for cluster_number in sorted(set(int(value) for value in labels.tolist())):
        cluster_indices = np.where(labels == cluster_number)[0]
        if cluster_indices.size <= 1:
            continue

        local_distance_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)].copy()
        local_scale_value = float(np.nanquantile(local_distance_matrix[local_distance_matrix > 0], 0.90)) if np.any(local_distance_matrix > 0) else 1.0
        if not np.isfinite(local_scale_value) or local_scale_value <= 1e-12:
            local_scale_value = 1.0

        def add_edge(global_source_index: int, global_target_index: int, edge_type: str) -> None:
            source_index = int(min(global_source_index, global_target_index))
            target_index = int(max(global_source_index, global_target_index))
            edge_key = (source_index, target_index)
            distance_value = float(distance_matrix[source_index, target_index])
            edge_strength = float(np.clip(1.0 - (distance_value / local_scale_value), 0.08, 1.0))
            if edge_key in edges_by_key:
                if edge_type not in str(edges_by_key[edge_key]["edge_type"]):
                    edges_by_key[edge_key]["edge_type"] = f"{edges_by_key[edge_key]['edge_type']}+{edge_type}"
                edges_by_key[edge_key]["edge_strength"] = max(float(edges_by_key[edge_key]["edge_strength"]), edge_strength)
                return
            edges_by_key[edge_key] = {
                "source": str(player_key_array[source_index]),
                "target": str(player_key_array[target_index]),
                "cluster": int(cluster_number),
                "edge_type": edge_type,
                "truth_distance": round(distance_value, 6),
                "edge_strength": round(edge_strength, 4),
            }

        for local_source, local_target in build_minimum_spanning_tree_edge_pairs(local_distance_matrix):
            add_edge(int(cluster_indices[local_source]), int(cluster_indices[local_target]), "mst")

        if same_cluster_knn_count > 0:
            for local_source_index, global_source_index in enumerate(cluster_indices):
                order = np.argsort(local_distance_matrix[local_source_index], kind="stable")
                neighbor_positions = [int(position) for position in order if int(position) != int(local_source_index)][:same_cluster_knn_count]
                for local_target_index in neighbor_positions:
                    add_edge(int(global_source_index), int(cluster_indices[local_target_index]), "knn")

    return list(edges_by_key.values())


def build_cluster_medoids_for_galaxy(
    guards: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    algorithm: str,
    distance_metric: str,
) -> List[Dict[str, object]]:
    medoids: List[Dict[str, object]] = []
    for cluster_number in sorted(set(int(value) for value in labels.tolist())):
        cluster_indices = np.where(labels == cluster_number)[0]
        if cluster_indices.size == 0:
            continue
        local_distance_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        local_mean_distances = np.mean(local_distance_matrix, axis=1)
        medoid_global_index = int(cluster_indices[int(np.argmin(local_mean_distances))])
        medoid_row = guards.iloc[medoid_global_index]
        medoids.append(
            {
                "cluster": int(cluster_number),
                "cluster_name": get_cluster_title(cluster_number, algorithm, distance_metric),
                "player_key": str(medoid_row["player_key"]),
                "player_name": str(medoid_row["Player Name"]),
                "season": str(medoid_row["Season"]),
                "teams_played": str(medoid_row["teams_played"]),
                "position": str(medoid_row["position"]),
                "mean_truth_distance": round(float(local_mean_distances.min()), 6),
            }
        )
    return medoids


def build_archetype_labels_for_galaxy(
    labels: np.ndarray,
    display_coordinates: np.ndarray,
    algorithm: str,
    distance_metric: str,
) -> List[Dict[str, object]]:
    label_rows: List[Dict[str, object]] = []
    for cluster_number in sorted(set(int(value) for value in labels.tolist())):
        cluster_indices = np.where(labels == cluster_number)[0]
        if cluster_indices.size == 0:
            continue
        center = np.mean(display_coordinates[cluster_indices], axis=0)
        label_rows.append(
            {
                "cluster": int(cluster_number),
                "cluster_name": get_cluster_title(cluster_number, algorithm, distance_metric),
                "x": round(float(center[0]), 6),
                "y": round(float(center[1]), 6),
                "z": round(float(center[2]), 6),
                "player_count": int(cluster_indices.size),
            }
        )
    return label_rows


def build_galaxy_payload(
    guards: pd.DataFrame,
    X_metric: np.ndarray,
    labels: np.ndarray,
    display_coordinates: np.ndarray,
    display_meta: Dict[str, object],
    algorithm: str,
    distance_metric: str,
    metric_meta: Optional[Dict[str, object]] = None,
    include_block_details: bool = False,
) -> Dict[str, object]:
    similarity_distance_metric = (
        str(metric_meta.get("similarity_distance_metric", metric_meta.get("distance_metric", distance_metric)))
        if metric_meta is not None
        else distance_metric
    )
    distance_matrix = pairwise_distance_matrix(X_metric, similarity_distance_metric)
    euclidean_override_names = []
    if metric_meta is not None:
        raw_override_names = metric_meta.get("euclidean_kmeans_locked_luka_euclidean_override_names", [])
        if isinstance(raw_override_names, list):
            euclidean_override_names = [str(name) for name in raw_override_names]
    euclidean_override_distance_matrix = (
        pairwise_distance_matrix(X_metric, LUKA_DONCIC_SIMILARITY_DISTANCE_METRIC)
        if euclidean_override_names
        else None
    )
    block_slices = None
    if metric_meta is not None and (include_block_details or bool(display_meta.get("precomputed", False))):
        # Block-level edge details are expensive to compute live and should come
        # from the precomputed galaxy asset. Keep live fallback fast so the
        # initial scatter/galaxy render does not sit on LOADING_SCATTER.
        raw_block_slices = metric_meta.get("euclidean_kmeans_locked_block_slices")
        if isinstance(raw_block_slices, dict):
            block_slices = raw_block_slices

    similarity_edges = build_similarity_edges_for_galaxy(
        guards=guards,
        labels=labels,
        distance_matrix=distance_matrix,
        top_n=GALAXY_SIMILAR_PLAYER_COUNT,
        X_metric=X_metric,
        block_slices=block_slices,
        default_similarity_metric=similarity_distance_metric,
        euclidean_override_distance_matrix=euclidean_override_distance_matrix,
    )
    cluster_edges = build_cluster_constellation_edges_for_galaxy(
        guards=guards,
        labels=labels,
        distance_matrix=distance_matrix,
        same_cluster_knn_count=GALAXY_CLUSTER_KNN_COUNT,
    )
    return {
        "enabled": True,
        "truth_space": (
            str(metric_meta.get("space_transform", EUCLIDEAN_KMEANS_LOCKED_SPACE_TRANSFORM))
            if is_locked_euclidean_kmeans_request(algorithm, distance_metric) and metric_meta is not None
            else "active_metric_space"
        ),
        "similarity_distance_metric": similarity_distance_metric,
        "euclidean_override_player_names": euclidean_override_names,
        "display_space": display_meta,
        "similar_player_count": int(GALAXY_SIMILAR_PLAYER_COUNT),
        "same_cluster_knn_count": int(GALAXY_CLUSTER_KNN_COUNT),
        "similarity_edges": similarity_edges,
        "cluster_edges": cluster_edges,
        "cluster_medoids": build_cluster_medoids_for_galaxy(
            guards=guards,
            labels=labels,
            distance_matrix=distance_matrix,
            algorithm=algorithm,
            distance_metric=distance_metric,
        ),
        "archetype_labels": build_archetype_labels_for_galaxy(
            labels=labels,
            display_coordinates=display_coordinates,
            algorithm=algorithm,
            distance_metric=distance_metric,
        ),
    }


def build_cluster_report_cache_key(cluster_cache_key: str, cluster_number: int) -> str:
    raw = json.dumps({
        "cluster_cache_key": cluster_cache_key,
        "cluster_number": int(cluster_number),
        "report_version": 3,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prepare_cluster_runtime(
    dataset_path: str,
    algorithm: str,
    distance_metric: str,
    k: int,
    features: List[str],
) -> Dict:
    dataset_meta = load_base_dataframe(dataset_path)

    if algorithm not in ALLOWED_ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Invalid algorithm: {algorithm}")

    if distance_metric not in ALLOWED_DISTANCE_METRICS:
        raise HTTPException(status_code=400, detail=f"Invalid distance metric: {distance_metric}")

    is_locked_euclidean_kmeans = is_locked_euclidean_kmeans_request(algorithm, distance_metric)
    if is_locked_euclidean_kmeans:
        k = EUCLIDEAN_KMEANS_LOCKED_K
        features = get_locked_euclidean_kmeans_feature_columns(raw=False)

    if is_locked_euclidean_kmeans:
        invalid_features = []
    else:
        invalid_features = [f for f in features if f not in ALLOWED_FEATURES]

    if invalid_features:
        raise HTTPException(status_code=400, detail=f"Invalid feature(s): {invalid_features}")

    if not features and not is_locked_euclidean_kmeans:
        raise HTTPException(status_code=400, detail="At least one feature must be selected.")

    base_guards = dataset_meta["guards"].copy()

    if is_locked_euclidean_kmeans:
        try:
            guards, X_metric, raw_labels, metric_meta = build_locked_euclidean_kmeans_space(base_guards)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        used_features_after_cleaning = metric_meta["euclidean_kmeans_locked_features"]
        dropped_all_nan_features = []
        dropped_zero_variance_features = []

        plot_pca = PCA(n_components=min(2, X_metric.shape[1]))
        X_plot = plot_pca.fit_transform(X_metric)
        pca_variance_captured = float(np.asarray(plot_pca.explained_variance_ratio_, dtype=float).sum())
        pca_components_used_for_clustering = int(X_metric.shape[1])
    else:
        guards = base_guards

        X_numeric = guards[features].apply(pd.to_numeric, errors="coerce")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

        all_nan_cols = X_numeric.columns[X_numeric.isna().all()].tolist()
        X_numeric = X_numeric.drop(columns=all_nan_cols)

        if X_numeric.shape[1] == 0:
            raise HTTPException(status_code=400, detail="No usable numeric feature columns remained after cleaning.")

        X_std_df = seasonwise_guard_standardize(X_numeric, guards["Season"])
        zero_var_mask = X_std_df.var(axis=0, ddof=0) == 0
        dropped_zero_var = X_std_df.columns[zero_var_mask].tolist()
        X_std_df = X_std_df.loc[:, ~zero_var_mask]

        if X_std_df.shape[1] == 0:
            raise HTTPException(status_code=400, detail="All selected features became zero-variance after standardization.")

        X_std = X_std_df.to_numpy(dtype=float)

        pca_full = PCA()
        pca_full.fit(X_std)
        cum_explained = np.cumsum(pca_full.explained_variance_ratio_)
        n_pca_components = int(np.searchsorted(cum_explained, PCA_EXPLAINED_VAR_TARGET) + 1)

        pca = PCA(n_components=n_pca_components)
        X_pca = pca.fit_transform(X_std)
        X_metric, metric_meta = prepare_metric_space(X_pca, pca, distance_metric)

        used_features_after_cleaning = X_std_df.columns.tolist()
        dropped_all_nan_features = all_nan_cols
        dropped_zero_variance_features = dropped_zero_var
        X_plot = X_pca
        pca_variance_captured = float(cum_explained[n_pca_components - 1])
        pca_components_used_for_clustering = int(n_pca_components)

    cluster_centers = None

    if algorithm == "kmeans":
        if is_locked_euclidean_kmeans:
            cluster_centers = recompute_centers(
                points_matrix=X_metric,
                cluster_labels=raw_labels,
                num_clusters=k,
            )
            memberships = np.eye(k, dtype=float)[raw_labels]
            algorithm_meta = {
                "kmeans_variant": "locked_precomputed_raw_equal_block_assignments",
                "similarity_pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
                "similarity_distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
                "kmeans_random_state": None,
                "kmeans_n_init": None,
                "kmeans_iterations_run": 0,
                "locked_assignment_source": str(EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH),
            }
        elif distance_metric == "cosine":
            cluster_centers, raw_labels, iteration_count = spherical_kmeans(
                X_metric,
                n_clusters=k,
                n_init=KMEANS_N_INIT,
                max_iter=KMEANS_MAX_ITER,
                tol=KMEANS_TOL,
                random_state=KMEANS_RANDOM_STATE,
            )
            memberships = np.eye(k, dtype=float)[raw_labels]
            algorithm_meta = {
                "kmeans_n_init": KMEANS_N_INIT,
                "kmeans_random_state": KMEANS_RANDOM_STATE,
                "kmeans_max_iter": KMEANS_MAX_ITER,
                "kmeans_tol": KMEANS_TOL,
                "kmeans_variant": "spherical_kmeans",
                "kmeans_iterations_run": int(iteration_count),
            }
        else:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=KMEANS_N_INIT,
                random_state=KMEANS_RANDOM_STATE,
                max_iter=KMEANS_MAX_ITER,
                tol=KMEANS_TOL,
            )
            raw_labels = kmeans.fit_predict(X_metric)
            cluster_centers = kmeans.cluster_centers_.copy()
            memberships = np.eye(k, dtype=float)[raw_labels]
            algorithm_meta = {
                "kmeans_n_init": KMEANS_N_INIT,
                "kmeans_random_state": KMEANS_RANDOM_STATE,
                "kmeans_max_iter": KMEANS_MAX_ITER,
                "kmeans_tol": KMEANS_TOL,
                "kmeans_variant": "standard_kmeans",
            }

    labels = raw_labels + 1
    guards = guards.reset_index(drop=True)

    if X_plot.shape[1] == 1:
        pc1 = X_plot[:, 0]
        pc2 = np.zeros(len(pc1))
    else:
        pc1 = X_plot[:, 0]
        pc2 = X_plot[:, 1]

    precomputed_galaxy_assets = get_matching_precomputed_galaxy_assets(guards)
    if precomputed_galaxy_assets is not None:
        galaxy_coordinates, galaxy_display_meta, galaxy_payload = precomputed_galaxy_assets
    else:
        galaxy_coordinates, galaxy_display_meta = compute_galaxy_display_coordinates(X_metric)
        galaxy_payload = build_galaxy_payload(
            guards=guards,
            X_metric=np.asarray(X_metric, dtype=float),
            labels=np.asarray(labels, dtype=int),
            display_coordinates=np.asarray(galaxy_coordinates, dtype=float),
            display_meta=galaxy_display_meta,
            algorithm=algorithm,
            distance_metric=distance_metric,
            metric_meta=metric_meta,
        )

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .reindex(range(1, k + 1), fill_value=0)
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    points = []
    for idx, row in guards.iterrows():
        point_memberships = memberships[idx].astype(float).tolist()
        dominant_cluster = int(labels[idx])
        dominant_probability = float(max(point_memberships))
        points.append(
            {
                "player_key": row["player_key"],
                "player_name": row["Player Name"],
                "season": row["Season"],
                "teams_played": row["teams_played"],
                "position": row["position"],
                **get_player_headshot_payload(row["Player Name"]),
                "cluster": dominant_cluster,
                "memberships": point_memberships,
                "dominant_probability": dominant_probability,
                "pc1": float(pc1[idx]),
                "pc2": float(pc2[idx]),
                "galaxy_x": float(galaxy_coordinates[idx, 0]),
                "galaxy_y": float(galaxy_coordinates[idx, 1]),
                "galaxy_z": float(galaxy_coordinates[idx, 2]),
            }
        )

    cluster_cache_key = build_cache_key(dataset_meta, algorithm, distance_metric, k, features)

    payload = {
        "cache_hit": False,
        "algorithm": algorithm,
        "distance_metric": distance_metric,
        "cluster_count": int(k),
        "k": int(k),
        "selected_features": features,
        "used_features_after_cleaning": used_features_after_cleaning,
        "dropped_all_nan_features": dropped_all_nan_features,
        "dropped_zero_variance_features": dropped_zero_variance_features,
        "pca_components_used_for_clustering": pca_components_used_for_clustering,
        "pca_variance_captured": pca_variance_captured,
        "points": points,
        "cluster_sizes": cluster_sizes,
        "algorithm_meta": algorithm_meta,
        "distance_metric_meta": metric_meta,
        "galaxy": galaxy_payload,
    }

    return {
        "dataset_meta": dataset_meta,
        "guards": guards,
        "memberships": np.asarray(memberships, dtype=float),
        "labels": np.asarray(labels, dtype=int),
        "X_metric": np.asarray(X_metric, dtype=float),
        "cluster_centers": None if cluster_centers is None else np.asarray(cluster_centers, dtype=float),
        "cluster_cache_key": cluster_cache_key,
        "payload": payload,
    }


def compute_cluster_payload(
    dataset_path: str,
    algorithm: str,
    distance_metric: str,
    k: int,
    features: List[str],
) -> Dict:
    dataset_meta = load_base_dataframe(dataset_path)
    cache_key = build_cache_key(dataset_meta, algorithm, distance_metric, k, features)
    cached = load_cluster_cache(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    runtime = prepare_cluster_runtime(dataset_path, algorithm, distance_metric, k, features)
    payload = runtime["payload"]
    save_cluster_cache(cache_key, payload)
    return payload


def compute_cluster_report_payload(
    dataset_path: str,
    algorithm: str,
    distance_metric: str,
    k: int,
    features: List[str],
    cluster_number: int,
) -> Dict:
    dataset_meta = load_base_dataframe(dataset_path)
    cluster_cache_key = build_cache_key(dataset_meta, algorithm, distance_metric, k, features)
    report_cache_key = build_cluster_report_cache_key(cluster_cache_key, cluster_number)
    cached = load_cluster_cache(report_cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    runtime = prepare_cluster_runtime(dataset_path, algorithm, distance_metric, k, features)
    guards = runtime["guards"].reset_index(drop=True)
    labels = runtime["labels"]
    X_metric = runtime["X_metric"]
    payload = runtime["payload"]

    if cluster_number < 1 or cluster_number > int(k):
        raise HTTPException(status_code=400, detail=f"Cluster number out of range: {cluster_number}")

    member_mask = labels == int(cluster_number)
    member_count = int(member_mask.sum())
    if member_count == 0:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_number} has no assigned players.")

    used_features = list(payload["used_features_after_cleaning"])
    numeric_df = guards[used_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    percentile_df = compute_season_percentiles(numeric_df.fillna(numeric_df.median(numeric_only=True)), guards["Season"]).fillna(0.0)

    member_numeric_df = numeric_df.loc[member_mask]
    other_numeric_df = numeric_df.loc[~member_mask]
    member_percentile_df = percentile_df.loc[member_mask]

    global_medians = numeric_df.median(axis=0, skipna=True).fillna(0.0)
    cluster_medians = member_numeric_df.median(axis=0, skipna=True).fillna(0.0)
    cluster_means = member_numeric_df.mean(axis=0, skipna=True).fillna(0.0)
    cluster_median_percentiles = member_percentile_df.median(axis=0, skipna=True).fillna(0.0)
    cluster_mean_percentiles = member_percentile_df.mean(axis=0, skipna=True).fillna(0.0)

    if other_numeric_df.empty:
        other_cluster_medians = global_medians.copy()
    else:
        other_cluster_medians = other_numeric_df.median(axis=0, skipna=True).fillna(0.0)

    mad_all = pd.Series({feature: median_absolute_deviation(numeric_df[feature]) for feature in used_features}, dtype=float)
    mad_denominator = mad_all.clip(lower=1e-12)

    robust_distinct_z = ((cluster_medians - other_cluster_medians) / mad_denominator).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    cluster_heat_z = ((cluster_medians - global_medians) / mad_denominator).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    cluster_member_metric_matrix = X_metric[member_mask]
    cluster_member_distances = pairwise_distance_matrix(cluster_member_metric_matrix, distance_metric)
    member_distance_sums = cluster_member_distances.sum(axis=1)

    member_points = pd.DataFrame(payload["points"]).loc[member_mask].reset_index(drop=True)
    member_points_sorted = member_points.assign(
        player_name_sort=member_points["player_name"].astype(str).str.lower(),
        season_sort=member_points["season"].astype(str),
    ).sort_values(["player_name_sort", "season_sort", "player_key"], kind="stable").reset_index(drop=True)

    heatmap_rows = [
        {
            "row_type": "cluster_summary",
            "label": "CLUSTER_MEDIAN_Z",
            "player_key": None,
            "player_name": f"Cluster {int(cluster_number)}",
            "season": "",
            "teams_played": "",
            "position": "",
            "values": [
                {
                    "feature": feature,
                    "heatmap_z": float(cluster_heat_z[feature]),
                    "raw_value": float(cluster_medians[feature]),
                    "percentile_value": float(cluster_median_percentiles[feature]),
                }
                for feature in used_features
            ],
        }
    ]

    for _, point_row in member_points_sorted.iterrows():
        player_key = point_row["player_key"]
        guard_idx = guards.index[guards["player_key"] == player_key]
        if guard_idx.empty:
            continue
        idx = int(guard_idx[0])
        player_values = []
        for feature in used_features:
            raw_value = float(0.0 if pd.isna(numeric_df.at[idx, feature]) else numeric_df.at[idx, feature])
            heat_value = float((raw_value - global_medians[feature]) / mad_denominator[feature])
            player_values.append({
                "feature": feature,
                "heatmap_z": heat_value,
                "raw_value": raw_value,
                "percentile_value": float(percentile_df.at[idx, feature]),
            })

        heatmap_rows.append({
            "row_type": "player",
            "label": point_row["player_name"],
            "player_key": player_key,
            "player_name": point_row["player_name"],
            "season": point_row["season"],
            "teams_played": point_row["teams_played"],
            "position": point_row["position"],
            "values": player_values,
        })

    feature_summaries = []
    for feature in used_features:
        feature_summaries.append({
            "feature": feature,
            "robust_z": float(robust_distinct_z[feature]),
            "cluster_heat_z": float(cluster_heat_z[feature]),
            "median_raw": float(cluster_medians[feature]),
            "mean_raw": float(cluster_means[feature]),
            "median_percentile": float(cluster_median_percentiles[feature]),
            "mean_percentile": float(cluster_mean_percentiles[feature]),
            "mad_all": float(mad_all[feature]),
            "global_median": float(global_medians[feature]),
            "other_cluster_median": float(other_cluster_medians[feature]),
        })

    feature_summaries_sorted = sorted(feature_summaries, key=lambda item: item["robust_z"], reverse=True)

    typical_indices = np.argsort(member_distance_sums)[: min(3, member_points.shape[0])].tolist()
    outlier_indices = np.argsort(member_distance_sums)[-min(3, member_points.shape[0]):][::-1].tolist()
    outlier_distance_lookup = {int(idx): float(member_distance_sums[int(idx)]) for idx in outlier_indices}

    medoid_local_index = int(typical_indices[0]) if typical_indices else None

    def summarize_member(index: int, outlier_mode: bool = False) -> Dict:
        point_row = member_points.iloc[int(index)]
        return {
            "player_key": point_row["player_key"],
            "player_name": point_row["player_name"],
            "season": point_row["season"],
            "teams_played": point_row["teams_played"],
            "position": point_row["position"],
            "distance_sum": float(member_distance_sums[int(index)]),
            "distance_to_center": float(outlier_distance_lookup.get(int(index), 0.0)) if outlier_mode else None,
            "is_medoid": bool(medoid_local_index is not None and int(index) == medoid_local_index),
        }

    report_payload = {
        "cache_hit": False,
        "algorithm": algorithm,
        "distance_metric": distance_metric,
        "cluster_number": int(cluster_number),
        "cluster_title": get_cluster_title(cluster_number, algorithm, distance_metric),
        "cluster_size": member_count,
        "description_text": get_cluster_description(cluster_number, algorithm, distance_metric),
        "feature_order": used_features,
        "heatmap_rows": heatmap_rows,
        "feature_summaries": feature_summaries_sorted,
        "top_features": feature_summaries_sorted[:5],
        "bottom_features": list(reversed(feature_summaries_sorted[-5:])),
        "typical_players": [summarize_member(idx) for idx in typical_indices],
        "notable_outliers": [summarize_member(idx, outlier_mode=True) for idx in outlier_indices],
        "heatmap_logic": {
            "summary_row": "(cluster median - global median) / MAD(feature across full guard pool)",
            "player_rows": "(player raw value - global median) / MAD(feature across full guard pool)",
            "feature_rankings": "(cluster median - other clusters median) / MAD(feature across full guard pool)",
        },
    }

    save_cluster_cache(report_cache_key, report_payload)
    return report_payload



def build_named_breakdown_payload(
    dataset_path: str,
    algorithm: str,
    distance_metric: str,
    k: int,
    features: List[str],
    player_key: str,
    group_order: List[str],
    group_features: Dict[str, List[str]],
    lower_is_better_by_group: Optional[Dict[str, set]] = None,
    local_percentile_rules_by_group: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    breakdown_kind: str = "skill_breakdown",
    score_logic: str = "Each component is the median season-adjusted percentile of that component's features.",
    excluded_features: Optional[List[str]] = None,
) -> Dict:
    runtime = prepare_cluster_runtime(dataset_path, algorithm, distance_metric, k, features)
    guards = runtime["guards"].copy().reset_index(drop=True)
    labels = np.asarray(runtime["labels"], dtype=int)

    if guards.empty:
        raise ValueError("No active guard rows are available for the breakdown.")

    player_key_string = str(player_key)
    matching_indices = guards.index[guards["player_key"].astype(str) == player_key_string].tolist()
    if not matching_indices:
        raise KeyError("Selected player is not present in the active clustering payload.")

    player_index = int(matching_indices[0])
    player_row = guards.iloc[player_index]
    player_cluster_number = int(labels[player_index])

    component_score_frame, used_group_features, missing_features = build_component_score_frame(
        guards=guards,
        group_order=group_order,
        group_features=group_features,
        lower_is_better_by_group=lower_is_better_by_group,
        local_percentile_rules_by_group=local_percentile_rules_by_group,
    )
    if missing_features:
        raise ValueError(f"Missing {breakdown_kind} feature columns: {missing_features}")

    def component_scores_for_row(row_index: int) -> Dict[str, float]:
        return {
            group_name: float(component_score_frame.at[row_index, group_name])
            for group_name in group_order
        }

    def component_median_scores(row_mask: np.ndarray) -> Dict[str, float]:
        selected_rows = component_score_frame.loc[row_mask, group_order]
        if selected_rows.empty:
            return {group_name: 0.0 for group_name in group_order}
        medians = selected_rows.median(axis=0, skipna=True).fillna(0.0)
        return {
            group_name: float(medians[group_name])
            for group_name in group_order
        }

    cluster_mask = labels == player_cluster_number

    player_name = str(player_row["Player Name"])
    player_season = str(player_row["Season"])
    player_team = str(player_row["teams_played"])
    player_position = str(player_row["position"])
    cluster_title = get_cluster_title(player_cluster_number, algorithm, distance_metric)
    local_lower_is_better = {
        group_name: sorted(list(features_for_group))
        for group_name, features_for_group in (lower_is_better_by_group or {}).items()
        if features_for_group
    }
    local_percentile_rules = local_percentile_rules_by_group or {}

    return {
        "breakdown_kind": breakdown_kind,
        "algorithm": algorithm,
        "distance_metric": distance_metric,
        "k": int(runtime["payload"].get("k", k)),
        "player_key": player_key_string,
        "cluster_number": player_cluster_number,
        "cluster_title": cluster_title,
        "axes": group_order,
        "feature_groups": used_group_features,
        "excluded_features": sorted(excluded_features or []),
        "lower_is_better_percentile_features": sorted(LOWER_IS_BETTER_PERCENTILE_FEATURES),
        "local_lower_is_better_by_group": local_lower_is_better,
        "local_percentile_rules_by_group": local_percentile_rules,
        "score_logic": score_logic,
        "player": {
            "label": player_name,
            "player_name": player_name,
            "season": player_season,
            "team": player_team,
            "position": player_position,
            **get_player_headshot_payload(player_name),
            "scores": component_scores_for_row(player_index),
        },
        "cluster_median": {
            "label": f"{cluster_title} Median",
            "cluster_number": player_cluster_number,
            "cluster_title": cluster_title,
            "scores": component_median_scores(cluster_mask),
        },
        "guard_median": {
            "label": f"Median Guard {player_season}",
            "season": player_season,
            "scores": component_median_scores(guards["Season"].astype(str).eq(player_season).to_numpy()),
        },
    }


def build_new_percentile_breakdown_payload(dataset_path: str, algorithm: str, distance_metric: str, k: int, features: List[str], player_key: str, breakdown_kind: str) -> Dict:
    runtime = prepare_cluster_runtime(dataset_path, algorithm, distance_metric, k, features)
    guards = runtime["guards"].copy().reset_index(drop=True)
    labels = np.asarray(runtime["labels"], dtype=int)
    if guards.empty:
        raise ValueError("No active guard rows are available for the breakdown.")
    player_key_string = str(player_key)
    matching_indices = guards.index[guards["player_key"].astype(str) == player_key_string].tolist()
    if not matching_indices:
        raise KeyError("Selected player is not present in the active clustering payload.")
    player_index = int(matching_indices[0])
    player_row = guards.iloc[player_index]
    player_cluster_number = int(labels[player_index])
    component_score_frame, subsection_frames, used_group_features, missing_features, required_features = build_skill_breakdown_score_frames(guards=guards, breakdown_kind=breakdown_kind)
    if missing_features:
        raise ValueError(f"Missing {breakdown_kind} feature columns: {missing_features}")
    group_order = THREE_PT_BREAKDOWN_GROUP_ORDER if breakdown_kind == "three_pt_breakdown" else SKILL_BREAKDOWN_GROUP_ORDER
    def component_scores_for_row(row_index: int) -> Dict[str, float]:
        return {group_name: float(component_score_frame.at[row_index, group_name]) for group_name in group_order}
    def subsection_scores_for_row(row_index: int) -> Dict[str, Dict[str, float]]:
        if breakdown_kind == "three_pt_breakdown":
            return {"3PT Shooting Talent": {group_name: float(component_score_frame.at[row_index, group_name]) for group_name in group_order}}
        return {group_name: {subsection_name: float(subsection_frames[group_name].at[row_index, subsection_name]) for subsection_name in subsection_frames[group_name].columns} for group_name in group_order if group_name in subsection_frames}
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
            medians = selected_rows.median(axis=0, skipna=True).fillna(0.0) if not selected_rows.empty else pd.Series(0.0, index=subsection_frame.columns)
            payload[group_name] = {column_name: float(medians[column_name]) for column_name in subsection_frame.columns}
        return payload
    cluster_mask = labels == player_cluster_number
    player_name = str(player_row["Player Name"])
    player_season = str(player_row["Season"])
    season_guard_mask = guards["Season"].astype(str).eq(player_season).to_numpy()
    cluster_title = get_cluster_title(player_cluster_number, algorithm, distance_metric)
    score_logic = "Same-season guard percentile scoring with LeBron James, Scottie Barnes, and Ben Simmons excluded from percentile peer pools."
    if breakdown_kind == "skill_breakdown":
        score_logic += " ThreePT, MidRange, and RimPressure use median subsection scores; Playmaking averages subsection scores; D-LEBRON uses the same-season guard percentile of D-LEBRON."
    return {
        "breakdown_kind": breakdown_kind,
        "algorithm": algorithm,
        "distance_metric": distance_metric,
        "k": int(runtime["payload"].get("k", k)),
        "player_key": player_key_string,
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
        "player": {"label": player_name, "player_name": player_name, "season": str(player_row["Season"]), "team": str(player_row["teams_played"]), "position": str(player_row["position"]), **get_player_headshot_payload(player_name), "scores": component_scores_for_row(player_index), "subsections": subsection_scores_for_row(player_index)},
        "cluster_median": {"label": f"{cluster_title} Median", "cluster_number": player_cluster_number, "cluster_title": cluster_title, "scores": component_median_scores(cluster_mask), "subsections": subsection_median_scores(cluster_mask)},
        "guard_median": {"label": f"Median Guard {player_season}", "season": player_season, "scores": component_median_scores(season_guard_mask), "subsections": subsection_median_scores(season_guard_mask)},
    }

def load_precomputed_breakdown_file(breakdown_kind: str) -> Optional[Dict[str, object]]:
    path = BREAKDOWN_PRECOMPUTED_PATHS.get(breakdown_kind)
    if path is None or not path.exists():
        return None

    mtime_ns = path.stat().st_mtime_ns
    cache_entry = _BREAKDOWN_PRECOMPUTED_CACHE.get(breakdown_kind)
    if (
        cache_entry is not None
        and cache_entry.get("path") == str(path)
        and cache_entry.get("mtime_ns") == mtime_ns
    ):
        return cache_entry.get("payload")  # type: ignore[return-value]

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    _BREAKDOWN_PRECOMPUTED_CACHE[breakdown_kind] = {
        "path": str(path),
        "mtime_ns": mtime_ns,
        "payload": payload,
    }
    return payload



def get_precomputed_percentile_breakdown_payload(
    dataset_path: str,
    algorithm: str,
    distance_metric: str,
    k: int,
    player_key: str,
    breakdown_kind: str,
) -> Optional[Dict[str, object]]:
    """
    Production-safe precomputed lookup.

    This intentionally does NOT reject precomputed JSON based on dataset mtime,
    feature signatures, or frontend request metadata. Those checks are brittle
    after Git/Render deployment because file mtimes can change. The website's
    locked production mode should serve the already-built JSON payloads directly.
    """
    payload = load_precomputed_breakdown_file(breakdown_kind)
    if payload is None:
        return None

    players = payload.get("players", {})
    if not isinstance(players, dict):
        return None

    requested_key = str(player_key)
    player_payload = players.get(requested_key)
    resolved_key = requested_key

    # Fallback: match by Player Name + Season if team/position formatting differs.
    if not isinstance(player_payload, dict):
        requested_parts = requested_key.split("||")
        requested_name = requested_parts[0].strip().lower() if len(requested_parts) >= 1 else ""
        requested_season = requested_parts[1].strip().lower() if len(requested_parts) >= 2 else ""

        for candidate_key, candidate_payload in players.items():
            candidate_parts = str(candidate_key).split("||")
            candidate_name = candidate_parts[0].strip().lower() if len(candidate_parts) >= 1 else ""
            candidate_season = candidate_parts[1].strip().lower() if len(candidate_parts) >= 2 else ""

            if (
                requested_name
                and requested_season
                and candidate_name == requested_name
                and candidate_season == requested_season
                and isinstance(candidate_payload, dict)
            ):
                player_payload = candidate_payload
                resolved_key = str(candidate_key)
                break

    if not isinstance(player_payload, dict):
        return None

    output_payload = dict(player_payload)
    output_payload["player_key"] = output_payload.get("player_key", resolved_key)
    output_payload["precomputed"] = True
    output_payload["cache_source"] = "precomputed_breakdown_json"
    output_payload["precomputed_validation"] = "forced_precomputed_no_mtime_check"
    return output_payload


def build_skill_breakdown_payload(dataset_path: str, algorithm: str, distance_metric: str, k: int, features: List[str], player_key: str) -> Dict:
    precomputed_payload = get_precomputed_percentile_breakdown_payload(
        dataset_path,
        algorithm,
        distance_metric,
        k,
        player_key,
        "skill_breakdown",
    )
    if precomputed_payload is not None:
        return precomputed_payload

    raise ValueError(f"No precomputed skill_breakdown payload found for player_key: {player_key}")


def build_three_pt_breakdown_payload(dataset_path: str, algorithm: str, distance_metric: str, k: int, features: List[str], player_key: str) -> Dict:
    precomputed_payload = get_precomputed_percentile_breakdown_payload(
        dataset_path,
        algorithm,
        distance_metric,
        k,
        player_key,
        "three_pt_breakdown",
    )
    if precomputed_payload is not None:
        return precomputed_payload

    raise ValueError(f"No precomputed three_pt_breakdown payload found for player_key: {player_key}")

def build_similar_players_response_from_galaxy(player_name: str, season: str) -> Dict[str, object]:
    runtime = prepare_cluster_runtime(
        DEFAULT_DATASET_PATH,
        "kmeans",
        "euclidean",
        EUCLIDEAN_KMEANS_LOCKED_K,
        get_locked_euclidean_kmeans_feature_columns(raw=False),
    )
    payload = runtime["payload"]
    guards = runtime["guards"].reset_index(drop=True)
    X_metric = np.asarray(runtime["X_metric"], dtype=float)
    block_slices = payload.get("distance_metric_meta", {}).get("euclidean_kmeans_locked_block_slices", {})
    if not isinstance(block_slices, dict):
        block_slices = {}

    points = payload.get("points", [])
    point_by_key = {str(point.get("player_key")): point for point in points}
    row_index_by_key = {str(row["player_key"]): int(row_index) for row_index, row in guards.iterrows()}

    source_point = None
    for point in points:
        if (
            normalize_filter_value(point.get("player_name")) == normalize_filter_value(player_name)
            and normalize_filter_value(point.get("season")) == normalize_filter_value(season)
        ):
            source_point = point
            break

    if source_point is None:
        raise HTTPException(status_code=404, detail=f"No similar players found for {player_name} {season}.")

    source_key = str(source_point.get("player_key"))
    similarity_edges = payload.get("galaxy", {}).get("similarity_edges", [])
    source_edges = sorted(
        [edge for edge in similarity_edges if str(edge.get("source")) == source_key],
        key=lambda edge: int(edge.get("rank", 999)),
    )[:GALAXY_SIMILAR_PLAYER_COUNT]

    source_index = row_index_by_key.get(source_key)

    similar_player_items = []
    for edge in source_edges:
        target_key = str(edge.get("target"))
        target_point = point_by_key.get(target_key)
        if target_point is None:
            continue

        block_detail_payload: Dict[str, object] = {
            "block_scores": edge.get("block_scores", {}) if isinstance(edge.get("block_scores"), dict) else {},
            "strongest_similarity_blocks": str(edge.get("strongest_similarity_blocks", "") or ""),
            "biggest_difference_blocks": str(edge.get("biggest_difference_blocks", "") or ""),
        }
        target_index = row_index_by_key.get(target_key)
        if source_index is not None and target_index is not None and block_slices:
            edge_distance_metric = str(edge.get("similarity_distance_metric", EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC))
            block_detail_payload = build_similarity_block_details(
                guards=guards,
                X_metric=X_metric,
                block_slices=block_slices,
                source_index=int(source_index),
                target_index=int(target_index),
                distance_metric=edge_distance_metric,
            )

        related_cluster_number = int(target_point.get("cluster", 0) or 0)
        related_cluster_name = get_cluster_title(related_cluster_number, "kmeans", "euclidean")
        similar_player_items.append(
            {
                "rank": int(edge.get("rank", 0) or 0),
                "player_name": str(target_point.get("player_name", "")),
                "season": str(target_point.get("season", "")),
                "team": str(target_point.get("teams_played", "")),
                "position": str(target_point.get("position", "")),
                "player_season_id": str(target_point.get("player_key", "")),
                "cluster_raw": related_cluster_number - 1,
                "cluster_number": related_cluster_number,
                "cluster_name": related_cluster_name,
                "archetype_name": related_cluster_name,
                **get_player_headshot_payload(target_point.get("player_name", "")),
                "similarity_score": float(edge.get("similarity_score", 0.0) or 0.0),
                "overall_distance": float(edge.get("truth_distance", 0.0) or 0.0),
                "similarity_distance_metric": str(edge.get("similarity_distance_metric", EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC)),
                "similarity_metric_used": str(edge.get("similarity_metric_used", edge.get("similarity_distance_metric", EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC))),
                "same_cluster": bool(edge.get("same_cluster", False)),
                "same_archetype": bool(edge.get("same_cluster", False)),
                "strongest_similarity_blocks": block_detail_payload.get("strongest_similarity_blocks", ""),
                "biggest_difference_blocks": block_detail_payload.get("biggest_difference_blocks", ""),
                "block_scores": block_detail_payload.get("block_scores", {}),
            }
        )

    source_cluster_number = int(source_point.get("cluster", 0) or 0)
    source_cluster_name = get_cluster_title(source_cluster_number, "kmeans", "euclidean")
    return {
        "source_file": "runtime_galaxy_truth_space",
        "player_name": str(source_point.get("player_name", player_name)),
        "season": str(source_point.get("season", season)),
        "team": str(source_point.get("teams_played", "")),
        "position": str(source_point.get("position", "")),
        "player_season_id": source_key,
        "cluster_raw": source_cluster_number - 1,
        "cluster_number": source_cluster_number,
        "cluster": source_cluster_name,
        "cluster_name": source_cluster_name,
        "archetype_name": source_cluster_name,
        **get_player_headshot_payload(source_point.get("player_name", player_name)),
        "similar_players": similar_player_items,
    }


def normalize_comparison_column_name(column_name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column_name).lower())


def build_comparison_column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for column_name in df.columns:
        lookup.setdefault(normalize_comparison_column_name(column_name), str(column_name))
    return lookup


def resolve_comparison_column(df: pd.DataFrame, column_lookup: Dict[str, str], feature_meta: Dict[str, object]) -> Optional[str]:
    candidates = [str(feature_meta.get("feature", ""))]
    candidates.extend(str(alias) for alias in feature_meta.get("aliases", []) if str(alias))
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        normalized_candidate = normalize_comparison_column_name(candidate)
        if normalized_candidate in column_lookup:
            return column_lookup[normalized_candidate]
    return None


def get_player_comparison_meta_columns(df: pd.DataFrame, column_lookup: Dict[str, str]) -> Dict[str, str]:
    candidate_map = {
        "Player Name": ["Player Name", "player_name", "PLAYER_NAME", "name"],
        "Season": ["Season", "season", "SEASON"],
        "teams_played": ["teams_played", "team", "TEAM", "Team", "team_abbreviation", "TEAM_ABBREVIATION"],
        "position": ["position", "Position", "POSITION", "pos", "POS"],
    }
    resolved: Dict[str, str] = {}
    for output_name, candidates in candidate_map.items():
        for candidate in candidates:
            if candidate in df.columns:
                resolved[output_name] = candidate
                break
            normalized_candidate = normalize_comparison_column_name(candidate)
            if normalized_candidate in column_lookup:
                resolved[output_name] = column_lookup[normalized_candidate]
                break
    missing = [name for name in ["Player Name", "Season"] if name not in resolved]
    if missing:
        raise ValueError(f"Player comparison CSV is missing required columns: {missing}")
    return resolved


def comparison_csv_candidate_paths(configured_path: str, default_filename: str) -> List[Path]:
    configured = Path(configured_path).expanduser()
    candidates = [
        configured,
        Path.cwd() / configured.name,
        BACKEND_DIR.parent / configured.name,
        BACKEND_DIR / configured.name,
        BACKEND_DATA_DIR / configured.name,
        Path("/Users/harsha/Desktop/PickPocketProjectOfficial") / configured.name,
        Path("/Users/harsha/Desktop") / configured.name,
    ]
    if configured.name != default_filename:
        candidates.extend(
            [
                BACKEND_DIR.parent / default_filename,
                BACKEND_DIR / default_filename,
                BACKEND_DATA_DIR / default_filename,
                Path("/Users/harsha/Desktop/PickPocketProjectOfficial") / default_filename,
                Path("/Users/harsha/Desktop") / default_filename,
            ]
        )
    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = str(candidate.expanduser())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(candidate.expanduser())
    return deduped


def find_readable_csv_path(configured_path: str, default_filename: str, label: str) -> Path:
    candidates = comparison_csv_candidate_paths(configured_path, default_filename)
    existing_files = [candidate for candidate in candidates if candidate.exists() and candidate.is_file()]
    non_empty_files = [candidate for candidate in existing_files if candidate.stat().st_size > 0]
    if non_empty_files:
        return non_empty_files[0]
    if existing_files:
        details = ", ".join(f"{candidate} ({candidate.stat().st_size} bytes)" for candidate in existing_files[:5])
        raise ValueError(
            f"{label} CSV was found but is empty: {details}. "
            "Re-export or replace the CSV, then restart the backend."
        )
    raise FileNotFoundError(
        f"{label} CSV not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates[:8])
    )


def read_csv_robust(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError(f"{label} CSV is empty: {path} (0 bytes)")

    attempts = [
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "latin1"},
        {"encoding": "latin1", "sep": None, "engine": "python"},
    ]
    errors: List[str] = []
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df is not None and len(df.columns) > 0:
                return df
            errors.append(f"{kwargs}: parsed zero columns")
        except pd.errors.EmptyDataError:
            errors.append(f"{kwargs}: no columns parsed")
        except UnicodeDecodeError as exc:
            errors.append(f"{kwargs}: unicode decode error: {exc}")
        except Exception as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")

    try:
        preview = path.read_bytes()[:180].decode("utf-8", errors="replace").replace("\n", "\\n")
    except Exception:
        preview = "<unable to read preview>"
    raise ValueError(
        f"Could not parse {label} CSV at {path} ({file_size} bytes). "
        f"Attempts: {' | '.join(errors)}. File starts with: {preview}"
    )


def load_player_comps_dataframe() -> tuple[pd.DataFrame, Dict[str, str], Path]:
    path = find_readable_csv_path(
        PLAYER_COMPS_DATASET_PATH,
        "fullseasonfeatures_player_comps_real.csv",
        "Player comparison",
    )

    mtime_ns = path.stat().st_mtime_ns
    if (
        _PLAYER_COMPS_DATA_CACHE.get("path") == str(path)
        and _PLAYER_COMPS_DATA_CACHE.get("mtime_ns") == mtime_ns
        and _PLAYER_COMPS_DATA_CACHE.get("dataframe") is not None
    ):
        return (
            _PLAYER_COMPS_DATA_CACHE["dataframe"],
            _PLAYER_COMPS_DATA_CACHE["column_map"],
            path,
        )

    df = read_csv_robust(path, "Player comparison")
    column_lookup = build_comparison_column_lookup(df)
    meta_columns = get_player_comparison_meta_columns(df, column_lookup)

    normalized_df = df.copy()
    for output_name, source_name in meta_columns.items():
        if output_name not in normalized_df.columns:
            normalized_df[output_name] = normalized_df[source_name]

    if "teams_played" not in normalized_df.columns:
        normalized_df["teams_played"] = ""
    if "position" not in normalized_df.columns:
        normalized_df["position"] = ""

    if "position" in normalized_df.columns:
        position_values = normalized_df["position"].astype(str).str.upper().str.strip()
        guard_mask = position_values.isin(["PG", "SG"])
        if guard_mask.any():
            normalized_df = normalized_df.loc[guard_mask].copy()

    normalized_df["Player Name"] = normalized_df["Player Name"].astype(str)
    normalized_df["Season"] = normalized_df["Season"].astype(str)
    normalized_df["teams_played"] = normalized_df["teams_played"].fillna("").astype(str)
    normalized_df["position"] = normalized_df["position"].fillna("").astype(str)
    normalized_df["player_key"] = normalized_df.apply(stable_player_key, axis=1)
    normalized_df = normalized_df.drop_duplicates(subset=["player_key"], keep="first").reset_index(drop=True)
    column_lookup = build_comparison_column_lookup(normalized_df)

    _PLAYER_COMPS_DATA_CACHE.update(
        {"path": str(path), "mtime_ns": mtime_ns, "dataframe": normalized_df, "column_map": column_lookup}
    )
    return normalized_df, column_lookup, path


def load_player_comps_pace_lookup() -> tuple[Dict[str, float], Optional[float], Optional[Path]]:
    try:
        path = find_readable_csv_path(
            PLAYER_COMPS_PACE_PATH,
            "league_average_pace_2016_17_to_2025_26.csv",
            "League-average pace",
        )
    except (FileNotFoundError, ValueError):
        return {}, None, None

    mtime_ns = path.stat().st_mtime_ns
    if (
        _PLAYER_COMPS_PACE_CACHE.get("path") == str(path)
        and _PLAYER_COMPS_PACE_CACHE.get("mtime_ns") == mtime_ns
        and _PLAYER_COMPS_PACE_CACHE.get("pace_by_season") is not None
    ):
        return (
            _PLAYER_COMPS_PACE_CACHE["pace_by_season"],
            _PLAYER_COMPS_PACE_CACHE["target_pace"],
            path,
        )

    try:
        pace_df = read_csv_robust(path, "League-average pace")
    except ValueError:
        return {}, None, path
    pace_column = None
    pace_column_lookup = build_comparison_column_lookup(pace_df)
    for candidate in ["league_avg_pace_weighted_by_minutes", "league_avg_pace_simple_mean", "PACE", "pace"]:
        if candidate in pace_df.columns:
            pace_column = candidate
            break
        normalized_candidate = normalize_comparison_column_name(candidate)
        if normalized_candidate in pace_column_lookup:
            pace_column = pace_column_lookup[normalized_candidate]
            break
    season_column = "season" if "season" in pace_df.columns else pace_column_lookup.get("season")
    if pace_column is None or season_column is None:
        return {}, None, path

    pace_by_season: Dict[str, float] = {}
    for _, row in pace_df.iterrows():
        season = str(row.get(season_column, ""))
        pace_value = pd.to_numeric(pd.Series([row.get(pace_column)]), errors="coerce").iloc[0]
        if season and pd.notna(pace_value) and float(pace_value) > 0:
            pace_by_season[season] = float(pace_value)

    target_pace = None
    if pace_by_season:
        sorted_seasons = sorted(pace_by_season.keys())
        if PLAYER_COMPS_TARGET_PACE_MODE == "mean":
            target_pace = float(np.mean(list(pace_by_season.values())))
        else:
            target_pace = pace_by_season[sorted_seasons[-1]]

    _PLAYER_COMPS_PACE_CACHE.update(
        {"path": str(path), "mtime_ns": mtime_ns, "pace_by_season": pace_by_season, "target_pace": target_pace}
    )
    return pace_by_season, target_pace, path



def player_comparison_identity_key(player_name: object, season: object) -> str:
    return f"{normalize_player_name_for_assignment_key(player_name)}||{str(season)}"


def load_player_comps_assignment_lookup() -> Dict[str, Dict[str, object]]:
    """Map player-season rows to the locked archetype assignment used by the galaxy."""
    assignment_path = BACKEND_DATA_DIR / "euclidean_kmeans_locked_assignments.csv"
    if not assignment_path.exists():
        return {}

    mtime_ns = assignment_path.stat().st_mtime_ns
    if (
        _PLAYER_COMPS_ASSIGNMENT_CACHE.get("path") == str(assignment_path)
        and _PLAYER_COMPS_ASSIGNMENT_CACHE.get("mtime_ns") == mtime_ns
        and _PLAYER_COMPS_ASSIGNMENT_CACHE.get("lookup") is not None
    ):
        return _PLAYER_COMPS_ASSIGNMENT_CACHE["lookup"]

    try:
        assignment_frame = pd.read_csv(assignment_path)
    except Exception:
        return {}

    lookup: Dict[str, Dict[str, object]] = {}
    for _, row in assignment_frame.iterrows():
        player_name = str(row.get("Player Name", ""))
        season = str(row.get("Season", ""))
        if not player_name or not season:
            continue
        archetype_name = str(
            row.get("archetype_name")
            or row.get("cluster_label")
            or row.get("cluster_name")
            or ""
        )
        cluster_number = _safe_float(row.get("cluster"), None)
        payload = {
            "archetype_name": archetype_name,
            "cluster_name": archetype_name,
            "cluster_number": int(cluster_number) if cluster_number is not None and pd.notna(cluster_number) else None,
        }
        lookup[player_comparison_identity_key(player_name, season)] = payload

    _PLAYER_COMPS_ASSIGNMENT_CACHE.update({"path": str(assignment_path), "mtime_ns": mtime_ns, "lookup": lookup})
    return lookup


def _badge_payload_from_row(badge_row: pd.Series, badge_frame_columns: set[str]) -> Dict[str, object]:
    components = {}
    demotion_reasons = []
    component_json = badge_row.get("component_percentiles_json", "")
    demotion_json = badge_row.get("demotion_reasons_json", "")
    try:
        if isinstance(component_json, str) and component_json.strip():
            parsed_components = json.loads(component_json)
            if isinstance(parsed_components, dict):
                components = parsed_components
    except Exception:
        components = {}
    try:
        if isinstance(demotion_json, str) and demotion_json.strip():
            parsed_reasons = json.loads(demotion_json)
            if isinstance(parsed_reasons, list):
                demotion_reasons = parsed_reasons
    except Exception:
        demotion_reasons = []

    return {
        "id": str(badge_row.get("badge_id", "")),
        "name": str(badge_row.get("badge_name", "")),
        "tier": str(badge_row.get("badge_tier", "")),
        "category": str(badge_row.get("badge_category", "")) if "badge_category" in badge_frame_columns else "",
        "score_percentile": _safe_float(badge_row.get("badge_score_percentile"), 0.0),
        "rarity_percent": _safe_float(badge_row.get("rarity_percent"), 0.0),
        "rarity_label": str(badge_row.get("rarity_label", "")),
        "components": components,
        "demotion_reasons": demotion_reasons,
    }


def load_player_comps_badge_lookups() -> tuple[Dict[str, List[Dict[str, object]]], Dict[str, List[Dict[str, object]]]]:
    """Load precomputed badges for Player Comps cards without recomputing badges live."""
    badge_path = BACKEND_DATA_DIR / "player_badges.csv"
    if not badge_path.exists():
        return {}, {}

    mtime_ns = badge_path.stat().st_mtime_ns
    if (
        _PLAYER_COMPS_BADGE_CACHE.get("path") == str(badge_path)
        and _PLAYER_COMPS_BADGE_CACHE.get("mtime_ns") == mtime_ns
        and _PLAYER_COMPS_BADGE_CACHE.get("by_player_key") is not None
        and _PLAYER_COMPS_BADGE_CACHE.get("by_identity_key") is not None
    ):
        return _PLAYER_COMPS_BADGE_CACHE["by_player_key"], _PLAYER_COMPS_BADGE_CACHE["by_identity_key"]

    try:
        badge_frame = pd.read_csv(badge_path)
    except Exception:
        return {}, {}

    required_columns = {"player_key", "badge_id", "badge_name", "badge_tier", "badge_score_percentile"}
    if not required_columns.issubset(set(badge_frame.columns)):
        return {}, {}
    badge_frame = attach_badge_rarity_columns(badge_frame)

    badge_frame_columns = set(str(column) for column in badge_frame.columns)
    by_player_key: Dict[str, List[Dict[str, object]]] = {}
    by_identity_key: Dict[str, List[Dict[str, object]]] = {}
    for _, badge_row in badge_frame.iterrows():
        badge_payload = _badge_payload_from_row(badge_row, badge_frame_columns)
        player_key = str(badge_row.get("player_key", ""))
        if player_key:
            by_player_key.setdefault(player_key, []).append(badge_payload)

        player_name = str(badge_row.get("Player Name", ""))
        season = str(badge_row.get("Season", ""))
        if player_name and season:
            identity_key = player_comparison_identity_key(player_name, season)
            by_identity_key.setdefault(identity_key, []).append(badge_payload)

    _PLAYER_COMPS_BADGE_CACHE.update(
        {
            "path": str(badge_path),
            "mtime_ns": mtime_ns,
            "by_player_key": by_player_key,
            "by_identity_key": by_identity_key,
        }
    )
    return by_player_key, by_identity_key


def build_player_comparison_player_payload(
    row: pd.Series,
    assignment_lookup: Dict[str, Dict[str, object]],
    badges_by_player_key: Dict[str, List[Dict[str, object]]],
    badges_by_identity_key: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    player_name = str(row.get("Player Name", ""))
    season = str(row.get("Season", ""))
    player_key = str(row.get("player_key", ""))
    identity_key = player_comparison_identity_key(player_name, season)
    assignment_payload = assignment_lookup.get(identity_key, {})
    badges = badges_by_player_key.get(player_key) or badges_by_identity_key.get(identity_key) or []
    return {
        "player_key": player_key,
        "player_name": player_name,
        "season": season,
        "team": str(row.get("teams_played", "")),
        "position": str(row.get("position", "")),
        "archetype_name": str(assignment_payload.get("archetype_name") or ""),
        "cluster_name": str(assignment_payload.get("cluster_name") or ""),
        "cluster_number": assignment_payload.get("cluster_number"),
        "badges": badges,
        **get_player_headshot_payload(player_name),
    }


def all_player_comparison_features() -> List[Dict[str, object]]:
    features: List[Dict[str, object]] = []
    seen_keys = set()
    for category_items in PLAYER_COMPARISON_FEATURES.values():
        for feature_meta in category_items:
            feature_key = str(feature_meta.get("feature", ""))
            dedupe_key = (feature_key, str(feature_meta.get("label", "")))
            if not feature_key or dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            features.append(feature_meta)
    return features


def get_comparison_numeric_series(df: pd.DataFrame, column_name: Optional[str]) -> pd.Series:
    if not column_name or column_name not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column_name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def should_show_comparison_feature(feature_meta: Dict[str, object], mode: str) -> bool:
    feature_name = str(feature_meta.get("feature", ""))
    kind = str(feature_meta.get("kind", "value"))
    lower_feature_name = feature_name.lower()
    is_frequency = kind == "frequency" or "frequency" in lower_feature_name
    is_fga_count = (
        kind == "volume"
        and any(token in lower_feature_name for token in ["fga", "fg3a", "fg2a", "2pa", "3pa"])
        and not is_frequency
    )

    if mode in {"same_season_percentile", "all_seasons_percentile"}:
        return True
    if mode in {"raw_stats", "pace_adjusted_raw_stats"}:
        return not is_frequency
    if mode in {"raw_frequencies", "raw_per_75"}:
        return not is_fga_count
    return True


def get_player_comparison_value(
    raw_value: float,
    feature_meta: Dict[str, object],
    mode: str,
    season: str,
    pace_by_season: Dict[str, float],
    target_pace: Optional[float],
) -> Optional[float]:
    if raw_value is None or pd.isna(raw_value):
        return None

    value = float(raw_value)
    kind = str(feature_meta.get("kind", "value"))
    feature_name = str(feature_meta.get("feature", ""))
    lower_feature_name = feature_name.lower()
    is_frequency = kind == "frequency" or "frequency" in lower_feature_name
    is_volume = kind == "volume"

    pace_factor = 1.0
    season_pace = pace_by_season.get(str(season))
    if target_pace and season_pace and season_pace > 0:
        pace_factor = float(target_pace) / float(season_pace)

    if mode == "pace_adjusted_raw_stats" and is_volume:
        value *= pace_factor
    elif mode == "raw_per_75" and is_frequency:
        value *= 75.0

    return value


def format_comparison_value(value: Optional[float], feature_meta: Dict[str, object], mode: str) -> str:
    if value is None or pd.isna(value):
        return "—"

    kind = str(feature_meta.get("kind", "value"))
    numeric_value = float(value)

    if mode in {"same_season_percentile", "all_seasons_percentile"}:
        return f"{numeric_value:.0f}"
    if kind == "frequency":
        if mode in {"same_season_percentile", "all_seasons_percentile"}:
            return f"{numeric_value:.0f}"
        formatted = f"{numeric_value:.1f}" if mode == "raw_per_75" else f"{numeric_value:.3f}"
        if formatted.startswith("0"):
            formatted = formatted[1:]
        if formatted.startswith("-0"):
            formatted = "-" + formatted[2:]
        return f"{formatted}%"
    if kind == "percentage":
        if abs(numeric_value) <= 1.5:
            formatted = f"{numeric_value:.3f}"
            if formatted.startswith("0"):
                formatted = formatted[1:]
            if formatted.startswith("-0"):
                formatted = "-" + formatted[2:]
            return formatted
        return f"{numeric_value:.1f}%"
    if kind in {"ratio", "rating"}:
        return f"{numeric_value:.2f}"
    if kind == "distance":
        return f"{numeric_value:.1f}"
    return f"{numeric_value:.1f}"


def percentile_rank(series: pd.Series, value: float, lower_is_better: bool = False) -> Optional[float]:
    valid_series = pd.to_numeric(series, errors="coerce").dropna()
    if valid_series.empty or value is None or pd.isna(value):
        return None
    if lower_is_better:
        percentile = (valid_series >= float(value)).mean() * 100.0
    else:
        percentile = (valid_series <= float(value)).mean() * 100.0
    return float(np.clip(percentile, 0.0, 100.0))


def build_player_comparison_options_payload() -> Dict[str, object]:
    df, column_lookup, csv_path = load_player_comps_dataframe()
    assignment_lookup = load_player_comps_assignment_lookup()
    badges_by_player_key, badges_by_identity_key = load_player_comps_badge_lookups()
    players = []
    sorted_df = df.sort_values(["Player Name", "Season", "teams_played", "position"], kind="stable")
    for _, row in sorted_df.iterrows():
        players.append(
            build_player_comparison_player_payload(
                row=row,
                assignment_lookup=assignment_lookup,
                badges_by_player_key=badges_by_player_key,
                badges_by_identity_key=badges_by_identity_key,
            )
        )

    available_features = {}
    missing_features = {}
    for category, features in PLAYER_COMPARISON_FEATURES.items():
        available_features[category] = []
        missing_features[category] = []
        for feature_meta in features:
            resolved_column = resolve_comparison_column(df, column_lookup, feature_meta)
            payload = {key: value for key, value in feature_meta.items() if key != "aliases"}
            payload["lower_is_better"] = str(feature_meta.get("feature")) in PLAYER_COMPS_LOWER_IS_BETTER_FEATURES
            payload["resolved_column"] = resolved_column
            if resolved_column:
                available_features[category].append(payload)
            else:
                missing_features[category].append(payload)

    pace_by_season, target_pace, pace_path = load_player_comps_pace_lookup()
    return {
        "source_file": str(csv_path),
        "pace_file": str(pace_path) if pace_path else None,
        "target_pace": target_pace,
        "players": players,
        "categories": PLAYER_COMPARISON_CATEGORIES,
        "modes": PLAYER_COMPARISON_MODES,
        "available_features": available_features,
        "missing_features": missing_features,
        "lower_is_better_features": sorted(PLAYER_COMPS_LOWER_IS_BETTER_FEATURES),
    }


def find_player_comparison_row(df: pd.DataFrame, player_key: str) -> pd.Series:
    matching_rows = df.loc[df["player_key"].astype(str) == str(player_key)]
    if matching_rows.empty:
        raise KeyError(f"Player comparison row not found for key: {player_key}")
    return matching_rows.iloc[0]


def build_player_comparison_payload(req: PlayerComparisonRequest) -> Dict[str, object]:
    category = req.category if req.category in PLAYER_COMPARISON_FEATURES else "traditional"
    allowed_modes = {item["value"] for item in PLAYER_COMPARISON_MODES}
    mode = req.mode if req.mode in allowed_modes else "raw_stats"

    df, column_lookup, csv_path = load_player_comps_dataframe()
    pace_by_season, target_pace, pace_path = load_player_comps_pace_lookup()

    selected_rows = [find_player_comparison_row(df, player_key) for player_key in req.player_keys]
    assignment_lookup = load_player_comps_assignment_lookup()
    badges_by_player_key, badges_by_identity_key = load_player_comps_badge_lookups()
    selected_players = [
        build_player_comparison_player_payload(
            row=row,
            assignment_lookup=assignment_lookup,
            badges_by_player_key=badges_by_player_key,
            badges_by_identity_key=badges_by_identity_key,
        )
        for row in selected_rows
    ]

    rows = []
    missing_features = []
    for feature_meta in PLAYER_COMPARISON_FEATURES[category]:
        if not should_show_comparison_feature(feature_meta, mode):
            continue

        feature_name = str(feature_meta.get("feature", ""))
        resolved_column = resolve_comparison_column(df, column_lookup, feature_meta)
        if not resolved_column:
            missing_features.append(feature_name)
            continue

        numeric_series = get_comparison_numeric_series(df, resolved_column)
        lower_is_better = feature_name in PLAYER_COMPS_LOWER_IS_BETTER_FEATURES
        values = []
        display_values = []
        raw_values = []
        for selected_row in selected_rows:
            raw_value = pd.to_numeric(pd.Series([selected_row.get(resolved_column)]), errors="coerce").iloc[0]
            raw_value = None if pd.isna(raw_value) else float(raw_value)
            raw_values.append(raw_value)
            if mode == "same_season_percentile":
                season_mask = df["Season"].astype(str) == str(selected_row.get("Season", ""))
                value = percentile_rank(numeric_series.loc[season_mask], raw_value, lower_is_better=lower_is_better)
            elif mode == "all_seasons_percentile":
                value = percentile_rank(numeric_series, raw_value, lower_is_better=lower_is_better)
            else:
                value = get_player_comparison_value(
                    raw_value=raw_value,
                    feature_meta=feature_meta,
                    mode=mode,
                    season=str(selected_row.get("Season", "")),
                    pace_by_season=pace_by_season,
                    target_pace=target_pace,
                )
            values.append(value)
            display_values.append(format_comparison_value(value, feature_meta, mode))

        winner = None
        if len(values) == 2 and values[0] is not None and values[1] is not None:
            first_value = float(values[0])
            second_value = float(values[1])
            if not np.isclose(first_value, second_value):
                if mode in {"same_season_percentile", "all_seasons_percentile"}:
                    winner = "left" if first_value > second_value else "right"
                elif lower_is_better:
                    winner = "left" if first_value < second_value else "right"
                else:
                    winner = "left" if first_value > second_value else "right"

        rows.append(
            {
                "feature": feature_name,
                "resolved_column": resolved_column,
                "label": str(feature_meta.get("label", feature_name)),
                "kind": str(feature_meta.get("kind", "value")),
                "lower_is_better": lower_is_better,
                "values": values,
                "display_values": display_values,
                "winner": winner,
            }
        )

    return {
        "source_file": str(csv_path),
        "pace_file": str(pace_path) if pace_path else None,
        "target_pace": target_pace,
        "category": category,
        "mode": mode,
        "players": selected_players,
        "rows": rows,
        "missing_features": missing_features,
        "categories": PLAYER_COMPARISON_CATEGORIES,
        "modes": PLAYER_COMPARISON_MODES,
    }


@app.get("/api/similar-players")
def similar_players(
    player_name: str,
    season: str,
    pipeline: Optional[str] = None,
    k: Optional[int] = None,
    pca_variance_target: Optional[str] = None,
):
    try:
        dataframe, csv_path = load_similar_players_dataframe()
    except FileNotFoundError:
        return build_similar_players_response_from_galaxy(player_name=player_name, season=season)
    except ValueError:
        return build_similar_players_response_from_galaxy(player_name=player_name, season=season)

    expected_signature = build_locked_euclidean_feature_signature()
    if "feature_signature" not in dataframe.columns:
        return build_similar_players_response_from_galaxy(player_name=player_name, season=season)

    dataframe = dataframe.loc[dataframe["feature_signature"].astype(str) == expected_signature].copy()
    if dataframe.empty:
        return build_similar_players_response_from_galaxy(player_name=player_name, season=season)

    candidate_rows = filter_similar_players_dataframe(
        dataframe=dataframe,
        player_name=player_name,
        season=season,
        pipeline=pipeline,
        k=k,
        pca_variance_target=pca_variance_target,
    )

    if candidate_rows.empty:
        return build_similar_players_response_from_galaxy(player_name=player_name, season=season)

    if not similar_players_rows_have_detail_payload(candidate_rows):
        return build_similar_players_response_from_galaxy(player_name=player_name, season=season)

    candidate_rows["_rank_numeric"] = pd.to_numeric(candidate_rows["rank"], errors="coerce")
    candidate_rows["_score_numeric"] = pd.to_numeric(candidate_rows.get("overall_similarity_score", 0.0), errors="coerce")
    candidate_rows = candidate_rows.sort_values(
        ["_rank_numeric", "_score_numeric"],
        ascending=[True, False],
        kind="stable",
    ).head(5)

    source_row = candidate_rows.iloc[0]
    source_cluster_number = row_int(
        source_row,
        "source_cluster_number",
        row_int(source_row, "cluster_number", row_int(source_row, "cluster", 0)),
    )
    source_cluster_raw = row_int(
        source_row,
        "source_cluster_raw",
        row_int(source_row, "cluster", source_cluster_number - 1),
    )
    source_cluster_name = (
        row_string(source_row, "source_archetype_name")
        or row_string(source_row, "archetype_name")
        or EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER.get(source_cluster_number, f"Cluster {source_cluster_number}")
    )

    similar_player_items = []
    for _, row in candidate_rows.iterrows():
        related_cluster_number = row_int(
            row,
            "related_cluster_number",
            row_int(row, "related_cluster_number_stable", row_int(row, "related_cluster", 0)),
        )
        related_cluster_raw = row_int(
            row,
            "related_cluster_raw",
            row_int(row, "related_cluster", related_cluster_number - 1),
        )
        related_cluster_name = (
            row_string(row, "related_archetype_name")
            or EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER.get(related_cluster_number, f"Cluster {related_cluster_number}")
        )

        similar_player_items.append(
            {
                "rank": row_int(row, "rank"),
                "player_name": row_string(row, "related_player_name"),
                "season": row_string(row, "related_season"),
                "team": row_string(row, "related_team"),
                "position": row_string(row, "related_position"),
                "player_season_id": row_string(row, "related_player_season_id"),
                "cluster_raw": related_cluster_raw,
                "cluster_number": related_cluster_number,
                "cluster_name": related_cluster_name,
                "archetype_name": related_cluster_name,
                **get_player_headshot_payload(row_string(row, "related_player_name")),
                "similarity_score": row_float(row, "overall_similarity_score"),
                "overall_distance": row_float(row, "overall_distance"),
                "similarity_distance_metric": row_string(row, "similarity_distance_metric", EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC),
                "similarity_metric_used": row_string(row, "similarity_metric_used", row_string(row, "similarity_distance_metric", EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC)),
                "same_cluster": parse_bool_value(row["same_cluster"]) if "same_cluster" in row.index else False,
                "same_archetype": parse_bool_value(row["same_archetype"]) if "same_archetype" in row.index else False,
                "strongest_similarity_blocks": row_string(row, "strongest_similarity_blocks"),
                "biggest_difference_blocks": row_string(row, "biggest_difference_blocks"),
                "block_scores": build_block_scores(row),
            }
        )

    return {
        "source_file": str(csv_path),
        "player_name": row_string(source_row, "player_name", player_name),
        "season": row_string(source_row, "season", season),
        "team": row_string(source_row, "team"),
        "position": row_string(source_row, "position"),
        "player_season_id": row_string(source_row, "source_player_season_id"),
        "cluster_raw": source_cluster_raw,
        "cluster_number": source_cluster_number,
        "cluster": source_cluster_name,
        "cluster_name": source_cluster_name,
        "archetype_name": source_cluster_name,
        **get_player_headshot_payload(row_string(source_row, "player_name", player_name)),
        "similar_players": similar_player_items,
    }


@app.get("/api/player-comparison-options")
def player_comparison_options():
    try:
        return build_player_comparison_options_payload()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/player-comparison")
def player_comparison(req: PlayerComparisonRequest):
    try:
        return build_player_comparison_payload(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/config")
def get_config():
    return {
        "dataset_path": DEFAULT_DATASET_PATH,
        "allowed_features": ALLOWED_FEATURES,
        "default_features": get_locked_euclidean_kmeans_feature_columns(raw=False),
        "euclidean_kmeans_locked_mode": True,
        "euclidean_kmeans_locked_k": EUCLIDEAN_KMEANS_LOCKED_K,
        "euclidean_kmeans_locked_features": get_locked_euclidean_kmeans_feature_columns(raw=False),
        "euclidean_kmeans_locked_features_raw": get_locked_euclidean_kmeans_feature_columns(raw=True),
        "euclidean_kmeans_locked_feature_groups": EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES,
        "euclidean_kmeans_locked_group_weights": EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS,
        "euclidean_kmeans_locked_pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
        "euclidean_kmeans_locked_space_transform": EUCLIDEAN_KMEANS_LOCKED_SPACE_TRANSFORM,
        "euclidean_kmeans_locked_similarity_distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
        "euclidean_kmeans_locked_luka_euclidean_override_metric": LUKA_DONCIC_SIMILARITY_DISTANCE_METRIC,
        "galaxy_similar_player_count": GALAXY_SIMILAR_PLAYER_COUNT,
        "galaxy_cluster_knn_count": GALAXY_CLUSTER_KNN_COUNT,
        "euclidean_kmeans_cluster_name_by_number": EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER,
        "euclidean_kmeans_cluster_description_by_number": EUCLIDEAN_KMEANS_CLUSTER_DESCRIPTION_BY_NUMBER,
        "cluster_name_by_number": CLUSTER_NAME_BY_NUMBER,
        "default_algorithm": DEFAULT_ALGORITHM,
        "allowed_distance_metrics": list(ALLOWED_DISTANCE_METRICS),
        "default_distance_metric": DEFAULT_DISTANCE_METRIC,
        "default_k": DEFAULT_KMEANS_K,
        "default_kmeans_k": DEFAULT_KMEANS_K,
        "allowed_algorithms": ["kmeans"],
        "pca_explained_var_target": PCA_EXPLAINED_VAR_TARGET,
        "lower_is_better_percentile_features": sorted(LOWER_IS_BETTER_PERCENTILE_FEATURES),
        "skill_breakdown_group_order": SKILL_BREAKDOWN_GROUP_ORDER,
        "skill_breakdown_group_features": SKILL_BREAKDOWN_GROUP_FEATURES,
        "skill_breakdown_excluded_features": sorted(SKILL_BREAKDOWN_EXCLUDED_FEATURES),
        "three_pt_breakdown_group_order": THREE_PT_BREAKDOWN_GROUP_ORDER,
        "three_pt_breakdown_group_features": THREE_PT_BREAKDOWN_GROUP_FEATURES,
        "three_pt_breakdown_lower_is_better_by_group": {
            group_name: sorted(list(feature_names))
            for group_name, feature_names in THREE_PT_BREAKDOWN_LOWER_IS_BETTER_BY_GROUP.items()
        },
        "style_tokens": {
            "bg": "#02090B",
            "panel": "#081316",
            "panel_alt": "#0A1518",
            "line": "#0F3B43",
            "text": "#DFF3F4",
            "muted": "#7FA5AA",
            "accent": "#00D4E0",
            "gold": "#F7CD67",
            "purple": "#B77AFE",
        },
    }


@app.post("/api/cluster")
def cluster(req: ClusterRequest):
    try:
        return compute_cluster_payload(
            DEFAULT_DATASET_PATH,
            req.algorithm,
            req.distance_metric,
            req.k,
            req.features,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/cluster-report")
def cluster_report(req: ClusterReportRequest):
    try:
        return compute_cluster_report_payload(
            DEFAULT_DATASET_PATH,
            req.algorithm,
            req.distance_metric,
            req.k,
            req.features,
            req.cluster_number,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/api/player-skill-breakdown")
def player_skill_breakdown(req: PlayerSkillBreakdownRequest):
    try:
        return build_skill_breakdown_payload(
            DEFAULT_DATASET_PATH,
            req.algorithm,
            req.distance_metric,
            req.k,
            req.features,
            req.player_key,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/player-three-pt-breakdown")
def player_three_pt_breakdown(req: PlayerSkillBreakdownRequest):
    try:
        return build_three_pt_breakdown_payload(
            DEFAULT_DATASET_PATH,
            req.algorithm,
            req.distance_metric,
            req.k,
            req.features,
            req.player_key,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def build_player_detail_payload(player_key: str) -> Dict[str, object]:
    try:
        dataset_meta = load_base_dataframe(DEFAULT_DATASET_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    details = dataset_meta["stats_lookup"].get(player_key)
    if details is None:
        raise HTTPException(status_code=404, detail="Player row not found.")

    player_comps_items = build_player_comps_feature_percentile_items(player_key, details["meta"])
    percentile_items = sorted(
        player_comps_items if player_comps_items is not None else [
            {
                "feature": feature,
                "label": humanize_player_comps_feature_label(feature),
                "value": details["stats"][feature],
                "percentile": details["percentiles"][feature],
                "lower_is_better": feature in LOWER_IS_BETTER_PERCENTILE_FEATURES,
            }
            for feature in ALLOWED_FEATURES
        ],
        key=lambda x: x["percentile"],
        reverse=True,
    )

    top_features = percentile_items[:5]
    bottom_features = list(reversed(percentile_items[-5:]))

    return {
        "meta": details["meta"],
        "stats": percentile_items,
        "top_features": top_features,
        "bottom_features": bottom_features,
        "badges": details.get("badges", []),
    }


@app.post("/api/player-details")
def player_details(req: PlayerDetailRequest):
    return build_player_detail_payload(req.player_key)
