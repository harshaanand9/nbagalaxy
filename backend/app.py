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


APP_VERSION = "0.26.0"
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
    "zero_to_one_drib_3PA_frequency",
    "zero_to_one_drib_3PA_accuracy",
    "three_to_six_drib_3FGA_frequency",
    "three_to_six_drib_3FGA_accuracy",
    "seven_plus_drib_3FGA_frequency",
    "seven_plus_drib_3FGA_accuracy",
    "box_creation",
    "offensive_load",
    "pts_created_from_assists_per_75",
    "potential_assists_and_ft_assists_per_75",
    "touches_per_75",
    "passes_received_per_75",
    "PTS_PER_100_ON_OFF",
    "p_r_ball_handler_frequency",
    "p_r_ball_handler_ppp",
    "iso_frequency",
    "isolation_ppp",
    "hand_off_frequency",
    "hand_off_ppp",
    "cut_frequency",
    "cut_ppp",
    "off_screen_frequency",
    "off_screen_ppp",
    "spot_up_frequency",
    "spot_up_ppp",
    "opp_players_fg_pct_difference_adjusted",
    "D-LEBRON_adjusted",
]

EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES = {
    "ThreePT": [
        "zero_to_one_drib_3PA_frequency",
        "zero_to_one_drib_3PA_accuracy",
        "three_to_six_drib_3FGA_frequency",
        "three_to_six_drib_3FGA_accuracy",
        "seven_plus_drib_3FGA_frequency",
        "seven_plus_drib_3FGA_accuracy",
        "3fga_frequency",
        "3P_Accuracy",
        "tight_very_tight_3fga_frequency",
        "tight_very_tight_3fga_accuracy",
        "open_3fga_frequency",
        "open_3fga_accuracy",
        "Wide_Open_3FGA_Frequency",
        "Wide_Open_3FG_PCT",
        "Avg3ptShotDistance",
        "pts_from_3s_per_75",
        "pct_fga_3FGA",
        "pct_3p_fg_assisted",
    ],
    "MidRange": [
        "MidRangeFrequency",
        "MidRangeAccuracy",
        "tight_very_tight_2fga_frequency",
        "tight_very_tight_2fga_accuracy",
        "open_2fga_frequency",
        "open_2fga_accuracy",
        "Avg2ptShotDistance",
        "pct_fga_MR",
        "pts_from_midrange_per_75",
        "pct_2p_fg_assisted",
    ],
    "RimPressure": [
        "RestrictedArea_Frequency",
        "RestrictedArea_Accuracy",
        "Paint_Non_RA_Frequency",
        "Paint_Non_RA_Accuracy",
        "drive_fga_frequency",
        "drive_fg_pct",
        "dunks_per_75_poss",
        "pts_from_drives_per_75",
        "pct_fga_drive_fga",
        "drive_fta_rate",
    ],
    "Playmaking": [
        "box_creation",
        "crafted_passer_rating",
        "offensive_load",
        "pts_created_from_assists_per_75",
        "potential_assists_and_ft_assists_per_75",
        "assist_frequency",
        "ASSISTS_ON_OFF",
        "EFG_PCT_ON_OFF",
        "potential_assist_tov_ratio",
        "PTS_PER_100_ON_OFF",
    ],
    "Defense": [
        "Opp_players_fga_per_75_poss",
        "contested_shot_frequency",
        "off_fouls_drawn_frequency",
        "Blocks_per_75",
        "Deflections_per_75",
        "opp_players_fg_pct_difference_adjusted",
        "D-LEBRON_adjusted",
    ],
    "Playtypes": [
        "avg_drib_per_touch",
        "avg_sec_per_touch",
        "touches_per_75",
        "passes_received_per_75",
        "p_r_ball_handler_frequency",
        "p_r_ball_handler_ppp",
        "iso_frequency",
        "isolation_ppp",
        "hand_off_frequency",
        "hand_off_ppp",
        "cut_frequency",
        "cut_ppp",
        "off_screen_frequency",
        "off_screen_ppp",
        "spot_up_frequency",
        "spot_up_ppp",
        "pull_up_3P_frequency",
        "pull_up_3P_accuracy",
        "catch_shoot_3P_frequency",
        "catch_shoot_3P_accuracy",
        "pull_up_2P_frequency",
        "pull_up_2P_accuracy",
    ],
}
_PLAYTYPES_WEIGHT     = 0.25
_NON_PLAYTYPES_WEIGHT = (1.0 - _PLAYTYPES_WEIGHT) / 5  # 5 non-Playtypes groups
EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS = {
    "ThreePT":     _NON_PLAYTYPES_WEIGHT,
    "MidRange":    _NON_PLAYTYPES_WEIGHT,
    "RimPressure": _NON_PLAYTYPES_WEIGHT,
    "Playmaking":  _NON_PLAYTYPES_WEIGHT,
    "Defense":     _NON_PLAYTYPES_WEIGHT,
    "Playtypes":   _PLAYTYPES_WEIGHT,
}
CONTESTED_SHOT_OPPORTUNITY_WEIGHT = 0.80
OPP_FGA_OPPORTUNITY_WEIGHT        = 0.20
EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER = ["ThreePT", "MidRange", "RimPressure", "Playmaking", "Defense", "Playtypes"]


def build_locked_euclidean_feature_signature() -> str:
    signature_payload = {
        "euclidean_kmeans_locked_group_features": EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES,
        "euclidean_kmeans_locked_group_weights": EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS,
        "euclidean_kmeans_locked_group_order": EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER,
        "euclidean_kmeans_locked_k": EUCLIDEAN_KMEANS_LOCKED_K,
        "euclidean_kmeans_locked_clip_zscore": EUCLIDEAN_KMEANS_LOCKED_CLIP_ZSCORE,
        "euclidean_kmeans_locked_pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
        "euclidean_kmeans_locked_similarity_distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
    }
    raw_signature = json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw_signature).hexdigest()
EUCLIDEAN_KMEANS_LOCKED_K = 16
EUCLIDEAN_KMEANS_LOCKED_PIPELINE = "freq_raw_6blocks_equal_cosine"
EUCLIDEAN_KMEANS_LOCKED_SPACE_TRANSFORM = "season_median_imputed_league_standardized_clipped_freq_6blocks_equal"
EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC = "cosine"
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
                    # The repo's own assignments are the source of truth. A stale
                    # local export must never silently outrank them.
                    BACKEND_DATA_DIR / "euclidean_kmeans_locked_assignments.csv",
                    LOCAL_PRODUCTION_OUTPUT_DIR / "euclidean_kmeans_locked_assignments.csv",
                ]
            )
        ),
    )
)
# Repo-local assets always take precedence over a stale local export.
GALAXY_PRECOMPUTED_PATHS = [
    BACKEND_DATA_DIR / "galaxy_precomputed.json",
    LOCAL_PRODUCTION_OUTPUT_DIR / "galaxy_precomputed.json",
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
    BACKEND_DATA_DIR / "similar_players_precomputed_production.csv",
    LOCAL_PRODUCTION_OUTPUT_DIR / "similar_players_precomputed_production.csv",
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
    "playtypes_similarity_score",
]
SIMILAR_PLAYERS_REQUIRED_DETAIL_COLUMNS = [
    "strongest_similarity_blocks",
    "biggest_difference_blocks",
    "threept_distance",
    "midrange_distance",
    "rimpressure_distance",
    "playmaking_distance",
    "defense_distance",
    "playtypes_distance",
    *SIMILAR_PLAYERS_BLOCK_SCORE_COLUMNS,
]

# ---------------------------------------------------------------------------
# v4 similarity model (sim.ipynb port -- see backend/similarity_engine.py)
# ---------------------------------------------------------------------------
# Comps come from the precomputed asset, never from a live fit: the engine is a
# population model that takes a couple of seconds to fit and must never sit in a
# request path. scripts/precompute_similarity_v4.py writes it.
SIMILARITY_V4_PATHS = [
    BACKEND_DATA_DIR / "similarity_v4.json",
    LOCAL_PRODUCTION_OUTPUT_DIR / "similarity_v4.json",
    BACKEND_DIR.parent / "data" / "similarity_v4.json",
    BACKEND_DIR.parent / "similarity_v4.json",
]
_SIMILARITY_V4_CACHE: Dict[str, object] = {"path": None, "mtime_ns": None, "payload": None}
SIMILARITY_V4_DOMAINS = ("overall", "offense", "defense")
# Positional layout of one comp record; mirrors meta.comp_record_fields.
_V4_TARGET, _V4_OFF_SIM, _V4_DEF_SIM, _V4_ALL_SIM = 0, 1, 2, 3
_V4_OFF_DIST, _V4_DEF_DIST, _V4_ALL_DIST, _V4_ALIKE, _V4_DIFFERENT = 4, 5, 6, 7, 8


def find_similarity_v4_path() -> Optional[Path]:
    for candidate_path in SIMILARITY_V4_PATHS:
        if candidate_path.exists():
            return candidate_path
    return None


def load_similarity_v4_payload() -> Optional[Dict[str, object]]:
    """Load and cache the v4 similarity asset, reloading when the file changes."""
    payload_path = find_similarity_v4_path()
    if payload_path is None:
        return None
    mtime_ns = payload_path.stat().st_mtime_ns
    if (
        _SIMILARITY_V4_CACHE["path"] == str(payload_path)
        and _SIMILARITY_V4_CACHE["mtime_ns"] == mtime_ns
        and _SIMILARITY_V4_CACHE["payload"] is not None
    ):
        return _SIMILARITY_V4_CACHE["payload"]  # type: ignore[return-value]
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or "players" not in payload:
        return None
    _SIMILARITY_V4_CACHE.update({"path": str(payload_path), "mtime_ns": mtime_ns, "payload": payload})
    return payload


def expand_similarity_v4_comp(
    record: List[object],
    payload: Dict[str, object],
    rank_index: int,
) -> Dict[str, object]:
    """Turn one positional comp record into named fields."""
    player_keys = payload.get("player_keys", [])
    block_labels = payload.get("block_labels", [])

    def block_names(ids: object) -> List[str]:
        if not isinstance(ids, list):
            return []
        return [str(block_labels[int(i)]) for i in ids if 0 <= int(i) < len(block_labels)]

    target_index = int(record[_V4_TARGET])
    return {
        "rank": int(rank_index),
        "player_key": str(player_keys[target_index]) if 0 <= target_index < len(player_keys) else "",
        "off_similarity": float(record[_V4_OFF_SIM]),
        "def_similarity": float(record[_V4_DEF_SIM]),
        "overall_similarity": float(record[_V4_ALL_SIM]),
        "off_distance": float(record[_V4_OFF_DIST]),
        "def_distance": float(record[_V4_DEF_DIST]),
        "overall_distance": float(record[_V4_ALL_DIST]),
        "most_alike_blocks": block_names(record[_V4_ALIKE]),
        "most_different_blocks": block_names(record[_V4_DIFFERENT]),
    }


def build_similarity_v4_attention(player_entry: Dict[str, object], payload: Dict[str, object]) -> Dict[str, object]:
    """The paper's two attention tables for one player, ranked highest-first."""
    family_labels = payload.get("family_labels", [])
    skillset_labels = payload.get("skillset_labels", [])

    def rows(labels: List[Dict[str, object]], attention_key: str, learned_key: str) -> List[Dict[str, object]]:
        attention = player_entry.get(attention_key, []) or []
        learned = player_entry.get(learned_key, []) or []
        built = []
        for index, label in enumerate(labels):
            if index >= len(attention):
                break
            built.append({
                **label,
                "attention_pct": float(attention[index]),
                "learned_w": float(learned[index]) if index < len(learned) else 0.0,
            })
        built.sort(key=lambda row: -row["attention_pct"])
        return built

    return {
        "off_weight": float(player_entry.get("off_weight", 0.5)),
        "def_weight": float(player_entry.get("def_weight", 0.5)),
        "families": rows(family_labels, "family_attention", "family_learned_w"),
        "skillsets": rows(skillset_labels, "skillset_attention", "skillset_learned_w"),
    }


def build_similarity_edges_from_v4(
    guards: pd.DataFrame,
    labels: np.ndarray,
    payload: Dict[str, object],
    top_n: int = GALAXY_SIMILAR_PLAYER_COUNT,
) -> Optional[List[Dict[str, object]]]:
    """Galaxy similarity edges taken from the v4 model.

    Emits the legacy edge shape so existing consumers keep working, plus the
    offense / defense / overall fields the v4 model adds. Edge geometry follows
    the OVERALL ranking; the other two rankings ride along on each source edge
    set so the detail panel can switch domains without another request.
    """
    players = payload.get("players", {})
    if not isinstance(players, dict) or not players:
        return None

    cluster_by_key: Dict[str, int] = {}
    meta_by_key: Dict[str, Dict[str, str]] = {}
    for row_index, row in guards.reset_index(drop=True).iterrows():
        player_key = str(row["player_key"])
        cluster_by_key[player_key] = int(labels[row_index])
        meta_by_key[player_key] = {
            "player_name": str(row["Player Name"]),
            "season": str(row["Season"]),
            "team": str(row["teams_played"]),
            "position": str(row["position"]),
        }

    edges: List[Dict[str, object]] = []
    for source_key, entry in players.items():
        source_meta = meta_by_key.get(str(source_key))
        if source_meta is None:
            continue
        comps = entry.get("comps", {})
        if not isinstance(comps, dict):
            continue
        by_domain = {
            domain: [
                expand_similarity_v4_comp(record, payload, rank)
                for rank, record in enumerate(comps.get(domain, []) or [], start=1)
            ]
            for domain in SIMILARITY_V4_DOMAINS
        }
        for comp in by_domain["overall"][:top_n]:
            target_key = comp["player_key"]
            target_meta = meta_by_key.get(target_key)
            if target_meta is None:
                continue
            edges.append({
                "source": str(source_key),
                "target": target_key,
                "source_player_name": source_meta["player_name"],
                "source_season": source_meta["season"],
                "source_team": source_meta["team"],
                "source_position": source_meta["position"],
                "target_player_name": target_meta["player_name"],
                "target_season": target_meta["season"],
                "target_team": target_meta["team"],
                "target_position": target_meta["position"],
                "rank": int(comp["rank"]),
                "truth_distance": comp["overall_distance"],
                "similarity_score": comp["overall_similarity"],
                "similarity_distance_metric": "v4_personalized",
                "similarity_metric_used": "v4_personalized",
                "off_similarity": comp["off_similarity"],
                "def_similarity": comp["def_similarity"],
                "overall_similarity": comp["overall_similarity"],
                "off_distance": comp["off_distance"],
                "def_distance": comp["def_distance"],
                "overall_distance": comp["overall_distance"],
                "strongest_similarity_blocks": ", ".join(comp["most_alike_blocks"]),
                "biggest_difference_blocks": ", ".join(comp["most_different_blocks"]),
                "block_scores": {},
                "same_cluster": bool(
                    cluster_by_key.get(str(source_key)) == cluster_by_key.get(target_key)
                ),
                "source_cluster": int(cluster_by_key.get(str(source_key), 0)),
                "target_cluster": int(cluster_by_key.get(target_key, 0)),
            })
    return edges or None


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
    str(
        next(
            (
                p for p in [
                    Path(__file__).resolve().parent / "data" / "fullseasonfeatures_13_14_25_26_pullup.csv",
                    Path("/Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_13_14_25_26_pullup.csv"),
                ]
                if p.exists()
            ),
            Path(__file__).resolve().parent / "data" / "fullseasonfeatures_13_14_25_26_pullup.csv",
        )
    ),
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

# Every player-season in the dataset is clustered. Nobody is held out of the
# archetype model any more: the model now spans all five positions, so the
# forwards and bigs that used to distort a guard-only fit are ordinary members
# of it.
EUCLIDEAN_KMEANS_LOCKED_EXCLUDED_NAMES = set()
EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER = {
    1: 'Primary Offensive Engine',
    2: 'Downhill Table-Setting Point',
    3: 'Pull-Up Shooting Combo Guard',
    4: 'Point-of-Attack Connector Guard',
    5: 'Isolation-Heavy Mid-Range Maestro',
    6: 'Two-Level Movement Shooter',
    7: '3PT-Reliant Sharpshooter',
    8: 'Limited-Playmaking Scoring Wing',
    9: 'Corner-Spacing 3-and-D Wing',
    10: 'High-Efficiency Off-Ball Forward',
    11: 'Interior Playmaking Hub',
    12: 'Skilled Two-Way Scoring Big',
    13: 'Floor-Spacing Stretch Big',
    14: 'Conventional Two-Way Big',
    15: 'Physical Paint Finisher',
    16: 'Vertical Spacing Rim Protector',
}

EUCLIDEAN_KMEANS_CLUSTER_DESCRIPTION_BY_NUMBER = {
    1: 'These players are the offensive centerpiece of everything their team does, and the heat chart shows it in the two blocks that matter most for on-ball creation. The playmaking block is the strongest in the entire model: box creation sits at +2.50, offensive load at +2.17, points created from assists at +1.87, assist frequency at +1.86, and crafted passer rating at +1.66. Nothing else in the league comes close to carrying this much of an offense.\n\nWhat separates this archetype from every other high-usage group is deep off-the-dribble shooting. Seven-plus-dribble 3PA frequency sits at +2.94 and three-to-six-dribble 3PA frequency at +2.51, both the highest values in the model, and pull-up 3PA frequency reaches +2.67. These are players who generate their own three-point looks from anywhere on the floor after long possessions, which is why average seconds per touch (+1.87) and dribbles per touch (+1.84) are also extreme. The strongly negative assisted-FGM rates on both twos and threes confirm the same thing from the other direction: almost nothing they score is set up by someone else.\n\nThey pressure the rim as well, with points from drives at +1.75 and drive FGA frequency at +1.65, so defenses cannot simply run them off the line. The one consistent weakness is defense, where the entire block sits at -0.36 and contested shot frequency at -0.98. That is partly a real limitation and partly a workload effect: carrying this much offense leaves less available on the other end.\n\nThe nearest comparison is the Isolation-Heavy Mid-Range Maestro, which matches the shot-making but not the passing. This group creates far more for teammates, and creates it from three rather than from the mid-range.',
    2: "These players run an offense from the paint rather than from the arc. The playmaking block sits at +0.92, driven by potential assists per 75 at +1.70, points created from assists at +1.67, assist frequency at +1.60, and crafted passer rating at +1.49. Just as importantly, potential assist-to-turnover ratio is at +1.30, so this is high-volume passing that stays under control.\n\nThe route to those assists is downhill pressure, not perimeter gravity. Percent of FGA from drives is the defining feature at +1.57, with drive FGA frequency at +1.14 and points from drives at +1.00. Touch metrics reinforce the picture: passes received per 75 at +1.64, dribbles per touch at +1.61, seconds per touch at +1.57, and touches per 75 at +1.38. These players hold the ball, get into the paint, and pass out of it.\n\nThe gap in the profile is three-point volume. Three-point accuracy is fine at +0.12, but zero-to-one dribble 3PA frequency is at -0.55, 3PA frequency at -0.37, and points from threes at -0.43. They can make an open three; they simply do not hunt them. Because they also finish poorly at the rim once they get there (restricted area accuracy -0.54, dunks -0.80), the drives function more as a passing mechanism than as a scoring one.\n\nThe contrast with the Primary Offensive Engine is entirely about self-created shooting. Both groups run an offense, but the Engine adds elite pull-up three-point volume, while this archetype's value stops at the point where the pass has to become a shot.",
    3: 'These players are microwave scorers who create their own looks from the perimeter without carrying a full offensive load. The signature is off-the-dribble three-point volume: three-to-six-dribble 3PA frequency at +1.28, seven-plus-dribble 3PA frequency at +0.76, and pull-up 3PA frequency at +1.16, all sitting on top of an above-average overall 3PA frequency of +0.81. Pick-and-roll ball handler frequency at +1.23 shows where most of those looks come from.\n\nThey add real rim pressure to that shooting. Drive FGA frequency is at +1.06 and points from drives at +1.00, and percent of FGA from drives is at +0.83. Combined with a mid-range block at +0.19, this is a genuine three-level scoring profile, and the sharply negative assisted-2PT-FGM rate (-1.03) confirms these shots are self-created.\n\nThe playmaking block is positive at +0.38, but the composition matters: offensive load (+0.99) and box creation (+0.77) are much stronger than the passing-quality metrics, and assists on/off is actually negative at -0.30. These are players who use possessions and generate shots, not players who make their teammates better. Defense is the clear weakness at -0.38 across the board.\n\nThe separation from the Primary Offensive Engine is scale and passing. This group shares the shot profile but not the responsibility: box creation is +0.77 here against +2.50 there. The separation from the Two-Level Movement Shooter is who creates the shot; that archetype gets its looks from screens and handoffs, while this one gets them off the dribble.',
    4: 'These players earn their minutes through defense and ball security rather than scoring. The defensive block sits at +0.04, which understates the group because the two features that actually measure perimeter disruption are strongly positive: offensive fouls drawn frequency at +0.81 and deflections per 75 at +0.78, with adjusted opponent FG% difference at +0.25. Blocks (-0.52) and contested shot frequency (-0.59) are negative simply because those are rim-protection statistics that no perimeter defender accumulates.\n\nThe offensive value is connective. Potential assist-to-turnover ratio is the strongest feature in the playmaking block at +0.85, ahead of potential assists per 75 (+0.74), assist frequency (+0.70), and points created from assists (+0.70). They hold the ball a fair amount (dribbles per touch +1.06, seconds per touch +0.98, passes received per 75 +0.93) and rarely waste possessions with it.\n\nScoring is the honest limitation. The mid-range block is at -0.33, rim pressure at -0.27, and the finishing features are poor: dunks at -0.74, restricted area frequency at -0.72, restricted area accuracy at -0.65. What shooting exists is spot-up and low-volume, with percent of FGA from three at +0.50 and wide-open 3PA frequency at +0.31 but tight 3PA frequency at -0.34. Defenses concede these threes and live with the result.\n\nAgainst the Downhill Table-Setting Point, the trade is clear: this group is meaningfully better defensively and far less productive as a passer and paint threat.',
    5: 'These players are the best pure shot-makers in the model, and the mid-range block makes it obvious. Points from mid-range per 75 is at +2.35, mid-range frequency at +2.29, open 2PA frequency at +2.15, tight/very tight 2PA frequency at +2.15, and percent of FGA from mid-range at +1.77. The assisted-2PT rate at -1.16 confirms these are self-created looks, and the accuracy features stay positive across the board, so the volume is not empty.\n\nThe playtype signature is equally clear: pull-up 2PA frequency at +2.45 and isolation frequency at +1.74, with pick-and-roll ball handler frequency at +1.52. They also shoot off the dribble from three, with three-to-six-dribble 3PA frequency at +1.18 and seven-plus-dribble 3PA frequency at +0.77, and they hit those shots (seven-plus-dribble accuracy +0.64). Rim pressure is real too, at +0.57 with points from drives at +1.51.\n\nThe playmaking block is positive at +0.85, but it is carried almost entirely by usage rather than passing: offensive load is at +1.58 and box creation at +1.48, while assists on/off sits at +0.10 and potential assist-to-turnover ratio at +0.15. They generate a lot of offense; they do not particularly elevate the players around them. Defense is the weakest part of the profile at -0.38.\n\nThe distinction from the Primary Offensive Engine is where the difficulty lives. Both are elite creators, but the Engine creates from three and passes at an elite level, while this archetype takes and makes the hardest two-point shots in basketball.',
    6: 'These players get open without the ball and punish defenses from two levels. The playtype block tells the story directly: off-screen frequency at +1.03, handoff frequency at +0.94, off-screen PPP at +0.80, catch-and-shoot 3PA frequency at +0.76, and handoff PPP at +0.71. This is movement shooting, not spot-up shooting, and the efficiency features confirm they are good at it rather than merely willing.\n\nThe three-point block sits at +0.51, with open 3PA frequency at +0.94, points from threes at +0.93, 3PA frequency at +0.83, and tight/very tight 3PA frequency at +0.80. The last of those matters: they take contested threes too, which is what separates a movement shooter from a stationary one.\n\nWhat makes this archetype distinct is the mid-range layer at +0.67. Average 2PT shot distance is at +1.32, percent of FGA from mid-range at +1.03, points from mid-range at +0.91, and pull-up 2PA frequency at +0.69. Coming off a screen or a handoff, they will take the pull-up two when the three is taken away.\n\nOutside those two levels the profile is thin. Rim pressure is at -0.25 with restricted area frequency at -0.91, playmaking is at -0.13, and defense is the weakest block at -0.45. Against the 3PT-Reliant Sharpshooter, the difference is both the mid-range game and the mechanism: this group runs to get open, while the Sharpshooter stands still and waits.',
    7: 'These players do one thing at a very high level and very little else. Percent of FGA from three sits at +1.44, zero-to-one dribble 3PA frequency at +1.38, points from threes per 75 at +1.13, 3PA frequency at +1.06, and open 3PA frequency at +0.90. Catch-and-shoot 3PA frequency at +1.35 and spot-up frequency at +0.72 confirm that essentially all of it comes off the catch.\n\nThe off-the-dribble features are the mirror image: seven-plus-dribble 3PT accuracy at -0.74, seven-plus-dribble frequency at -0.46, and three-to-six-dribble frequency at -0.42. Isolation PPP at -1.26, touches per 75 at -1.01, and seconds per touch at -0.78 all say the same thing. These players do not hold the ball, and when they do, nothing good happens.\n\nEverything inside the arc is a weakness. The rim pressure block is at -0.55, with restricted area frequency at -1.12, paint non-RA frequency at -1.03, and drive FGA frequency at -0.75. Playmaking is at -0.45 and defense at -0.36. The value here is entirely spacing: they bend a defense by standing in the right place and making the shot.\n\nThe comparison to the Two-Level Movement Shooter is the cleanest in the model. Both groups are high-volume, high-accuracy shooters, but that archetype adds a mid-range counter and generates its own separation through screens, while this one is dependent on someone else creating the look.',
    8: "This is the largest archetype in the model, and it is defined more by the absence of a specialty than by the presence of one. The three-point block sits at +0.01, mid-range at +0.06, and rim pressure at +0.01. Almost every scoring feature is within a fraction of the league average, which is exactly why these players cluster together: there is no dimension on which they clearly separate from the field.\n\nThe one consistently negative block is playmaking, at -0.43. The on/off features are the worst part of it, with assists on/off at -0.67, eFG% on/off at -0.63, and points per 100 on/off at -0.57. Teams do not run better when these players are on the floor. Defense is also below average at -0.30, with D-LEBRON at -0.56.\n\nThere is a mild self-creation lean inside the arc, with percent of FGA from drives at +0.48 and drive FGA frequency at +0.29, and the playtype efficiency features are slightly positive across spot-ups, handoffs, and isolations. These are competent scorers who can be given the ball in a variety of situations without the offense breaking.\n\nThe honest summary is that this archetype is the league's replacement level for a scoring wing. The players in it can fill a role, but they neither space the floor at the level of the 3PT-Reliant Sharpshooter, defend at the level of the Corner-Spacing 3-and-D Wing, nor create at the level of the Pull-Up Shooting Combo Guard.",
    9: 'These players are built to defend, space the corners, and touch the ball as little as possible in between. The defensive block sits at -0.01, which is a poor summary of the group: deflections per 75 is at +0.39, adjusted opponent FG% difference at +0.04, and adjusted D-LEBRON at +0.03, all achieved without the rim-protection volume that inflates the block for bigs. Among perimeter players, this is the strongest defensive profile in the model outside the Point-of-Attack Connector Guard.\n\nOffensively the role is narrow by design. The assisted 3PT-FGM rate is at +0.64 and percent of FGA from three at +0.39, but every off-the-dribble shooting feature collapses: three-to-six-dribble 3PT accuracy at -0.84, seven-plus-dribble accuracy at -0.79, three-to-six-dribble frequency at -0.67. Touches per 75 at -0.97 and passes received at -0.99 confirm they are not asked to do more.\n\nThe mid-range block at -0.82 is the weakest of any wing archetype, and mid-range accuracy at -1.65 is close to the worst value in the model. Playmaking is at -0.66, with offensive load at -1.03 and box creation at -0.94. This is a genuinely one-way role: defend, stand in the corner, take the open three.\n\nAgainst the 3PT-Reliant Sharpshooter, the trade is explicit. That archetype is a far better and higher-volume shooter; this one is a far better defender and takes a fraction of the shots.',
    10: "This archetype is the clearest example in the model of a group defined by efficiency rather than volume. Nearly every rate statistic is positive and nearly every usage statistic is negative. Off-screen PPP sits at +0.69, spot-up frequency at +0.65, handoff PPP at +0.59, pick-and-roll ball handler PPP at +0.54, and isolation PPP at +0.51, while pick-and-roll ball handler frequency is at -0.58, touches per 75 at -0.48, dribbles per touch at -0.48, and pull-up 2PA frequency at -0.49.\n\nThe shooting is real and it is almost entirely assisted. Assisted 3PT-FGM rate is at +0.52, zero-to-one dribble 3PA frequency at +0.45, wide-open 3PA frequency at +0.41, and three-point accuracy at +0.39. Inside the arc they finish well without hunting shots: restricted area accuracy at +0.27 and drive FG% at +0.13 against a paint non-RA frequency of -0.42.\n\nThe most distinctive feature is the on/off block. Assists on/off is at +0.61, eFG% on/off at +0.60, and points per 100 on/off at +0.54 — the highest impact values in the model outside the two engine archetypes — even though assist frequency (-0.52) and potential assists (-0.55) are well below average. These players do not create offense; they make the offense around them work, and the scoreboard notices.\n\nThe comparison to the Limited-Playmaking Scoring Wing is instructive, since the two occupy similar positions and similar usage. The difference is entirely conversion and impact: this group's on/off features are strongly positive where that group's are the worst in the model.",
    11: 'This archetype runs an offense from inside the arc, and it contains the most recognizable non-guard playmakers in basketball. The playmaking block sits at +0.78, with points created from assists at +1.08, assist frequency at +1.04, crafted passer rating at +1.03, potential assists at +0.98, and assists on/off at +0.91. Touches per 75 at +1.24 confirms the offense genuinely runs through them.\n\nThey combine that with real interior scoring. The rim pressure block is at +0.50, with restricted area frequency at +1.23, paint non-RA frequency at +0.83, restricted area accuracy at +0.60, and drive FG% at +0.55. Cut frequency at +0.90 shows they also move without the ball rather than only operating as a hub.\n\nThe limitation is perimeter shooting, and it is severe. Tight/very tight 3PT accuracy is at -1.19, open 3PA frequency at -1.12, percent of FGA from three at -1.07, 3PA frequency at -1.05, and points from threes at -1.02. Catch-and-shoot 3PA frequency at -1.02 and pull-up 3PT accuracy at -1.05 close off both routes to the arc. Potential assist-to-turnover ratio at +0.12 is also modest for a group with this much passing volume, so the creation comes with real turnover cost.\n\nDefensively they hold up well, with adjusted D-LEBRON at +0.89 and deflections at +0.48. Against the Skilled Two-Way Scoring Big, this group passes far better and shoots far worse; against the Primary Offensive Engine, it is the same role played entirely inside the three-point line.',
    12: 'These players are the most offensively complete bigs in the model, and the mid-range block is what separates them. Mid-range frequency sits at +1.16, points from mid-range at +1.13, percent of FGA from mid-range at +1.03, and open 2PA frequency at +0.93, with all three accuracy features positive. Paint non-RA frequency at +1.10 extends the same skill closer in. This is a genuine face-up and short-range scoring game, not just finishing.\n\nThey protect the rim at a high level while doing it. Blocks per 75 is at +1.39, opponent FGA per 75 at +1.31, contested shot frequency at +1.23, and adjusted D-LEBRON at +0.98. Worth noting: adjusted opponent FG% difference reads -0.93 for this group, which looks contradictory until you account for shot location. Interior defenders contest a much higher share of shots at the rim, where opponents convert at a far higher baseline rate than on the perimeter, so a raw FG%-difference metric systematically penalizes the players doing the most defensive work. The volume features are the more honest read here.\n\nThey also stretch the floor a little, with three-point accuracy at +0.28 and wide-open 3PA frequency at +0.25, although the volume features are negative and the off-the-dribble three is absent entirely.\n\nPlaymaking is neutral at +0.04, held up by assists on/off (+0.79) rather than by passing volume, and potential assist-to-turnover ratio at -0.77 is a real weakness. Against the Floor-Spacing Stretch Big, this group scores far more and from far more places; against the Vertical Spacing Rim Protector, it trades a little rim deterrence for an actual offensive skillset.',
    13: "These players exist to pull a defense's biggest defender away from the basket. The shooting profile is modest in raw terms but decisive in context: assisted 3PT-FGM rate at +0.70, wide-open 3PA frequency at +0.68, percent of FGA from three at +0.46, zero-to-one dribble 3PA frequency at +0.41, and three-point accuracy at +0.24. Catch-and-shoot 3PA frequency at +0.61 and spot-up frequency at +0.52 confirm the shots are stationary and assisted.\n\nEverything else on offense is deliberately absent. The rim pressure block is at -0.38, with drive FGA frequency at -0.94, percent of FGA from drives at -0.94, points from drives at -0.89, and paint non-RA frequency at -0.77. Playmaking is at -0.61, with potential assists at -0.84 and box creation at -0.80. Every playtype efficiency feature outside of spot-ups is deeply negative: pick-and-roll PPP at -1.51, handoff PPP at -1.38, isolation PPP at -1.33.\n\nDefensively they are useful without being anchors. Contested shot frequency at +0.95 and opponent FGA per 75 at +0.81 show they defend interior volume, while blocks at +0.30 and adjusted D-LEBRON at +0.29 put the rim protection itself at slightly above average.\n\nThe distinction from the Skilled Two-Way Scoring Big is scoring range and self-creation. Both shoot; that archetype also posts, faces up, and scores from the mid-range at high volume, while this group's offensive contribution begins and ends with standing at the arc and making the open shot.",
    14: 'This is the traditional interior big, and it is the most populous big archetype in the model. The defensive block sits at +0.49, with contested shot frequency at +1.51, opponent FGA per 75 at +1.42, blocks per 75 at +1.10, and adjusted D-LEBRON at +0.86. Offensively the value is concentrated at the basket: dunks per 75 at +1.14, restricted area frequency at +1.11, restricted area accuracy at +0.73, and cut frequency at +1.73.\n\nThe perimeter game is largely absent but not quite nonexistent. 3PA frequency at -1.43, percent of FGA from three at -1.43, and points from threes at -1.39 are all deeply negative, yet three-point accuracy is only at -0.13 and the assisted 3PT rate is at +0.75. These are players who will take the occasional open corner three and are not embarrassed by it; they simply are not asked to.\n\nThe gaps are playmaking and self-creation. The playmaking block is at -0.52, with potential assist-to-turnover ratio at -0.91 the worst feature in it. Percent of FGA from drives at -1.15 and drive FGA frequency at -1.03 confirm they do not attack off the bounce, and pick-and-roll PPP at -1.51 and handoff PPP at -1.40 show they are not effective in the actions that would let them.\n\nThe separation from the Vertical Spacing Rim Protector is a matter of degree in both directions: that archetype protects the rim harder and finishes more vertically, but cannot shoot at all, where this group retains a functional if unused jumper.',
    15: "These players do their work in a narrow band of the floor and do it physically. Dunks per 75 sits at +1.68, restricted area frequency at +1.32, restricted area accuracy at +0.76, and paint non-RA frequency at +0.51, with cut frequency at +1.92. Defensively they cover a lot of ground: opponent FGA per 75 at +1.46, contested shot frequency at +1.33, adjusted D-LEBRON at +1.20, and blocks at +0.96.\n\nWhat separates this archetype from every other big group is the complete absence of shooting. Wide-open 3PT% is at -2.94, assisted 3PT-FGM rate at -2.88, three-point accuracy at -2.80, and zero-to-one dribble 3PT accuracy at -2.74. Catch-and-shoot accuracy at -2.86 and spot-up frequency at -1.84 confirm there is no perimeter option at all. Average 2PT shot distance at -1.10 shows how tightly their shot profile is compressed toward the basket.\n\nUnlike the Vertical Spacing Rim Protector, though, they have interior craft. The mid-range block at -0.34 is meaningfully better than that archetype's -1.06, mid-range accuracy is actually positive at +0.19, and paint non-RA frequency at +0.51 against that group's -0.56 shows real short-range scoring rather than pure lob finishing. Assists on/off at +0.40 also hints at competent short-roll passing.\n\nThe overall shape is a high-volume, high-efficiency interior scorer and defender with no floor spacing whatsoever. Whether that is valuable depends almost entirely on whether the other four players on the floor can shoot.",
    16: 'This is the most extreme archetype in the model, and it is extreme in both directions. On defense, contested shot frequency sits at +1.81, blocks per 75 at +1.69, opponent FGA per 75 at +1.65, and adjusted D-LEBRON at +1.43 — the strongest rim-protection profile in basketball. On offense, dunks per 75 at +2.49 and restricted area frequency at +1.45 describe a player who scores almost exclusively at the basket, mostly off cuts and rolls (cut frequency +1.98).\n\nEverything else collapses. The three-point block is at -1.74, with assisted 3PT-FGM rate at -3.29, three-point accuracy at -3.18, zero-to-one dribble accuracy at -3.12, and wide-open 3PT% at -2.96. The mid-range block is at -1.06, with mid-range accuracy at -1.88 and average 2PT shot distance at -1.67. Spot-up PPP at -3.33 and catch-and-shoot accuracy at -2.86 are the lowest values anywhere in the model. Playmaking is at -0.71, with crafted passer rating at -1.12.\n\nOne measurement note is important for reading this group fairly. Adjusted opponent FG% difference reads -1.13, which appears to contradict everything else in the defensive block. It does not: these players contest a far higher share of shots at the rim than anyone else, and rim attempts convert at a much higher baseline rate than perimeter attempts, so a raw opponent FG% differential penalizes exactly the players doing the most valuable defensive work. The volume and impact features are the more reliable signal.\n\nThe result is the sharpest specialist profile in the league: a player who single-handedly changes what an opponent can do at the rim, and who gives back essentially all of the floor on the other end.',
}


DEFAULT_FEATURES = []
ADDITIONAL_ALLOWED_FEATURES = []
ALLOWED_FEATURES = list(dict.fromkeys(CURRENT_CSV_FEATURES + ADDITIONAL_ALLOWED_FEATURES + BADGE_REQUIRED_FEATURES))
LOWER_IS_BETTER_PERCENTILE_FEATURES = {
    "opp_players_fg_pct_difference",
    "OPP_SHOT_QUALITY_ON_OFF",
    "OPP_EFG_PCT_ON_OFF",
}

# Percentiles are computed against every player-season in the same season.
# No player is excluded from the peer pool.
PERCENTILE_AND_BADGE_EXCLUDED_NAMES: set = set()

DEFENSE_SCORE_CAPPED_NAMES: set = set()

ROWS_TO_REMOVE = []

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

CLUSTER_NAME_BY_NUMBER = dict(EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER)


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


app = FastAPI(title="NBA Player Cluster Explorer API")

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

    # Defensive opportunity score: composite reliability weight used for both
    # opp_players_fg_pct_difference_adjusted and D-LEBRON_adjusted.
    # 80% contested_shot_frequency season percentile + 20% Opp_players_fga_per_75_poss season percentile.
    season_col = output["Season"] if "Season" in output.columns else pd.Series("unknown", index=output.index)

    def _defensive_opportunity_score(grp: pd.DataFrame) -> pd.Series:
        contested = pd.to_numeric(grp.get("contested_shot_frequency", pd.Series(np.nan, index=grp.index)), errors="coerce")
        opp_fga   = pd.to_numeric(grp.get("Opp_players_fga_per_75_poss", pd.Series(np.nan, index=grp.index)), errors="coerce")
        c_pct = contested.rank(pct=True, method="average").fillna(0.0)
        f_pct = opp_fga.rank(pct=True, method="average").fillna(0.0)
        return (CONTESTED_SHOT_OPPORTUNITY_WEIGHT * c_pct + OPP_FGA_OPPORTUNITY_WEIGHT * f_pct).astype(float)

    if "opp_players_fg_pct_difference" in output.columns:
        adj = pd.Series(np.nan, index=output.index, dtype=float)
        for _, grp in output.groupby(season_col):
            x    = pd.to_numeric(grp["opp_players_fg_pct_difference"], errors="coerce")
            mu_s = x.median()
            if pd.isna(mu_s):
                mu_s = 0.0
            opp  = _defensive_opportunity_score(grp)
            t_s  = opp.median()
            if pd.isna(t_s) or t_s == 0:
                adj.loc[grp.index] = x
            else:
                w = opp / (opp + t_s)
                adj.loc[grp.index] = mu_s + w * (x - mu_s)
        output["opp_players_fg_pct_difference_adjusted"] = adj

    if "D-LEBRON" in output.columns:
        d_adj = pd.Series(np.nan, index=output.index, dtype=float)
        for _, grp in output.groupby(season_col):
            d    = pd.to_numeric(grp["D-LEBRON"], errors="coerce")
            mu_s = d.median()
            if pd.isna(mu_s):
                mu_s = 0.0
            opp  = _defensive_opportunity_score(grp)
            t_s  = opp.median()
            if pd.isna(t_s) or t_s == 0:
                d_adj.loc[grp.index] = d
            else:
                w = opp / (opp + t_s)
                d_adj.loc[grp.index] = mu_s + w * (d - mu_s)
        output["D-LEBRON_adjusted"] = d_adj

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
        "Playtypes": {
            "distance": row_float(row, "playtypes_distance"),
            "similarity_score": row_float(row, "playtypes_similarity_score"),
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

    eligible = source.copy()
    excluded_names = {normalize_player_name_for_percentile_pool(name) for name in PERCENTILE_AND_BADGE_EXCLUDED_NAMES}
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
    """Attach cumulative same-season league-wide rarity labels to badge rows.

    Rarity is cumulative by tier: a gold badge counts as gold, silver, and bronze
    for the same badge skill. Therefore bronze rarity is the most common, silver
    means silver-or-better, gold means gold-or-better, and diamond means diamond.
    """
    if badge_frame.empty or not {"Season", "badge_id", "badge_tier"}.issubset(set(badge_frame.columns)):
        return badge_frame

    output = badge_frame.copy()
    excluded_names = {normalize_player_name_for_percentile_pool(name) for name in PERCENTILE_AND_BADGE_EXCLUDED_NAMES}

    if guards is not None and not guards.empty and {"Season", "Player Name", "position"}.issubset(set(guards.columns)):
        eligible = guards.copy()
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
        rarity_labels.append(f"{rarity_percent:.2f}% of {season} players" if season else f"{rarity_percent:.2f}% of players")

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
        # Fill absent optional columns with 0.0 so missing data never propagates into
        # median/std calculations or shifts similarity scores.
        df = pd.concat(
            [df, pd.DataFrame({missing_feature: 0.0 for missing_feature in missing_features}, index=df.index)],
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

    players = df.copy()
    if players.empty:
        raise ValueError("No player-seasons found in the dataset.")
    guards = players

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
        for col in missing_feature_columns:
            base_guards[col] = 0.0

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
    numeric_feature_frame = numeric_feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    numeric_feature_frame["Season"] = locked_guards["Season"].values

    standardized_feature_frame = season_standardize_clip_for_locked_euclidean(
        dataframe=numeric_feature_frame,
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
        # The assignments file's contents decide the archetypes, so its mtime has
        # to be part of the key. Without it a re-clustering keeps serving the old
        # cached labels under the new cluster names.
        "euclidean_kmeans_locked_assignments_mtime_ns": (
            EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH.stat().st_mtime_ns
            if EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH.exists()
            else 0
        ),
        "euclidean_kmeans_locked_group_features": EUCLIDEAN_KMEANS_LOCKED_GROUP_FEATURES,
        "euclidean_kmeans_locked_group_weights": EUCLIDEAN_KMEANS_LOCKED_GROUP_WEIGHTS,
        "euclidean_kmeans_locked_group_order": EUCLIDEAN_KMEANS_LOCKED_GROUP_ORDER,
        "euclidean_kmeans_locked_pipeline": EUCLIDEAN_KMEANS_LOCKED_PIPELINE,
        "euclidean_kmeans_locked_similarity_distance_metric": EUCLIDEAN_KMEANS_LOCKED_SIMILARITY_DISTANCE_METRIC,
        # Similar-player edges are read from the v4 asset, so its contents belong in
        # the key -- otherwise a re-run of the similarity precompute keeps serving
        # the previously cached comparisons.
        "similarity_v4_path": str(find_similarity_v4_path() or ""),
        "similarity_v4_mtime_ns": (
            find_similarity_v4_path().stat().st_mtime_ns
            if find_similarity_v4_path() is not None
            else 0
        ),
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
        source_distance_metric = default_similarity_metric
        source_distance_matrix = distance_matrix
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
    block_slices = None
    if metric_meta is not None and (include_block_details or bool(display_meta.get("precomputed", False))):
        # Block-level edge details are expensive to compute live and should come
        # from the precomputed galaxy asset. Keep live fallback fast so the
        # initial scatter/galaxy render does not sit on LOADING_SCATTER.
        raw_block_slices = metric_meta.get("euclidean_kmeans_locked_block_slices")
        if isinstance(raw_block_slices, dict):
            block_slices = raw_block_slices

    # Similar-player edges come from the v4 model when its asset is present. Only
    # the similarity layer moves: archetypes are locked from the assignments file
    # and constellation edges, medoids and labels still describe the clustered
    # space, so they keep using the locked distance matrix below.
    similarity_edges = None
    similarity_v4_payload = load_similarity_v4_payload()
    if similarity_v4_payload is not None:
        similarity_edges = build_similarity_edges_from_v4(
            guards=guards,
            labels=labels,
            payload=similarity_v4_payload,
            top_n=GALAXY_SIMILAR_PLAYER_COUNT,
        )
    if similarity_edges is None:
        similarity_edges = build_similarity_edges_for_galaxy(
            guards=guards,
            labels=labels,
            distance_matrix=distance_matrix,
            top_n=GALAXY_SIMILAR_PLAYER_COUNT,
            X_metric=X_metric,
            block_slices=block_slices,
            default_similarity_metric=similarity_distance_metric,
        )
        similarity_model = "locked_block_weighted"
    else:
        similarity_model = "v4_personalized"

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
        "similarity_model": similarity_model,
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
            "label": f"Median Player {player_season}",
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
    score_logic = "Same-season guard percentile scoring with Scottie Barnes excluded from percentile peer pools."
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
        "guard_median": {"label": f"Median Player {player_season}", "season": player_season, "scores": component_median_scores(season_guard_mask), "subsections": subsection_median_scores(season_guard_mask)},
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

def build_similar_players_response_v4(
    source_point: Dict[str, object],
    source_key: str,
    entry: Dict[str, object],
    payload: Dict[str, object],
    point_by_key: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Similar-players response backed by the v4 model.

    ``similar_players`` stays the OVERALL ranking so existing callers are
    unaffected; ``comps`` carries all three rankings and ``attention`` carries
    the paper's skill tables for the queried player.
    """
    comps = entry.get("comps", {})
    if not isinstance(comps, dict):
        comps = {}

    def build_domain(domain: str) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        for rank, record in enumerate(comps.get(domain, []) or [], start=1):
            comp = expand_similarity_v4_comp(record, payload, rank)
            target_point = point_by_key.get(comp["player_key"])
            if target_point is None:
                continue
            cluster_number = int(target_point.get("cluster", 0) or 0)
            cluster_name = get_cluster_title(cluster_number, "kmeans", "euclidean")
            items.append({
                "rank": comp["rank"],
                "player_name": str(target_point.get("player_name", "")),
                "season": str(target_point.get("season", "")),
                "team": str(target_point.get("teams_played", "")),
                "position": str(target_point.get("position", "")),
                "player_season_id": comp["player_key"],
                "cluster_raw": cluster_number - 1,
                "cluster_number": cluster_number,
                "cluster_name": cluster_name,
                "archetype_name": cluster_name,
                **get_player_headshot_payload(target_point.get("player_name", "")),
                "similarity_score": comp["overall_similarity"],
                "overall_distance": comp["overall_distance"],
                "off_similarity": comp["off_similarity"],
                "def_similarity": comp["def_similarity"],
                "overall_similarity": comp["overall_similarity"],
                "off_distance": comp["off_distance"],
                "def_distance": comp["def_distance"],
                "similarity_distance_metric": "v4_personalized",
                "similarity_metric_used": "v4_personalized",
                "same_cluster": bool(cluster_number == int(source_point.get("cluster", 0) or 0)),
                "same_archetype": bool(cluster_number == int(source_point.get("cluster", 0) or 0)),
                "most_alike_blocks": comp["most_alike_blocks"],
                "most_different_blocks": comp["most_different_blocks"],
                "strongest_similarity_blocks": ", ".join(comp["most_alike_blocks"]),
                "biggest_difference_blocks": ", ".join(comp["most_different_blocks"]),
                "block_scores": {},
            })
        return items

    by_domain = {domain: build_domain(domain) for domain in SIMILARITY_V4_DOMAINS}
    source_cluster_number = int(source_point.get("cluster", 0) or 0)
    source_cluster_name = get_cluster_title(source_cluster_number, "kmeans", "euclidean")
    meta = payload.get("meta", {})
    return {
        "source_file": "similarity_v4",
        "similarity_model": "v4_personalized",
        "similarity_model_meta": {
            "engine": meta.get("engine", ""),
            "paper": meta.get("paper", ""),
            "blocks": meta.get("blocks", 0),
            "subgroups": meta.get("subgroups", 0),
            "fitting_population": meta.get("fitting_population", 0),
            "candidate_pool": meta.get("candidate_pool", 0),
            "similarity_transform": meta.get("similarity_transform", ""),
        },
        "player_name": str(source_point.get("player_name", "")),
        "season": str(source_point.get("season", "")),
        "team": str(source_point.get("teams_played", "")),
        "position": str(source_point.get("position", "")),
        "player_season_id": source_key,
        "cluster_raw": source_cluster_number - 1,
        "cluster_number": source_cluster_number,
        "cluster": source_cluster_name,
        "cluster_name": source_cluster_name,
        "archetype_name": source_cluster_name,
        **get_player_headshot_payload(source_point.get("player_name", "")),
        "attention": build_similarity_v4_attention(entry, payload),
        "comps": by_domain,
        "similar_players": by_domain["overall"],
    }


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

    # v4 path: three ranked lists plus the model's attention profile.
    similarity_v4_payload = load_similarity_v4_payload()
    v4_entry = None
    if similarity_v4_payload is not None:
        v4_entry = similarity_v4_payload.get("players", {}).get(source_key)
    if v4_entry is not None:
        return build_similar_players_response_v4(
            source_point=source_point,
            source_key=source_key,
            entry=v4_entry,
            payload=similarity_v4_payload,
            point_by_key=point_by_key,
        )
    if similarity_v4_payload is not None:
        # In the galaxy but absent from the similarity model's source data. Say so
        # rather than falling through to a legacy path that has no edges either.
        source_cluster_number = int(source_point.get("cluster", 0) or 0)
        source_cluster_name = get_cluster_title(source_cluster_number, "kmeans", "euclidean")
        return {
            "source_file": "similarity_v4",
            "similarity_model": "v4_personalized",
            "unavailable_reason": (
                "This player-season is not in the similarity model's feature source, "
                "so no comparisons can be computed for it."
            ),
            "player_name": str(source_point.get("player_name", "")),
            "season": str(source_point.get("season", "")),
            "team": str(source_point.get("teams_played", "")),
            "position": str(source_point.get("position", "")),
            "player_season_id": source_key,
            "cluster_raw": source_cluster_number - 1,
            "cluster_number": source_cluster_number,
            "cluster": source_cluster_name,
            "cluster_name": source_cluster_name,
            "archetype_name": source_cluster_name,
            **get_player_headshot_payload(source_point.get("player_name", "")),
            "attention": None,
            "comps": {domain: [] for domain in SIMILARITY_V4_DOMAINS},
            "similar_players": [],
        }

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

    # No position filter: player comparison covers every player in the model.
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


@app.get("/api/similar-players")
def similar_players(
    player_name: str,
    season: str,
    pipeline: Optional[str] = None,
    k: Optional[int] = None,
    pca_variance_target: Optional[str] = None,
):
    # The v4 model is the site's similarity engine when its asset is present.
    # The legacy CSV path below stays as a fallback for older exports.
    if load_similarity_v4_payload() is not None:
        return build_similar_players_response_from_galaxy(player_name=player_name, season=season)

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
