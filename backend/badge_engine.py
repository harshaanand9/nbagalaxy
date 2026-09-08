import json
import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BADGE_TIER_ORDER = ["diamond", "gold", "silver", "bronze"]
BADGE_TIER_RANK = {tier: index for index, tier in enumerate(BADGE_TIER_ORDER)}
BADGE_CATEGORY_ORDER = {
    "three_pt": 0,
    "midrange": 1,
    "interior": 2,
    "rim_pressure": 3,
    "scoring": 4,
    "playmaking": 5,
    "defense": 6,
}

BADGE_REQUIRED_FEATURES = [
    "Avg3ptShotDistance",
    "3fga_frequency",
    "3P_Accuracy",
    "catch_shoot_3P_frequency",
    "catch_shoot_3P_accuracy",
    "avg_closest_defender_3FGA",
    "pct_3fga_wide_open",
    "tight_very_tight_3fga_frequency",
    "pull_up_3P_frequency",
    "pull_up_3P_accuracy",
    "traditional_fg3a",
    "traditional_pts",
    "pts_per_75",
    "fga_per_75",
    "MidRangeFrequency",
    "MidRangeAccuracy",
    "by_zone_statistics_mid_range_fga",
    "tight_very_tight_2fga_frequency",
    "pull_up_2P_frequency",
    "pull_up_2P_accuracy",
    "drives_drive_fga",
    "drives_drives",
    "pts_from_drives_per_75",
    "drive_fga_frequency",
    "drive_fg_pct",
    "traditional_fta",
    "fta_rate",
    "fta_per_75",
    "drive_ast_per_75",
    "drives_per_75",
    "drive_tov_per_75",
    "drive_passes_per_75",
    "drives_drive_ast",
    "pts_from_midrange_per_75",
    "dunks_per_75_poss",
    "Blocks_per_75",
    "Steals_per_75",
    "Deflections_per_75",
    "off_fouls_drawn_frequency",
    "opp_players_fg_pct_difference",
    "Opp_players_fga_per_75_poss",
    "crafted_cdpm",
    "D-LEBRON",
    "assist_frequency",
    "potential_assist_frequency",
    "potential_assist_tov_ratio",
    "assists_tov_ratio",
    # Interior scoring, rim protection and screening. None of these were needed
    # while the site was guards-only; they carry the badges that forwards and
    # centers actually earn.
    "RestrictedArea_Frequency",
    "RestrictedArea_Accuracy",
    "Paint_Non_RA_Frequency",
    "Paint_Non_RA_Accuracy",
    "cut_frequency",
    "cut_ppp",
    "contested_shot_frequency",
    "screen_assist_points_per_game",
]

BADGE_LOWER_IS_BETTER_FEATURES = {
    "pct_3fga_wide_open",
    "avg_closest_defender_3FGA",
    "drive_tov_per_75",
    "opp_players_fg_pct_difference",
}

# Badges are computed for every player-season, and every percentile is taken
# against every other player-season in the same season regardless of position.
#
# The by-name exclusion lists that used to live here existed only because the
# pool was guards: LeBron, Simmons and Barnes were forwards being ranked against
# guards, and Harden / Lillard / Curry / Doncic scored as "lockdown" defenders
# because a guard-only pool contained no rim protection to compare them
# against. A league-wide pool removes both problems at the source, so the manual
# overrides are gone. Eligibility is decided by opportunity gates on the badges
# themselves, never by a player's name.
PERCENTILE_AND_BADGE_EXCLUDED_NAMES: set = set()

BADGE_DEFINITIONS = {
    "deep_range_bomber": {"name": "Deep Range Bomber", "category": "three_pt"},
    "catch_and_shoot_converter": {"name": "Catch and Shoot Converter", "category": "three_pt"},
    "contested_3pt_maker": {"name": "Contested-3PT Maker", "category": "three_pt"},
    "pull_up_3pt_machine": {"name": "Pull-Up 3PT Machine", "category": "three_pt"},
    "volume_3pt_shooter": {"name": "Volume 3PT Shooter", "category": "three_pt"},
    "three_pt_sniper": {"name": "3PT Sniper", "category": "three_pt"},
    "volume_mid_range_shooter": {"name": "Volume Mid-Range Shooter", "category": "midrange"},
    "mid_range_assassin": {"name": "Mid-Range Assassin", "category": "midrange"},
    "volume_slasher": {"name": "Volume Slasher", "category": "rim_pressure"},
    "efficient_driver": {"name": "Efficient Driver", "category": "rim_pressure"},
    "free_throw_generator": {"name": "Free Throw Generator", "category": "rim_pressure"},
    "drive_and_kicker": {"name": "Drive and Kicker", "category": "playmaking"},
    "inside_the_arc_scorer": {"name": "Inside-The-Arc Scorer", "category": "rim_pressure"},
    "rim_finisher": {"name": "Rim Finisher", "category": "interior"},
    "paint_craftsman": {"name": "Paint Craftsman", "category": "interior"},
    "cut_finisher": {"name": "Lob and Cut Finisher", "category": "interior"},
    "inside_out_threat": {"name": "Inside-Out Threat", "category": "three_pt"},
    "screen_assist_machine": {"name": "Screen Assist Machine", "category": "playmaking"},
    "rim_protector": {"name": "Rim Protector", "category": "defense"},
    "perimeter_stopper": {"name": "Perimeter Stopper", "category": "defense"},
    "walking_bucket": {"name": "Walking Bucket", "category": "scoring"},
    "dunker": {"name": "Dunker", "category": "rim_pressure"},
    "active_hands": {"name": "Active Hands", "category": "defense"},
    "defensive_lock_down": {"name": "Defensive Lock-Down", "category": "defense"},
    "assist_generator": {"name": "Assist Generator", "category": "playmaking"},
    "efficient_passer": {"name": "Efficient Passer", "category": "playmaking"},
}


def normalize_player_name_for_badges(player_name: object) -> str:
    text = "" if player_name is None else str(player_name)
    text = text.replace("ı", "i").replace("İ", "I").replace("ø", "o").replace("Ø", "O").replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_numeric_frame(dataframe: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    numeric_frame = dataframe[feature_names].apply(pd.to_numeric, errors="coerce")
    return numeric_frame.replace([np.inf, -np.inf], np.nan)


def compute_feature_percentile_by_season(feature_values: pd.Series, seasons: pd.Series, lower_is_better: bool = False, peer_mask: Optional[pd.Series] = None, min_peer_count: int = 5) -> pd.Series:
    percentile_series = pd.Series(0.0, index=feature_values.index, dtype=float)
    if peer_mask is None:
        peer_mask = pd.Series(True, index=feature_values.index, dtype=bool)
    peer_mask = peer_mask.reindex(feature_values.index).fillna(False).astype(bool)

    for _, season_indices in seasons.groupby(seasons).groups.items():
        season_indices = pd.Index(season_indices)
        season_values = feature_values.loc[season_indices]
        season_peer_indices = season_indices[peer_mask.loc[season_indices].to_numpy()]
        peer_values = feature_values.loc[season_peer_indices].dropna()

        if len(peer_values) < min_peer_count:
            peer_values = season_values.dropna()
        if peer_values.empty:
            continue

        ranked_peer_values = peer_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
        percentile_series.loc[ranked_peer_values.index] = ranked_peer_values.astype(float)

        non_peer_indices = season_indices.difference(ranked_peer_values.index)
        for row_index in non_peer_indices:
            player_value = feature_values.loc[row_index]
            if pd.isna(player_value):
                percentile_series.loc[row_index] = 0.0
                continue
            comparison_values = pd.concat([peer_values, pd.Series([float(player_value)], index=["__player__"])])
            ranked = comparison_values.rank(method="average", pct=True, ascending=not lower_is_better) * 100.0
            percentile_series.loc[row_index] = float(ranked.loc["__player__"])

    return percentile_series.fillna(0.0)


def compute_percentile_frame(dataframe: pd.DataFrame, seasons: pd.Series, feature_names: List[str], peer_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    numeric_frame = _clean_numeric_frame(dataframe, feature_names)
    percentile_frame = pd.DataFrame(index=dataframe.index)
    for feature_name in feature_names:
        percentile_frame[feature_name] = compute_feature_percentile_by_season(numeric_frame[feature_name], seasons, lower_is_better=feature_name in BADGE_LOWER_IS_BETTER_FEATURES, peer_mask=peer_mask)
    return percentile_frame.fillna(0.0)


def compute_local_percentile(dataframe: pd.DataFrame, numeric_frame: pd.DataFrame, percentile_frame: pd.DataFrame, row_index: int, target_feature: str, anchor_feature: str, mode: str = "floor_to_100", lower_is_better: Optional[bool] = None, peer_gate_feature: Optional[str] = None, peer_gate_percentile_min: Optional[float] = None, min_peer_count: int = 5, peer_mask: Optional[pd.Series] = None) -> float:
    if row_index not in dataframe.index or target_feature not in numeric_frame.columns or anchor_feature not in percentile_frame.columns:
        return 0.0
    player_value = numeric_frame.at[row_index, target_feature]
    player_anchor_percentile = percentile_frame.at[row_index, anchor_feature]
    if pd.isna(player_value) or pd.isna(player_anchor_percentile):
        return 0.0
    if peer_mask is None:
        peer_mask = dataframe["_badge_peer_eligible"].astype(bool) if "_badge_peer_eligible" in dataframe.columns else pd.Series(True, index=dataframe.index, dtype=bool)
    peer_mask = peer_mask.reindex(dataframe.index).fillna(False).astype(bool)
    season_mask = dataframe["Season"].astype(str).eq(str(dataframe.at[row_index, "Season"]))
    lower_bound = max(0.0, float(player_anchor_percentile) - 10.0)
    if mode == "pm_10":
        upper_bound = min(100.0, float(player_anchor_percentile) + 10.0)
        local_mask = percentile_frame[anchor_feature].between(lower_bound, upper_bound, inclusive="both")
    else:
        local_mask = percentile_frame[anchor_feature] >= lower_bound
    selected_mask = season_mask & peer_mask & local_mask
    if peer_gate_feature is not None and peer_gate_percentile_min is not None and peer_gate_feature in percentile_frame.columns:
        selected_mask = selected_mask & (percentile_frame[peer_gate_feature] >= float(peer_gate_percentile_min))
    peer_values = numeric_frame.loc[selected_mask, target_feature].dropna()
    fallback_mask = season_mask & peer_mask
    if peer_gate_feature is not None and peer_gate_percentile_min is not None and peer_gate_feature in percentile_frame.columns:
        fallback_mask = fallback_mask & (percentile_frame[peer_gate_feature] >= float(peer_gate_percentile_min))
    if len(peer_values) < min_peer_count:
        peer_values = numeric_frame.loc[fallback_mask, target_feature].dropna()
    if len(peer_values) < min_peer_count:
        peer_values = numeric_frame.loc[season_mask & peer_mask, target_feature].dropna()
    if peer_values.empty:
        peer_values = numeric_frame.loc[season_mask, target_feature].dropna()
    if peer_values.empty:
        return 0.0
    lower_flag = target_feature in BADGE_LOWER_IS_BETTER_FEATURES if lower_is_better is None else bool(lower_is_better)
    if row_index in peer_values.index:
        ranked = peer_values.rank(method="average", pct=True, ascending=not lower_flag) * 100.0
        return float(ranked.loc[row_index])
    comparison_values = pd.concat([peer_values, pd.Series([float(player_value)], index=["__player__"])])
    ranked = comparison_values.rank(method="average", pct=True, ascending=not lower_flag) * 100.0
    return float(ranked.loc["__player__"])

def median_score(component_values: List[float]) -> float:
    cleaned_values = [float(value) for value in component_values if value is not None and not pd.isna(value)]
    if not cleaned_values:
        return 0.0
    return float(np.median(cleaned_values))


def tier_from_score(score: float, diamond: float, gold: float, silver: float, bronze: Optional[float] = None) -> Optional[str]:
    if score >= diamond:
        return "diamond"
    if score >= gold:
        return "gold"
    if score >= silver:
        return "silver"
    if bronze is not None and score >= bronze:
        return "bronze"
    return None



# ---------------------------------------------------------------------------
# Tier thresholds
#
# A badge score is already a same-season percentile taken against ALL player
# seasons, so a threshold is directly a rarity statement. These numbers are not
# hand-picked: they are solved so every badge lands on about the same share of
# the league, which stops a badge from becoming ordinary just because the skill
# underneath it is common.
#
#   diamond  ~0.6% of the league     gold    ~2.4% (cumulative)
#   silver   ~6%   (cumulative)      bronze  ~12%  (cumulative)
#
# Bronze is therefore "great at this", diamond is "the best in the league at
# this". Players with no elite or near-elite skill earn no badges, which is the
# intended behaviour rather than a gap to be filled.
#
# When a badge's opportunity gate is tighter than the 12% target, the gate is
# doing the work and everyone who clears it earns at least bronze.
# Re-solve with scripts/calibrate_badge_thresholds.py after a data refresh.
# ---------------------------------------------------------------------------
BADGE_TIER_THRESHOLDS = {
    "deep_range_bomber": (95.3, 89.7, 80.6, 50.2),
    "catch_and_shoot_converter": (93.0, 88.0, 80.4, 69.3),
    "contested_3pt_maker": (95.7, 93.0, 88.5, 81.8),
    "pull_up_3pt_machine": (94.7, 89.6, 83.1, 69.3),
    "volume_3pt_shooter": (99.2, 96.8, 93.0, 86.0),
    "three_pt_sniper": (95.6, 91.7, 86.1, 78.0),
    "volume_mid_range_shooter": (99.4, 97.5, 93.9, 87.6),
    "mid_range_assassin": (99.4, 97.1, 93.3, 86.3),
    "volume_slasher": (99.4, 97.6, 93.7, 87.5),
    "efficient_driver": (99.4, 97.0, 93.1, 87.0),
    "free_throw_generator": (99.4, 97.4, 93.7, 87.0),
    "drive_and_kicker": (98.7, 96.3, 92.3, 86.3),
    "inside_the_arc_scorer": (97.2, 93.5, 88.1, 80.5),
    "rim_finisher": (98.5, 95.8, 91.4, 82.4),
    "paint_craftsman": (95.5, 90.1, 82.5, 71.4),
    "cut_finisher": (98.3, 95.1, 90.5, 82.8),
    "inside_out_threat": (86.7, 75.3, 65.6, 55.0),
    "screen_assist_machine": (99.7, 97.7, 94.1, 88.1),
    "rim_protector": (98.1, 95.4, 91.0, 85.2),
    "perimeter_stopper": (95.0, 88.1, 78.1, 65.4),
    "walking_bucket": (99.4, 97.6, 94.0, 88.0),
    "dunker": (99.7, 97.7, 94.1, 88.1),
    "active_hands": (99.1, 96.7, 92.6, 86.0),
    "defensive_lock_down": (96.4, 92.4, 87.4, 80.9),
    "assist_generator": (99.5, 97.7, 94.1, 88.1),
    "efficient_passer": (96.2, 89.8, 82.3, 73.3),
}


def tier_for(badge_id: str, score: float) -> Optional[str]:
    """Tier a badge score against that badge's calibrated thresholds."""
    diamond, gold, silver, bronze = BADGE_TIER_THRESHOLDS[badge_id]
    return tier_from_score(score, diamond, gold, silver, bronze)


def demote_tier(tier: Optional[str], steps: int = 1) -> Optional[str]:
    if tier is None:
        return None
    current_index = BADGE_TIER_RANK[tier]
    next_index = current_index + int(steps)
    if next_index >= len(BADGE_TIER_ORDER):
        return None
    return BADGE_TIER_ORDER[next_index]


def cap_tier(tier: Optional[str], max_tier: str) -> Optional[str]:
    if tier is None:
        return None
    return tier if BADGE_TIER_RANK[tier] >= BADGE_TIER_RANK[max_tier] else max_tier


def better_tier(current_tier: Optional[str], candidate_tier: Optional[str]) -> Optional[str]:
    if candidate_tier is None:
        return current_tier
    if current_tier is None:
        return candidate_tier
    return candidate_tier if BADGE_TIER_RANK[candidate_tier] < BADGE_TIER_RANK[current_tier] else current_tier


def build_badge_payload(
    badge_id: str,
    tier: Optional[str],
    score: float,
    components: Dict[str, float],
    demotion_reasons: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    if tier is None:
        return None
    definition = BADGE_DEFINITIONS[badge_id]
    return {
        "id": badge_id,
        "name": definition["name"],
        "tier": tier,
        "category": definition["category"],
        "score_percentile": round(float(score), 3),
        "components": {key: round(float(value), 3) for key, value in components.items()},
        "demotion_reasons": demotion_reasons or [],
    }


def _pct(percentile_frame: pd.DataFrame, row_index: int, feature_name: str) -> float:
    return float(percentile_frame.at[row_index, feature_name]) if feature_name in percentile_frame.columns else 0.0


def _raw(numeric_frame: pd.DataFrame, row_index: int, feature_name: str) -> float:
    if feature_name not in numeric_frame.columns:
        return float("nan")
    value = numeric_frame.at[row_index, feature_name]
    return float(value) if pd.notna(value) else float("nan")


def _below_average(value: float) -> bool:
    return pd.isna(value) or float(value) < 50.0


def _season_zscore_series(dataframe: pd.DataFrame, numeric_frame: pd.DataFrame, feature_name: str, lower_is_better: bool = False, peer_mask: Optional[pd.Series] = None) -> pd.Series:
    zscore_series = pd.Series(0.0, index=dataframe.index, dtype=float)
    if peer_mask is None:
        peer_mask = dataframe["_badge_peer_eligible"].astype(bool) if "_badge_peer_eligible" in dataframe.columns else pd.Series(True, index=dataframe.index, dtype=bool)
    peer_mask = peer_mask.reindex(dataframe.index).fillna(False).astype(bool)
    for _, season_indices in dataframe["Season"].groupby(dataframe["Season"]).groups.items():
        season_indices = pd.Index(season_indices)
        peer_indices = season_indices[peer_mask.loc[season_indices].to_numpy()]
        peer_values = numeric_frame.loc[peer_indices, feature_name].dropna()
        if len(peer_values) < 2:
            peer_values = numeric_frame.loc[season_indices, feature_name].dropna()
        if peer_values.empty:
            continue
        center = float(peer_values.mean())
        scale = float(peer_values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        z = (numeric_frame.loc[season_indices, feature_name].astype(float) - center) / scale
        if lower_is_better:
            z = -z
        zscore_series.loc[season_indices] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return zscore_series.fillna(0.0)


def _season_zscore_sum_percentile(dataframe: pd.DataFrame, numeric_frame: pd.DataFrame, feature_names: List[str], peer_mask: Optional[pd.Series] = None) -> pd.Series:
    zsum = pd.Series(0.0, index=dataframe.index, dtype=float)
    for feature_name in feature_names:
        zsum = zsum + _season_zscore_series(dataframe, numeric_frame, feature_name, lower_is_better=feature_name in BADGE_LOWER_IS_BETTER_FEATURES, peer_mask=peer_mask)
    return compute_feature_percentile_by_season(zsum, dataframe["Season"], lower_is_better=False, peer_mask=peer_mask)


# Overall defensive activity still counts blocks: Defensive Lock-Down rewards
# defensive value wherever it comes from.
DEFENSIVE_LOCKDOWN_ACTIVITY_FEATURES = [
    "Blocks_per_75",
    "Steals_per_75",
    "Deflections_per_75",
    "off_fouls_drawn_frequency",
]

# Active Hands is the perimeter-disruption badge and deliberately excludes
# blocks, which are Rim Protector territory. Splitting them this way stops one
# league-wide percentile pool from collapsing every defensive badge onto centers.
ACTIVE_HANDS_FEATURES = [
    "Steals_per_75",
    "Deflections_per_75",
    "off_fouls_drawn_frequency",
]


def compute_defensive_lockdown_components(dataframe: pd.DataFrame, numeric_frame: pd.DataFrame, percentile_frame: pd.DataFrame, row_index: int, peer_mask: Optional[pd.Series] = None) -> Tuple[float, Dict[str, float]]:
    frame = compute_defensive_lockdown_component_frame(
        dataframe=dataframe,
        numeric_frame=numeric_frame,
        percentile_frame=percentile_frame,
        peer_mask=peer_mask,
    )
    components = {
        "opp_players_fg_pct_difference": float(frame.at[row_index, "opp_players_fg_pct_difference"]),
        "avg_D-LEBRON_crafted_cdpm": float(frame.at[row_index, "avg_D-LEBRON_crafted_cdpm"]),
        "D-LEBRON": float(frame.at[row_index, "D-LEBRON"]),
        "crafted_cdpm": float(frame.at[row_index, "crafted_cdpm"]),
        "defensive_activity_zsum_percentile": float(frame.at[row_index, "defensive_activity_zsum_percentile"]),
    }
    return float(frame.at[row_index, "score"]), components


def compute_defensive_lockdown_component_frame(
    dataframe: pd.DataFrame,
    numeric_frame: pd.DataFrame,
    percentile_frame: pd.DataFrame,
    peer_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Precompute Defensive Lock-Down components once per badge run.

    Formula:
      15% opp_players_fg_pct_difference percentile, lower-is-better
      65% average of same-season D-LEBRON and crafted_cdpm percentiles
      20% percentile of the summed same-season z-scores for defensive activity
    """
    if peer_mask is None:
        peer_mask = dataframe["_badge_peer_eligible"].astype(bool) if "_badge_peer_eligible" in dataframe.columns else pd.Series(True, index=dataframe.index, dtype=bool)

    opp_fg_component = percentile_frame["opp_players_fg_pct_difference"].astype(float).fillna(0.0)
    dlebron_component = percentile_frame["D-LEBRON"].astype(float).fillna(0.0)
    cdpm_component = percentile_frame["crafted_cdpm"].astype(float).fillna(0.0)
    defensive_metric_average = ((dlebron_component + cdpm_component) / 2.0).fillna(0.0)

    activity_sum = pd.Series(0.0, index=dataframe.index, dtype=float)
    for feature_name in DEFENSIVE_LOCKDOWN_ACTIVITY_FEATURES:
        activity_sum = activity_sum + _season_zscore_series(
            dataframe, numeric_frame, feature_name, peer_mask=peer_mask
        )
    activity_component = compute_feature_percentile_by_season(
        activity_sum, dataframe["Season"], peer_mask=peer_mask
    ).fillna(0.0).astype(float)

    score = (
        (0.1500 * opp_fg_component)
        + (0.6500 * defensive_metric_average)
        + (0.2000 * activity_component)
    )

    frame = pd.DataFrame(index=dataframe.index)
    frame["score"] = score.fillna(0.0).astype(float)
    frame["opp_players_fg_pct_difference"] = opp_fg_component.astype(float)
    frame["avg_D-LEBRON_crafted_cdpm"] = defensive_metric_average.astype(float)
    frame["D-LEBRON"] = dlebron_component.astype(float)
    frame["crafted_cdpm"] = cdpm_component.astype(float)
    frame["defensive_activity_zsum_percentile"] = activity_component.astype(float)
    return frame

def compute_badges_for_guards(guards: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    missing_features = [feature_name for feature_name in BADGE_REQUIRED_FEATURES if feature_name not in guards.columns]
    if missing_features:
        raise ValueError(f"Missing badge feature columns: {missing_features}")
    if "player_key" not in guards.columns:
        raise ValueError("Missing player_key column before badge computation.")

    dataframe = guards.copy()
    dataframe["_badge_normalized_name"] = dataframe["Player Name"].map(normalize_player_name_for_badges)
    dataframe["_badge_peer_eligible"] = ~dataframe["_badge_normalized_name"].isin(PERCENTILE_AND_BADGE_EXCLUDED_NAMES)
    badge_peer_mask = dataframe["_badge_peer_eligible"].astype(bool)
    numeric_frame = _clean_numeric_frame(dataframe, BADGE_REQUIRED_FEATURES)
    percentile_frame = compute_percentile_frame(dataframe, dataframe["Season"], BADGE_REQUIRED_FEATURES, peer_mask=badge_peer_mask)
    defensive_lockdown_frame = compute_defensive_lockdown_component_frame(
        dataframe, numeric_frame, percentile_frame, peer_mask=badge_peer_mask
    )

    badges_by_player_key: Dict[str, List[Dict[str, object]]] = {}

    for row_index, row in dataframe.iterrows():
        player_key = str(row["player_key"])
        normalized_name = normalize_player_name_for_badges(row.get("Player Name"))
        if normalized_name in PERCENTILE_AND_BADGE_EXCLUDED_NAMES:
            badges_by_player_key[player_key] = []
            continue

        row_badges: List[Dict[str, object]] = []

        def append_badge(badge: Optional[Dict[str, object]]) -> None:
            if badge is not None:
                row_badges.append(badge)

        # Shot-contest volume is the behavioural test for "does this player
        # defend the interior". Three badges gate on it, so it is computed once.
        contested_pct = _pct(percentile_frame, row_index, "contested_shot_frequency")

        # Deep Range Bomber
        local_3p_accuracy_deep = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="3P_Accuracy", anchor_feature="Avg3ptShotDistance", mode="floor_to_100"
        )
        components = {
            "Avg3ptShotDistance": _pct(percentile_frame, row_index, "Avg3ptShotDistance"),
            "3fga_frequency": _pct(percentile_frame, row_index, "3fga_frequency"),
            "3P_Accuracy_local_by_avg_3pt_distance_floor_to_100": local_3p_accuracy_deep,
        }
        score = median_score(list(components.values()))
        tier = None
        if components["Avg3ptShotDistance"] >= 80.0 and components["3fga_frequency"] >= 50.0:
            tier = tier_for("deep_range_bomber", score)
            if tier == "diamond" and components["Avg3ptShotDistance"] < 95.0:
                tier = "gold"
            demotions = []
            if local_3p_accuracy_deep < 50.0:
                tier = demote_tier(tier)
                demotions.append("Below-average local 3P accuracy after distance adjustment")
            append_badge(build_badge_payload("deep_range_bomber", tier, score, components, demotions))

        # Catch and Shoot Converter
        local_catch_accuracy = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="catch_shoot_3P_accuracy", anchor_feature="avg_closest_defender_3FGA", mode="floor_to_100"
        )
        components = {
            "catch_shoot_3P_frequency": _pct(percentile_frame, row_index, "catch_shoot_3P_frequency"),
            "catch_shoot_3P_accuracy_local_by_closest_defender_floor_to_100": local_catch_accuracy,
        }
        score = median_score(list(components.values()))
        if components["catch_shoot_3P_frequency"] >= 80.0:
            tier = tier_for("catch_and_shoot_converter", score)
            demotions = []
            if local_catch_accuracy < 50.0:
                tier = demote_tier(tier)
                demotions.append("Below-average catch-and-shoot accuracy after closest-defender adjustment")
            append_badge(build_badge_payload("catch_and_shoot_converter", tier, score, components, demotions))

        # Contested-3PT Maker
        local_contested_accuracy = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="3P_Accuracy", anchor_feature="avg_closest_defender_3FGA", mode="floor_to_100",
            peer_gate_feature="3fga_frequency", peer_gate_percentile_min=50.0,
        )
        components = {
            "pct_3fga_wide_open_lower_is_better": _pct(percentile_frame, row_index, "pct_3fga_wide_open"),
            "avg_closest_defender_3FGA_lower_is_better": _pct(percentile_frame, row_index, "avg_closest_defender_3FGA"),
            "3P_Accuracy_local_by_closest_defender_floor_to_100": local_contested_accuracy,
            "tight_very_tight_3fga_frequency": _pct(percentile_frame, row_index, "tight_very_tight_3fga_frequency"),
        }
        score = median_score(list(components.values()))
        if components["avg_closest_defender_3FGA_lower_is_better"] >= 75.0 and _pct(percentile_frame, row_index, "3fga_frequency") >= 50.0:
            tier = tier_for("contested_3pt_maker", score)
            demotions = []
            if any(_below_average(value) for value in components.values()):
                tier = demote_tier(tier)
                demotions.append("At least one contested-3PT component is below average")
            if tier in {"diamond", "gold"} and (_raw(numeric_frame, row_index, "3P_Accuracy") < 0.335):
                tier = demote_tier(tier)
                demotions.append("3P accuracy below .335 gold/diamond gate")
            append_badge(build_badge_payload("contested_3pt_maker", tier, score, components, demotions))

        # Pull-Up 3PT Machine
        local_pullup_accuracy = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="pull_up_3P_accuracy", anchor_feature="pull_up_3P_frequency", mode="floor_to_100",
            peer_gate_feature="3fga_frequency", peer_gate_percentile_min=50.0,
        )
        components = {
            "pull_up_3P_frequency": _pct(percentile_frame, row_index, "pull_up_3P_frequency"),
            "pull_up_3P_accuracy_local_by_pull_up_frequency_floor_to_100": local_pullup_accuracy,
        }
        score = median_score(list(components.values()))
        if components["pull_up_3P_frequency"] >= 75.0:
            tier = tier_for("pull_up_3pt_machine", score)
            if components["pull_up_3P_frequency"] >= 95.0:
                if local_pullup_accuracy >= 75.0:
                    tier = "diamond"
                elif local_pullup_accuracy >= 60.0 and tier not in {"diamond"}:
                    tier = "gold"
                elif local_pullup_accuracy >= 40.0 and tier not in {"diamond", "gold"}:
                    tier = "silver"
                elif tier is None:
                    tier = "bronze"
            demotions = []
            if any(_below_average(value) for value in components.values()):
                tier = demote_tier(tier)
                demotions.append("At least one pull-up 3PT component is below average")
            append_badge(build_badge_payload("pull_up_3pt_machine", tier, score, components, demotions))

        # Volume 3PT Shooter
        components = {
            "3fga_frequency": _pct(percentile_frame, row_index, "3fga_frequency"),
            "traditional_fg3a": _pct(percentile_frame, row_index, "traditional_fg3a"),
        }
        score = median_score(list(components.values()))
        if components["3fga_frequency"] >= 75.0:
            tier = tier_for("volume_3pt_shooter", score)
            demotions = []
            if any(_below_average(value) for value in components.values()):
                tier = demote_tier(tier)
                demotions.append("At least one volume 3PT component is below average")
            append_badge(build_badge_payload("volume_3pt_shooter", tier, score, components, demotions))

        # 3PT Sniper
        local_3p_accuracy_sniper = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="3P_Accuracy", anchor_feature="avg_closest_defender_3FGA", mode="floor_to_100",
        )
        components = {
            "3P_Accuracy_local_by_closest_defender_floor_to_100": local_3p_accuracy_sniper,
            "3fga_frequency": _pct(percentile_frame, row_index, "3fga_frequency"),
        }
        score = (0.70 * components["3P_Accuracy_local_by_closest_defender_floor_to_100"]) + (0.30 * components["3fga_frequency"])
        if _pct(percentile_frame, row_index, "3P_Accuracy") >= 60.0 and components["3fga_frequency"] >= 60.0:
            tier = tier_for("three_pt_sniper", score)
            append_badge(build_badge_payload("three_pt_sniper", tier, score, components))

        # Volume Mid-Range Shooter
        components = {
            "MidRangeFrequency": _pct(percentile_frame, row_index, "MidRangeFrequency"),
            "by_zone_statistics_mid_range_fga": _pct(percentile_frame, row_index, "by_zone_statistics_mid_range_fga"),
        }
        score = median_score(list(components.values()))
        if components["MidRangeFrequency"] >= 75.0:
            tier = tier_for("volume_mid_range_shooter", score)
            demotions = []
            if any(_below_average(value) for value in components.values()):
                tier = demote_tier(tier)
                demotions.append("At least one mid-range volume component is below average")
            append_badge(build_badge_payload("volume_mid_range_shooter", tier, score, components, demotions))

        # Mid-Range Assassin
        local_mid_accuracy = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="MidRangeAccuracy", anchor_feature="tight_very_tight_2fga_frequency", mode="floor_to_100",
            peer_gate_feature="MidRangeFrequency", peer_gate_percentile_min=50.0,
        )
        components = {
            "MidRangeFrequency": _pct(percentile_frame, row_index, "MidRangeFrequency"),
            "MidRangeAccuracy_local_by_tight_2fga_floor_to_100": local_mid_accuracy,
            "by_zone_statistics_mid_range_fga": _pct(percentile_frame, row_index, "by_zone_statistics_mid_range_fga"),
        }
        score = median_score(list(components.values()))
        if components["MidRangeFrequency"] >= 75.0:
            tier = tier_for("mid_range_assassin", score)
            demotions = []
            if local_mid_accuracy < 50.0:
                tier = demote_tier(tier)
                demotions.append("Below-average mid-range accuracy after tight-2FGA adjustment")
            append_badge(build_badge_payload("mid_range_assassin", tier, score, components, demotions))

        # Volume Slasher
        components = {
            "drives_drive_fga": _pct(percentile_frame, row_index, "drives_drive_fga"),
            "drives_drives": _pct(percentile_frame, row_index, "drives_drives"),
            "pts_from_drives_per_75": _pct(percentile_frame, row_index, "pts_from_drives_per_75"),
        }
        score = median_score(list(components.values()))
        if components["pts_from_drives_per_75"] >= 75.0:
            tier = tier_for("volume_slasher", score)
            demotions = []
            if components["pts_from_drives_per_75"] < 50.0 or components["drives_drive_fga"] < 50.0:
                tier = demote_tier(tier)
                demotions.append("Drive FGA or drive points component is below average")
            append_badge(build_badge_payload("volume_slasher", tier, score, components, demotions))

        # Efficient Driver
        local_drive_fg = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="drive_fg_pct", anchor_feature="drive_fga_frequency", mode="floor_to_100"
        )
        components = {
            "drive_fga_frequency": _pct(percentile_frame, row_index, "drive_fga_frequency"),
            "drive_fg_pct_local_by_drive_fga_frequency_floor_to_100": local_drive_fg,
            "pts_from_drives_per_75": _pct(percentile_frame, row_index, "pts_from_drives_per_75"),
        }
        score = median_score(list(components.values()))
        if components["pts_from_drives_per_75"] >= 65.0:
            tier = tier_for("efficient_driver", score)
            demotions = []
            if components["drive_fga_frequency"] < 50.0 or local_drive_fg < 50.0:
                tier = demote_tier(tier)
                demotions.append("Drive volume or local drive efficiency is below average")
            append_badge(build_badge_payload("efficient_driver", tier, score, components, demotions))

        # Free Throw Generator
        components = {
            "traditional_fta": _pct(percentile_frame, row_index, "traditional_fta"),
            "fta_rate": _pct(percentile_frame, row_index, "fta_rate"),
            "fta_per_75": _pct(percentile_frame, row_index, "fta_per_75"),
        }
        score = median_score(list(components.values()))
        if components["fta_rate"] >= 65.0:
            tier = tier_for("free_throw_generator", score)
            demotions = []
            if components["fta_per_75"] < 50.0:
                tier = demote_tier(tier)
                demotions.append("FTA per 75 is below average")
            append_badge(build_badge_payload("free_throw_generator", tier, score, components, demotions))

        # Drive and Kicker
        local_drive_tov = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="drive_tov_per_75", anchor_feature="drives_per_75", mode="floor_to_100", lower_is_better=True
        )
        drive_ast_pct = _pct(percentile_frame, row_index, "drive_ast_per_75")
        components = {
            "drive_ast_per_75": drive_ast_pct,
            "drives_per_75": _pct(percentile_frame, row_index, "drives_per_75"),
            "drive_tov_per_75_local_lower_is_better_by_drives_per_75_floor_to_100": local_drive_tov,
            "drive_passes_per_75": _pct(percentile_frame, row_index, "drive_passes_per_75"),
            "drives_drive_ast": _pct(percentile_frame, row_index, "drives_drive_ast"),
            "drives_drives": _pct(percentile_frame, row_index, "drives_drives"),
        }
        score = median_score(list(components.values()))
        if drive_ast_pct >= 65.0:
            tier = tier_for("drive_and_kicker", score)
            demotions = []
            if drive_ast_pct < 50.0 or (local_drive_tov < 50.0 and not (drive_ast_pct >= 97.0 and local_drive_tov >= 25.0)):
                tier = demote_tier(tier)
                demotions.append("Drive assists or adjusted drive turnover control failed gate")
            append_badge(build_badge_payload("drive_and_kicker", tier, score, components, demotions))

        # Inside-The-Arc Scorer
        local_mid_accuracy_arc = local_mid_accuracy
        local_drive_fg_arc = local_drive_fg
        components = {
            "drives_drive_fga": _pct(percentile_frame, row_index, "drives_drive_fga"),
            "by_zone_statistics_mid_range_fga": _pct(percentile_frame, row_index, "by_zone_statistics_mid_range_fga"),
            "MidRangeAccuracy_local_by_tight_2fga_floor_to_100": local_mid_accuracy_arc,
            "MidRangeFrequency": _pct(percentile_frame, row_index, "MidRangeFrequency"),
            "drive_fga_frequency": _pct(percentile_frame, row_index, "drive_fga_frequency"),
            "drive_fg_pct_local_by_drive_fga_frequency_floor_to_100": local_drive_fg_arc,
            "pts_from_midrange_per_75": _pct(percentile_frame, row_index, "pts_from_midrange_per_75"),
            "pts_from_drives_per_75": _pct(percentile_frame, row_index, "pts_from_drives_per_75"),
        }
        score = median_score(list(components.values()))
        if components["drive_fga_frequency"] >= 60.0 and components["MidRangeFrequency"] >= 60.0:
            tier = tier_for("inside_the_arc_scorer", score)
            append_badge(build_badge_payload("inside_the_arc_scorer", tier, score, components))

        # Rim Finisher
        local_ra_accuracy = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="RestrictedArea_Accuracy", anchor_feature="RestrictedArea_Frequency",
            mode="floor_to_100",
        )
        components = {
            "RestrictedArea_Frequency": _pct(percentile_frame, row_index, "RestrictedArea_Frequency"),
            "RestrictedArea_Accuracy_local_by_restricted_area_frequency_floor_to_100": local_ra_accuracy,
            "dunks_per_75_poss": _pct(percentile_frame, row_index, "dunks_per_75_poss"),
        }
        score = median_score(list(components.values()))
        if components["RestrictedArea_Frequency"] >= 70.0:
            tier = tier_for("rim_finisher", score)
            demotions = []
            if local_ra_accuracy < 50.0:
                tier = demote_tier(tier)
                demotions.append("Below-average restricted-area accuracy after volume adjustment")
            append_badge(build_badge_payload("rim_finisher", tier, score, components, demotions))

        # Paint Craftsman
        local_paint_accuracy = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="Paint_Non_RA_Accuracy", anchor_feature="Paint_Non_RA_Frequency",
            mode="floor_to_100",
        )
        components = {
            "Paint_Non_RA_Frequency": _pct(percentile_frame, row_index, "Paint_Non_RA_Frequency"),
            "Paint_Non_RA_Accuracy_local_by_paint_non_ra_frequency_floor_to_100": local_paint_accuracy,
        }
        score = median_score(list(components.values()))
        if components["Paint_Non_RA_Frequency"] >= 70.0:
            tier = tier_for("paint_craftsman", score)
            demotions = []
            if local_paint_accuracy < 50.0:
                tier = demote_tier(tier)
                demotions.append("Below-average non-restricted-area paint accuracy after volume adjustment")
            append_badge(build_badge_payload("paint_craftsman", tier, score, components, demotions))

        # Lob and Cut Finisher
        local_cut_ppp = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="cut_ppp", anchor_feature="cut_frequency", mode="floor_to_100",
        )
        components = {
            "cut_frequency": _pct(percentile_frame, row_index, "cut_frequency"),
            "cut_ppp_local_by_cut_frequency_floor_to_100": local_cut_ppp,
            "dunks_per_75_poss": _pct(percentile_frame, row_index, "dunks_per_75_poss"),
        }
        score = median_score(list(components.values()))
        if components["cut_frequency"] >= 75.0:
            tier = tier_for("cut_finisher", score)
            demotions = []
            if local_cut_ppp < 50.0:
                tier = demote_tier(tier)
                demotions.append("Below-average cut efficiency after cut-volume adjustment")
            append_badge(build_badge_payload("cut_finisher", tier, score, components, demotions))

        # Inside-Out Threat
        # Gated on behaviour rather than on a listed position. Attempting shots
        # at the rim does NOT identify an interior player, because driving guards
        # do that too; what identifies one is defending the interior. So the gate
        # pairs the same contest-volume test Rim Protector uses with real
        # three-point volume: a player who guards the paint and spaces the floor.
        local_3p_accuracy_inside_out = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="3P_Accuracy", anchor_feature="3fga_frequency", mode="floor_to_100",
        )
        components = {
            "contested_shot_frequency": contested_pct,
            "3fga_frequency": _pct(percentile_frame, row_index, "3fga_frequency"),
            "3P_Accuracy_local_by_3fga_frequency_floor_to_100": local_3p_accuracy_inside_out,
        }
        score = median_score(list(components.values()))
        if contested_pct >= 65.0 and components["3fga_frequency"] >= 55.0:
            tier = tier_for("inside_out_threat", score)
            demotions = []
            if local_3p_accuracy_inside_out < 50.0:
                tier = demote_tier(tier)
                demotions.append("Below-average three-point accuracy after volume adjustment")
            append_badge(build_badge_payload("inside_out_threat", tier, score, components, demotions))

        # Screen Assist Machine
        components = {
            "screen_assist_points_per_game": _pct(percentile_frame, row_index, "screen_assist_points_per_game"),
        }
        score = components["screen_assist_points_per_game"]
        if score >= 70.0:
            tier = tier_for("screen_assist_machine", score)
            append_badge(build_badge_payload("screen_assist_machine", tier, score, components))

        # Walking Bucket
        components = {
            "pts_per_75": _pct(percentile_frame, row_index, "pts_per_75"),
            "fga_per_75": _pct(percentile_frame, row_index, "fga_per_75"),
            "traditional_pts": _pct(percentile_frame, row_index, "traditional_pts"),
        }
        score = median_score(list(components.values()))
        if components["fga_per_75"] >= 65.0:
            tier = tier_for("walking_bucket", score)
            demotions = []
            if any(_below_average(value) for value in components.values()):
                tier = demote_tier(tier)
                demotions.append("At least one high-volume scoring component is below average")
            append_badge(build_badge_payload("walking_bucket", tier, score, components, demotions))

        # Dunker
        components = {"dunks_per_75_poss": _pct(percentile_frame, row_index, "dunks_per_75_poss")}
        score = components["dunks_per_75_poss"]
        if score >= 70.0:
            tier = tier_for("dunker", score)
            append_badge(build_badge_payload("dunker", tier, score, components))

        # Active Hands
        components = {
            feature_name: _pct(percentile_frame, row_index, feature_name)
            for feature_name in ACTIVE_HANDS_FEATURES
        }
        score = median_score(list(components.values()))
        tier = tier_for("active_hands", score)
        append_badge(build_badge_payload("active_hands", tier, score, components))

        # Rim Protector
        # Opponent FG% difference is compared only against players who contest a
        # similar share of shots. Contesting at the rim produces a worse raw FG%
        # difference than contesting on the perimeter simply because rim attempts
        # go in more often, so the unadjusted feature punishes exactly the players
        # doing the most interior work.
        local_opp_fg_by_contest = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="opp_players_fg_pct_difference", anchor_feature="contested_shot_frequency",
            mode="pm_10", lower_is_better=True,
        )
        components = {
            "Blocks_per_75": _pct(percentile_frame, row_index, "Blocks_per_75"),
            "contested_shot_frequency": contested_pct,
            "Opp_players_fga_per_75_poss": _pct(percentile_frame, row_index, "Opp_players_fga_per_75_poss"),
            "opp_players_fg_pct_difference_local_by_contested_frequency_pm_10": local_opp_fg_by_contest,
        }
        score = median_score(list(components.values()))
        if contested_pct >= 65.0:
            tier = tier_for("rim_protector", score)
            demotions = []
            if components["Blocks_per_75"] < 50.0:
                tier = demote_tier(tier)
                demotions.append("Blocks per 75 is below average")
            if local_opp_fg_by_contest < 40.0:
                tier = demote_tier(tier)
                demotions.append("Opponents shoot better than expected for this contest volume")
            append_badge(build_badge_payload("rim_protector", tier, score, components, demotions))

        # Perimeter Stopper
        # The mirror of Rim Protector. Both gate on where a player defends, which
        # is behaviour rather than a listed position, so the two badges partition
        # the league by defensive role instead of by height.
        components_opponent_volume_pct = _pct(percentile_frame, row_index, "Opp_players_fga_per_75_poss")
        components = {
            "Deflections_per_75": _pct(percentile_frame, row_index, "Deflections_per_75"),
            "Steals_per_75": _pct(percentile_frame, row_index, "Steals_per_75"),
            "opp_players_fg_pct_difference_local_by_contested_frequency_pm_10": local_opp_fg_by_contest,
            "D-LEBRON": _pct(percentile_frame, row_index, "D-LEBRON"),
        }
        score = median_score(list(components.values()))
        # Contest rate alone is not enough: a low-effort centre can slip under a
        # contest gate. Opponent FGA per 75 is what actually separates interior
        # from perimeter assignments, so both have to say "perimeter".
        if contested_pct <= 55.0 and components_opponent_volume_pct <= 55.0:
            tier = tier_for("perimeter_stopper", score)
            demotions = []
            if local_opp_fg_by_contest < 40.0:
                tier = demote_tier(tier)
                demotions.append("Opponents shoot better than expected for this contest volume")
            append_badge(build_badge_payload("perimeter_stopper", tier, score, components, demotions))

        # Defensive Lock-Down
        score = float(defensive_lockdown_frame.at[row_index, "score"])
        components = {
            "opp_players_fg_pct_difference": float(defensive_lockdown_frame.at[row_index, "opp_players_fg_pct_difference"]),
            "avg_D-LEBRON_crafted_cdpm": float(defensive_lockdown_frame.at[row_index, "avg_D-LEBRON_crafted_cdpm"]),
            "D-LEBRON": float(defensive_lockdown_frame.at[row_index, "D-LEBRON"]),
            "crafted_cdpm": float(defensive_lockdown_frame.at[row_index, "crafted_cdpm"]),
            "defensive_activity_zsum_percentile": float(defensive_lockdown_frame.at[row_index, "defensive_activity_zsum_percentile"]),
        }
        defensive_metric_average_pct = components["avg_D-LEBRON_crafted_cdpm"]
        tier = tier_for("defensive_lock_down", score)
        # A player whose two headline defensive-impact metrics are both elite
        # should not be held down by the activity components, but the promotion
        # has to be as rare as the tier it grants, or it overwrites the
        # calibrated tiering for a tenth of the league.
        if defensive_metric_average_pct >= 99.4:
            tier = better_tier(tier, "diamond")
        elif defensive_metric_average_pct >= 97.5:
            tier = better_tier(tier, "gold")
        append_badge(build_badge_payload("defensive_lock_down", tier, score, components))

        # Assist Generator
        components = {
            "assist_frequency": _pct(percentile_frame, row_index, "assist_frequency"),
            "potential_assist_frequency": _pct(percentile_frame, row_index, "potential_assist_frequency"),
        }
        score = median_score(list(components.values()))
        if components["assist_frequency"] >= 70.0:
            tier = tier_for("assist_generator", score)
            append_badge(build_badge_payload("assist_generator", tier, score, components))

        # Efficient Passer
        local_potential_ast_tov = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="potential_assist_tov_ratio", anchor_feature="potential_assist_frequency", mode="floor_to_100"
        )
        local_assists_tov = compute_local_percentile(
            dataframe, numeric_frame, percentile_frame, row_index,
            target_feature="assists_tov_ratio", anchor_feature="assist_frequency", mode="floor_to_100"
        )
        components = {
            "assist_frequency": _pct(percentile_frame, row_index, "assist_frequency"),
            "potential_assist_frequency": _pct(percentile_frame, row_index, "potential_assist_frequency"),
            "potential_assist_tov_ratio_local_by_potential_assist_frequency_floor_to_100": local_potential_ast_tov,
            "assists_tov_ratio_local_by_assist_frequency_floor_to_100": local_assists_tov,
        }
        score = median_score(list(components.values()))
        if components["potential_assist_frequency"] >= 65.0:
            tier = tier_for("efficient_passer", score)
            append_badge(build_badge_payload("efficient_passer", tier, score, components))

        row_badges.sort(
            key=lambda badge: (
                BADGE_TIER_RANK.get(str(badge["tier"]), 999),
                BADGE_CATEGORY_ORDER.get(str(badge.get("category", "")), 999),
                -float(badge.get("score_percentile", 0.0)),
                str(badge.get("name", "")),
            )
        )
        badges_by_player_key[player_key] = row_badges

    return badges_by_player_key


def build_badge_rows(guards: pd.DataFrame, badges_by_player_key: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    meta_by_key = guards.set_index("player_key", drop=False)
    for player_key, badges in badges_by_player_key.items():
        if player_key not in meta_by_key.index:
            continue
        row = meta_by_key.loc[player_key]
        for badge in badges:
            rows.append(
                {
                    "player_key": str(player_key),
                    "Player Name": str(row.get("Player Name", "")),
                    "Season": str(row.get("Season", "")),
                    "teams_played": str(row.get("teams_played", "")),
                    "position": str(row.get("position", "")),
                    "badge_id": badge["id"],
                    "badge_name": badge["name"],
                    "badge_tier": badge["tier"],
                    "badge_score_percentile": badge["score_percentile"],
                    "component_percentiles_json": json.dumps(badge.get("components", {}), sort_keys=True),
                    "demotion_reasons_json": json.dumps(badge.get("demotion_reasons", [])),
                }
            )
    return rows
