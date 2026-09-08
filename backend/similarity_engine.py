"""v4 player-season similarity engine -- a port of sim.ipynb, cell `8a1e766d`.

This is the algorithm described in "NBA Player-Season Similarity Algorithm"
(Harsha Anand). The math is carried over from the notebook unchanged; the only
edits are structural:

  * ``DATA_PATH`` / ``IMPACT_PATHS`` resolve to the checked-in slim dataset
    (see scripts/build_similarity_dataset.py) and are overridable by env var.
  * the notebook's held-out validator, sharpening diagnostics, candidate
    evaluator, invariant tests and case-study tables are dropped -- none of them
    feed the production path, and the validator was already gated off behind
    SHOW_SETUP_DIAGNOSTICS with its RNG state restored, so removing it leaves
    every downstream draw bit-identical.
  * retrieval helpers at the bottom add a candidate mask, so comps can be
    restricted to the site's roster while the model still FITS on the full
    population (z-scores, PCA, peer percentiles and the continuity learner are
    all population statistics).

Pipeline, in the paper's order:
  1. within-season z-score every raw feature
  2. per-subgroup PCA -> whiten with a domain ridge
  3. non-negative logistic regression on same-player adjacent-season pairs,
     learned separately per position group x domain (six models)
  4. per-player distinctiveness vs same-season, same-position peers
  5. blend learned + distinctive weights, then adaptive hierarchical sharpening
  6. role gates on Paint Defense and Defensive Rebounding
  7. offense/defense identity balance from the LEBRON family
  8. symmetric pair-averaged distance + soft position penalty
  9. sim = 100 / (1 + (d/median)^2)

Importing this module FITS the model (a few seconds). It is used by the
precompute scripts only -- the web app serves the precomputed assets and never
imports this.
"""

from __future__ import annotations

import os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _REPO_ROOT / "backend" / "data" / "similarity_model_dataset.csv"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_PATH        = os.environ.get("SIMILARITY_MODEL_DATASET", str(_DEFAULT_DATASET))
RANDOM_STATE     = 0
SHOW_SETUP_DIAGNOSTICS = False  # keep notebook output focused on player comparison results
GUARDING_PIPELINE_MODE = 'reliability_pca'  # "original" or "reliability_pca"
# Redistribute guarded position/archetype shares into the Perimeter/Paint Defense Matchups
# blocks, and split "Defensive Matchups + Versatility" into Defensive Matchups + Defensive
# Versatility blocks (the latter carrying an engineered guarded-position-spread signal).
MATCHUP_RESTRUCTURE = True
VERSATILITY_MEGA_MODE = "shared"  # "shared" = two blocks share one mega; "separate" = own megas
EXPLAINED_VAR    = {"Offense": 1.0, "Defense": 1.0}   # PCA variance kept per subgroup
WHITEN_SHRINKAGE = {"Offense": 0.3, "Defense": 0.3}   # eigenvalue ridge before whitening
N_NEG_PER_POS    = 20     # random negatives per positive during metric learning
L2_REG           = 1.0    # ridge on learned subgroup weights
LEARN_BLEND      = 1.0    # continuity learner output before fair-prior stabilization
LEARNED_BLOCK_SHRINKAGE = 1.00    # 0 = mega-block-fair group prior, 1 = continuity-only
LEARNED_SUBGROUP_SHRINKAGE = 0.50 # 0 = uniform within block, 1 = continuity-only
PROFILE_BLEND    = 1.00   # additive personalization: 0 = stabilized group weights,
                          # 1 = pure "what this player is distinctive at"
WEIGHT_SHARPEN_GAMMA = 1.50   # mild post-blend sharpening of personalized block weights
                          # (per domain, renormalized to sum 1). 1.0 = no change; >1 concentrates
                          # attention on the player's top blocks and shrinks the rest.
SHARPENING_STRATEGY = "adaptive_mega"  # "fixed", "adaptive_mega", "adaptive_both", or "mixture_mega"

ADAPTIVE_SHARPEN_TAU = 0.35       # positive raw-distinctiveness evidence threshold in z units

ADAPTIVE_SHARPEN_ALPHA = 1.00     # higher = specialists stay closer to fixed gamma

ADAPTIVE_SHARPEN_STRENGTH = 0.60  # evidence scale for weak-flat-profile rejection

ADAPTIVE_CHILD_GAMMA = WEIGHT_SHARPEN_GAMMA  # selected model adapts only across mega-blocks

MAX_BLOCK_ATTENTION = 0.25      # optional hierarchical cap; 1.0 disables it
DISTINCTIVENESS_SOFT_SCALE = 0.25  # spread of the dimension-fair empirical RMS score.
DISTINCT_MAGNITUDE = 0.4   # 0 = distinctiveness is pure peer RANK (saturates: every
                          # top-percentile block ties at the ceiling, so a historic skill
                          # (Curry pull-up 3PT) cannot outrank a merely-elite one). >0
                          # blends in the per-block-standardized RMS MAGNITUDE so a player's
                          # genuinely most-extreme trait dominates their weight profile.
OFF_WEIGHT_GAMMA   = 1.0   # sharpening: pushes one-way players to extremes (two-way ~unaffected)
OFFENSE_BIAS       = 1.50   # neutral point: offense impact has more spread than defense, so a
                           # player equally elite on both leans offense. 1.0 = no bias; 2.0 ->
                           # two-way stars (PG/Kawhi/Giannis) ~0.70; higher = more offense.
MIN_DOMAIN_WEIGHT  = 0.10  # preserve at least 10% offense and 10% defense in overall similarity
OVERALL_OFF_WEIGHT = None  # None = auto per-player weight (recommended); or float in 0..1
REBOUNDING_WEIGHT_SCALE = 0.5  # multiplier on the rebounding families' within-domain share.
                               # 1.0 = original; 0.5 = rebounding counts half as much vs other
                               # offensive/defensive skills, freeing budget for more robust
                               # offense/defense comps (applied in _final_sub_weights).
USE_GAP2_PAIRS   = False  # also train on same-player (t, t+2) pairs

# Offense/defense IDENTITY from impact metrics (joined from the files below by
# player+season). Each side is a season-position-relative percentile composite; the
# offense/defense balance is off_weight = O^g/(O^g+D^g). Offensive identity
# blends scoring/playmaking impact (O-LEBRON) with creation centrality (Box
# Creation) so role-players whose offense is system-driven (Gobert, Capela) read
# low. Defensive identity uses luck-adjusted D-LEBRON, which resists team
# inflation far better than D-DPM / Defensive RAPTOR. You CAN add basket members
# with weights for robustness (e.g. "Defense Impact on Opponent Shot Quality":
# 1.0), but that re-inflates weak-defense stars, so keep D-LEBRON dominant.
IMPACT_PATHS = [
    path for path in os.environ.get("SIMILARITY_IMPACT_PATHS", "").split(os.pathsep) if path
]
OFF_IDENTITY_METRICS = {"O-LEBRON": 1.0, "On-Ball Action Share": 1.0}   # Box Creation dropped:
# it penalizes non-shooting creators and deflated their off_weight (off/def balance).
# Defensive IDENTITY (drives the off/def balance) = position-relative percentile composite
# built from curated perimeter/rim/help defensive features, gated by group-relative Matchup
# Difficulty (discounts defenders who face easy assignments), blended with D-LEBRON, then
# scaled by a per-position defensive-impact ceiling -- guards structurally carry less
# defensive value than wings/bigs, so an elite-for-a-guard composite still counts for less
# than an equivalent wing/big composite.
DEF_PERIMETER_FEATS = ["Perimeter Isolation Defense", "Ball Screen Navigation", "Off-Ball Chaser Defense", "Defensive Positional Versatility", "Defensive Role Versatility"]
DEF_RIM_FEATS = ["Stable Blocked Shots At Rim Per 75", "Rim Protection", "Rim Deterrence Per 100", "Rim Disruption",
                 "Rim Points Saved Per 75 Possessions", "Help Defense Talent",
                 "Help Defensive Activity", "Help Effectiveness Rating"]
DEF_REBOUND_FEATS = ["Defensive Rebounding Talent", "Defensive Rebounding Impact",
                      "Defensive Rebounding Consistency"]
DEF_MATCHUP_FEAT = "Matchup Difficulty"
DEF_BASE_RIM = {"guard": 0.15, "wing": 0.38, "big": 0.75}   # base rim-vs-perimeter weight by position
DEF_BOOST_THRESH = 0.80   # league-wide RIM score above this boosts rim weight (non-bigs with
                          # genuinely elite rim activity, e.g. Mobley/Gobert type seasons)
DEF_BOOST_GAIN = 3.0
DEF_GATE_FLOOR = 0.6      # PERIM score is regressed toward 0.5 when group-relative Matchup
DEF_GATE_POWER = 1.0      # Difficulty is low (discounts defenders who face easy assignments)
DEF_DLEBRON_W = 0.3       # weight of D-LEBRON in the def-skill composite
DEF_GROUP_CEILING = {"guard": 0.85, "wing": 1.0, "big": 1.0}   # defensive-impact ceiling:

# Continuous role-relevance gate for Paint Defense and Defensive Rebounding.
# It is strongest for perimeter-oriented players with limited defensive identity,
# while large/interior-role players and defense-defined players remain near 1.0.
ROLE_GATE_IDENTITY_STRENGTH = 0.85
ROLE_GATE_FLOOR = 0.40
ROLE_GATE_THRESHOLD = 0.58
ROLE_GATE_WIDTH = 0.08
                          # guards structurally carry less defensive value than wings/bigs

rng = np.random.default_rng(RANDOM_STATE)

def _setup_print(*args, **kwargs):
    if SHOW_SETUP_DIAGNOSTICS:
        print(*args, **kwargs)

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False).reset_index(drop=True)
df["_yr"] = df["season"].map(lambda s: int(str(s)[:4]))
players = df["player"].values
N = len(df)
_setup_print(f"Loaded {N} player-seasons, {df.shape[1]} columns")

# Exact playmaking-efficiency ratio requested for the similarity model. Validate
# before feature bookkeeping so unsafe denominators fail loudly by player-season.
_RATIO_AST_COL = "Stable Assists Per 75"
_RATIO_TOV_COL = "Stable TOV Per 75"
_missing_ratio_cols = [c for c in (_RATIO_AST_COL, _RATIO_TOV_COL) if c not in df.columns]
if _missing_ratio_cols:
    raise KeyError(f"Missing stable AST/TOV source columns: {_missing_ratio_cols}")
for _c in (_RATIO_AST_COL, _RATIO_TOV_COL):
    df[_c] = pd.to_numeric(df[_c], errors="coerce")
    _bad = df[_c].isna() | ~np.isfinite(df[_c])
    if _bad.any():
        _affected = df.loc[_bad, ["player", "season"]].head(30).to_dict("records")
        raise ValueError(f"Non-numeric/nonfinite values in {_c}: {_affected}")
_bad_denom = df[_RATIO_TOV_COL] <= 0
if _bad_denom.any():
    _affected = df.loc[_bad_denom, ["player", "season", _RATIO_TOV_COL]].to_dict("records")
    raise ValueError(f"Unsafe stable_ast_tov_ratio denominators: {_affected}")
df["stable_ast_tov_ratio"] = df[_RATIO_AST_COL] / df[_RATIO_TOV_COL]
assert df["stable_ast_tov_ratio"].notna().all() and np.isfinite(df["stable_ast_tov_ratio"]).all(),     "stable_ast_tov_ratio must be finite and null-free"
_setup_print("stable_ast_tov_ratio ready: "
      f"min={df['stable_ast_tov_ratio'].min():.3f}, "
      f"max={df['stable_ast_tov_ratio'].max():.3f}")
IN_SITE_ROSTER = (
    df.pop("in_site_roster").astype(bool).to_numpy()
    if "in_site_roster" in df.columns
    else np.ones(len(df), dtype=bool)
)
# Height/Weight/Minutes arrive from the impact side-load and are season-median
# filled a few lines below, so they are exempt here (in the notebook they were
# simply not yet joined at this point).
_role_context_pending = [c for c in ("Height", "Weight", "Minutes") if c in df.columns]
assert df.drop(columns=["_yr"] + _role_context_pending).isna().sum().sum() == 0, "Dataset still has nulls"

# ----------------------------------------------------------------------------
# Join offense/defense IMPACT metrics for the per-player off/def balance.
# The complete dataset was assembled from the IMPACT_PATHS files, so most impact
# columns (Box Creation, D-LEBRON) are already present; we only pull in the ones
# that are missing (e.g. O-LEBRON) by (player, season).
# ----------------------------------------------------------------------------
ROLE_CONTEXT_COLS = {"Height", "Weight", "Minutes"}
_impact_cols = sorted((set(OFF_IDENTITY_METRICS) | ROLE_CONTEXT_COLS) - {None})
_need_imp = [c for c in _impact_cols if c not in df.columns]
if _need_imp:
    _imp_parts = []
    for _p in IMPACT_PATHS:
        _src = pd.read_csv(_p, low_memory=False)
        _imp_parts.append(_src[["player", "season"] + [c for c in _need_imp if c in _src.columns]])
    _imp = pd.concat(_imp_parts, ignore_index=True).drop_duplicates(["player", "season"])
    df = df.merge(_imp, on=["player", "season"], how="left")
    assert len(df) == N, "impact join changed the row count"
_bad_imp = [c for c in OFF_IDENTITY_METRICS if c not in df.columns or df[c].isna().any()]
assert not _bad_imp, f"identity metrics missing/null after join: {_bad_imp}"
for _c in ROLE_CONTEXT_COLS:
    _x = pd.to_numeric(df[_c], errors="coerce")
    df[_c] = _x.fillna(_x.groupby(df["season"]).transform("median")).fillna(_x.median())
_setup_print(f"Impact metrics ready: off-identity {list(OFF_IDENTITY_METRICS)}, "
      f"def-identity curated perimeter/rim/help features + D-LEBRON")

# Position GROUPS (PG+SG = guard, SF+PF = wing, C = big) -- a STABLE per-PLAYER
# attribute (career-modal position group, tie-broken by the most recent season),
# NOT a per-season label. ~15% of players carry multiple position-GROUP labels
# across their career (e.g. LeBron: wing 8x, guard 2x, big 1x) -- using the raw
# per-season label would bounce a player between different learned models /
# comparison populations season to season, breaking continuity. Used for the
# specialized models AND for every position-relative comparison (distinctiveness,
# identity, off_weight, and role-subgroup standardization).
GROUP_MAP = {"PG": "guard", "SG": "guard", "SF": "wing", "PF": "wing", "C": "big"}
GROUPS = ["guard", "wing", "big"]
_row_grp = df["position"].map(GROUP_MAP)
_player_grp = {}
for _pl, _g in pd.DataFrame({"player": df["player"], "_yr": df["_yr"], "row_grp": _row_grp}).groupby("player"):
    _counts = _g["row_grp"].value_counts()
    _top = _counts[_counts == _counts.max()].index.tolist()
    _player_grp[_pl] = _top[0] if len(_top) == 1 else _g.sort_values("_yr")["row_grp"].iloc[-1]
PLAYER_GROUP = df["player"].map(_player_grp).values


BLOCKS = {
    "3PT Shooting": {
        "Pull-Up + Self Created 3PT": [
            "3PT Pull Up FGA Per 75", "3PT Pull Up Openness Rating", "3PT Pull Up Shot Creation",
            "3PT Pull Up Shot Making", "3PT Pull Up Shot Making Efficiency", "3PT Pull Up Shot Quality",
            "3PT Pull Up Talent", "Stable Pull Up 3PT%", "3PT Shot Creation", "pct_3fga_unassisted",
            "On-Ball Gravity", "three_to_six_drib_3FGA_per_75", "three_to_six_drib_3FGA_accuracy",
            "seven_plus_drib_3FGA_per_75", "seven_plus_drib_3FGA_accuracy",
            "Deep 3PT Shooting Talent", "Stable Avg 3PT Shot Distance",
        ],
        "C+S + Off-Ball 3PT": [
            "Stable C&S 3PT%", "C&S 3PT Openness Rating", "C&S 3PT Shooting Talent", "C&S 3PT Shot Making",
            "C&S 3PT Shot Making Efficiency", "C&S 3PT Shot Quality", "C&S to Pull Up Ratio",
            "one_two_drib_3fga_per_75", "one_two_drib_3PA_accuracy", "Off-Ball Gravity", "pct_3fga_assisted",
            "Stable Corner 3PT%", "Corner 3 to ATB Ratio", "catch_shoot_3fga_per_75",
        ],
    },

    "Mid Range Shooting": {
        "Pull Up Midrange": [
            "Midrange Pull Up FGA Per 75", "Midrange Pull Up FG%", "Midrange Pull Up Shot Creation",
            "Midrange Pull Up Shot Making", "Midrange Pull Up Shot Making Efficiency",
            "Midrange Pull Up Shot Quality", "Midrange Pull Up Talent",
        ],
        "Non Paint Midrange": [
            "Non-Paint Midrange Shooting Talent", "Non-Paint Midrange Shot Creation",
            "Non-Paint Midrange Shot Making", "Non-Paint Midrange Shot Making Efficiency",
            "Non-Paint Midrange Shot Quality", "non_paint_midrange_fga_per_75", "non_paint_midrange_fg_pct",
            "Stable Avg 2PT Shot Distance", "pct_2fga_assisted", "pct_2fga_unassisted",
        ],
        "Short Midrange": [
            "SMR FGA PER 75", "SMR FG%",
        ],
        "Long Midrange": [
            "LMR FGA PER 75", "LMR FG%",
        ],
    },

    "Rim Pressure": {
        "Rim": [
            "% of Shots at Rim Unassisted", "Stable Rim FG%", "Rim Openness Rating",
            "Rim Shot Attempts Per 75 Possessions", "Rim Shot Creation", "Rim Shot Making",
            "Rim Shot Making Efficiency", "Rim Shot Quality", "Rim Attempt Consistency",
            "Rim Makes Consistency", "RA_fga_per_75", "RA_fg%", "dunks_per_game", "Finishing Talent",
            "Contact Finish Rate",
        ],
        "Paint - Non Rim": [
            "Paint Openness Rating", "Paint Shooting Talent", "Paint Shot Creation", "Paint Shot Making",
            "Paint Shot Making Efficiency", "Paint Shot Quality", "paint_non_ra_fga_per_75", "non_ra_fg%",
            "Floater FG%", "Floater Shot Creation", "Floater Shot Making", "Floater Shot Making Efficiency",
            "Floater Shot Quality", "Floater Talent",
        ],
        "Drive Tendencies": [
            "Drives Per 75 Possessions", "drive_fga_per_75",
            "Drive Pass Rate", "Drive TOV Rate", "drive_fg_pct", "drive_PPP",
        ],
    },

    "Cut": {
        "Cut": [
            "Stable Cut PPP", "Cuts Per 75 Possessions", "Cut Scoring Impact Per 75", "cut_tov_freq",
        ],
    },

    "Handoff": {
        "Handoff": [
            "Stable Handoff PPP", "handoff_per_75", "handoff_tov_freq",
        ],
    },

    "Isolations": {
        "Isolations": [
            "Stable Isolation PPP", "Isolation Turnover Rate", "Isolation Shooting Talent",
            "Isolation Shot Making", "Isolation Shot Making Efficiency", "Isolation Shot Quality",
            "isolation_per_75",
        ],
    },

    "Off-Ball Screens": {
        "Off-Ball Screens": [
            "Stable Off-Ball Screen PPP", "Off-Ball Screen Possessions Per 75 Possessions",
            "Off-Screen Impact Per 75", "off_screen_tov_freq",
        ],
    },

    "P&R Ball Handler": {
        "P&R Ball Handler": [
            "Stable P&R Ball Handler PPP", "P&R Creation Rate", "PnR Ball Handler Shooting Talent",
            "PnR Ball Handler Shot Making", "PnR Ball Handler Shot Making Efficiency",
            "PnR Ball Handler Shot Quality", "p_r_ball_handler_per_75", "p_r_ball_handler_tov_freq",
        ],
    },

    "P&R Roll Man": {
        "P&R Roll Man": [
            "Stable P&R Roll Man PPP", "Roll Man Impact Per 75 Possessions",
            "Roll Man Poss Per 75 Possessions", "PnR Screener Shooting Talent", "PnR Screener Shot Making",
            "PnR Screener Shot Making Efficiency", "PnR Screener Shot Quality", "p_r_roll_man_tov_freq",
        ],
    },

    "Spot Up": {
        "Spot Up": [
            "Stable Spot Up PPP", "Spot Up%", "spot_up_tov_freq", "spot_up_per_75",
        ],
    },

    "Transition": {
        "Transition": [
            "Stable Transition PPP", "Transition Shooting Talent", "Transition Shot Creation",
            "Transition Shot Making", "Transition Shot Making Efficiency", "Transition Shot Quality",
            "Transition Frequency Impact", "Offensive Transition Frequency Impact",
            "transition_per_75", "transition_tov_freq",
        ],
    },

    

    "Post Ups": {
        "Post Ups": [
            "Stable Post Up PPP", "Post Up Impact Per 75 Possessions", "Post Up Shooting Talent",
            "Post Up Shot Making", "Post Up Shot Making Efficiency", "Post Up Shot Quality",
            "post_up_per_75", "post_up_tov_freq",
        ],
    },

    "Touches / Ball Dominance": {
        "Ball Dominance": [
            "Ball Dominance %", "time_of_poss_new",
        ],
        "Dribbling Tendencies": [
            "avg_drib_per_touch", "Dribbles Per Second on Offense", "avg_sec_per_touch",
        ],
        "Touches Tendencies": [
            "touches_per_75", "Front_court_touches_per_75", "elbow_touches_per_75", "paint_touches_per_75", "post_touches_per_75", "On-Ball Action Share",
        ],
    },

    "Playmaking": {
        "Passing Volume": [
            "Box Creation", "Potential Assists Per 100 Passes", "Potential Assists Per 75 Possessions",
            "Lob Passing Creation Rate", "Passing Creation Volume", "Assist Consistency",
            "Stable Assists Per 75", "Passing Versatility", "Playmaking Talent",
        ],
        "Passing Efficiency": [
            "Passing Creation Quality", "Passing Efficiency",
            "stable_ast_tov_ratio", "potential_ast_tov_ratio",
        ],
    },

    "Guarded By Data": {
        # NOTE: the old code had a second subgroup "Guarded By Matchup Difficulty"
        # containing the SINGLE feature "Guarded by Matchup Difficulty" which is
        # already inside "Guarded By Roles". Because every subgroup is rescaled to
        # total variance 1 and concatenated, that one feature was counted twice with
        # full weight. The duplicate subgroup is removed.
        "Guarded By Roles": [
            "Guarded by Ball Screen Navigation", "Guarded by Help Defensive Activity",
            "Guarded by Matchup Difficulty", "Guarded by Off-Ball Chaser Defense",
            "Guarded by Perimeter Isolation Defense", "Guarded by Rim Protection",
        ],
    },

    "Perimeter Defense": {
        "On-Ball Defense": [
            #"3PT Contests Per 75 Possessions",
            "Ball Screen Navigation", "Perimeter Isolation Defense",
        ],
        "Off-Ball / Chaser Defense": [
            "Off-Ball Chaser Defense",
        ],
        "Matchups": [
            "% of Time Guarding Movement Shooters", "% of Time Guarding Off-Screen Shooters",
            "% of Time Guarding Point Guards", "% of Time Guarding Primary Ball Handlers",
            "% of Time Guarding Secondary Ball Handlers", "% of Time Guarding Shooting Guards",
            "% of Time Guarding Small Forwards", "% of Time Guarding Shot Creators",
            "% of Time Guarding Slashers", "% of Time Guarding Stationary Shooters",
            "Guarded Pick & Roll Ball Handler Frequency%",
        ],
    },

    "Paint Defense": {
        "Rim Protection": [
            #"Post Defense", 
            "Rim Deterrence Per 100", "Rim Disruption", "Rim Points Saved Per 75 Possessions",
            "Rim Protection", 
        ],
        "Shot Blocking": [
            "Stable Blocked Shots At Rim Per 75", #"Block Rate on Contests",
        ],
        "Matchups": [
            "% of Time Guarding Athletic Finishers", "% of Time Guarding Centers",
            "% of Time Guarding Post Scorers", "% of Time Guarding Power Forwards",
            "% of Time Guarding Roll & Cut Bigs", "% of Time Guarding Stretch Bigs",
            "% of Time Guarding Versatile Bigs", "Guarded Offensive Rebounding Talent",
        ],
        "Help Defense": [
            "Help Defense Talent", "Help Defensive Activity", "Help Effectiveness Rating",
        ],
    },

    "Defensive Matchups + Versatility": {
        "Positions / Archetypes Guarded / Versatility": [
            "Defensive Positional Versatility", "Defensive Role Versatility", "Defensive Portability",
        ],
        "Matchup Difficulty": [
            "Matchup Difficulty",
            "% of Time Guarding Usage Tier 1 Players", "% of Time Guarding Usage Tier 2 Players",
            "% of Time Guarding Usage Tier 3 Players", "% of Time Guarding Usage Tier 4 Players",
            "% of Time Guarding Usage Tier 5 Players", "% of Time Guarding Usage Tier 6 Players",
            "Guarded 3PT Shooting Talent", "Guarded Finishing Talent", "Guarded Midrange Talent",
            "Guarded O-LEBRON", "Guarded Offensive Involvement Rate", "Guarded On-Ball %",
            "Guarded One on One Talent", "Guarded Playmaking Talent",
            "Guarded Rim Shot Making","Guarded USG%",
            "Primary Defender Involvement Rate",
        ],
    },

    "Defensive Advanced / Impact Stats": {
            "Defensive Advanced / Impact Stats": [
           "D-DPM", "D-LEBRON", "Defense Impact on Opponent Shot Quality",
            "Defense Impact on Possession Quality", "Defensive Possession Volume Impact",
            "Defensive eFG% Impact", "Defensive RAPTOR", "Defensive FT Rate Impact",
        ],
    },

    "Free Throw Generation": {
        "Free Throw Generation": [
            "Stable Shooting Fouls Drawn Per 75", "Drive Foul Drawn Rate",
            "Isolation Foul Drawn Rate", "3PT Foul Rate", "Offensive FT Rate Impact", "spot_up_sf_freq",
            "post_up_sf_freq", "transition_sf_freq", "cut_sf_freq", "handoff_sf_freq", "off_screen_sf_freq",
            "p_r_ball_handler_sf_freq",
        ],
    },

    # Rebounding is split into two blocks so it competes by salience inside the
    # offense / defense domains instead of inheriting a whole domain budget alone.
    "Offensive Rebounding": {
        "Offensive Rebounding": [
            "Offensive Rebounding Talent",
            "Offensive Rebounding Chances Per 75 Possessions",
            "Adjusted Offensive Rebounding Success Rate",
            "Stable Offensive Rebounds Per 75",
        ],
        "Putbacks": [
            "Stable Self OReb Percent of FGA",
            "Stable Self Offensive Rebounds Per 75",
            "Putback Scoring Impact Per 75 Possessions",
            "Stable Putback Points Per 75",
            "Stable Putbacks PPP",
        ],
    },

    "Defensive Rebounding": {
        "Defensive Rebounding": [
            "Defensive Rebounding Talent",
            "Defensive Rebounding Impact", "Defensive Rebounding Consistency",
        ],
    },
}



# GUARDED DEFENSIVE MATCHUP PIPELINE
# `original` reproduces the pre-stabilization representation exactly.
FORBIDDEN_MODEL_FEATURES = {'% of Time Guarding Low Minute Players', 'Guarded Projected Usage%', 'pass_tov_ratio'}
GUARDED_POSITION_COMPONENTS = ('% of Time Guarding Point Guards', '% of Time Guarding Shooting Guards', '% of Time Guarding Small Forwards', '% of Time Guarding Power Forwards', '% of Time Guarding Centers')
GUARDED_ARCHETYPE_BINS = MappingProxyType({
    'Guarded Perimeter Off Ball': ('% of Time Guarding Movement Shooters', '% of Time Guarding Off-Screen Shooters', '% of Time Guarding Stationary Shooters'), 'Guarded Primary Ball Handlers': ('% of Time Guarding Primary Ball Handlers',), 'Guarded Shot Creators': ('% of Time Guarding Shot Creators',), 'Guarded Secondary Wings': ('% of Time Guarding Secondary Ball Handlers', '% of Time Guarding Slashers'), 'Guarded Interior Finishers': ('% of Time Guarding Athletic Finishers', '% of Time Guarding Roll & Cut Bigs'), 'Guarded Skilled Bigs': ('% of Time Guarding Post Scorers', '% of Time Guarding Stretch Bigs', '% of Time Guarding Versatile Bigs')
})
GUARDED_USAGE_BINS = MappingProxyType({
    'Guarded High Usage': ('% of Time Guarding Usage Tier 1 Players', '% of Time Guarding Usage Tier 2 Players'), 'Guarded Medium Usage': ('% of Time Guarding Usage Tier 3 Players', '% of Time Guarding Usage Tier 4 Players'), 'Guarded Low Usage': ('% of Time Guarding Usage Tier 5 Players', '% of Time Guarding Usage Tier 6 Players')
})
MATCHUP_CONTEXT_FEATURES = ('Guarded Pick & Roll Ball Handler Frequency%',)
MATCHUP_RESPONSIBILITY_FEATURES = ('Guarded On-Ball %', 'Guarded Offensive Involvement Rate', 'Primary Defender Involvement Rate', 'Guarded USG%', 'Matchup Difficulty')
MATCHUP_OPPONENT_QUALITY_FEATURES = ('Guarded 3PT Shooting Talent', 'Guarded Finishing Talent', 'Guarded Midrange Talent', 'Guarded O-LEBRON', 'Guarded One on One Talent', 'Guarded Playmaking Talent', 'Guarded Rim Shot Making', 'Guarded Offensive Rebounding Talent')
GUARDING_RELIABILITY_KAPPA = MappingProxyType({'position_composition': 1000.0, 'archetype_composition': 2500.0, 'usage_composition': 4000.0, 'responsibility': 4000.0, 'opponent_quality': 4000.0})
GUARDING_COMPOSITION_TOLERANCES = (1.0, 5.0, 10.0)
GUARDING_PREPROCESSING_VERSION = "guarding-stabilization-v1"

def _remove_guard_features(_features):
    _remove = set(_features)
    for _block_subs in BLOCKS.values():
        for _sub_name in list(_block_subs):
            _block_subs[_sub_name] = [
                _feature
                for _feature in _block_subs[_sub_name]
                if _feature not in _remove
            ]
            if not _block_subs[_sub_name]:
                del _block_subs[_sub_name]

def _assert_forbidden_absent(_manifest, _label):
    _active = set(_manifest)
    _bad = FORBIDDEN_MODEL_FEATURES & _active
    assert not _bad, f"Forbidden model features in {_label}: {sorted(_bad)}"

_assert_forbidden_absent(
    [
        _feature
        for _subgroups in BLOCKS.values()
        for _features in _subgroups.values()
        for _feature in _features
    ],
    "raw BLOCKS",
)

_production_guard_dir = Path('/Users/harsha/Desktop/testing_galaxy/guarding_stabilization_outputs/production_candidate')
_production_guard_dir.mkdir(parents=True, exist_ok=True)
_GUARD_COMPOSITION_AUDIT = []
_GUARD_PRIOR_ROWS = []
_GUARD_REPRESENTATION_MANIFEST = []
_GUARD_DUAL_USE_MANIFEST = []

if GUARDING_PIPELINE_MODE == "original":
    _GUARD_AGGREGATIONS = {'Guarded Perimeter Off Ball': ['% of Time Guarding Movement Shooters', '% of Time Guarding Off-Screen Shooters', '% of Time Guarding Stationary Shooters'], 'Guarded Secondary Wings': ['% of Time Guarding Secondary Ball Handlers', '% of Time Guarding Slashers'], 'Guarded Interior Finishers': ['% of Time Guarding Athletic Finishers', '% of Time Guarding Roll & Cut Bigs'], 'Guarded Skilled Bigs': ['% of Time Guarding Post Scorers', '% of Time Guarding Stretch Bigs', '% of Time Guarding Versatile Bigs'], 'Guarded High Usage': ['% of Time Guarding Usage Tier 1 Players', '% of Time Guarding Usage Tier 2 Players'], 'Guarded Medium Usage': ['% of Time Guarding Usage Tier 3 Players', '% of Time Guarding Usage Tier 4 Players'], 'Guarded Low Usage': ['% of Time Guarding Usage Tier 5 Players', '% of Time Guarding Usage Tier 6 Players']}
    _assert_forbidden_absent(
        [
            _feature
            for _members in _GUARD_AGGREGATIONS.values()
            for _feature in _members
        ],
        "original aggregation definitions",
    )
    for _new, _members in _GUARD_AGGREGATIONS.items():
        df[_new] = df[_members].sum(axis=1)
    _remove_guard_features(
        [_feature for _members in _GUARD_AGGREGATIONS.values() for _feature in _members]
    )
    BLOCKS["Perimeter Defense"]["Matchups"].extend([
        "Guarded Perimeter Off Ball",
        "Guarded Secondary Wings",
    ])
    BLOCKS["Paint Defense"]["Matchups"].extend([
        "Guarded Interior Finishers",
        "Guarded Skilled Bigs",
    ])
    BLOCKS["Defensive Matchups + Versatility"]["Matchup Difficulty"].extend([
        "Guarded High Usage",
        "Guarded Medium Usage",
        "Guarded Low Usage",
    ])
else:
    if GUARDING_PIPELINE_MODE != "reliability_pca":
        raise ValueError(
            "GUARDING_PIPELINE_MODE must be 'original' or 'reliability_pca'"
        )
    _archetype_sources = [
        _feature
        for _members in GUARDED_ARCHETYPE_BINS.values()
        for _feature in _members
    ]
    _usage_sources = [
        _feature
        for _members in GUARDED_USAGE_BINS.values()
        for _feature in _members
    ]
    assert len(_archetype_sources) == len(set(_archetype_sources))
    assert len(_usage_sources) == len(set(_usage_sources))
    _assert_forbidden_absent(
        list(GUARDED_POSITION_COMPONENTS)
        + _archetype_sources
        + _usage_sources
        + list(MATCHUP_CONTEXT_FEATURES)
        + list(MATCHUP_RESPONSIBILITY_FEATURES)
        + list(MATCHUP_OPPONENT_QUALITY_FEATURES),
        "stabilized metadata",
    )

    _GUARD_AGGREGATIONS = {
        **{_name: list(_members) for _name, _members in GUARDED_ARCHETYPE_BINS.items()},
        **{_name: list(_members) for _name, _members in GUARDED_USAGE_BINS.items()},
    }
    for _new, _members in _GUARD_AGGREGATIONS.items():
        _numeric = df[_members].apply(pd.to_numeric, errors="coerce")
        df[_new] = _numeric.sum(axis=1, min_count=len(_members))

    _evidence = pd.to_numeric(df["Minutes"], errors="coerce")
    _evidence = _evidence.fillna(
        _evidence.groupby(df["season"]).transform("median")
    ).fillna(_evidence.median()).to_numpy()
    assert np.all(_evidence >= 0.0)

    def _guard_position_prior(_values, _valid, _family, _columns):
        _prior = np.full_like(_values, np.nan, dtype=float)
        for _season in df["season"].astype(str).unique():
            for _group in GROUPS:
                _reference = (
                    _valid
                    & (df["season"].astype(str).values == _season)
                    & (PLAYER_GROUP == _group)
                )
                if not _reference.any():
                    _reference = _valid & (PLAYER_GROUP == _group)
                if not _reference.any():
                    _reference = _valid
                _center = np.nanmedian(_values[_reference], axis=0)
                _center = np.clip(_center, 0.0, None)
                _center /= _center.sum()
                _target = (
                    (df["season"].astype(str).values == _season)
                    & (PLAYER_GROUP == _group)
                )
                _prior[_target] = _center
                for _component, _value in zip(_columns, _center):
                    _GUARD_PRIOR_ROWS.append({
                        "family": _family,
                        "season": _season,
                        "position_group": _group,
                        "component": _component,
                        "prior": _value,
                    })
        return _prior

    def _prepare_guard_composition(_family, _columns):
        _raw = df[list(_columns)].apply(pd.to_numeric, errors="coerce").to_numpy()
        _negative = np.any(_raw < 0.0, axis=1)
        _missing = np.any(~np.isfinite(_raw), axis=1)
        _sums = np.nansum(_raw, axis=1)
        _valid = (~_negative) & (~_missing) & (_sums > 0.0)
        for _season, _indexes in df.groupby("season").groups.items():
            _indexes = np.asarray(list(_indexes), dtype=int)
            _season_sums = _sums[_indexes]
            _row = {
                "family": _family,
                "season": _season,
                "rows": len(_indexes),
                "valid_rows": int(_valid[_indexes].sum()),
                "negative_rows": int(_negative[_indexes].sum()),
                "missing_rows": int(_missing[_indexes].sum()),
                "minimum_sum": float(np.nanmin(_season_sums)),
                "maximum_sum": float(np.nanmax(_season_sums)),
                "median_sum": float(np.nanmedian(_season_sums)),
                "mean_absolute_deviation_from_100": float(
                    np.nanmean(np.abs(_season_sums - 100.0))
                ),
            }
            for _tolerance in GUARDING_COMPOSITION_TOLERANCES:
                _row[f"percent_outside_{_tolerance:g}"] = float(
                    100.0
                    * np.mean(np.abs(_season_sums - 100.0) > _tolerance)
                )
            _GUARD_COMPOSITION_AUDIT.append(_row)
        _current = np.full_like(_raw, np.nan, dtype=float)
        _current[_valid] = _raw[_valid] / _sums[_valid, None]
        _prior = _guard_position_prior(
            _current, _valid, _family, _columns
        )
        _current[~_valid] = _prior[~_valid]
        _kappa = float(GUARDING_RELIABILITY_KAPPA[_family])
        _reliability = _evidence / (_evidence + _kappa)
        assert np.all((_reliability >= 0.0) & (_reliability <= 1.0))
        _stable = (
            _reliability[:, None] * _current
            + (1.0 - _reliability[:, None]) * _prior
        )
        _stable /= _stable.sum(axis=1, keepdims=True)
        assert np.allclose(_stable.sum(axis=1), 1.0, atol=1e-12)
        return _stable, _reliability

    _composition_specs = {
        "position_composition": list(GUARDED_POSITION_COMPONENTS),
        "archetype_composition": list(GUARDED_ARCHETYPE_BINS),
        "usage_composition": list(GUARDED_USAGE_BINS),
    }
    _composition_subgroups = {}
    for _family, _columns in _composition_specs.items():
        _stable, _reliability = _prepare_guard_composition(_family, _columns)
        _features = []
        for _index, _component in enumerate(_columns):
            _feature = f"__guard_{_family}_{_index}"
            df[_feature] = 100.0 * _stable[:, _index]
            _features.append(_feature)
            _GUARD_REPRESENTATION_MANIFEST.append({
                "family": _family,
                "component": _component,
                "embedding_feature": _feature,
                "representation": "reliability_shrunk_proportion_pca",
                "kappa": float(GUARDING_RELIABILITY_KAPPA[_family]),
            })
            # Secondary identity and role-gate paths need interpretable stabilized raw shares.
            if _family == "position_composition":
                df[_component] = 100.0 * _stable[:, _index]
        _composition_subgroups[_family] = _features

    def _guard_scalar_prior(_feature):
        _values = pd.to_numeric(df[_feature], errors="coerce")
        _prior = np.full(N, np.nan)
        for _season in df["season"].astype(str).unique():
            for _group in GROUPS:
                _reference = (
                    (df["season"].astype(str).values == _season)
                    & (PLAYER_GROUP == _group)
                )
                _center = pd.to_numeric(
                    df.loc[_reference, _feature], errors="coerce"
                ).median()
                if not np.isfinite(_center):
                    _center = pd.to_numeric(
                        df.loc[PLAYER_GROUP == _group, _feature], errors="coerce"
                    ).median()
                _target = _reference
                _prior[_target] = _center
                _GUARD_PRIOR_ROWS.append({
                    "family": "scalar",
                    "season": _season,
                    "position_group": _group,
                    "component": _feature,
                    "prior": _center,
                })
        return _values.fillna(pd.Series(_prior)).to_numpy(), _prior

    for _family, _features in (
        ("responsibility", MATCHUP_RESPONSIBILITY_FEATURES),
        ("opponent_quality", MATCHUP_OPPONENT_QUALITY_FEATURES),
    ):
        _kappa = float(GUARDING_RELIABILITY_KAPPA[_family])
        _reliability = _evidence / (_evidence + _kappa)
        for _feature in _features:
            _current, _prior = _guard_scalar_prior(_feature)
            df[_feature] = (
                _reliability * _current
                + (1.0 - _reliability) * _prior
            )

    _remove_guard_features(
        list(GUARDED_POSITION_COMPONENTS)
        + _archetype_sources
        + _usage_sources
    )
    _responsibility_quality = (
        list(MATCHUP_RESPONSIBILITY_FEATURES)
        + [
            _feature
            for _feature in MATCHUP_OPPONENT_QUALITY_FEATURES
            if _feature != "Guarded Offensive Rebounding Talent"
        ]
    )
    _pos_comp = _composition_subgroups["position_composition"]   # [PG, SG, SF, PF, C]
    _arc_comp = _composition_subgroups["archetype_composition"]  # [OffBall, PrimaryBH, ShotCreators, SecWings, IntFinishers, SkilledBigs]
    _usg_comp = _composition_subgroups["usage_composition"]      # [High, Medium, Low]
    if not MATCHUP_RESTRUCTURE:
        BLOCKS["Perimeter Defense"]["Matchups"] = list(MATCHUP_CONTEXT_FEATURES)
        BLOCKS["Paint Defense"]["Matchups"] = ["Guarded Offensive Rebounding Talent"]
        _base_matchup = BLOCKS["Defensive Matchups + Versatility"]
        _base_matchup.pop("Matchup Difficulty", None)
        _base_matchup["Matchup Responsibility + Quality"] = _responsibility_quality
        BLOCKS["Matchup Deployment"] = {
            "Guarded Position Composition": _pos_comp,
            "Guarded Archetype Composition": _arc_comp,
            "Guarded Usage Composition": _usg_comp,
        }
    else:
        # Guarded position + archetype shares move into the perimeter / paint matchup blocks so
        # "who you defend" competes by salience inside those defensive families.
        BLOCKS["Perimeter Defense"]["Matchups"] = (
            list(MATCHUP_CONTEXT_FEATURES) + _pos_comp[0:3] + _arc_comp[0:4]
        )
        BLOCKS["Paint Defense"]["Matchups"] = (
            ["Guarded Offensive Rebounding Talent"] + _pos_comp[3:5] + _arc_comp[4:6]
        )
        # Engineered VERSATILITY signal: Shannon entropy of the (stabilized) guarded-position
        # distribution, percentile-ranked within season x position group so multi-position
        # defenders (Davis, Adebayo) rank high relative to their positional peers.
        _pos_share = (
            df[list(GUARDED_POSITION_COMPONENTS)]
            .apply(pd.to_numeric, errors="coerce").to_numpy()
        )
        _pos_share = _pos_share / np.clip(_pos_share.sum(axis=1, keepdims=True), 1e-9, None)
        _pos_entropy = -np.nansum(
            np.where(_pos_share > 0.0, _pos_share * np.log(_pos_share), 0.0), axis=1
        )
        df["Guarded Position Spread"] = (
            pd.Series(_pos_entropy)
            .groupby([df["season"].astype(str), pd.Series(PLAYER_GROUP)])
            .rank(pct=True).values
        )
        # Split "Defensive Matchups + Versatility" into two blocks (REGROUP below assigns megas):
        #   Defensive Matchups    = responsibility/quality + usage-guarded composition
        #   Defensive Versatility = versatility traits + engineered position-spread signal
        _base_matchup = BLOCKS["Defensive Matchups + Versatility"]
        _base_matchup.pop("Matchup Difficulty", None)
        _versatility_feats = _base_matchup.pop("Positions / Archetypes Guarded / Versatility")
        _base_matchup["Matchup Responsibility + Quality"] = _responsibility_quality
        _base_matchup["Guarded Usage Composition"] = _usg_comp
        _base_matchup["Defensive Versatility"] = _versatility_feats + ["Guarded Position Spread"]

    pd.DataFrame(_GUARD_COMPOSITION_AUDIT).to_csv(
        _production_guard_dir / "composition_audits.csv", index=False
    )
    pd.DataFrame(_GUARD_PRIOR_ROWS).to_csv(
        _production_guard_dir / "prior_values.csv", index=False
    )
    pd.DataFrame(_GUARD_REPRESENTATION_MANIFEST).to_csv(
        _production_guard_dir / "guarding_representation_manifest.csv",
        index=False,
    )


BLOCK_DOMAIN = {
    "3PT Shooting": "Offense",
    "Mid Range Shooting": "Offense",
    "Rim Pressure": "Offense",
    "Cut": "Offense",
    "Handoff": "Offense",
    "Isolations": "Offense",
    "Off-Ball Screens": "Offense",
    "P&R Ball Handler": "Offense",
    "P&R Roll Man": "Offense",
    "Spot Up": "Offense",
    "Transition": "Offense",
    "Post Ups": "Offense",
    "Touches / Ball Dominance": "Offense",
    "Playmaking": "Offense",
    "Guarded By Data": "Offense",
    "Free Throw Generation": "Offense",
    "Perimeter Defense": "Defense",
    "Paint Defense": "Defense",
    "Defensive Matchups + Versatility": "Defense",
    "Defensive Advanced / Impact Stats": "Defense",
    "Offensive Rebounding": "Offense",
    "Defensive Rebounding": "Defense",
}

# ----------------------------------------------------------------------------
if GUARDING_PIPELINE_MODE != "original" and not MATCHUP_RESTRUCTURE:
    BLOCK_DOMAIN["Matchup Deployment"] = "Defense"

# Restructure blocks before learning. Two mechanisms:
#  SPLIT_TO_SUBGROUPS: fully explode a block into one block per subgroup, ALL
#    sharing the original mega-block (so 3PT-as-a-whole keeps one fair-prior budget
#    — pull-up/C&S/deep just get their own distinctiveness channels).
#  REGROUP: custom-merge a block's subgroups into named blocks; each entry is
#    (new block name, [subgroups], mega-block). The mega-block controls fair-prior
#    INITIAL INFLUENCE: give every new block its OWN mega when the parts are
#    genuinely different signals (defensive MATCHUPS = role vs SKILL), or share ONE
#    mega when the parts are facets of a single skill that should still count as one
#    mega-block of influence (Rim Pressure, Touches / Ball Dominance).
# These mappings define the hierarchy used by the fair prior, distinctiveness,
# personalization, sharpening, reporting, and final distance allocations.
# ----------------------------------------------------------------------------
SPLIT_TO_SUBGROUPS = ["3PT Shooting"]
REGROUP = {
    # Pull Up / Non Paint / Short / Long midrange = distinct identity channels (own
    # blocks) but they SHARE one "Mid Range Shooting" mega, so their COMBINED
    # fair-prior budget is unchanged from when midrange was a single block (mega =
    # budget unit).
    "Mid Range Shooting": [
        ("Pull Up Midrange",   ["Pull Up Midrange"],   "Mid Range Shooting"),
        ("Non Paint Midrange", ["Non Paint Midrange"], "Mid Range Shooting"),
        ("Short Midrange",     ["Short Midrange"],     "Mid Range Shooting"),
        ("Long Midrange",      ["Long Midrange"],      "Mid Range Shooting"),
    ],
    # Passing VOLUME and Passing EFFICIENCY are distinct identity axes -> own megas.
    "Playmaking": [
        ("Playmaking Volume",     ["Passing Volume"],     "Playmaking Volume"),
        ("Playmaking Efficiency", ["Passing Efficiency"], "Playmaking Efficiency"),
    ],
    # skill vs matchups = separate BLOCKS but SHARE one mega (one mega of influence each)
    "Perimeter Defense": [
        ("Perimeter Defense",          ["On-Ball Defense", "Off-Ball / Chaser Defense"], "Perimeter Defense"),
        ("Perimeter Defense Matchups", ["Matchups"],                                                     "Perimeter Defense"),
    ],
    "Paint Defense": [
        ("Paint Defense",          ["Rim Protection", "Shot Blocking", "Help Defense"], "Paint Defense"),
        ("Paint Defense Matchups", ["Matchups"],                                        "Paint Defense"),
    ],
    # Rim / Paint - Non Rim / Drive Tendencies remain separate child signals, but
    # share one Rim Pressure mega-block, analogous to the 3PT Shooting children.
    # (Floaters merged into Paint - Non Rim at the BLOCKS-dict level above)
    "Rim Pressure": [
        ("Rim",              ["Rim"],              "Rim Pressure"),
        ("Paint - Non Rim",  ["Paint - Non Rim"],  "Rim Pressure"),
        ("Drive Tendencies", ["Drive Tendencies"], "Rim Pressure"),
    ],
    # touches (volume), ball-dominance (usage / on-ball role), and dribbling (handling
    # mechanics) are distinct signals -> own megas. Ball Dominance is the primary on-ball axis.
    "Touches / Ball Dominance": [
        ("Touches",              ["Touches Tendencies"],   "Touches"),
        ("Ball Dominance",       ["Ball Dominance"],       "Ball Dominance"),
        ("Dribbling Tendencies", ["Dribbling Tendencies"], "Dribbling Tendencies"),
    ],
}
if GUARDING_PIPELINE_MODE != "original" and not MATCHUP_RESTRUCTURE:
    REGROUP["Matchup Deployment"] = [
        ("Matchup Deployment", [
            "Guarded Position Composition",
            "Guarded Archetype Composition",
            "Guarded Usage Composition",
        ], "Defensive Matchups + Versatility"),
    ]
if GUARDING_PIPELINE_MODE != "original" and MATCHUP_RESTRUCTURE:
    # Two child blocks; mega assignment controlled by VERSATILITY_MEGA_MODE.
    _dm_mega = ("Defensive Matchups + Versatility" if VERSATILITY_MEGA_MODE == "shared"
                else "Defensive Matchups")
    _dv_mega = ("Defensive Matchups + Versatility" if VERSATILITY_MEGA_MODE == "shared"
                else "Defensive Versatility")
    REGROUP["Defensive Matchups + Versatility"] = [
        ("Defensive Matchups", ["Matchup Responsibility + Quality",
                                "Guarded Usage Composition"], _dm_mega),
        ("Defensive Versatility", ["Defensive Versatility"], _dv_mega),
    ]

_BLK2, _DOM2, BLOCK_MEGA = {}, {}, {}   # BLOCK_MEGA: block -> mega-block (fair-prior grouping)
for _b, _subs in BLOCKS.items():
    if _b in SPLIT_TO_SUBGROUPS:
        for _sub, _feats in _subs.items():
            _BLK2[f"{_b}: {_sub}"] = {_sub: _feats}
            _DOM2[f"{_b}: {_sub}"] = BLOCK_DOMAIN[_b]
            BLOCK_MEGA[f"{_b}: {_sub}"] = _b      # split blocks share the original mega
    elif _b in REGROUP:
        for _newname, _sublist, _mega in REGROUP[_b]:
            _BLK2[_newname] = {s: _subs[s] for s in _sublist}
            _DOM2[_newname] = BLOCK_DOMAIN[_b]
            BLOCK_MEGA[_newname] = _mega          # explicit: own mega (role/skill) or shared (facets)
    else:
        _BLK2[_b] = _subs
        _DOM2[_b] = BLOCK_DOMAIN[_b]
        BLOCK_MEGA[_b] = _b
BLOCKS, BLOCK_DOMAIN = _BLK2, _DOM2
_REQUESTED_OWN_MEGAS = (
    "3PT Shooting: Pull-Up + Self Created 3PT",
    "3PT Shooting: C+S + Off-Ball 3PT",
    "Rim",
    "Paint - Non Rim",
    "Drive Tendencies",
    "Perimeter Defense",
    "Perimeter Defense Matchups",
    "Paint Defense",
    "Paint Defense Matchups",
    "Defensive Matchups",
    "Defensive Versatility",
)
_missing_requested_own_megas = [b for b in _REQUESTED_OWN_MEGAS if b not in BLOCK_MEGA]
if _missing_requested_own_megas:
    raise KeyError(f"Requested split blocks missing after regroup: {_missing_requested_own_megas}")
for _own_mega_block in _REQUESTED_OWN_MEGAS:
    BLOCK_MEGA[_own_mega_block] = _own_mega_block


# Combine the perimeter skill/matchup and paint skill/matchup families.
# The original blocks remain equal child blocks, so hierarchy size cannot inflate
# the combined mega prior. Existing within-block subgroup balancing is preserved.
_COMBINED_DEFENSE_MEGAS = {
    "Perimeter Defense + Matchups": (
        "Perimeter Defense", "Perimeter Defense Matchups",
    ),
    "Paint Defense + Matchups": (
        "Paint Defense", "Paint Defense Matchups",
    ),
}
for _combined_mega, _children in _COMBINED_DEFENSE_MEGAS.items():
    for _child in _children:
        BLOCK_MEGA[_child] = _combined_mega

# ----------------------------------------------------------------------------
# FREQUENCY / QUALITY SPLIT: many offense subgroups bundle a volume/frequency
# feature (how OFTEN a player does something) with shot-making/efficiency/talent
# features (how WELL they do it) into one PCA-whitened composite. That couples two
# independent signals -- a high-volume, low-efficiency shooter (Otto Porter Jr.
# 2016-17, 43% on low volume) ends up looking like a high-volume, high-efficiency
# shooter (Aaron Gordon) because the composite can't tell "does it a lot" from
# "does it well". Splitting each into its own "X: Frequency" / "X: Quality"
# subgroup (still inside the same block / same mega-block fair-prior budget) lets
# the continuity regression and personalization weight volume and quality
# independently. Playmaking Volume is additionally ENRICHED with the broader set
# of assist/passing features (some previously commented out / unused).
# ----------------------------------------------------------------------------
SPLIT_FREQ_QUALITY = {
    ("3PT Shooting: Pull-Up + Self Created 3PT", "Pull-Up + Self Created 3PT"): {
        "Frequency": ["3PT Pull Up FGA Per 75", "pct_3fga_unassisted",
                       "three_to_six_drib_3FGA_per_75", "seven_plus_drib_3FGA_per_75", ],
        "Shot Quality": ["3PT Pull Up Openness Rating", "3PT Pull Up Shot Creation", "3PT Pull Up Shot Quality",
                          "3PT Shot Creation", "On-Ball Gravity", "Stable Avg 3PT Shot Distance"],
        "Efficiency": ["3PT Pull Up Shot Making", "3PT Pull Up Shot Making Efficiency", "3PT Pull Up Talent",
                        "Stable Pull Up 3PT%", "three_to_six_drib_3FGA_accuracy", "seven_plus_drib_3FGA_accuracy",
                        "Deep 3PT Shooting Talent"],
    },
    ("3PT Shooting: C+S + Off-Ball 3PT", "C+S + Off-Ball 3PT"): {
        "Frequency": ["C&S to Pull Up Ratio", "one_two_drib_3fga_per_75", "pct_3fga_assisted",
                       "Corner 3 to ATB Ratio", "catch_shoot_3fga_per_75"],
        "Shot Quality": ["C&S 3PT Openness Rating", "C&S 3PT Shot Quality", "Off-Ball Gravity"],
        "Efficiency": ["Stable C&S 3PT%", "C&S 3PT Shooting Talent", "C&S 3PT Shot Making",
                        "C&S 3PT Shot Making Efficiency", "one_two_drib_3PA_accuracy", "Stable Corner 3PT%"],
    },
    ("Pull Up Midrange", "Pull Up Midrange"): {
        "Frequency": ["Midrange Pull Up FGA Per 75"],
        "Shot Quality": ["Midrange Pull Up Shot Creation", "Midrange Pull Up Shot Quality"],
        "Efficiency": ["Midrange Pull Up FG%", "Midrange Pull Up Shot Making",
                        "Midrange Pull Up Shot Making Efficiency", "Midrange Pull Up Talent"],
    },
    ("Non Paint Midrange", "Non Paint Midrange"): {
        "Frequency": ["non_paint_midrange_fga_per_75", "Stable Avg 2PT Shot Distance",
                       "pct_2fga_assisted", "pct_2fga_unassisted"],
        "Shot Quality": ["Non-Paint Midrange Shot Creation", "Non-Paint Midrange Shot Quality"],
        "Efficiency": ["Non-Paint Midrange Shooting Talent", "Non-Paint Midrange Shot Making",
                        "Non-Paint Midrange Shot Making Efficiency", "non_paint_midrange_fg_pct"],
    },
    ("Short Midrange", "Short Midrange"): {
        "Frequency": ["SMR FGA PER 75"],
        "Efficiency": ["SMR FG%"],
    },
    ("Long Midrange", "Long Midrange"): {
        "Frequency": ["LMR FGA PER 75"],
        "Efficiency": ["LMR FG%"],
    },
    ("Rim", "Rim"): {
        "Frequency": ["Rim Shot Attempts Per 75 Possessions", "RA_fga_per_75", "dunks_per_game", "Rim Attempt Consistency", "Rim Makes Consistency"],
        "Shot Quality": ["% of Shots at Rim Unassisted", "Rim Openness Rating", "Rim Shot Creation", "Rim Shot Quality"],
        "Efficiency": ["Stable Rim FG%", "Rim Shot Making", "Rim Shot Making Efficiency","RA_fg%", "Finishing Talent", "Contact Finish Rate"],
    },
    ("Paint - Non Rim", "Paint - Non Rim"): {
        "Frequency": ["paint_non_ra_fga_per_75"],
        "Shot Quality": ["Paint Openness Rating", "Paint Shot Creation", "Paint Shot Quality",
                          "Floater Shot Creation", "Floater Shot Quality"],
        "Efficiency": ["Paint Shooting Talent", "Paint Shot Making", "Paint Shot Making Efficiency", "non_ra_fg%",
                        "Floater FG%", "Floater Shot Making", "Floater Shot Making Efficiency", "Floater Talent"],
    },
    ("Drive Tendencies", "Drive Tendencies"): {
        "Frequency": ["Drives Per 75 Possessions", "drive_fga_per_75", "Drive Pass Rate"],
        "Quality": ["Drive TOV Rate", "drive_fg_pct", "drive_PPP"],
    },
    ("Cut", "Cut"): {
        "Frequency": ["Cuts Per 75 Possessions"],
        "Quality": ["Stable Cut PPP", "Cut Scoring Impact Per 75", "cut_tov_freq"],
    },
    ("Handoff", "Handoff"): {
        "Frequency": ["handoff_per_75"],
        "Quality": ["Stable Handoff PPP", "handoff_tov_freq"],
    },
    ("Isolations", "Isolations"): {
        "Frequency": ["isolation_per_75"],
        "Quality": ["Stable Isolation PPP", "Isolation Turnover Rate", "Isolation Shooting Talent",
                     "Isolation Shot Making", "Isolation Shot Making Efficiency", "Isolation Shot Quality"],
    },
    ("Off-Ball Screens", "Off-Ball Screens"): {
        "Frequency": ["Off-Ball Screen Possessions Per 75 Possessions"],
        "Quality": ["Stable Off-Ball Screen PPP", "Off-Screen Impact Per 75", "off_screen_tov_freq"],
    },
    ("P&R Ball Handler", "P&R Ball Handler"): {
        "Frequency": ["P&R Creation Rate", "p_r_ball_handler_per_75"],
        "Quality": ["Stable P&R Ball Handler PPP", "PnR Ball Handler Shooting Talent",
                     "PnR Ball Handler Shot Making", "PnR Ball Handler Shot Making Efficiency",
                     "PnR Ball Handler Shot Quality", "p_r_ball_handler_tov_freq"],
    },
    ("P&R Roll Man", "P&R Roll Man"): {
        "Frequency": ["Roll Man Poss Per 75 Possessions"],
        "Quality": ["Stable P&R Roll Man PPP", "Roll Man Impact Per 75 Possessions",
                     "PnR Screener Shooting Talent", "PnR Screener Shot Making",
                     "PnR Screener Shot Making Efficiency", "PnR Screener Shot Quality",
                     "p_r_roll_man_tov_freq"],
    },
    ("Spot Up", "Spot Up"): {
        "Frequency": ["Spot Up%", "spot_up_per_75"],
        "Quality": ["Stable Spot Up PPP", "spot_up_tov_freq"],
    },
    ("Transition", "Transition"): {
        "Frequency": ["Transition Frequency Impact", "Offensive Transition Frequency Impact",
                       "transition_per_75"],
        "Quality": ["Stable Transition PPP", "Transition Shooting Talent", "Transition Shot Creation",
                     "Transition Shot Making", "Transition Shot Making Efficiency", "Transition Shot Quality",
                     "transition_tov_freq"],
    },
    ("Post Ups", "Post Ups"): {
        "Frequency": ["post_up_per_75"],
        "Quality": ["Stable Post Up PPP", "Post Up Impact Per 75 Possessions", "Post Up Shooting Talent",
                     "Post Up Shot Making", "Post Up Shot Making Efficiency", "Post Up Shot Quality",
                     "post_up_tov_freq"],
    },
}

for _blk, _sub in list(SPLIT_FREQ_QUALITY.keys()):
    _splits = SPLIT_FREQ_QUALITY[(_blk, _sub)]
    _old_feats = BLOCKS[_blk].pop(_sub)
    _new_feats = sorted(f for v in _splits.values() for f in v)
    if _blk != "Playmaking":
        assert sorted(_old_feats) == _new_feats, (
            f"{_blk} | {_sub}: feature mismatch\n"
            f"  old-only: {sorted(set(_old_feats) - set(_new_feats))}\n"
            f"  new-only: {sorted(set(_new_feats) - set(_old_feats))}")
    for _new_sub, _feats in _splits.items():
        BLOCKS[_blk][f"{_sub}: {_new_sub}"] = _feats

# ----------------------------------------------------------------------------
# Validate block features + bookkeeping (blocks, subgroups, domains)
# ----------------------------------------------------------------------------
missing = [(g, sub, f)
           for g, subs in BLOCKS.items()
           for sub, feats in subs.items()
           for f in feats if f not in df.columns]
if missing:
    for g, sub, f in missing:
        print(f"  MISSING  {g} | {sub} | {f}")
    raise ValueError(f"{len(missing)} block features missing from dataset.")

BLOCK_NAMES = list(BLOCKS.keys())
G = len(BLOCK_NAMES)
OFF_BLOCKS = [g for g in BLOCK_NAMES if BLOCK_DOMAIN[g] == "Offense"]
DEF_BLOCKS = [g for g in BLOCK_NAMES if BLOCK_DOMAIN[g] == "Defense"]
IS_OFF = np.array([BLOCK_DOMAIN[g] == "Offense" for g in BLOCK_NAMES])
_OFF_IDX = [BLOCK_NAMES.index(g) for g in OFF_BLOCKS]
_DEF_IDX = [BLOCK_NAMES.index(g) for g in DEF_BLOCKS]

SUBS = [(g, sub, feats) for g in BLOCK_NAMES for sub, feats in BLOCKS[g].items()]
_assert_forbidden_absent(
    [feature for _, _, features in SUBS for feature in features],
    "SUBS",
)
NSUB = len(SUBS)
SUB_BLOCK = np.array([BLOCK_NAMES.index(g) for g, _, _ in SUBS])     # block of each subgroup
SUB_NAMES = [f"{g} | {sub}" for g, sub, _ in SUBS]
SUB_IS_OFF = np.array([BLOCK_DOMAIN[BLOCK_NAMES[b]] == "Offense" for b in SUB_BLOCK])
OFF_SUB_IDX = np.where(SUB_IS_OFF)[0]
DEF_SUB_IDX = np.where(~SUB_IS_OFF)[0]
_setup_print(f"{G} blocks / {NSUB} subgroups ({len(OFF_SUB_IDX)} offense + {len(DEF_SUB_IDX)} defense subgroups)")

# ----------------------------------------------------------------------------
# HIERARCHICALLY FAIR PRIOR: every mega-block receives an equal domain share, that
# share is split equally among its child blocks, and each block's share is split
# equally among its subgroups. Thus feature count, retained PCA count, subgroup
# count, and child-block count cannot mechanically increase an allocation.
# Continuity learning can still move weight when the data provides evidence.
# ----------------------------------------------------------------------------
MEGA_OF_SUB = np.array([BLOCK_MEGA[BLOCK_NAMES[b]] for b in SUB_BLOCK])

def _megafair_prior(sub_idx):
    megas = MEGA_OF_SUB[sub_idx]
    uniq = list(dict.fromkeys(megas))                       # mega-blocks in this domain
    per_mega = 1.0 / len(uniq)
    p = np.zeros(len(sub_idx))
    for m in uniq:
        local = np.where(megas == m)[0]
        blocks = SUB_BLOCK[sub_idx[local]]
        uniq_blocks = np.unique(blocks)
        per_block = per_mega / len(uniq_blocks)
        for b in uniq_blocks:
            members = local[blocks == b]
            p[members] = per_block / len(members)
    return p / p.sum()

PRIOR_OFF_SUB = _megafair_prior(OFF_SUB_IDX)
PRIOR_DEF_SUB = _megafair_prior(DEF_SUB_IDX)
_PRIOR_SUB_ALL = np.zeros(NSUB)
_PRIOR_SUB_ALL[OFF_SUB_IDX] = PRIOR_OFF_SUB
_PRIOR_SUB_ALL[DEF_SUB_IDX] = PRIOR_DEF_SUB
PRIOR_BLOCK = np.array([_PRIOR_SUB_ALL[SUB_BLOCK == gi].sum() for gi in range(G)])
for _domain_idx in (OFF_SUB_IDX, DEF_SUB_IDX):
    _mega_totals = [PRIOR_BLOCK[[BLOCK_MEGA[g] == m for g in BLOCK_NAMES]].sum()
                    for m in dict.fromkeys(MEGA_OF_SUB[_domain_idx])]
    assert np.ptp(_mega_totals) < 1e-12, "mega-block prior depends on hierarchy size"


# ----------------------------------------------------------------------------
# 1-2. Within-season z-score, per-subgroup PCA + whitening -> embedding E
# ----------------------------------------------------------------------------
def standardize_within_season(features):
    """z = (x - mean) / std computed within each season (removes era / pace)."""
    cols = {}
    for c in features:
        x = pd.to_numeric(df[c], errors="coerce")
        m = x.groupby(df["season"]).transform("mean")
        sd = x.groupby(df["season"]).transform("std").replace(0.0, np.nan)
        cols[c] = ((x - m) / sd).fillna(0.0)
    return pd.concat(cols, axis=1).values


sub_slices = []                 # per-subgroup column range in E
sub_n_components = []            # retained PCA dimensions for each subgroup
block_slices = {}               # per-block column range in E (its subgroups are contiguous)
_parts, _white, _cur = [], [], 0
for _g, _sub, _feats in SUBS:
    _dom = BLOCK_DOMAIN[_g]
    Z = standardize_within_season(_feats)
    pca = PCA(n_components=min(Z.shape), random_state=RANDOM_STATE).fit(Z)
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), EXPLAINED_VAR[_dom]) + 1)
    k = max(1, min(k, Z.shape[1]))
    scores = pca.transform(Z)[:, :k]
    eig = pca.explained_variance_[:k]
    white = scores / np.sqrt(eig + WHITEN_SHRINKAGE[_dom] * eig.mean())  # ~unit var per PC
    _white.append(white)
    _parts.append(white)
    sub_slices.append((_cur, _cur + k))
    sub_n_components.append(k)
    _cur += k
for _gi, _g in enumerate(BLOCK_NAMES):
    _ss = [sub_slices[u] for u in range(NSUB) if SUB_BLOCK[u] == _gi]
    block_slices[_g] = (_ss[0][0], _ss[-1][1])
SUB_N_COMPONENTS = np.asarray(sub_n_components, dtype=int)
E = np.hstack(_parts)            # whitened embedding; aggregation normalizes dimensions explicitly
E_WHITE = np.hstack(_white)      # whitened embedding used for distinctiveness
_PCA_CUSTOM_EMBEDDING_MANIFEST = [
    {"subgroup": SUB_NAMES[index], "features": list(SUBS[index][2]),
     "retained_components": int(SUB_N_COMPONENTS[index])}
    for index in range(NSUB)
]
_assert_forbidden_absent(
    [feature for row in _PCA_CUSTOM_EMBEDDING_MANIFEST for feature in row["features"]],
    "PCA/custom embedding manifest",
)
_setup_print(f"Embedding: {E.shape[0]} player-seasons x {E.shape[1]} dims")


# ----------------------------------------------------------------------------
# DISTINCTIVENESS: per player x block, SEASON-AND-POSITION-GROUP-RELATIVE on
# whitened coords (vs guard/wing/big peers from the same season). First compute
# RMS standardized PC energy. Components are averaged within each subgroup first,
# then subgroup energies are averaged within the block. This prevents either more
# retained PCs or more subgroups from mechanically increasing block attention.
# Then rank that RMS within the exact peer pool and apply a soft normal-score map.
# The exponential keeps every block positive for the downstream squared gate,
# while scale<1 preserves tail separation without collapsing ordinary traits.
# ----------------------------------------------------------------------------
DISTINCTIVENESS = np.zeros((N, G))
for _gi, _g in enumerate(BLOCK_NAMES):
    _s, _e = block_slices[_g]
    _Xg = E_WHITE[:, _s:_e]
    _dev = np.empty_like(_Xg)
    for _season in df["season"].unique():
        for _p in np.unique(PLAYER_GROUP):
            _m = (df["season"].values == _season) & (PLAYER_GROUP == _p)
            _group_x = _Xg[_m]
            _group_sd = _group_x.std(axis=0)
            _group_sd[_group_sd == 0.0] = 1.0
            _dev[_m] = (_group_x - _group_x.mean(axis=0)) / _group_sd
    _sub_energy = []
    for _u in np.where(SUB_BLOCK == _gi)[0]:
        _us, _ue = sub_slices[_u]
        _sub_energy.append(np.mean(_dev[:, _us - _s:_ue - _s] ** 2, axis=1))
    _rms = np.sqrt(np.mean(_sub_energy, axis=0))
    for _season in df["season"].unique():
        for _p in np.unique(PLAYER_GROUP):
            _m = (df["season"].values == _season) & (PLAYER_GROUP == _p)
            _rank = pd.Series(_rms[_m]).rank(method="average").to_numpy()
            _pct = (_rank - 0.5) / _m.sum()       # finite mid-rank plotting positions
            _rank_z = norm.ppf(_pct)              # ordinal score (saturates in the tail)
            _sd = _rms[_m].std()                  # magnitude score: RMS standardized WITHIN the
            _mag_z = (_rms[_m] - _rms[_m].mean()) / (_sd if _sd > 0 else 1.0)   # peer pool (so it
            #                                       stays comparable across blocks) -- preserves how
            #                                       many SD out this player is, not just their rank.
            _z = (1.0 - DISTINCT_MAGNITUDE) * _rank_z + DISTINCT_MAGNITUDE * _mag_z
            DISTINCTIVENESS[_m, _gi] = np.exp(DISTINCTIVENESS_SOFT_SCALE * _z)


# ----------------------------------------------------------------------------
# 3. Continuity training at SUBGROUP resolution (offense / defense x position group)
# ----------------------------------------------------------------------------
def _sub_sqdists(i_idx, j_idx, sub_idx):
    """Mean squared PC distance per subgroup; invariant to retained component count."""
    diff = E[i_idx] - E[j_idx]
    out = np.empty((len(i_idx), len(sub_idx)))
    for c, u in enumerate(sub_idx):
        s, e = sub_slices[u]
        out[:, c] = (diff[:, s:e] ** 2).mean(axis=1)
    return out


positive_pairs, gap2_pairs = [], []
for _pl, _idxs in df.groupby("player").groups.items():
    ymap = {int(df.at[ix, "_yr"]): ix for ix in _idxs}
    for y, ix in ymap.items():
        if y + 1 in ymap:
            positive_pairs.append((ix, ymap[y + 1]))
        if y + 2 in ymap:
            gap2_pairs.append((ix, ymap[y + 2]))
positive_pairs = np.array(positive_pairs)
gap2_pairs = np.array(gap2_pairs)
_setup_print(f"{len(positive_pairs)} same-player consecutive-season pairs"
      + (f" (+{len(gap2_pairs)} two-season-gap pairs)" if USE_GAP2_PAIRS else ""))


def _train_pool(pos_pairs, extra_gap2):
    return np.vstack([pos_pairs, extra_gap2]) if (USE_GAP2_PAIRS and len(extra_gap2)) else pos_pairs


def _sample_negatives(rows, n):
    a = rng.choice(rows, n); b = rng.choice(rows, n)
    bad = players[a] == players[b]
    while bad.any():
        b[bad] = rng.choice(rows, bad.sum()); bad = players[a] == players[b]
    return np.stack([a, b], axis=1)


def learn_sub_weights(pos_pairs, candidate_rows, sub_idx, prior):
    """Non-negative logistic regression over subgroup sq-distances -> weights (sum 1).
    Regularization is a RIDGE TOWARD THE MEGA-BLOCK-FAIR PRIOR (not toward 0): absent
    re-identification signal the weights sit at `prior` (every mega-block equal), and
    L2_REG controls how much signal is needed to move a block off that baseline."""
    nb = len(sub_idx)
    neg = _sample_negatives(candidate_rows, N_NEG_PER_POS * len(pos_pairs))
    Dp = _sub_sqdists(pos_pairs[:, 0], pos_pairs[:, 1], sub_idx)
    Dn = _sub_sqdists(neg[:, 0], neg[:, 1], sub_idx)
    X = np.vstack([Dp, Dn]); y = np.r_[np.ones(len(Dp)), np.zeros(len(Dn))]
    scale = Dn.mean(axis=0) + 1e-9
    Xs = X / scale
    t = prior * scale          # prior expressed in theta-space (weight = theta / scale)

    def obj(theta):
        a, w = theta[0], theta[1:]
        z = a - Xs @ w
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * L2_REG * np.sum((w - t) ** 2) / len(y)
        r = (p - y) / len(y)
        return loss, np.r_[r.sum(), -(Xs.T @ r) + L2_REG * (w - t) / len(y)]

    res = minimize(obj, np.r_[2.0, t], jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] + [(0.0, None)] * nb)
    w = res.x[1:] / scale
    return (1.0 - LEARN_BLEND) * prior + LEARN_BLEND * (w / w.sum())


# Production models: one per position group, offense and defense separately.
W_OFF_SUB_BY_GROUP, W_DEF_SUB_BY_GROUP = {}, {}
for _grp in GROUPS:
    _rows = np.where(PLAYER_GROUP == _grp)[0]
    _pos = positive_pairs[PLAYER_GROUP[positive_pairs[:, 0]] == _grp]
    _g2 = gap2_pairs[PLAYER_GROUP[gap2_pairs[:, 0]] == _grp] if len(gap2_pairs) else gap2_pairs
    _tp = _train_pool(_pos, _g2)
    W_OFF_SUB_BY_GROUP[_grp] = learn_sub_weights(_tp, _rows, OFF_SUB_IDX, PRIOR_OFF_SUB)
    W_DEF_SUB_BY_GROUP[_grp] = learn_sub_weights(_tp, _rows, DEF_SUB_IDX, PRIOR_DEF_SUB)

# Stabilize continuity weights before personalization. Block weights are shrunk
# toward the mega-block-fair prior, and within-block subgroup proportions are
# independently shrunk toward uniform. This preserves empirical continuity signal
# without letting sparse role/frequency identifiers define the entire metric.
W_OFF_BY_GROUP, W_DEF_BY_GROUP, _MULT_BY_GROUP, _BLK_BY_GROUP = {}, {}, {}, {}
for _grp in GROUPS:
    _w_all = np.zeros(NSUB)
    _w_all[OFF_SUB_IDX] = W_OFF_SUB_BY_GROUP[_grp]
    _w_all[DEF_SUB_IDX] = W_DEF_SUB_BY_GROUP[_grp]
    _raw_blk = np.array([_w_all[SUB_BLOCK == gi].sum() for gi in range(G)])
    _prop = np.empty(NSUB)
    for gi in range(G):
        _m = SUB_BLOCK == gi
        _learned_prop = (_w_all[_m] / _raw_blk[gi]) if _raw_blk[gi] > 0 else np.full(_m.sum(), 1.0 / _m.sum())
        _uniform_prop = np.full(_m.sum(), 1.0 / _m.sum())
        _prop[_m] = ((1.0 - LEARNED_SUBGROUP_SHRINKAGE) * _uniform_prop
                     + LEARNED_SUBGROUP_SHRINKAGE * _learned_prop)
    _blk = ((1.0 - LEARNED_BLOCK_SHRINKAGE) * PRIOR_BLOCK
            + LEARNED_BLOCK_SHRINKAGE * _raw_blk)
    _blk[_OFF_IDX] /= _blk[_OFF_IDX].sum()
    _blk[_DEF_IDX] /= _blk[_DEF_IDX].sum()
    _BLK_BY_GROUP[_grp] = _blk
    _MULT_BY_GROUP[_grp] = _prop  # sums to 1 inside every block
    W_OFF_BY_GROUP[_grp] = {g: float(_blk[BLOCK_NAMES.index(g)]) for g in OFF_BLOCKS}
    W_DEF_BY_GROUP[_grp] = {g: float(_blk[BLOCK_NAMES.index(g)]) for g in DEF_BLOCKS}

# Per-player signatures: block weights (N x G) + within-block proportions (N x NSUB).
BASE_W = np.zeros((N, G))
SUB_MULT = np.zeros((N, NSUB))
for _grp in GROUPS:
    _m = PLAYER_GROUP == _grp
    BASE_W[_m] = _BLK_BY_GROUP[_grp]
    SUB_MULT[_m] = _MULT_BY_GROUP[_grp]

# Offense/defense IDENTITY from impact metrics (see config). Always season- and
# position-group-relative; drives the per-player off/def balance (off_weight).
def _position_identity(metric_weights):
    """Season-position-relative percentile composite of impact metrics -> [0, 1].
    Each metric is ranked within season x position GROUP (guard/wing/big),
    weight-averaged, then re-ranked within the same season x group. Blending
    O-LEBRON with Box Creation pulls
    non-creator bigs (Gobert) down on offense; D-LEBRON (luck-adjusted) resists
    defensive team-inflation. This drives off_weight ONLY; the which-skills-spike
    personalization still uses the peer-calibrated RMS DISTINCTIVENESS above."""
    grp = pd.Series(PLAYER_GROUP)
    season = df["season"]
    acc = np.zeros(N); wsum = 0.0
    for _c, _w in metric_weights.items():
        x = pd.to_numeric(df[_c], errors="coerce")
        acc += _w * x.groupby([season, grp]).rank(pct=True).values
        wsum += _w
    return pd.Series(acc / wsum).groupby([season, grp]).rank(pct=True).values

OFF_IDENTITY = _position_identity(OFF_IDENTITY_METRICS)

# Defensive IDENTITY = season-relative percentile composite of curated perimeter/rim/help
# features, gated by season-position-relative Matchup Difficulty,
# blended with D-LEBRON, then scaled by a per-position defensive-impact ceiling (see
# DEF_GROUP_CEILING above).
def _lw_pct(col):
    x = pd.to_numeric(df[col], errors="coerce")
    return x.groupby(df["season"]).rank(pct=True).values

DEF_PERIM_SCORE = np.mean([_lw_pct(f) for f in DEF_PERIMETER_FEATS], axis=0)
DEF_RIM_SCORE = np.mean([_lw_pct(f) for f in DEF_RIM_FEATS], axis=0)
DEF_REB_SCORE = np.mean([_lw_pct(f) for f in DEF_REBOUND_FEATS], axis=0)
_def_match = pd.to_numeric(df[DEF_MATCHUP_FEAT], errors="coerce")
DEF_MATCH_GRP = _def_match.groupby([df["season"], pd.Series(PLAYER_GROUP)]).rank(pct=True).values

def _def_skill_identity():
    grp = pd.Series(PLAYER_GROUP)
    base_rim = np.array([DEF_BASE_RIM[g] for g in PLAYER_GROUP])
    rim_w = np.clip(base_rim + DEF_BOOST_GAIN * np.maximum(0.0, DEF_RIM_SCORE - DEF_BOOST_THRESH), 0.0, 0.95)
    gate = DEF_GATE_FLOOR + (1 - DEF_GATE_FLOOR) * DEF_MATCH_GRP ** DEF_GATE_POWER
    gated_perim = 0.5 + (DEF_PERIM_SCORE - 0.5) * gate
    skill = rim_w * DEF_RIM_SCORE + (1 - rim_w) * gated_perim
    skill_id = pd.Series(skill).groupby([df["season"], grp]).rank(pct=True).values
    def_id = (1 - DEF_DLEBRON_W) * skill_id + DEF_DLEBRON_W * DLEBRON_ID
    ceiling = np.array([DEF_GROUP_CEILING[g] for g in PLAYER_GROUP])
    return def_id * ceiling

DLEBRON_ID = _position_identity({"D-LEBRON": 1.0})
DEF_IDENTITY = _def_skill_identity()


def _off_weight_from_identity(off_id, def_id):
    """Bounded o/(o+d): impact sets the balance without erasing either domain."""
    o = OFFENSE_BIAS * off_id ** OFF_WEIGHT_GAMMA
    d = def_id ** OFF_WEIGHT_GAMMA
    raw = o / (o + d + 1e-12)
    return np.clip(raw, MIN_DOMAIN_WEIGHT, 1.0 - MIN_DOMAIN_WEIGHT)


def auto_off_weight(q_pos):
    return float(_off_weight_from_identity(OFF_IDENTITY[q_pos], DEF_IDENTITY[q_pos]))


def _peer_pct(col):
    x = pd.to_numeric(df[col], errors="coerce")
    return x.groupby([df["season"], pd.Series(PLAYER_GROUP)]).rank(pct=True).fillna(0.5).values


def _league_pct(col):
    x = pd.to_numeric(df[col], errors="coerce")
    return x.groupby(df["season"]).rank(pct=True).fillna(0.5).values


_minutes = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0).values
_minute_reliability = np.clip(_minutes / 1500.0, 0.0, 1.0)
_physical = np.mean([_league_pct("Height"), _league_pct("Weight")], axis=0)
_interior_matchup = np.mean([
    _league_pct("% of Time Guarding Power Forwards"),
    _league_pct("% of Time Guarding Centers"),
    _league_pct("Avg Height Guarded"),
], axis=0)

_paint_opportunity = np.mean([
    _peer_pct("Rim Contests Per 75 Possessions"),
    _peer_pct("Stable Rim DFGA Per 75 Possessions"),
    _peer_pct("Primary Defender Involvement Rate"),
    _peer_pct("% of Time Guarding Power Forwards"),
    _peer_pct("% of Time Guarding Centers"),
], axis=0)
_paint_performance = np.mean([
    _peer_pct("Stable Rim dFG% vs. Expected"),
    _peer_pct("Rim Points Saved Per 75 Possessions"),
    _peer_pct("Stable Blocked Shots At Rim Per 75"),
    _peer_pct("Rim Deterrence Per 100"),
], axis=0)
_paint_support = np.mean([
    _paint_opportunity, _paint_performance, _interior_matchup, _physical
], axis=0)
_paint_reliability = _minute_reliability * np.sqrt(np.clip(_paint_opportunity, 0.0, 1.0))
_paint_role = np.mean([
    _paint_opportunity, _paint_support, _interior_matchup, _physical
], axis=0)
_paint_peripheral = 1.0 - np.sqrt(_physical * _interior_matchup)

_drb_opportunity = np.mean([
    _peer_pct("Defensive Rebonding Chances Per 75 Possessions"),
    _peer_pct("Stable Defensive Rebounds Per 75"),
    _peer_pct("Percentage of Defensive Rebounds Contested"),
], axis=0)
_drb_performance = np.mean([
    _peer_pct("Adjusted Defensive Rebounding Success Rate"),
    _peer_pct("Defensive Rebounding Conversion Skill"),
    _peer_pct("Defensive Rebounding Crashing Skill"),
], axis=0)
_drb_support = np.mean([_drb_opportunity, _drb_performance, _physical], axis=0)
_drb_reliability = _minute_reliability * np.sqrt(np.clip(_drb_opportunity, 0.0, 1.0))
_drb_role = np.mean([
    _drb_opportunity, _drb_support, _drb_performance, _physical
], axis=0)
_drb_peripheral = 1.0 - _physical


def _role_relevance_gate(role_score, reliability, peripheral_probability):
    relevance = ROLE_GATE_FLOOR + (1.0 - ROLE_GATE_FLOOR) / (
        1.0 + np.exp(-(role_score - ROLE_GATE_THRESHOLD) / ROLE_GATE_WIDTH)
    )
    gate = 1.0 - peripheral_probability * (1.0 - relevance)
    gate *= 1.0 - (
        0.20 * peripheral_probability * (1.0 - relevance) * (1.0 - reliability)
    )
    defensive_weight = 1.0 - _off_weight_from_identity(OFF_IDENTITY, DEF_IDENTITY)
    defensive_role = np.clip((defensive_weight - MIN_DOMAIN_WEIGHT) / 0.50, 0.0, 1.0)
    gate *= 1.0 - (
        ROLE_GATE_IDENTITY_STRENGTH
        * peripheral_probability
        * (1.0 - defensive_role)
    )
    return np.clip(gate, 0.25, 1.0)


PAINT_ROLE_GATE = _role_relevance_gate(
    _paint_role, _paint_reliability, _paint_peripheral
)
DRB_ROLE_GATE = _role_relevance_gate(
    _drb_role, _drb_reliability, _drb_peripheral
)


def percentile_similarity(dist):
    """0-100 distance-calibrated similarity; monotone, so match order is unchanged."""
    d = np.asarray(dist, dtype=float)
    valid = np.isfinite(d)
    positive = d[valid & (d > 0)]
    if positive.size == 0:
        return np.where(valid, 100.0, 0.0)
    scale = np.median(positive)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = positive.max() if positive.max() > 1e-12 else 1.0
    sim = 100.0 / (1.0 + (d / scale) ** 2)
    return np.where(valid, sim, 0.0)


# ----------------------------------------------------------------------------
# Symmetric personalized distances
# ----------------------------------------------------------------------------
def _hierarchical_profile(scores):
    """Normalize block scores as mega -> block, removing child-block-count effects."""
    out = np.zeros_like(scores, dtype=float)
    for _idx in (_OFF_IDX, _DEF_IDX):
        _idx = np.asarray(_idx)
        _megas = np.array([BLOCK_MEGA[BLOCK_NAMES[i]] for i in _idx])
        _uniq = list(dict.fromkeys(_megas))
        _mega_scores = np.column_stack([scores[:, _idx[_megas == m]].mean(axis=1) for m in _uniq])
        _mega_scores /= _mega_scores.sum(axis=1, keepdims=True)
        for _mi, _m in enumerate(_uniq):
            _children = _idx[_megas == _m]
            _within = scores[:, _children]
            _within /= _within.sum(axis=1, keepdims=True)
            out[:, _children] = _mega_scores[:, [_mi]] * _within
    return out


def _cap_simplex(weights, cap):
    """Cap row-wise simplex weights and redistribute excess proportionally."""
    if cap >= 1.0:
        return weights
    out = weights.copy()
    for _ in range(out.shape[1]):
        over = out > cap
        if not over.any():
            break
        excess = np.where(over, out - cap, 0.0).sum(axis=1, keepdims=True)
        out = np.where(over, cap, out)
        under = ~over
        under_sum = (out * under).sum(axis=1, keepdims=True)
        out += np.where(under, excess * out / np.where(under_sum == 0, 1.0, under_sum), 0.0)
    return out


def _row_power_sharpen(values, gamma):

    gamma = np.asarray(gamma, dtype=float)

    if gamma.ndim == 0:

        if float(gamma) == 1.0:

            return values

        out = values ** float(gamma)

    else:

        out = values ** gamma[:, None]

    return out / out.sum(axis=1, keepdims=True)




def _resolve_sharpen_config(gamma):

    if gamma is None:

        return {"mode": SHARPENING_STRATEGY,

                "base_gamma": WEIGHT_SHARPEN_GAMMA,

                "child_gamma": ADAPTIVE_CHILD_GAMMA,

                "tau": ADAPTIVE_SHARPEN_TAU,

                "alpha": ADAPTIVE_SHARPEN_ALPHA,

                "strength": ADAPTIVE_SHARPEN_STRENGTH}

    if isinstance(gamma, dict):

        cfg = {"mode": "fixed", "base_gamma": WEIGHT_SHARPEN_GAMMA,

               "child_gamma": WEIGHT_SHARPEN_GAMMA, "tau": ADAPTIVE_SHARPEN_TAU,

               "alpha": ADAPTIVE_SHARPEN_ALPHA, "strength": ADAPTIVE_SHARPEN_STRENGTH}

        cfg.update(gamma)

        return cfg

    return {"mode": "fixed", "base_gamma": float(gamma), "child_gamma": float(gamma),

            "tau": ADAPTIVE_SHARPEN_TAU, "alpha": ADAPTIVE_SHARPEN_ALPHA,

            "strength": ADAPTIVE_SHARPEN_STRENGTH}




def _domain_mega_evidence(distinctiveness, _idx, _megas, _uniq, tau):

    raw_z = np.log(np.maximum(distinctiveness[:, _idx], 1e-12)) / DISTINCTIVENESS_SOFT_SCALE

    evidence = np.maximum(raw_z - tau, 0.0)

    return np.column_stack([evidence[:, _megas == m].mean(axis=1) for m in _uniq])




def _versatility_from_evidence(mega_evidence, strength_scale):

    pos = mega_evidence.sum(axis=1)

    sq = (mega_evidence ** 2).sum(axis=1)

    eff = np.where(pos > 0, pos ** 2 / (sq + 1e-12), 1.0)

    breadth = (eff - 1.0) / max(mega_evidence.shape[1] - 1, 1)

    strength_gate = pos / (pos + strength_scale + 1e-12)

    return np.where(pos > 0, np.clip(breadth * strength_gate, 0.0, 1.0), 0.0)




def _hierarchical_sharpen(weights, gamma, distinctiveness=None):

    """Sharpen mega allocations and within-mega allocations separately.



    Fixed numeric gamma reproduces the previous behavior. The adaptive modes use

    raw pre-normalization DISTINCTIVENESS values, aggregated by child-block mean

    into each mega, so a weak flat profile does not masquerade as versatility and

    child-block count cannot mechanically raise a mega's influence.

    """

    cfg = _resolve_sharpen_config(gamma)

    mode = cfg["mode"]

    base_gamma = float(cfg["base_gamma"])

    child_gamma = float(cfg.get("child_gamma", base_gamma))

    distinctiveness = DISTINCTIVENESS if distinctiveness is None else distinctiveness

    out = np.zeros_like(weights)

    for _idx in (_OFF_IDX, _DEF_IDX):

        _idx = np.asarray(_idx)

        _megas = np.array([BLOCK_MEGA[BLOCK_NAMES[i]] for i in _idx])

        _uniq = list(dict.fromkeys(_megas))

        _mega = np.column_stack([weights[:, _idx[_megas == m]].sum(axis=1) for m in _uniq])

        if mode == "fixed":

            _mega_out = _row_power_sharpen(_mega, base_gamma)

            _domain_versatility = np.zeros(weights.shape[0])

        else:

            mega_evidence = _domain_mega_evidence(distinctiveness, _idx, _megas, _uniq, float(cfg["tau"]))

            _domain_versatility = _versatility_from_evidence(mega_evidence, float(cfg["strength"]))

            gamma_mega = 1.0 + (base_gamma - 1.0) * (1.0 - _domain_versatility) ** float(cfg["alpha"])

            sharp = _row_power_sharpen(_mega, gamma_mega)

            if mode == "mixture_mega":

                _mega_out = (1.0 - _domain_versatility[:, None]) * sharp + _domain_versatility[:, None] * _mega

                _mega_out /= _mega_out.sum(axis=1, keepdims=True)

            else:

                _mega_out = sharp

        _mega_out = _cap_simplex(_mega_out, MAX_BLOCK_ATTENTION)

        for _mi, _m in enumerate(_uniq):

            _children = _idx[_megas == _m]

            _within = weights[:, _children]

            _within /= _within.sum(axis=1, keepdims=True)

            if mode == "adaptive_both" and len(_children) > 1:

                raw_z = np.log(np.maximum(distinctiveness[:, _children], 1e-12)) / DISTINCTIVENESS_SOFT_SCALE

                child_evidence = np.maximum(raw_z - float(cfg["tau"]), 0.0)

                child_versatility = _versatility_from_evidence(child_evidence, float(cfg["strength"]))

                child_gamma_vec = 1.0 + (base_gamma - 1.0) * (1.0 - child_versatility) ** float(cfg["alpha"])

                _within = _row_power_sharpen(_within, child_gamma_vec)

            else:

                _within = _row_power_sharpen(_within, child_gamma)

            _within = _cap_simplex(_within, MAX_BLOCK_ATTENTION)

            out[:, _children] = _mega_out[:, [_mi]] * _within

    return out



DISTINCT_PROFILE = _hierarchical_profile(DISTINCTIVENESS)
_neutral_profile = _hierarchical_profile(np.ones((1, G)))
for _idx in (_OFF_IDX, _DEF_IDX):
    _megas = [BLOCK_MEGA[BLOCK_NAMES[i]] for i in _idx]
    _totals = [_neutral_profile[0, [j for j in _idx if BLOCK_MEGA[BLOCK_NAMES[j]] == m]].sum()
               for m in dict.fromkeys(_megas)]
    assert np.ptp(_totals) < 1e-12, "personalization depends on child-block count"


def personalized_base(blend=PROFILE_BLEND, gamma=None):
    """Per-player per-BLOCK weight = ADDITIVE blend of the learned group weight and
    the player's OWN distinctiveness, normalized mega -> block within each domain.
        blend = 0 -> pure learned weights (what re-identifies players in general)
        blend = 1 -> pure "what THIS player is distinctive at"
    The group base is already stabilized toward the fair prior; distinctiveness
    then lets genuine player-specific signatures earn most of the allocation.
    After blending, a SHARPENING power `gamma` (default WEIGHT_SHARPEN_GAMMA) is
    applied separately across mega-blocks and across children within each mega, so
    gamma cannot reward or punish a family merely for having more child blocks."""
    _bw = BASE_W.copy()
    for _idx in (_OFF_IDX, _DEF_IDX):
        _bw[:, _idx] /= _bw[:, _idx].sum(axis=1, keepdims=True)
    blended = (1.0 - blend) * _bw + blend * DISTINCT_PROFILE
    return _hierarchical_sharpen(blended, gamma)


def _final_sub_weights(blend=PROFILE_BLEND, gamma=None):

    """N x NSUB distance weights after continuous role-relevance gating.

    Paint Defense and Defensive Rebounding retain their original learned and
    player-distinctive allocations when role evidence supports them. Otherwise
    their within-defense allocations are softly reduced and redistributed across
    the other defensive families before subgroup allocation."""

    block_weights = personalized_base(blend, gamma)
    gated = block_weights.copy()
    paint_blocks = np.array([g == "Paint Defense" for g in BLOCK_NAMES])
    drb_blocks = np.array([BLOCK_MEGA[g] == "Defensive Rebounding" for g in BLOCK_NAMES])
    gated[:, paint_blocks] *= PAINT_ROLE_GATE[:, None]
    gated[:, drb_blocks] *= DRB_ROLE_GATE[:, None]
    reb_blocks = np.array([BLOCK_MEGA[g] in ("Offensive Rebounding", "Defensive Rebounding")
                           for g in BLOCK_NAMES])
    gated[:, reb_blocks] *= REBOUNDING_WEIGHT_SCALE  # halve rebounding's relative share; the
    # domain renorm below restores the off/def totals, redistributing the freed budget across
    # the other offensive/defensive families (mirrors the role-gate redistribution above).
    gated[:, _OFF_IDX] *= (
        block_weights[:, _OFF_IDX].sum(axis=1, keepdims=True)
        / gated[:, _OFF_IDX].sum(axis=1, keepdims=True)
    )
    gated[:, _DEF_IDX] *= (
        block_weights[:, _DEF_IDX].sum(axis=1, keepdims=True)
        / gated[:, _DEF_IDX].sum(axis=1, keepdims=True)
    )
    return gated[:, SUB_BLOCK] * SUB_MULT


# Global offense/defense scales from the default-blend, UNSHARPENED signatures
# (pair-averaged) so the offense/defense balance is calibrated independent of
# WEIGHT_SHARPEN_GAMMA. D(a,b) == D(b,a) preserved.
_FW0 = _final_sub_weights(PROFILE_BLEND, gamma=1.0)
_si = rng.choice(N, 5000); _sj = rng.choice(N, 5000)
_pw0 = 0.5 * (_FW0[_si] + _FW0[_sj])
_sq0 = _sub_sqdists(_si, _sj, np.arange(NSUB))
S_OFF = float((_pw0[:, SUB_IS_OFF] * _sq0[:, SUB_IS_OFF]).sum(axis=1).mean())
S_DEF = float((_pw0[:, ~SUB_IS_OFF] * _sq0[:, ~SUB_IS_OFF]).sum(axis=1).mean())

def _domain_normalized(fw_sub, ow):
    """Rescale fw_sub (..., NSUB) so the offense subgroups sum to `ow` and the defense
    subgroups sum to `1 - ow` (per row) -- makes off_weight mean exactly what it says
    in the blended distance / attention table, independent of how many subgroups each
    domain has or how spread out their distances are (S_OFF != S_DEF because the
    domains have different subgroup structures and whitening strength --
    both would otherwise leak into the off/def blend on top of off_weight itself)."""
    off_sum = (fw_sub * SUB_IS_OFF).sum(axis=-1, keepdims=True)
    def_sum = (fw_sub * ~SUB_IS_OFF).sum(axis=-1, keepdims=True)
    ow = np.asarray(ow)
    if ow.ndim > 0:
        ow = ow[..., None]
    bal = np.where(SUB_IS_OFF, ow, 1.0 - ow)
    denom = np.where(SUB_IS_OFF, off_sum, def_sum)
    return bal * fw_sub / denom

# Additive POSITIONAL-FIT term: penalizes cross-position-group matches, graduated by
# how far apart the groups are on a guard(0) - wing(1) - big(2) perimeter-to-paint
# spectrum, so guard<->big (e.g. PG vs C) is penalized 4x as hard as guard<->wing or
# wing<->big (the SG/SF and PF/C boundaries) -- without banning cross-group comps
# outright. Calibrated like the quality terms above.
POSITION_AXIS = {"guard": 0.0, "wing": 1.0, "big": 2.0}
POSITION_PENALTY_WEIGHT = 0.19
PG_AXIS = np.array([POSITION_AXIS[g] for g in PLAYER_GROUP])
PTERM_REF = float(((PG_AXIS[_si] - PG_AXIS[_sj]) ** 2).mean()) + 1e-12

def _player_balance(off_weight):
    """Per-player offense balance (auto, or a fixed constant for everyone)."""
    if off_weight is None:
        return _off_weight_from_identity(OFF_IDENTITY, DEF_IDENTITY)
    return np.full(N, float(off_weight))


def symmetric_distances(q_pos, blend=PROFILE_BLEND, off_weight=OVERALL_OFF_WEIGHT, gamma=None):
    """Symmetric off / def / overall distances; pair weight = mean of both players'
    blended signatures, so D(a,b) == D(b,a)."""
    fw_sub = _final_sub_weights(blend, gamma)
    ow = _player_balance(off_weight)
    fw_all = _domain_normalized(fw_sub, ow)

    sq = _sub_sqdists(np.arange(N), np.full(N, q_pos), np.arange(NSUB))
    pw_sub = 0.5 * (fw_sub[q_pos] + fw_sub)
    _pterm = (PG_AXIS[q_pos] - PG_AXIS) ** 2 / PTERM_REF
    off_core2 = (pw_sub * sq * SUB_IS_OFF[None, :]).sum(axis=1)
    off_d = np.sqrt(off_core2 + POSITION_PENALTY_WEIGHT * S_OFF * _pterm)
    def_core2 = (pw_sub * sq * (~SUB_IS_OFF)[None, :]).sum(axis=1)
    def_d = np.sqrt(def_core2 + POSITION_PENALTY_WEIGHT * S_DEF * _pterm)
    pw_all = 0.5 * (fw_all[q_pos] + fw_all)
    overall_d = np.sqrt((pw_all * sq).sum(axis=1) + POSITION_PENALTY_WEIGHT * _pterm)
    return off_d, def_d, overall_d



# ----------------------------------------------------------------------------
# Reporting / API
# ----------------------------------------------------------------------------
def show_block_weights(top=6):
    for grp in GROUPS:
        for side, wmap in [("OFFENSE", W_OFF_BY_GROUP[grp]), ("DEFENSE", W_DEF_BY_GROUP[grp])]:
            wt = (pd.DataFrame({"block": list(wmap), "weight": list(wmap.values())})
                  .assign(weight=lambda d: d.weight / d.weight.sum())
                  .sort_values("weight", ascending=False).head(top))
            print(f"\n[{grp}] top {side} blocks: " +
                  ", ".join(f"{b} {w:.2f}" for b, w in zip(wt["block"], wt["weight"])))


def show_subgroup_weights(group="wing"):
    """Learned subgroup weights for one position group (within-domain shares)."""
    w_all = np.zeros(NSUB)
    w_all[OFF_SUB_IDX] = W_OFF_SUB_BY_GROUP[group]
    w_all[DEF_SUB_IDX] = W_DEF_SUB_BY_GROUP[group]
    tbl = (pd.DataFrame({"subgroup": SUB_NAMES,
                         "domain": ["Offense" if o else "Defense" for o in SUB_IS_OFF],
                         "weight": w_all})
           .sort_values("weight", ascending=False).reset_index(drop=True))
    print(f"\n[{group}] learned subgroup weights (each domain sums to 1):")
    print(tbl.to_string(index=False, formatters={"weight": "{:.3f}".format}))
    return tbl


def _resolve(player_name, season):
    nm = df["player"].str.lower() == player_name.lower()
    if not nm.any():
        hits = df.loc[df["player"].str.lower().str.contains(player_name.lower(), na=False),
                      ["player", "season", "team", "position"]].drop_duplicates()
        print(hits.head(40).to_string(index=False))
        raise ValueError(f"No exact match for '{player_name}'")
    if season is not None:
        rm = nm & (df["season"].astype(str) == str(season))
        if not rm.any():
            print(df.loc[nm, ["player", "season", "team"]].to_string(index=False))
            raise ValueError(f"No row for '{player_name}' in '{season}'")
        return df.index[rm][0]
    return df.loc[df.index[nm], "season"].astype(str).sort_values().index[-1]


def _query_weight_table(q, blend, off_weight, gamma=None):
    """Exact per-block feature allocation used by the query side of distance."""
    ow = auto_off_weight(q) if off_weight is None else off_weight
    fw_sub = _domain_normalized(_final_sub_weights(blend, gamma)[q], ow)
    fw = np.array([fw_sub[SUB_BLOCK == gi].sum() for gi in range(G)])
    share = 100.0 * fw       # sums to 100; additive position/quality penalties are separate
    return (pd.DataFrame({"block": BLOCK_NAMES,
                          "domain": [BLOCK_DOMAIN[g] for g in BLOCK_NAMES],
                          "learned_w": BASE_W[q],               # learned group weight
                          "distinct": DISTINCTIVENESS[q],       # peer-calibrated RMS PC energy
                          "attention_%": share})               # sum of final subgroup allocations
            .sort_values("attention_%", ascending=False).reset_index(drop=True))


# Megas whose child blocks are reported as SEPARATE skill families -- each child
# block gets its own row in the family table even though the blocks SHARE the
# mega's single fair-prior budget. Lets a conceptually split skill (Pull Up vs
# Overall midrange) be read independently while still drawing the COMBINED initial
# influence of ONE family. Budget/fairness is unchanged; only the grouping label is.
REPORT_SPLIT_MEGAS = {"Mid Range Shooting"}

def _report_family(block):
    return block if BLOCK_MEGA[block] in REPORT_SPLIT_MEGAS else BLOCK_MEGA[block]


def _query_skill_weight_table(q, blend, off_weight, gamma=None):
    """Exact final allocation at the most specific modeled skill level.

    Multi-subgroup blocks emit one row per subgroup. Blocks with only one subgroup
    stay as a single block row so the report does not repeat identical labels.
    `learned_w` is the stabilized pre-personalization subgroup allocation; `attention_%`
    is the exact final query-side weight after personalization and off/def balancing.
    """
    ow = auto_off_weight(q) if off_weight is None else off_weight
    fw_sub = _domain_normalized(_final_sub_weights(blend, gamma)[q], ow)
    learned_sub = BASE_W[q, SUB_BLOCK] * SUB_MULT[q]
    block_counts = np.bincount(SUB_BLOCK, minlength=G)
    rows = []
    for u, (block, subgroup, _) in enumerate(SUBS):
        is_subgroup = block_counts[SUB_BLOCK[u]] > 1
        rows.append({
            "skillset": subgroup if is_subgroup else block,
            "type": "subgroup" if is_subgroup else "block",
            "family": _report_family(block),
            "parent_block": block if is_subgroup else "",
            "domain": BLOCK_DOMAIN[block],
            "learned_w": learned_sub[u],
            "attention_%": 100.0 * fw_sub[u],
        })
    return (pd.DataFrame(rows)
            .sort_values("attention_%", ascending=False).reset_index(drop=True))


def _query_skill_family_table(q, blend, off_weight, gamma=None):
    """Comparable top-level ranking: aggregate every split block/subgroup by mega-block
    (REPORT_SPLIT_MEGAS megas are decomposed into one family per child block)."""
    skills = _query_skill_weight_table(q, blend, off_weight, gamma)
    return (skills.groupby(["family", "domain"], as_index=False, sort=False)
            .agg(**{"learned_w": ("learned_w", "sum"),
                    "attention_%": ("attention_%", "sum")})
            .sort_values("attention_%", ascending=False).reset_index(drop=True))


def show_query_weights(player_name, season=None, blend=PROFILE_BLEND,
                       off_weight=OVERALL_OFF_WEIGHT, top=None, gamma=None):
    q = _resolve(player_name, season)
    ow = auto_off_weight(q) if off_weight is None else off_weight
    families = _query_skill_family_table(q, blend, off_weight, gamma)
    tbl = _query_skill_weight_table(q, blend, off_weight, gamma)
    if top:
        families = families.head(top)
        tbl = tbl.head(top)
    r = df.loc[q]
    print(f"\n{r['player']} ({r['season']}) distance weights  [off_weight={ow:.2f}, blend={blend}]")
    print("\nSKILL FAMILIES (comparable aggregate ranking):")
    print(families.to_string(index=False, formatters={"learned_w": "{:.3f}".format,
          "attention_%": "{:.1f}".format}))
    print("\nINDIVIDUAL SKILLSETS (split families appear as smaller rows):")
    print(tbl.to_string(index=False, formatters={"learned_w": "{:.3f}".format,
          "attention_%": "{:.1f}".format}))
    return tbl


def find_similar_players(player_name, season=None, top_n=10, blend=PROFILE_BLEND,
                         off_weight=OVERALL_OFF_WEIGHT, exclude_same_player=True,
                         dedupe_players=False, show=True, show_weights=True, gamma=None):
    q = _resolve(player_name, season)
    off_d, def_d, overall_d = symmetric_distances(q, blend, off_weight, gamma)
    ow = auto_off_weight(q) if off_weight is None else off_weight  # query's own balance (display)
    res = df[["player", "season", "team", "position", "age"]].copy()
    res["off_dist"] = off_d
    res["def_dist"] = def_d
    res["overall_dist"] = overall_d   # combined distance, LOWER = more similar
    m = res.index != q
    if exclude_same_player:
        m &= df["player"] != df.at[q, "player"]
    res = res[m]

    def _view(col, ascending):
        d = res.sort_values(col, ascending=ascending)
        if dedupe_players:                       # one row per player (their best season)
            d = d.drop_duplicates("player", keep="first")
        return d.head(top_n).reset_index(drop=True)

    def _view_overall():
        d = res.sort_values("overall_dist", ascending=True)
        if dedupe_players:
            d = d.drop_duplicates("player", keep="first")
        return d.head(top_n).reset_index(drop=True)
    views = {"OFFENSE": _view("off_dist", True),
             "DEFENSE": _view("def_dist", True),
             "OVERALL": _view_overall()}
    if show:
        r = df.loc[q]
        print(f"\n================ {r['player']} ({r['season']}, {r['team']}, {r['position']}) "
              f"=================  [blend={blend}, off_weight={ow:.2f}]")
        if show_weights:
            _fam = _query_skill_family_table(q, blend, off_weight, gamma)
            _wt = _query_skill_weight_table(q, blend, off_weight, gamma)
            print("\nSKILL FAMILIES (comparable aggregate ranking):")
            print(_fam.to_string(index=False, formatters={"learned_w": "{:.3f}".format,
                  "attention_%": "{:.1f}".format}))
            print("\nINDIVIDUAL SKILLSETS (exact subgroup/block weights):")
            print(_wt.to_string(index=False, formatters={"learned_w": "{:.3f}".format,
                  "attention_%": "{:.1f}".format}))
        fmt = {"off_dist": "{:.3f}".format, "def_dist": "{:.3f}".format, "overall_dist": "{:.3f}".format}
        for name, v in views.items():
            print(f"\n--- TOP {name} MATCHES ---")
            print(v.reset_index(drop=True).to_string(index=False, formatters=fmt))
    return views


def why_similar(name_a, season_a, name_b, season_b, blend=PROFILE_BLEND):
    qa, qb = _resolve(name_a, season_a), _resolve(name_b, season_b)
    fw_sub = _final_sub_weights(blend)
    ow = _player_balance(None)
    fw_a = _domain_normalized(fw_sub[qa], ow[qa])
    fw_b = _domain_normalized(fw_sub[qb], ow[qb])
    pw = 0.5 * (fw_a + fw_b)                     # symmetric pair weights
    sq = _sub_sqdists(np.array([qa]), np.array([qb]), np.arange(NSUB))[0]
    contrib_sub = pw * sq
    rows = []
    for gi, g in enumerate(BLOCK_NAMES):
        rows.append((g, BLOCK_DOMAIN[g], float(contrib_sub[SUB_BLOCK == gi].sum())))
    tbl = pd.DataFrame(rows, columns=["block", "side", "contribution"]).sort_values("contribution")
    print(f"\n{df.at[qa,'player']} ({df.at[qa,'season']})  vs  {df.at[qb,'player']} ({df.at[qb,'season']})")
    print("most alike (top) -> most different (bottom):")
    print(tbl.to_string(index=False, formatters={"contribution": "{:.3f}".format}))
    return tbl


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Retrieval API for the site precompute (additions to the notebook)
# ----------------------------------------------------------------------------
# The model FITS on the full BBall Index population above. These helpers let the
# caller restrict which rows may be RETURNED as comps -- the site only wants
# player-seasons that exist in its galaxy -- without touching that fit.

SITE_CANDIDATE_MASK = IN_SITE_ROSTER


def resolve_index(player_name, season):
    """Row index for a player-season, or None when absent (no printing)."""
    match = (df["player"].str.lower() == str(player_name).lower()) & (
        df["season"].astype(str) == str(season)
    )
    hits = np.where(match.to_numpy())[0]
    return int(hits[0]) if len(hits) else None


def comps_for(q_pos, top_n=10, candidate_mask=None, exclude_same_player=True,
              blend=PROFILE_BLEND, off_weight=OVERALL_OFF_WEIGHT, gamma=None):
    """Top-N offense / defense / overall comps for one player-season.

    Similarity scores use the paper's transform, sim = 100 / (1 + (d/m)^2), with
    the median m taken per domain over the eligible candidate distances -- so a
    score is always relative to the pool the comp was actually drawn from.
    """
    off_d, def_d, overall_d = symmetric_distances(q_pos, blend, off_weight, gamma)

    eligible = np.ones(N, dtype=bool) if candidate_mask is None else np.asarray(candidate_mask, dtype=bool).copy()
    eligible[q_pos] = False
    if exclude_same_player:
        eligible &= (df["player"] != df.at[q_pos, "player"]).to_numpy()

    scores = {}
    for name, distances in (("off", off_d), ("def", def_d), ("overall", overall_d)):
        masked = np.where(eligible, distances, np.nan)
        scores[name] = percentile_similarity(masked)

    rows = np.where(eligible)[0]
    views = {}
    for name, distances in (("offense", off_d), ("defense", def_d), ("overall", overall_d)):
        ordered = rows[np.argsort(distances[rows], kind="stable")][:top_n]
        views[name] = [
            {
                "row": int(index),
                "player": str(df.at[index, "player"]),
                "season": str(df.at[index, "season"]),
                "team": str(df.at[index, "team"]),
                "position": str(df.at[index, "position"]),
                "age": int(pd.to_numeric(df.at[index, "age"], errors="coerce")),
                "off_distance": float(off_d[index]),
                "def_distance": float(def_d[index]),
                "overall_distance": float(overall_d[index]),
                "off_similarity": float(scores["off"][index]),
                "def_similarity": float(scores["def"][index]),
                "overall_similarity": float(scores["overall"][index]),
            }
            for index in ordered
        ]
    return views


def attention_for(q_pos, blend=PROFILE_BLEND, off_weight=OVERALL_OFF_WEIGHT, gamma=None):
    """The paper's two attention tables for one player-season.

    ``families`` is the comparable aggregate ranking (section 16's "Blocks Ranked
    By Attention"); ``skillsets`` is the exact per-subgroup allocation
    ("Individual Skillsets Ranked By Attention").
    """
    families = _query_skill_family_table(q_pos, blend, off_weight, gamma)
    skillsets = _query_skill_weight_table(q_pos, blend, off_weight, gamma)
    return {
        "off_weight": auto_off_weight(q_pos) if off_weight is None else float(off_weight),
        "families": families.to_dict("records"),
        "skillsets": skillsets.to_dict("records"),
    }


def block_decomposition(q_pos, target_rows, blend=PROFILE_BLEND,
                        off_weight=OVERALL_OFF_WEIGHT, gamma=None):
    """Per-block breakdown of the OVERALL distance between q_pos and each target.

    This is the notebook's ``why_similar`` decomposition, vectorized over many
    targets. Two quantities are returned per block:

      contribution -- the block's share of the pair's squared overall distance
                      (pair weight x squared subgroup distance, summed). This is
                      what ``why_similar`` ranks, and across all blocks it sums
                      to the squared distance before the position penalty.
      divergence   -- contribution divided by the block's pair weight, i.e. the
                      weight-averaged squared distance inside the block. Unlike
                      contribution this does NOT shrink just because the model
                      pays a block little attention, so it answers "how alike are
                      these two here" rather than "how much did this block move
                      the total".
    """
    fw_sub = _final_sub_weights(blend, gamma)
    ow = _player_balance(off_weight)
    fw_all = _domain_normalized(fw_sub, ow)

    targets = np.asarray(list(target_rows), dtype=int)
    if targets.size == 0:
        return {}
    sq = _sub_sqdists(targets, np.full(len(targets), q_pos), np.arange(NSUB))
    pair_weight = 0.5 * (fw_all[q_pos][None, :] + fw_all[targets])
    contribution = pair_weight * sq

    out = {}
    for position, target in enumerate(targets):
        blocks = {}
        for block_index, block in enumerate(BLOCK_NAMES):
            members = SUB_BLOCK == block_index
            block_contribution = float(contribution[position, members].sum())
            block_weight = float(pair_weight[position, members].sum())
            blocks[block] = {
                "domain": BLOCK_DOMAIN[block],
                "contribution": block_contribution,
                "divergence": block_contribution / block_weight if block_weight > 1e-12 else 0.0,
                "pair_weight": block_weight,
            }
        out[int(target)] = blocks
    return out
