"""Weighted blocked-PCA embedding and stability-selected KMeans clustering.

This module is intentionally independent of the website's locked clustering mode.
It implements the hierarchy and weighting contract used by
``scripts/run_weighted_blocked_pca_kmeans.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


EXPECTED_SOURCE_BLOCKS = 29
EXPECTED_SUBGROUPS = 59
EXPECTED_FEATURES = 239
DOMAIN_TOTAL_WEIGHT = {"Offense": 0.58, "Defense": 0.42}


@dataclass(frozen=True)
class HierarchySubgroup:
    name: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class HierarchyBlock:
    name: str
    subgroups: tuple[HierarchySubgroup, ...]


@dataclass(frozen=True)
class ParsedHierarchy:
    blocks: tuple[HierarchyBlock, ...]
    ignored_feature_like_lines: tuple[dict[str, Any], ...]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(
            feature
            for block in self.blocks
            for subgroup in block.subgroups
            for feature in subgroup.features
        )


@dataclass(frozen=True)
class EffectiveBlockSpec:
    name: str
    domain: str
    requested_domain_weight: float
    source_blocks: tuple[str, ...]
    selected_subgroups: tuple[str, ...] | None = None

    @property
    def final_weight(self) -> float:
        return DOMAIN_TOTAL_WEIGHT[self.domain] * self.requested_domain_weight


@dataclass(frozen=True)
class WeightedSubgroup:
    effective_block: str
    domain: str
    source_block: str
    subgroup: str
    features: tuple[str, ...]
    effective_block_weight: float
    subgroup_weight: float


@dataclass
class EmbeddingResult:
    matrix: np.ndarray
    weighted_subgroups: list[WeightedSubgroup]
    subgroup_slices: list[tuple[int, int]]
    subgroup_manifest: list[dict[str, Any]]
    standardization_manifest: list[dict[str, Any]]
    contribution_audit: pd.DataFrame
    model_bundle: dict[str, Any]


@dataclass
class BalancedKMeansResult:
    labels_: np.ndarray
    cluster_centers_: np.ndarray
    inertia_: float
    n_iter_: int
    base_inertia_: float
    min_cluster_size: int
    max_cluster_size: int
    balance_tolerance: float


DEFAULT_EFFECTIVE_BLOCKS: tuple[EffectiveBlockSpec, ...] = (
    EffectiveBlockSpec(
        "3PT Shooting",
        "Offense",
        0.10,
        (
            "3PT Shooting: Pull-Up + Self Created 3PT",
            "3PT Shooting: C+S + Off-Ball 3PT",
        ),
    ),
    EffectiveBlockSpec(
        "Pull-Up Midrange",
        "Offense",
        0.05,
        ("Mid Range Shooting",),
        (
            "Pull Up Midrange: Frequency",
            "Pull Up Midrange: Shot Quality",
            "Pull Up Midrange: Efficiency",
        ),
    ),
    EffectiveBlockSpec(
        "Overall Midrange",
        "Offense",
        0.05,
        ("Mid Range Shooting",),
        (
            "Overall Midrange: Frequency",
            "Overall Midrange: Shot Quality",
            "Overall Midrange: Efficiency",
        ),
    ),
    EffectiveBlockSpec(
        "Rim Pressure", "Offense", 0.10, ("Rim", "Paint - Non Rim", "Drive Tendencies")
    ),
    EffectiveBlockSpec("Cut", "Offense", 0.175 / 9.0, ("Cut",)),
    EffectiveBlockSpec("Handoff", "Offense", 0.175 / 9.0, ("Handoff",)),
    EffectiveBlockSpec("Isolations", "Offense", 0.175 / 9.0, ("Isolations",)),
    EffectiveBlockSpec(
        "Off-Ball Screens", "Offense", 0.175 / 9.0, ("Off-Ball Screens",)
    ),
    EffectiveBlockSpec(
        "P&R Ball Handler", "Offense", 0.175 / 9.0, ("P&R Ball Handler",)
    ),
    EffectiveBlockSpec("P&R Roll Man", "Offense", 0.175 / 9.0, ("P&R Roll Man",)),
    EffectiveBlockSpec("Spot Up", "Offense", 0.175 / 9.0, ("Spot Up",)),
    EffectiveBlockSpec("Transition", "Offense", 0.175 / 9.0, ("Transition",)),
    EffectiveBlockSpec("Post Ups", "Offense", 0.175 / 9.0, ("Post Ups",)),
    EffectiveBlockSpec(
        "Ball Dominance + Touches", "Offense", 0.05, ("Ball Dominance", "Touches")
    ),
    EffectiveBlockSpec(
        "Dribbling Tendencies", "Offense", 0.05, ("Dribbling Tendencies",)
    ),
    EffectiveBlockSpec(
        "Playmaking",
        "Offense",
        0.30,
        ("Playmaking Volume", "Playmaking Efficiency"),
    ),
    EffectiveBlockSpec("Guarded By Data", "Offense", 0.05, ("Guarded By Data",)),
    EffectiveBlockSpec(
        "Offensive Rebounding", "Offense", 0.025, ("Offensive Rebounding",)
    ),
    EffectiveBlockSpec(
        "Free Throw Generation", "Offense", 0.05, ("Free Throw Generation",)
    ),
    EffectiveBlockSpec(
        "Perimeter Defense + Matchups",
        "Defense",
        0.20,
        ("Perimeter Defense + Matchups",),
    ),
    EffectiveBlockSpec(
        "Paint Defense + Matchups", "Defense", 0.20, ("Paint Defense + Matchups",)
    ),
    EffectiveBlockSpec(
        "Defensive Matchups", "Defense", 0.10, ("Defensive Matchups",)
    ),
    EffectiveBlockSpec(
        "Defensive Versatility", "Defense", 0.20, ("Defensive Versatility",)
    ),
    EffectiveBlockSpec(
        "Defensive Advanced / Impact Stats",
        "Defense",
        0.15,
        ("Defensive Advanced / Impact Stats",),
    ),
    EffectiveBlockSpec(
        "Defensive Rebounding", "Defense", 0.15, ("Defensive Rebounding",)
    ),
)


GROUP_MAP = {"PG": "guard", "SG": "guard", "SF": "wing", "PF": "wing", "C": "big"}
GUARDED_POSITION_COMPONENTS = (
    "% of Time Guarding Point Guards",
    "% of Time Guarding Shooting Guards",
    "% of Time Guarding Small Forwards",
    "% of Time Guarding Power Forwards",
    "% of Time Guarding Centers",
)
GUARDED_ARCHETYPE_BINS: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    (
        (
            "Guarded Perimeter Off Ball",
            (
                "% of Time Guarding Movement Shooters",
                "% of Time Guarding Off-Screen Shooters",
                "% of Time Guarding Stationary Shooters",
            ),
        ),
        ("Guarded Primary Ball Handlers", ("% of Time Guarding Primary Ball Handlers",)),
        ("Guarded Shot Creators", ("% of Time Guarding Shot Creators",)),
        (
            "Guarded Secondary Wings",
            (
                "% of Time Guarding Secondary Ball Handlers",
                "% of Time Guarding Slashers",
            ),
        ),
        (
            "Guarded Interior Finishers",
            (
                "% of Time Guarding Athletic Finishers",
                "% of Time Guarding Roll & Cut Bigs",
            ),
        ),
        (
            "Guarded Skilled Bigs",
            (
                "% of Time Guarding Post Scorers",
                "% of Time Guarding Stretch Bigs",
                "% of Time Guarding Versatile Bigs",
            ),
        ),
    )
)
GUARDED_USAGE_BINS: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    (
        (
            "Guarded High Usage",
            (
                "% of Time Guarding Usage Tier 1 Players",
                "% of Time Guarding Usage Tier 2 Players",
            ),
        ),
        (
            "Guarded Medium Usage",
            (
                "% of Time Guarding Usage Tier 3 Players",
                "% of Time Guarding Usage Tier 4 Players",
            ),
        ),
        (
            "Guarded Low Usage",
            (
                "% of Time Guarding Usage Tier 5 Players",
                "% of Time Guarding Usage Tier 6 Players",
            ),
        ),
    )
)
def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hierarchy(
    path: str | Path,
    *,
    expected_blocks: int | None = EXPECTED_SOURCE_BLOCKS,
    expected_subgroups: int | None = EXPECTED_SUBGROUPS,
    expected_features: int | None = EXPECTED_FEATURES,
) -> ParsedHierarchy:
    """Parse the hierarchy while treating indentation as presentation only."""

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    block_rows: list[tuple[str, list[tuple[str, list[str]]]]] = []
    current_block: tuple[str, list[tuple[str, list[str]]]] | None = None
    current_subgroup: tuple[str, list[str]] | None = None
    ignored: list[dict[str, Any]] = []

    for line_number, line in enumerate(lines, start=1):
        if line.startswith("MEGA BLOCK: "):
            name = line.removeprefix("MEGA BLOCK: ").strip()
            current_block = (name, [])
            block_rows.append(current_block)
            current_subgroup = None
        elif line.startswith("  SUBGROUP: "):
            if current_block is None:
                raise ValueError(f"Subgroup before mega block at line {line_number}.")
            name = line.removeprefix("  SUBGROUP: ").strip()
            current_subgroup = (name, [])
            current_block[1].append(current_subgroup)
        elif line.lstrip().startswith("- "):
            if current_subgroup is None:
                raise ValueError(f"Feature before subgroup at line {line_number}.")
            current_subgroup[1].append(line.lstrip().removeprefix("- ").strip())

    blocks = tuple(
        HierarchyBlock(
            name=block_name,
            subgroups=tuple(
                HierarchySubgroup(name=subgroup_name, features=tuple(features))
                for subgroup_name, features in subgroups
            ),
        )
        for block_name, subgroups in block_rows
    )
    parsed = ParsedHierarchy(blocks=blocks, ignored_feature_like_lines=tuple(ignored))

    block_names = [block.name for block in blocks]
    subgroup_names = [
        (block.name, subgroup.name) for block in blocks for subgroup in block.subgroups
    ]
    features = list(parsed.features)
    if len(block_names) != len(set(block_names)):
        raise ValueError("Hierarchy contains duplicate mega-block names.")
    if len(subgroup_names) != len(set(subgroup_names)):
        raise ValueError("Hierarchy contains duplicate subgroup names inside a mega block.")
    if len(features) != len(set(features)):
        duplicates = sorted({feature for feature in features if features.count(feature) > 1})
        raise ValueError(f"Hierarchy contains duplicate features: {duplicates}")
    empty = [
        f"{block.name} / {subgroup.name}"
        for block in blocks
        for subgroup in block.subgroups
        if not subgroup.features
    ]
    if empty:
        raise ValueError(f"Hierarchy contains empty subgroups: {empty}")

    actual = (len(blocks), len(subgroup_names), len(features))
    expected = (expected_blocks, expected_subgroups, expected_features)
    labels = ("mega blocks", "subgroups", "features")
    mismatches = [
        f"{label}: expected {want}, found {got}"
        for label, want, got in zip(labels, expected, actual)
        if want is not None and want != got
    ]
    if mismatches:
        raise ValueError("Hierarchy contract mismatch: " + "; ".join(mismatches))
    return parsed


def build_weighted_subgroups(
    hierarchy: ParsedHierarchy,
    specs: Sequence[EffectiveBlockSpec] = DEFAULT_EFFECTIVE_BLOCKS,
) -> list[WeightedSubgroup]:
    block_by_name = {block.name: block for block in hierarchy.blocks}
    all_subgroup_keys = {
        (block.name, subgroup.name)
        for block in hierarchy.blocks
        for subgroup in block.subgroups
    }
    selected_by_spec: list[list[tuple[str, HierarchySubgroup]]] = []
    claimed_keys: list[tuple[str, str]] = []
    for spec in specs:
        unknown_sources = sorted(set(spec.source_blocks) - set(block_by_name))
        if unknown_sources:
            raise ValueError(f"Unknown source mega blocks for {spec.name}: {unknown_sources}")
        selected: list[tuple[str, HierarchySubgroup]] = []
        for source in spec.source_blocks:
            for subgroup in block_by_name[source].subgroups:
                if spec.selected_subgroups is None or subgroup.name in spec.selected_subgroups:
                    selected.append((source, subgroup))
        if spec.selected_subgroups is not None:
            matched_names = {subgroup.name for _, subgroup in selected}
            missing_names = sorted(set(spec.selected_subgroups) - matched_names)
            if missing_names:
                raise ValueError(f"Unknown selected subgroups for {spec.name}: {missing_names}")
        if not selected:
            raise ValueError(f"Effective block {spec.name} contains no subgroups.")
        selected_by_spec.append(selected)
        claimed_keys.extend((source, subgroup.name) for source, subgroup in selected)
    duplicated = sorted({key for key in claimed_keys if claimed_keys.count(key) > 1})
    missing = sorted(all_subgroup_keys - set(claimed_keys))
    if duplicated or missing:
        raise ValueError(
            f"Effective-block subgroup mapping mismatch; duplicated={duplicated}, unmapped={missing}"
        )

    for domain, domain_total in DOMAIN_TOTAL_WEIGHT.items():
        requested_total = sum(
            spec.requested_domain_weight for spec in specs if spec.domain == domain
        )
        if not np.isclose(requested_total, 1.0, atol=1e-12):
            raise ValueError(f"{domain} requested weights sum to {requested_total}, not 1.0.")
        final_total = sum(spec.final_weight for spec in specs if spec.domain == domain)
        if not np.isclose(final_total, domain_total, atol=1e-12):
            raise ValueError(f"{domain} final weights sum to {final_total}, not {domain_total}.")
    if not np.isclose(sum(spec.final_weight for spec in specs), 1.0, atol=1e-12):
        raise ValueError("Final effective-block weights do not sum to 1.0.")

    weighted: list[WeightedSubgroup] = []
    for spec, source_subgroups in zip(specs, selected_by_spec):
        subgroup_weight = spec.final_weight / len(source_subgroups)
        for source, subgroup in source_subgroups:
            weighted.append(
                WeightedSubgroup(
                    effective_block=spec.name,
                    domain=spec.domain,
                    source_block=source,
                    subgroup=subgroup.name,
                    features=subgroup.features,
                    effective_block_weight=spec.final_weight,
                    subgroup_weight=subgroup_weight,
                )
            )
    return weighted


def restrict_weighted_subgroups_to_domain(
    weighted_subgroups: Sequence[WeightedSubgroup], domain: str
) -> list[WeightedSubgroup]:
    """Select one domain and renormalize its subgroup weights to sum to one."""

    selected = [row for row in weighted_subgroups if row.domain == domain]
    if not selected:
        raise ValueError(f"No weighted subgroups found for domain {domain!r}.")
    domain_total = sum(row.subgroup_weight for row in selected)
    if domain_total <= 0.0:
        raise ValueError(f"Domain {domain!r} has nonpositive total weight.")
    return [
        replace(
            row,
            effective_block_weight=row.effective_block_weight / domain_total,
            subgroup_weight=row.subgroup_weight / domain_total,
        )
        for row in selected
    ]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"Missing {context} columns: {missing}")


def _numeric_frame(df: pd.DataFrame, columns: Sequence[str], context: str) -> pd.DataFrame:
    numeric = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(numeric.to_numpy(dtype=float))
    if bad.any():
        row_indexes, column_indexes = np.where(bad)
        examples = [
            {
                "row": int(row),
                "player": str(df.iloc[row].get("player", "")),
                "season": str(df.iloc[row].get("season", "")),
                "feature": columns[column],
                "value": df.iloc[row][columns[column]],
            }
            for row, column in zip(row_indexes[:20], column_indexes[:20])
        ]
        raise ValueError(f"Non-numeric or nonfinite values in {context}: {examples}")
    return numeric


def _stable_player_groups(df: pd.DataFrame) -> np.ndarray:
    _require_columns(df, ("player", "season", "position"), "player identity")
    row_groups = df["position"].map(GROUP_MAP)
    if row_groups.isna().any():
        bad = sorted(df.loc[row_groups.isna(), "position"].astype(str).unique().tolist())
        raise ValueError(f"Unsupported positions: {bad}")
    start_year = pd.to_numeric(df["season"].astype(str).str[:4], errors="coerce")
    if start_year.isna().any():
        raise ValueError("Every season must begin with a four-digit year.")

    stable_by_player: dict[str, str] = {}
    grouping = pd.DataFrame(
        {
            "player": df["player"].astype(str),
            "start_year": start_year,
            "row_group": row_groups,
        }
    )
    for player, rows in grouping.groupby("player", sort=False):
        counts = rows["row_group"].value_counts()
        top = counts[counts == counts.max()].index.tolist()
        stable_by_player[player] = (
            top[0]
            if len(top) == 1
            else rows.sort_values("start_year")["row_group"].iloc[-1]
        )
    return grouping["player"].map(stable_by_player).to_numpy()


def engineer_notebook_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build hierarchy-required ratios and unshrunk raw guarding compositions."""

    output = df.copy().reset_index(drop=True)
    player_group = _stable_player_groups(output)

    ratio_columns = ("Stable Assists Per 75", "Stable TOV Per 75")
    ratio = _numeric_frame(output, ratio_columns, "stable AST/TOV ratio")
    bad_denominator = ratio[ratio_columns[1]] <= 0.0
    if bad_denominator.any():
        examples = output.loc[
            bad_denominator, ["player", "season", ratio_columns[1]]
        ].head(20)
        raise ValueError(
            "Stable TOV Per 75 must be positive: " + str(examples.to_dict("records"))
        )
    output["stable_ast_tov_ratio"] = ratio[ratio_columns[0]] / ratio[ratio_columns[1]]

    for new_name, members in (*GUARDED_ARCHETYPE_BINS.items(), *GUARDED_USAGE_BINS.items()):
        _require_columns(output, members, f"{new_name} guarding aggregation")
        output[new_name] = _numeric_frame(output, members, new_name).sum(axis=1)

    raw_compositions: dict[str, np.ndarray] = {}
    composition_audit: list[dict[str, Any]] = []
    composition_specs: OrderedDict[str, tuple[str, ...]] = OrderedDict(
        (
            ("position_composition", GUARDED_POSITION_COMPONENTS),
            ("archetype_composition", tuple(GUARDED_ARCHETYPE_BINS.keys())),
            ("usage_composition", tuple(GUARDED_USAGE_BINS.keys())),
        )
    )
    for family, columns in composition_specs.items():
        raw = _numeric_frame(output, columns, family).to_numpy(dtype=float)
        negative = np.any(raw < 0.0, axis=1)
        sums = raw.sum(axis=1)
        invalid = negative | ~np.isfinite(sums) | (sums <= 0.0)
        if invalid.any():
            examples = output.loc[invalid, ["player", "season"]].head(20).to_dict("records")
            raise ValueError(f"Invalid raw guarding composition for {family}: {examples}")
        composition = raw / sums[:, None]
        raw_compositions[family] = composition
        for index in range(composition.shape[1]):
            output[f"__guard_{family}_{index}"] = 100.0 * composition[:, index]
        for season, indexes in output.groupby("season", sort=False).groups.items():
            indexes_array = np.asarray(list(indexes), dtype=int)
            season_sums = sums[indexes_array]
            composition_audit.append(
                {
                    "family": family,
                    "season": str(season),
                    "rows": int(len(indexes_array)),
                    "invalid_rows": int(invalid[indexes_array].sum()),
                    "minimum_raw_sum": float(season_sums.min()),
                    "maximum_raw_sum": float(season_sums.max()),
                    "median_raw_sum": float(np.median(season_sums)),
                    "mean_absolute_deviation_from_100": float(
                        np.mean(np.abs(season_sums - 100.0))
                    ),
                    "normalized_sum_max_error": float(
                        np.max(np.abs(composition[indexes_array].sum(axis=1) - 1.0))
                    ),
                }
            )

    position_shares = raw_compositions["position_composition"]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(position_shares > 0.0, position_shares * np.log(position_shares), 0.0),
            axis=1,
        )
    output["Guarded Position Spread"] = (
        pd.Series(entropy)
        .groupby([output["season"].astype(str), pd.Series(player_group)])
        .rank(pct=True)
        .to_numpy()
    )

    generated = ["stable_ast_tov_ratio", "Guarded Position Spread"] + [
        f"__guard_{family}_{index}"
        for family, values in raw_compositions.items()
        for index in range(values.shape[1])
    ]
    if len(generated) != 16:
        raise AssertionError(f"Expected 16 engineered features, found {len(generated)}.")
    if not np.isfinite(output[generated].to_numpy(dtype=float)).all():
        raise ValueError("Engineered features contain nonfinite values.")
    return output, {
        "generated_features": generated,
        "guarding_preprocessing_version": "raw-guarding-composition-v1",
        "guarding_compositions": "raw shares normalized row-wise; no Minutes or prior shrinkage",
        "position_group": player_group.tolist(),
        "composition_audit": composition_audit,
    }


def standardize_within_season(
    df: pd.DataFrame, features: Sequence[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    _require_columns(df, ("season", *features), "standardization")
    standardized: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    seasons = df["season"].astype(str)
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        bad = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
        if bad.any():
            examples = df.loc[bad, ["player", "season", feature]].head(20).to_dict("records")
            raise ValueError(f"Non-numeric/nonfinite values in {feature}: {examples}")
        means = values.groupby(seasons).transform("mean")
        standard_deviations = values.groupby(seasons).transform("std")
        constant = standard_deviations.isna() | (standard_deviations == 0.0)
        zscores = ((values - means) / standard_deviations.mask(constant)).fillna(0.0)
        if not np.isfinite(zscores.to_numpy(dtype=float)).all():
            raise ValueError(f"Within-season standardization failed for {feature}.")
        standardized.append(zscores.to_numpy(dtype=float))
        for season in pd.unique(seasons):
            season_mask = seasons == season
            manifest.append(
                {
                    "feature": feature,
                    "season": str(season),
                    "mean": float(means.loc[season_mask].iloc[0]),
                    "std": (
                        None
                        if constant.loc[season_mask].iloc[0]
                        else float(standard_deviations.loc[season_mask].iloc[0])
                    ),
                    "constant": bool(constant.loc[season_mask].iloc[0]),
                }
            )
    return np.column_stack(standardized), manifest


def fit_weighted_embedding(
    df: pd.DataFrame,
    weighted_subgroups: Sequence[WeightedSubgroup],
    *,
    random_state: int = 42,
) -> EmbeddingResult:
    parts: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    subgroup_manifest: list[dict[str, Any]] = []
    standardization_manifest: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    cursor = 0

    for subgroup in weighted_subgroups:
        zscores, standardization_rows = standardize_within_season(df, subgroup.features)
        standardization_manifest.extend(standardization_rows)
        pca = PCA(n_components=min(zscores.shape), svd_solver="full", random_state=random_state)
        scores = pca.fit_transform(zscores)
        eigenvalues = pca.explained_variance_
        maximum = float(eigenvalues.max(initial=0.0))
        tolerance = maximum * max(zscores.shape) * np.finfo(float).eps
        retained = np.flatnonzero(eigenvalues > tolerance)
        if retained.size == 0:
            raise ValueError(
                f"Subgroup {subgroup.source_block} / {subgroup.subgroup} has no nonzero variance."
            )
        whitened = scores[:, retained] / np.sqrt(eigenvalues[retained])
        component_scale = np.sqrt(subgroup.subgroup_weight / retained.size)
        weighted_part = whitened * component_scale
        if not np.isfinite(weighted_part).all():
            raise ValueError(f"Nonfinite PCA embedding for subgroup {subgroup.subgroup}.")
        parts.append(weighted_part)
        subgroup_slice = (cursor, cursor + retained.size)
        slices.append(subgroup_slice)
        cursor = subgroup_slice[1]
        empirical_variance = float(np.var(weighted_part, axis=0, ddof=1).sum())
        subgroup_manifest.append(
            {
                **asdict(subgroup),
                "input_feature_count": len(subgroup.features),
                "pca_component_count": int(len(eigenvalues)),
                "retained_component_count": int(retained.size),
                "discarded_zero_variance_components": int(len(eigenvalues) - retained.size),
                "eigenvalue_tolerance": tolerance,
                "component_scale": float(component_scale),
                "slice_start": subgroup_slice[0],
                "slice_stop": subgroup_slice[1],
                "empirical_variance_sum": empirical_variance,
            }
        )
        transforms.append(
            {
                "weighted_subgroup": subgroup,
                "pca": pca,
                "retained_component_indexes": retained,
                "component_scale": component_scale,
                "slice": subgroup_slice,
            }
        )

    matrix = np.hstack(parts)
    if not np.isfinite(matrix).all():
        raise ValueError("Final embedding contains nonfinite values.")
    audit = build_contribution_audit(matrix, weighted_subgroups, slices)
    return EmbeddingResult(
        matrix=matrix,
        weighted_subgroups=list(weighted_subgroups),
        subgroup_slices=slices,
        subgroup_manifest=subgroup_manifest,
        standardization_manifest=standardization_manifest,
        contribution_audit=audit,
        model_bundle={
            "sklearn_version": sklearn.__version__,
            "weighted_subgroups": list(weighted_subgroups),
            "transforms": transforms,
            "standardization_manifest": standardization_manifest,
        },
    )


def build_contribution_audit(
    matrix: np.ndarray,
    weighted_subgroups: Sequence[WeightedSubgroup],
    slices: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subgroup, (start, stop) in zip(weighted_subgroups, slices):
        part = matrix[:, start:stop]
        variance_sum = float(np.var(part, axis=0, ddof=1).sum())
        target = subgroup.subgroup_weight
        rows.append(
            {
                "level": "subgroup",
                "domain": subgroup.domain,
                "effective_block": subgroup.effective_block,
                "source_block": subgroup.source_block,
                "subgroup": subgroup.subgroup,
                "target_variance_weight": target,
                "empirical_variance_weight": variance_sum,
                "target_mean_pairwise_squared_distance": 2.0 * target,
                "empirical_mean_pairwise_squared_distance": 2.0 * variance_sum,
                "absolute_error": abs(variance_sum - target),
            }
        )
    subgroup_frame = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    for level, columns in (
        ("effective_block", ["domain", "effective_block"]),
        ("domain", ["domain"]),
        ("total", []),
    ):
        grouped: Iterable[tuple[Any, pd.DataFrame]]
        if columns:
            grouped = subgroup_frame.groupby(columns, sort=False, dropna=False)
        else:
            grouped = [((), subgroup_frame)]
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "level": level,
                "domain": "",
                "effective_block": "",
                "source_block": "",
                "subgroup": "",
            }
            row.update(dict(zip(columns, keys)))
            target = float(group["target_variance_weight"].sum())
            empirical = float(group["empirical_variance_weight"].sum())
            row.update(
                {
                    "target_variance_weight": target,
                    "empirical_variance_weight": empirical,
                    "target_mean_pairwise_squared_distance": 2.0 * target,
                    "empirical_mean_pairwise_squared_distance": 2.0 * empirical,
                    "absolute_error": abs(empirical - target),
                }
            )
            aggregate_rows.append(row)
    result = pd.concat([subgroup_frame, pd.DataFrame(aggregate_rows)], ignore_index=True)
    if float(result["absolute_error"].max()) > 1e-10:
        raise AssertionError("Variance contribution audit exceeded 1e-10 tolerance.")
    return result


def stratified_subsample_indexes(
    seasons: Sequence[Any], fraction: float, rng: np.random.Generator
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("subsample fraction must be in (0, 1].")
    season_values = np.asarray(seasons).astype(str)
    selected: list[np.ndarray] = []
    for season in pd.unique(season_values):
        indexes = np.flatnonzero(season_values == season)
        count = max(1, int(np.floor(fraction * len(indexes))))
        selected.append(np.sort(rng.choice(indexes, size=count, replace=False)))
    return np.sort(np.concatenate(selected))


def _fit_subsample(
    matrix: np.ndarray,
    seasons: np.ndarray,
    *,
    k: int,
    fraction: float,
    seed: int,
    n_init: int,
    max_iter: int,
    tolerance: float,
    balance_tolerance: float | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indexes = stratified_subsample_indexes(seasons, fraction, rng)
    if len(indexes) < k:
        raise ValueError(f"Subsample contains {len(indexes)} rows, fewer than k={k}.")
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tolerance,
        random_state=seed,
        algorithm="lloyd",
    ).fit(matrix[indexes])
    if balance_tolerance is None:
        labels = model.predict(matrix).astype(np.int16)
    else:
        target_size = len(matrix) / k
        minimum_size = int(np.floor(target_size * (1.0 - balance_tolerance)))
        maximum_size = int(np.ceil(target_size * (1.0 + balance_tolerance)))
        labels = _bounded_cluster_assignment(
            matrix,
            model.cluster_centers_,
            minimum_size=minimum_size,
            maximum_size=maximum_size,
        ).astype(np.int16)
    return {
        "seed": seed,
        "subsample_size": int(len(indexes)),
        "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_),
        "labels": labels,
    }


def evaluate_k_stability(
    matrix: np.ndarray,
    seasons: Sequence[Any],
    *,
    k_values: Sequence[int] = tuple(range(20, 27)),
    resamples: int = 100,
    fraction: float = 0.8,
    seed: int = 42,
    n_init: int = 50,
    max_iter: int = 1000,
    tolerance: float = 1e-4,
    jobs: int = 1,
    balance_tolerance: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]]]:
    if resamples < 2:
        raise ValueError("At least two resamples are required for AMI/ARI stability.")
    from joblib import Parallel, delayed

    matrix = np.asarray(matrix, dtype=float)
    seasons_array = np.asarray(seasons)
    metric_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    candidates: dict[int, dict[str, Any]] = {}

    for k in sorted(set(int(value) for value in k_values)):
        seed_sequence = np.random.SeedSequence([seed, k])
        run_seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in seed_sequence.spawn(resamples)
        ]
        fits = Parallel(n_jobs=jobs)(
            delayed(_fit_subsample)(
                matrix,
                seasons_array,
                k=k,
                fraction=fraction,
                seed=run_seed,
                n_init=n_init,
                max_iter=max_iter,
                tolerance=tolerance,
                balance_tolerance=balance_tolerance,
            )
            for run_seed in run_seeds
        )
        label_matrix = np.stack([fit["labels"] for fit in fits])
        ami_values: list[float] = []
        ari_values: list[float] = []
        for first, second in combinations(range(resamples), 2):
            ami = float(adjusted_mutual_info_score(label_matrix[first], label_matrix[second]))
            ari = float(adjusted_rand_score(label_matrix[first], label_matrix[second]))
            ami_values.append(ami)
            ari_values.append(ari)
            pairwise_rows.append(
                {
                    "k": k,
                    "run_a": first,
                    "run_b": second,
                    "seed_a": fits[first]["seed"],
                    "seed_b": fits[second]["seed"],
                    "ami": ami,
                    "ari": ari,
                }
            )
        mean_ami = float(np.mean(ami_values))
        mean_ari = float(np.mean(ari_values))
        std_ami = float(np.std(ami_values, ddof=1)) if len(ami_values) > 1 else 0.0
        std_ari = float(np.std(ari_values, ddof=1)) if len(ari_values) > 1 else 0.0
        metric_rows.append(
            {
                "k": k,
                "resamples": resamples,
                "pairwise_comparisons": len(ami_values),
                "mean_ami": mean_ami,
                "std_ami": std_ami,
                "p05_ami": float(np.quantile(ami_values, 0.05)),
                "mean_ari": mean_ari,
                "std_ari": std_ari,
                "p05_ari": float(np.quantile(ari_values, 0.05)),
                "selection_score": 0.5 * (mean_ami + mean_ari),
                "mean_inertia": float(np.mean([fit["inertia"] for fit in fits])),
                "mean_iterations": float(np.mean([fit["iterations"] for fit in fits])),
            }
        )
        candidates[k] = {
            "labels": label_matrix,
            "seeds": np.asarray(run_seeds, dtype=np.uint32),
            "subsample_sizes": np.asarray(
                [fit["subsample_size"] for fit in fits], dtype=np.int32
            ),
            "inertias": np.asarray([fit["inertia"] for fit in fits], dtype=float),
            "iterations": np.asarray([fit["iterations"] for fit in fits], dtype=np.int32),
        }
    metrics = pd.DataFrame(metric_rows).sort_values("k").reset_index(drop=True)
    pairwise = pd.DataFrame(pairwise_rows)
    return metrics, pairwise, candidates


def select_k(metrics: pd.DataFrame) -> int:
    required = {"k", "selection_score"}
    if not required.issubset(metrics.columns) or metrics.empty:
        raise ValueError("Metrics must contain nonempty k and selection_score columns.")
    ranked = metrics.sort_values(
        ["selection_score", "k"], ascending=[False, True], kind="mergesort"
    )
    return int(ranked.iloc[0]["k"])


def fit_final_kmeans(
    matrix: np.ndarray,
    *,
    k: int,
    seed: int = 42,
    n_init: int = 200,
    max_iter: int = 1000,
    tolerance: float = 1e-4,
) -> KMeans:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tolerance,
        random_state=seed,
        algorithm="lloyd",
    ).fit(matrix)
    if len(np.unique(model.labels_)) != k:
        raise AssertionError(f"Final KMeans produced fewer than {k} nonempty clusters.")
    return model


def _bounded_cluster_assignment(
    matrix: np.ndarray,
    centers: np.ndarray,
    *,
    minimum_size: int,
    maximum_size: int,
) -> np.ndarray:
    """Solve the minimum-distance assignment under per-cluster size bounds."""

    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, vstack

    row_count, cluster_count = len(matrix), len(centers)
    if minimum_size * cluster_count > row_count or maximum_size * cluster_count < row_count:
        raise ValueError(
            f"Infeasible cluster bounds {minimum_size}..{maximum_size} for "
            f"{row_count} rows and k={cluster_count}."
        )
    squared_distances = (
        np.square(matrix).sum(axis=1, keepdims=True)
        + np.square(centers).sum(axis=1)[None, :]
        - 2.0 * matrix @ centers.T
    )
    squared_distances = np.maximum(squared_distances, 0.0)
    variable_count = row_count * cluster_count
    variable_indexes = np.arange(variable_count)
    point_constraints = csr_matrix(
        (
            np.ones(variable_count),
            (np.repeat(np.arange(row_count), cluster_count), variable_indexes),
        ),
        shape=(row_count, variable_count),
    )
    cluster_constraints = csr_matrix(
        (
            np.ones(variable_count),
            (np.tile(np.arange(cluster_count), row_count), variable_indexes),
        ),
        shape=(cluster_count, variable_count),
    )
    upper_constraints = vstack([cluster_constraints, -cluster_constraints], format="csr")
    upper_bounds = np.concatenate(
        [
            np.full(cluster_count, maximum_size, dtype=float),
            np.full(cluster_count, -minimum_size, dtype=float),
        ]
    )
    result = linprog(
        squared_distances.ravel(),
        A_ub=upper_constraints,
        b_ub=upper_bounds,
        A_eq=point_constraints,
        b_eq=np.ones(row_count),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Balanced assignment failed: {result.message}")
    assignment = result.x.reshape(row_count, cluster_count)
    rounded = np.rint(assignment)
    fractional_error = float(np.max(np.abs(assignment - rounded)))
    if fractional_error > 1e-6:
        raise RuntimeError(
            f"Balanced assignment was unexpectedly fractional (max error={fractional_error})."
        )
    labels = rounded.argmax(axis=1).astype(int)
    sizes = np.bincount(labels, minlength=cluster_count)
    if sizes.min() < minimum_size or sizes.max() > maximum_size:
        raise AssertionError(f"Balanced assignment sizes violate bounds: {sizes.tolist()}")
    return labels


def fit_size_balanced_kmeans(
    matrix: np.ndarray,
    *,
    k: int,
    seed: int = 42,
    n_init: int = 200,
    max_iter: int = 20,
    tolerance: float = 1e-4,
    balance_tolerance: float = 0.10,
) -> BalancedKMeansResult:
    """Fit k-means++ then alternate bounded assignments and centroid updates."""

    if not 0.0 <= balance_tolerance < 1.0:
        raise ValueError("balance_tolerance must be in [0, 1).")
    matrix = np.asarray(matrix, dtype=float)
    target_size = len(matrix) / k
    minimum_size = int(np.floor(target_size * (1.0 - balance_tolerance)))
    maximum_size = int(np.ceil(target_size * (1.0 + balance_tolerance)))
    base_model = fit_final_kmeans(
        matrix,
        k=k,
        seed=seed,
        n_init=n_init,
        max_iter=1000,
        tolerance=tolerance,
    )
    centers = base_model.cluster_centers_.copy()
    previous_labels: np.ndarray | None = None
    labels: np.ndarray | None = None
    iterations = 0
    for iterations in range(1, max_iter + 1):
        labels = _bounded_cluster_assignment(
            matrix,
            centers,
            minimum_size=minimum_size,
            maximum_size=maximum_size,
        )
        new_centers = np.vstack([matrix[labels == cluster].mean(axis=0) for cluster in range(k)])
        center_shift = float(np.linalg.norm(new_centers - centers))
        stable_labels = previous_labels is not None and np.array_equal(labels, previous_labels)
        centers = new_centers
        if stable_labels or center_shift <= tolerance:
            break
        previous_labels = labels.copy()
    if labels is None:
        raise AssertionError("Balanced k-means did not produce assignments.")
    inertia = float(np.square(matrix - centers[labels]).sum())
    return BalancedKMeansResult(
        labels_=labels,
        cluster_centers_=centers,
        inertia_=inertia,
        n_iter_=iterations,
        base_inertia_=float(base_model.inertia_),
        min_cluster_size=minimum_size,
        max_cluster_size=maximum_size,
        balance_tolerance=balance_tolerance,
    )


def save_model_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
