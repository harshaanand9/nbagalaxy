from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.weighted_blocked_pca import (
    GUARDED_ARCHETYPE_BINS,
    GUARDED_POSITION_COMPONENTS,
    GUARDED_USAGE_BINS,
    HierarchyBlock,
    HierarchySubgroup,
    ParsedHierarchy,
    WeightedSubgroup,
    build_weighted_subgroups,
    engineer_notebook_features,
    evaluate_k_stability,
    fit_size_balanced_kmeans,
    fit_weighted_embedding,
    parse_hierarchy,
    restrict_weighted_subgroups_to_domain,
    select_k,
)


class HierarchyTests(unittest.TestCase):
    def test_parser_accepts_feature_bullets_with_inconsistent_indentation(self) -> None:
        text = """MEGA BLOCK: A
  SUBGROUP: One
    FEATURES:
      - valid
     - malformed
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hierarchy.txt"
            path.write_text(text, encoding="utf-8")
            parsed = parse_hierarchy(
                path, expected_blocks=1, expected_subgroups=1, expected_features=2
            )
        self.assertEqual(parsed.features, ("valid", "malformed"))
        self.assertEqual(parsed.ignored_feature_like_lines, ())

    def test_default_effective_weights_are_exact(self) -> None:
        subgroup_counts = {
            "3PT Shooting: Pull-Up + Self Created 3PT": 3,
            "3PT Shooting: C+S + Off-Ball 3PT": 3,
            "Mid Range Shooting": 6,
            "Rim": 3,
            "Paint - Non Rim": 3,
            "Drive Tendencies": 2,
            "Cut": 2,
            "Handoff": 2,
            "Isolations": 2,
            "Off-Ball Screens": 2,
            "P&R Ball Handler": 2,
            "P&R Roll Man": 2,
            "Spot Up": 2,
            "Transition": 2,
            "Post Ups": 2,
            "Touches": 1,
            "Ball Dominance": 1,
            "Dribbling Tendencies": 1,
            "Playmaking Volume": 1,
            "Playmaking Efficiency": 1,
            "Guarded By Data": 1,
            "Perimeter Defense + Matchups": 3,
            "Paint Defense + Matchups": 4,
            "Defensive Matchups": 2,
            "Defensive Versatility": 1,
            "Defensive Advanced / Impact Stats": 1,
            "Free Throw Generation": 1,
            "Offensive Rebounding": 2,
            "Defensive Rebounding": 1,
        }
        counter = 0
        blocks = []
        for block_name, count in subgroup_counts.items():
            subgroups = []
            subgroup_names = (
                [
                    "Pull Up Midrange: Frequency",
                    "Pull Up Midrange: Shot Quality",
                    "Pull Up Midrange: Efficiency",
                    "Overall Midrange: Frequency",
                    "Overall Midrange: Shot Quality",
                    "Overall Midrange: Efficiency",
                ]
                if block_name == "Mid Range Shooting"
                else [f"{block_name} subgroup {number}" for number in range(count)]
            )
            for subgroup_number, subgroup_name in enumerate(subgroup_names):
                counter += 1
                subgroups.append(
                    HierarchySubgroup(
                        name=subgroup_name,
                        features=(f"feature_{counter}",),
                    )
                )
            blocks.append(HierarchyBlock(name=block_name, subgroups=tuple(subgroups)))
        weighted = build_weighted_subgroups(ParsedHierarchy(tuple(blocks), ()))
        self.assertEqual(len(weighted), 59)
        self.assertAlmostEqual(
            sum(row.subgroup_weight for row in weighted if row.domain == "Offense"), 0.58
        )
        self.assertAlmostEqual(
            sum(row.subgroup_weight for row in weighted if row.domain == "Defense"), 0.42
        )
        play_type_names = (
            "Cut",
            "Handoff",
            "Isolations",
            "Off-Ball Screens",
            "P&R Ball Handler",
            "P&R Roll Man",
            "Spot Up",
            "Transition",
            "Post Ups",
        )
        for name in play_type_names:
            play_type = [row for row in weighted if row.effective_block == name]
            self.assertEqual(len(play_type), 2)
            self.assertTrue(
                all(
                    np.isclose(row.subgroup_weight, 0.58 * (0.175 / 9.0) / 2.0)
                    for row in play_type
                )
            )
        for name in ("Pull-Up Midrange", "Overall Midrange"):
            midrange = [row for row in weighted if row.effective_block == name]
            self.assertEqual(len(midrange), 3)
            self.assertTrue(
                all(np.isclose(row.subgroup_weight, 0.58 * 0.05 / 3.0) for row in midrange)
            )
        shooting = [row for row in weighted if row.effective_block == "3PT Shooting"]
        self.assertEqual(len(shooting), 6)
        self.assertTrue(
            all(np.isclose(row.subgroup_weight, 0.58 * 0.10 / 6.0) for row in shooting)
        )
        rim_pressure = [row for row in weighted if row.effective_block == "Rim Pressure"]
        self.assertEqual(len(rim_pressure), 8)
        self.assertTrue(
            all(np.isclose(row.subgroup_weight, 0.58 * 0.10 / 8.0) for row in rim_pressure)
        )
        passing = [row for row in weighted if row.effective_block == "Playmaking"]
        self.assertEqual(len(passing), 2)
        self.assertTrue(all(np.isclose(row.subgroup_weight, 0.087) for row in passing))
        revised_offense_totals = {
            "Ball Dominance + Touches": 0.029,
            "Dribbling Tendencies": 0.029,
            "Playmaking": 0.174,
            "Guarded By Data": 0.029,
            "Offensive Rebounding": 0.0145,
            "Free Throw Generation": 0.029,
        }
        for name, expected_weight in revised_offense_totals.items():
            rows = [row for row in weighted if row.effective_block == name]
            self.assertAlmostEqual(sum(row.subgroup_weight for row in rows), expected_weight)
        defensive_totals = {
            "Perimeter Defense + Matchups": 0.084,
            "Paint Defense + Matchups": 0.084,
            "Defensive Matchups": 0.042,
            "Defensive Versatility": 0.084,
            "Defensive Advanced / Impact Stats": 0.063,
            "Defensive Rebounding": 0.063,
        }
        for name, expected_weight in defensive_totals.items():
            rows = [row for row in weighted if row.effective_block == name]
            self.assertAlmostEqual(sum(row.subgroup_weight for row in rows), expected_weight)
        offense_only = restrict_weighted_subgroups_to_domain(weighted, "Offense")
        defense_only = restrict_weighted_subgroups_to_domain(weighted, "Defense")
        self.assertAlmostEqual(sum(row.subgroup_weight for row in offense_only), 1.0)
        self.assertAlmostEqual(sum(row.subgroup_weight for row in defense_only), 1.0)
        self.assertTrue(all(row.domain == "Offense" for row in offense_only))
        self.assertTrue(all(row.domain == "Defense" for row in defense_only))


class EmbeddingTests(unittest.TestCase):
    def test_guarding_compositions_are_raw_normalized_shares_without_minutes(self) -> None:
        rows = []
        source_columns = set(GUARDED_POSITION_COMPONENTS)
        source_columns.update(
            feature for members in GUARDED_ARCHETYPE_BINS.values() for feature in members
        )
        source_columns.update(
            feature for members in GUARDED_USAGE_BINS.values() for feature in members
        )
        for index in range(2):
            row = {
                "player": f"P{index}",
                "season": "2024-25",
                "position": "PG",
                "Stable Assists Per 75": 8.0,
                "Stable TOV Per 75": 2.0,
            }
            row.update({column: 1.0 for column in source_columns})
            rows.append(row)
        data, manifest = engineer_notebook_features(pd.DataFrame(rows))
        self.assertNotIn("Minutes", data.columns)
        self.assertEqual(manifest["guarding_preprocessing_version"], "raw-guarding-composition-v1")
        np.testing.assert_allclose(
            data[[f"__guard_position_composition_{i}" for i in range(5)]].sum(axis=1),
            100.0,
        )
        np.testing.assert_allclose(
            data[[f"__guard_archetype_composition_{i}" for i in range(6)]].iloc[0],
            100.0 * np.array([3, 1, 1, 2, 2, 3]) / 12.0,
        )
        np.testing.assert_allclose(
            data[[f"__guard_usage_composition_{i}" for i in range(3)]].iloc[0],
            np.repeat(100.0 / 3.0, 3),
        )

    def test_component_count_and_duplicate_features_cannot_inflate_weight(self) -> None:
        rng = np.random.default_rng(11)
        rows = 240
        data = pd.DataFrame(
            {
                "player": [f"P{index}" for index in range(rows)],
                "season": np.where(np.arange(rows) < rows // 2, "2023-24", "2024-25"),
                "one": rng.normal(size=rows),
                "many_a": rng.normal(size=rows),
                "many_b": rng.normal(size=rows),
                "many_c": rng.normal(size=rows),
                "duplicate": rng.normal(size=rows),
            }
        )
        data["duplicate_copy"] = data["duplicate"]
        weighted = [
            WeightedSubgroup("A", "Offense", "A", "one", ("one",), 0.2, 0.2),
            WeightedSubgroup(
                "B",
                "Offense",
                "B",
                "many",
                ("many_a", "many_b", "many_c"),
                0.3,
                0.3,
            ),
            WeightedSubgroup(
                "C",
                "Defense",
                "C",
                "duplicate",
                ("duplicate", "duplicate_copy"),
                0.5,
                0.5,
            ),
        ]
        result = fit_weighted_embedding(data, weighted)
        subgroup = result.contribution_audit[result.contribution_audit["level"] == "subgroup"]
        np.testing.assert_allclose(
            subgroup["empirical_variance_weight"], subgroup["target_variance_weight"], atol=1e-12
        )
        duplicate_manifest = result.subgroup_manifest[2]
        self.assertEqual(duplicate_manifest["retained_component_count"], 1)
        self.assertEqual(duplicate_manifest["discarded_zero_variance_components"], 1)


class StabilityTests(unittest.TestCase):
    def test_balanced_kmeans_respects_size_band(self) -> None:
        rng = np.random.default_rng(19)
        matrix = np.vstack(
            [
                rng.normal(loc=-3.0, scale=0.4, size=(70, 2)),
                rng.normal(loc=0.0, scale=0.4, size=(20, 2)),
                rng.normal(loc=3.0, scale=0.4, size=(10, 2)),
            ]
        )
        model = fit_size_balanced_kmeans(
            matrix, k=4, seed=42, n_init=5, max_iter=5, balance_tolerance=0.10
        )
        sizes = np.bincount(model.labels_, minlength=4)
        self.assertGreaterEqual(sizes.min(), model.min_cluster_size)
        self.assertLessEqual(sizes.max(), model.max_cluster_size)
        self.assertEqual(sizes.sum(), len(matrix))

    def test_stability_sweep_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        first = rng.normal(loc=-2.0, scale=0.2, size=(25, 3))
        second = rng.normal(loc=2.0, scale=0.2, size=(25, 3))
        matrix = np.vstack([first, second])
        seasons = np.array(["2023-24"] * 25 + ["2024-25"] * 25)
        kwargs = dict(
            k_values=(2, 3),
            resamples=3,
            fraction=0.8,
            seed=42,
            n_init=2,
            max_iter=100,
            jobs=1,
        )
        metrics_a, pairs_a, candidates_a = evaluate_k_stability(matrix, seasons, **kwargs)
        metrics_b, pairs_b, candidates_b = evaluate_k_stability(matrix, seasons, **kwargs)
        pd.testing.assert_frame_equal(metrics_a, metrics_b)
        pd.testing.assert_frame_equal(pairs_a, pairs_b)
        for k in candidates_a:
            np.testing.assert_array_equal(candidates_a[k]["labels"], candidates_b[k]["labels"])
        self.assertIn(select_k(metrics_a), (2, 3))


if __name__ == "__main__":
    unittest.main()
