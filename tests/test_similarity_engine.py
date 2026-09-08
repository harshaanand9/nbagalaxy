"""Guardrails for the v4 similarity engine (the sim.ipynb port).

These lock the properties the paper claims, so a future edit to
``backend/similarity_engine.py`` that quietly changes the model fails here
rather than silently reshuffling every comparison on the site.

Run:
    python3 -m unittest tests/test_similarity_engine.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DATASET = BACKEND_DIR / "data" / "similarity_model_dataset.csv"


@unittest.skipUnless(DATASET.exists(), f"{DATASET.name} not built")
class SimilarityEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import similarity_engine

        cls.engine = similarity_engine

    def test_hierarchy_shape_matches_the_shipped_model(self) -> None:
        self.assertEqual(self.engine.G, 34, "block count changed")
        self.assertEqual(self.engine.NSUB, 63, "subgroup count changed")
        self.assertEqual(len(self.engine.BLOCK_NAMES), self.engine.G)
        self.assertEqual(len(self.engine.SUB_NAMES), self.engine.NSUB)

    def test_offense_defense_identity_matches_the_paper(self) -> None:
        """Section 16/17 of the paper publishes these two identity splits."""
        for player, season, expected_off in [
            ("Keyonte George", "2025-26", 0.83),
            ("Walker Kessler", "2024-25", 0.31),
        ]:
            row = self.engine.resolve_index(player, season)
            self.assertIsNotNone(row, f"{player} {season} missing from the dataset")
            self.assertAlmostEqual(
                self.engine.auto_off_weight(row), expected_off, places=2,
                msg=f"{player} {season} offensive identity moved",
            )

    def test_identity_never_erases_either_side_of_the_ball(self) -> None:
        weights = self.engine._player_balance(None)
        floor = self.engine.MIN_DOMAIN_WEIGHT
        self.assertGreaterEqual(weights.min(), floor - 1e-9)
        self.assertLessEqual(weights.max(), 1.0 - floor + 1e-9)

    def test_distance_is_symmetric(self) -> None:
        """Pair-averaged weights are what make d(a, b) == d(b, a)."""
        a = self.engine.resolve_index("Nikola Jokic", "2022-23")
        b = self.engine.resolve_index("Domantas Sabonis", "2022-23")
        from_a = self.engine.symmetric_distances(a)
        from_b = self.engine.symmetric_distances(b)
        for name, left, right in zip(("off", "def", "overall"), from_a, from_b):
            self.assertAlmostEqual(left[b], right[a], places=12, msg=f"{name} asymmetric")
        self.assertAlmostEqual(float(from_a[2][a]), 0.0, places=12)

    def test_attention_is_a_percentage_split(self) -> None:
        row = self.engine.resolve_index("Walker Kessler", "2024-25")
        table = self.engine._query_skill_weight_table(row, self.engine.PROFILE_BLEND, None, None)
        self.assertAlmostEqual(float(table["attention_%"].sum()), 100.0, places=6)
        self.assertGreaterEqual(float(table["attention_%"].min()), 0.0)

    def test_attention_splits_along_the_identity_weight(self) -> None:
        """Offense subgroups must carry exactly off_weight of the attention."""
        row = self.engine.resolve_index("Walker Kessler", "2024-25")
        off_weight = self.engine.auto_off_weight(row)
        balanced = self.engine._domain_normalized(self.engine._final_sub_weights()[row], off_weight)
        self.assertAlmostEqual(float(balanced[self.engine.SUB_IS_OFF].sum()), off_weight, places=9)
        self.assertAlmostEqual(float(balanced[~self.engine.SUB_IS_OFF].sum()), 1.0 - off_weight, places=9)

    def test_block_decomposition_reconstructs_the_distance(self) -> None:
        """Contributions must sum to the squared distance less the position penalty."""
        row = self.engine.resolve_index("Keyonte George", "2025-26")
        comps = self.engine.comps_for(row, top_n=3, candidate_mask=self.engine.SITE_CANDIDATE_MASK)
        targets = [comp["row"] for comp in comps["overall"]]
        decomposition = self.engine.block_decomposition(row, targets)
        _off, _def, overall = self.engine.symmetric_distances(row)
        for target in targets:
            penalty = self.engine.POSITION_PENALTY_WEIGHT * (
                (self.engine.PG_AXIS[row] - self.engine.PG_AXIS[target]) ** 2 / self.engine.PTERM_REF
            )
            total = sum(block["contribution"] for block in decomposition[target].values())
            self.assertAlmostEqual(total, overall[target] ** 2 - penalty, places=9)

    def test_comps_exclude_self_and_the_same_player(self) -> None:
        row = self.engine.resolve_index("Stephen Curry", "2021-22")
        comps = self.engine.comps_for(row, top_n=10, candidate_mask=self.engine.SITE_CANDIDATE_MASK)
        for domain, rows in comps.items():
            self.assertEqual(len(rows), 10, f"{domain} returned {len(rows)} comps")
            for comp in rows:
                self.assertNotEqual(comp["row"], row)
                self.assertNotEqual(comp["player"], "Stephen Curry", f"self-match in {domain}")

    def test_candidate_mask_is_respected(self) -> None:
        row = self.engine.resolve_index("Nikola Jokic", "2022-23")
        mask = self.engine.SITE_CANDIDATE_MASK
        for rows in self.engine.comps_for(row, top_n=10, candidate_mask=mask).values():
            for comp in rows:
                self.assertTrue(bool(mask[comp["row"]]), "comp drawn from outside the pool")

    def test_similarity_score_is_monotone_in_distance(self) -> None:
        distances = np.linspace(0.1, 4.0, 200)
        scores = self.engine.percentile_similarity(distances)
        self.assertTrue(np.all(np.diff(scores) <= 1e-9), "score is not monotone")
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 100.0)))


if __name__ == "__main__":
    unittest.main()
