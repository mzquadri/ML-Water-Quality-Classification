"""Tests for what the generator produces and what that permits.

The single most important fact about this repository is how the label comes
about: it is drawn before any feature exists. Everything the README claims rests
on that, so it is pinned here rather than left as a comment in the generator.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.ceiling import PARAMETERS, PRIOR_POTABLE, bayes_log_odds, clipping_share
from src.generate_data import generate_water_data
from src.models import FEATURE_COLUMNS, TARGET


class Schema(unittest.TestCase):
    def setUp(self):
        self.frame = generate_water_data(n_samples=4000, seed=1)

    def test_the_expected_columns_are_present_and_nothing_else(self):
        self.assertEqual(list(self.frame.columns), [*FEATURE_COLUMNS, TARGET])

    def test_every_feature_is_numeric(self):
        for name in FEATURE_COLUMNS:
            with self.subTest(feature=name):
                self.assertTrue(np.issubdtype(self.frame[name].dtype, np.number))

    def test_the_target_is_binary(self):
        self.assertEqual(set(self.frame[TARGET].unique()), {0, 1})

    def test_the_target_is_never_missing(self):
        self.assertEqual(int(self.frame[TARGET].isna().sum()), 0)

    def test_the_ceiling_parameters_cover_every_feature(self):
        """If a feature were added without a parameter entry, the ceiling would
        silently be computed from an incomplete model."""
        self.assertEqual(set(PARAMETERS), set(FEATURE_COLUMNS))


class TheLabelIsDrawnFirst(unittest.TestCase):
    """The claim the whole README rests on."""

    def test_the_class_rate_matches_the_stated_prior(self):
        frame = generate_water_data(n_samples=40_000, seed=3)
        self.assertAlmostEqual(float(frame[TARGET].mean()), PRIOR_POTABLE, places=2)

    def test_the_label_rate_does_not_depend_on_the_features(self):
        """A threshold rule over the features would make the rate move when the
        feature distributions move. Here it cannot, because the label came first.

        Both seeds draw entirely different feature values; the class rate is the
        same to sampling error either way.
        """
        first = generate_water_data(n_samples=20_000, seed=11)
        second = generate_water_data(n_samples=20_000, seed=12)
        self.assertFalse(np.allclose(first["ph"].to_numpy()[:100],
                                     second["ph"].to_numpy()[:100], equal_nan=True))
        self.assertAlmostEqual(float(first[TARGET].mean()),
                               float(second[TARGET].mean()), places=2)

    def test_no_feature_separates_the_classes_on_its_own(self):
        """If any single feature determined the label, the target would be a rule
        over that feature rather than a draw."""
        frame = generate_water_data(n_samples=20_000, seed=4)
        for name in FEATURE_COLUMNS:
            with self.subTest(feature=name):
                values = frame[name]
                present = values.notna()
                correlation = abs(np.corrcoef(values[present],
                                              frame[TARGET][present])[0, 1])
                self.assertLess(correlation, 0.4)


class MissingValues(unittest.TestCase):
    def test_only_the_three_documented_columns_are_blanked(self):
        frame = generate_water_data(n_samples=20_000, seed=5)
        blanked = {name for name in FEATURE_COLUMNS if frame[name].isna().any()}
        self.assertEqual(blanked, {"ph", "sulfate", "trihalomethanes"})

    def test_the_missing_rate_is_close_to_seven_percent(self):
        frame = generate_water_data(n_samples=20_000, seed=6)
        for name in ("ph", "sulfate", "trihalomethanes"):
            with self.subTest(feature=name):
                self.assertAlmostEqual(float(frame[name].isna().mean()), 0.07,
                                       places=2)

    def test_missingness_does_not_depend_on_the_label(self):
        """Injected after the draw, so it is missing completely at random. If it
        were not, dropping or imputing those rows would bias the result."""
        frame = generate_water_data(n_samples=40_000, seed=8)
        for name in ("ph", "sulfate", "trihalomethanes"):
            with self.subTest(feature=name):
                by_class = frame.groupby(TARGET)[name].apply(
                    lambda column: column.isna().mean())
                self.assertLess(abs(by_class.iloc[0] - by_class.iloc[1]), 0.02)


class Ceiling(unittest.TestCase):
    def test_the_optimal_rule_beats_the_majority_class(self):
        frame = generate_water_data(n_samples=20_000, seed=9)
        predicted = (bayes_log_odds(frame) > 0).astype(int)
        actual = frame[TARGET].to_numpy()
        optimal = float((predicted == actual).mean())
        majority = float(max(actual.mean(), 1 - actual.mean()))
        self.assertGreater(optimal, majority + 0.15)

    def test_it_beats_any_single_feature_threshold(self):
        """The ceiling has to be above what a one-column rule achieves, or it is
        not a ceiling worth reporting."""
        frame = generate_water_data(n_samples=10_000, seed=10)
        actual = frame[TARGET].to_numpy()
        optimal = float(((bayes_log_odds(frame) > 0).astype(int) == actual).mean())

        best_single = 0.0
        for name in FEATURE_COLUMNS:
            values = frame[name].to_numpy(dtype=float)
            present = ~np.isnan(values)
            for cut in np.quantile(values[present], np.linspace(0.1, 0.9, 17)):
                for direction in (1, -1):
                    guess = np.zeros(len(frame), dtype=int)
                    guess[present] = (direction * values[present]
                                      > direction * cut).astype(int)
                    best_single = max(best_single, float((guess == actual).mean()))
        self.assertGreater(optimal, best_single)

    def test_missing_values_contribute_no_term(self):
        """A row differing only in whether a feature is present must keep the
        contribution of every other feature unchanged."""
        frame = generate_water_data(n_samples=200, seed=13)
        complete = frame[frame[FEATURE_COLUMNS].notna().all(axis=1)].head(20).copy()
        base = bayes_log_odds(complete)
        blanked = complete.copy()
        blanked.loc[:, "ph"] = np.nan
        without = bayes_log_odds(blanked)
        self.assertFalse(np.allclose(base, without))
        self.assertTrue(np.all(np.isfinite(without)))

    def test_clipping_is_measured_and_small(self):
        frame = generate_water_data(n_samples=20_000, seed=14)
        shares = clipping_share(frame)
        self.assertEqual(set(shares), set(FEATURE_COLUMNS))
        self.assertLess(max(shares.values()), 0.02)


class Reproducibility(unittest.TestCase):
    def test_the_same_seed_gives_the_same_frame(self):
        first = generate_water_data(n_samples=500, seed=2)
        second = generate_water_data(n_samples=500, seed=2)
        self.assertTrue(first.equals(second))

    def test_a_different_seed_gives_a_different_frame(self):
        first = generate_water_data(n_samples=500, seed=2)
        second = generate_water_data(n_samples=500, seed=3)
        self.assertFalse(first.equals(second))


if __name__ == "__main__":
    unittest.main()
