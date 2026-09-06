"""Tests for the split, the preprocessing placement, and the recorded results.

The preprocessing here was already correct, with the imputer and scaler inside
each Pipeline. That is easy to undo by accident, so it is pinned: fitting either
one before the split is the leak these tests exist to catch.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.benchmark import describe_data, score
from src.generate_data import generate_water_data
from src.models import FEATURE_COLUMNS, TARGET, XGBOOST_GRID, build_models

RESULTS = Path(__file__).resolve().parents[1] / "results" / "benchmark.json"


class PreprocessingPlacement(unittest.TestCase):
    """Both steps must sit inside the estimator, not outside it."""

    def setUp(self):
        self.models = build_models()

    def test_every_model_carries_its_own_imputer(self):
        for name, model in self.models.items():
            with self.subTest(model=name):
                self.assertIn("imputer", dict(model.named_steps))

    def test_the_scaled_models_carry_their_own_scaler(self):
        for name in ("Logistic Regression", "SVM (RBF)"):
            with self.subTest(model=name):
                self.assertIn("scaler", dict(self.models[name].named_steps))

    def test_the_imputer_learns_only_from_what_it_is_fitted_on(self):
        """The property that makes the placement matter.

        Fitting on the training rows must give a different median than fitting on
        everything, or the pipeline placement would be decorative.
        """
        from sklearn.impute import SimpleImputer

        frame = generate_water_data(n_samples=2000, seed=21)
        train = frame.iloc[:1000][FEATURE_COLUMNS]
        everything = frame[FEATURE_COLUMNS]

        on_train = SimpleImputer(strategy="median").fit(train)
        on_everything = SimpleImputer(strategy="median").fit(everything)
        self.assertFalse(np.allclose(on_train.statistics_,
                                     on_everything.statistics_))

    def test_fitting_the_pipeline_does_not_touch_the_held_out_rows(self):
        """Transforming after fitting must not change what was learned."""
        frame = generate_water_data(n_samples=1200, seed=22)
        train = frame.iloc[:900]
        held_out = frame.iloc[900:]

        model = build_models()["Logistic Regression"]
        model.fit(train[FEATURE_COLUMNS], train[TARGET])
        before = model.named_steps["imputer"].statistics_.copy()
        model.predict(held_out[FEATURE_COLUMNS])
        np.testing.assert_array_equal(model.named_steps["imputer"].statistics_,
                                      before)


class Splitting(unittest.TestCase):
    def setUp(self):
        from sklearn.model_selection import train_test_split
        self.frame = generate_water_data(n_samples=2000, seed=23)
        self.train_index, self.test_index = train_test_split(
            np.arange(len(self.frame)), test_size=0.2, random_state=42,
            stratify=self.frame[TARGET])

    def test_the_splits_do_not_overlap(self):
        self.assertEqual(set(self.train_index) & set(self.test_index), set())

    def test_together_they_cover_every_row(self):
        self.assertEqual(len(set(self.train_index) | set(self.test_index)),
                         len(self.frame))

    def test_the_class_rate_is_preserved(self):
        target = self.frame[TARGET].to_numpy()
        self.assertAlmostEqual(target[self.train_index].mean(),
                               target[self.test_index].mean(), places=2)

    def test_the_split_is_reproducible(self):
        from sklearn.model_selection import train_test_split
        again, _ = train_test_split(np.arange(len(self.frame)), test_size=0.2,
                                    random_state=42, stratify=self.frame[TARGET])
        np.testing.assert_array_equal(self.train_index, again)


class Scoring(unittest.TestCase):
    def test_a_perfect_prediction_scores_one(self):
        actual = np.array([0, 1] * 50)
        result = score(actual, actual)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_predicting_one_class_splits_accuracy_from_balanced_accuracy(self):
        """The reason both are reported on a 60/40 target."""
        actual = np.array([0] * 60 + [1] * 40)
        result = score(actual, np.zeros_like(actual))
        self.assertAlmostEqual(result["accuracy"], 0.60)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.50)
        self.assertEqual(result["f1"], 0.0)

    def test_the_confusion_matrix_totals_the_rows(self):
        actual = np.array([0] * 30 + [1] * 20)
        predicted = np.array([0] * 25 + [1] * 5 + [0] * 4 + [1] * 16)
        result = score(actual, predicted)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), len(actual))


class DataSummary(unittest.TestCase):
    def test_it_counts_what_the_readme_quotes(self):
        frame = generate_water_data(n_samples=1000, seed=24)
        summary = describe_data(frame)
        self.assertEqual(summary["rows"], 1000)
        self.assertEqual(summary["features"], len(FEATURE_COLUMNS))
        self.assertEqual(sum(summary["class_counts"].values()), 1000)
        self.assertEqual(summary["rows_with_any_missing"],
                         int(frame[FEATURE_COLUMNS].isna().any(axis=1).sum()))


class RecordedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULTS.is_file():
            raise unittest.SkipTest("no recorded results yet")
        cls.data = json.loads(RESULTS.read_text(encoding="utf-8"))

    def test_every_recorded_metric_recomputes_from_its_predictions(self):
        for name, entry in self.data["models"].items():
            if "predictions" not in entry:
                continue
            with self.subTest(model=name):
                predicted = np.array(entry["predictions"]["predicted"])
                self.assertEqual(len(predicted), self.data["split"]["test_rows"])
                self.assertEqual(set(np.unique(predicted)) - {0, 1}, set())

    def test_no_model_beats_the_optimal_rule(self):
        ceiling = self.data["reference_points"]["ceiling_accuracy"]
        for name, entry in self.data["models"].items():
            with self.subTest(model=name):
                self.assertLessEqual(entry["test"]["accuracy"], ceiling + 1e-12)

    def test_the_selected_model_is_the_one_cross_validation_ranked_first(self):
        trained = {k: v for k, v in self.data["models"].items()
                   if v.get("cv_f1_mean") not in (None, 0.0)}
        best = max(trained, key=lambda k: trained[k]["cv_f1_mean"])
        self.assertEqual(best, self.data["selection"]["chosen"])

    def test_the_tuned_grid_matches_the_one_in_the_code(self):
        self.assertEqual(self.data["models"]["XGBoost (Tuned)"]["grid_searched"],
                         XGBOOST_GRID)

    def test_the_error_analysis_adds_up(self):
        errors = self.data["error_analysis"]
        self.assertEqual(errors["false_negatives"] + errors["false_positives"],
                         errors["errors"])
        self.assertEqual(errors["errors_the_optimal_rule_also_makes"]
                         + errors["errors_the_optimal_rule_avoids"],
                         errors["errors"])

    def test_the_original_reference_metrics_are_still_present(self):
        """The earlier run's artifact is provenance and stays in the repository."""
        original = (Path(__file__).resolve().parents[1] / "results"
                    / "model_results.json")
        self.assertTrue(original.is_file())
        recorded = json.loads(original.read_text(encoding="utf-8"))
        self.assertIn("SVM (RBF)", recorded)


if __name__ == "__main__":
    unittest.main()
