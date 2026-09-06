"""Fail if the README stops agreeing with the recorded results.

    python scripts/check_repository.py

Checks that the documented files exist and compile, that the recorded numbers are
internally consistent, and that every figure the README quotes still matches
`results/benchmark.json`.

Each claim is matched with its surrounding words included, so a value that has
drifted into a different sentence does not accidentally satisfy a check.
"""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark.json"

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "src/generate_data.py",
    "src/models.py",
    "src/benchmark.py",
    "src/ceiling.py",
    "src/error_analysis.py",
    "src/pipeline.py",
    "tests/test_data.py",
    "tests/test_pipeline.py",
    "scripts/figures/generate_figures.py",
    "docs/diagrams/pipeline.svg",
    "docs/figures/01_against_the_ceiling.png",
    "docs/figures/02_what_the_data_is.png",
    "docs/figures/03_errors.png",
    "results/benchmark.json",
    "results/model_results.json",
)

ROW_ORDER = (
    ("majority_class", "Always the larger class"),
    ("Logistic Regression", "Logistic Regression"),
    ("XGBoost", "XGBoost"),
    ("Random Forest", "Random Forest"),
    ("XGBoost (Tuned)", "XGBoost (Tuned)"),
    ("SVM (RBF)", "SVM (RBF)"),
    ("bayes_optimal", "The generator's own rule"),
)


def claims(data: dict) -> list[tuple[str, str]]:
    """Every quoted number, paired with enough context to anchor it."""
    info = data["data"]
    models = data["models"]
    reference = data["reference_points"]
    errors = data["error_analysis"]
    ceiling = data["ceiling"]
    chosen = data["selection"]["chosen"]
    selected = models[chosen]

    out = [
        (f"| Rows | {info['rows']:,} |", "the row count"),
        (f"| Features | {info['features']} numeric |", "the feature count"),
        (f"| Class balance | {info['class_counts']['0']:,} class 0, "
         f"{info['class_counts']['1']:,} class 1 |", "the class balance row"),
        (f"| Majority class share | {info['majority_class_share'] * 100:.2f} percent |",
         "the majority share row"),
        (f"| Rows with a missing value | {info['rows_with_any_missing']:,} |",
         "the missing rows row"),
        (f"| Duplicate feature rows | {info['duplicate_feature_rows']} |",
         "the duplicate row"),
        (f"giving {data['split']['train_rows']:,} training and "
         f"{data['split']['test_rows']:,} test rows", "the split sizes"),
        (f"{data['split']['folds']}-fold stratified cross-validation",
         "the fold count"),

        (f"reaches\n{selected['share_of_available_signal'] * 100:.1f} percent of the "
         f"distance", "the share of available signal"),
        (f"selects \\*\\*{chosen}\\*\\* at {selected['cv_f1_mean'] * 100:.2f} CV F1",
         "the selected model and its CV score"),
        (f"optimal rule reaches {reference['ceiling_accuracy'] * 100:.2f}",
         "the ceiling on this split"),
        (f"it reaches {ceiling['accuracy'] * 100:.2f} with a 95 percent interval of "
         f"{ceiling['accuracy_95_interval'][0] * 100:.2f} to "
         f"{ceiling['accuracy_95_interval'][1] * 100:.2f}", "the population ceiling"),
        (f"gap between the selected model and the optimal rule is "
         f"{(reference['ceiling_accuracy'] - selected['test']['accuracy']) * 100:.2f} "
         f"accuracy points", "the gap to the ceiling"),

        (f"makes {errors['errors']} errors on "
         f"{errors['test_rows']:,} held-out rows", "the error count"),
        (f"{errors['false_negatives']} false negatives\nand "
         f"{errors['false_positives']} false positives", "the error split"),
        (f"\\*\\*{errors['errors_the_optimal_rule_also_makes']} of those "
         f"{errors['errors']}, or "
         f"{errors['share_of_errors_that_are_irreducible'] * 100:.1f} percent",
         "the irreducible share"),
        (f"Only {errors['errors_the_optimal_rule_avoids']} are errors a perfect "
         f"model would have avoided", "the avoidable errors"),
        (f"between {errors['ambiguous_band'][0]} and {errors['ambiguous_band'][1]}, "
         f"covering {errors['rows_in_the_ambiguous_band']} rows",
         "the ambiguous band"),
        (f"error rate is {errors['error_rate_inside_the_band'] * 100:.1f} percent; "
         f"outside it, {errors['error_rate_outside_the_band'] * 100:.1f} percent",
         "the band error rates"),
        (f"{errors['model_confidence']['mean_on_wrong']} against "
         f"{errors['model_confidence']['mean_on_correct']}", "the confidence gap"),
        (f"{max(abs(v) for v in errors['standardised_feature_shift_on_errors'].values()):.2f} "
         f"standard\ndeviations", "the largest feature shift"),

        (f"at most {max(ceiling['clipped_value_share'].values()) * 100:.2f} percent "
         f"of values sit on a bound", "the clipping share"),
    ]

    for key, label in ROW_ORDER:
        entry = models[key]
        test = entry["test"]
        share = entry["share_of_available_signal"] * 100
        if key in ("majority_class", "bayes_optimal"):
            continue
        out.append((
            f"| {label} | {test['accuracy'] * 100:.2f} | "
            f"{test['balanced_accuracy'] * 100:.2f} | {test['f1'] * 100:.2f} | "
            f"{test['roc_auc'] * 100:.2f} | {test['pr_auc'] * 100:.2f} | "
            f"{entry['cv_f1_mean'] * 100:.2f} | {share:.1f}% |",
            f"the {key} results row"))

    best = models["XGBoost (Tuned)"]["best_params"]
    out.append((
        f"selected `max_depth` {best['clf__max_depth']} and `n_estimators` "
        f"{best['clf__n_estimators']}", "the tuned parameters"))
    return out


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"  missing required files: {', '.join(missing)}")
    for source in sorted((ROOT / "src").glob("*.py")):
        py_compile.compile(source, doraise=True)
    print(f"  {len(REQUIRED_FILES)} required files present, src/ compiles")

    # The pickle may exist locally, because running src/pipeline.py writes one.
    # What must not happen is it being committed: a pickle executes on load, so
    # shipping one asks a reader to trust a binary to inspect the repository.
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "results/best_model_xgboost.pkl"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    if tracked.returncode == 0:
        raise SystemExit("  results/best_model_xgboost.pkl is tracked again. A "
                         "pickle executes on load; the README says it is not "
                         "committed.")
    print("  no model pickle is tracked")

    if not RESULTS.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    # The selected model must be the one cross-validation actually picked.
    trained = {k: v for k, v in data["models"].items()
               if v.get("cv_f1_mean") not in (None, 0.0)}
    best = max(trained, key=lambda k: trained[k]["cv_f1_mean"])
    if best != data["selection"]["chosen"]:
        raise SystemExit(f"  the results record {data['selection']['chosen']} as "
                         f"selected but cross-validation ranks {best} highest")
    print(f"  cross-validation selection is consistent: {best}")

    # No model may exceed the optimal rule. If one does, the ceiling is wrong.
    ceiling = data["reference_points"]["ceiling_accuracy"]
    above = [k for k, v in data["models"].items()
             if v["test"]["accuracy"] > ceiling + 1e-12]
    if above:
        raise SystemExit(f"  {', '.join(above)} scores above the optimal rule, so "
                         f"the ceiling is not a ceiling")
    print(f"  no model exceeds the optimal rule at {ceiling * 100:.2f}")

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    failures = []
    checks = claims(data)
    for expected, description in checks:
        if re.sub(r"\s+", " ", expected.replace("\\*", "*")) not in flat:
            failures.append(f"  README does not state {description}: expected "
                            f"{re.sub(r'\\s+', ' ', expected).strip()!r}")
    print(f"  {len(checks) - len(failures)} of {len(checks)} recorded numbers "
          f"found in the README")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} claims in the README no longer match "
                         f"results/benchmark.json")

    print("  README and results/benchmark.json agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
