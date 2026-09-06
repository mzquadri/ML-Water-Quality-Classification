"""Rerun the benchmark into a temporary file and check the conclusions survive.

    python scripts/check_reproducibility.py

The committed results file is what the README is written from, so this must not
overwrite it. The rerun goes somewhere else and the two are compared.

Two standards, as elsewhere. The findings the README argues from are qualitative
and must hold exactly, because if one flips the README is saying something false.
The numbers are compared within a tolerance, because XGBoost and SVM do not
promise identical output across platforms and a run that differs in the fourth
decimal is not a regression.

Writes nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "results" / "benchmark.json"

TOLERANCE = 0.03


def findings(data: dict) -> dict:
    """The claims the README rests on."""
    models = data["models"]
    reference = data["reference_points"]
    chosen = data["selection"]["chosen"]
    selected = models[chosen]
    trained = {k: v for k, v in models.items()
               if v.get("cv_f1_mean") not in (None, 0.0)}

    return {
        "the majority class sits at the floor":
            abs(models["majority_class"]["test"]["accuracy"]
                - reference["floor_accuracy"]) < 1e-9,
        "the optimal rule sits at the ceiling":
            abs(models["bayes_optimal"]["test"]["accuracy"]
                - reference["ceiling_accuracy"]) < 1e-9,
        "no model beats the optimal rule": all(
            entry["test"]["accuracy"] <= reference["ceiling_accuracy"] + 1e-12
            for entry in models.values()),
        "every model beats the majority class": all(
            entry["test"]["accuracy"] > reference["floor_accuracy"]
            for name, entry in trained.items()),
        "cross-validation picks the model the results record":
            max(trained, key=lambda k: trained[k]["cv_f1_mean"]) == chosen,
        "the selected model covers more than nine tenths of the range":
            selected["share_of_available_signal"] > 0.9,
        "the selected model is within half a point of the ceiling":
            (reference["ceiling_accuracy"] - selected["test"]["accuracy"]) < 0.005,
        "logistic regression is the weakest of the four":
            models["Logistic Regression"]["test"]["accuracy"] == min(
                entry["test"]["accuracy"] for entry in trained.values()),
        "balanced accuracy separates the majority baseline from the models":
            models["majority_class"]["test"]["balanced_accuracy"] < 0.51,
        "most of the errors are ones the optimal rule also makes":
            data["error_analysis"]["share_of_errors_that_are_irreducible"] > 0.5,
        "the error rate is higher inside the ambiguous band":
            data["error_analysis"]["error_rate_inside_the_band"]
            > data["error_analysis"]["error_rate_outside_the_band"],
        "the model is less confident when it is wrong":
            data["error_analysis"]["model_confidence"]["mean_on_wrong"]
            < data["error_analysis"]["model_confidence"]["mean_on_correct"],
        "the label rate is close to the stated prior":
            abs(data["data"]["positive_rate"] - 0.40) < 0.02,
        "clipping affects a negligible share of values":
            max(data["ceiling"]["clipped_value_share"].values()) < 0.02,
    }


def run(module: str, out: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(completed.stdout[-1500:])
        print(completed.stderr[-1500:])
        raise SystemExit(f"  {module} failed with exit code {completed.returncode}")


def main() -> int:
    if not RECORDED.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as directory:
        fresh_path = Path(directory) / "benchmark.json"
        print("  rerunning the ceiling, the benchmark and the error analysis")
        for module in ("src.ceiling", "src.benchmark", "src.error_analysis"):
            run(module, fresh_path)
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))

    failures = []
    recorded_findings, fresh_findings = findings(recorded), findings(fresh)
    for description, holds in recorded_findings.items():
        if not holds:
            failures.append(f"  the recorded results no longer support: {description}")
        elif not fresh_findings[description]:
            failures.append(f"  a rerun no longer supports: {description}")
    held = sum(1 for key in recorded_findings
               if recorded_findings[key] and fresh_findings[key])
    print(f"  {len(recorded_findings)} findings checked, {held} hold")

    drifted = 0
    compared = 0
    for name, entry in recorded["models"].items():
        for metric in ("accuracy", "f1"):
            was = entry["test"][metric]
            now = fresh["models"][name]["test"][metric]
            compared += 1
            if was and abs(now - was) > TOLERANCE:
                drifted += 1
                failures.append(f"  {name} {metric} moved from {was:.4f} to "
                                f"{now:.4f}, more than {TOLERANCE}")
    ceiling_shift = abs(fresh["ceiling"]["accuracy"] - recorded["ceiling"]["accuracy"])
    compared += 1
    if ceiling_shift > 0.01:
        drifted += 1
        failures.append(f"  the ceiling moved by {ceiling_shift:.4f}")
    print(f"  {compared} values compared, {compared - drifted} within tolerance")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} checks failed")

    print("  the rerun supports the same conclusions as the recorded results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
