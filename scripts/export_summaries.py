"""Write small flat summaries for use outside this repository.

    python scripts/export_summaries.py

results/benchmark.json carries per-row predictions and is not a convenient thing
to fetch from a web page. These four files hold only what a summary view needs.
No feature values are exported: the dataset is regenerated from
src/generate_data.py in one command, so shipping a copy would add weight without
adding anything.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark.json"
OUT = ROOT / "results" / "summary"


def main() -> int:
    if not RESULTS.is_file():
        raise SystemExit("run `python -m src.benchmark` first")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    written = {
        "dataset_summary.json": {
            "source": data["data"]["source"],
            "rows": data["data"]["rows"],
            "features": data["data"]["features"],
            "target": data["data"]["target"],
            "class_counts": data["data"]["class_counts"],
            "majority_class_share": data["data"]["majority_class_share"],
            "rows_with_any_missing": data["data"]["rows_with_any_missing"],
            "duplicate_feature_rows": data["data"]["duplicate_feature_rows"],
            "split": data["split"],
        },
        "feature_summary.json": {
            "features": [
                {"name": name,
                 "missing": data["data"]["missing_per_feature"][name],
                 "missing_share": data["data"]["missing_share_per_feature"][name]}
                for name in data["data"]["feature_names"]],
            "note": "all numeric, all drawn independently given the label",
        },
        "model_metrics.json": {
            "selected": data["selection"]["chosen"],
            "selection_criterion": data["selection"]["criterion"],
            "floor_accuracy": data["reference_points"]["floor_accuracy"],
            "ceiling_accuracy": data["reference_points"]["ceiling_accuracy"],
            "models": {
                name: {
                    "accuracy": entry["test"]["accuracy"],
                    "balanced_accuracy": entry["test"]["balanced_accuracy"],
                    "f1": entry["test"]["f1"],
                    "roc_auc": entry["test"].get("roc_auc"),
                    "pr_auc": entry["test"].get("pr_auc"),
                    "cv_f1_mean": entry.get("cv_f1_mean"),
                    "share_of_available_signal": entry["share_of_available_signal"],
                }
                for name, entry in data["models"].items()},
        },
        "error_summary.json": {
            key: value for key, value in data["error_analysis"].items()
            if key != "standardised_feature_shift_on_errors"},
    }

    for name, payload in written.items():
        (OUT / name).write_text(json.dumps(payload, indent=2) + "\n",
                                encoding="utf-8", newline="\n")
        print(f"  wrote {(OUT / name).relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
