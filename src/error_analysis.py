"""Look at what the selected model gets wrong, and whether it could get it right.

    python -m src.error_analysis

Two questions, and the second is the one that matters here.

Where do the errors fall? On a 60/40 split the two error types are not symmetric,
and a label called potability makes the false negative the one a reader will
worry about, so both are counted separately and by confidence.

Could a better model fix them? That is answerable on this data, because the
generating process is known. Every test row has a true posterior probability, and
a row where that probability sits near 0.5 is one the generator made ambiguous.
An error there is not a modelling failure, it is the data. Separating the two is
the point of this file.

Nothing here says anything about water. The label is a coin flip drawn before any
feature existed, so a false negative is a misread of a synthetic draw, not a
missed contaminant.

Reads and updates results/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models import FEATURE_COLUMNS, SEED, TARGET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"

#: A test row whose true posterior sits inside this band is one the generator
#: made close to a coin flip. Errors there are irreducible.
AMBIGUOUS = (0.35, 0.65)


def analyse(frame_test, actual, predicted, probability, true_posterior) -> dict:
    """Split the mistakes into the reducible and the irreducible."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    wrong = actual != predicted

    false_negative = (actual == 1) & (predicted == 0)
    false_positive = (actual == 0) & (predicted == 1)
    ambiguous = (true_posterior >= AMBIGUOUS[0]) & (true_posterior <= AMBIGUOUS[1])

    # An error the optimal rule also makes cannot be fixed by a better model on
    # this data. One it avoids is, in principle, still available.
    optimal = (true_posterior > 0.5).astype(int)
    optimal_wrong = actual != optimal

    result = {
        "test_rows": len(actual),
        "errors": int(wrong.sum()),
        "false_negatives": int(false_negative.sum()),
        "false_positives": int(false_positive.sum()),
        "errors_the_optimal_rule_also_makes": int((wrong & optimal_wrong).sum()),
        "errors_the_optimal_rule_avoids": int((wrong & ~optimal_wrong).sum()),
        "share_of_errors_that_are_irreducible": round(
            float((wrong & optimal_wrong).sum() / max(wrong.sum(), 1)), 4),
        "errors_in_the_ambiguous_band": int((wrong & ambiguous).sum()),
        "ambiguous_band": list(AMBIGUOUS),
        "rows_in_the_ambiguous_band": int(ambiguous.sum()),
        "error_rate_inside_the_band": round(
            float(wrong[ambiguous].mean()) if ambiguous.any() else 0.0, 4),
        "error_rate_outside_the_band": round(
            float(wrong[~ambiguous].mean()) if (~ambiguous).any() else 0.0, 4),
    }

    # Model confidence on the rows it got wrong, against the rows it got right.
    result["model_confidence"] = {
        "mean_on_correct": round(float(np.abs(probability - 0.5)[~wrong].mean()), 4),
        "mean_on_wrong": round(float(np.abs(probability - 0.5)[wrong].mean()), 4),
        "note": "distance from 0.5, so larger means more confident",
    }

    # Feature ranges where the model errs, against the split as a whole. Reported
    # as a standardised shift so features on different scales can be compared.
    shifts = {}
    for name in FEATURE_COLUMNS:
        values = frame_test[name].to_numpy(dtype=float)
        present = ~np.isnan(values)
        overall = values[present]
        errored = values[present & wrong]
        if len(errored) < 5 or overall.std() == 0:
            continue
        shifts[name] = round(
            float((errored.mean() - overall.mean()) / overall.std()), 4)
    result["standardised_feature_shift_on_errors"] = dict(
        sorted(shifts.items(), key=lambda kv: -abs(kv[1])))

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    if not arguments.out.exists():
        raise SystemExit("run `python -m src.benchmark` first")
    payload = json.loads(arguments.out.read_text(encoding="utf-8"))
    chosen = payload["selection"]["chosen"]
    if "predictions" not in payload["models"].get(chosen, {}):
        raise SystemExit(f"no stored predictions for {chosen}; rerun the benchmark")

    from sklearn.model_selection import train_test_split

    from .benchmark import load_frame
    from .ceiling import bayes_log_odds

    frame = load_frame()
    _, index_test = train_test_split(
        np.arange(len(frame)), test_size=payload["split"]["test_fraction"],
        random_state=SEED, stratify=frame[TARGET])
    frame_test = frame.iloc[index_test]

    log_odds = bayes_log_odds(frame_test)
    true_posterior = 1.0 / (1.0 + np.exp(-log_odds))

    entry = payload["models"][chosen]
    result = analyse(
        frame_test,
        frame_test[TARGET].to_numpy(),
        np.array(entry["predictions"]["predicted"]),
        np.array(entry["predictions"]["probability"]),
        true_posterior)
    result["model"] = chosen
    payload["error_analysis"] = result

    arguments.out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8", newline="\n")

    print(f"  {chosen} on {result['test_rows']:,} held-out rows")
    print(f"    errors {result['errors']}  "
          f"({result['false_negatives']} false negative, "
          f"{result['false_positives']} false positive)")
    print(f"    of those, {result['errors_the_optimal_rule_also_makes']} are ones "
          f"the optimal rule also makes")
    print(f"    which is {result['share_of_errors_that_are_irreducible'] * 100:.1f} "
          f"percent of them")
    print(f"    error rate inside the ambiguous band "
          f"{result['error_rate_inside_the_band'] * 100:.1f}, outside it "
          f"{result['error_rate_outside_the_band'] * 100:.1f}")
    print(f"    confidence on correct {result['model_confidence']['mean_on_correct']}, "
          f"on wrong {result['model_confidence']['mean_on_wrong']}")
    print("    largest standardised feature shifts on errors:")
    for name, shift in list(result["standardised_feature_shift_on_errors"].items())[:4]:
        print(f"      {name:<18} {shift:+.3f}")
    print(f"\n  wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
