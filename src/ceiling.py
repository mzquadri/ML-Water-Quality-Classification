"""Compute the best accuracy any classifier could reach on this generated data.

    python -m src.ceiling

The dataset here is not measured. `src/generate_data.py` draws the label first,
as a coin flip with probability 0.40, and then draws each of the nine features
from one of two normal distributions chosen by that label. The label is not
derived from the features; the features are derived from the label.

That has a consequence worth more than any model score in this repository. The
generative process is known exactly, and the features are drawn independently
given the label, so the optimal classifier is written down rather than learned:
it is naive Bayes carrying the generator's own parameters. Its accuracy is the
ceiling. No model trained on this data can do better than it except by chance,
and a model reported without it is a number with nothing to be read against.

Two details matter for getting the ceiling right.

Missing values are missing completely at random, injected after the features are
drawn, so the optimal rule simply drops the term for an absent feature rather
than imputing anything.

The generator clips each feature to a plausible range, which turns the tails into
point masses and makes the true density something other than normal there. The
share of values affected is measured below rather than assumed away.

Writes the ceiling into results/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"

#: The generator's own parameters, transcribed from src/generate_data.py. Each
#: entry is (mean, standard deviation) for the potable class and then for the
#: non-potable class, with the clipping bounds applied afterwards.
PARAMETERS = {
    "ph": ((7.2, 0.6), (6.8, 1.2), (2, 14)),
    "hardness": ((180, 40), (210, 60), (50, 400)),
    "solids": ((18000, 5000), (22000, 8000), (300, 60000)),
    "chloramines": ((7.0, 1.2), (7.5, 1.8), (1, 13)),
    "sulfate": ((320, 40), (340, 55), (100, 500)),
    "conductivity": ((400, 70), (430, 90), (180, 800)),
    "organic_carbon": ((13, 3), (15, 4), (2, 30)),
    "trihalomethanes": ((60, 15), (68, 20), (5, 130)),
    "turbidity": ((3.5, 0.8), (4.0, 1.0), (1, 7)),
}
PRIOR_POTABLE = 0.40


def log_normal_density(values, mean, sigma):
    """Log density of a normal, evaluated where the values are present."""
    z = (values - mean) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def bayes_log_odds(frame) -> np.ndarray:
    """Log posterior odds of the potable class under the true generative model.

    Each feature contributes an independent term because the generator draws them
    independently given the label. A feature that is missing contributes nothing,
    which is the correct handling when values are missing completely at random.
    """
    total = np.full(len(frame), np.log(PRIOR_POTABLE / (1.0 - PRIOR_POTABLE)))
    for name, (potable, other, _) in PARAMETERS.items():
        values = frame[name].to_numpy(dtype=float)
        present = ~np.isnan(values)
        contribution = np.zeros(len(frame))
        contribution[present] = (
            log_normal_density(values[present], *potable)
            - log_normal_density(values[present], *other))
        total = total + contribution
    return total


def clipping_share(frame) -> dict:
    """How much of the data sits exactly on a clipping bound.

    Where clipping bites, the true density is not the normal density used above,
    so this is the size of the approximation rather than a footnote.
    """
    affected = {}
    for name, (_, _, (low, high)) in PARAMETERS.items():
        values = frame[name].to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        at_bound = int(((present <= low) | (present >= high)).sum())
        affected[name] = round(at_bound / max(len(present), 1), 6)
    return affected


def compute(n_samples: int, seed: int) -> dict:
    from .generate_data import generate_water_data

    frame = generate_water_data(n_samples=n_samples, seed=seed)
    actual = frame["potability"].to_numpy()
    log_odds = bayes_log_odds(frame)
    predicted = (log_odds > 0).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    accuracy = float(accuracy_score(actual, predicted))
    # A binomial interval on the accuracy, because the ceiling is estimated from
    # a finite sample and quoting it to four decimals would be overstating it.
    margin = 1.96 * np.sqrt(accuracy * (1 - accuracy) / len(actual))

    return {
        "samples_used": int(n_samples),
        "seed": int(seed),
        "accuracy": round(accuracy, 5),
        "accuracy_95_interval": [round(accuracy - margin, 5),
                                 round(accuracy + margin, 5)],
        "balanced_accuracy": round(
            float(balanced_accuracy_score(actual, predicted)), 5),
        "f1": round(float(f1_score(actual, predicted)), 5),
        "precision": round(float(precision_score(actual, predicted)), 5),
        "recall": round(float(recall_score(actual, predicted)), 5),
        "roc_auc": round(float(roc_auc_score(actual, log_odds)), 5),
        "majority_class_accuracy": round(
            float(max(np.mean(actual), 1 - np.mean(actual))), 5),
        "clipped_value_share": clipping_share(frame),
        "method": "naive Bayes carrying the generator's own parameters, which is "
                  "the optimal rule because the features are drawn independently "
                  "given the label; missing values contribute no term",
        "caveat": "the generator clips each feature, so the true density is not "
                  "normal at the bounds. The share of values sitting on a bound is "
                  "recorded above, and the ceiling is an estimate on a finite "
                  "sample rather than a closed form",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    print(f"  drawing {arguments.samples:,} samples from the generator")
    result = compute(arguments.samples, arguments.seed)

    print(f"  the best any classifier can do on this data: "
          f"{result['accuracy'] * 100:.2f} percent "
          f"({result['accuracy_95_interval'][0] * 100:.2f} to "
          f"{result['accuracy_95_interval'][1] * 100:.2f})")
    print(f"  balanced accuracy {result['balanced_accuracy'] * 100:.2f}, "
          f"F1 {result['f1'] * 100:.2f}, ROC-AUC {result['roc_auc']:.4f}")
    print(f"  always predicting the larger class: "
          f"{result['majority_class_accuracy'] * 100:.2f} percent")
    worst = max(result["clipped_value_share"].items(), key=lambda kv: kv[1])
    print(f"  most clipped feature: {worst[0]} at {worst[1] * 100:.3f} percent "
          f"of values on a bound")

    payload = {}
    if arguments.out.exists():
        payload = json.loads(arguments.out.read_text(encoding="utf-8"))
    payload["ceiling"] = result
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"\n  wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
