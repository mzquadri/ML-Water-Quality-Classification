"""Render the README figures from results/benchmark.json.

    python scripts/figures/generate_figures.py

Each figure answers one question:

  01  how far up the available range does each model get
  02  what the generated data actually contains
  03  what the selected model gets wrong, and whether it could have got it right

Everything is read from the recorded results rather than recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import portfolio_style as ps

RESULTS = ROOT / "results" / "benchmark.json"
FIGURES = ROOT / "docs" / "figures"

ORDER = ("majority_class", "Logistic Regression", "XGBoost", "Random Forest",
         "XGBoost (Tuned)", "SVM (RBF)", "bayes_optimal")
SHORT = {
    "majority_class": "Always the\nlarger class",
    "Logistic Regression": "Logistic\nregression",
    "XGBoost": "XGBoost",
    "Random Forest": "Random\nforest",
    "XGBoost (Tuned)": "XGBoost\ntuned",
    "SVM (RBF)": "SVM\nRBF",
    "bayes_optimal": "The generator's\nown rule",
}


def load() -> dict:
    if not RESULTS.exists():
        raise SystemExit("run `python -m src.benchmark` first")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    for key in ("models", "reference_points", "error_analysis", "ceiling"):
        if key not in data:
            raise SystemExit(f"results/benchmark.json has no '{key}'; run "
                             f"src.ceiling, src.benchmark and src.error_analysis")
    return data


def figure_01_against_the_ceiling(data: dict) -> None:
    """How far up the available range does each model get?"""
    models = data["models"]
    floor = data["reference_points"]["floor_accuracy"] * 100
    roof = data["reference_points"]["ceiling_accuracy"] * 100
    chosen = data["selection"]["chosen"]

    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    fig.subplots_adjust(left=0.085, right=0.955, top=0.735, bottom=0.245)

    values = [models[name]["test"]["accuracy"] * 100 for name in ORDER]
    colours = []
    for name in ORDER:
        if name == "majority_class":
            colours.append(ps.AMBER)
        elif name == "bayes_optimal":
            colours.append(ps.GREEN)
        elif name == chosen:
            colours.append(ps.BLUE)
        else:
            colours.append(ps.SLATE)

    positions = np.arange(len(ORDER))
    ax.bar(positions, values, color=colours, width=0.58)
    # Inside the bars: several of them sit within a point of the ceiling line, so
    # a label placed above would land on top of it.
    for index, value in enumerate(values):
        ax.text(index, value - 4.5, f"{value:.2f}", ha="center", fontsize=10.6,
                color="white", fontweight="600")

    ax.axhline(floor, color=ps.AMBER, linewidth=1.4, linestyle=(0, (5, 3)), zorder=4)
    ax.axhline(roof, color=ps.GREEN, linewidth=1.4, linestyle=(0, (5, 3)), zorder=4)
    ax.text(len(ORDER) - 0.42, roof + 1.6, f"ceiling {roof:.2f}", fontsize=9.8,
            color=ps.GREEN, ha="right")
    ax.text(-0.42, floor + 1.6, f"floor {floor:.2f}", fontsize=9.8, color=ps.AMBER)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[name] for name in ORDER], fontsize=9.8, color=ps.INK)
    ax.set_ylabel("Accuracy on the held-out split", fontsize=11, color=ps.MUTED)
    ax.set_ylim(0, 100)
    ps.clean(ax, grid_axis="y")

    selected = models[chosen]["share_of_available_signal"] * 100
    ps.title_block(
        fig, "The models have run out of signal to find",
        "Every score sits between what one class alone achieves and what a rule "
        "carrying the generator's own\nparameters achieves. The second is a "
        "ceiling, not a competitor.")

    ps.footnote(fig, [
        f"{chosen} covers {selected:.1f} percent of the distance between the two "
        f"dashed lines. The remaining {100 - selected:.1f} percent is not a better "
        f"model waiting",
        "to be found: it is the part of the label the generator drew at random and "
        "left no trace of in the features.",
    ])
    ps.save(fig, FIGURES, "01_against_the_ceiling")


def figure_02_what_the_data_is(data: dict) -> None:
    """What does the generated dataset actually contain?"""
    info = data["data"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 6.0),
                                      gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(left=0.085, right=0.965, top=0.70, bottom=0.245, wspace=0.30)

    counts = [info["class_counts"]["0"], info["class_counts"]["1"]]
    left.bar([0, 1], counts, color=[ps.SLATE, ps.TEAL], width=0.56)
    for index, value in enumerate(counts):
        left.text(index, value + 60, f"{value:,}", ha="center", fontsize=11,
                  color=ps.INK, fontweight="600")
    left.set_xticks([0, 1], ["class 0\n(not potable)", "class 1\n(potable)"],
                    fontsize=10, color=ps.INK)
    left.set_ylabel("Rows", fontsize=10.6, color=ps.MUTED)
    left.set_ylim(0, max(counts) * 1.2)
    left.set_title("Class balance", fontsize=12, color=ps.INK, pad=10, loc="left")
    ps.clean(left, grid_axis="y")

    missing = info["missing_share_per_feature"]
    names = list(missing)
    shares = [missing[name] * 100 for name in names]
    order = np.argsort(shares)
    right.barh(np.arange(len(names)), [shares[i] for i in order],
               color=[ps.RED if shares[i] > 0 else ps.HAIR for i in order],
               height=0.62)
    for index, i in enumerate(order):
        if shares[i] > 0:
            right.text(shares[i] + 0.12, index, f"{shares[i]:.1f}%", va="center",
                       fontsize=9.6, color=ps.INK)
    right.set_yticks(np.arange(len(names)), [names[i] for i in order], fontsize=9.6,
                     color=ps.INK)
    right.set_xlabel("Share of rows missing", fontsize=10.6, color=ps.MUTED)
    right.set_xlim(0, max(shares) * 1.35 if max(shares) else 1)
    right.set_title("Missing values by feature", fontsize=12, color=ps.INK, pad=10,
                    loc="left")
    ps.clean(right, grid_axis="x")

    ps.title_block(
        fig, "What the generator produced",
        f"{info['rows']:,} rows, {info['features']} features, no measurement "
        f"involved. The label was drawn first and the\nfeatures were drawn from it.")

    ps.footnote(fig, [
        "Missing values were injected into three features after the data was "
        "drawn, so they are missing completely at random and",
        f"carry no information. {info['rows_with_any_missing']:,} rows have at "
        f"least one, and {info['duplicate_feature_rows']} rows are duplicated across "
        f"the feature columns.",
    ])
    ps.save(fig, FIGURES, "02_what_the_data_is")


def figure_03_errors(data: dict) -> None:
    """What does the selected model get wrong, and could it have got it right?"""
    errors = data["error_analysis"]
    chosen = errors["model"]
    matrix = np.array(data["models"][chosen]["test"]["confusion_matrix"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 6.2),
                                      gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.10, right=0.965, top=0.70, bottom=0.245, wspace=0.32)

    labels = ["not potable", "potable"]
    left.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            value = int(matrix[row][column])
            left.text(column, row, f"{value:,}", ha="center", va="center",
                      fontsize=13,
                      color="white" if value > matrix.max() * 0.55 else ps.INK,
                      fontweight="600")
    left.set_xticks([0, 1], labels, fontsize=10)
    left.set_yticks([0, 1], labels, fontsize=10)
    left.set_xlabel("predicted", fontsize=10.4, color=ps.MUTED)
    left.set_ylabel("actual", fontsize=10.4, color=ps.MUTED)
    left.set_title(f"{chosen}, held-out split", fontsize=12, color=ps.INK, pad=10,
                   loc="left")
    for spine in left.spines.values():
        spine.set_visible(False)
    left.tick_params(length=0)

    irreducible = errors["errors_the_optimal_rule_also_makes"]
    fixable = errors["errors_the_optimal_rule_avoids"]
    right.barh([1], [irreducible], color=ps.SLATE, height=0.5,
               label="the optimal rule gets these wrong too")
    right.barh([0], [fixable], color=ps.RED, height=0.5,
               label="the optimal rule gets these right")
    right.text(irreducible + 2, 1, f"{irreducible}", va="center", fontsize=11.5,
               color=ps.INK, fontweight="600")
    right.text(fixable + 2, 0, f"{fixable}", va="center", fontsize=11.5,
               color=ps.INK, fontweight="600")
    right.set_yticks([0, 1], ["in principle\nstill available", "irreducible\non this data"],
                     fontsize=10, color=ps.INK)
    right.set_xlabel("Errors", fontsize=10.6, color=ps.MUTED)
    right.set_xlim(0, irreducible * 1.25)
    right.legend(loc="lower right", frameon=False, fontsize=9.8, labelcolor=ps.MUTED)
    right.set_title(f"Where the {errors['errors']} errors come from", fontsize=12,
                    color=ps.INK, pad=10, loc="left")
    ps.clean(right, grid_axis="x")

    ps.title_block(
        fig, "Most of the mistakes are the data, not the model",
        "The generative process is known, so each error can be checked against "
        "what the optimal rule does with\nthe same row.")

    ps.footnote(fig, [
        f"{errors['share_of_errors_that_are_irreducible'] * 100:.1f} percent of the "
        f"errors are rows the optimal rule also gets wrong. Inside the band where "
        f"the true probability sits",
        f"between {errors['ambiguous_band'][0]} and {errors['ambiguous_band'][1]}, "
        f"the error rate is {errors['error_rate_inside_the_band'] * 100:.1f} percent "
        f"against {errors['error_rate_outside_the_band'] * 100:.1f} outside it. The "
        f"model is also less confident when",
        f"it is wrong, {errors['model_confidence']['mean_on_wrong']} against "
        f"{errors['model_confidence']['mean_on_correct']} measured as distance from "
        f"a half.",
    ], y=0.105)
    ps.save(fig, FIGURES, "03_errors")


def main() -> int:
    ps.apply()
    data = load()
    print(f"  reading {RESULTS.relative_to(ROOT).as_posix()}")
    figure_01_against_the_ceiling(data)
    figure_02_what_the_data_is(data)
    figure_03_errors(data)
    print(f"  figures in {FIGURES.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
