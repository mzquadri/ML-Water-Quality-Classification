"""
End-to-end ML classification pipeline for water potability prediction.

Pipeline:
1. Data loading and preprocessing (imputation, scaling)
2. Exploratory visualizations
3. Model training: Logistic Regression, Random Forest, XGBoost, SVM
4. Hyperparameter tuning (GridSearchCV)
5. Evaluation: accuracy, precision, recall, F1, ROC-AUC
6. Feature importance analysis
7. Professional result visualization
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    f1_score,
)
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

FEATURE_COLS = [
    "ph",
    "hardness",
    "solids",
    "chloramines",
    "sulfate",
    "conductivity",
    "organic_carbon",
    "trihalomethanes",
    "turbidity",
]
TARGET = "potability"


def run_pipeline(csv_path: str, output_dir: str = "../results"):
    """Run the full ML classification pipeline."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df[TARGET].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()}\n")

    X = df[FEATURE_COLS]
    y = df[TARGET]

    # ============================================================
    # 2. EDA VISUALIZATIONS
    # ============================================================

    # 2a. Feature distributions by class
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for idx, col in enumerate(FEATURE_COLS):
        ax = axes[idx // 3, idx % 3]
        for label, color in [(0, "#F44336"), (1, "#4CAF50")]:
            subset = df[df[TARGET] == label][col].dropna()
            ax.hist(
                subset,
                bins=40,
                alpha=0.6,
                color=color,
                label=f"{'Potable' if label else 'Not Potable'}",
                edgecolor="white",
            )
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Feature Distributions by Potability Class",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "01_feature_distributions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: 01_feature_distributions.png")

    # 2b. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[FEATURE_COLS + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: 02_correlation_matrix.png")

    # ============================================================
    # 3. PREPROCESSING & SPLIT
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # ============================================================
    # 4. DEFINE MODELS
    # ============================================================
    models = {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric="logloss",
                        verbosity=0,
                    ),
                ),
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
    }

    # ============================================================
    # 5. TRAIN & EVALUATE ALL MODELS
    # ============================================================
    results = {}
    all_predictions = {}

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit on full train set
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
        }
        all_predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}

        print(f"  Test Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # ============================================================
    # 6. HYPERPARAMETER TUNING (Best model — XGBoost)
    # ============================================================
    print("\nHyperparameter tuning for XGBoost...")
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1, 0.2],
    }
    grid_search = GridSearchCV(
        models["XGBoost"],
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")

    # Evaluate tuned model
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    y_proba_tuned = best_model.predict_proba(X_test)[:, 1]

    results["XGBoost (Tuned)"] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_tuned)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_tuned)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba_tuned)), 4),
        "best_params": grid_search.best_params_,
    }
    all_predictions["XGBoost (Tuned)"] = {
        "y_pred": y_pred_tuned,
        "y_proba": y_proba_tuned,
    }

    # Save best model
    joblib.dump(best_model, os.path.join(output_dir, "best_model_xgboost.pkl"))

    # ============================================================
    # 7. VISUALIZATIONS
    # ============================================================

    # 7a. Model Comparison Bar Chart
    model_names = list(results.keys())
    metrics_to_plot = ["accuracy", "f1_score", "roc_auc"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, metric in enumerate(metrics_to_plot):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=colors[i],
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classification Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "03_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved: 03_model_comparison.png")

    # 7b. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 8))
    colors_roc = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for idx, (name, preds) in enumerate(all_predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, preds["y_proba"])
        auc_val = roc_auc_score(y_test, preds["y_proba"])
        ax.plot(
            fpr,
            tpr,
            color=colors_roc[idx % len(colors_roc)],
            linewidth=2,
            label=f"{name} (AUC={auc_val:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_roc_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved: 04_roc_curves.png")

    # 7c. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, name in enumerate(["Random Forest", "XGBoost (Tuned)", "SVM (RBF)"]):
        ax = axes[idx]
        cm = confusion_matrix(y_test, all_predictions[name]["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Not Potable", "Potable"],
            yticklabels=["Not Potable", "Potable"],
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "05_confusion_matrices.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: 05_confusion_matrices.png")

    # 7d. Feature Importance (XGBoost)
    xgb_model = best_model.named_steps["clf"]
    importance = xgb_model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], color="#673AB7")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([FEATURE_COLS[i].replace("_", " ").title() for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "06_feature_importance.png"), dpi=150)
    plt.close(fig)
    print("Saved: 06_feature_importance.png")

    # 7e. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, (name, preds) in enumerate(all_predictions.items()):
        prec, rec, _ = precision_recall_curve(y_test, preds["y_proba"])
        ax.plot(
            rec, prec, color=colors_roc[idx % len(colors_roc)], linewidth=2, label=name
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "07_precision_recall.png"), dpi=150)
    plt.close(fig)
    print("Saved: 07_precision_recall.png")

    # ============================================================
    # 8. SAVE RESULTS
    # ============================================================
    with open(os.path.join(output_dir, "model_results.json"), "w") as f:
        # Convert numpy types for JSON serialization
        serializable = {}
        for k, v in results.items():
            serializable[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    serializable[k][kk] = {
                        kkk: (int(vvv) if isinstance(vvv, np.integer) else vvv)
                        for kkk, vvv in vv.items()
                    }
                else:
                    serializable[k][kk] = vv
        json.dump(serializable, f, indent=2)

    # Classification report for best model
    report = classification_report(
        y_test, y_pred_tuned, target_names=["Not Potable", "Potable"]
    )
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write("XGBoost (Tuned) — Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"\n{'=' * 50}")
    print("FINAL MODEL COMPARISON")
    print("=" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "water_quality.csv"
    )
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    run_pipeline(csv_path=data_path, output_dir=results_dir)
