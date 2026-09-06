"""The model definitions, in one place so two callers cannot drift apart.

`src/pipeline.py` and `src/benchmark.py` both train these. Keeping a second copy
in either file would let the reference run and the verified run diverge silently,
which is the sort of difference that only shows up as an unexplained gap between
two numbers in the same repository.

Each estimator is a scikit-learn Pipeline with its imputer and scaler inside it.
That placement is what keeps them honest: fitted on the training fold during
cross-validation and on the training split for the final fit, never on data they
are later scored against.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "ph", "hardness", "solids", "chloramines", "sulfate",
    "conductivity", "organic_carbon", "trihalomethanes", "turbidity",
]
TARGET = "potability"
SEED = 42


def build_models(seed: int = SEED) -> dict:
    """The four classifiers the reference run compares, unchanged."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=seed,
                                           n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=seed, eval_metric="logloss",
                                  verbosity=0)),
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]),
    }


#: The grid the reference run searched, transcribed exactly. Only XGBoost was
#: tuned, which is worth stating whenever its result is compared with the others.
#:
#: The search selected max_depth 4, the smallest value offered. A parameter chosen
#: at the edge of its grid is a sign the grid may not contain the optimum, so the
#: tuned result should be read as the best of what was searched rather than the
#: best available.
XGBOOST_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [4, 6, 8],
    "clf__learning_rate": [0.05, 0.1, 0.2],
}
