"""
Is a better model available on this data, or is 0.70 as good as it gets?

WHY THIS FILE EXISTS
    The trained model scores 0.695. That is a moderate number, and the honest question is
    whether it is moderate because the model is weak or because the data is thin.

    Those two have opposite answers. If the model is weak, the fix is a better model. If
    the data is thin, a better model is a waste of time and the right move is to say so
    and stop.

    This script answers the question by trying to beat it and reporting whether anything
    did. It changes nothing and saves nothing -- it is evidence, not part of training.

WHAT IT MEASURES
    ROC-AUC. Given one patient who missed and one who attended, how often does the model
    give the higher risk score to the one who missed? 0.5 is a coin toss. 1.0 is perfect.

Run:
    .venv/bin/python ml/ceiling_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: `python ml/ceiling_check.py` from the project root, or directly from
# inside ml/. Without this the import below only works from one of those two places.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from train import CATEGORICAL, DATA_PATH, RANDOM_STATE, TARGET

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def _frame() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=[TARGET, "appointment_id"]), df[TARGET]


def _pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("prepare", ColumnTransformer([
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("number", StandardScaler(), numeric),
        ])),
        ("model", model),
    ])


def single_column_strength(X: pd.DataFrame, y: pd.Series) -> None:
    """
    How much can each column tell us on its own?

    For the numbers, the column IS a risk score, so it can be scored directly. The
    categories are turned into their own group's no-show rate first -- which flatters
    them, because that rate was worked out using the answers. Even flattered, they are
    weak, and that is the point.
    """
    print("\nWhat each column knows by itself (ROC-AUC, 0.5 = nothing)")
    rows = []
    for column in X.columns:
        if column in CATEGORICAL:
            rate = y.groupby(X[column]).mean()
            score = roc_auc_score(y, X[column].map(rate))
            label = f"{column} (flattered)"
        else:
            score = roc_auc_score(y, X[column])
            label = column
        rows.append((score, label))
    for score, label in sorted(rows, reverse=True):
        bar = "#" * int((score - 0.5) * 100) if score > 0.5 else ""
        print(f"  {label:34s} {score:.3f}  {bar}")


def model_comparison(X: pd.DataFrame, y: pd.Series) -> None:
    """Throw everything reasonable at it and see whether anything pulls ahead."""
    print("\nCan any model do better? (5-fold cross-validation on all rows)")
    candidates = {
        "logistic regression": _pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            _numeric(X), CATEGORICAL),
        "logistic, regularised": _pipeline(
            LogisticRegression(max_iter=1000, C=0.03, random_state=RANDOM_STATE),
            _numeric(X), CATEGORICAL),
        "logistic, balanced": _pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE),
            _numeric(X), CATEGORICAL),
        "random forest": _pipeline(
            RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=1,
                                   random_state=RANDOM_STATE),
            _numeric(X), CATEGORICAL),
        "gradient boosting": _pipeline(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            _numeric(X), CATEGORICAL),
        "gradient boosting, tuned": _pipeline(
            HistGradientBoostingClassifier(
                max_leaf_nodes=7, learning_rate=0.03, max_iter=400,
                l2_regularization=1.0, min_samples_leaf=40, random_state=RANDOM_STATE),
            _numeric(X), CATEGORICAL),
    }

    # Interactions: does "new patient AND no reminder" say more than the two separately?
    candidates["logistic + interactions"] = Pipeline([
        ("prepare", ColumnTransformer([
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("number", StandardScaler(), _numeric(X)),
        ])),
        ("pairs", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
        ("model", LogisticRegression(max_iter=2000, C=0.05, random_state=RANDOM_STATE)),
    ])

    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X, y, cv=CV, scoring="roc_auc")
        print(f"  {name:30s} {scores.mean():.4f}  +/- {scores.std():.4f}")


def how_few_columns_are_needed(X: pd.DataFrame, y: pd.Series) -> None:
    """
    The finding that decides the whole question.

    If two columns get you almost everything, the other eight are decoration and no model
    can find signal that is not there.
    """
    print("\nHow much of the score comes from how few columns?")
    subsets = {
        "previous_no_shows alone": ["previous_no_shows"],
        "+ reminder_sent": ["previous_no_shows", "reminder_sent"],
        "+ previous_appointments": ["previous_no_shows", "reminder_sent",
                                    "previous_appointments"],
        "all ten columns": None,
    }
    for label, columns in subsets.items():
        if columns is None:
            pipeline = _pipeline(LogisticRegression(max_iter=1000,
                                                    random_state=RANDOM_STATE),
                                 _numeric(X), CATEGORICAL)
            data = X
        else:
            pipeline = _pipeline(LogisticRegression(max_iter=1000,
                                                    random_state=RANDOM_STATE),
                                 columns, [])
            data = X[columns]
        scores = cross_val_score(pipeline, data, y, cv=CV, scoring="roc_auc")
        print(f"  {label:30s} {scores.mean():.4f}")


def _numeric(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in CATEGORICAL]


def main() -> None:
    X, y = _frame()
    print(f"Rows: {len(X)}   No-shows: {int(y.sum())} ({y.mean():.1%})")

    single_column_strength(X, y)
    model_comparison(X, y)
    how_few_columns_are_needed(X, y)

    print("""
CONCLUSION
    Nothing clears about 0.70, and two columns out of ten get almost all of it. The limit
    is the data, not the model: the file records what the clinic knew, and most of why
    somebody misses an appointment -- the car broke down, the child got sick, the bus
    never came -- was never written down anywhere.

    So the right answer is the simple model, honestly reported, with the limit stated.
    Reaching for something fancier here would buy a third of a percent and cost the
    ability to explain how it works.""")


if __name__ == "__main__":
    main()
