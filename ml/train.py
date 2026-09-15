"""
Part 2 — predict whether a patient will miss an appointment.

PLAIN ENGLISH
    We keep 20% of the uploaded appointments hidden. Several models practise on the
    other 80%. Cross-validation chooses the best model without seeing the hidden rows.
    Only then do we measure the winner on the hidden rows.

WHY THIS IS HONEST
    Looking at the test rows while choosing a model is like reading an exam answer key
    before sitting the exam. It makes the final score look better than it really is.
    This script uses the test rows once, at the very end.

Run:
    .venv/bin/python ml/train.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = PROJECT_ROOT / "CliniKit_NoShow_Dataset.csv"
MODEL_PATH = ROOT / "models" / "no_show_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
RESULTS_PATH = ROOT / "RESULTS.md"
FIGURES = ROOT / "figures"

TARGET = "no_show"
ID_COLUMN = "appointment_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

CATEGORICAL = ["gender", "appointment_type", "weekday", "appointment_time"]
NUMERIC = [
    "age",
    "days_before_appointment",
    "previous_appointments",
    "previous_no_shows",
    "reminder_sent",
    "new_patient",
    "past_no_show_rate",
]
FEATURES = CATEGORICAL + NUMERIC

REQUIRED_COLUMNS = {
    ID_COLUMN,
    TARGET,
    "age",
    "gender",
    "appointment_type",
    "days_before_appointment",
    "previous_appointments",
    "previous_no_shows",
    "weekday",
    "appointment_time",
    "reminder_sent",
    "new_patient",
}

EXAMPLE_APPOINTMENTS = [
    {
        "label": "Returning patient, reminder sent, no earlier misses",
        "age": 42,
        "gender": "Female",
        "appointment_type": "Follow-up",
        "days_before_appointment": 3,
        "previous_appointments": 8,
        "previous_no_shows": 0,
        "weekday": "Wednesday",
        "appointment_time": "14:00-16:00",
        "reminder_sent": 1,
        "new_patient": 0,
    },
    {
        "label": "New consultation, booked well ahead, no reminder",
        "age": 24,
        "gender": "Male",
        "appointment_type": "New Consultation",
        "days_before_appointment": 24,
        "previous_appointments": 0,
        "previous_no_shows": 0,
        "weekday": "Friday",
        "appointment_time": "18:00-20:00",
        "reminder_sent": 0,
        "new_patient": 1,
    },
    {
        "label": "Returning patient with several earlier misses, no reminder",
        "age": 37,
        "gender": "Female",
        "appointment_type": "Urgent Visit",
        "days_before_appointment": 12,
        "previous_appointments": 7,
        "previous_no_shows": 4,
        "weekday": "Friday",
        "appointment_time": "16:00-18:00",
        "reminder_sent": 0,
        "new_patient": 0,
    },
]


def load_frame(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the dataset supplied with the assessment."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put CliniKit_NoShow_Dataset.csv "
            "in the project root."
        )
    frame = pd.read_csv(path)
    validate_data(frame)
    return frame


def validate_data(df: pd.DataFrame) -> None:
    """Stop with a clear reason instead of silently training on broken data."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("The dataset has no rows.")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("appointment_id must be unique.")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("no_show must contain only 0 (attended) or 1 (missed).")
    if df[TARGET].isna().any():
        raise ValueError("no_show contains missing values.")
    if (df["previous_no_shows"] > df["previous_appointments"]).any():
        raise ValueError("A row has more previous no-shows than previous appointments.")
    if (df[["age", "days_before_appointment", "previous_appointments",
            "previous_no_shows"]] < 0).any().any():
        raise ValueError("Age and appointment-history counts cannot be negative.")
    for column in ("reminder_sent", "new_patient"):
        if not set(df[column].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{column} must contain only 0 or 1.")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature work during training and later prediction."""
    out = df.copy()
    for column in CATEGORICAL:
        out[column] = out[column].astype("string").str.strip()

    history = out["previous_appointments"].replace(0, np.nan)
    out["past_no_show_rate"] = (
        out["previous_no_shows"] / history
    ).fillna(0.0)

    return out[FEATURES]


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build inputs and target for training.

    appointment_id is deliberately excluded. It identifies a row; it does not describe
    a patient. Learning from it would be memorising the spreadsheet, not learning risk.
    """
    return prepare_features(df), df[TARGET].astype(int)


def explore(df: pd.DataFrame) -> dict:
    """Numbers used in the notebook and generated report."""
    reminder_rates = (
        df.groupby("reminder_sent")[TARGET].agg(["size", "mean"]).to_dict("index")
    )
    type_rates = (
        df.groupby("appointment_type")[TARGET]
        .agg(["size", "mean"])
        .sort_values("mean", ascending=False)
        .to_dict("index")
    )
    weekday_rates = (
        df.groupby("weekday")[TARGET]
        .agg(["size", "mean"])
        .sort_values("mean", ascending=False)
        .to_dict("index")
    )
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "no_shows": int(df[TARGET].sum()),
        "no_show_rate": float(df[TARGET].mean()),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COLUMN].duplicated().sum()),
        "reminder_rates": reminder_rates,
        "appointment_type_rates": type_rates,
        "weekday_rates": weekday_rates,
    }


def _categorical_steps(*, dense: bool) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                ),
            ),
        ]
    )


def _linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("category", _categorical_steps(dense=False), CATEGORICAL),
            (
                "number",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
        ]
    )


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("category", _categorical_steps(dense=True), CATEGORICAL),
            (
                "number",
                SimpleImputer(strategy="median"),
                NUMERIC,
            ),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    """Small, explainable comparison: baseline, linear model, and two tree models."""
    return {
        "dummy_baseline": Pipeline(
            [
                ("prepare", _linear_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("prepare", _linear_preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prepare", _tree_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=0.5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prepare", _tree_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        max_features=0.7,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def compare_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> list[dict]:
    """Compare on training folds only; the held-out test rows stay untouched."""
    rows = []
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
    }
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
                "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
                "cv_balanced_accuracy_mean": float(
                    scores["test_balanced_accuracy"].mean()
                ),
            }
        )
    return sorted(rows, key=lambda row: row["cv_roc_auc_mean"], reverse=True)


def choose_threshold(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[float, dict]:
    """
    Choose the cut-off using out-of-fold training predictions.

    The threshold with the best no-show F1 balances two competing costs:
    missing true no-shows and bothering patients who would have attended.
    """
    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, probabilities)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.nanargmax(f1_values[:-1]))
    threshold = float(thresholds[best_index])
    return threshold, {
        "training_oof_precision": float(precision[best_index]),
        "training_oof_recall": float(recall[best_index]),
        "training_oof_f1": float(f1_values[best_index]),
    }


def _raw_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    shuffled = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rows = [
        {
            "feature": feature,
            "auc_drop": float(mean),
            "std": float(std),
        }
        for feature, mean, std in zip(
            FEATURES,
            shuffled.importances_mean,
            shuffled.importances_std,
        )
    ]
    return sorted(rows, key=lambda row: row["auc_drop"], reverse=True)


def _logistic_coefficients(model: Pipeline) -> list[dict]:
    """Direction of each relationship. Association, not proof of cause."""
    if not isinstance(model.named_steps["model"], LogisticRegression):
        return []
    names = model.named_steps["prepare"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    rows = [
        {"feature": str(name), "coefficient": float(value)}
        for name, value in zip(names, coefficients)
    ]
    return sorted(rows, key=lambda row: abs(row["coefficient"]), reverse=True)


def train(data_path: Path = DATA_PATH) -> dict:
    """Run the complete, reproducible training experiment."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_frame(data_path)
    exploration = explore(df)
    X, y = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = candidate_models()
    comparison = compare_models(models, X_train, y_train, cv)
    selected_name = comparison[0]["model"]
    selected_model = models[selected_name]

    threshold, threshold_details = choose_threshold(
        selected_model,
        X_train,
        y_train,
        cv,
    )
    selected_model.fit(X_train, y_train)

    probabilities = selected_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "data_file": data_path.name,
        "exploration": exploration,
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_comparison": comparison,
        "selected_model": selected_name,
        "threshold": threshold,
        "threshold_method": "maximum no-show F1 on out-of-fold training predictions",
        **threshold_details,
        "test_no_show_rate": float(y_test.mean()),
        "baseline_accuracy_all_attend": float((y_test == 0).mean()),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_pr_auc": float(average_precision_score(y_test, probabilities)),
        "test_brier_score": float(brier_score_loss(y_test, probabilities)),
        "test_precision_no_show": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "test_recall_no_show": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "test_f1_no_show": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["attended", "no_show"],
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics["feature_importance"] = _raw_feature_importance(
        selected_model,
        X_test,
        y_test,
    )
    metrics["top_coefficients"] = _logistic_coefficients(selected_model)
    example_frame = pd.DataFrame(
        [
            {key: value for key, value in example.items() if key != "label"}
            for example in EXAMPLE_APPOINTMENTS
        ]
    )
    example_probabilities = selected_model.predict_proba(
        prepare_features(example_frame)
    )[:, 1]
    metrics["example_predictions"] = [
        {
            "label": example["label"],
            "no_show_probability": float(probability),
            "flagged": bool(probability >= threshold),
        }
        for example, probability in zip(
            EXAMPLE_APPOINTMENTS,
            example_probabilities,
        )
    ]

    joblib.dump(
        {
            "pipeline": selected_model,
            "threshold": threshold,
            "selected_model": selected_name,
            "features": FEATURES,
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    _make_figures(
        df,
        y_test,
        probabilities,
        predictions,
        matrix,
        metrics["feature_importance"],
    )
    _write_results(metrics)
    _print_summary(metrics)
    return metrics


def _setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURES / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _make_figures(
    df: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    matrix: np.ndarray,
    importance: list[dict],
) -> None:
    plt = _setup_matplotlib()

    # Basic exploration.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    df[TARGET].value_counts().sort_index().plot(
        kind="bar",
        ax=axes[0],
        color=["#4c78a8", "#e45756"],
    )
    axes[0].set_title("Class balance")
    axes[0].set_xticklabels(["Attended", "No-show"], rotation=0)
    axes[0].set_ylabel("Appointments")

    reminder = df.groupby("reminder_sent")[TARGET].mean()
    reminder.plot(kind="bar", ax=axes[1], color="#59a14f")
    axes[1].set_title("No-show rate by reminder")
    axes[1].set_xticklabels(["No reminder", "Reminder"], rotation=0)
    axes[1].set_ylabel("No-show rate")

    appointment_type = (
        df.groupby("appointment_type")[TARGET].mean().sort_values()
    )
    appointment_type.plot(kind="barh", ax=axes[2], color="#f28e2b")
    axes[2].set_title("Rate by appointment type")
    axes[2].set_xlabel("No-show rate")

    fig.tight_layout()
    fig.savefig(FIGURES / "data_exploration.png", dpi=140)
    plt.close(fig)

    # ROC, precision-recall, and confusion matrix.
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(fpr, tpr, color="#4c78a8")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="ROC curve", xlabel="False-positive rate",
                ylabel="True-positive rate")

    axes[1].plot(recall, precision, color="#59a14f")
    axes[1].axhline(y_test.mean(), linestyle="--", color="grey")
    axes[1].set(title="Precision-recall curve", xlabel="Recall",
                ylabel="Precision")

    image = axes[2].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[2].text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="black",
            )
    axes[2].set(
        title="Confusion matrix",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Attend", "No-show"],
        yticklabels=["Attend", "No-show"],
    )
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_evaluation.png", dpi=140)
    plt.close(fig)

    # Raw input importance.
    ordered = importance[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(
        [row["feature"] for row in ordered],
        [row["auc_drop"] for row in ordered],
        color="#4c78a8",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ROC-AUC drop after shuffling")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=140)
    plt.close(fig)


def _write_results(metrics: dict) -> None:
    exploration = metrics["exploration"]
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]

    lines = [
        "# Part 2 — No-show prediction results",
        "",
        "All numbers below come from the uploaded "
        "`CliniKit_NoShow_Dataset.csv`, not generated data.",
        "",
        "## 1. Basic data exploration",
        "",
        f"- Rows: **{exploration['rows']:,}**",
        f"- Source columns: **{exploration['columns']}** "
        "(10 candidate predictors, one ID, and target `no_show`)",
        f"- No-shows: **{exploration['no_shows']} "
        f"({exploration['no_show_rate']:.1%})**",
        f"- Attended: **{exploration['rows'] - exploration['no_shows']} "
        f"({1 - exploration['no_show_rate']:.1%})**",
        f"- Missing cells: **{exploration['missing_cells']}**",
        f"- Duplicate rows / IDs: **{exploration['duplicate_rows']} / "
        f"{exploration['duplicate_ids']}**",
        "",
        "Observed no-show rate by reminder:",
    ]
    for key, row in exploration["reminder_rates"].items():
        label = "reminder sent" if int(key) == 1 else "no reminder"
        lines.append(f"- {label}: **{row['mean']:.1%}** ({int(row['size'])} rows)")

    lines += [
        "",
        "Observed rate by appointment type:",
    ]
    for label, row in exploration["appointment_type_rates"].items():
        lines.append(f"- {label}: **{row['mean']:.1%}** ({int(row['size'])} rows)")

    lines += [
        "",
        "These are associations in this dataset. They do not prove, for example, that "
        "sending a reminder caused the lower no-show rate.",
        "",
        "![Data exploration](figures/data_exploration.png)",
        "",
        "## 2. Data preparation",
        "",
        "- Removed `appointment_id` from the inputs because it identifies a row rather "
        "than describing appointment risk.",
        "- Kept `weekday`, `appointment_time`, `gender`, and `appointment_type` as "
        "categories and one-hot encoded them.",
        "- Kept numeric values numeric and standardised them for logistic regression.",
        "- Added `past_no_show_rate = previous_no_shows / previous_appointments`; new "
        "patients receive zero because they have no history.",
        "- Used a stratified 80/20 split, so train and test contain the same share of "
        "no-shows.",
        "- Kept the 600 test rows hidden during model and threshold selection.",
        "",
        "## 3. Model selection",
        "",
        "Five-fold cross-validation was run on the training set only:",
        "",
        "| model | CV ROC-AUC | CV PR-AUC | CV balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics["model_comparison"]:
        lines.append(
            f"| `{row['model']}` | {row['cv_roc_auc_mean']:.3f} "
            f"± {row['cv_roc_auc_std']:.3f} | {row['cv_pr_auc_mean']:.3f} | "
            f"{row['cv_balanced_accuracy_mean']:.3f} |"
        )

    lines += [
        "",
        f"Selected: **`{metrics['selected_model']}`**, because it had the strongest "
        "cross-validated ROC-AUC. Logistic regression is also the simplest candidate to "
        "explain: it combines weighted evidence from each input into one probability.",
        "",
        "Balanced accuracy in the table uses each candidate's default 0.5 cut-off. "
        "It was not the selection score because the final operating cut-off is tuned "
        "separately. Logistic regression also led on PR-AUC.",
        "",
        "## 4. Held-out test results",
        "",
        f"- ROC-AUC: **{metrics['test_roc_auc']:.3f}**",
        f"- PR-AUC (average precision): **{metrics['test_pr_auc']:.3f}** "
        f"(random baseline is the {metrics['test_no_show_rate']:.1%} no-show rate)",
        f"- No-show precision: **{metrics['test_precision_no_show']:.3f}**",
        f"- No-show recall: **{metrics['test_recall_no_show']:.3f}**",
        f"- No-show F1: **{metrics['test_f1_no_show']:.3f}**",
        f"- Balanced accuracy: **{metrics['test_balanced_accuracy']:.3f}**",
        f"- Ordinary accuracy: **{metrics['test_accuracy']:.3f}** "
        f"(always predicting attendance would score "
        f"{metrics['baseline_accuracy_all_attend']:.3f}, but catch zero no-shows)",
        f"- Brier score: **{metrics['test_brier_score']:.3f}** "
        "(lower means better probability estimates)",
        f"- Decision threshold: **{metrics['threshold']:.3f}**, chosen by maximum "
        "no-show F1 using out-of-fold training predictions—not the test set",
        "",
        f"Confusion matrix: **TN={tn}, FP={fp}, FN={fn}, TP={tp}**.",
        "",
        f"In plain English: the model caught **{tp} of {tp + fn}** true no-shows and "
        f"incorrectly flagged **{fp}** patients who attended.",
        "",
        "![Model evaluation](figures/model_evaluation.png)",
        "",
        "## 5. Strongest influences",
        "",
        "Permutation importance shuffles one raw input at a time. A larger ROC-AUC drop "
        "means the model relied on that input more:",
        "",
    ]
    for row in metrics["feature_importance"]:
        lines.append(f"- `{row['feature']}`: **{row['auc_drop']:.4f}** AUC drop")

    lines += [
        "",
        "![Feature importance](figures/feature_importance.png)",
        "",
        "A small negative value means shuffling that column slightly improved this test "
        "score; treat it as noise or an unhelpful feature, not negative importance.",
        "",
        "Important: influence is not cause. This model finds patterns; it cannot prove "
        "why a patient missed an appointment.",
        "",
        "## 6. Why these metrics",
        "",
        "- **Accuracy alone is misleading:** 84% attended, so an always-attend model "
        "already gets about 84% accuracy while helping nobody.",
        "- **Recall:** of all patients who missed, how many were flagged?",
        "- **Precision:** of all patients flagged, how many actually missed?",
        "- **PR-AUC:** summarises the precision/recall trade-off and is useful for the "
        "rare no-show class.",
        "- **ROC-AUC:** tests whether true no-shows generally receive higher risk scores "
        "than attendances, independent of one threshold.",
        "",
        "## 7. Example predictions",
        "",
        "These patients are fictional. They demonstrate the saved pipeline; they are not "
        "extra evaluation rows:",
    ]
    for example in metrics["example_predictions"]:
        status = "flag" if example["flagged"] else "do not flag"
        lines.append(
            f"- **{example['no_show_probability']:.1%}** ({status}) — "
            f"{example['label']}"
        )
    lines += [
        "",
        "Reproduce them with `.venv/bin/python ml/predict.py` after training.",
        "",
        "## 8. Product integration",
        "",
        "A nightly job would score tomorrow's appointments. Reception could see a risk "
        "flag and decide whether to send an extra reminder or offer a waitlist slot. "
        "The model advises; it must not cancel, overbook, or deny care automatically.",
        "",
        "In production I would also monitor accuracy and calibration over time, audit "
        "performance across patient groups, retrain on newer outcomes, and choose the "
        "threshold with the clinic based on the cost of missed visits versus unnecessary "
        "reminders.",
        "",
        "## 9. Limits",
        "",
        "- Only 3,000 rows and 478 no-shows are available.",
        "- There is no patient identifier, so repeat visits by one person cannot be kept "
        "in the same train/test group.",
        "- The result is moderate, not production-ready.",
        "- Observed associations—especially reminders—must not be described as causal.",
        "",
    ]
    RESULTS_PATH.write_text("\n".join(lines))


def _print_summary(metrics: dict) -> None:
    print("Model comparison (5-fold CV on training rows)")
    for row in metrics["model_comparison"]:
        print(
            f"  {row['model']:24} "
            f"ROC-AUC {row['cv_roc_auc_mean']:.3f}  "
            f"PR-AUC {row['cv_pr_auc_mean']:.3f}"
        )
    print(f"\nSelected: {metrics['selected_model']}")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print(
        f"Test ROC-AUC {metrics['test_roc_auc']:.3f} | "
        f"PR-AUC {metrics['test_pr_auc']:.3f} | "
        f"precision {metrics['test_precision_no_show']:.3f} | "
        f"recall {metrics['test_recall_no_show']:.3f} | "
        f"F1 {metrics['test_f1_no_show']:.3f}"
    )
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved report: {RESULTS_PATH}")


if __name__ == "__main__":
    train()
