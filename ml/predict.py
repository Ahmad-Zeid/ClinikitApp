"""
Example predictions for Part 2.

These patients are fictional. They show how the saved pipeline accepts a normal clinic
record and returns an estimated probability of missing the appointment.

Run:
    .venv/bin/python ml/train.py
    .venv/bin/python ml/predict.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
    # Works when imported from the project root or notebook.
    from ml.train import EXAMPLE_APPOINTMENTS, MODEL_PATH, prepare_features
except ModuleNotFoundError:
    # Works for the documented command: python ml/predict.py
    from train import EXAMPLE_APPOINTMENTS, MODEL_PATH, prepare_features


def predict_examples(model_path: Path = MODEL_PATH) -> list[dict]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run `.venv/bin/python ml/train.py` first."
        )

    bundle = joblib.load(model_path)
    labels = [example["label"] for example in EXAMPLE_APPOINTMENTS]
    rows = [
        {key: value for key, value in example.items() if key != "label"}
        for example in EXAMPLE_APPOINTMENTS
    ]
    X = prepare_features(pd.DataFrame(rows))
    probabilities = bundle["pipeline"].predict_proba(X)[:, 1]
    threshold = float(bundle["threshold"])

    return [
        {
            "label": label,
            "no_show_probability": float(probability),
            "flagged": bool(probability >= threshold),
        }
        for label, probability in zip(labels, probabilities)
    ]


def main() -> None:
    bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if bundle is None:
        raise SystemExit(
            "No trained model yet. Run: .venv/bin/python ml/train.py"
        )

    threshold = float(bundle["threshold"])
    print("Example predictions (estimated probability of a no-show)\n")
    for result in predict_examples():
        flag = "FLAG" if result["flagged"] else "ok"
        print(
            f"  [{flag:4}] {result['no_show_probability']:5.1%}  "
            f"{result['label']}"
        )
    print(f"\nFlagging threshold: {threshold:.3f}")
    print("These are risk estimates for workflow support, not automatic decisions.")


if __name__ == "__main__":
    main()
