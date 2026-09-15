"""Fast, offline checks for the real-data Part 2 pipeline."""

import pytest

from ml.train import (
    DATA_PATH,
    FEATURES,
    REQUIRED_COLUMNS,
    candidate_models,
    load_frame,
    prepare,
    validate_data,
)


def test_supplied_dataset_is_clean_and_has_the_brief_fields():
    df = load_frame(DATA_PATH)

    assert len(df) == 3_000
    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert df["appointment_id"].is_unique
    assert df.isna().sum().sum() == 0
    assert set(df["no_show"].unique()) == {0, 1}


def test_preparation_drops_identifier_and_target():
    X, y = prepare(load_frame(DATA_PATH).iloc[:100])

    assert list(X.columns) == FEATURES
    assert "appointment_id" not in X
    assert "no_show" not in X
    assert X["past_no_show_rate"].between(0, 1).all()
    assert set(y.unique()).issubset({0, 1})


def test_validation_rejects_impossible_history():
    broken = load_frame(DATA_PATH).iloc[:2].copy()
    broken.loc[broken.index[0], "previous_no_shows"] = (
        broken.loc[broken.index[0], "previous_appointments"] + 1
    )

    with pytest.raises(ValueError, match="more previous no-shows"):
        validate_data(broken)


def test_selected_model_family_can_fit_and_return_probabilities():
    df = load_frame(DATA_PATH).iloc[:500]
    X, y = prepare(df)
    model = candidate_models()["logistic_regression"]

    model.fit(X.iloc[:400], y.iloc[:400])
    probabilities = model.predict_proba(X.iloc[400:])[:, 1]

    assert len(probabilities) == 100
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
