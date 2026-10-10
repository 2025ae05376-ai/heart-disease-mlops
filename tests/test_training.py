from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_raw_with_target.csv"


def test_dataset_exists_and_has_binary_target():
    assert DATA_PATH.exists(), f"Dataset missing: {DATA_PATH}"

    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0
    assert "target" in df.columns
    assert set(df["target"].dropna().unique()).issubset({0, 1})


def test_training_data_has_expected_features():
    df = pd.read_csv(DATA_PATH)

    required = {
        "age", "sex", "cp", "trestbps", "chol", "thalach", "target"
    }
    assert required.issubset(df.columns)


def test_target_columns_are_not_model_features():
    df = pd.read_csv(DATA_PATH)
    excluded = {"target", "target_original", "num"}
    features = [c for c in df.columns if c not in excluded]

    assert "target" not in features
    assert "target_original" not in features
    assert "num" not in features
    assert len(features) > 0
