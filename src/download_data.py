"""Download and prepare UCI Heart Disease dataset ID 45."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "heart_disease_raw_with_target.csv"


def main():
    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features.copy()
    targets = dataset.data.targets
    raw = pd.to_numeric(targets.iloc[:, 0], errors="coerce")
    if raw.isna().any():
        raise ValueError("The UCI target contains missing or non-numeric values.")
    df = features.copy()
    df["num"] = raw.astype(int)
    df["target_original"] = raw.astype(int)
    df["target"] = (raw > 0).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved:", OUT)
    print("Shape:", df.shape)
    print("Target counts:", df["target"].value_counts().sort_index().to_dict())
    print("Missing values:", df.isna().sum()[df.isna().sum() > 0].to_dict())


if __name__ == "__main__":
    main()
