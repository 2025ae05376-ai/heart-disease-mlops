from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_raw_with_target.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        raise ValueError("Dataset must contain the binary 'target' column.")

    y = df["target"].astype(int)
    excluded = {"target", "target_original", "num"}
    X = df.drop(columns=[c for c in excluded if c in df.columns])

    categorical = [
        c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        if c in X.columns
    ]
    numerical = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    }

    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, prob))
        }
        results[name] = metrics
        fitted[name] = pipeline
        print(f"\n{name}:")
        print(json.dumps(metrics, indent=2))

    # Choose the final model using training-set cross-validation, not test results.
    winner = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(fitted[winner], model_path)

    summary = {
        "selected_model": winner,
        "selection_metric": "mean 5-fold cross-validation ROC-AUC",
        "models": results
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSelected model: {winner}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {MODEL_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
