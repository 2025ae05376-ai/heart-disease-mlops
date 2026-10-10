from pathlib import Path
import json

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_raw_with_target.csv"
OUTPUT_PATH = ROOT / "reports" / "tuning_results.json"

def main():
    df = pd.read_csv(DATA_PATH)
    y = df["target"].astype(int)
    excluded = {"target", "target_original", "num"}
    X = df.drop(columns=[c for c in excluded if c in df.columns])

    categorical = [
        c for c in [
            "sex", "cp", "fbs", "restecg", "exang",
            "slope", "ca", "thal"
        ] if c in X.columns
    ]
    numerical = [c for c in X.columns if c not in categorical]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    searches = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 2],
                "model__class_weight": ["balanced"],
            },
        ),
    }

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment("heart-disease-classification-linux")

    results = {}

    for name, (estimator, grid) in searches.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=False,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        best = search.best_index_
        result = {
            "model": name,
            "search_method": "GridSearchCV",
            "selection_metric": "mean 5-fold CV ROC-AUC",
            "parameter_combinations_tested": len(search.cv_results_["params"]),
            "best_params": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "best_cv_roc_auc_std": float(
                search.cv_results_["std_test_score"][best]
            ),
        }
        results[name] = result

        with mlflow.start_run(run_name=f"tuning_grid_search_{name}"):
            mlflow.log_param("model_type", name)
            mlflow.log_param("search_method", "GridSearchCV")
            mlflow.log_param("cv_strategy", "StratifiedKFold")
            mlflow.log_param("cv_splits", 5)
            mlflow.log_param("selection_metric", "roc_auc")
            mlflow.log_param(
                "parameter_combinations_tested",
                result["parameter_combinations_tested"]
            )
            for key, value in search.best_params_.items():
                mlflow.log_param(f"best_{key}", value)
            mlflow.log_metric("best_cv_roc_auc", result["best_cv_roc_auc"])
            mlflow.log_metric(
                "best_cv_roc_auc_std", result["best_cv_roc_auc_std"]
            )
            mlflow.log_dict(result, "tuning_results.json")

        print(f"\n{name}")
        print(json.dumps(result, indent=2))

    winner = max(results, key=lambda name: results[name]["best_cv_roc_auc"])
    summary = {
        "note": (
            "Separate tuning experiment. Existing production model and "
            "deployment were not changed. Test split was not used for tuning."
        ),
        "selected_by_cv_roc_auc": winner,
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved tuning summary: {OUTPUT_PATH}")
    print(f"Best CV model in this experiment: {winner}")
    print("Existing saved model was not overwritten.")

if __name__ == "__main__":
    main()
