import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # <-- add this
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

from .data_load import build_dataset

NUMERICS = ["grid_position", "driver_points_before"]
CATEGORICAL = ["team", "code"]

def train(save_dir: str = "models", year_start: int = 2018, year_end: int = 2025):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    df = build_dataset(year_start, year_end).dropna(subset=["grid_position"])

    X = df[NUMERICS + CATEGORICAL]
    y = df["top10"]

    # 🔧 add imputers so NaNs don’t break LogisticRegression
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERICS),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
        ]
    )

    clf = Pipeline(steps=[
        ("pre", pre),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    score = clf.score(Xte, yte)

    joblib.dump(clf, f"{save_dir}/top10_logreg.joblib")
    pd.Series({"test_accuracy": float(score)}).to_json(f"{save_dir}/metrics.json")
    print(f"Saved model to {save_dir}/top10_logreg.joblib (test acc={score:.3f})")

if __name__ == "__main__":
    # keep 2023 for speed during setup
    train(year_start=2023, year_end=2023)
