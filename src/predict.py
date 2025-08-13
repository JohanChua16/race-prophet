import joblib
import pandas as pd
from pathlib import Path
from .data_load import load_race_dataframe

MODEL_PATH = Path("models/top10_logreg.joblib")

def predict_event_top10(year: int, event_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame with per-driver Top-10 probability for the given event,
    using only pre-race features (grid + driver_points_before).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run: python -m src.train")

    model = joblib.load(MODEL_PATH)
    df = load_race_dataframe(year, event_name)

    # Keep the feature columns the model expects
    X = df[["grid_position", "driver_points_before", "team", "code"]]
    proba = model.predict_proba(X)[:, 1]  # P(top10==1)

    out = df[["code", "team", "grid_position", "final_position"]].copy()
    out["top10_prob"] = proba
    return out.sort_values("top10_prob", ascending=False).reset_index(drop=True)
