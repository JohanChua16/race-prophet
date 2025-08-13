import joblib
import pandas as pd
from pathlib import Path
from .data_load import load_race_dataframe

MODEL_PATH = Path("models/top10_logreg.joblib")

def _ensure_model():
    if MODEL_PATH.exists():
        return
    # Train a small model on the fly if not present (Cloud first run)
    from .train import train
    # small, self-contained year range for a quick bootstrap
    train(year_start=2023, year_end=2023)

def predict_event_top10(year: int, event_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame with per-driver Top-10 probability for the given event,
    using only pre-race features (grid + driver_points_before).
    """
    _ensure_model()
    model = joblib.load(MODEL_PATH)

    df = load_race_dataframe(year, event_name)
    X = df[["grid_position", "driver_points_before", "team", "code"]]
    proba = model.predict_proba(X)[:, 1]  # P(top10==1)

    out = df[["code", "team", "grid_position", "final_position"]].copy()
    out["top10_prob"] = proba
    return out.sort_values("top10_prob", ascending=False).reset_index(drop=True)
