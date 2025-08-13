import streamlit as st
import pandas as pd
from src.data_load import get_event_schedule
from src.predict import predict_event_top10

st.set_page_config(page_title="Race Prophet", page_icon="🏎️", layout="wide")
st.title("Race Prophet — Top 10 Finish Probability")

# --- Caching for smoother UX (and fewer API calls on first load) ---
@st.cache_data(show_spinner=False, ttl=60*30)
def cached_schedule(year: int) -> pd.DataFrame:
    return get_event_schedule(year)

@st.cache_data(show_spinner=True, ttl=60*30)
def cached_predict(year: int, event_name: str) -> pd.DataFrame:
    return predict_event_top10(year, event_name)

with st.sidebar:
    st.header("Select Event")
    year = st.number_input("Season", min_value=2018, max_value=2025, value=2023, step=1)
    schedule = cached_schedule(int(year))
    options = schedule["EventName"].tolist()
    # Default to a well-known race if it exists; else first event
    default_event = "Bahrain Grand Prix" if "Bahrain Grand Prix" in options else (options[0] if options else "")
    event_name = st.selectbox("Grand Prix", options, index=options.index(default_event) if default_event in options else 0)

# --- Auto-run on load for instant demo (great for recruiters) ---
st.caption("This app predicts each driver's probability of finishing in the top 10 using simple, explainable features (grid position & pre-race points).")

run_now = st.session_state.get("auto_run_done") is None
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict Top 10 Probabilities") or run_now:
        try:
            preds = cached_predict(int(year), event_name)
            # Nice human labels
            preds_display = preds.rename(columns={
                "code": "Driver",
                "team": "Team",
                "grid_position": "Grid",
                "final_position": "Actual Finish",
                "top10_prob": "Top-10 Probability"
            })
            st.dataframe(preds_display, use_container_width=True)

            # Simple bar chart of probabilities (top 10 rows)
            st.subheader("Top-10 finish probability (highest to lowest)")
            chart_df = preds_display[["Driver", "Top-10 Probability"]].head(10).set_index("Driver")
            st.bar_chart(chart_df)

            st.session_state["auto_run_done"] = True
        except Exception as e:
            st.error(str(e))

with col2:
    st.subheader("How this works")
    st.markdown(
        "- **Data**: FastF1 sessions + pre-race driver standings\n"
        "- **Model**: Logistic Regression (balanced)\n"
        "- **Features**: Grid position, pre-race points\n"
        "- **Output**: Probability of finishing in the **Top 10**"
    )
    st.subheader("Train the model (CLI)")
    st.code("python -m src.train", language="bash")
    st.caption("Train once locally; the saved model in /models powers predictions.")
