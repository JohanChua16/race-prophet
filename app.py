import streamlit as st
from src.data_load import get_event_schedule
from src.predict import predict_event_top10

st.set_page_config(page_title="Race Prophet", page_icon="🏎️", layout="wide")
st.title("Race Prophet — Top 10 Finish Probability")

with st.sidebar:
    st.header("Select Event")
    year = st.number_input("Season", min_value=2018, max_value=2025, value=2023, step=1)
    schedule = get_event_schedule(int(year))
    options = schedule["EventName"].tolist()
    event_name = st.selectbox("Grand Prix", options)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict Top 10 Probabilities"):
        try:
            preds = predict_event_top10(int(year), event_name)
            st.dataframe(preds, use_container_width=True)
        except Exception as e:
            st.error(str(e))

with col2:
    st.subheader("How to train (CLI)")
    st.code("python -m src.train", language="bash")
    st.caption("Train once; predictions will use the saved model in /models.")
