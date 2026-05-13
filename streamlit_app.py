from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
DATA_PATH = BASE_DIR / "health_data.csv"
PREDICTION_THRESHOLD = 0.35

FEATURES = [
    "sex",
    "age",
    "smokes",
    "total_cholesterol",
    "fasting_blood_sugar",
    "vigorous_minutes",
    "moderate_minutes",
    "walking_cycling_minutes",
    "sitting_minutes",
    "fruit_servings_per_week",
    "vegetable_servings_per_week",
    "systolic_bp",
    "diastolic_bp",
]


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def build_input_frame(values):
    return pd.DataFrame([{feature: values[feature] for feature in FEATURES}])


st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.title("Heart Disease Predictor")

try:
    model = load_model()
except Exception as exc:  # noqa: BLE001 - visible local app error.
    st.error(f"Could not load model: {exc}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Patient Inputs")

    left, middle, right = st.columns(3)

    with left:
        sex_label = st.selectbox("Sex", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
        smokes_label = st.selectbox("Smokes tobacco or cigarettes", ["No", "Yes"])
        total_cholesterol = st.number_input("Total cholesterol", min_value=0.0, value=145.0, step=1.0)
        fasting_blood_sugar = st.number_input("Fasting blood sugar (mg/dl)", min_value=0.0, value=94.0, step=1.0)

    with middle:
        systolic_bp = st.number_input("Systolic blood pressure (mmHg)", min_value=0.0, value=120.0, step=1.0)
        diastolic_bp = st.number_input("Diastolic blood pressure (mmHg)", min_value=0.0, value=80.0, step=1.0)
        sitting_minutes = st.number_input("Sitting time (minutes per day)", min_value=0.0, value=240.0, step=10.0)
        fruit_servings_per_week = st.number_input("Fruit servings per week", min_value=0.0, value=7.0, step=1.0)
        vegetable_servings_per_week = st.number_input("Vegetable servings per week", min_value=0.0, value=7.0, step=1.0)

    with right:
        vigorous_minutes = st.number_input("Vigorous activity (minutes per week)", min_value=0.0, value=0.0, step=10.0)
        moderate_minutes = st.number_input("Moderate activity (minutes per week)", min_value=0.0, value=0.0, step=10.0)
        walking_cycling_minutes = st.number_input("Walking or cycling (minutes per week)", min_value=0.0, value=0.0, step=10.0)

    submitted = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)

if submitted:
    values = {
        "sex": 1.0 if sex_label == "Male" else 0.0,
        "age": float(age),
        "smokes": 1.0 if smokes_label == "Yes" else 0.0,
        "total_cholesterol": float(total_cholesterol),
        "fasting_blood_sugar": float(fasting_blood_sugar),
        "vigorous_minutes": float(vigorous_minutes),
        "moderate_minutes": float(moderate_minutes),
        "walking_cycling_minutes": float(walking_cycling_minutes),
        "sitting_minutes": float(sitting_minutes),
        "fruit_servings_per_week": float(fruit_servings_per_week),
        "vegetable_servings_per_week": float(vegetable_servings_per_week),
        "systolic_bp": float(systolic_bp),
        "diastolic_bp": float(diastolic_bp),
    }

    probability = float(model.predict_proba(build_input_frame(values))[0][1])
    label = "Higher predicted risk" if probability >= PREDICTION_THRESHOLD else "Lower predicted risk"

    st.divider()
    result_cols = st.columns([1, 2])
    result_cols[0].metric("Predicted Risk", f"{probability * 100:.1f}%")
    result_cols[1].subheader(label)
    result_cols[1].progress(min(probability, 1.0))
    result_cols[1].warning(
        "This model is for screening/demo use only. It was trained on an imbalanced dataset and is not a clinical diagnosis."
    )
