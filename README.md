# Heart Disease Predictor

Live app: `https://heartdiseases-fa4sj2wwcfsymoinxgxg73.streamlit.app/`

This project predicts heart disease risk using an XGBoost model trained on health, lifestyle, and blood pressure data. The app is built with Streamlit.

## Features

- Streamlit web interface
- XGBoost classification model
- Median imputation for missing values
- Feature scaling inside the saved model pipeline
- Class weighting to reduce bias toward the majority class
- Threshold-based risk prediction

## Project Structure

```text
.
├── health_data.csv
├── streamlit_app.py
├── model_training.py
├── model_evaluation.py
├── requirements.txt
├── xgb_model.pkl
├── test_data.pkl
├── feature_schema.pkl
└── README.md
```

## Dataset

The current dataset is `health_data.csv`.

The target column is `Y`:

- `0`: no heart disease
- `1`: heart disease

The dataset is highly imbalanced:

- No heart disease: `5480`
- Heart disease: `107`

Because of this imbalance, accuracy alone is not a good metric for this project.

## Model Inputs

The model uses these inputs:

- Sex
- Age
- Smoking status
- Total cholesterol
- Fasting blood sugar
- Vigorous activity minutes
- Moderate activity minutes
- Walking or cycling minutes
- Sitting time
- Fruit servings per week
- Vegetable servings per week
- Systolic blood pressure
- Diastolic blood pressure

## Training

Train the model with:

```bash
python3 model_training.py
```

This creates:

- `xgb_model.pkl`
- `test_data.pkl`
- `feature_schema.pkl`

## Evaluation

Evaluate the model with:

```bash
python3 model_evaluation.py
```

Current evaluation summary:

```text
ROC-AUC: 0.6855
PR-AUC: 0.1336
```

At threshold `0.35`, the model catches more positive heart disease cases, but it also produces many false positives. This is expected because the positive class is rare.

## Run Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the Streamlit app:

```bash
python3 -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```


```text
streamlit_app.py
```

## Important Note

This project is for educational and demonstration purposes only. It is not a medical diagnosis tool. Anyone concerned about heart disease should consult a qualified healthcare professional.
