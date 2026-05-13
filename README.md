# Heart Diseases

This project trains and serves a heart disease prediction model from `health_data.csv`.

## Project Files

- `health_data.csv` is the current source dataset.
- `model_training.py` cleans column names, imputes missing values, trains an XGBoost classifier with class weighting, and writes `xgb_model.pkl`, `test_data.pkl`, and `feature_schema.pkl`.
- `model_evaluation.py` evaluates the saved model with ROC-AUC, PR-AUC, confusion matrices, and threshold-specific classification reports.
- `streamlit_app.py` serves the preferred local Streamlit prediction UI.
- `web_app.py` serves an optional local Flask prediction UI using the saved model pipeline.

## Run Locally

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
python3 model_training.py
python3 model_evaluation.py
python3 -m streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

## Model Inputs

The model uses:

- `sex`
- `age`
- `smokes`
- `total_cholesterol`
- `fasting_blood_sugar`
- `vigorous_minutes`
- `moderate_minutes`
- `walking_cycling_minutes`
- `sitting_minutes`
- `fruit_servings_per_week`
- `vegetable_servings_per_week`
- `systolic_bp`
- `diastolic_bp`

The target is `Y`, where `0` means no heart disease and `1` means heart disease.

## Current Model Limitation

The target class is very imbalanced: `5480` no-heart-disease rows and `107` heart-disease rows. The training pipeline uses XGBoost `scale_pos_weight`, but evaluation should focus on class-1 recall, precision, PR-AUC, and threshold behavior instead of accuracy alone.
