import pickle

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_PATH = "health_data.csv"
MODEL_PATH = "xgb_model.pkl"
TEST_DATA_PATH = "test_data.pkl"
FEATURE_SCHEMA_PATH = "feature_schema.pkl"
TARGET = "heart_disease"

COLUMN_RENAMES = {
    "sex": "sex",
    "age": "age",
    "Y": TARGET,
    "Smoke tobacoo or ciggrate": "smokes",
    "Total cholesterol": "total_cholesterol",
    "Fasting Blood Sugar": "fasting_blood_sugar",
    "vigorous_minutes": "vigorous_minutes",
    "moderate_minutes": "moderate_minutes",
    "cycling or walking minutes": "walking_cycling_minutes",
    "minutes sitting per day moderate-intensity sports": "sitting_minutes",
    "serving of foods per week": "fruit_servings_per_week",
    "Vegetable_serving per week": "vegetable_servings_per_week",
    "Blood presure Systolic ": "systolic_bp",
    "Blood Pressure Diastolic": "diastolic_bp",
}

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


def load_health_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAMES)

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df


def main():
    df = load_health_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    feature_schema = {
        "features": FEATURES,
        "target": TARGET,
        "positive_label": 1,
        "negative_label": 0,
        "column_renames": COLUMN_RENAMES,
        "scale_pos_weight": scale_pos_weight,
    }

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(pipeline, model_file)
    with open(TEST_DATA_PATH, "wb") as test_file:
        pickle.dump((X_test, y_test), test_file)
    with open(FEATURE_SCHEMA_PATH, "wb") as schema_file:
        pickle.dump(feature_schema, schema_file)

    print(f"Rows: {len(df)}")
    print(f"No heart disease: {(y == 0).sum()}")
    print(f"Heart disease: {(y == 1).sum()}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Saved model pipeline to {MODEL_PATH}")
    print(f"Saved test data to {TEST_DATA_PATH}")
    print(f"Saved feature schema to {FEATURE_SCHEMA_PATH}")


if __name__ == "__main__":
    main()
