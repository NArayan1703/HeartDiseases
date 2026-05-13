import pickle

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


MODEL_PATH = "xgb_model.pkl"
TEST_DATA_PATH = "test_data.pkl"
THRESHOLDS = [0.5, 0.35, 0.25, 0.15, 0.10]


def main():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(TEST_DATA_PATH, "rb") as test_file:
        X_test, y_test = pickle.load(test_file)

    y_proba = model.predict_proba(X_test)[:, 1]

    print("Model Evaluation Results")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC Score: {average_precision_score(y_test, y_proba):.4f}")

    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        print(f"\nThreshold: {threshold}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
