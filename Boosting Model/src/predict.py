import joblib
import numpy as np
import pandas as pd

def load_model():
    data = joblib.load("model/ensemble_models.pkl")
    return data["lgb_models"], data["rf_models"], data["threshold"]

def predict(df, X):
    lgb_models, rf_models, threshold = load_model()

    # Predict average across folds
    lgb_probs = np.mean([model.predict_proba(X)[:, 1] for model in lgb_models], axis=0)
    rf_probs = np.mean([model.predict_proba(X)[:, 1] for model in rf_models], axis=0)

    # ensemble
    y_prob = (lgb_probs + rf_probs) / 2

    # threshold
    y_pred = (y_prob > threshold).astype(int)

    # output
    result = df[["account_id"]].copy()
    result["fraud_probability"] = y_prob
    result["prediction"] = y_pred

    return result