import joblib
import pandas as pd

def load_model():
    lgb_model = joblib.load("model/lgb_model.pkl")
    rf_model = joblib.load("model/rf_model.pkl")
    return lgb_model, rf_model


def predict(df, X):
    lgb_model, rf_model = load_model()

    # predict probability
    lgb_prob = lgb_model.predict_proba(X)[:,1]
    rf_prob = rf_model.predict_proba(X)[:,1]

    # ensemble
    y_prob = (lgb_prob + rf_prob) / 2

    # threshold
    y_pred = (y_prob > 0.3).astype(int)

    # output
    result = df[["account_id"]].copy()
    result["fraud_probability"] = y_prob
    result["prediction"] = y_pred

    return result