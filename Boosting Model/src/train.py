import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos = (y_train == 1).sum()
    scale_pos_weight = (y_train == 0).sum() / pos if pos > 0 else 1

    # ===== LightGBM =====
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=256,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight
    )

    lgb_model.fit(X_train, y_train)

    # ===== RandomForest =====
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15)
    rf_model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)

    joblib.dump(lgb_model, "model/lgb_model.pkl")
    joblib.dump(rf_model, "model/rf_model.pkl")

    return lgb_model, rf_model, X_test, y_test