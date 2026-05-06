import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, fbeta_score
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X, y):
    # Keep a hold-out test set for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lgb_models = []
    rf_models = []
    
    oof_preds = np.zeros(len(X_train))
    categorical_features = ["community_id"] if "community_id" in X.columns else "auto"
    feature_importances = np.zeros(X.shape[1])

    print("\n[K-Fold Cross Validation] Bắt đầu huấn luyện...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np, y_train_np)):
        print(f"--- Fold {fold+1} ---")
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train_np[train_idx], y_train_np[val_idx]
        
        pos = (y_t == 1).sum()
        scale_pos_weight = (y_t == 0).sum() / pos if pos > 0 else 1
        
        # --- LightGBM ---
        lgb_model = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.01, num_leaves=256,
            max_depth=10,
            subsample=0.8, colsample_bytree=0.8, 
            reg_alpha=0.1, reg_lambda=0.1,
            min_child_samples=20,
            scale_pos_weight=scale_pos_weight,
            max_bin=512,
            verbose=-1,
            random_state=42, n_jobs=-1
        )
        
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        lgb_model.fit(
            X_t, y_t, eval_set=[(X_v, y_v)], eval_metric="auc",
            categorical_feature=categorical_features, callbacks=callbacks
        )
        
        # --- Random Forest ---
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
            class_weight='balanced_subsample'
        )
        rf_model.fit(X_t, y_t)
        
        lgb_models.append(lgb_model)
        rf_models.append(rf_model)
        
        # --- Out-of-fold Prediction ---
        lgb_prob = lgb_model.predict_proba(X_v)[:, 1]
        rf_prob = rf_model.predict_proba(X_v)[:, 1]
        oof_preds[val_idx] = (lgb_prob + rf_prob) / 2
        
        feature_importances += lgb_model.feature_importances_ / kf.n_splits

    # Find optimal threshold: Recall >= 0.70, maximize F1 within that constraint
    from sklearn.metrics import recall_score, precision_score
    
    best_threshold = 0.1
    best_f1_constrained = 0
    min_recall = 0.70  # Bắt buộc bắt được >= 70% fraud
    
    thresholds = np.arange(0.01, 0.9, 0.01)
    for thresh in thresholds:
        preds = (oof_preds > thresh).astype(int)
        rec = recall_score(y_train_np, preds, zero_division=0)
        if rec >= min_recall:
            f1 = f1_score(y_train_np, preds, zero_division=0)
            if f1 > best_f1_constrained:
                best_f1_constrained = f1
                best_threshold = thresh
            
    # Nếu không tìm được threshold nào đạt Recall >= 0.70, chọn threshold thấp nhất
    if best_f1_constrained == 0:
        print(f"[WARNING] Không tìm được threshold với Recall >= {min_recall:.0%}. Dùng F2-score fallback.")
        best_f2 = 0
        for thresh in thresholds:
            preds = (oof_preds > thresh).astype(int)
            f2 = fbeta_score(y_train_np, preds, beta=2.0)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = thresh

    if best_f1_constrained > 0:
        print(f"\n[Optimal Threshold] Ngưỡng: {best_threshold:.2f} (Recall >= {min_recall:.0%}, OOF F1: {best_f1_constrained:.4f})")
    else:
        print(f"\n[Optimal Threshold] Ngưỡng fallback: {best_threshold:.2f}")

    os.makedirs("model", exist_ok=True)
    # Save the ensemble
    joblib.dump({"lgb_models": lgb_models, "rf_models": rf_models, "threshold": best_threshold}, "model/ensemble_models.pkl")

    # Feature Importance Plot
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette="viridis")
    plt.title('LightGBM Feature Importance (K-Fold Average)')
    plt.tight_layout()
    plt.savefig('model/feature_importance.png')
    plt.close()
    print("Đã lưu biểu đồ Feature Importance tại model/feature_importance.png")

    return lgb_models, rf_models, best_threshold, X_test, y_test