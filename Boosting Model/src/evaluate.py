from sklearn.metrics import classification_report, roc_auc_score

def evaluate(lgb_model, rf_model, X_test, y_test):

    lgb_prob = lgb_model.predict_proba(X_test)[:,1]
    rf_prob = rf_model.predict_proba(X_test)[:,1]

    y_prob = (lgb_prob + rf_prob) / 2
    y_pred = (y_prob > 0.3).astype(int)

    print("===== REPORT =====")
    print(classification_report(y_test, y_pred))

    print("AUC:", roc_auc_score(y_test, y_prob))