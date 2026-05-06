from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate(lgb_models, rf_models, optimal_threshold, X_test, y_test):

    # Average predictions over all folds
    lgb_probs = np.mean([model.predict_proba(X_test)[:, 1] for model in lgb_models], axis=0)
    rf_probs = np.mean([model.predict_proba(X_test)[:, 1] for model in rf_models], axis=0)

    y_prob = (lgb_probs + rf_probs) / 2
    y_pred = (y_prob > optimal_threshold).astype(int)

    print("\n===== REPORT ON TEST SET =====")
    print(f"Sử dụng Optimal Threshold: {optimal_threshold:.2f}")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC (Average Precision):", average_precision_score(y_test, y_prob))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold = {optimal_threshold:.2f})")
    
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/confusion_matrix.png")
    plt.close()
    print("Đã lưu Confusion Matrix tại model/confusion_matrix.png")