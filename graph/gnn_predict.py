"""
GNN Predictions Loader
=======================
Chuẩn hóa file predictions từ GNN Model của Trung thành format
dùng được cho pipeline graph_mining.py

Hỗ trợ các format input:
  - fraud_predictions_v3.csv (output từ Kaggle pipeline)
  - Hoặc bất kỳ CSV nào có cột: account_id/id + fraud_prob/fraud_probability

Output: output-trustscore/gnn_predictions.csv
  Columns: id, gnn_fraud_prob, gnn_prediction

Cách chạy:
  python graph/gnn_predict.py
  python graph/gnn_predict.py --input path/to/gnn_output.csv
"""

import os
import sys
import io
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output-trustscore"
OUTPUT_CSV = OUTPUT_DIR / "gnn_predictions.csv"

# Danh sách file GNN output có thể có (thứ tự ưu tiên)
CANDIDATE_FILES = [
    OUTPUT_DIR / "fraud_predictions_v3.csv",
    OUTPUT_DIR / "gnn_output.csv",
    OUTPUT_DIR / "gnn_predictions_raw.csv",
    BASE_DIR / "fraud_predictions_v3.csv",
    BASE_DIR / "GNN Model" / "fraud_predictions_v3.csv",
]

# GNN threshold (từ model: f1_constrained, min_recall=0.7)
DEFAULT_THRESHOLD = 0.935


def find_gnn_csv(custom_path=None):
    """Tìm file GNN predictions CSV."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    for f in CANDIDATE_FILES:
        if f.exists():
            return f

    return None


def load_and_normalize(csv_path, threshold=DEFAULT_THRESHOLD):
    """
    Load file CSV từ GNN và chuẩn hóa tên cột.

    Hỗ trợ các format:
    - account_id, fraud_prob, prediction, label  (Kaggle pipeline output)
    - id, fraud_probability, prediction           (alternative format)
    """
    print(f"\n[Load] Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")

    # ── Chuẩn hóa cột ID ─────────────────────────────────────
    if "account_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"account_id": "id"})
    elif "AccountID" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"AccountID": "id"})

    if "id" not in df.columns:
        raise ValueError(f"Không tìm thấy cột ID. Columns: {df.columns.tolist()}")

    # ── Chuẩn hóa cột fraud probability ──────────────────────
    prob_col = None
    for candidate in ["fraud_prob", "fraud_probability", "gnn_fraud_prob", "probability", "prob"]:
        if candidate in df.columns:
            prob_col = candidate
            break

    if prob_col is None:
        raise ValueError(f"Không tìm thấy cột fraud probability. Columns: {df.columns.tolist()}")

    # ── Chuẩn hóa cột prediction ─────────────────────────────
    pred_col = None
    for candidate in ["prediction", "gnn_prediction", "pred", "is_fraud"]:
        if candidate in df.columns:
            pred_col = candidate
            break

    # ── Tạo output DataFrame ─────────────────────────────────
    result = pd.DataFrame({
        "id": df["id"],
        "gnn_fraud_prob": df[prob_col].astype(float),
    })

    if pred_col:
        result["gnn_prediction"] = df[pred_col].astype(int)
    else:
        # Tự tính prediction từ threshold
        result["gnn_prediction"] = (result["gnn_fraud_prob"] >= threshold).astype(int)
        print(f"  [INFO] Tự tính prediction với threshold={threshold:.4f}")

    return result


def validate_and_report(df):
    """Kiểm tra và báo cáo thống kê."""
    total = len(df)
    n_fraud = df["gnn_prediction"].sum()
    probs = df["gnn_fraud_prob"]

    print(f"\n[Validate] GNN Predictions Summary:")
    print(f"  Total accounts:    {total:,}")
    print(f"  Predicted fraud:   {n_fraud:,} ({n_fraud/total*100:.3f}%)")
    print(f"  Predicted normal:  {total - n_fraud:,}")
    print(f"\n  Probability distribution:")
    print(f"    Min:    {probs.min():.4f}")
    print(f"    Q1:     {probs.quantile(0.25):.4f}")
    print(f"    Median: {probs.median():.4f}")
    print(f"    Q3:     {probs.quantile(0.75):.4f}")
    print(f"    Max:    {probs.max():.4f}")
    print(f"    Mean:   {probs.mean():.4f}")

    # Kiểm tra tỉ lệ fraud hợp lý (~0.14% theo ground truth)
    fraud_pct = n_fraud / total * 100
    if fraud_pct > 50:
        print(f"\n  ⚠️  WARNING: Tỉ lệ fraud = {fraud_pct:.1f}% — quá cao!")
        print(f"     Dataset thực tế chỉ có ~0.14% fraud.")
        print(f"     Có thể cần điều chỉnh threshold hoặc kiểm tra lại input.")
    elif fraud_pct > 10:
        print(f"\n  ⚠️  WARNING: Tỉ lệ fraud = {fraud_pct:.1f}% — cao hơn expected (~0.14%)")
    else:
        print(f"\n  ✓ Tỉ lệ fraud {fraud_pct:.3f}% — hợp lý.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Chuẩn hóa GNN predictions cho pipeline")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Đường dẫn file GNN predictions CSV")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Threshold cho prediction (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GNN PREDICTIONS LOADER")
    print("=" * 60)

    # 1. Tìm file
    csv_path = find_gnn_csv(args.input)

    if csv_path is None:
        print("\n❌ Không tìm thấy file GNN predictions!")
        print("\nCách giải quyết:")
        print("  1. Nhắn Trung gửi file 'fraud_predictions_v3.csv'")
        print("     (File này tự xuất khi train GNN trên Kaggle)")
        print(f"  2. Đặt file vào: {OUTPUT_DIR}/")
        print("  3. Chạy lại: python graph/gnn_predict.py")
        print("\n  Hoặc chỉ định file trực tiếp:")
        print("     python graph/gnn_predict.py --input path/to/file.csv")
        return

    # 2. Load và chuẩn hóa
    df = load_and_normalize(csv_path, threshold=args.threshold)

    # 3. Validate
    validate_and_report(df)

    # 4. Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Output] Saved: {OUTPUT_CSV}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # 5. Preview
    print("\n── TOP 10 HIGHEST GNN FRAUD PROBABILITY ──")
    top = df.nlargest(10, "gnn_fraud_prob")
    print(top.to_string(index=False))

    print("\n" + "=" * 60)
    print("DONE — Giờ chạy: python graph/graph_mining.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
