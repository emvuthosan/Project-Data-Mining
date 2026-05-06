import numpy as np

def preprocess(df):
    if "account_id" in df.columns:
        df["account_id"] = df["account_id"].astype(str).str.strip()
    
    return df


def feature_engineering(df):
    df["tx_ratio"] = df["total_sent"] / (df["total_received"] + 1)
    df["avg_amount_per_day"] = df["total_sent"] / (df.get("active_days", 1) + 1)

    if "avg_tx_amount" in df.columns:
        df["z_score_amount"] = (
            (df["avg_tx_amount"] - df["avg_tx_amount"].mean()) /
            (df["avg_tx_amount"].std() + 1e-5)
        )

    # ===== INTERACTION FEATURES =====
    # Kết hợp Trust Score với cấu trúc đồ thị
    if "trust_score" in df.columns and "out_degree" in df.columns:
        df["trust_x_out_degree"] = df["trust_score"] * df["out_degree"]

    # Tỷ lệ rủi ro: anti_trustrank cao + pagerank thấp = rất nghi ngờ
    if "anti_trustrank" in df.columns and "pagerank" in df.columns:
        df["risk_density"] = df["anti_trustrank"] / (df["pagerank"] + 1e-8)

    # Tỷ lệ bậc vào/ra: Fraud thường gửi ra nhiều hơn nhận vào
    if "out_degree" in df.columns and "in_degree" in df.columns:
        df["degree_ratio"] = df["out_degree"] / (df["in_degree"] + 1)

    # Số tiền trung bình mỗi lần gửi so với mỗi lần nhận
    if "total_sent" in df.columns and "out_degree" in df.columns:
        df["avg_sent_per_tx"] = df["total_sent"] / (df["out_degree"] + 1)
    if "total_received" in df.columns and "in_degree" in df.columns:
        df["avg_recv_per_tx"] = df["total_received"] / (df["in_degree"] + 1)

    # Chênh lệch giữa gửi và nhận (dòng tiền ròng)
    if "total_sent" in df.columns and "total_received" in df.columns:
        df["net_flow"] = df["total_sent"] - df["total_received"]
        df["net_flow_ratio"] = df["net_flow"] / (df["total_sent"] + df["total_received"] + 1)

    # Mức độ phân tán: gửi tiền cho nhiều người nhưng nhận từ ít người
    if "unique_receivers" in df.columns and "unique_senders" in df.columns:
        df["scatter_score"] = df["unique_receivers"] / (df["unique_senders"] + 1)

    return df


def prepare_xy(df):

    # fallback label
    if df["label"].nunique() == 1:
        df["label"] = (
            (df["is_known_fraud"] == 1) |
            (df["is_in_cycle"] == 1)
        ).astype(int)

    X = df.drop(columns=["account_id", "node_id", "label", "bank_id"], errors="ignore")
    X = X.select_dtypes(include=["int64", "float64", "float32", "int32"])
    
    # Giảm bộ nhớ: float64 → float32 (tiết kiệm 50% RAM)
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype("int32")
    
    y = df["label"]

    return X, y