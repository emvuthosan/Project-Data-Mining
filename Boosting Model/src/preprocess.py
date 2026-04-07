import numpy as np

def preprocess(features, nodes):
    features["account_id"] = features["account_id"].astype(str).str.strip()
    nodes["account_id"] = nodes["account_id"].astype(str).str.strip()

    df = features.merge(nodes, on="account_id", how="left")

    return df


def feature_engineering(df):
    df["tx_ratio"] = df["total_sent"] / (df["total_received"] + 1)
    df["avg_amount_per_day"] = df["total_sent"] / (df.get("active_days", 1) + 1)

    if "avg_tx_amount" in df.columns:
        df["z_score_amount"] = (
            (df["avg_tx_amount"] - df["avg_tx_amount"].mean()) /
            (df["avg_tx_amount"].std() + 1e-5)
        )

    return df


def prepare_xy(df):

    # fallback label
    if df["label"].nunique() == 1:
        df["label"] = (
            (df["is_known_fraud"] == 1) |
            (df["is_in_cycle"] == 1)
        ).astype(int)

    X = df.drop(columns=["account_id", "label", "bank_id"], errors="ignore")
    X = X.select_dtypes(include=["int64", "float64"])
    y = df["label"]

    return X, y