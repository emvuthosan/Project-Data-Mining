from sklearn.preprocessing import LabelEncoder
import pandas as pd

def build_graph_features(df):
    # encode community since it is categorical
    if "community_id" in df.columns:
        le = LabelEncoder()
        df["community_id"] = le.fit_transform(df["community_id"].astype(str))

    # ===== Encode trust_level (ordinal) =====
    if "trust_level" in df.columns:
        trust_order = {"VERY_LOW": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "VERY_HIGH": 4}
        df["trust_level_encoded"] = df["trust_level"].map(trust_order).fillna(-1).astype(int)

    # ===== Encode credit_level (ordinal) =====
    if "credit_level" in df.columns:
        credit_order = {"Very Poor": 0, "Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
        df["credit_level_encoded"] = df["credit_level"].map(credit_order).fillna(-1).astype(int)

    # fill NaN values for some graph metrics if any
    cols_to_fill = [
        "trust_score", "trustrank", "pagerank", 
        "anti_trustrank", "simrank"
    ]
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df