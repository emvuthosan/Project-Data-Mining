from sklearn.preprocessing import LabelEncoder
import pandas as pd

def build_graph_features(df, communities, fraud_accounts, cycles):

    # ===== FIX COMMUNITY =====
    communities = communities.rename(columns={
        "AccountID": "account_id",
        "communityId": "community_id"
    })

    df = df.merge(communities, on="account_id", how="left")

    # encode community
    if "community_id" in df.columns:
        le = LabelEncoder()
        df["community_id"] = le.fit_transform(df["community_id"].astype(str))

    # ===== FIX FRAUD ACCOUNTS =====
    fraud_accounts = fraud_accounts.rename(columns={
        "Suspicious_Account": "account_id"
    })

    fraud_accounts["is_known_fraud"] = 1

    df = df.merge(
        fraud_accounts[["account_id", "is_known_fraud"]],
        on="account_id",
        how="left"
    )

    df["is_known_fraud"] = df["is_known_fraud"].fillna(0)

    # ===== FIX CYCLES =====
    # chuyển từ A,B,C → 1 list account_id
    cycle_accounts = pd.concat([
        cycles["Account_A"],
        cycles["Account_B"],
        cycles["Account_C"]
    ]).dropna().unique()

    cycle_df = pd.DataFrame({
        "account_id": cycle_accounts,
        "is_in_cycle": 1
    })

    df = df.merge(cycle_df, on="account_id", how="left")
    df["is_in_cycle"] = df["is_in_cycle"].fillna(0)

    return df