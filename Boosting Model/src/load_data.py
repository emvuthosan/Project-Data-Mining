import pandas as pd
import glob

def load_data():
    features = pd.concat([pd.read_csv(f) for f in glob.glob("data/features_df_csv/*.csv")])
    nodes = pd.concat([pd.read_csv(f) for f in glob.glob("data/node_df_csv/*.csv")])

    communities = pd.read_csv("data/account_communities.csv")
    fraud_accounts = pd.read_csv("data/classified_fraud_accounts.csv")
    cycles = pd.read_csv("data/suspicious_cycles.csv")

    return features, nodes, communities, fraud_accounts, cycles