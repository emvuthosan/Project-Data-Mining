import pandas as pd
import glob
import os
import json
import joblib

def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    cache_file = os.path.join(data_dir, "cached_nodes_df.pkl")

    if os.path.exists(cache_file):
        cached = joblib.load(cache_file)
        # Kiểm tra cache có cột node_id không (cache cũ thì không có)
        if "node_id" in cached.columns:
            print(f"Loading cached parsed data from {cache_file}...")
            return cached
        else:
            print("Cache cũ thiếu cột node_id, đang parse lại...")

    print("Parsing JSON from nodes_*.csv files... This may take a moment.")
    nodes_dir = os.path.join(data_dir, "nodes")
    nodes_files = glob.glob(os.path.join(nodes_dir, "nodes_*.csv"))

    if not nodes_files:
        raise FileNotFoundError(f"Không tìm thấy file nodes_*.csv trong {nodes_dir}.")

    parsed_rows = []
    all_node_ids = []
    
    def parse_json(row):
        try:
            return json.loads(row)['properties']
        except Exception:
            return {}

    for file in nodes_files:
        print(f"Reading {os.path.basename(file)}...")
        df_part = pd.read_csv(file)
        # Giữ lại node_id gốc để merge với rels data sau này
        all_node_ids.extend(df_part['node_id'].tolist())
        # Apply json parser
        parsed = df_part['n'].apply(parse_json).tolist()
        parsed_rows.extend(parsed)

    print("Creating dataframe...")
    df = pd.DataFrame(parsed_rows)
    df['node_id'] = all_node_ids
    
    # Đổi tên cột id thành account_id để tương thích với các bước sau
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'account_id'})
        
    print(f"Saving parsed dataframe to {cache_file}...")
    joblib.dump(df, cache_file)

    return df