import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.connect import Neo4jConnection

def run_classification(conn):
    print(" Đang truy quét mô-típ Fan-in (Gom tiền) và Fan-out (Phân tán)...")

    # 1. Tìm Fan-in: Tài khoản nhận tiền từ >= 5 người khác nhau
    query_fan_in = """
    MATCH (src:Account)-[r:TRANSFER]->(dst:Account)
    WITH dst, count(src) AS num_senders, sum(r.amount) AS total_received
    WHERE num_senders >= 5
    RETURN dst.id AS Suspicious_Account, 'Fan-in' AS Fraud_Type
    LIMIT 1000
    """
    
    # 2. Tìm Fan-out: Tài khoản chuyển tiền cho >= 5 người khác nhau
    query_fan_out = """
    MATCH (src:Account)-[r:TRANSFER]->(dst:Account)
    WITH src, count(dst) AS num_receivers, sum(r.amount) AS total_sent
    WHERE num_receivers >= 5
    RETURN src.id AS Suspicious_Account, 'Fan-out' AS Fraud_Type
    LIMIT 1000
    """

    res_in = pd.DataFrame(conn.run_query(query_fan_in))
    res_out = pd.DataFrame(conn.run_query(query_fan_out))

    # Gộp lại và xuất file
    df_final = pd.concat([res_in, res_out], ignore_index=True)
    df_final.to_csv("classified_fraud_accounts.csv", index=False)
    print(f" Đã tạo xong file 'classified_fraud_accounts.csv' với {len(df_final)} bản ghi.")

if __name__ == "__main__":
    db_conn = Neo4jConnection(uri="bolt://127.0.0.1:7687", user="neo4j", password="ldrltnd17")
    try:
        run_classification(db_conn)
    finally:
        db_conn.close()