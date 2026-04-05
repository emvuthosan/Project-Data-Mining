import sys
import os
import pandas as pd

# Thêm đường dẫn để gọi 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.connect import Neo4jConnection

def detect_cycles(conn):
    # Tối ưu siêu nhẹ: 
    # 1. Chỉ lấy 5000 tài khoản ra quét thử (chia nhỏ)
    # 2. Bỏ qua các giao dịch lặt vặt dưới 50.000 (giảm tải RAM)
    query = """
    MATCH (a:Account)
    WITH a LIMIT 50000
    MATCH (a)-[r1:TRANSFER]->(b:Account)-[r2:TRANSFER]->(c:Account)-[r3:TRANSFER]->(a)
    WHERE a.id <> b.id AND b.id <> c.id AND c.id <> a.id   // <-- Thêm dòng này
      AND r1.amount > 10000 
      AND r2.amount > 10000 
      AND r3.amount > 10000
    RETURN DISTINCT a.id AS Account_A,                     // <-- Thêm DISTINCT
           b.id AS Account_B, 
           c.id AS Account_C, 
           (r1.amount + r2.amount + r3.amount) AS Total_Volume
    ORDER BY Total_Volume DESC
    """
    print("Đang quét cụm 50000 tài khoản đầu tiên để tìm vòng lặp...")
    results = conn.run_query(query) 
    
    if results:
        df = pd.DataFrame(results)
        print("\nCẢNH BÁO - Đã phát hiện các cụm tài khoản đảo tiền:")
        print(df.head(10)) 
        
        df.to_csv("suspicious_cycles.csv", index=False)
        print("\n Đã xuất danh sách tài khoản nghi vấn ra file 'suspicious_cycles.csv'")
    else:
        print("Mạng lưới 50000 tài khoản đầu tiên an toàn, chưa phát hiện vòng lặp.")

if __name__ == "__main__":
    db_conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="ldrltnd17")
    
    try:
        detect_cycles(db_conn)
    finally:
        db_conn.close()