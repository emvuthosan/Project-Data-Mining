import sys
import os
import pandas as pd

# Kết nối với module graph của dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.connect import Neo4jConnection

def build_node_features(conn):
    print("BƯỚC 1: Đang trích xuất Bậc vào (In-degree) và Bậc ra (Out-degree) từ Neo4j...")
    
    # Lấy thông tin tổng giao dịch nhận và gửi của từng tài khoản
    query_degree = """
    MATCH (n:Account)
    RETURN n.id AS AccountID,
           count{ (n)<-[:TRANSFER]-() } AS in_degree,
           count{ (n)-[:TRANSFER]->() } AS out_degree
    """
    degree_data = conn.run_query(query_degree)
    df_features = pd.DataFrame(degree_data)
    print(f"   -> Đã lấy xong dữ liệu đặc trưng cho {len(df_features)} tài khoản.")

    print("\nBƯỚC 2: Đang trích xuất Đặc trưng Vòng lặp (Cycles)...")
    try:
        df_cycles = pd.read_csv("mining-result/suspicious_cycles.csv")
        # Đếm tần suất xuất hiện của mỗi tài khoản trong các vòng lặp
        all_cycle_ids = pd.concat([df_cycles['Account_A'], df_cycles['Account_B'], df_cycles['Account_C']])
        cycle_counts = all_cycle_ids.value_counts().reset_index()
        cycle_counts.columns = ['AccountID', 'num_cycles']
        
        # Nối vào bảng tính năng chính
        df_features = df_features.merge(cycle_counts, on='AccountID', how='left')
        df_features['num_cycles'] = df_features['num_cycles'].fillna(0)
        print("   -> Đã tính xong tần suất tham gia vòng lặp.")
    except FileNotFoundError:
        print(" Không tìm thấy 'mining-result/suspicious_cycles.csv'")

    print("\nBƯỚC 3: Đang trích xuất Đặc trưng Băng đảng (Community)...")
    try:
        df_communities = pd.read_csv("mining-result/account_communities.csv")
        # Tính kích thước của từng băng đảng
        comm_sizes = df_communities['communityId'].value_counts().reset_index()
        comm_sizes.columns = ['communityId', 'community_size']
        df_communities = df_communities.merge(comm_sizes, on='communityId', how='left')
        
        # Nối ID băng đảng và Kích thước băng đảng vào bảng tính năng chính
        df_features = df_features.merge(df_communities[['AccountID', 'communityId', 'community_size']], on='AccountID', how='left')
        df_features['community_size'] = df_features['community_size'].fillna(0)
        print("   -> Đã gắn thông tin quy mô tổ chức cho từng tài khoản.")
    except FileNotFoundError:
        print("  Không tìm thấy 'mining-result/account_communities.csv' ")

    print("\nBƯỚC 4: Tính toán tỷ lệ Nhận/Chuyển (In/Out Ratio)...")
    # Tỷ lệ này giúp AI phát hiện các trạm trung chuyển (nhận bao nhiêu chuyển bấy nhiêu)
    # Thêm 1e-5 để tránh lỗi chia cho 0
    df_features['in_out_ratio'] = df_features['in_degree'] / (df_features['out_degree'] + 1e-5)
    
    # Làm tròn các con số cho đẹp
    df_features = df_features.round(4)

    print("\nBƯỚC 5: Xuất dữ liệu huấn luyện...")
    output_path = "mining-result/node_features.csv"
    df_features.to_csv(output_path, index=False)
    
    print(f"\n File đã được lưu tại: {output_path}")
    
    # Hiển thị Top 5 tài khoản đáng ngờ nhất dựa trên số vòng lặp
    print("\n--- TOP 5 TÀI KHOẢN ĐẶC TRƯNG NHẤT ---")
    print(df_features.sort_values(by=['num_cycles', 'in_degree', 'out_degree'], ascending=False).head(5))

if __name__ == "__main__":
    db_conn = Neo4jConnection(uri="bolt://127.0.0.1:7687", user="neo4j", password="ldrltnd17")
    try:
        build_node_features(db_conn)
    finally:
        db_conn.close()