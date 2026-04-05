import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.connect import Neo4jConnection

def run_louvain_algorithm(conn):
    print("Đang nạp đồ thị vào bộ nhớ (Graph Projection)...")
    project_query = """
    CALL gds.graph.project(
      'aml_graph',
      'Account',
      'TRANSFER'
    )
    """
    try:
        conn.run_query(project_query)
    except Exception as e:
        print("Đồ thị 'aml_graph' có thể đã tồn tại trong RAM. Tiếp tục chạy thuật toán...")

    print("Đang chạy thuật toán phân cụm cộng đồng Louvain...")
    louvain_query = """
    CALL gds.louvain.stream('aml_graph')
    YIELD nodeId, communityId
    RETURN gds.util.asNode(nodeId).id AS AccountID, communityId
    ORDER BY communityId ASC
    LIMIT 20000
    """
    results = conn.run_query(louvain_query)
    
    print("Đang giải phóng RAM...")
    conn.run_query("CALL gds.graph.drop('aml_graph', false)")
    
    if results:
        df = pd.DataFrame(results)
        
        # Đếm xem mỗi băng đảng (communityId) có bao nhiêu thành viên
        community_sizes = df['communityId'].value_counts().reset_index()
        community_sizes.columns = ['communityId', 'member_count']
        
        print("\n Kích thước các cộng đồng (băng đảng) được phát hiện:")
        print(community_sizes.head(10))
        
        df.to_csv("account_communities.csv", index=False)
        print("\n Đã xuất dữ liệu cộng đồng ra file 'account_communities.csv'")

if __name__ == "__main__":
    db_conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="ldrltnd17")
    try:
        run_louvain_algorithm(db_conn)
    finally:
        db_conn.close()