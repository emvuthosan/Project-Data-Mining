from connect import Neo4jConnection
import re, os

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "ldrltnd17"

def create_index(conn):
    print("Creating index...")
    query = """
    CREATE INDEX account_id IF NOT EXISTS
    FOR (a:Account)
    ON (a.id);
    """
    conn.run_query(query)

def import_nodes(conn, files):
    print("Importing nodes...")
    
    for f in files:
        query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///node_df_csv/{f}' AS row
        CALL {{
            WITH row
            MERGE (a:Account {{id: row.account_id}})
            SET a.bank_id = toInteger(row.bank_id),
                a.entity_id = row.entity_id,
                a.label = toInteger(row.label)
        }} IN TRANSACTIONS OF 1000 ROWS
        """
        
        print(f"Node file: {f}")
        conn.run_query(query)

def import_edges(conn, files):
    print("Importing edges...")
    
    for f in files:
        query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///edge_df_csv/{f}' AS row
        CALL {{
            WITH row
            MATCH (src:Account {{id: row.src}})
            MATCH (dst:Account {{id: row.dst}})

            CREATE (src)-[t:TRANSFER]->(dst)

            SET t.amount = toFloat(row.amount),
                t.timestamp = row.timestamp,
                t.payment_format = row.payment_format,
                t.is_laundering = toInteger(row.is_laundering_tx)
        }} IN TRANSACTIONS OF 5000 ROWS
        """
        
        print(f"Edge file: {f}")
        conn.run_query(query)

def import_features(conn, files):
    print("Importing features...")
    
    for f in files:
        query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///features_df_csv/{f}' AS row
        CALL {{
            WITH row
            MATCH (a:Account {{id: row.account_id}})
            SET 
                a.in_degree = toInteger(row.in_degree),
                a.out_degree = toInteger(row.out_degree),
                a.unique_senders = toInteger(row.unique_senders),
                a.unique_receivers = toInteger(row.unique_receivers),
                a.total_sent = toFloat(row.total_sent),
                a.total_received = toFloat(row.total_received),
                a.avg_tx_amount = toFloat(row.avg_tx_amount),
                a.avg_tx_per_day = toFloat(row.avg_tx_per_day),
                a.avg_time_gap = toFloat(row.avg_time_gap)
        }} IN TRANSACTIONS OF 1000 ROWS
        """
        
        print(f"Feature file: {f}")
        conn.run_query(query)

def run_import_pipeline():
    conn = Neo4jConnection(URI, USER, PASSWORD)

    files_nodes = os.listdir("data/node_df_csv")
    files_edges = os.listdir("data/edge_df_csv")
    files_features = os.listdir("data/features_df_csv")

    create_index(conn)
    import_nodes(conn, files_nodes)
    import_edges(conn, files_edges)
    import_features(conn, files_features)

    conn.close()
    print("✅ Import completed!")

# RUN
if __name__ == "__main__":
    run_import_pipeline()