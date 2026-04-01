import config
from neo4j_connector import Neo4jConnector


def drop_projection_if_exists(conn: Neo4jConnector, graph_name: str = config.GRAPH_NAME):
    """Xoá projection cũ nếu tồn tại để tránh xung đột."""
    check_query = """
        CALL gds.graph.exists($graphName)
        YIELD exists
        RETURN exists
    """
    result = conn.run(check_query, {"graphName": graph_name})
    if result and result[0]["exists"]:
        drop_query = "CALL gds.graph.drop($graphName) YIELD graphName"
        conn.run(drop_query, {"graphName": graph_name})
        print(f"[Graph Projection] Đã xoá projection cũ: '{graph_name}'")


def create_projection(conn: Neo4jConnector, graph_name: str = config.GRAPH_NAME):
    """
    Tạo GDS native projection từ graph Neo4j.
    
    - Node: Account  (có thuộc tính label, id)
    - Edge: TRANSFER  (có thuộc tính amount, timestamp)
    - Hướng: NATURAL (directed, theo chiều giao dịch thực)
    """
    drop_projection_if_exists(conn, graph_name)

    project_query = """
        CALL gds.graph.project(
            $graphName,
            {
                Account: {
                    properties: ['label']
                }
            },
            {
                TRANSFER: {
                    orientation: 'NATURAL',
                    properties: {
                        amount: { defaultValue: 1.0 }
                    }
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
    """
    result = conn.run(project_query, {"graphName": graph_name})
    if result:
        r = result[0]
        print(
            f"[Graph Projection] Đã tạo '{r['graphName']}': "
            f"{r['nodeCount']} nodes, {r['relationshipCount']} edges"
        )
    return result


def create_undirected_projection(conn: Neo4jConnector,
                                  graph_name: str = "aml_graph_undirected"):
    """
    Tạo GDS projection không có hướng — dùng cho SimRank.
    SimRank yêu cầu undirected hoặc reversed graph.
    """
    drop_projection_if_exists(conn, graph_name)

    project_query = """
        CALL gds.graph.project(
            $graphName,
            {
                Account: {
                    properties: ['label']
                }
            },
            {
                TRANSFER: {
                    orientation: 'UNDIRECTED',
                    properties: {
                        amount: { defaultValue: 1.0 }
                    }
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
    """
    result = conn.run(project_query, {"graphName": graph_name})
    if result:
        r = result[0]
        print(
            f"[Graph Projection Undirected] '{r['graphName']}': "
            f"{r['nodeCount']} nodes, {r['relationshipCount']} edges"
        )
    return result