from neo4j import GraphDatabase
import config


class Neo4jConnector:
    """Singleton connector cho Neo4j."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    def run(self, query: str, params: dict = None):
        """Chạy một Cypher query và trả về danh sách record."""
        params = params or {}
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def run_write(self, query: str, params: dict = None):
        """Chạy write transaction."""
        params = params or {}
        with self.driver.session() as session:
            return session.execute_write(
                lambda tx: tx.run(query, **params).consume()
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()