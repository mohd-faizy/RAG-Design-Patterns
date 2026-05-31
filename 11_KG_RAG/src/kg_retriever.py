import os
from neo4j import GraphDatabase


class KGRetriever:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
        except Exception as e:
            self.driver = None
            print(f"[Neo4j Retriever Init Warning] {e}")

    # --------------------------------
    # GRAPH QUERY
    # --------------------------------
    def retrieve(self, query):
        if not self.driver:
            print("[Neo4j Warning] Neo4j driver not initialized. Skipping DB retrieval.")
            return []

        try:
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"[Neo4j Warning] Database is offline or unreachable. Skipping DB retrieval. (Error: {e})")
            return []

        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (a)-[r]->(b)
                WHERE
                toLower(a.name) CONTAINS toLower($query)
                OR
                toLower(b.name) CONTAINS toLower($query)
                RETURN
                a.name AS source,
                type(r) AS relation,
                b.name AS target
                LIMIT 10
                """
                result = session.run(
                    cypher,
                    query=query
                )

                outputs = []
                for record in result:
                    outputs.append(
                        f"{record['source']} --{record['relation']}--> {record['target']}"
                    )
                return outputs
        except Exception as e:
            print(f"[Neo4j Database Retrieval Warning] Cypher query failed: {e}")
            return []
