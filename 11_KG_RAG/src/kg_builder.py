import os
import re
from neo4j import GraphDatabase
from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


class KGBuilder:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
        except Exception as e:
            self.driver = None
            print(f"[Neo4j Initialization Warning] {e}")

    # --------------------------------
    # EXTRACT STRUCTURED TRIPLETS
    # --------------------------------
    def extract_triplets(self, text):
        prompt = f"""
        Extract knowledge graph triplets.

        Return format:
        ENTITY1 | RELATION | ENTITY2

        Use uppercase relationships.

        Text:
        {text}
        """

        try:
            response = llm.invoke(prompt)
            lines = response.content.split("\n")
        except Exception as e:
            print(f"[LLM Triplet Extraction Warning] LLM call failed: {e}")
            return []

        triplets = []
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 3:
                    triplets.append([
                        p.strip()
                        for p in parts
                    ])
        return triplets

    # --------------------------------
    # STORE IN NEO4J
    # --------------------------------
    def store_triplets(self, triplets):
        if not self.driver:
            print("[Neo4j Warning] Neo4j driver not initialized. Skipping DB write.")
            return

        try:
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"[Neo4j Warning] Database is offline or unreachable. Skipping DB write. (Error: {e})")
            return

        print(f"Storing {len(triplets)} semantic triplets in Neo4j...")
        try:
            with self.driver.session() as session:
                for source, rel, target in triplets:
                    # Sanitize relation label to be alphanumeric to prevent Cypher syntax errors
                    clean_rel = re.sub(r'[^A-Za-z0-9_]', '_', rel).upper()
                    if not clean_rel:
                        clean_rel = "RELATED_TO"

                    query = f"""
                    MERGE (a:Entity {{name: $source}})
                    MERGE (b:Entity {{name: $target}})
                    MERGE (a)-[:{clean_rel}]->(b)
                    """
                    session.run(query, source=source, target=target)
            print("Successfully persisted triplets in Neo4j!")
        except Exception as e:
            print(f"[Neo4j Database Warning] Failed to execute query session: {e}")
