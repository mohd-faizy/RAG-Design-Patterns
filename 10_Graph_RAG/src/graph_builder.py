from neo4j import GraphDatabase
from langchain_groq import ChatGroq
import os

class GraphBuilder:
    def __init__(self):
        # Gracefully handle default Neo4j properties
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            self.driver.verify_connectivity()
            self.online = True
        except Exception as e:
            print(f"[GraphBuilder] Warning: Could not connect to local Neo4j database: {e}")
            print("[GraphBuilder] Running in offline mode.")
            self.online = False

    def close(self):
        if self.online:
            self.driver.close()

    def extract_triplets(self, text: str, llm: ChatGroq) -> list[list[str]]:
        """Uses LLM to extract key entity triplets in format ENTITY1 | RELATIONSHIP | ENTITY2"""
        prompt = f"""
        Extract key entities and their relationships from the given text.
        Return ONLY relationships in the following format:
        ENTITY1 | RELATIONSHIP | ENTITY2

        Provide each relationship on a new line. Do not write any explanations, introductory remarks, or formatting outside this structure.

        Text:
        {text}
        """
        response = llm.invoke(prompt)
        lines = response.content.split("\n")
        
        triplets = []
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 3:
                    triplets.append([p.strip() for p in parts])
        return triplets

    def store_triplets(self, triplets: list[list[str]]):
        """Stores triplets in local Neo4j database using Cypher commands."""
        if not self.online:
            print("[GraphBuilder] Offline: Skipping Neo4j write.")
            return

        with self.driver.session() as session:
            for source, rel, target in triplets:
                # Merge entities and merge relationships between them
                query = """
                MERGE (a:Entity {name: $source})
                MERGE (b:Entity {name: $target})
                MERGE (a)-[r:REL {type: $rel}]->(b)
                """
                session.run(query, source=source, target=target, rel=rel)
