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
        
        print(f"[KGBuilder] Connecting to Neo4j URI: {uri} (Username: {username})")

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
                    source = parts[0].strip()
                    rel = parts[1].strip()
                    target = parts[2].strip()
                    
                    # Clean source and target by removing leading numbered list prefixes, bold markers, and quotes
                    source = re.sub(r'^\d+\.\s*', '', source)  # remove "1. ", "12. "
                    source = re.sub(r'^[-*+]\s*', '', source)   # remove "- ", "* "
                    source = source.strip().strip('*"`')
                    
                    target = re.sub(r'^\d+\.\s*', '', target)
                    target = re.sub(r'^[-*+]\s*', '', target)
                    target = target.strip().strip('*"`')
                    
                    # Filter out helper introductory lines or headers
                    if (source.lower().startswith("here are") or 
                        source.lower().startswith("entity") or 
                        len(source) > 60 or len(target) > 60):
                        continue
                        
                    if source and rel and target:
                        triplets.append([source, rel, target])
        return triplets

    # --------------------------------
    # STORE IN NEO4J
    # --------------------------------
    def store_triplets(self, triplets):
        if not self.driver:
            print("[Neo4j Warning] Neo4j driver not initialized. Saving to local_graph.json...")
            self._store_local(triplets)
            return

        try:
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"[Neo4j Warning] Database is offline or unreachable. Saving to local_graph.json... (Error: {e})")
            self._store_local(triplets)
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

    def _store_local(self, triplets):
        from pathlib import Path
        import json
        local_path = str(Path(__file__).resolve().parent.parent / "local_graph.json")
        existing = []
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        
        for t in triplets:
            if t not in existing:
                existing.append(t)
        
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"[KGBuilder] Successfully saved {len(triplets)} relationships to local_graph.json!")
