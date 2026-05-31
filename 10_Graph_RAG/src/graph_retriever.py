from neo4j import GraphDatabase
import os

MOCK_TRIPLETS = [
    ("LangGraph", "is a framework for", "building graph-based AI systems"),
    ("Graph RAG", "uses", "knowledge graphs for retrieval"),
    ("Neo4j", "is", "a graph database"),
    ("Groq", "provides", "fast inference for open-source LLMs"),
    ("ReAct", "combines", "reasoning and acting")
]

class GraphRetriever:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            self.online = True
        except Exception:
            self.online = False

    def close(self):
        if self.online:
            self.driver.close()

    def retrieve(self, query: str) -> list[str]:
        """Queries Neo4j database using Cypher, falling back to local search in offline mode."""
        outputs = []
        
        if not self.online:
            print("[GraphRetriever] Database offline. Using local relationships fallback...")
            from pathlib import Path
            import json
            import re
            
            local_path = str(Path(__file__).resolve().parent.parent / "local_graph.json")
            triplets = MOCK_TRIPLETS
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        triplets = json.load(f)
                    print(f"[GraphRetriever] Loaded {len(triplets)} real relationships from local_graph.json")
                except Exception as e:
                    print(f"[GraphRetriever Warning] Failed to load local_graph.json: {e}")
            
            # Smart token/keyword overlap matching
            query_words = set(w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2)
            if not query_words:
                query_words = {query.lower()}

            scored_triplets = []
            for item in triplets:
                # Handle both lists [source, rel, target] and tuples (source, rel, target)
                source, rel, target = item[0], item[1], item[2]
                source_lower = source.lower()
                target_lower = target.lower()
                rel_lower = rel.lower()
                
                match_count = 0
                for word in query_words:
                    if word in source_lower or word in target_lower or word in rel_lower:
                        match_count += 1
                        
                if match_count > 0:
                    scored_triplets.append((match_count, f"{source} -[{rel}]-> {target}"))
            
            scored_triplets.sort(key=lambda x: x[0], reverse=True)
            return [text for count, text in scored_triplets[:10]]

        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (a)-[r:REL]->(b)
                WHERE toLower(a.name) CONTAINS toLower($query)
                   OR toLower(b.name) CONTAINS toLower($query)
                RETURN a.name AS source, r.type AS relation, b.name AS target
                LIMIT 10
                """
                result = session.run(cypher, query=query)
                for record in result:
                    outputs.append(f"{record['source']} -[{record['relation']}]-> {record['target']}")
        except Exception as e:
            print(f"[GraphRetriever] Error: Neo4j traversal failed: {e}")
            
        return outputs
