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
            print("[GraphRetriever] Database offline. Using local mock relationships fallback...")
            query_lower = query.lower()
            for source, rel, target in MOCK_TRIPLETS:
                if (query_lower in source.lower() or 
                    query_lower in target.lower() or 
                    query_lower in rel.lower()):
                    outputs.append(f"{source} -[{rel}]-> {target}")
            return outputs[:10]

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
