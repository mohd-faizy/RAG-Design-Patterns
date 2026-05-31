import os
from neo4j import GraphDatabase


class KGRetriever:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        print(f"[KGRetriever] Connecting to Neo4j URI: {uri} (Username: {username})")

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
            print("[Neo4j Warning] Neo4j driver not initialized. Falling back to local search...")
            return self._retrieve_local(query)

        try:
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"[Neo4j Warning] Database is offline or unreachable. Falling back to local search... (Error: {e})")
            return self._retrieve_local(query)

        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (a)-[r]->(b)
                WHERE toLower(a.name) CONTAINS toLower($query)
                   OR toLower($query) CONTAINS toLower(a.name)
                   OR toLower(b.name) CONTAINS toLower($query)
                   OR toLower($query) CONTAINS toLower(b.name)
                RETURN a.name AS source,
                       type(r) AS relation,
                       b.name AS target
                LIMIT 10
                """
                result = session.run(cypher, {"query": query})

                outputs = []
                for record in result:
                    outputs.append(
                        f"{record['source']} --{record['relation']}--> {record['target']}"
                    )
                return outputs
        except Exception as e:
            print(f"[Neo4j Database Retrieval Warning] Cypher query failed: {e}. Falling back to local search...")
            return self._retrieve_local(query)

    def _retrieve_local(self, query):
        from pathlib import Path
        import json
        import re
        
        local_path = str(Path(__file__).resolve().parent.parent / "local_graph.json")
        if not os.path.exists(local_path):
            print("[KGRetriever Warning] local_graph.json not found. Returning empty list.")
            return []
            
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                triplets = json.load(f)
            print(f"[KGRetriever] Loaded {len(triplets)} relationships from local_graph.json for offline RAG")
        except Exception as e:
            print(f"[KGRetriever Warning] Failed to load local_graph.json: {e}")
            return []
            
        # Smart keyword overlap matching
        query_words = set(w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2)
        if not query_words:
            query_words = {query.lower()}

        scored_triplets = []
        for item in triplets:
            source, rel, target = item[0], item[1], item[2]
            source_lower = source.lower()
            target_lower = target.lower()
            rel_lower = rel.lower()
            
            match_count = 0
            for word in query_words:
                if word in source_lower or word in target_lower or word in rel_lower:
                    match_count += 1
                    
            if match_count > 0:
                scored_triplets.append((match_count, f"{source} --{rel}--> {target}"))
        
        scored_triplets.sort(key=lambda x: x[0], reverse=True)
        return [text for count, text in scored_triplets[:10]]
