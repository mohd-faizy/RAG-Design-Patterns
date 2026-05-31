from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from src.graph_builder import GraphBuilder
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

def build_knowledge_graph():
    print("Initializing Graph RAG Ingestion Pipeline...")
    
    loader = TextLoader(str(BASE_DIR.parent / "_data" / "source.txt"))
    docs = loader.load()

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    
    graph_builder = GraphBuilder()
    
    if graph_builder.online:
        try:
            # Clear database to start fresh in local development
            with graph_builder.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("Cleared existing local Neo4j database records.")
        except Exception as e:
            print(f"[Ingestion] Warning: Failed to clear Neo4j: {e}")
    else:
        print("[Ingestion] Running in OFFLINE mock database mode.")

    try:
        for doc in docs:
            text = doc.page_content
            triplets = graph_builder.extract_triplets(text, llm)
            print(f"\nExtracted relationships from chunk:\n" + "\n".join(f"  - {t}" for t in triplets))
            graph_builder.store_triplets(triplets)
            
        print("\nIngestion Completed successfully!")
    finally:
        graph_builder.close()
