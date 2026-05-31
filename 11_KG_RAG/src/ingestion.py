from pathlib import Path
from langchain_community.document_loaders import TextLoader
from src.kg_builder import KGBuilder


def build_knowledge_graph():
    print("Initiating KG ingestion pipeline...")
    try:
        loader = TextLoader(str(Path(__file__).resolve().parent.parent.parent / "_data" / "source.txt"))
        docs = loader.load()
    except Exception as e:
        print(f"[Ingestion Error] Failed to load shared data/source.txt: {e}")
        return

    builder = KGBuilder()

    if builder.driver:
        try:
            with builder.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("Cleared existing local/cloud Neo4j database records to start fresh.")
        except Exception as e:
            print(f"[Ingestion Warning] Failed to clear Neo4j database: {e}")

    for doc in docs:
        triplets = builder.extract_triplets(
            doc.page_content
        )
        builder.store_triplets(triplets)
