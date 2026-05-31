from dotenv import find_dotenv, load_dotenv

from src.ingestion import (
    create_hierarchical_chunks,
    create_vectorstore
)
from src.retriever import HierarchicalRetriever
from src.graph import build_graph

load_dotenv(find_dotenv())


def main():
    # -------------------------
    # CREATE HIERARCHY
    # -------------------------
    child_docs, parent_map = create_hierarchical_chunks()

    # -------------------------
    # VECTOR STORE
    # -------------------------
    create_vectorstore(child_docs)

    # -------------------------
    # RETRIEVER
    # -------------------------
    retriever = HierarchicalRetriever(parent_map)

    # -------------------------
    # BUILD GRAPH
    # -------------------------
    app = build_graph(retriever)

    print("\n" + "="*40)
    print(" Hierarchical RAG Active (Parent-Child Expansion)")
    print(" Type 'exit' to quit.")
    print("="*40)

    while True:
        question = input("\nAsk Question: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down workspace.")
            break

        result = app.invoke({
            "question": question
        })

        print("\nAnswer:\n")
        print(result["answer"])
        print("-" * 20)


if __name__ == "__main__":
    main()
