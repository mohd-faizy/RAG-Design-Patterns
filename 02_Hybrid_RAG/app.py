from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


from src.ingestion import (
    load_and_split_documents,
    create_vector_store
)

from src.graph import build_graph



def main():
    # -----------------------
    # CREATE VECTOR STORE
    # -----------------------
    docs = load_and_split_documents()
    create_vector_store(docs)

    # -----------------------
    # BUILD GRAPH
    # -----------------------
    app = build_graph()

    print("\n" + "="*40)
    print(" Hybrid RAG (BM25 + Chroma) Active")
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