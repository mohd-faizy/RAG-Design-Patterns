from dotenv import find_dotenv, load_dotenv

from src.ingestion import (
    load_and_contextualize,
    create_vectorstore
)

from src.graph import build_graph

load_dotenv(find_dotenv())


def main():
    # -------------------------
    # CONTEXTUAL INGESTION
    # -------------------------
    docs = load_and_contextualize()
    create_vectorstore(docs)

    # -------------------------
    # BUILD GRAPH
    # -------------------------
    app = build_graph()

    print("\n" + "="*40)
    print(" Contextual RAG Active (Anthropic Contextualization)")
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
