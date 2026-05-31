from dotenv import load_dotenv

from src.ingestion import (
    load_documents,
    create_vectorstore
)

from src.graph import build_graph

load_dotenv()


def main():
    docs = load_documents()
    create_vectorstore(docs)

    app = build_graph()

    print("\n" + "="*40)
    print(" Self-RAG Active (State Reflection Loop)")
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
            "question": question,
            "retries": 0
        })

        print("\nAnswer:\n")
        print(result["answer"])
        print("-" * 20)


if __name__ == "__main__":
    main()
