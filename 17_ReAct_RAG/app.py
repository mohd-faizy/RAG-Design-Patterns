from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)


from src.ingestion import (
    load_documents,
    create_vectorstore
)
from src.graph import build_graph



def main():
    docs = load_documents()
    create_vectorstore(docs)

    app = build_graph()

    print("\n" + "="*40)
    print(" ReAct RAG Agent Active (Thought-Action-Observation Loop)")
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
            "messages": [
                (
                    "user",
                    question
                )
            ]
        })

        print("\nAnswer:\n")
        print(result["messages"][-1].content)
        print("-" * 20)


if __name__ == "__main__":
    main()