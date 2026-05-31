from dotenv import find_dotenv, load_dotenv
from src.ingestion import (
    load_documents,
    create_vectorstore
)
from src.graph import build_graph

# Load environment variables from central root .env or local
load_dotenv(find_dotenv())


def main():
    # 1. Build local vector database if needed
    docs = load_documents()
    create_vectorstore(docs)

    # 2. Compile state graph workflow
    app = build_graph()

    print("\n" + "="*60)
    print(" Deep Research RAG Engine Active (Multi-Step Autonomous Research)")
    print(" Planner -> Dual-Source Research (Chroma + Web) -> Synthesizer")
    print(" Type 'exit' to quit.")
    print("="*60)

    while True:
        question = input("\nResearch Question: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down Deep Research RAG workspace.")
            break

        try:
            result = app.invoke({
                "question": question
            })

            print("\n" + "-"*40)
            print(" Research Report:")
            print("-"*40)
            print(result["answer"])
            print("-" * 60)
        except Exception as e:
            print(f"\n[Execution Error] Failed to complete research task: {e}")


if __name__ == "__main__":
    main()
