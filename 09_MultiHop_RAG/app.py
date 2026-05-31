from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from src.ingestion import load_documents, create_vectorstore
from src.graph import build_graph

# Load environment variables from central root .env

def main():
    # 1. Build local vector database if needed
    docs = load_documents()
    create_vectorstore(docs)

    # 2. Compile state graph workflow
    app = build_graph()

    print("\n" + "="*50)
    print(" Multi-Hop RAG Engine Active (2-Hop Reasoning)")
    print(" Type 'exit' to quit.")
    print("="*50)

    while True:
        question = input("\nAsk Question: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down Multi-Hop RAG workspace.")
            break

        result = app.invoke({"question": question})

        print("\n[Answer]:")
        print(result["answer"])
        print("-" * 30)

if __name__ == "__main__":
    main()