from dotenv import find_dotenv, load_dotenv
from src.ingestion import load_documents
from src.colbert_retriever import build_collection, build_index
from src.graph import build_graph

# Load environment variables from central root .env or local
load_dotenv(find_dotenv())


def main():
    # 1. Load data
    chunks = load_documents()

    # 2. Build collection input file
    build_collection(chunks)

    # 3. Compile ColBERT Index
    build_index()

    # 4. Compile state graph workflow
    app = build_graph()

    print("\n" + "="*60)
    print(" ColBERT RAG Engine Active (Contextual Late Interaction)")
    print(" Chunks -> Token Embedding Encoding -> MaxSim Late Interaction Scoring")
    print(" Type 'exit' to quit.")
    print("="*60)

    while True:
        question = input("\nAsk Question: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down ColBERT RAG workspace.")
            break

        try:
            result = app.invoke({
                "question": question
            })

            print("\n" + "-"*40)
            print(" Answer:")
            print("-"*40)
            print(result["answer"])
            print("-" * 60)
        except Exception as e:
            print(f"\n[Execution Error] Failed to complete ColBERT RAG workflow: {e}")


if __name__ == "__main__":
    main()
