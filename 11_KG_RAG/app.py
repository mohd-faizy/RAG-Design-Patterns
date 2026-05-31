from dotenv import find_dotenv, load_dotenv
from src.ingestion import build_knowledge_graph
from src.graph import build_graph

# Load environment variables from central root .env or local
load_dotenv(find_dotenv())


def main():
    # 1. Load data and seed local knowledge graph in Neo4j if database is active
    build_knowledge_graph()

    # 2. Compile state graph workflow
    app = build_graph()

    print("\n" + "="*60)
    print(" KG-RAG Engine Active (Knowledge Graph & Semantic Reasoning)")
    print(" Ingested Data -> Neo4j Subgraph Retrieval -> Groq Llama")
    print(" Type 'exit' to quit.")
    print("="*60)

    while True:
        question = input("\nAsk Question: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down KG-RAG workspace.")
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
            print(f"\n[Execution Error] Failed to complete KG-RAG workflow: {e}")


if __name__ == "__main__":
    main()
