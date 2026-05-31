from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


from src.ingestion import build_raptor_tree
from src.graph import build_graph



def main():
    # -------------------------
    # BUILD RAPTOR TREE
    # -------------------------
    tree = build_raptor_tree()

    # -------------------------
    # BUILD GRAPH
    # -------------------------
    app = build_graph(tree)

    print("\n" + "="*40)
    print(" RAPTOR RAG Active (Recursive Clustering & Summarization)")
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