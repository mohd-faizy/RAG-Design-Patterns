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
    
    print("\n" + "="*55)
    print(" Adaptive RAG Engine Active (Dynamic Routing)")
    print(" Routes: no_retrieval | vector | hybrid | web")
    print(" Type 'exit' to quit.")
    print("="*55)
    
    while True:
        question = input("\nAsk Question: ").strip()
        
        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down Adaptive RAG workspace.")
            break
            
        result = app.invoke({"question": question})
        
        print("\n[Answer]:")
        print(result["answer"])
        print("-" * 30)

if __name__ == "__main__":
    main()