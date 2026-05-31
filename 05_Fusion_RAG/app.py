from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from src.ingestion import create_vector_store
from src.graph import build_graph

# Load environment variables from central root .env

def main():
    # 1. Initialize database locally if needed
    create_vector_store()
    
    # 2. Compile our state graph workflow
    app = build_graph()
    
    print("\n" + "="*50)
    print(" Fusion RAG Engine Active (Multi-Query + RRF)")
    print(" Type 'exit' to quit.")
    print("="*50)
    
    while True:
        question = input("\nAsk Question: ").strip()
        
        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down Fusion RAG workspace.")
            break
            
        # Run state through the LangGraph engine
        result = app.invoke({"question": question})
        
        print("\n[Answer]:")
        print(result["answer"])
        print("-" * 30)

if __name__ == "__main__":
    main()