from pathlib import Path
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

from src.ingestion import build_knowledge_graph
from src.graph import build_graph

# Load environment variables from central root .env

def main():
    # 1. Triplet Ingestion pipeline
    build_knowledge_graph()
    
    # 2. Compile our state graph workflow
    app = build_graph()
    
    print("\n" + "="*50)
    print(" Graph RAG Engine Active (Neo4j + LangGraph)")
    print(" Type 'exit' to quit.")
    print("="*50)
    
    while True:
        question = input("\nAsk Question: ").strip()
        
        if not question:
            continue
        if question.lower() == "exit":
            print("Shutting down Graph RAG workspace.")
            break
            
        # Run state through the LangGraph engine
        result = app.invoke({"question": question})
        
        print("\n[Answer]:")
        print(result["answer"])
        print("-" * 30)

if __name__ == "__main__":
    main()