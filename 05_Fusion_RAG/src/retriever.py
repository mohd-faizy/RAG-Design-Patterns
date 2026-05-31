from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.fusion import reciprocal_rank_fusion

CHROMA_PATH = "chroma_db"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def generate_queries(question: str, llm) -> list[str]:
    """Generates 4 alternative search queries for the user question."""
    prompt = f"""
    Generate 4 alternative search queries for the following question to retrieve the most relevant information.
    Provide each query on a new line. Do NOT include numbers, bullets, or any introductory text. Just output the raw query text.

    Question:
    {question}
    """
    response = llm.invoke(prompt)
    
    # Extract and clean queries
    queries = []
    for line in response.content.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue
        # Remove numbers and bullet symbols like '1. ', '- ' at the start of a query variation
        if len(line_clean) > 2 and line_clean[0].isdigit() and (line_clean[1] == '.' or line_clean[1] == ')'):
            line_clean = line_clean[2:].strip()
        elif line_clean.startswith('-') or line_clean.startswith('*'):
            line_clean = line_clean[1:].strip()
        
        if line_clean:
            queries.append(line_clean)
            
    # Always include the original query as one of the search queries to ensure coverage,
    # and cap at 4 queries.
    if question not in queries:
        queries.insert(0, question)
        
    return queries[:4]

def fusion_retrieve(question: str, retriever, llm) -> list:
    """Coordinates multi-query generation, parallel retrieval, and rank fusion."""
    queries = generate_queries(question, llm)
    print(f"\n[Fusion RAG] Generated Queries for Retrieval:")
    for idx, q in enumerate(queries, 1):
        print(f"  {idx}. {q}")
    
    all_results = []
    for query in queries:
        docs = retriever.invoke(query)
        all_results.append(docs)
        
    # Fuse documents using Reciprocal Rank Fusion
    fused_docs = reciprocal_rank_fusion(all_results)
    
    # Return top 5 fused documents
    return fused_docs[:5]
