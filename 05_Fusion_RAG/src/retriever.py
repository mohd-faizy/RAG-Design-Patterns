from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.fusion import reciprocal_rank_fusion

BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root
CHROMA_PATH = str(BASE_DIR / "chroma_db")


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
    """Uses Groq Llama-3.3 to generate 4 alternative search queries."""
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
        # Remove leading numbering/bullets like '1. ', '- ', '* '
        if len(line_clean) > 2 and line_clean[0].isdigit() and line_clean[1] in (".", ")"):
            line_clean = line_clean[2:].strip()
        elif line_clean[0] in ("-", "*"):
            line_clean = line_clean[1:].strip()

        if line_clean:
            queries.append(line_clean)

    # Always include the original query; cap at 4 total
    if question not in queries:
        queries.insert(0, question)

    return queries[:4]


def fusion_retrieve(question: str, retriever, llm) -> list:
    """
    Parallel Retrieval Layer (matches architectural diagram):
      1. Generate 4 query variations via Groq Llama-3.3
      2. Retrieve from ChromaDB for EACH query in PARALLEL (ThreadPoolExecutor)
      3. Merge results with Reciprocal Rank Fusion (RRF)
    """
    queries = generate_queries(question, llm)
    print(f"\n[Fusion RAG] Generated {len(queries)} queries for parallel retrieval:")
    for idx, q in enumerate(queries, 1):
        print(f"  {idx}. {q}")

    # True parallel retrieval — one thread per query variation
    all_results: list = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        future_to_idx = {
            executor.submit(retriever.invoke, query): idx
            for idx, query in enumerate(queries)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            all_results[idx] = future.result()

    # Reciprocal Rank Fusion across all per-query result lists
    fused_docs = reciprocal_rank_fusion(all_results)

    # Return top 5 fused documents for the generate node
    return fused_docs[:5]
