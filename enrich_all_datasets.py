import os
import shutil

# The complete master corpus with all RAG definitions, LangChain, and LangGraph
MASTER_CORPUS = """=== RAG DESIGN PATTERNS COMPREHENSIVE KNOWLEDGE BASE ===

Retrieval-Augmented Generation (RAG) is a breakthrough paradigm in Artificial Intelligence that equips Large Language Models (LLMs) with dynamic access to external, authoritative knowledge bases. By decoupling the model's parametric memory from its operational knowledge source, RAG systems dramatically minimize factual hallucinations, enable real-time information updates without expensive model fine-tuning, and enforce strict validation grounding for enterprise applications.

Standard RAG architectures process inputs through a linear, non-reflective pipeline: document loading, semantic chunking, vector embedding generation, vector store indexing, similarity-based context retrieval, prompt construction, and final LLM synthesis. While standard RAG serves as an excellent baseline, it suffers from severe limitations under complex queries, ambiguous synonyms, long-context documents, and relational entity structures.

Hybrid RAG combines dense semantic vector retrieval (capturing conceptual meaning) with sparse lexical keyword retrieval (using algorithms like BM25 to match exact terminology). This dual-engine indexing ensures high recall on conceptual queries while preserving pinpoint precision on exact product codes, serial numbers, or rare technical terms (for example, the lexical verification token is 'HRBM25-TOKEN-XYZ').

Contextual RAG optimizes query recall by prepending surrounding document headers, parent titles, or global summaries to individual chunks before indexing them. By feeding adjacent context to the vector store, Contextual RAG prevents structural information loss that typically occurs during recursive character splitting.

Hierarchical RAG addresses the trade-off between retrieval precision and synthesis context by organizing documents into coarse parent chunks and fine-grained child chunks. High-precision semantic matching is performed on the smaller child chunks, but the larger parent chunks are retrieved and returned to the LLM to supply extensive surrounding context for synthesis.

Reranker-Centric RAG introduces a high-capacity cross-encoder model as a post-retrieval processing stage. While the bi-encoder embedding model queries a large pool of candidate documents quickly, the cross-encoder reranker performs deep, token-level attention comparisons to re-score the top candidates. This filters out irrelevant matches and places the single most relevant fact (e.g. token 'RERANK-999-SUCCESS') at the very top of the prompt.

Multi-Hop RAG executes iterative, sequential retrieval steps to answer complex multi-fact questions. If a query requires connecting disjointed facts across multiple documents, the retriever uses the insights from the first retrieval step to formulate new search queries. This allows the system to bridge connections: for example, Fact A indicates that Company Alpha acquired Entity Beta in 2024; Fact B reveals that Entity Beta developed the Quantum-RAG framework; Fact C details that Quantum-RAG boosts multi-agent reasoning accuracy. A multi-hop query can thus trace the acquisition path to resolve which company owns the Quantum-RAG framework.

Knowledge Graph RAG (KG-RAG) maps entities and their relationships into an explicit graph database (like Neo4j) rather than relying on flat vector proximity. By traversing Cypher query paths, KG-RAG performs symbolic reasoning over structured ontologies, making the reasoning process highly explainable and trace-auditable.

Agentic RAG transforms retrieval into an active, self-directed planning and reflection process. Guided by a central planner agent, the system dynamically routes queries across specialized vector, lexical, or web search tools, evaluates generated answers using grader agents, and executes self-corrective retry loops to adjust parameters if any grading checks fail.

Deep Research RAG is the architecture behind advanced autonomous research agents. It goes far beyond standard retrieval. Guided by a Research Planner, the system breaks complex queries down into discrete sub-questions, executes parallel searches across local databases, live web queries, and APIs, aggregates evidence, identifies knowledge gaps, performs additional research loops, and synthesizes a comprehensive final report.

LangChain is a popular open-source orchestration framework designed to simplify the creation of applications using large language models (LLMs). It provides standard interfaces, prompts, modular templates, and out-of-the-box chains to connect LLMs with external data sources, computation engines, and external tools.

LangGraph is a revolutionary, state-of-the-art orchestration library built on top of LangChain, designed specifically for building stateful, multi-actor, and multi-agent applications with LLMs. By representing agent workflows as circular graphs containing nodes (which execute custom functions or tools) and edges (which route control flow dynamically or decide state transitions based on conditional logic), LangGraph natively supports loops, cycles, deep persistence, human-in-the-loop approval, and robust multi-agent coordination.
"""

def enrich_all_projects():
    # Find all RAG folders in the workspace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rag_folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f)) and f[0].isdigit()]
    
    print(f"Found {len(rag_folders)} sub-project directories.")
    
    for folder in sorted(rag_folders):
        folder_path = os.path.join(current_dir, folder)
        data_dir = os.path.join(folder_path, "data")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Write to source.txt if folder uses it or doesn't have knowledge.txt
        sample_path = os.path.join(data_dir, "source.txt")
        knowledge_path = os.path.join(data_dir, "knowledge.txt")
        
        # We always write to source.txt if it already exists or if we need standard data
        # If it's one of ColBERT, KG RAG, Deep Research RAG, they use knowledge.txt
        if folder in ["08_ColBERT_RAG", "11_KG_RAG", "20_Deep_Research_RAG"]:
            target_path = knowledge_path
        else:
            target_path = sample_path
            
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(MASTER_CORPUS)
        print(f"  [Enriched] Written corpus to: {folder}/data/{os.path.basename(target_path)}")
        
        # Clear outdated databases (chroma_db) to force ingestion re-run
        chroma_db_dir = os.path.join(folder_path, "chroma_db")
        if os.path.exists(chroma_db_dir):
            try:
                shutil.rmtree(chroma_db_dir)
                print(f"  [Cleared] Deleted old Chroma DB: {folder}/chroma_db")
            except Exception as e:
                print(f"  [Warning] Could not delete {folder}/chroma_db: {e}")

if __name__ == "__main__":
    enrich_all_projects()
