<div align="center">

  <h1>⚡ RAG Design Patterns</h1>

  <p>
    <b>A comprehensive, production-ready collection of 20 Retrieval-Augmented Generation (RAG) architecture implementations using Python & LangChain.</b>
  </p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/mohd-faizy/RAG-Design-Patterns/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License" />
    </a>
    <a href="https://github.com/mohd-faizy/RAG-Design-Patterns/stargazers">
      <img src="https://img.shields.io/github/stars/mohd-faizy/RAG-Design-Patterns?style=for-the-badge&logo=github&color=D97706" alt="GitHub Stars" />
    </a>
    <a href="https://github.com/mohd-faizy/RAG-Design-Patterns/network/members">
      <img src="https://img.shields.io/github/forks/mohd-faizy/RAG-Design-Patterns?style=for-the-badge&logo=github&color=2563EB" alt="GitHub Forks" />
    </a>
    <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
    <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
    <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI" />
    <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
    <img src="https://img.shields.io/badge/FAISS-Vector%20Store-00ADD8?style=for-the-badge" alt="FAISS" />
    <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-FF6B35?style=for-the-badge" alt="ChromaDB" />
    <img src="https://img.shields.io/badge/Architectures-20-brightgreen?style=for-the-badge" alt="20 Architectures" />
  </p>

  <br/>

</div>

---

## 📖 Overview

This repository is a **comprehensive reference implementation** of **20 RAG (Retrieval-Augmented Generation) design patterns**, built with **Python** and **LangChain**. Each architecture is implemented as a clean, well-documented Jupyter Notebook — covering everything from the foundational Standard RAG to cutting-edge Agentic and Deep Research RAG systems.

It serves as both a **learning curriculum** and a **production-ready reference** for AI engineers and researchers who want to deeply understand, compare, and deploy the right RAG strategy for their use case.

> 💡 **What is RAG?**
> Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval, enabling LLMs to access up-to-date, domain-specific information at inference time — dramatically reducing hallucinations and improving factual accuracy.

---

## 🗺️ RAG Architecture Roadmap

<p align="center">
  <b>🗺️ RAG Architecture Roadmap</b><br>
  <img src="_assets/rag-architecture-roadmap.png" alt="RAG Architecture Roadmap" width="100%">
</p>

The 20 architectures are organized into **5 progressive learning phases**, from foundational patterns to fully autonomous agentic systems:

```
Phase 1: Fundamentals         →  Standard, Hybrid, Contextual, Hierarchical RAG
Phase 2: Better Retrieval     →  Fusion, Multi-Source, Reranker-Centric, ColBERT RAG
Phase 3: Better Reasoning     →  Multi-Hop, Graph, KG-RAG, RAPTOR RAG
Phase 4: Self-Improving       →  Self-RAG, Corrective (CRAG), Feedback, Adaptive RAG
Phase 5: Agentic Systems      →  ReAct, Memory-Augmented, Agentic, Deep Research RAG
```

---

## 🏗️ The 20 RAG Architectures

### 🟢 Phase 1 — Fundamentals

| # | Architecture | Category | Key Idea | Best For | Primary Challenge |
|:---:|:---|:---|:---|:---|:---|
| 01 | **Standard RAG** | Foundation | Retrieve relevant chunks → Generate answer | General Q&A, simple apps | Baseline information retrieval |
| 02 | **Hybrid RAG** | Retrieval Enhancement | BM25 (sparse) + Vector (dense) search | Improved retrieval quality | Keyword vs semantic mismatch |
| 03 | **Contextual RAG** | Context Optimization | Expand query with surrounding context | Ambiguous queries, chatbots | Information loss during chunking |
| 04 | **Hierarchical RAG** | Large Corpus Retrieval | Coarse retrieval → Fine retrieval | Large documents, corpora | High noise in large search spaces |

### 🔵 Phase 2 — Better Retrieval

| # | Architecture | Category | Key Idea | Best For | Primary Challenge |
|:---:|:---|:---|:---|:---|:---|
| 05 | **Fusion RAG** | Query Expansion | Multiple queries → Fuse results | Boost recall & robustness | Poor/ambiguous user queries |
| 06 | **Multi-Source RAG** | Multi-Repository | Combine multiple knowledge sources | Enterprise, multi-doc systems | Fragmented data silos |
| 07 | **Reranker-Centric RAG** | Precision Retrieval | Retrieve many → Rerank → Use best | High-precision retrieval needs | Irrelevant results in top-K |
| 08 | **ColBERT RAG** | Advanced Retrieval | Token-level late interaction scoring | State-of-the-art retrieval | Missed fine-grained semantic matches |

### 🟠 Phase 3 — Better Reasoning

| # | Architecture | Category | Key Idea | Best For | Primary Challenge |
|:---:|:---|:---|:---|:---|:---|
| 09 | **Multi-Hop RAG** | Iterative Reasoning | Retrieve → Reason → Retrieve again | Complex multi-fact questions | Dispersed/multi-step information |
| 10 | **Graph RAG** | Graph-Based Retrieval | Traverse graph relationships | Relationship-heavy data | Disconnected entity discovery |
| 11 | **KG-RAG** | Knowledge Graphs | Entities + Relations → Graph traversal | Knowledge-intensive domains | Exact relational query failure |
| 12 | **RAPTOR RAG** | Recursive Summarization | Build a tree of summaries → Retrieve | Long documents, research | Loss of global context in details |

### 🟣 Phase 4 — Self-Improving Systems

| # | Architecture | Category | Key Idea | Best For | Primary Challenge |
|:---:|:---|:---|:---|:---|:---|
| 13 | **Self-RAG** | Self-Correcting | LLM decides when & what to retrieve | Dynamic, adaptive QA | Fixed/unnecessary retrieval calls |
| 14 | **Corrective RAG (CRAG)** | Self-Correcting | Critique → Detect error → Correct | High-accuracy applications | Hallucinating from wrong retrieval |
| 15 | **Corrective Feedback RAG** | Feedback-Driven | Collect feedback → Improve generation | Learning systems | Static errors in generated text |
| 16 | **Adaptive RAG** | Dynamic Routing | Route: vector / hybrid / web / none | Cost & latency optimization | Single-path query handling |

### 🔴 Phase 5 — Agentic Systems

| # | Architecture | Category | Key Idea | Best For | Primary Challenge |
|:---:|:---|:---|:---|:---|:---|
| 17 | **ReAct RAG** | Reasoning + Acting | Reason → Act (retrieve/search) → Reason | Complex tasks needing tools | Lack of tool-use orchestration |
| 18 | **Memory-Augmented RAG** | Persistent Memory | Retrieve from short & long-term memory | Conversational agents | Loss of multi-turn user context |
| 19 | **Agentic RAG** | Autonomous Agents | Plan → Act → Observe → Iterate | Autonomous, long-horizon tasks | Multi-step task decomposition |
| 20 | **Deep Research RAG** | Autonomous Research | Plan → Search → Synthesize → Report | Research, analysis, reports | Time-consuming multi-page search |

---

## 🔄 Production-Ready RAG Pipeline (2026 Stack)

The optimal modern production agent architecture combines the best of all phases:

```
User Query
    ↓
Adaptive Router          ← [Query Type, Domain, Complexity, History]
    ↓
Hybrid Retrieval         ← [BM25 (sparse) + Dense (semantic) search]
    ↓
Reranker                 ← [Cross-Encoder / ColBERT / Cohere Rerank]
    ↓
Multi-Hop Retrieval      ← [Iterative Search → Sub-Question Decomp.]
    ↓
Memory Retrieval         ← [Short-Term + Long-Term + Knowledge Memory]
    ↓
Agent Planning           ← [ReAct / Plan & Execute / Tree of Thoughts]
    ↓
Tool Usage               ← [Web Search / Code / APIs / Databases]
    ↓
Reflection (Self/CRAG)   ← [Hallucination Check / Consistency Check]
    ↓
Final Answer
```

---

## What Top AI Companies Use in Production

| Company / System | Architecture Stack |
|:---|:---|
| 1️⃣ **OpenAI Deep Research** | Deep Research RAG + Agentic RAG + Memory |
| 2️⃣ **Microsoft GraphRAG** | Graph RAG + KG-RAG |
| 3️⃣ **Perplexity AI** | Fusion RAG + Multi-Hop RAG + Rerankers |
| 4️⃣ **Anthropic Claude** | Agentic RAG + Tool Use + Memory |
| 5️⃣ **Google Gemini** | Deep Research RAG + Multi-Hop RAG |
| 6️⃣ **Meta Retrieval** | ColBERT + Hybrid Retrieval |
| 7️⃣ **LinkedIn Search** | ColBERT-style Retrieval |
| 8️⃣ **Netflix Search** | Graph + Knowledge Retrieval |

---

## 📂 Repository Structure

```
RAG-Design-Patterns/
│
├── 📁 01_Fundamentals/
│   ├── 01_Standard_RAG.ipynb
│   ├── 02_Hybrid_RAG.ipynb
│   ├── 03_Contextual_RAG.ipynb
│   └── 04_Hierarchical_RAG.ipynb
│
├── 📁 02_Better_Retrieval/
│   ├── 05_Fusion_RAG.ipynb
│   ├── 06_MultiSource_RAG.ipynb
│   ├── 07_Reranker_Centric_RAG.ipynb
│   └── 08_ColBERT_RAG.ipynb
│
├── 📁 03_Better_Reasoning/
│   ├── 09_MultiHop_RAG.ipynb
│   ├── 10_Graph_RAG.ipynb
│   ├── 11_KG_RAG.ipynb
│   └── 12_RAPTOR_RAG.ipynb
│
├── 📁 04_Self_Improving/
│   ├── 13_Self_RAG.ipynb
│   ├── 14_Corrective_RAG.ipynb
│   ├── 15_Feedback_RAG.ipynb
│   └── 16_Adaptive_RAG.ipynb
│
├── 📁 05_Agentic_Systems/
│   ├── 17_ReAct_RAG.ipynb
│   ├── 18_Memory_Augmented_RAG.ipynb
│   ├── 19_Agentic_RAG.ipynb
│   └── 20_Deep_Research_RAG.ipynb
│
├── 📁 _assets/                  # Architecture diagrams & cheat sheets
├── 📄 requirements.txt
├── 📄 .env.example
└── 📄 README.md
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.9+**
- **Git**
- **API Keys**: OpenAI / HuggingFace / Cohere *(as needed per notebook)*

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/mohd-faizy/RAG-Design-Patterns.git
cd RAG-Design-Patterns
```

**2. Set up the environment**

*Using `uv` (Recommended — fastest):*

```bash
uv venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
uv add -r requirements.txt
```

*Using `pip`:*

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

**3. Configure API keys**

```bash
cp .env.example .env
```

```ini
# .env
OPENAI_API_KEY=sk-...
HUGGINGFACEHUB_API_TOKEN=hf_...
COHERE_API_KEY=...
LANGCHAIN_API_KEY=...
```

**4. Launch Jupyter**

```bash
jupyter notebook
```

### Minimal Standard RAG Example

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. Load & Split Documents
loader = TextLoader("data/my_docs.txt")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(loader.load())

# 2. Create Vector Store
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 3. Build RAG Chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

# 4. Query
response = rag_chain.run("What is the key concept here?")
print(response)
```

---

## 🛠️ Tech Stack

| Component | Tools |
|:---|:---|
| **Framework** | LangChain, LangGraph |
| **LLMs** | OpenAI GPT-4o, HuggingFace, Ollama |
| **Embeddings** | OpenAI, HuggingFace (BAAI/bge), Cohere |
| **Vector Stores** | FAISS, ChromaDB, Pinecone, Weaviate |
| **Rerankers** | Cohere Rerank, Cross-Encoder, ColBERT |
| **Graph DBs** | Neo4j, NetworkX |
| **Notebooks** | Jupyter, Google Colab |

---

## 💾 Database & Storage Selection

| **Type** | **Examples** | **Strengths** | **Weaknesses** | **Best for** |
| --- | --- | --- | --- | --- |
| **Vector DB** | Pinecone; Qdrant; Weaviate; Milvus; FAISS | Fast semantic search; filtering; scalable; built for embeddings | Cost (managed); ops for self‑hosted FAISS/Milvus | Primary semantic retriever in RAG. |
| **Search Engine** | Elasticsearch / OpenSearch | Excellent full‑text, BM25, aggregations, mature tooling | Heavier infra; not native vectors (needs plugin) | Hybrid search, analytics, metadata filtering. |
| **Relational / NoSQL** | PostgreSQL (with pgvector); MongoDB | Strong structured queries, ACID, joins; pgvector adds embeddings | Not optimized for large vector scale | Small RAG corpora, metadata joins, transactional data. |
| **Graph DB** | Neo4j; Amazon Neptune | Models relationships, reasoning, path queries | Different query model; not for dense vector search | Knowledge graphs, entity linking, provenance. |
| **Object Storage** | AWS S3; GCS; Azure Blob | Cheap storage for raw docs, PDFs, media; easy ingestion | Not searchable by itself; needs indexing | Long‑term storage and batch ingestion. |

---

## 🧠 Embedding Models Selection

| **Embedding Model** | **Provider** | **Dimensionality** | **Strengths** | **Weaknesses** | **Best For** |
| --- | --- | --- | --- | --- | --- |
| **text-embedding-3-small/large** | OpenAI | 1536 / 3072 | Highly scalable, cheap, excellent general multilingual support, dynamic dimensions | API-dependent, potential latency spikes, data privacy concerns | Production general-purpose multilingual RAG |
| **BAAI/bge-large-en-v1.5** | Hugging Face (Local) | 1024 | Pinpoint English retrieval, open-source, runs fully local/private, zero cost | Requires local compute (GPU/CPU overhead) | Offline RAG, private/sensitive data pipelines |
| **cohere-embed-v3** | Cohere | 1024 | Built-in compression (binary/int8), state-of-the-art multilingual search, search-intent optimization | API cost, requires internet connection | Production global multi-language applications |
| **voyage-3** | Voyage AI | 1024 / 2048 | Domain-specific context (code, finance, law), extremely high retrieval accuracy | Less known, API cost, limits on concurrency | Highly specialized domains (code repositories, legal) |
| **multilingual-e5-large** | Microsoft (Local) | 1024 | Strong multilingual local retrieval, pre-trained on diverse datasets | Large memory footprint for local CPU | High-accuracy local multi-language deployments |

---

## ✂️ Chunking Strategies Selection

| **Strategy** | **Description** | **Strengths** | **Weaknesses** | **Best For** |
| --- | --- | --- | --- | --- |
| **Character / Token Splitting** | Splits text by a fixed number of characters or tokens with a defined overlap. | Simple, predictable, fast execution, low computation cost | Disregards semantic boundaries; splits sentences or paragraphs in half | Base RAG pipelines, simple structured text (logs, FAQs). |
| **Recursive Character Splitting** | Splits by a list of separators (e.g., `\n\n`, `\n`, ` `, `""`) recursively to keep paragraphs/sentences intact. | Extremely robust, maintains natural context groups, highly configurable | Can still lose high-level document hierarchy | General-purpose text, manuals, markdown, essays. |
| **Semantic Chunking** | Uses embedding similarity between sentences to find natural transition points and group text dynamically. | Groups text by actual semantic meaning; prevents splitting cohesive concepts | High computational overhead; slow indexing speeds | Research articles, complex novels, conversational transcriptions. |
| **Parent-Child (Hierarchical)** | Stores small chunks (child) for precise vector matching, but retrieves larger chunks (parent) for LLM context. | High precision matching with rich generation context | Complex indexing; higher storage footprint | Long-form books, legal contracts, comprehensive technical specs. |
| **Agentic / Semantic Summarization** | Summarizes entire sections or pages recursively (like RAPTOR) to capture high-level ideas. | Preserves global thematic context across huge documents | Very high LLM api cost and indexing time | Comprehensive academic summaries, multi-document thematic search. |

---

## 🎯 Reranker Models Selection

| **Reranker Model / Strategy** | **Type** | **Latency** | **Strengths** | **Weaknesses** | **Best For** |
| --- | --- | --- | --- | --- | --- |
| **Cohere Rerank v3** | Managed API | Medium (50-150ms) | Multilingual out-of-the-box, highly optimized for search relevance, handles diverse documents, cheap API cost | API dependency, requires network call | High-volume production multilingual systems |
| **Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2`)** | Local Model | Low-Medium (on GPU) | Fast, open-source, runs local (zero cost), excellent for English lexical/semantic mapping | Computationally heavy on CPU, limits sequence length (~512 tokens) | Local, private, or fast English-based RAG pipelines |
| **BAAI/bge-reranker-large** | Local Model | Medium (on GPU) | Exceptional multilingual performance, high capacity, state-of-the-art accuracy | Higher memory footprint and GPU requirement | High-precision open-source applications |
| **ColBERT Late Interaction (MaxSim)** | Multi-Vector | Very Fast | High token-level alignment accuracy, extremely fast query execution | High indexing storage overhead, complex index build stage | Dynamic, real-time token-level search pipelines |
| **LLM-as-a-Reranker (e.g., GPT-4o-mini)** | Generative | Slow (500ms+) | Extremely deep reasoning, matches complex instructions, highly flexible | Very expensive, extremely slow, susceptible to LLM output parsing failures | Complex reasoning tasks, low-volume/high-value RAG |

---

## 📊 Architecture Comparison Matrix

<p align="center">
  <img src="_assets/rag-comparison-matrix.png" alt="RAG Architecture Comparison Matrix" width="100%">
</p>

---

## ⚖️ Performance & Complexity Trade-offs

| Architecture | Complexity | Accuracy | Latency | Cost |
|:---|:---:|:---:|:---:|:---:|
| **01. Standard RAG** | 🟢 Low | ⭐⭐⭐ | Fast | $ |
| **02. Hybrid RAG** | 🟡 Medium | ⭐⭐⭐⭐ | Medium | $$ |
| **03. Contextual RAG** | 🟡 Medium | ⭐⭐⭐⭐ | Medium | $$ |
| **04. Hierarchical RAG** | 🟡 Medium | ⭐⭐⭐⭐ | Medium | $$ |
| **05. Fusion RAG** | 🟡 Medium | ⭐⭐⭐⭐ | Medium | $$ |
| **06. Multi-Source RAG** | 🟡 Medium | ⭐⭐⭐⭐ | Medium | $$ |
| **07. Reranker-Centric RAG** | 🟡 Medium | ⭐⭐⭐⭐⭐ | Medium | $$ |
| **08. ColBERT RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Fast | $$ |
| **09. Multi-Hop RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Slower | $$$ |
| **10. Graph RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Medium | $$$ |
| **11. KG-RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Medium | $$$ |
| **12. RAPTOR RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Medium | $$$ |
| **13. Self-RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Slower | $$$ |
| **14. Corrective RAG (CRAG)** | 🟠 High | ⭐⭐⭐⭐⭐ | Slower | $$$ |
| **15. Corrective Feedback RAG** | 🟠 High | ⭐⭐⭐⭐ | Medium | $$$ |
| **16. Adaptive RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Dynamic | $$ to $$$ |
| **17. ReAct RAG** | 🔴 V.High | ⭐⭐⭐⭐⭐ | Slow | $$$$ |
| **18. Memory-Augmented RAG** | 🟠 High | ⭐⭐⭐⭐⭐ | Medium | $$$ |
| **19. Agentic RAG** | 🔴 V.High | ⭐⭐⭐⭐⭐ | Slowest | $$$$ |
| **20. Deep Research RAG** | 🔴 V.High | ⭐⭐⭐⭐⭐ | Slowest | $$$$ |

---

## ✅ Key Considerations for Production RAG

- **Data Quality** — Keep data fresh and relevant; up-to-date docs are critical
- **Chunking Strategy** — Right chunk size improves retrieval significantly
- **Embedding Model** — Domain-specific embeddings work better
- **Retrieval Strategy** — Hybrid retrieval + reranking is powerful
- **Prompt Engineering** — Clear instructions improve results
- **Evaluation** — Use metrics: Faithfulness, Relevance, Recall
- **Latency vs Accuracy** — Balance query optimization with quality needs
- **Cost Management** — Optimize tokens, caching, and retrieval calls
- **Monitor for Hallucinations** — Implement reflection and self-critique loops

---

## 🤝 Contributing

Contributions are welcome! Whether it's a new architecture variant, improved documentation, or bug fixes — feel free to open an issue or submit a pull request.

```bash
# Fork → Clone → Branch → Commit → PR
git checkout -b feature/your-architecture-improvement
git commit -m "feat: add XYZ improvement to Agentic RAG"
git push origin feature/your-architecture-improvement
```

If you find this repository useful, please consider giving it a ⭐ — it helps others discover it!

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

## 🌐 Connect with Me

  <p>
    <a href="https://twitter.com/F4izy">
      <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter"/>
    </a>
    <a href="https://www.linkedin.com/in/mohd-faizy/">
      <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
    </a>
    <a href="https://github.com/mohd-faizy">
      <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
    </a>
  </p>

  <p>
    <sub>
      ⭐ If this repository helped you, please star it<br/>
      Made by <a href="https://github.com/mohd-faizy"><b>mohd-faizy</b></a>
    </sub>
  </p>

</div>
