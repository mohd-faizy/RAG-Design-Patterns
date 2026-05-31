# KG-RAG (Knowledge Graph RAG) using LangGraph + Neo4j + Groq

A stateful, production-structured, and highly explainable implementation of the **Knowledge Graph Retrieval-Augmented Generation (KG-RAG)** pattern.

---

## 📖 What is KG-RAG?

Traditional RAG models retrieve unstructured, flat text chunks based on semantic proximity:
```
Question → Chunks → Answer
```

However, flat chunks fail to resolve complex, relational, or symbolic reasoning tasks where entities span multiple paragraphs or sources. 

**KG-RAG** models structured knowledge graphs to traverse entity links and semantic relations:
```
Question → Entity Linking → KG Traversal → Context Subgraph → LLM
```

By traversing connections explicitly in Neo4j, KG-RAG provides unparalleled explainability, robust multi-hop reasoning, and strict factual grounding.

---

## 🏗️ Architecture & State Workflow

### 1. Information Extraction & Retrieval Flow

```mermaid
graph TD
    Docs([Documents]) --> EntityExt["Entity Extraction"]
    EntityExt --> RelationExt["Relation Extraction"]
    RelationExt --> KG["Knowledge Graph {Neo4j}"]
    
    subgraph Knowledge Base Core
        KG --> Entities["Entities"]
        KG --> Relations["Relations"]
        KG --> Ontology["Ontology {Ontology Definitions}"]
    end
    
    Entities --> QueryEngine["Graph Query Engine {Cypher Query}"]
    Relations --> QueryEngine
    Ontology --> QueryEngine
    
    QueryEngine --> Subgraph["Relevant Subgraph"]
    Subgraph --> Groq["Groq LLM {Llama-3.3-70b}"]
    Groq --> Final([Final Answer])

    style Docs fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style Final fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style QueryEngine fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

### 2. State-Based Graph Schema

```
                      +-------------------+
                      |   retrieve_node   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   generate_node   |
                      +---------+---------+
                                |
                                v
                            [  END  ]
```

---

## 📁 Project Structure

The project has a modular, scalable design:

```bash
11_KG_RAG/
│
├── app.py               # Main CLI interactive loop entrypoint
├── requirements.txt     # Local project packages
├── .env                 # Environment variables configuration
│
│
└── src/
    ├── __init__.py      # Package initialization
    ├── state.py         # GraphState schema using TypedDict
    ├── prompts.py       # Modularized RAG prompt template
    ├── ontology.py      # Ontological definitions (Entities & Relations)
    ├── kg_builder.py    # Extracts semantic relations and merges into Neo4j
    ├── kg_retriever.py  # Localized Cypher subgraph query retriever
    ├── ingestion.py     # Document loader and builder executor
    └── graph.py         # LangGraph workflow compiler
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have configured the **centralized `.env`** file in the root folder of the repository workspace with your Groq and Neo4j connection keys:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### 2. Install Dependencies
Navigate to this directory and install the required modules:
```bash
pip install -r requirements.txt
```

### 3. Setup Neo4j Database
Ensure you have Neo4j running locally (e.g. Neo4j Desktop or local console):
```bash
neo4j console
```

### 4. Run the Sandbox
Boot the interactive knowledge graph console:
```bash
python app.py
```
*(If your Neo4j database is offline or unreachable, the system will output clean, descriptive warnings and continue to run gracefully.)*

---

## ⚖️ Strategic Advantage

| Feature | Traditional RAG | KG-RAG |
| :--- | :--- | :--- |
| **Data Format** | Unstructured, flat text chunks | **Structured entities & relationships** |
| **Query Mechanism** | Vector cosine similarity search | **Case-insensitive Cypher subgraph traversal** |
| **Reasoning Model** | Implicit semantic association | **Explicit symbolic reasoning** |
| **Explainability** | Low (hard to audit vector indices) | **High (relationships are traced and auditable)** |
| **Multi-Hop Traversal** | Very poor (fails to span chunks) | **Excellent (scales recursively across graph paths)** |
