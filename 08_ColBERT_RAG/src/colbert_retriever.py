import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import native colbert libraries
import platform
try:
    if platform.system() == "Windows":
        # Native Stanford ColBERT relies on C++ extensions, GCC, CUDA, and Linux-specific APIs (e.g. fcntl)
        # We force simulation mode on Windows to guarantee stable, error-free execution
        COLBERT_AVAILABLE = False
    else:
        from colbert.infra import Run, RunConfig
        from colbert import Indexer, Searcher
        COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False

INDEX_PATH = "colbert_index"
COLLECTION_PATH = "collection.txt"


# --------------------------------
# CREATE COLLECTION FILE
# --------------------------------
def build_collection(chunks):
    print("Writing document chunks to collection...")
    with open(COLLECTION_PATH, "w") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    print("Successfully wrote collection.txt!")


# --------------------------------
# BUILD COLBERT INDEX
# --------------------------------
def build_index():
    if COLBERT_AVAILABLE:
        print("[ColBERT] Building native index using Stanford ColBERT...")
        try:
            with Run().context(RunConfig(nranks=1)):
                config = {
                    "nbits": 2
                }
                indexer = Indexer(checkpoint="colbert-ir/colbertv2.0")
                indexer.index(
                    name=INDEX_PATH,
                    collection=COLLECTION_PATH,
                    overwrite=True
                )
            print("[ColBERT] Native index created successfully!")
            return
        except Exception as e:
            print(f"[ColBERT Native Indexing Warning] Failed: {e}. Falling back to simulation mode.")
    
    print("[ColBERT] Running in Simulation Mode. Late Interaction Simulator does not require index files.")


# --------------------------------
# SEARCH USING COLBERT
# --------------------------------
class ColBERTRetriever:
    def __init__(self):
        self.native_active = False
        self.initialized = False
        self.collection = []
        self.model = None

    def _lazy_init(self):
        if self.initialized:
            return

        print("[ColBERT] Performing lazy initialization of retriever...")
        if os.path.exists(COLLECTION_PATH):
            with open(COLLECTION_PATH, "r") as f:
                self.collection = [line.strip() for line in f if line.strip()]
        else:
            self.collection = []
            print("[ColBERT Warning] collection.txt not found. Collection is empty.")

        if COLBERT_AVAILABLE:
            try:
                self.searcher = Searcher(index=INDEX_PATH)
                self.native_active = True
                print("[ColBERT] Native Stanford ColBERT Searcher initialized.")
            except Exception as e:
                print(f"[ColBERT Warning] Native searcher failed to initialize: {e}. Activating Simulator.")

        if not self.native_active:
            print("[ColBERT] Activating Late Interaction MaxSim Simulator (BAAI/bge-small-en-v1.5)...")
            self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            print("[ColBERT] Late Interaction Simulator ready.")

        self.initialized = True

    def retrieve(self, query, top_k=3):
        self._lazy_init()

        if not self.collection:
            print("[ColBERT Warning] Retrieval queried, but collection is empty!")
            return []

        if self.native_active:
            try:
                results = self.searcher.search(query, k=top_k)
                # Map native results back to collection
                passages = []
                for passage_id in results[0]:
                    if passage_id < len(self.collection):
                        passages.append(self.collection[passage_id])
                return passages
            except Exception as e:
                print(f"[ColBERT Native Search Warning] {e}. Falling back to simulator.")

        # Late Interaction Simulation Mode
        # 1. Tokenize query into words (tokens)
        q_tokens = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 1]
        if not q_tokens:
            q_tokens = [query.lower()]

        # Encode query tokens (shape: [N_Q, D])
        q_embeddings = self.model.encode(q_tokens)
        if len(q_embeddings.shape) == 1:
            q_embeddings = np.expand_dims(q_embeddings, axis=0)

        scores = []
        for passage in self.collection:
            # Tokenize document passage into words (tokens)
            d_tokens = [w.lower() for w in re.findall(r'\b\w+\b', passage) if len(w) > 1]
            if not d_tokens:
                d_tokens = [passage.lower()]

            # Encode document tokens (shape: [N_D, D])
            d_embeddings = self.model.encode(d_tokens)
            if len(d_embeddings.shape) == 1:
                d_embeddings = np.expand_dims(d_embeddings, axis=0)

            # Normalize embeddings to calculate cosine similarity via dot product
            q_norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
            d_norms = np.linalg.norm(d_embeddings, axis=1, keepdims=True)
            
            q_normed = q_embeddings / (q_norms + 1e-9)
            d_normed = d_embeddings / (d_norms + 1e-9)

            # Matrix of cosine similarities: [N_Q, N_D]
            sim_matrix = np.dot(q_normed, d_normed.T)

            # MaxSim calculation: for each query token, take the maximum similarity over all document tokens
            max_sims = np.max(sim_matrix, axis=1)

            # Score is the sum of MaxSims
            score = np.sum(max_sims)
            scores.append((passage, score))

        # Sort in descending order and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [passage for passage, _ in scores[:top_k]]
