import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "BAAI/bge-reranker-base"

# Lazy loader for reranker model and tokenizer to prevent slowdowns on module import
_tokenizer = None
_model = None

def get_reranker():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading Cross-Encoder Reranker: {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model

def rerank_documents(query: str, docs: list, top_k: int = 3) -> list:
    """
    Reranks documents using BAAI/bge-reranker-base cross-encoder model.
    """
    if not docs:
        return []

    tokenizer, model = get_reranker()

    # Form query-document pairs
    pairs = [[query, doc.page_content] for doc in docs]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        # squeeze(-1) to handle batch of size 1 correctly (avoiding scalar issues)
        scores = logits.squeeze(-1)
        # If batch size is 1, scores will be a scalar. Convert it to a 1-element list/tensor first.
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        
    scored_docs = list(zip(docs, scores.tolist()))
    
    # Sort by cross-encoder score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Cross-Encoder Reranking Scores ---")
    for i, (doc, score) in enumerate(scored_docs):
        snippet = doc.page_content[:60].replace("\n", " ")
        print(f"[{i+1}] Score: {score:+.4f} | Snippet: \"{snippet}...\"")
    print("--------------------------------------")

    # Select top k
    reranked_docs = [doc for doc, score in scored_docs[:top_k]]
    return reranked_docs
