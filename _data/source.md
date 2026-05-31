# Advanced RAG Design Patterns

This document outlines key technical patterns for modern Retrieval-Augmented Generation (RAG) systems.

## Architectural Trade-Offs

| Pattern | Retrieval Latency | Quality Gain | Use Case |
|---|---|---|---|
| Standard RAG | Low | Baseline | Simple question answering |
| Hybrid RAG | Medium | High | Mixed keyword/conceptual search |
| Contextual RAG | Low-Medium | High | Documents with vital structure |
| Hierarchical RAG | Medium | Very High | Long context reasoning |
| Reranker-Centric RAG | High | Maximum | Precision-critical domains |
| Multi-Hop RAG | Very High | Maximum | Complex multi-document synthesis |

## Best Practices
1. Always evaluate chunk sizes. 256-512 tokens are usually optimal.
2. Use cross-encoders for reranking rather than simple semantic cosine-similarity where latency permits.
3. Incorporate metadata filtering to restrict retrieval scope dynamically.
