"""Memory subsystem — vector index on top of the Markdown wiki.

- `embeddings.py`: Ollama `/api/embed` client (nomic-embed-text, 768-d).
- `index.py`: sqlite-vec wrapper for upsert + top-K search.
- `build_index.py`: one-shot + on-startup rebuild over `wiki/`.

Retrieval sits on top of this in `processors/vector_retrieval.py`. The
wiki itself is the source of truth; this package is a retrieval shortcut.
"""
