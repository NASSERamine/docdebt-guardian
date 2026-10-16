"""Retrieval-augmented generation (RAG): the project's knowledge base.

* ``embeddings``: turn text into vectors (Ollama locally, OpenAI when the
  repository allows the cloud) or fall back to lexical BM25 scoring.
* ``knowledge``: the per-repository index (documented methods, tests,
  README sections, Javadoc validated by reviewers), stored in SQLite.
* ``retriever``: build the context for one method from the file it lives
  in and from the knowledge base, with a record of every excerpt used.
"""
