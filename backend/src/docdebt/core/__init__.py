"""Deterministic core: Java parsing, Javadoc parsing, diff reading, structural audit.

Nothing in this package calls an LLM. Everything here is exact and cheap,
which is why it always runs first in the pipeline.
"""
