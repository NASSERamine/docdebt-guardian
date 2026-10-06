"""LangGraph orchestration: one compiled graph processes one method at a time.

The graph itself contains no business logic — every node calls a component
already validated on its own (Auditor, Router/Judge, Repairer). This package
only wires them together, adds the human-review pause, and makes the whole
run resumable across a process restart via a SQLite checkpointer.
"""
