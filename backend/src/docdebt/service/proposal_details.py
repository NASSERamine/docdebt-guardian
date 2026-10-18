"""Everything the dashboard shows about one proposal before a decision.

Nothing new is stored for this: the graph already checkpoints the method,
the findings, the original Javadoc and the verified repair for every
thread, so the details are read back from the checkpoint. A proposal whose
paused run is gone (an expired one) simply has no details.
"""

from __future__ import annotations

from difflib import unified_diff

from langgraph.checkpoint.base import BaseCheckpointSaver

from docdebt.llm.prompts.registry import SCREENED_ON, is_screened


def read_thread_values(checkpointer: BaseCheckpointSaver, thread_id: str) -> dict | None:
    """The latest saved state of a graph thread, without compiling the graph
    (so without building any model client); ``None`` if the thread is unknown."""
    saved = checkpointer.get_tuple({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
    if saved is None:
        return None
    return dict(saved.checkpoint.get("channel_values") or {})


def _finding_json(finding: dict) -> dict:
    keys = ("kind", "severity", "rule", "message", "field", "line", "quoted_doc", "quoted_code")
    return {key: finding.get(key) for key in keys}


def build_details(values: dict) -> dict:
    """The JSON the proposal page renders, from a thread's saved state."""
    unit = values["unit"]
    signature = unit["signature"]
    params = ", ".join(f"{p['type']} {p['name']}" for p in signature["parameters"])
    returns = "" if signature["is_constructor"] else f"{signature['return_type']} "
    throws = f" throws {', '.join(signature['throws'])}" if signature["throws"] else ""

    findings = [_finding_json(f) for f in values.get("structural_findings", [])]
    findings += [_finding_json(f) for f in values.get("semantic_findings", [])]
    original = unit.get("javadoc") or ""
    repaired = values.get("repaired_doc")
    indent = unit.get("indent", "")
    diff = (
        list(
            unified_diff(
                (indent + original).splitlines(),
                (indent + repaired).splitlines(),
                "current",
                "proposed",
                n=10,
                lineterm="",
            )
        )
        if repaired
        else []
    )
    return {
        "method": {
            "class_name": unit["class_name"],
            "name": signature["name"],
            "signature": f"{returns}{signature['name']}({params}){throws}",
            "file_path": unit["file_path"],
            "start_line": unit["start_line"],
            "end_line": unit["end_line"],
            "code": unit["code"],
            "javadoc": original,
            # Javadoc and code as they read in the file: their first lines were
            # cut at the first character, the indentation is stored apart.
            "source": "\n".join(unit.get("indent", "") + part for part in (original, unit["code"]) if part),
        },
        # Errors are the debt; warnings are hygiene notes that never count as debt.
        "debt": [f for f in findings if f["severity"] == "error"],
        "warnings": [f for f in findings if f["severity"] != "error"],
        "detection": {
            "status": values.get("detection_status"),
            "model": values.get("detection_model"),
            "error": values.get("detection_error"),
            # A model reusing the prompts screened for a sibling model (e.g. gpt-5.6-luna -> gpt-5.6).
            "screened_on": SCREENED_ON.get(values.get("detection_model") or ""),
            "screened": is_screened(values.get("detection_model") or "")
            if values.get("detection_model")
            else None,
        },
        "repair": {
            "status": values.get("repair_status"),
            "model": values.get("repair_model"),
            "attempts": values.get("repair_attempts"),
            "prompt_version": values.get("repair_prompt_version"),
            "repaired_doc": repaired,
            "changes": values.get("changes", []),
            "mechanism": "replacement"
            if values.get("repair_mechanism") == "hybrid"
            else (values.get("repair_mechanism") or "regeneration"),
            # Hybrid mechanism: what the post-validation did with each field the model answered.
            "field_log": values.get("field_log", []),
        },
        # The retrieved project context given to the model (RAG), excerpt by excerpt.
        "rag_context": values.get("rag_context", []),
        "indent": indent,
        "diff": diff[2:],  # without the "--- current" / "+++ proposed" header lines
        "final": values.get("final"),
    }
