"""Assemble the graph's components for one repository.

Fixed, deterministic configuration:

* **one model**, chosen on the Model page (:mod:`docdebt.service.llm_settings`),
  for detection and repair alike, at temperature 0;
* **one detection on the whole Javadoc**: the exact rules of the Auditor,
  then one call with the model's fixed zero-shot wording; debt when a rule
  fails OR the model says so, with the model's explanation;
* **retrieval always on** (siblings + hybrid BM25/vector search), given to
  detection and repair;
* **Replacement** (the research's hybrid mechanism) computes the proposal;
  **Regeneration** is computed on demand from the proposal page.

Without a saved model configuration, only the exact rules run and nothing
is proposed (``GraphDeps.router`` is ``None`` and the repairer cannot be
built): the pull request and the status banner say why.
"""

from __future__ import annotations

import logging
import warnings

from docdebt.core.auditor import Auditor
from docdebt.detection.judge import Judge, JudgeConfig
from docdebt.detection.router import Router, RouterPolicy
from docdebt.graph.nodes import GraphDeps
from docdebt.llm.client import ChatModelFactory, LLMClient, UsageTracker
from docdebt.llm.config import ModelSpec
from docdebt.llm.prompts.registry import PromptRegistry
from docdebt.rag.knowledge import KnowledgeBase
from docdebt.rag.retriever import RagProvider
from docdebt.repair.hybrid_repairer import HybridRepairer
from docdebt.repair.registry import LOCKED_REPAIR_WORDINGS, RepairRegistry
from docdebt.repair.repairer import RepairConfig, Repairer

logger = logging.getLogger("docdebt.service")

# Tests replace the provider calls here (a scripted chat model); ``None`` = the real providers.
CHAT_FACTORY: ChatModelFactory | None = None

DETECTION_PATTERN = "zeroshot"  # the best mean pattern of the research, with and without context
REGENERATION_PATTERN = "zeroshot"  # no few-shot file needed; the wording is fixed per model


def service_judge_config() -> JudgeConfig:
    """One vote, no threshold, every explanation kept, context always given."""
    return JudgeConfig(
        patterns=[DETECTION_PATTERN],
        samples_per_pattern=1,
        explain=True,
        require_citation=False,
        confidence_threshold=0.0,
        use_context=True,
        keep_unverified_explanations=True,
    )


def build_regenerator(
    spec: ModelSpec, *, tracker: UsageTracker | None = None, factory: ChatModelFactory | None = None
) -> Repairer:
    """The Regeneration repairer of ``spec``'s model (asked from the proposal page).

    Models without a regeneration wording screened in the research use the
    first zero-shot wording; the proposal says the wording was not screened.
    """
    overrides = {} if spec.id in LOCKED_REPAIR_WORDINGS else {spec.id: {REGENERATION_PATTERN: 0}}
    client = LLMClient(spec, factory=factory or CHAT_FACTORY, tracker=tracker)
    return Repairer(client, RepairRegistry(overrides=overrides), config=RepairConfig(pattern=REGENERATION_PATTERN))


def regeneration_screened(model_id: str) -> bool:
    return REGENERATION_PATTERN in LOCKED_REPAIR_WORDINGS.get(model_id, {})


def build_service_deps(
    spec: ModelSpec | None,
    *,
    owner: str | None = None,
    repo: str | None = None,
    tracker: UsageTracker | None = None,
    factory: ChatModelFactory | None = None,
    knowledge: KnowledgeBase | None = None,
    embedder=None,
) -> GraphDeps:
    """Everything the graph calls for one repository, with the model ``spec``
    (``None``: no model configured, exact rules only)."""
    tracker = tracker or UsageTracker()
    factory = factory or CHAT_FACTORY
    rag = RagProvider(knowledge, owner, repo, embedder=embedder) if owner and repo else None
    auditor = Auditor()
    if spec is None:
        return GraphDeps(auditor=auditor, router=None, repairer=None, rag=rag)  # type: ignore[arg-type]

    client = LLMClient(spec, factory=factory, tracker=tracker)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        judge = Judge(client, PromptRegistry(), service_judge_config())
    router = Router(
        {spec.id: judge},
        RouterPolicy(cascade=[spec.id], allow_cloud=True),
        tracker,
        context_provider=rag,
    )
    repairer = HybridRepairer(client)
    return GraphDeps(
        auditor=auditor,
        router=router,
        repairer=repairer,
        repairers={spec.id: repairer},
        rag=rag,
        rag_repair=rag is not None,
        rag_detection=rag is not None,
    )
