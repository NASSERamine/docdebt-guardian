"""Assemble the graph's components for one repository, from the service settings.

What changes from one repository to another:

* the cascade: the local model always, the cloud model (``gpt-5.6-luna``)
  only if one is configured and the repository allows sending code to it;
* the Judge's confidence threshold (per repository, else the global one);
* the knowledge base used for retrieved context, if RAG is on.

A debt found by a model of the cascade is repaired by that same model, with
the configured mechanism. A model that has no screened regeneration wording
(every cloud model) always uses the hybrid mechanism, which needs none.
"""

from __future__ import annotations

import logging

from docdebt.core.auditor import Auditor
from docdebt.graph.build import _build_router
from docdebt.graph.nodes import GraphDeps
from docdebt.llm.catalog import BUILTIN_MODELS, get_model_spec
from docdebt.llm.client import ChatModelFactory, LLMClient, UsageTracker
from docdebt.pipeline import build_repairer
from docdebt.rag.embeddings import choose_embedder
from docdebt.rag.knowledge import KnowledgeBase
from docdebt.rag.retriever import RagProvider
from docdebt.repair.hybrid_repairer import HybridRepairer
from docdebt.repair.registry import LOCKED_REPAIR_WORDINGS
from docdebt.service.config import ServiceSettings
from docdebt.service.runtime_settings import RuntimeSettings, to_docdebt_settings
from docdebt.settings import Settings

logger = logging.getLogger("docdebt.service")


def model_overrides(service: ServiceSettings) -> dict[str, dict]:
    """Model specs adjusted by the environment: the cloud model's prices."""
    cloud = service.cloud_model
    if not cloud or (service.cloud_price_in_per_m is None and service.cloud_price_out_per_m is None):
        return {}
    try:
        spec = get_model_spec(cloud).model_dump()
    except KeyError:
        return {}
    spec.pop("id", None)
    if service.cloud_price_in_per_m is not None:
        spec["usd_per_1k_input"] = service.cloud_price_in_per_m / 1000
    if service.cloud_price_out_per_m is not None:
        spec["usd_per_1k_output"] = service.cloud_price_out_per_m / 1000
    return {cloud: spec}


def cloud_model_known(service: ServiceSettings) -> bool:
    return bool(service.cloud_model) and service.cloud_model in BUILTIN_MODELS


def build_repairer_for(
    settings: Settings,
    model_id: str,
    mechanism: str,
    *,
    factory: ChatModelFactory | None = None,
    tracker: UsageTracker | None = None,
):
    """The repairer that repairs with ``model_id``."""
    if mechanism == "regeneration" and model_id in LOCKED_REPAIR_WORDINGS:
        return build_repairer(settings, model_id, factory=factory, tracker=tracker)
    client = LLMClient(get_model_spec(model_id, settings.model_specs()), factory=factory, tracker=tracker)
    return HybridRepairer(client, auditor_config=settings.auditor.to_config())


def build_service_deps(
    service: ServiceSettings,
    runtime: RuntimeSettings,
    policy: dict,
    *,
    owner: str | None = None,
    repo: str | None = None,
    tracker: UsageTracker | None = None,
    factory: ChatModelFactory | None = None,
    knowledge: KnowledgeBase | None = None,
    embedder=None,
) -> GraphDeps:
    """Everything the graph calls, for one repository (``policy`` is its
    ``ReviewStore.repo_settings``)."""
    tracker = tracker or UsageTracker()
    allow_cloud = bool(policy.get("allow_cloud")) and cloud_model_known(service)
    settings = to_docdebt_settings(
        runtime,
        cloud_model=service.cloud_model,
        allow_cloud=allow_cloud,
        confidence_threshold=policy.get("confidence_threshold"),
        models=model_overrides(service),
    )
    router = None if runtime.structural_only else _build_router(settings, None, factory, tracker)
    repairers = {
        model: build_repairer_for(settings, model, runtime.repair_mechanism, factory=factory, tracker=tracker)
        for model in settings.router.cascade
    }
    rag = None
    use_rag = bool(policy.get("rag_enabled", True)) and (runtime.rag_repair or runtime.rag_detection)
    if use_rag and owner and repo:
        rag = RagProvider(
            knowledge,
            owner,
            repo,
            sources=tuple(runtime.rag_sources),
            top_k=runtime.rag_top_k,
            embedder=embedder
            if embedder is not None
            else choose_embedder(service.rag_embeddings, allow_cloud=allow_cloud, probe=knowledge is not None),
        )
        if router is not None and runtime.rag_detection:
            router.context_provider = rag
    return GraphDeps(
        auditor=Auditor(settings.auditor.to_config()),
        router=router,
        repairer=repairers[settings.router.cascade[0]],
        repairers=repairers,
        rag=rag,
        rag_repair=rag is not None and runtime.rag_repair,
        rag_detection=rag is not None and runtime.rag_detection,
    )
