"""``/health`` — dependency status, reported honestly.

The status this endpoint returns is the point of it. Every backend in this system has a
fallback: no Neo4j means the in-memory graph, no Mongo means the in-memory audit store, no API
key means the deterministic mock provider. Those fallbacks keep the prototype runnable from a
clean checkout — but a system that silently substituted an ephemeral store for a durable one, or
a scripted model for a real one, would be lying about what it is.

So ``ok`` means every configured dependency answered as configured, and ``degraded`` means
something is running on a substitute. ``degraded`` is not an error: the pipeline still evaluates
every request and every decision is still made and recorded. It is a statement about what the
answers are worth.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import ContainerDep
from app.services.audit.memory_repo import InMemoryAuditRepository

router = APIRouter(tags=["health"])


class DependencyStatus(BaseModel):
    name: str
    backend: str
    healthy: bool
    #: True when this backend is a substitute for something that was configured but is not
    #: answering, or for a real dependency that was never configured at all.
    fallback: bool = False
    detail: str = ""


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when nothing is running on a substitute backend")
    app: str
    environment: str
    default_policy_profile: str
    policy_profiles: list[str] = Field(default_factory=list)
    dependencies: list[DependencyStatus] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@router.get("/health", response_model=HealthResponse, summary="Dependency and backend status")
async def health(container: ContainerDep) -> HealthResponse:
    settings = container.settings
    deps: list[DependencyStatus] = []
    notes: list[str] = []

    # --- model provider ---
    provider = await container.llm.health_check()
    provider_fallback = settings.llm_provider == "anthropic" and container.llm.name != "anthropic"
    deps.append(
        DependencyStatus(
            name="llm",
            backend=container.llm.name,
            healthy=provider.healthy,
            fallback=provider_fallback,
            detail=provider.detail,
        )
    )
    if provider_fallback:
        notes.append(
            "CP_LLM_PROVIDER=anthropic was requested but the provider could not be "
            "initialised; responses come from deterministic fixtures."
        )
    elif container.llm.name == "mock":
        notes.append(
            "Running on the deterministic mock provider. Responses come from "
            "data/demo/scenarios.yaml, not from a live model."
        )

    # --- graph ---
    graph = await container.graph.health_check()
    graph_fallback = settings.neo4j_configured and container.graph.backend != "neo4j"
    deps.append(
        DependencyStatus(
            name="graph",
            backend=container.graph.backend,
            healthy=graph.healthy,
            fallback=graph_fallback,
            detail=graph.detail
            or f"{graph.entity_count or 0} entities, {graph.fact_count or 0} facts",
        )
    )
    if graph_fallback:
        notes.append(
            "CP_NEO4J_URI is set but Neo4j did not answer; grounding is reading the in-memory "
            "graph seeded from the same file."
        )

    # --- audit store ---
    audit = await container.audit.health_check()
    audit_fallback = settings.mongo_configured and audit.backend != "mongo"
    deps.append(
        DependencyStatus(
            name="audit",
            backend=audit.backend,
            healthy=audit.available,
            fallback=audit_fallback,
            detail=audit.detail,
        )
    )
    if audit_fallback:
        notes.append(
            "CP_MONGO_URI is set but MongoDB did not answer; audit records are being written "
            "to memory and will not survive a restart."
        )
    elif isinstance(container.audit_repo, InMemoryAuditRepository):
        notes.append(
            "Audit records are held in memory only. Set CP_MONGO_URI for durable storage."
        )

    unhealthy = any(not d.healthy for d in deps)
    substituted = any(d.fallback for d in deps)

    return HealthResponse(
        status="degraded" if (unhealthy or substituted) else "ok",
        app=settings.app_name,
        environment=settings.environment,
        default_policy_profile=container.policies.default_id,
        policy_profiles=container.policies.ids(),
        dependencies=deps,
        notes=notes,
    )


__all__ = ["DependencyStatus", "HealthResponse", "router"]
