"""FastAPI dependencies.

Thin accessors over the single :class:`~app.container.Container` built during startup. No
service is constructed per request: the graph driver, Mongo client and provider client all hold
connection pools that must outlive one request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.container import Container
from app.core.config import Settings
from app.demo.scenarios import ScenarioLibrary
from app.services.audit.service import AuditService
from app.services.orchestrator import Orchestrator
from app.services.policy.loader import PolicyRegistry


def get_container(request: Request) -> Container:
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only reachable if lifespan did not run
        raise RuntimeError("application container is not initialised")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings_dep(container: ContainerDep) -> Settings:
    return container.settings


def get_orchestrator(container: ContainerDep) -> Orchestrator:
    return container.orchestrator


def get_audit_service(container: ContainerDep) -> AuditService:
    return container.audit


def get_policies(container: ContainerDep) -> PolicyRegistry:
    return container.policies


def get_scenarios(container: ContainerDep) -> ScenarioLibrary:
    return container.scenarios


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
AuditDep = Annotated[AuditService, Depends(get_audit_service)]
PoliciesDep = Annotated[PolicyRegistry, Depends(get_policies)]
ScenariosDep = Annotated[ScenarioLibrary, Depends(get_scenarios)]


__all__ = [
    "AuditDep",
    "ContainerDep",
    "OrchestratorDep",
    "PoliciesDep",
    "ScenariosDep",
    "SettingsDep",
    "get_audit_service",
    "get_container",
    "get_orchestrator",
    "get_policies",
    "get_scenarios",
]
