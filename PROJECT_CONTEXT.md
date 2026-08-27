# ControlPlane.ai — Project Context for AI Coding Agents

## Executive Summary

We are building **ControlPlane.ai**, a working prototype for the Accenture Innovation Challenge 2026 Round 2, Problem Track 1.

The official brief expands the Round 1 Responsible AI Checker concept into a working prototype. The brief explicitly says the architecture is open-ended, proprietary enterprise data is not required, and reasonable assumptions/simulated data are acceptable.

Our prototype is a **Responsible AI control layer** that evaluates LLM outputs for:
- performance/grounding risk,
- cost/performance signals,
- responsibility/safety/privacy risks,

then chooses a configurable action:
- allow,
- edit/redact,
- flag for review,
- block.

## Why This Project

The team already has:
- Graph RAG experience
- Distributed backend/rate-limiter experience

We want to use those capabilities where they make architectural sense.

### Graph RAG use
Use Graph RAG for trusted grounding verification and policy/entity relationships.

### Distributed-systems influence
Use the fast-path/deep-path philosophy:
- cheap checks quickly for normal responses
- expensive checks selectively for suspicious responses
- asynchronous audit work where appropriate

Do NOT blindly copy a rate limiter into ControlPlane.

## Core User Story

A business uses an LLM for an internal or customer-facing workflow.

1. User sends a prompt.
2. LLM generates a response.
3. ControlPlane intercepts/observes the response.
4. ControlPlane runs configured checks.
5. Risk signals are combined.
6. A policy determines the action.
7. User receives:
   - original response,
   - edited response,
   - or safe fallback/block message.
8. Every decision is auditable.

## Key Engineering Principles

### Principle 1 — The LLM is not the quantitative source of truth

Use deterministic logic, databases, statistics, retrieval, and rules where appropriate.

### Principle 2 — Evidence before certainty

If grounding evidence is unavailable or contradictory:
- mark uncertainty,
- do not invent certainty,
- flag/abstain when the configured policy requires it.

### Principle 3 — Explain every decision

A risk decision must answer:
- what went wrong?
- which detector found it?
- what evidence supports the detector?
- which policy triggered?
- what action was taken?

### Principle 4 — Configuration over hard-coded policy

Risk thresholds, enabled checks, and action rules should be configurable.

### Principle 5 — Prototype first

The deadline is close. Build a working, demonstrable MVP before adding sophisticated infrastructure.

## Current Preferred Technology Direction

Use the existing repository's stack where possible.

Preferred backend:
- Python
- FastAPI

Preferred AI:
- provider-agnostic LLM adapter

Preferred graph:
- Neo4j or the graph DB already present

Preferred relational persistence:
- PostgreSQL or existing relational DB

Optional:
- Redis for caching/temporary fast state
- Celery/background worker for async audit tasks

Frontend:
- keep the repository's existing frontend if present; otherwise React/Next.js is acceptable.

## Expected Core Services

### LLM Service
Responsible only for model interaction.

### Grounding Service
Responsible for claim extraction and graph/document evidence retrieval.

### PII Service
Responsible for detection and redaction.

### Safety/Policy Service
Responsible for safety and configurable policy checks.

### Risk Service
Combines detector results into component and overall scores.

### Decision Service
Maps risk + policy to allow/edit/review/block.

### Audit Service
Persists traceable events.

## Example Risk Response Schema

```json
{
  "request_id": "req_123",
  "decision": "BLOCK",
  "overall_risk_score": 0.91,
  "signals": {
    "grounding": {
      "status": "contradicted",
      "score": 0.95,
      "evidence": ["Australia-CAPITAL-Canberra"]
    },
    "pii": {
      "detected": false,
      "score": 0.0
    },
    "safety": {
      "status": "pass",
      "score": 0.0
    },
    "cost": {
      "token_count": 580,
      "latency_ms": 420
    }
  },
  "reason": "Generated claim contradicts trusted evidence."
}
```

## Coding Instructions

Before coding:
1. inspect repository structure;
2. run the current application;
3. identify what already exists;
4. avoid replacing working code unnecessarily;
5. identify missing modules against `SYSTEM_REQUIREMENTS.md`.

While coding:
- use typed Python/Pydantic models where appropriate;
- isolate provider-specific code;
- keep secrets in env vars;
- provide mock mode for demos/tests;
- do not make tests depend on live LLM calls;
- handle dependency failures explicitly;
- add logging without leaking sensitive payloads;
- write small modules;
- keep API contracts stable.

After coding:
- run tests;
- run the application;
- test at least safe, hallucination, PII, unsafe, and uncertain scenarios;
- update documentation;
- only claim completed features that actually run.

## What Claude Should Prioritize

### Priority 0 — Must work
- backend starts
- `/health`
- `/chat`
- LLM adapter
- grounding detector
- PII detector/redactor
- safety/policy check
- risk score
- allow/edit/block
- audit logging
- deterministic demo fixtures
- tests

### Priority 1 — Demo quality
- frontend/dashboard
- risk charts
- evidence display
- audit table
- latency/token telemetry
- policy configuration

### Priority 2 — Advanced
- bias probing
- drift
- asynchronous deep analysis
- feedback learning
- multi-model routing

## Official-brief Alignment

Round 2 specifically asks the prototype to address the deeper realities of:
- different risk tolerance/latency budgets across AI use cases,
- overlapping bias/hallucination/privacy risks,
- lack of reliable real-time ground truth,
- alert fatigue,
- multi-turn/agentic downstream risk,
- changing regulatory expectations,
- external foundation-model APIs.

The brief highlights solutioning areas including:
- rule-based and statistical anomaly checks,
- AI-as-judge,
- retrieval/source verification,
- PII/entity detection,
- confidence scoring,
- tiered decisions,
- inline/pre-response/post-hoc placement,
- parallel checks,
- configurable governance/policy layers,
- audit trails,
- feedback loops,
- false-positive/false-negative metrics.

We do not need to implement all of these. We should select a coherent subset that is demonstrable.

## Definition of a Strong Prototype

A reviewer should be able to:
1. send a prompt;
2. see an LLM response;
3. see ControlPlane evaluate it;
4. see evidence from the graph/source;
5. see risk signals;
6. see a decision;
7. see why that decision happened;
8. inspect the audit record.

That end-to-end path is more important than adding many unfinished features.
