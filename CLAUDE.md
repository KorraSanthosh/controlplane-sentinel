# CLAUDE.md — ControlPlane.ai

## 1. Project Identity

**Project:** ControlPlane.ai  
**Hackathon:** Accenture Innovation Challenge 2026 — Round 2  
**Track:** Problem Track 1 — ControlPlane.ai  
**Goal:** Build a working prototype of a Responsible AI Checker that sits around an AI application/model and evaluates AI responses for performance, cost, and responsibility risks before or immediately after delivery.

## 2. Source of Truth

The official Round 2 brief is the primary source for requirements. It explicitly states:

- Round 2 requires a more complete solution and a working prototype demonstrating the core mechanism, even on a limited or simulated scope.
- There is no single mandated architecture.
- Real enterprise proprietary data is not expected; simulated/illustrative data is acceptable.
- The solution can selectively use the suggested solutioning areas; it does not need to implement every listed idea.

When implementation choices are made beyond the brief, treat them as project design decisions, not Accenture-mandated requirements.

## 3. Core Product Concept

ControlPlane is an AI governance/control middleware layer.

High-level flow:

User -> AI Application -> LLM -> ControlPlane Analysis -> Decision -> Response

The ControlPlane evaluates the model response across:

1. Performance
   - Grounding / hallucination risk
   - Confidence / consistency signals
   - Optional drift/anomaly signals

2. Cost
   - Token usage
   - Latency
   - Estimated model/API cost
   - Rework/retry signals where available

3. Responsibility
   - PII / sensitive-data leakage
   - Unsafe content
   - Policy violations
   - Optional bias probing

The final decision is tiered:

- ALLOW
- EDIT / REDACT
- FLAG / HUMAN REVIEW
- BLOCK

The system must produce an audit record explaining why the decision was made.

## 4. Important Design Principle

Do NOT build a collection of unrelated AI checks.

Build one coherent pipeline:

Request
  -> Model
  -> Fast screening
  -> Risk signals
  -> Risk aggregation
  -> Policy decision
  -> Action
  -> Audit/telemetry

Use a fast path for cheap checks and a deep path for suspicious cases.

## 5. Graph RAG / Knowledge Graph Role

The team already has Graph RAG experience. Use it deliberately for **grounding verification**, not as a buzzword.

Expected flow:

AI response
  -> claim/entity extraction
  -> retrieve relevant graph evidence
  -> compare claim with trusted graph facts
  -> produce grounding status/confidence
  -> contribute to risk score

Example:

Claim: "The capital of Australia is Sydney."

Trusted graph:
Australia --CAPITAL--> Canberra

Result:
- contradiction / unsupported claim
- grounding risk increases
- decision engine may block or flag depending on policy

Do not claim that Graph RAG proves causality. It only provides evidence/relationship-based verification.

## 6. Architecture Requirements

Recommended logical modules:

- API layer
- LLM adapter
- Risk orchestration engine
- Grounding checker
- PII detector/redactor
- Safety/policy checker
- Cost/latency telemetry
- Risk scoring engine
- Decision engine
- Audit store
- Optional asynchronous deep-analysis worker
- Dashboard/frontend

Keep modules independently testable.

## 7. Implementation Guidance

### Backend
Use Python + FastAPI.

### LLM
Use a provider abstraction. Do not hard-code a model-specific implementation throughout the codebase.

Create an interface such as:

- generate()
- evaluate()
- count_usage() / parse_usage()
- health_check()

### Graph
Use Neo4j or another graph layer behind a small repository/service abstraction.

### Persistent storage
Use a relational database such as PostgreSQL for audit events, policies, requests, and metrics summaries.

### Fast state/cache
Redis may be used for caching, temporary state, request counters, or async job coordination. It is optional for the first functional MVP.

### Frontend
Use React/Next.js or the existing frontend stack already present in the repository. Prefer reuse over rewriting.

### Deployment
Dockerize the backend and frontend only if the existing codebase makes that practical.

## 8. Risk Model

Every analyzed response should produce structured signals, for example:

- grounding_score
- pii_risk
- safety_risk
- bias_risk (optional/when enabled)
- cost_score
- latency_ms
- overall_risk_score
- decision

Do not present arbitrary thresholds as industry standards. Thresholds are prototype policy configuration and must be documented as assumptions.

## 9. Decision Policy

A policy maps risk signals to actions.

Example prototype policy:

LOW:
- allow
- log

MEDIUM:
- redact/edit when possible
- warn or flag
- optionally human review

HIGH:
- block
- provide safe fallback
- create high-severity audit event

The exact thresholds must be configuration-driven.

## 10. Auditability

Every decision should be traceable.

Minimum audit fields:

- request_id
- timestamp
- use_case
- model/provider
- response hash or safe response reference
- risk signals
- overall risk score
- triggered policies
- final decision
- processing latency
- token usage if available
- estimated cost if available

Never store secrets or unnecessary sensitive payloads in plaintext.

## 11. Performance / Latency Philosophy

The official brief asks how the checker can protect latency.

Implement the architecture so that:

- cheap checks can run inline
- independent checks can run in parallel where practical
- expensive analysis can be triggered only for suspicious cases
- non-critical audit work can run asynchronously
- the system records latency overhead

Do not make every request pass through every expensive check.

## 12. Safety Rules

Never create a fake claim of medical/legal/financial certainty just to make a demo look stronger.

The prototype should clearly distinguish:

- detected risk
- evidence available
- model-generated judgment
- policy decision

When evidence is insufficient, prefer:
- flag
- review
- abstain
rather than inventing certainty.

## 13. Engineering Rules for Claude

1. Inspect the existing repository before changing code.
2. Reuse existing code where possible.
3. Do not rewrite the whole project unless necessary.
4. Do not introduce a new framework when an equivalent existing dependency is present.
5. Keep secrets in environment variables.
6. Never commit API keys.
7. Add `.env.example`.
8. Add unit tests for risk logic and decision logic.
9. Add an end-to-end demo path.
10. Keep the first implementation runnable with simulated data.
11. Document setup and run commands.
12. Prefer small, reversible changes.
13. After each major change, run tests/lint/type checks available in the repository.
14. Do not claim a feature works unless it is actually implemented and tested.

## 14. Definition of Done for the MVP

The prototype is considered functional when:

- a user can send an AI request;
- an LLM response is generated or simulated;
- the response passes through ControlPlane;
- at least one grounding check works;
- PII detection works;
- a safety/policy check works;
- risk signals are aggregated;
- the decision engine can ALLOW, EDIT/REDACT, or BLOCK;
- a decision is persisted as an audit record;
- a dashboard/API can show the decision and reason;
- test scenarios demonstrate safe and unsafe cases;
- the project can be run from documented steps.

## 15. Demo Scenarios

At minimum create these deterministic scenarios:

1. Safe/grounded response -> ALLOW
2. Unsupported or contradicted fact -> high grounding risk -> BLOCK or FLAG
3. Response containing detectable PII -> REDACT
4. Unsafe/policy-violating response -> BLOCK
5. High-token/slow response -> elevated cost/performance signal
6. Ambiguous/insufficient evidence -> FLAG / HUMAN REVIEW

## 16. Do Not Overbuild

Round 2 asks for a working proof-of-concept, not production-grade enterprise infrastructure. Focus on a credible core mechanism, clean engineering, traceable decisions, and a strong demo.

## 17. Priority Order

P0:
- FastAPI/API flow
- LLM adapter
- Risk orchestration
- Grounding
- PII
- Safety/policy
- Risk score
- Decision engine
- Audit log
- Demo

P1:
- Dashboard
- Async deep checks
- Redis caching
- Better telemetry
- Policy configuration UI

P2:
- Bias probing
- drift detection
- feedback learning loop
- multi-model routing
- advanced enterprise connectors
