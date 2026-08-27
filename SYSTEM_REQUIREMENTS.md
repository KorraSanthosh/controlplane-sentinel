# ControlPlane.ai — System Requirements Specification

## 1. Purpose

Build a prototype Responsible AI Checker for enterprise generative-AI use cases.

The system sits between an AI application and its end user, evaluates model outputs, assigns risk, and takes a configurable action.

The official Round 2 brief says the prototype may use realistic, simulated, or sample data and does not require production-grade architecture.

## 2. Problem Scope

The system must address three broad risk dimensions:

### 2.1 Performance Risk

Detect or score:
- unsupported factual claims
- hallucination/grounding problems
- confidence/consistency signals where feasible

The prototype's strongest implementation is source-grounded verification using Graph RAG / a knowledge graph.

### 2.2 Cost Risk

Capture:
- input token usage if provider exposes it
- output token usage if provider exposes it
- total token usage
- response latency
- estimated cost
- repeated user retries/rework when available

### 2.3 Responsibility Risk

Detect:
- PII/sensitive data leakage
- unsafe content
- configurable policy violations
- optional bias signals

## 3. Functional Requirements

### FR-01: Request intake

The API shall accept:
- user prompt/message
- optional use-case identifier
- optional user/tenant identifier
- optional policy profile

### FR-02: Model invocation

The system shall invoke an LLM through a provider-agnostic adapter.

The adapter shall return:
- generated text
- provider/model identifier
- token usage when available
- latency
- raw provider metadata needed for telemetry

### FR-03: Risk orchestration

The system shall send the model output to enabled risk checks and aggregate their results.

Checks should be independently switchable by policy.

### FR-04: Grounding verification

The grounding module shall:
1. extract factual claims or relevant entities;
2. query the trusted knowledge source;
3. compare generated claims with retrieved evidence;
4. produce a structured grounding result.

Minimum output:
- status: grounded / unsupported / contradicted / unavailable
- confidence score
- evidence references
- explanation

### FR-05: PII detection

The system shall identify at least common detectable PII such as:
- email
- phone number
- address
- other configured sensitive entities

The system shall support redaction.

### FR-06: Safety/policy check

The system shall evaluate responses against configurable rules and/or a secondary safety evaluator.

Minimum result:
- pass/fail/uncertain
- severity
- policy identifiers
- reason

### FR-07: Cost and latency measurement

Every request shall record:
- start/end time
- latency
- token usage if available
- estimated cost if a pricing configuration is available

### FR-08: Risk scoring

The system shall convert individual signals into a single overall risk score.

The scoring formula and weights must be configurable.

Do not hard-code a claim that any specific weights are universal or industry standard.

### FR-09: Decision engine

The system shall support at least:
- ALLOW
- EDIT/REDACT
- BLOCK

The architecture should also support:
- FLAG / HUMAN_REVIEW

### FR-10: Audit trail

Every decision shall create an audit record containing enough information to explain:
- what was checked
- which checks triggered
- risk scores
- policy used
- final action
- latency
- model/provider
- token usage/cost if available

### FR-11: Error handling

If a dependency fails:
- do not silently mark a check as passed;
- return an explicit unavailable/unknown state;
- apply the configured fail-safe policy.

### FR-12: Configuration

Use environment variables/configuration files for:
- LLM provider
- model
- API keys
- database URL
- Neo4j URL/credentials
- Redis URL
- policy thresholds
- feature flags

## 4. Non-Functional Requirements

### NFR-01: Security
- never commit credentials
- sanitize logs
- minimize storage of sensitive content
- hash or reference raw responses where appropriate

### NFR-02: Performance
- cheap checks should be possible inline
- independent checks may run in parallel
- expensive checks should be selectively triggered
- audit/telemetry should support async processing where practical

### NFR-03: Explainability
Every final decision must have a human-readable reason and machine-readable evidence.

### NFR-04: Reliability
A single detector failure must not crash the entire service.

### NFR-05: Testability
Risk modules and decision logic must be unit-testable without live LLM calls.

### NFR-06: Reproducibility
Provide deterministic demo fixtures for the main scenarios.

## 5. API Requirements

Suggested endpoints:

### POST /api/v1/chat
Generate a response and pass it through ControlPlane.

Request example:
```json
{
  "message": "What is the capital of Australia?",
  "use_case": "knowledge_assistant",
  "policy_profile": "default"
}
```

Response example:
```json
{
  "request_id": "req_123",
  "answer": "The capital of Australia is Sydney.",
  "risk": {
    "overall_score": 0.91,
    "grounding": {
      "status": "contradicted",
      "score": 0.95
    },
    "pii": {
      "detected": false
    },
    "safety": {
      "status": "pass"
    }
  },
  "decision": "BLOCK",
  "reason": "Generated claim contradicts trusted grounding evidence."
}
```

### GET /api/v1/audits
List/filter audit events.

### GET /api/v1/audits/{request_id}
Get one complete audit record.

### GET /api/v1/metrics
Return summary metrics for the dashboard.

### GET /api/v1/health
Health check.

## 6. Data Model

Minimum relational entities:

### requests
- id
- timestamp
- use_case
- policy_profile
- provider
- model
- latency_ms
- input_tokens
- output_tokens
- estimated_cost
- final_decision

### risk_events
- id
- request_id
- type
- severity
- score
- status
- explanation
- evidence_reference
- created_at

### policies
- id
- name
- use_case
- thresholds/config
- enabled_checks
- action_rules
- version
- created_at

### feedback
- id
- request_id
- reviewer
- override_decision
- comment
- created_at

## 7. Graph Schema (Prototype)

Use a small controlled graph for demo grounding.

Example nodes:
- Entity
- Document
- Fact
- Policy
- SensitiveCategory

Example relationships:
- Entity -[HAS_FACT]-> Fact
- Fact -[SUPPORTED_BY]-> Document
- Entity -[CLASSIFIED_AS]-> SensitiveCategory
- Policy -[FORBIDS]-> SensitiveCategory

Keep the graph small and deterministic for the demo.

## 8. Risk Scoring Model

Illustrative structure:

overall_risk =
  weighted(performance risk)
  + weighted(cost risk)
  + weighted(responsibility risk)

Possible signal set:
- grounding risk
- safety risk
- privacy risk
- bias risk
- cost anomaly
- latency anomaly
- model confidence/consistency

The system must store both:
- component scores
- final score

## 9. Policy Decision Matrix

Use configurable rules.

Example:

| Condition | Action |
|---|---|
| no significant risk | ALLOW |
| moderate PII | EDIT/REDACT |
| uncertain/ambiguous evidence | FLAG / HUMAN_REVIEW |
| clear sensitive-data exposure | BLOCK |
| clear unsafe policy violation | BLOCK |
| strong factual contradiction | BLOCK or FLAG, depending on policy |

## 10. Demo Acceptance Criteria

A complete demo must prove:

### Scenario A — Safe
Input -> correct/grounded answer -> ALLOW.

### Scenario B — Hallucination
Input -> intentionally incorrect/unsupported answer -> grounding failure -> BLOCK/FLAG.

### Scenario C — PII
Input -> response contains PII -> REDACT -> modified response returned.

### Scenario D — Safety
Input -> unsafe policy violation -> BLOCK.

### Scenario E — Cost/latency
Show telemetry where a request has unusually high token usage or latency.

### Scenario F — Uncertainty
Evidence cannot support a definitive conclusion -> system does not invent certainty -> FLAG/ABSTAIN.

## 11. Repository Requirements

Recommended structure:

```text
controlplane-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── llm/
│   │   │   ├── grounding/
│   │   │   ├── pii/
│   │   │   ├── safety/
│   │   │   ├── risk/
│   │   │   ├── policy/
│   │   │   └── audit/
│   │   └── main.py
│   └── tests/
├── frontend/
├── graph/
│   ├── seed/
│   └── queries/
├── data/
│   └── demo/
├── docs/
├── scripts/
├── .env.example
├── docker-compose.yml
└── README.md
```

Adapt this to the existing repository instead of blindly replacing its structure.

## 12. Out of Scope for the First MVP

- training a foundation model
- production-scale multi-region deployment
- full enterprise identity platform
- every possible PII type
- perfect automated causal inference
- complete regulatory compliance engine
- full autonomous agent governance
- advanced online learning
- enterprise connectors to every data source

## 13. Implementation Sequence

P0:
1. Inspect existing codebase.
2. Make the current app run.
3. Implement LLM adapter.
4. Implement ControlPlane orchestration.
5. Implement grounding checker.
6. Implement PII detector/redactor.
7. Implement safety/policy check.
8. Implement risk score.
9. Implement decision engine.
10. Persist audit data.
11. Create deterministic demo scenarios.
12. Add tests.

P1:
13. Dashboard.
14. Async deep analysis.
15. Redis caching.
16. Advanced telemetry.
17. Configurable policy UI.

P2:
18. Bias probing.
19. Drift detection.
20. Feedback learning loop.
21. Multi-model routing.
