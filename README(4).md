# ControlPlane Sentinel

## AI Governance & Risk Control Plane

ControlPlane Sentinel is an AI governance control plane that evaluates model-generated responses before delivery. It combines five governance checks—Grounding, Safety, Privacy/PII, Bias/Fairness, and Cost/Overhead—into an adaptive risk assessment and policy decision.

### Decision Actions

- **ALLOW** — deliver the response
- **REDACT** — mask detected sensitive information
- **FLAG** — escalate for human review
- **BLOCK** — withhold the response and use the configured fallback

## Architecture

```text
User Request
     |
     v
LLM Provider
     |
     v
Orchestrator
     |
     +-- Grounding / Graph RAG
     +-- Safety
     +-- PII / Privacy
     +-- Bias / Fairness
     +-- Cost / Latency
     |
     v
Risk Scorer
     |
     v
Decision Engine
     |
     +-- ALLOW
     +-- REDACT
     +-- FLAG
     +-- BLOCK
     |
     v
Audit Trail
     |
     v
Dashboard
```

## Five Governance Checks

### 1. Grounding / Performance
Uses claim extraction and knowledge-graph verification.

- `CONTRADICTED` → **BLOCK**
- `UNSUPPORTED` → **FLAG** + human review
- Default risk weight: **0.40**

### 2. Safety
Uses deterministic rule checks on the fast path and an LLM judge on the deep path when required.

- `CRITICAL` → **BLOCK**
- `HIGH` → **FLAG** + human review
- Default risk weight: **0.30**

### 3. Privacy / PII
Detects email, phone, account number, credit card, DOB, and postal address.

Redactable PII is replaced with placeholders such as `[REDACTED: email]`.

- Redactable PII → **REDACT**
- Non-redactable PII → **BLOCK**
- Default risk weight: **0.20**

### 4. Bias / Fairness
Deterministic rules cover categories such as gender, age, ethnicity, postcode proxy/redlining, and disability.

- High/critical bias risk → **FLAG** + human review
- Default risk weight: **0.00**, configurable through policy profiles

### 5. Cost / Overhead
Evaluates token usage, latency, estimated USD cost, and configured budgets.

A cost warning is recorded without blocking an otherwise correct response.

- Default risk weight: **0.10**

## Adaptive Risk Engine

Default weights:

| Signal | Weight |
|---|---:|
| Grounding | 0.40 |
| Safety | 0.30 |
| PII | 0.20 |
| Bias | 0.00 |
| Cost | 0.10 |
| **Total** | **1.00** |

For usable detectors:

```text
Overall Risk =
SUM(weight × score) / SUM(usable weights)
```

Unavailable or skipped detectors are excluded from the denominator. The final score is clamped to `[0.0, 1.0]` and rounded to four decimal places.

Severity mapping:

| Severity | Score |
|---|---:|
| PASS | 0.0 |
| LOW | 0.2 |
| MEDIUM | 0.5 |
| HIGH | 0.8 |
| CRITICAL | 1.0 |

## Fast Path and Deep Path

The orchestrator first performs inexpensive checks. More expensive grounding verification and LLM safety judging are selectively triggered when triage conditions require them.

## Demo Scenarios

| Scenario | Finding | Decision |
|---|---|---|
| A — Grounded Safe | Grounded response | **ALLOW** |
| B — Hallucination Contradicted | Knowledge-graph contradiction | **BLOCK** |
| C — PII Leakage | Redactable PII | **REDACT** |
| D — Unsafe Policy Violation | Critical safety violation | **BLOCK** |
| E — Cost Anomaly | Token/latency warning | **ALLOW** |
| F — Insufficient Evidence | Unsupported claim | **FLAG** + human review |

## Audit Trail

Audit records include request ID, timestamps, policy information, model/provider metadata, hashes and sanitized previews, risk, decision, fired rules, unavailable/skipped checks, human-review status, and telemetry.

Raw prompts and raw model responses are not persisted. SHA-256 hashes and sanitized/truncated previews are used for traceability.

### Audit APIs

```text
GET  /api/v1/audits
GET  /api/v1/audits/{request_id}
POST /api/v1/audits/{request_id}/feedback
GET  /api/v1/audits/{request_id}/feedback
```

Production persistence uses MongoDB; an in-memory repository is available for fallback/offline development and testing.

## Dashboard

The dashboard displays:

- Five governance checks
- Individual risk scores and statuses
- Overall risk and decision
- Human-review indicator
- Grounding claims/evidence
- PII findings
- Safety violations
- Bias findings
- Deep-path information
- Telemetry
- Audit log explorer

## Repository Structure

```text
controlplane-sentinel/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   │   ├── grounding/
│   │   │   ├── safety/
│   │   │   ├── pii/
│   │   │   ├── bias/
│   │   │   ├── cost/
│   │   │   ├── risk/
│   │   │   ├── policy/
│   │   │   ├── audit/
│   │   │   └── llm/
│   │   └── static/
│   └── tests/
├── data/demo/
├── graph/seed/
├── policies/
├── .env.example
├── .gitignore
├── CLAUDE.md
├── PROJECT_CONTEXT.md
├── SYSTEM_REQUIREMENTS.md
└── README.md
```

## Running and Testing

For the verified test suite:

```bash
cd backend
./venv/bin/python -m pytest -v
```

Verified result:

```text
170 passed in 0.88s
```

The test suite covers REST APIs, bias, grounding, orchestrator behavior, PII detection/redaction, policy profiles, risk scoring, and scenarios A–F.

## Security

`.gitignore` protects local secrets and credentials, including `.env`, secrets/credentials, key/certificate files, virtual environments, and `node_modules`. `.env.example` is the committed configuration template.

## Current Limitation

Automated generative claim repair is **not implemented**.

Current safe behavior:

- Contradicted claims → **BLOCK**
- Unsupported claims → **FLAG** + human review
- Privacy-sensitive spans → **REDACT** when safely redactable

`REDACT` therefore means privacy span masking, not generative factual-claim rewriting.

## Future Enhancements

Potential production enhancements include live Redis caching, an active Neo4j deployment, a learning loop from reviewer feedback, automated generative claim repair, additional governance detectors, and expanded observability.

## Project Status

**MVP COMPLETE**

The verified repository contains all five governance checks, adaptive risk scoring, policy decisions, scenarios A–F, audit and feedback APIs, dashboard integration, and **170 passing automated tests**.
