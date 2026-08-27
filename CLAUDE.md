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

## 18. Version Control Discipline (Mandatory)

Repository: `origin` -> `https://github.com/KorraSanthosh/controlplane-sentinel.git`
Working branch: `main`, tracking `origin/main`.

**Rule: after every successful code change or safe point, commit and push to `main`.**
Permission for `git add` / `git commit` / `git push origin main` in this repository is granted in
advance. Do not ask before each one. Do not batch a day of work into a single commit.

### What counts as a safe point

All four must hold:

1. the change is complete — no half-wired module, no import that does not resolve;
2. the backend test suite passes **and the output was actually observed** (see §19);
3. no secret, `.env`, virtualenv, or cache file is staged;
4. every staged file can be explained by the commit message.

If the tree is knowingly incomplete and a checkpoint is still needed, commit it as
`wip(<area>): ...` and state in the body exactly what is not yet wired. Prefer finishing the
slice over checkpointing it.

### Commit message format

```
<area>: <imperative summary, <= 72 chars>

<why this change exists — the reasoning the diff cannot show>
<what is deliberately left undone, if anything>
<which existing tests changed and why, if any>
```

Areas: `signals`, `scoring`, `policy`, `orchestrator`, `grounding`, `pii`, `safety`, `bias`,
`repair`, `audit`, `api`, `demo`, `tests`, `docs`, `dashboard`, `chore`.

Good: `bias: add BiasSignal and wire fairness into RiskSignals`
Avoid: messages that describe a whole work session rather than one change ("update code",
"prototype done"). They cannot be reviewed and cannot be reverted cleanly.

### Hard limits

- One logical change per commit. Small and reversible beats large and tidy (§13.12).
- Run `git status --short` before staging. Stage deliberately; do not `git add -A` blindly.
- Never commit API keys, `.env`, `backend/venv/`, `.pytest_cache/`, `.DS_Store`. `.gitignore`
  already covers these — that is a safety net, not a substitute for looking.
- Never force-push, never rewrite pushed history, never `git reset --hard` a pushed commit.
- If a push is rejected, fetch and rebase or merge. Do not `--force` your way past it.

### Why this matters here

Chat history is not stored in this repository and cannot be relied on across sessions. If every
safe point is committed and pushed, `git log` becomes the project's actual memory. Session
transcripts are not a backup; commits are.

## 19. Verification Gate

`pytest` is the **only** configured quality gate. `backend/pyproject.toml` defines pytest
settings and nothing else — there is no ruff and no mypy configuration in this repository. Do not
report that a linter or type checker ran.

```
cd backend && ./venv/bin/python -m pytest -q
```

**Known environment constraint.** The Bash safety classifier in this environment intermittently
refuses to execute `pytest` and `python -c` while still permitting short `git` / `grep` / `ls`
commands. When that happens:

- do not guess the result;
- do not mark the step done;
- ask the user to run it in-session so the output lands in the conversation:
  `! cd backend && ./venv/bin/python -m pytest -q`

§13.14 with teeth: **an unrun test suite is not a passing test suite.** If the suite was not
observed to pass, say so in those words. Such a change is not a safe point under §18 — either
get the suite run, or label the commit body `unverified: test suite not executed`.

**Open environment issue (unverified):** `backend/requirements.txt` declares Python 3.12+, but
`backend/venv` is Python 3.9. Pydantic models across the codebase annotate fields as `X | None`,
which Pydantic resolves at runtime and which 3.9 cannot evaluate. If the suite fails on import,
rebuild the virtualenv on Python 3.12 — do not rewrite annotations across the codebase to
accommodate an old interpreter.

## 20. Environment and Run Commands

- Install: `cd backend && ./venv/bin/python -m pip install -r requirements.txt`
- Run the API: `cd backend && ./venv/bin/python -m uvicorn app.main:app --reload`
- Dashboard: served by the same app at `/` and `/dashboard` from `backend/app/static/index.html`.
  Extend that file; do not start a parallel frontend without reason (§7, reuse over rewrite).
- Configuration is environment variables only; the template is `.env.example`. Never inline a
  key, not even in a test or a demo fixture.
- Policy YAMLs load eagerly at startup, so a malformed profile aborts boot by design. If the
  server stops starting after a policy edit, read the loader's error before changing code.

## 21. Adding a Risk Check Is a Cross-Cutting Change

`policy/loader.py` rejects unknown check names at load time, so a new detector cannot be dropped
in one file at a time. These move together or startup and tests fail:

1. `schemas/signals.py` — signal model, `RiskSignals` field, `as_dict()`, `__all__`
2. `services/<check>/service.py` — rule layer inline, optional LLM probe on the deep path;
   model it on `services/safety/service.py`
3. `services/risk/scoring.py` — `DEFAULT_WEIGHTS`
4. `services/policy/loader.py` — `CHECK_NAMES`; also `NON_PROFILE_FILES` if the check adds a
   non-profile YAML, otherwise it is parsed as an extra policy profile and boot fails
5. `services/policy/conditions.py` — `FIELDS` entries, so rules can reference the new signal
6. all three profiles in `policies/` — `weights` (must sum to 1.0), `enabled_checks`,
   `on_unavailable`, and rules
7. `container.py` construction and `orchestrator.py` fast path, placeholder, and action
8. `tests/factories.py` factory plus a dedicated `tests/test_<check>.py`

Two rules that follow from this:

- Adding a check changes the weighted denominator, so existing score assertions legitimately
  move. Update them deliberately and state in the commit body that the change is arithmetic,
  not a regression.
- A detector that did not run reports `SKIPPED` or `UNAVAILABLE`, never `PASS`. A check that
  never ran must not read as a check that found nothing.

## 22. Never Reference a File That Does Not Exist

Already present in this repository and to be fixed, not repeated: `scripts/run_demo.py` and
`backend/tests/test_scenarios.py` are referenced as real from `data/demo/scenarios.yaml`,
`backend/app/api/v1/demo.py`, `backend/app/demo/scenarios.py`, `backend/app/container.py`,
`backend/tests/test_orchestrator.py`, and `backend/requirements.txt`. A top-level `README.md` is
referenced from `backend/app/main.py`. None of these files exist.

Rule: create the file in the same change that references it, or do not reference it. Confirm a
path exists before writing it into a docstring, comment, or YAML. A phantom reference is worse
than no reference — it asserts coverage that no test provides, which is exactly the failure
§13.14 exists to prevent.

## 23. Recovering Context in a New Session

There is no `.claude/logs/` directory and no chat history in the repository. Read, in order:

1. `CLAUDE.md`, `PROJECT_CONTEXT.md`, `SYSTEM_REQUIREMENTS.md`
2. `git log --oneline` — what has actually shipped
3. `git status --short` and `git diff` — what was in flight when the last session ended

Uncommitted work in `git diff` is the most reliable record of unfinished work, which is another
reason to keep §18 tight: the smaller the diff at any moment, the less context a new session has
to reconstruct.
