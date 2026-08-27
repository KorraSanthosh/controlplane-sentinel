"""API integration tests.

Tests the FastAPI endpoints:
- POST /api/v1/chat (governed generation + async audit task)
- GET  /api/v1/audits (decision trail list)
- GET  /api/v1/audits/{request_id} (full traceable decision record)
- POST /api/v1/audits/{request_id}/feedback & GET feedback
- GET  /api/v1/health (system status & backend health)
- GET  /api/v1/policies (list policy profiles)
- GET  /api/v1/metrics (telemetry summary)
- GET  /api/v1/demo/scenarios (demo scenario list)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.decision import Decision


@pytest.fixture
def client(settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    deps = {d["name"]: d for d in body["dependencies"]}
    assert "llm" in deps
    assert "graph" in deps
    assert "audit" in deps


def test_policies_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/policies")
    assert resp.status_code == 200
    data = resp.json()
    assert "profiles" in data
    profiles = data["profiles"]
    ids = [p["id"] for p in profiles]
    assert "default" in ids
    assert "strict" in ids
    assert "lenient" in ids


def test_demo_scenarios_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/demo/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert "scenarios" in data
    scenarios = data["scenarios"]
    assert len(scenarios) >= 6


def test_chat_and_audit_flow(client: TestClient) -> None:
    # 1. POST /chat with a prompt that triggers PII redaction
    payload = {
        "message": "Can you confirm details for my account? My email is john.doe@example.com",
        "use_case": "support_assistant",
        "policy_profile": "default",
    }
    chat_resp = client.post("/api/v1/chat", json=payload)
    assert chat_resp.status_code == 200
    data = chat_resp.json()

    req_id = data["request_id"]
    assert data["decision"] in (Decision.REDACT, Decision.FLAG, Decision.ALLOW)
    assert "risk" in data
    assert "grounding" in data["risk"]["signals"]
    assert "pii" in data["risk"]["signals"]
    assert "safety" in data["risk"]["signals"]
    assert "bias" in data["risk"]["signals"]
    assert "cost" in data["risk"]["signals"]
    assert "telemetry" in data

    # 2. GET /audits to list decisions
    audit_list_resp = client.get("/api/v1/audits")
    assert audit_list_resp.status_code == 200
    list_body = audit_list_resp.json()
    assert list_body["total"] >= 1
    req_ids = [item["request_id"] for item in list_body["items"]]
    assert req_id in req_ids

    # 3. GET /audits/{request_id} to fetch traceable record
    audit_detail_resp = client.get(f"/api/v1/audits/{req_id}")
    assert audit_detail_resp.status_code == 200
    detail = audit_detail_resp.json()
    assert detail["request_id"] == req_id
    assert detail["decision"] == data["decision"]
    assert "prompt_preview" in detail
    assert "response_sha256" in detail
    assert "risk" in detail

    # 4. POST /audits/{request_id}/feedback
    fb_payload = {
        "reviewer": "audit_reviewer_1",
        "agrees_with_decision": True,
        "comment": "Verified PII redaction action.",
    }
    fb_post_resp = client.post(f"/api/v1/audits/{req_id}/feedback", json=fb_payload)
    assert fb_post_resp.status_code == 201
    fb_data = fb_post_resp.json()
    assert fb_data["request_id"] == req_id
    assert fb_data["reviewer"] == "audit_reviewer_1"

    # 5. GET /audits/{request_id}/feedback
    fb_get_resp = client.get(f"/api/v1/audits/{req_id}/feedback")
    assert fb_get_resp.status_code == 200
    fbs = fb_get_resp.json()
    assert len(fbs) == 1
    assert fbs[0]["comment"] == "Verified PII redaction action."


def test_metrics_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_requests" in body
