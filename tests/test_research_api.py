"""Serving boundaries, legacy routes and research-only failure handling."""

import json

from fastapi.testclient import TestClient

from dail_llm.api.app import create_app
from dail_llm.api.runtime import InferenceGate


def test_research_page_coexists_with_published_plots(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("dail_llm.api.app")
    (tmp_path / "index.html").write_text("<h1>Dáil LLM</h1>", encoding="utf-8")
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "loss.png").write_bytes(b"old plot")
    public = tmp_path / "research-data/pilot"
    public.mkdir(parents=True)
    (public / "summary.json").write_text(json.dumps({"release_id": "pilot-hash", "seed": 42}))
    monkeypatch.setattr(module, "PLOTS_DIR", plots)
    monkeypatch.setenv("DAIL_FRONTEND_DIST", str(tmp_path))
    monkeypatch.delenv("DAIL_RESEARCH_RUN", raising=False)
    with TestClient(create_app(False)) as client:
        for path in ("/research", "/research/", "/research?view=history", "/lab"):
            assert "<h1>" in client.get(path).text
        assert client.get("/research/loss.png").content == b"old plot"
        assert client.get("/research-data/pilot/summary.json").json()["release_id"] == "pilot-hash"
        assert client.get("/api/v1/research/capabilities").json()["live"] is False
        response = client.post(
            "/api/v1/research/inspect",
            json={"release_id": "pilot-hash", "prefix": "The Minister for", "policy": "uniform"},
        )
        assert response.status_code == 503
        assert "Recorded examples" in response.json()["detail"]


def test_live_requests_validate_and_share_gate(monkeypatch):
    monkeypatch.delenv("DAIL_RESEARCH_RUN", raising=False)

    class Reader:
        release_id = "pilot-hash"

        def inspect(self, prefix, policy, excluded):
            if excluded == "missing":
                raise ValueError("Unknown speech")
            return {"prefix": prefix, "policy": policy, "excluded_speech": excluded}

    app = create_app(False)
    with TestClient(app) as client:
        app.state.research_reader = Reader()
        payload = {"release_id": "pilot-hash", "prefix": "é🐈", "policy": "uniform"}
        assert client.post("/api/v1/research/inspect", json=payload).json()["prefix"] == "é🐈"
        assert (
            client.post(
                "/api/v1/research/inspect", json={**payload, "prefix": "🐈" * 257}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/research/inspect", json={**payload, "policy": "../private"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/research/inspect", json={**payload, "release_id": "old"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v1/research/inspect", json={**payload, "excluded_speech": "missing"}
            ).status_code
            == 422
        )
        app.state.gate = InferenceGate(0, 0)
        response = client.post("/api/v1/research/inspect", json=payload)
        assert response.status_code == 429
        assert "retry-after" in response.headers
