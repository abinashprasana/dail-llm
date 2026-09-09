"""Local adversarial checks; these do not probe a deployed service."""

import pickle

import pytest
import torch
from fastapi.testclient import TestClient

from dail_llm.api.app import create_app
from dail_llm.api.runtime import RateLimiter
from dail_llm.research.common import ROOT, Budget, BudgetExceeded, initialize, read_json, write_json
from dail_llm.research.training import load


class StubService:
    def generate(self, prompt, max_new_tokens, temperature):
        return {
            "text": prompt,
            "prompt": prompt,
            "generated_characters": 0,
            "elapsed_ms": 0,
            "filtered_characters": [],
        }


def test_forwarded_header_cannot_bypass_rate_limit():
    with TestClient(create_app(load_model_on_start=False)) as client:
        client.app.state.model_service = StubService()
        client.app.state.rate_limiter = RateLimiter(1, 60)
        payload = {"prompt": "The Minister for", "max_new_tokens": 50}
        assert (
            client.post(
                "/api/v1/generate", json=payload, headers={"x-forwarded-for": "198.51.100.1"}
            ).status_code
            == 200
        )
        response = client.post(
            "/api/v1/generate", json=payload, headers={"x-forwarded-for": "198.51.100.2"}
        )
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "x" * 257},
        {"prompt": []},
        {"prompt": "x", "max_new_tokens": 10**9},
        {"prompt": "x", "temperature": "NaN"},
        {"prompt": "x", "temperature": "Infinity"},
    ],
)
def test_hostile_generation_requests_rejected(payload):
    with TestClient(create_app(load_model_on_start=False)) as client:
        assert client.post("/api/v1/generate", json=payload).status_code == 422


def test_cors_and_static_traversal(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    secret = "private sentinel outside public directory"
    (tmp_path / "secret.txt").write_text(secret)
    monkeypatch.setenv("DAIL_FRONTEND_DIST", str(public))
    with TestClient(create_app(load_model_on_start=False)) as client:
        for path in ("/%2e%2e/secret.txt", "/..%5csecret.txt", "/%2e%2e%2fsecret.txt"):
            assert secret not in client.get(path).text
        response = client.options(
            "/api/v1/generate",
            headers={
                "origin": "https://untrusted.example",
                "access-control-request-method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    "location", ["frontend/../data/attack", "outputs/checkpoints/attack", "dail_llm/attack"]
)
def test_research_cannot_target_protected_paths(location):
    with pytest.raises(ValueError, match="isolated"):
        initialize(ROOT / location, ROOT / "experiments/pilot.toml")


def test_budget_lock_and_interruption(tmp_path, monkeypatch):
    write_json(tmp_path / "config.json", {"memory_gib": 4, "max_seconds": 0})
    with pytest.raises(BudgetExceeded, match="memory"):
        with Budget(tmp_path, "adversarial") as budget:
            with pytest.raises(RuntimeError, match="lock"):
                with Budget(tmp_path, "collision"):
                    pass
            monkeypatch.setattr("dail_llm.research.common.peak_bytes", lambda: 5 * 1024**3)
            budget.check()
    assert not (tmp_path / ".execution.lock").exists()
    assert read_json(tmp_path / "status.json")["stages"]["adversarial"]["status"] == "incomplete"


class UnsupportedCheckpointObject:
    pass


def test_checkpoint_rejects_untrusted_pickle_objects(tmp_path):
    checkpoint = tmp_path / "untrusted.pt"
    torch.save(UnsupportedCheckpointObject(), checkpoint)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load(tmp_path, checkpoint)
