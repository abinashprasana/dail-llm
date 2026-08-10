from fastapi.testclient import TestClient

from dail_llm.api.app import create_app


class FakeService:
    def generate(self, prompt, max_new_tokens, temperature):
        return {
            "text": prompt + " generated",
            "prompt": prompt,
            "generated_characters": max_new_tokens,
            "elapsed_ms": 12,
            "filtered_characters": [],
        }

    def metadata(self):
        return {"name": "Dáil LLM", "architecture": {}, "dataset": None}

    def evaluation(self):
        return {"metrics": {}, "samples": []}


def test_health_is_degraded_without_model():
    app = create_app(load_model_on_start=False)
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False
        assert response.headers["x-content-type-options"] == "nosniff"


def test_generate_contract_and_bounds():
    app = create_app(load_model_on_start=False)
    with TestClient(app) as client:
        app.state.model_service = FakeService()
        response = client.post(
            "/api/v1/generate",
            json={"prompt": "The Minister for", "max_new_tokens": 50, "temperature": 0.8},
        )
        assert response.status_code == 200
        assert response.json()["generated_characters"] == 50

        invalid = client.post(
            "/api/v1/generate",
            json={"prompt": "x", "max_new_tokens": 20, "temperature": 0.8},
        )
        assert invalid.status_code == 422
