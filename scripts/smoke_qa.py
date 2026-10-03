"""Run a local end-to-end Q&A API smoke against the indexed Official Report."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from dail_llm.api.app import create_app


def main() -> None:
    with TestClient(create_app(load_model_on_start=False)) as client:
        capabilities = client.get("/api/v1/qa/capabilities").json()
        if not capabilities["available"]:
            raise RuntimeError("Build the local index before running the Q&A smoke")
        response = client.post("/api/v1/qa/ask", json={
            "question": "What was said about housing supply on 1 October 2026?",
            "start_date": "2026-10-01", "end_date": "2026-10-01",
        })
        response.raise_for_status()
        result = response.json()
        sources = {source["passage_id"]: source for source in result["sources"]}
        if not set(result["citation_ids"]) <= set(sources):
            raise AssertionError("The API returned a citation absent from its sources")
        if not all(source["source_url"].startswith(
            "https://www.oireachtas.ie/en/debates/debate/dail/"
        ) for source in sources.values()):
            raise AssertionError("The API returned a non-official citation URL")
        print(json.dumps({"status": result["status"], "source_count": len(sources),
                          "coverage": result["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
