"""Exercise the production HTTP surface of a running container or local server."""

import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    def get(path):
        with urlopen(args.url + path, timeout=10) as response:
            return response.read()

    for _ in range(30):
        try:
            if json.loads(get("/api/v1/health"))["model_loaded"]:
                break
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError("Model did not become ready")
    for path in ("/", "/lab", "/research", "/research/", "/research?view=history"):
        assert b'<div id="root"' in get(path), path
    assert get("/research/loss.png").startswith(b"\x89PNG")
    summary = json.loads(get("/research-data/pilot/summary.json"))
    capabilities = json.loads(get("/api/v1/research/capabilities"))
    assert capabilities["release_id"] == summary["release_id"]
    payload = {"prompt": "The Minister for", "max_new_tokens": 50, "temperature": 0.8}
    request = Request(
        args.url + "/api/v1/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        generated = json.load(response)
    assert generated["prompt"] == payload["prompt"] and generated["generated_characters"] == 50
    if capabilities["live"]:
        request = Request(
            args.url + "/api/v1/research/inspect",
            data=json.dumps(
                {
                    "release_id": summary["release_id"],
                    "prefix": "The Minister for",
                    "policy": "uniform",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            inspection = json.load(response)
        assert inspection["release_id"] == summary["release_id"]
        assert abs(sum(d["after"] for d in inspection["distributions"]) - 1) < 1e-6
    print(
        f"PASS: production routes, legacy plots, generation; live research={capabilities['live']}"
    )


if __name__ == "__main__":
    main()
