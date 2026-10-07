"""Incrementally cache official Dáil XML from 2014 onward.

Usage: python -m scripts.download_debates --start 2026-06-01 --end 2026-06-30
The default range is 2014-01-01 through today. Raw XML remains under ignored data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from dail_llm.qa.corpus import parse_debate

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.oireachtas.ie/v1/debates"
USER_AGENT = "DailLLMResearch/1.0 (cached public retrieval)"


def fetch(url: str, attempts: int = 3) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def months(start: date, end: date):
    cursor = start.replace(day=1)
    while cursor <= end:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(start, cursor), min(end, next_month - timedelta(days=1))
        cursor = next_month


def records(start: date, end: date):
    for first, last in months(start, end):
        skip = 0
        while True:
            query = urllib.parse.urlencode({
                "date_start": first.isoformat(), "date_end": last.isoformat(),
                "limit": 100, "skip": skip,
            })
            payload = json.loads(fetch(f"{BASE}?{query}"))
            batch = payload.get("results", [])
            for item in batch:
                record = item["debateRecord"]
                if record.get("chamber", {}).get("uri", "").endswith("/house/dail"):
                    yield record
            skip += len(batch)
            if not batch or skip >= payload["head"]["counts"]["resultCount"]:
                break
            time.sleep(0.25)
        time.sleep(0.25)


def download(start: date, end: date, root: Path = ROOT / "data" / "oireachtas",
             refresh: bool = False) -> dict:
    if start < date(2014, 1, 1) or end < start or end > date.today():
        raise ValueError("Expected a date range from 2014-01-01 through today")
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    entries = manifest.get("records", {})
    seen: set[str] = set()
    for record in records(start, end):
        day = record["date"]
        seen.add(day)
        xml_url = record.get("formats", {}).get("xml") or {}
        xml_url = xml_url.get("uri")
        if not xml_url or not xml_url.startswith("https://data.oireachtas.ie/"):
            raise ValueError(f"Missing official XML URL for {day}")
        updated = record.get("lastUpdated")
        target = raw_dir / f"{day}.xml"
        old = entries.get(day, {})
        if not refresh and target.exists() and old.get("last_updated") == updated:
            continue
        xml = fetch(xml_url)
        checksum = hashlib.sha256(xml).hexdigest()
        if target.exists() and old.get("sha256") == checksum:
            old["last_updated"] = updated
            old["checked_at"] = datetime.now(UTC).isoformat()
            entries[day] = old
            continue
        passages = parse_debate(xml, day)
        # Some official sitting records contain only summaries or documents.
        # Keep their checksum and zero count so a later revision is detectable.
        temporary = target.with_suffix(".xml.tmp")
        temporary.write_bytes(xml)
        temporary.replace(target)
        entries[day] = {
            "xml_url": xml_url, "last_updated": updated,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": checksum,
            "bytes": len(xml), "passages": len(passages),
        }
        manifest_path.write_text(json.dumps({
            "source": BASE, "licence": "https://www.oireachtas.ie/en/open-data/license/",
            "records": dict(sorted(entries.items())),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{day}: {len(passages)} speeches")
        time.sleep(0.25)
    manifest_path.write_text(json.dumps({
        "source": BASE, "licence": "https://www.oireachtas.ie/en/open-data/license/",
        "records": dict(sorted(entries.items())),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"listed_days": len(seen), "cached_days": len(entries),
            "first_cached": min(entries) if entries else None,
            "last_cached": max(entries) if entries else None,
            "speeches": sum(row["passages"] for row in entries.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=date(2014, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--refresh", action="store_true",
                        help="Refetch XML and compare SHA-256 even if metadata is unchanged")
    args = parser.parse_args()
    print(json.dumps(download(args.start, args.end, refresh=args.refresh), indent=2))


if __name__ == "__main__":
    main()
