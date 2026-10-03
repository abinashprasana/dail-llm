"""Resume-safe, bounded upload of one local year shard to Cloudflare D1.

Requires a reviewed D1 database, CLOUDFLARE_ACCOUNT_ID and
CLOUDFLARE_D1_WRITE_TOKEN. Run at most one year at a time. The default
40,000 passages targets 80,000 explicit rows. D1's actual billable writes,
including FTS and other account use, must be measured before relying on the
100,000-row daily free allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path


def d1_query(database_id: str, sql: str, params: list | None = None) -> list[dict]:
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_D1_WRITE_TOKEN"]
    url = (f"https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/"
           f"{database_id}/query")
    request = urllib.request.Request(url, data=json.dumps({
        "sql": sql, "params": params or [],
    }).encode("utf-8"), headers={"Authorization": f"Bearer {token}",
                               "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not result.get("success") or not result.get("result", [{}])[0].get("success"):
        raise RuntimeError(f"D1 query failed: {result.get('errors')}")
    return result["result"][0].get("results", [])


def initialize(database_id: str) -> None:
    d1_query(database_id, "CREATE TABLE IF NOT EXISTS passages (passage_id TEXT PRIMARY KEY, "
             "date TEXT NOT NULL, speaker TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, "
             "source_url TEXT NOT NULL, language TEXT, party TEXT)")
    d1_query(database_id, "CREATE INDEX IF NOT EXISTS passages_date ON passages(date)")
    d1_query(
        database_id,
        "CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5("
        "passage_id UNINDEXED, text, title, speaker, "
        "tokenize='unicode61 remove_diacritics 0')",
    )


def publish(index: Path, year: int, database_id: str, max_passages: int = 40_000,
            refresh: bool = False) -> dict:
    if not index.exists():
        raise FileNotFoundError(index)
    if max_passages < 1 or max_passages > 40_000:
        raise ValueError("max_passages must be between 1 and 40000")
    state_path = index.parent / f"d1-upload-{year}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state and state.get("database_id") != database_id:
        raise ValueError("Upload state belongs to another D1 database")
    last_id = "" if refresh else state.get("last_id", "")
    local = sqlite3.connect(index)
    local.row_factory = sqlite3.Row
    uploaded = 0
    try:
        initialize(database_id)
        while uploaded < max_passages:
            rows = local.execute("SELECT * FROM passages WHERE date>=? AND date<? "
                                 "AND passage_id>? ORDER BY passage_id LIMIT ?",
                                 (f"{year}-01-01", f"{year+1}-01-01", last_id,
                                  min(8, max_passages - uploaded))).fetchall()
            if not rows:
                break
            ids = [row["passage_id"] for row in rows]
            marks = ",".join("?" for _ in rows)
            d1_query(database_id, f"DELETE FROM passage_fts WHERE passage_id IN ({marks})", ids)
            passage_marks = ",".join("(" + ",".join("?" for _ in range(8)) + ")"
                                     for _ in rows)
            values = [row[key] for row in rows for key in
                      ("passage_id", "date", "speaker", "title", "text", "source_url",
                       "language", "party")]
            d1_query(database_id, "INSERT OR REPLACE INTO passages VALUES " + passage_marks,
                     values)
            fts_marks = ",".join("(?,?,?,?)" for _ in rows)
            fts_values = [row[key] for row in rows for key in
                          ("passage_id", "text", "title", "speaker")]
            d1_query(database_id, "INSERT INTO passage_fts(passage_id,text,title,speaker) "
                     "VALUES " + fts_marks, fts_values)
            last_id = ids[-1]
            uploaded += len(rows)
            state_path.write_text(json.dumps({"database_id": database_id,
                                              "last_id": last_id}) + "\n", encoding="utf-8")
            time.sleep(0.1)
        remaining = local.execute("SELECT 1 FROM passages WHERE date>=? AND date<? "
                                  "AND passage_id>? LIMIT 1",
                                  (f"{year}-01-01", f"{year+1}-01-01", last_id)).fetchone()
        return {"year": year, "uploaded_this_run": uploaded, "last_id": last_id,
                "complete": remaining is None}
    finally:
        local.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/oireachtas/passages.sqlite"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--database-id", required=True)
    parser.add_argument("--max-passages", type=int, default=40_000)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.index, args.year, args.database_id,
                             args.max_passages, args.refresh), indent=2))


if __name__ == "__main__":
    main()
