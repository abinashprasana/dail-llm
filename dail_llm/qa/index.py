"""SQLite FTS5 index for attributed Dáil speech passages."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Protocol

from .corpus import Passage, parse_debate, query_terms

MONTH_NAMES = {"january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"}


def match_queries(terms: list[str], start: str | None, end: str | None) -> list[str]:
    """Try a selective match first for a short date window, then retain OR recall."""
    def quoted(values: list[str], join: str) -> str:
        return join.join('"' + term.replace('"', '') + '"' for term in values)
    broad = quoted(terms, " OR ")
    if not start or not end:
        return [broad]
    try:
        short_window = 0 <= (date.fromisoformat(end) - date.fromisoformat(start)).days <= 7
    except ValueError:
        return [broad]
    topics = [term for term in terms if not term.isdigit() and term not in MONTH_NAMES]
    if not short_window or len(topics) < 2:
        return [broad]
    selective = quoted(topics[:2], " AND ")
    return [selective, broad] if selective != broad else [broad]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            date TEXT PRIMARY KEY, sha256 TEXT NOT NULL, passage_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS passages (
            passage_id TEXT PRIMARY KEY, date TEXT NOT NULL, speaker TEXT NOT NULL,
            title TEXT NOT NULL, text TEXT NOT NULL, source_url TEXT NOT NULL,
            language TEXT, party TEXT
        );
        CREATE INDEX IF NOT EXISTS passages_date ON passages(date);
        CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(
            passage_id UNINDEXED, text, title, speaker, tokenize='unicode61 remove_diacritics 0'
        );
    """)
    return db


def index_corpus(raw_dir: Path, db_path: Path, year: int | None = None) -> dict:
    db = connect(db_path)
    changed = 0
    try:
        pattern = f"{year}-*.xml" if year else "*.xml"
        for source in sorted(raw_dir.glob(pattern)):
            day = source.stem
            xml = source.read_bytes()
            checksum = hashlib.sha256(xml).hexdigest()
            old = db.execute("SELECT sha256 FROM sources WHERE date=?", (day,)).fetchone()
            if old and old["sha256"] == checksum:
                continue
            passages = parse_debate(xml, day)
            with db:
                db.execute("DELETE FROM passage_fts WHERE passage_id IN "
                           "(SELECT passage_id FROM passages WHERE date=?)", (day,))
                db.execute("DELETE FROM passages WHERE date=?", (day,))
                db.executemany("""
                    INSERT INTO passages VALUES (:passage_id,:date,:speaker,:title,:text,
                                                :source_url,:language,:party)
                """, [p.as_dict() for p in passages])
                db.executemany("INSERT INTO passage_fts(passage_id,text,title,speaker) "
                               "VALUES (?,?,?,?)",
                               [(p.passage_id, p.text, p.title, p.speaker) for p in passages])
                db.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?)",
                           (day, checksum, len(passages)))
            changed += 1
        row = db.execute("SELECT COUNT(*) days, SUM(passage_count) passages, "
                         "MIN(date) first_date, MAX(date) last_date FROM sources").fetchone()
        return {"changed_days": changed, **dict(row)}
    finally:
        db.close()


class SQLiteRetriever:
    def __init__(self, path: Path):
        self.path = path

    def coverage(self) -> dict:
        if not self.path.exists():
            return {"first_date": None, "last_date": None, "days": 0, "passages": 0}
        db = connect(self.path)
        try:
            row = db.execute("SELECT MIN(date) first_date, MAX(date) last_date, "
                             "COUNT(*) days, COALESCE(SUM(passage_count),0) passages "
                             "FROM sources").fetchone()
            return dict(row)
        finally:
            db.close()

    def search(self, query: str, start: str | None = None, end: str | None = None,
               limit: int = 5) -> list[Passage]:
        terms = query_terms(query)
        if not terms or not self.path.exists():
            return []
        db = connect(self.path)
        try:
            rows = []
            for match in match_queries(terms, start, end):
                rows = db.execute("""
                    SELECT p.* FROM passage_fts f JOIN passages p ON p.passage_id=f.passage_id
                    WHERE passage_fts MATCH ? AND (? IS NULL OR p.date>=?)
                      AND (? IS NULL OR p.date<=?)
                    ORDER BY bm25(passage_fts, 0, 1, 2, 0.5) LIMIT ?
                """, (match, start, start, end, end, limit)).fetchall()
                if rows:
                    break
            return [Passage(**dict(row)) for row in rows]
        finally:
            db.close()


class Retriever(Protocol):
    def coverage(self) -> dict: ...

    def search(self, query: str, start: str | None = None, end: str | None = None,
               limit: int = 5) -> list[Passage]: ...


class D1Retriever:
    """Search free-tier year shards through Cloudflare's documented D1 REST API."""

    def __init__(self, databases: dict[int, str], coverage: dict):
        self.databases = databases
        self._coverage = coverage
        self.account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        self.token = os.environ["CLOUDFLARE_D1_READ_TOKEN"]

    def coverage(self) -> dict:
        return self._coverage

    def _query(self, database_id: str, sql: str, params: list) -> list[dict]:
        url = ("https://api.cloudflare.com/client/v4/accounts/"
               f"{self.account}/d1/database/{database_id}/query")
        request = urllib.request.Request(url, data=json.dumps({"sql": sql, "params": params})
                                         .encode("utf-8"), headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
        })
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.load(response)
        if not payload.get("success") or not payload.get("result", [{}])[0].get("success"):
            raise RuntimeError("D1 search failed")
        return payload["result"][0]["results"]

    def search(self, query: str, start: str | None = None, end: str | None = None,
               limit: int = 5) -> list[Passage]:
        terms = query_terms(query)
        if not terms:
            return []
        years = [year for year in self.databases
                 if (not start or year >= int(start[:4])) and
                 (not end or year <= int(end[:4]))]
        # Adjacent years can share one free-tier D1 database. Query each
        # database once, then apply the exact date filter in SQL.
        database_ids = list(dict.fromkeys(self.databases[year] for year in sorted(years)))
        sql = ("SELECT p.*, bm25(passage_fts, 0, 1, 2, 0.5) score "
               "FROM passage_fts f JOIN passages p ON p.passage_id=f.passage_id "
               "WHERE passage_fts MATCH ? AND (? IS NULL OR p.date>=?) "
               "AND (? IS NULL OR p.date<=?) ORDER BY score LIMIT ?")
        rows = []
        for match in match_queries(terms, start, end):
            params = [match, start, start, end, end, limit]
            with ThreadPoolExecutor(max_workers=4) as pool:
                batches = list(pool.map(lambda database_id, params=params:
                                        self._query(database_id, sql, params),
                                        database_ids))
            rows = sorted((row for batch in batches for row in batch),
                          key=lambda row: row["score"])
            if rows:
                break
        return [Passage(**{key: row[key] for key in Passage.__dataclass_fields__})
                for row in rows[:limit]]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/oireachtas/raw"))
    parser.add_argument("--index", type=Path, default=Path("data/oireachtas/passages.sqlite"))
    parser.add_argument("--year", type=int)
    args = parser.parse_args()
    print(json.dumps(index_corpus(args.raw, args.index, args.year), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
