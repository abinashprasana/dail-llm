"""Stream the documented archive format and select whole debate groups."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import ftfy

from .common import Budget, digest, fingerprint, initialize, write_json

COLUMNS = [
    "speechID",
    "memberID",
    "partyID",
    "constID",
    "title",
    "date",
    "member_name",
    "party_name",
    "const_name",
    "speech",
]
PERIODS = {
    "train": ("2008-01-01", "2009-12-31"),
    "validation": ("2010-01-01", "2010-06-30"),
    "earlier": ("2010-07-01", "2010-12-31"),
    "later": ("2011-04-01", "2011-09-30"),
}
ROLE_SOURCE = "https://www.oireachtas.ie/en/debates/debate/dail/2011-03-09/4/"
CORPUS_SOURCE = "https://alexherzog.net/files/Herzog_Mikhaylov_FADS_2017.pdf"


def normalize(text: str) -> str:
    return " ".join(ftfy.fix_text(text).split())


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", text.lower())


def partition(day: str) -> str | None:
    return next((key for key, (start, end) in PERIODS.items() if start <= day <= end), None)


def role(party: str, day: str) -> str:
    name = normalize(party).casefold()
    # Only the stable three-party cohort is mapped. Transition day is excluded.
    if name not in {"fianna fáil", "fine gael", "labour party", "the labour party", "labour"}:
        return "unmapped"
    if not "2008-01-01" <= day <= "2011-09-30" or day == "2011-03-09":
        return "unmapped"
    ff = name == "fianna fáil"
    government = ff if day < "2011-03-09" else not ff
    return "government" if government else "opposition"


def deduplicate(rows: list[dict], budget=None) -> tuple[list[dict], list[dict]]:
    """Exact set-similarity join; postings only nominate, Jaccard verifies."""
    exact, shingles, postings = {}, {}, defaultdict(set)
    kept, removed = [], []
    for index, row in enumerate(sorted(rows, key=lambda r: (r["date"], int(r["speech_id"])))):
        if budget and index % 25 == 0:
            budget.check()
        text = row["text"]
        key = hashlib.sha256(text.encode()).hexdigest()
        duplicate = exact.get(key)
        reason = "exact"
        tokens = words(text)
        grams = (
            {tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)}
            if len(tokens) >= 50
            else set()
        )
        if duplicate is None and grams:
            counts = Counter(candidate for gram in grams for candidate in postings.get(gram, ()))
            for candidate, overlap in counts.items():
                other = shingles[candidate]
                if overlap >= math.ceil(0.9 / 1.9 * (len(grams) + len(other))):
                    if len(grams & other) / len(grams | other) >= 0.9:
                        duplicate, reason = candidate, "near"
                        break
        if duplicate is not None:
            removed.append({"speech_id": row["speech_id"], "kept_id": duplicate, "reason": reason})
            continue
        sid = row["speech_id"]
        exact[key] = sid
        if grams:
            shingles[sid] = grams
            for gram in grams:
                postings[gram].add(sid)
        kept.append(row)
    return kept, removed


def choose_groups(groups: list[tuple], cap: int) -> list[str]:
    months = defaultdict(list)
    for gid, month, size in groups:
        if size <= cap:
            months[month].append((fingerprint(gid), gid, size))
    for values in months.values():
        values.sort(reverse=True)
    chosen, total = [], 0
    while any(months.values()):
        for month in sorted(months):
            if months[month]:
                _, gid, size = months[month].pop()
                if total + size <= cap:
                    chosen.append(gid)
                    total += size
    return chosen


def prepare(run: Path, profile: Path, source: Path):
    config = initialize(run, profile)
    with Budget(run, "prepare") as budget:
        counts = Counter()
        staging = run / "selection.sqlite"
        con = sqlite3.connect(staging)
        try:
            con.execute("CREATE TABLE rows (gid TEXT, month TEXT, split TEXT, size INT, row TEXT)")
            # Raw byte hashing includes all records, including those outside the selected periods.
            sha = hashlib.sha256()
            with (
                source.open("rb") as stream,
                (run / "malformed.jsonl").open("w", encoding="utf-8") as errors,
            ):
                header = stream.readline()
                sha.update(header)
                if header.decode("utf-8-sig").rstrip("\r\n").split("\t") != COLUMNS:
                    raise ValueError("Archive header differs from the documented ten columns")
                pending = []
                for line_number, raw in enumerate(stream, 2):
                    sha.update(raw)
                    counts["rows_seen"] += 1
                    if line_number % 10000 == 0:
                        budget.check()
                    try:
                        line = raw.decode("utf-8").rstrip("\r\n")
                        values = next(csv.reader([line], delimiter="\t", quoting=csv.QUOTE_NONE))
                        if len(values) != 10:
                            raise ValueError("column_count")
                        day = date.fromisoformat(values[5]).isoformat()
                        int(values[0])
                    except (UnicodeError, ValueError, csv.Error) as error:
                        counts["malformed"] += 1
                        errors.write(
                            json.dumps(
                                {
                                    "line": line_number,
                                    "reason": str(error),
                                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                }
                            )
                            + "\n"
                        )
                        continue
                    split = partition(day)
                    if split is None:
                        counts["outside_periods"] += 1
                        continue
                    text = normalize(values[9])
                    if len(text) < 2:
                        counts["too_short"] += 1
                        continue
                    counts["normalized"] += text != values[9]
                    title = normalize(values[4])
                    gid = fingerprint([day, title.casefold()])
                    row = dict(
                        zip(
                            [
                                "speech_id",
                                "member_id",
                                "party_id",
                                "constituency_id",
                                "title",
                                "date",
                                "member_name",
                                "party",
                                "constituency",
                                "original_text",
                            ],
                            values,
                            strict=True,
                        )
                    )
                    row.update(
                        text=text,
                        debate_id=gid,
                        split=split,
                        role=role(values[7], day),
                        source_line=line_number,
                        raw_sha256=hashlib.sha256(raw).hexdigest(),
                    )
                    pending.append(
                        (gid, day[:7], split, len(text), json.dumps(row, ensure_ascii=False))
                    )
                    counts["eligible"] += 1
                    if len(pending) >= 1000:
                        con.executemany("INSERT INTO rows VALUES (?,?,?,?,?)", pending)
                        pending.clear()
                con.executemany("INSERT INTO rows VALUES (?,?,?,?,?)", pending)
            con.execute("CREATE INDEX group_lookup ON rows(gid)")
            selected, selection = [], {}
            for split in PERIODS:
                cap = config["train_chars"] if split == "train" else config["eval_chars"]
                groups = con.execute(
                    "SELECT gid, month, SUM(size) FROM rows WHERE split=? GROUP BY gid", (split,)
                ).fetchall()
                gids = choose_groups(groups, cap)
                selection[split] = {
                    "candidate_groups": len(groups),
                    "selected_groups": len(gids),
                    "oversize_groups": sum(size > cap for _, _, size in groups),
                    "cap": cap,
                }
                for gid in gids:
                    selected.extend(
                        json.loads(row[0])
                        for row in con.execute("SELECT row FROM rows WHERE gid=?", (gid,))
                    )
            retained, duplicates = deduplicate(selected, budget)
            with (run / "speeches.jsonl").open("w", encoding="utf-8") as stream:
                for row in retained:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["duplicate_exclusions"] = len(duplicates)
            write_json(run / "duplicates.json", duplicates)
            for split in PERIODS:
                subset = [r for r in retained if r["split"] == split]
                selection[split].update(
                    speeches=len(subset), characters=sum(len(r["text"]) for r in subset)
                )
                if not subset:
                    raise ValueError(f"No complete debate groups fit the {split} budget")
            write_json(
                run / "manifest.json",
                {
                    "schema_version": 2,
                    "source_sha256": sha.hexdigest(),
                    "source_bytes": source.stat().st_size,
                    "citation": (
                        "Alexander Herzog and Slava J. Mikhaylov (2017). "
                        "Database of Parliamentary Speeches in Ireland, 1919–2013."
                    ),
                    "source": CORPUS_SOURCE,
                    "role_source": ROLE_SOURCE,
                    "periods": PERIODS,
                    "counts": dict(counts),
                    "selection": selection,
                    "language": "unclassified parliamentary text",
                    "dedup_scope": (
                        "selected speeches across all four partitions, chronological retention"
                    ),
                    "records_sha256": digest(run / "speeches.jsonl"),
                },
            )
        finally:
            con.close()
            # Staging is local derived data, never the source archive.
            staging.unlink(missing_ok=True)
