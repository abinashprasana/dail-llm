"""Create source-linked evaluation candidates for independent human review.

These are candidate questions, not a reviewed release dataset. Review each source,
answerability label, and gold passage before using them as held-out evidence.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

INDEX = Path("data/oireachtas/passages.sqlite")
OUT = Path("eval/candidates.jsonl")


def main() -> None:
    db = sqlite3.connect(INDEX)
    db.row_factory = sqlite3.Row
    rows: list[dict] = []
    for year in range(2014, 2027):
        candidates = db.execute("""
            SELECT passage_id, date, speaker, title, text, source_url
            FROM passages WHERE date BETWEEN ? AND ?
              AND speaker != 'Unknown speaker' AND length(text) BETWEEN 300 AND 1800
              AND length(title) BETWEEN 20 AND 100
              AND title NOT LIKE '%Questions%' AND title NOT LIKE '%Business%'
            ORDER BY date, passage_id
        """, (f"{year}-01-01", f"{year}-12-31")).fetchall()
        if not candidates:
            continue
        stride = max(1, len(candidates) // 3)
        for row in candidates[::stride][:3]:
            phrase = re.sub(r"\s+", " ", row["text"]).strip()[:130]
            rows.append({
                "question": f"On {row['date']}, who said: “{phrase}…”?",
                "start_date": row["date"], "end_date": row["date"],
                "answerable": True, "gold_passage_ids": [row["passage_id"]],
                "source_url": row["source_url"], "expected_speaker": row["speaker"],
                "reviewed": False,
            })
        rows.append({
            "question": f"On 1 January {year}, who discussed the imaginary Zyxqv bill?",
            "start_date": f"{year}-01-01", "end_date": f"{year}-01-01",
            "answerable": False, "gold_passage_ids": [], "reviewed": False,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                   encoding="utf-8")
    print(f"Wrote {len(rows)} unreviewed candidates to {OUT}")


if __name__ == "__main__":
    main()
