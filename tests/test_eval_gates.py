"""Release gates on the LangSmith experiment path; no network or credentials needed."""

import json
from pathlib import Path

import pytest

from scripts.evaluate_qa import load_questions, publish_langsmith

ROW = {
    "question": "Who spoke about housing?",
    "gold_passage_ids": ["dail:2025-06-01:dbsect_4:spk_2"],
    "answerable": True,
}


def test_langsmith_upload_refuses_unreviewed_questions(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    for rows in ([ROW], [{**ROW, "reviewed": False}], [{**ROW, "reviewed": True}, ROW]):
        with pytest.raises(ValueError, match="independently reviewed"):
            publish_langsmith(rows, None, "dail-test")


def test_langsmith_upload_requires_credentials(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY"):
        publish_langsmith([{**ROW, "reviewed": True}], None, "dail-test")


def test_questions_need_gold_passages_and_answerability(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"question": "No gold set"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gold_passage_ids"):
        load_questions(path)
    path.write_text(json.dumps(ROW) + "\n\n", encoding="utf-8")
    assert load_questions(path) == [ROW]
