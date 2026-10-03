"""Compare lexical retrieval with the grounded graph on reviewed questions."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from dail_llm.qa.answer import ask
from dail_llm.qa.index import SQLiteRetriever


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any("question" not in row or "gold_passage_ids" not in row or
                       "answerable" not in row for row in rows):
        raise ValueError("Questions need question, gold_passage_ids and answerable")
    return rows


def evaluate_local(rows: list[dict], retriever: SQLiteRetriever) -> dict:
    results = []
    for row in rows:
        started = time.perf_counter()
        baseline = retriever.search(row["question"], row.get("start_date"),
                                    row.get("end_date"), 5)
        baseline_ms = round((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        graph = ask(row["question"], retriever, row.get("start_date"), row.get("end_date"))
        graph_ms = round((time.perf_counter() - started) * 1000)
        gold = set(row["gold_passage_ids"])
        baseline_ids = {passage.passage_id for passage in baseline}
        graph_ids = {passage["passage_id"] for passage in graph["sources"]}
        results.append({
            "question": row["question"], "answerable": row["answerable"],
            "gold_passage_ids": sorted(gold),
            "baseline_recall_at_5": len(gold & baseline_ids) / len(gold) if gold else None,
            "graph_recall_at_5": len(gold & graph_ids) / len(gold) if gold else None,
            "baseline_ms": baseline_ms, "graph_ms": graph_ms,
            "graph_status": graph["status"],
            "refusal_correct": (graph["status"] == "insufficient_evidence")
            == (not row["answerable"]),
            "citation_ids_valid": set(graph["citation_ids"]) <= graph_ids,
            "claim_support_reviewed": False,
        })
    def mean(values):
        return sum(values) / len(values) if values else None

    return {"generated_at": datetime.now(UTC).isoformat(), "question_count": len(rows),
            "coverage": retriever.coverage(),
            "summary": {
                "baseline_recall_at_5": mean([x["baseline_recall_at_5"] for x in results
                                              if x["baseline_recall_at_5"] is not None]),
                "graph_recall_at_5": mean([x["graph_recall_at_5"] for x in results
                                           if x["graph_recall_at_5"] is not None]),
                "refusal_accuracy": mean([int(x["refusal_correct"]) for x in results]),
                "citation_id_validity": mean([int(x["citation_ids_valid"]) for x in results]),
                "baseline_latency_ms": mean([x["baseline_ms"] for x in results]),
                "graph_latency_ms": mean([x["graph_ms"] for x in results]),
            }, "results": results}


def publish_langsmith(rows: list[dict], retriever: SQLiteRetriever, name: str) -> None:
    """Upload a reviewed set and run a named graph experiment when credentials exist."""
    if any(not row.get("reviewed", False) for row in rows):
        raise ValueError("LangSmith upload requires every question to be independently reviewed")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY before publishing an experiment")
    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    if client.has_dataset(dataset_name=name):
        raise ValueError(f"Dataset {name} already exists; use a new immutable version name")
    client.create_dataset(name, description="Reviewed Dáil Official Report Q&A passages")
    client.create_examples(dataset_name=name, examples=[{
        "inputs": {"question": row["question"], "start_date": row.get("start_date"),
                   "end_date": row.get("end_date")},
        "outputs": {"gold_passage_ids": row["gold_passage_ids"],
                    "answerable": row["answerable"]},
    } for row in rows])

    def target(inputs: dict) -> dict:
        return ask(inputs["question"], retriever, inputs.get("start_date"),
                   inputs.get("end_date"))

    def retrieval_recall(outputs: dict, reference_outputs: dict) -> dict:
        gold = set(reference_outputs["gold_passage_ids"])
        found = {source["passage_id"] for source in outputs["sources"]}
        return {"key": "recall_at_5", "score": len(gold & found) / len(gold) if gold else 1.0}

    def refusal(outputs: dict, reference_outputs: dict) -> dict:
        correct = (outputs["status"] == "insufficient_evidence") == (
            not reference_outputs["answerable"])
        return {"key": "refusal_correct", "score": int(correct)}

    evaluate(target, data=name, evaluators=[retrieval_recall, refusal],
             experiment_prefix="dail-grounded-qa", max_concurrency=1, client=client)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("--index", type=Path, default=Path("data/oireachtas/passages.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/local.json"))
    parser.add_argument("--langsmith-dataset")
    args = parser.parse_args()
    rows = load_questions(args.questions)
    retriever = SQLiteRetriever(args.index)
    report = evaluate_local(rows, retriever)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if args.langsmith_dataset:
        publish_langsmith(rows, retriever, args.langsmith_dataset)


if __name__ == "__main__":
    main()
