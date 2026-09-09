"""Read-only research inspection and explicit, sanitized public releases.

This module deliberately leaves the hash-bound scientific implementation untouched.
Run with ``python -m dail_llm.research.publication --help``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from .common import binding, digest, fingerprint, read_json, records, write_json
from .memory import POLICIES, Memory
from .modeling import evaluating
from .training import load


class ResearchReader:
    """Validated, cached assets; every query gets its own search state."""

    def __init__(self, run: Path, seed: int = 42):
        self.run, self.seed = run, seed
        self.config = read_json(run / "config.json")
        self.binding = binding(run)
        self.report = read_json(run / "report.json")
        verification = read_json(run / "verification.json")
        status = read_json(run / "status.json")
        if (
            self.report.get("status") != "complete"
            or self.report.get("binding") != self.binding
            or verification.get("binding") != self.binding
            or verification.get("status") != "passed"
            or not verification.get("checks")
            or not all(verification["checks"].values())
            or self.report.get("verification", {}).get("sha256")
            != digest(run / "verification.json")
            or any(s["status"] != "complete" for s in status["stages"].values())
        ):
            raise ValueError("A compatible, completed and verified research run is required")
        self.docs = {}
        for name, expected in self.report["evaluation_hashes"].items():
            if Path(name).name != name or digest(run / "evaluations" / name) != expected:
                raise ValueError("Evaluation hash mismatch")
            doc = read_json(run / "evaluations" / name)
            if doc["binding"] != self.binding:
                raise ValueError("Evaluation binding mismatch")
            if "checkpoint_sha256" in doc:
                path = run / "models" / f"{doc.get('mode', 'none')}-{doc['seed']}" / "best.pt"
                if digest(path) != doc["checkpoint_sha256"]:
                    raise ValueError("Evaluation checkpoint mismatch")
            self.docs[Path(name).stem] = doc
        self.checkpoint = run / "models" / f"none-{seed}" / "best.pt"
        self.model, self.tokenizer, self.state = load(run, self.checkpoint)
        self.model.eval()
        self.sources = {r["speech_id"]: r for r in records(run)}
        self.memories, self.parameters = {}, {}
        for policy in POLICIES:
            path = run / "memory" / str(seed) / policy
            doc = self.docs[f"memory-{seed}-{policy}"]
            if digest(path / "manifest.json") != doc["memory_manifest_sha256"]:
                raise ValueError("Evaluation memory mismatch")
            self.memories[policy] = Memory(run, self.checkpoint, self.tokenizer, path)
            self.parameters[policy] = doc["tuning"]["parameters"]
        if len({len(m.entries) for m in self.memories.values()}) != 1:
            raise ValueError("Published memory comparisons require equal entry counts")
        self.provenance = {
            "binding": self.binding,
            "checkpoint_sha256": digest(self.checkpoint),
            "tokenizer_sha256": fingerprint(self.tokenizer.state()),
            "report_sha256": digest(run / "report.json"),
        }
        self.release_id = "pilot-" + fingerprint(self.provenance)[:12]

    def inspect(self, prefix: str, policy: str, excluded_speech: str | None = None):
        if policy not in POLICIES or not 1 <= len(prefix) <= self.model.block_size:
            raise ValueError("Choose a known policy and a prefix of 1–256 characters")
        original = self.memories[policy]
        if excluded_speech and not any(e["speech_id"] == excluded_speech for e in original.entries):
            raise ValueError("The excluded speech is not in this memory")
        before, after = copy.copy(original), copy.copy(original)
        after.allowed = np.array(
            [i for i in original.allowed if original.entries[i]["speech_id"] != excluded_speech],
            dtype=np.int64,
        )
        before.seconds = after.seconds = 0.0
        before.queries = after.queries = 0
        params = self.parameters[policy]
        with evaluating(self.model):
            encoded = self.tokenizer.encode(prefix)
            logits, hidden = self.model.predict(torch.tensor([encoded]))
            base, query = logits[0, -1:].softmax(-1), hidden[0, -1:]
            a, b = before.mix(query, base, **params), after.mix(query, base, **params)

            def neighbours(memory):
                distances, ids = memory.nearest(query, params["k"])
                weights = torch.softmax(-distances / params["temperature"], -1)[0]
                result = []
                for index, weight in zip(ids[0].tolist(), weights.tolist(), strict=True):
                    entry = memory.entries[index]
                    row, offset = self.sources[entry["speech_id"]], entry["offset"]
                    start = max(0, offset - 64)
                    result.append(
                        {
                            **entry,
                            "memory_mass": weight,
                            "mixture_mass": weight * params["weight"],
                            "source_span": row["text"][start : offset + 32],
                            "span_start": start,
                            "member": row["member_name"],
                            "title": row.get("title", ""),
                        }
                    )
                return result

            return {
                "schema_version": 1,
                "release_id": self.release_id,
                "policy": policy,
                "prefix": prefix,
                "excluded_speech": excluded_speech,
                "parameters": params,
                "remaining_entries": len(after.allowed),
                "removed_entries": len(before.allowed) - len(after.allowed),
                "unknown_characters": sum(i == 0 for i in encoded),
                "neighbours_before": neighbours(before),
                "neighbours_after": neighbours(after),
                "distributions": [
                    {
                        "character": self.tokenizer.label(i),
                        "base": float(base[0, i]),
                        "before": float(a[0, i]),
                        "after": float(b[0, i]),
                        "delta": float(b[0, i] - a[0, i]),
                    }
                    for i in range(self.tokenizer.size)
                ],
                "provenance": self.provenance,
            }


def metrics(result):
    """Allowlist measured values; no raw passages or filesystem identifiers."""
    allowed = (
        "targets",
        "supported",
        "unknown",
        "mapped_token_loss",
        "supported_target_bpc",
        "coverage",
        "supported_accuracy",
    )
    return {key: result[key] for key in allowed if key in result}


def public_summary(reader):
    from .evaluation import summarize

    manifest = read_json(reader.run / "manifest.json")
    rows = list(reader.sources.values())
    matching = reader.docs["matching"]
    continuing = {r["member_id"] for r in rows if r["split"] == "earlier"} & {
        r["member_id"] for r in rows if r["split"] == "later"
    }
    history, memory = {}, {}
    for name in ("none", "party", "role", "ngram"):
        doc = reader.docs[name if name == "ngram" else f"{name}-{reader.seed}"]
        history[name] = {"parameters": doc.get("parameters", 0), "splits": {}}
        for split, result in doc["splits"].items():
            if name == "ngram":
                # Same speech-level cohorts as the decoder evaluations; no fitting.
                matched = {p[split] for p in matching["pairs"]}
                result = {
                    **result,
                    "continuing_speakers": summarize(
                        result["per_speech"][r["speech_id"]]
                        for r in rows
                        if r["split"] == split and r["member_id"] in continuing
                    ),
                    "matched": summarize(result["per_speech"][sid] for sid in matched),
                }
            history[name]["splits"][split] = {
                **{
                    k: metrics(result[k])
                    for k in ("aggregate", "continuing_speakers", "matched")
                    if k in result
                },
                "perturbations": {
                    key: {
                        "masked_characters": item["masked_characters"],
                        "masked_input": metrics(item["masked_input"]),
                        "same_target_control": metrics(item["same_target_control"]),
                    }
                    for key, item in result.get("perturbations", {}).items()
                },
            }
    for policy in POLICIES:
        doc = reader.docs[f"memory-{reader.seed}-{policy}"]
        slices = {}
        for label, item in doc["result"]["slices"].items():
            slices[label] = {
                "spans": item["supported_spans"],
                "bpc": item["loss"] / item["characters"] / math.log(2)
                if item["characters"]
                else None,
                "accuracy": item["correct_spans"] / item["supported_spans"]
                if item["supported_spans"]
                else None,
            }
        generations = doc.get("generations", [])
        copying = [
            s["copied_word_8gram_rate"]
            for s in generations
            if s["copied_word_8gram_rate"] is not None
        ]
        memory[policy] = {
            "metrics": metrics(doc["result"]["aggregate"]),
            "slices": slices,
            "entries": len(reader.memories[policy].entries),
            "parameters": reader.parameters[policy],
            "retrieval": doc["retrieval"],
            "copying": sum(copying) / len(copying) if copying else None,
            "tokens_per_second": sum(s["tokens_per_second"] for s in generations) / len(generations)
            if generations
            else None,
            "comparison": reader.report["comparisons"].get(f"uniform_vs_{policy}/{reader.seed}"),
        }
    matching = reader.docs["matching"]
    rows = list(reader.sources.values())
    continuing = {r["member_id"] for r in rows if r["split"] == "earlier"} & {
        r["member_id"] for r in rows if r["split"] == "later"
    }
    return {
        "schema_version": 1,
        "release_id": reader.release_id,
        "label": "CPU pilot",
        "evidence": "Exploratory results",
        "seed": reader.seed,
        "steps": reader.config["steps"],
        "batch_size": reader.config["batch_size"],
        "provenance": reader.provenance,
        "periods": manifest["periods"],
        "selection": manifest["selection"],
        "counts": manifest["counts"],
        "source": manifest["source"],
        "source_sha256": manifest["source_sha256"],
        "history": history,
        "memory": memory,
        "cohort": {
            "pairs": len(matching["pairs"]),
            "unmatched_earlier": matching["unmatched_earlier"],
            "unmatched_later": matching["unmatched_later"],
            "continuing_speakers": len(continuing),
            "continuing_speeches": {
                s: sum(r["split"] == s and r["member_id"] in continuing for r in rows)
                for s in ("earlier", "later")
            },
        },
        "verification": reader.report["verification"],
        "references": reader.report["references"],
        "human_review": reader.report["human_review"],
        "training_history": reader.state.get("history", []),
        "execution": {
            k: reader.report["execution"][k] for k in ("elapsed_seconds", "peak_memory_bytes")
        },
    }


def example_prefixes(reader):
    from .evaluation import phrases

    rows = sorted(reader.sources.values(), key=lambda r: fingerprint(r["speech_id"]))
    recurring = phrases([r for r in rows if r["split"] == "train"])
    choices = [{"prefix": "The Minister for", "kind": "Fixed prefix"}]
    pools = {"Recurring phrase (heuristic)": [], "Ordinary continuation (heuristic)": []}
    for row in rows:
        if row["split"] != "earlier":
            continue
        tokens = list(re.finditer(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", row["text"]))
        for i in range(len(tokens) - 4):
            if tuple(t.group().lower() for t in tokens[i : i + 5]) in recurring:
                end = tokens[i + 3].end()
                pools["Recurring phrase (heuristic)"].append(
                    row["text"][max(0, end - min(64, reader.model.block_size)) : end]
                )
                break
        pools["Ordinary continuation (heuristic)"].append(
            row["text"][: min(64, reader.model.block_size)]
        )
    seen = {choices[0]["prefix"]}
    while any(pools.values()) and len(choices) < 8:
        for kind, pool in pools.items():
            if not pool or len(choices) == 8:
                continue
            prefix = pool.pop(0)
            if prefix and prefix not in seen:
                choices.append({"prefix": prefix, "kind": kind})
                seen.add(prefix)
    return choices


def export(run: Path, destination: Path, seed=42):
    run, destination = run.resolve(), destination.resolve()
    if destination == run or destination.is_relative_to(run):
        raise ValueError("Public releases must be separate from the original run")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Choose an empty release directory")
    reader = ResearchReader(run, seed)
    summary = public_summary(reader)
    summary["examples"] = []
    # A separate file per policy/example keeps each request small. Exclusions only
    # store changed fields; the browser composes them with the original inspection.
    for number, choice in enumerate(example_prefixes(reader)):
        item = {**choice, "id": f"example-{number + 1}", "files": {}}
        for policy in POLICIES:
            before = reader.inspect(choice["prefix"], policy)
            exclusions = {}
            source_entries = {}
            for sid in sorted({e["speech_id"] for e in before["neighbours_before"]}):
                after = reader.inspect(choice["prefix"], policy, sid)
                exclusions[sid] = {
                    k: after[k] for k in ("excluded_speech", "remaining_entries", "removed_entries")
                }
                exclusions[sid]["after"] = [d["after"] for d in after["distributions"]]
                exclusions[sid]["neighbours"] = []
                for entry in after["neighbours_after"]:
                    key = f"{entry['speech_id']}:{entry['offset']}"
                    source_entries[key] = {
                        k: v for k, v in entry.items() if k not in ("memory_mass", "mixture_mass")
                    }
                    exclusions[sid]["neighbours"].append([key, entry["memory_mass"]])
            name = f"{item['id']}-{policy}.json"
            value = {"original": before, "exclusions": exclusions, "source_entries": source_entries}
            # Compact JSON keeps the actual measurements intact, without rounding.
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if len(encoded.encode()) > 250_000:
                raise ValueError(f"Example exceeds public payload budget: {name}")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / name).write_text(encoded + "\n", encoding="utf-8")
            item["files"][policy] = {"name": name, "sha256": digest(destination / name)}
        summary["examples"].append(item)
    write_json(destination / "summary.json", summary)
    if (destination / "summary.json").stat().st_size > 150_000:
        raise ValueError("Summary exceeds public payload budget")
    return summary


def private_bundle(run: Path, destination: Path, seed=42):
    """Copy only the artifacts needed for validation and read-only serving."""
    run, destination = run.resolve(), destination.resolve()
    if destination == run or destination.is_relative_to(run):
        raise ValueError("A private bundle must be separate from the original run")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Choose an empty private bundle directory")
    reader = ResearchReader(run, seed)
    names = {
        "config.json",
        "manifest.json",
        "speeches.jsonl",
        "report.json",
        "verification.json",
        "status.json",
    }
    for name in reader.report["evaluation_hashes"]:
        names.add(f"evaluations/{name}")
        doc = read_json(run / "evaluations" / name)
        if "checkpoint_sha256" in doc:
            names.add(f"models/{doc.get('mode', 'none')}-{doc['seed']}/best.pt")
    for policy in POLICIES:
        names.update(
            f"memory/{seed}/{policy}/{name}"
            for name in ("manifest.json", "entries.json", "keys.npy")
        )
    for name in sorted(names):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / name, target)
        if digest(run / name) != digest(target):
            raise ValueError("Private artifact copy mismatch")
    if ResearchReader(destination, seed).release_id != reader.release_id:
        raise ValueError("Private bundle release mismatch")
    return {"label": "Private research bundle", "release_id": reader.release_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--private-bundle",
        action="store_true",
        help="Copy a private runtime bundle instead of public examples",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    operation = private_bundle if args.private_bundle else export
    result = operation(args.run, args.destination, args.seed)
    print(f"Published {result['label']}: {result['release_id']}")


if __name__ == "__main__":
    main()
