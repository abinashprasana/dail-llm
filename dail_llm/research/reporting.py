"""Reports derived from verified artifacts; no generated quality judgments."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import binding, digest, fingerprint, read_json, records, write_json
from .evaluation import phrases
from .memory import POLICIES

REFERENCES = [
    ("Herzog and Mikhaylov (2017), corpus and metadata", "https://arxiv.org/abs/1708.04557"),
    ("Khandelwal et al. (ICLR 2020), kNN-LM", "https://arxiv.org/abs/1911.00172"),
    (
        "Yogatama et al. (TACL 2021), character-level memory",
        "https://aclanthology.org/2021.tacl-1.22/",
    ),
    (
        "Nishida et al. (NAACL Findings 2025), rare-target evaluation",
        "https://aclanthology.org/2025.findings-naacl.331/",
    ),
    (
        "Hirst et al. (2014), institutional-role confounding",
        "https://www.benjamins.com/catalog/dapsac.55.05hir",
    ),
    (
        "Rheault and Cochrane (Political Analysis 2020), metadata conditioning",
        "https://doi.org/10.1017/pan.2019.26",
    ),
]


def bootstrap(left, right, rows, repetitions=1000):
    """Paired BPC difference, resampled at debate level (right minus left)."""
    groups = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for row in rows:
        sid = row["speech_id"]
        if sid not in left or sid not in right:
            continue
        a, b = left[sid], right[sid]
        if a["supported"] != b["supported"]:
            raise ValueError("Paired comparison has different target coverage")
        item = groups[row["debate_id"]]
        item[0] += a["supported_loss"]
        item[1] += a["supported"]
        item[2] += b["supported_loss"]
        item[3] += b["supported"]
    data = np.array(list(groups.values()))
    if len(data) < 2 or not data[:, 1].sum():
        return {"debates": len(data), "interval": None, "reason": "Insufficient independent groups"}
    rng = np.random.default_rng(42)
    values = []
    for _ in range(repetitions):
        sums = data[rng.integers(len(data), size=len(data))].sum(axis=0)
        if sums[1] and sums[3]:
            values.append((sums[2] / sums[3] - sums[0] / sums[1]) / math.log(2))
    sums = data.sum(axis=0)
    return {
        "debates": len(data),
        "difference_bpc": (sums[2] / sums[3] - sums[0] / sums[1]) / math.log(2),
        "interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "replicates": len(values),
        "method": "Paired debate bootstrap; right minus left; no multiplicity correction",
    }


def review_candidates(rows, documents):
    """Reserve space for each available heuristic stratum before filling gaps."""
    pools = defaultdict(list)
    for system, doc in documents.items():
        for sample in doc.get("generations", []):
            pools["ordinary continuation"].append((system, {"kind": "continuation", **sample}))
    recurring = phrases([r for r in rows if r["split"] == "train"])
    names = sorted(
        {
            re.sub(
                r"^(?:Deputy|Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+", "", r["member_name"], flags=re.I
            ).strip()
            for r in rows
        }
    )
    for row in rows:
        if row["split"] == "train":
            continue
        text = row["text"]
        spans = []
        for name in names:
            if len(name) >= 4:
                match = re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I)
                if match:
                    spans.append(("member-name span", match.start(), match.end()))
                    break
        tokens = list(re.finditer(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", text))
        for i in range(len(tokens) - 4):
            if tuple(t.group().lower() for t in tokens[i : i + 5]) in recurring:
                spans.append(("frequent phrase", tokens[i].start(), tokens[i + 4].end()))
                break
        for label, start, end in spans:
            pools[label].append(
                (
                    "source",
                    {
                        "kind": "source passage",
                        "speech_id": row["speech_id"],
                        "prefix": text[max(0, start - 64) : start],
                        "continuation": text[start : max(end, start + 192)],
                    },
                )
            )
    rng = random.Random(42)
    selected, remaining = [], []
    for label in sorted(pools):
        rng.shuffle(pools[label])
        annotated = [
            (system, {**sample, "heuristic_stratum": label}) for system, sample in pools[label]
        ]
        selected.extend(annotated[:40])
        remaining.extend(annotated[40:])
    rng.shuffle(remaining)
    selected.extend(remaining[: 120 - len(selected)])
    rng.shuffle(selected)
    return selected


def report(run: Path):
    if not (run / "manifest.json").exists():
        status = read_json(run / "status.json")
        summary = {
            "schema_version": 2,
            "status": "incomplete",
            "missing_artifacts": ["manifest.json"],
            "execution": status,
            "config_sha256": digest(run / "config.json"),
            "reason": (
                "Data preparation has not produced a verified corpus; no scores are available."
            ),
        }
        write_json(run / "report.json", summary)
        (run / "report.md").write_text(
            "# Dáil LLM research results\n\nStatus: incomplete.\n\n" + summary["reason"] + "\n",
            encoding="utf-8",
        )
        return summary
    config, expected = read_json(run / "config.json"), binding(run)
    rows = records(run)
    files = sorted((run / "evaluations").glob("*.json"))
    documents, ledger = {}, {}
    for path in files:
        doc = read_json(path)
        if doc.get("binding") != expected:
            raise ValueError(f"Incompatible evaluation: {path.name}")
        if "checkpoint_sha256" in doc:
            mode = doc.get("mode", "none")
            checkpoint = run / "models" / f"{mode}-{doc['seed']}" / "best.pt"
            if digest(checkpoint) != doc["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed: {path.name}")
        if "memory_manifest_sha256" in doc:
            memory = run / "memory" / str(doc["seed"]) / doc["policy"]
            manifest = read_json(memory / "manifest.json")
            if (
                digest(memory / "manifest.json") != doc["memory_manifest_sha256"]
                or digest(memory / "keys.npy") != manifest["keys_sha256"]
                or digest(memory / "entries.json") != manifest["entries_sha256"]
            ):
                raise ValueError(f"Memory changed: {path.name}")
        documents[path.stem] = doc
        ledger[path.name] = digest(path)
    required = {f"{mode}-{seed}" for mode in ("none", "party", "role") for seed in config["seeds"]}
    required |= {f"memory-{seed}-{policy}" for policy in POLICIES for seed in config["seeds"]}
    required |= {"ngram", "matching"}
    missing = sorted(required - documents.keys())
    status = read_json(run / "status.json")
    comparisons = {}
    for seed in config["seeds"]:
        base = documents.get(f"none-{seed}")
        uniform = documents.get(f"memory-{seed}-uniform")
        if base:
            for policy in POLICIES:
                memory = documents.get(f"memory-{seed}-{policy}")
                if memory:
                    earlier = [r for r in rows if r["split"] == "earlier"]
                    comparisons[f"base_vs_{policy}/{seed}"] = bootstrap(
                        base["splits"]["earlier"]["per_speech"],
                        memory["result"]["per_speech"],
                        earlier,
                    )
                    if uniform and policy != "uniform":
                        comparisons[f"uniform_vs_{policy}/{seed}"] = bootstrap(
                            uniform["result"]["per_speech"], memory["result"]["per_speech"], earlier
                        )
            for mode in ("party", "role"):
                candidate = documents.get(f"{mode}-{seed}")
                if candidate:
                    for split in ("earlier", "later"):
                        comparisons[f"base_vs_{mode}/{seed}/{split}"] = bootstrap(
                            base["splits"][split]["per_speech"],
                            candidate["splits"][split]["per_speech"],
                            [r for r in rows if r["split"] == split],
                        )
    incomplete = {key: val for key, val in status["stages"].items() if val["status"] != "complete"}
    review, answers = [], {}
    for system, sample in review_candidates(rows, documents):
        rid = fingerprint([system, sample])[:16]
        review.append(
            {
                "review_id": rid,
                "kind": sample["kind"],
                "heuristic_stratum": sample["heuristic_stratum"],
                "prefix": sample["prefix"],
                "text": sample["continuation"],
                "readability": None,
                "substantive_or_formulaic": None,
                "name_span_offsets": [],
                "notes": "",
            }
        )
        answers[rid] = {"system": system, "speech_id": sample["speech_id"]}
    write_json(run / "review-pack.json", review)
    write_json(run / "review-answer-key.json", answers)
    seed_variation = {}
    for mode in ("none", "party", "role"):
        for split in ("earlier", "later"):
            values = [
                documents[f"{mode}-{seed}"]["splits"][split]["aggregate"]["supported_target_bpc"]
                for seed in config["seeds"]
                if f"{mode}-{seed}" in documents
            ]
            values = [v for v in values if v is not None]
            if values:
                seed_variation[f"{mode}/{split}"] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
    summary = {
        "schema_version": 2,
        "binding": expected,
        "evaluation_hashes": ledger,
        "status": "incomplete" if missing or incomplete else "complete",
        "evidence_level": "exploratory pilot"
        if config["name"] == "cpu-pilot"
        else "automated study; human review pending",
        "missing_artifacts": missing,
        "incomplete_stages": incomplete,
        "execution": status,
        "comparisons": comparisons,
        "seed_variation": seed_variation,
        "human_review": {
            "status": "pending",
            "items": len(review),
            "heuristic_strata": {
                label: sum(item["heuristic_stratum"] == label for item in review)
                for label in ("ordinary continuation", "member-name span", "frequent phrase")
            },
        },
        "references": REFERENCES,
    }
    for policy in POLICIES:
        values = [
            documents[f"memory-{seed}-{policy}"]["result"]["aggregate"]["supported_target_bpc"]
            for seed in config["seeds"]
            if f"memory-{seed}-{policy}" in documents
        ]
        values = [v for v in values if v is not None]
        if values:
            seed_variation[f"memory/{policy}/earlier"] = {
                "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    verification_path = run / "verification.json"
    if verification_path.exists():
        verification = read_json(verification_path)
        if verification["binding"] != expected:
            raise ValueError("Verification artifact is incompatible")
        summary["verification"] = {
            "sha256": digest(verification_path),
            "status": verification["status"],
            "checks": len(verification["checks"]),
        }
    write_json(run / "report.json", summary)
    lines = [
        "# Dáil LLM research results",
        "",
        f"Status: {summary['status']}. Evidence: {summary['evidence_level']}.",
        "",
        "Scores use the selected speeches and the recorded checkpoint. "
        "They are not comparable to the earlier published checkpoint's scores.",
        "",
        "| System | Partition | Supported-target BPC | Coverage |",
        "|---|---|---:|---:|",
    ]
    for name, doc in documents.items():
        results = doc.get("splits", {"earlier": doc["result"]} if "result" in doc else {})
        for split, measured in results.items():
            aggregate = measured["aggregate"]
            bpc = aggregate["supported_target_bpc"]
            coverage = aggregate["coverage"]
            formatted_bpc = f"{bpc:.4f}" if bpc is not None else "unavailable"
            formatted_coverage = f"{coverage:.2%}" if coverage is not None else "unavailable"
            lines.append(f"| {name} | {split} | {formatted_bpc} | {formatted_coverage} |")

    def number(value):
        return "unavailable" if value is None else f"{value:.4f}"

    lines += [
        "",
        "## Historical cohorts",
        "",
        "| System | Partition | Continuing-speaker BPC | Matched-passage BPC | Matched targets |",
        "|---|---|---:|---:|---:|",
    ]
    for name, doc in documents.items():
        for split, result in doc.get("splits", {}).items():
            if "matched" not in result:
                continue
            continuing, matched = result["continuing_speakers"], result["matched"]
            lines.append(
                f"| {name} | {split} | {number(continuing['supported_target_bpc'])} "
                f"| {number(matched['supported_target_bpc'])} | {matched['targets']} |"
            )
    cohort = documents.get("matching", {})
    lines += ["", "| Model | Parameters | Additional metadata parameters |", "|---|---:|---:|"]
    for seed in config["seeds"]:
        base = documents.get(f"none-{seed}")
        if base:
            for mode in ("none", "party", "role"):
                candidate = documents.get(f"{mode}-{seed}")
                if candidate:
                    lines.append(
                        f"| {mode}-{seed} | {candidate['parameters']} | "
                        f"{candidate['parameters'] - base['parameters']} |"
                    )
    lines += [
        "",
        f"Matched pairs: {len(cohort.get('pairs', []))}. "
        f"Unmatched earlier passages: {cohort.get('unmatched_earlier', 'unavailable')}; "
        f"unmatched later passages: {cohort.get('unmatched_later', 'unavailable')}. "
        f"Small cohort: {cohort.get('small_cohort', 'unavailable')}.",
        "",
        "## Memory and generation measurements",
        "",
        "| System | Entries | Neighbours / weight / temperature | Retrieval ms per target |",
        "|---|---:|---|---:|",
    ]
    for name, doc in documents.items():
        if "policy" in doc:
            manifest = read_json(
                run / "memory" / str(doc["seed"]) / doc["policy"] / "manifest.json"
            )
            parameters = doc["tuning"]["parameters"]
            lines.append(
                f"| {name} | {manifest['entry_count']} | {parameters['k']} / "
                f"{parameters['weight']} / {parameters['temperature']} | "
                f"{number(doc['retrieval']['ms_per_query'])} |"
            )
    lines += [
        "",
        "| System | Mean generation tokens/s | Mean copied training eight-gram rate |",
        "|---|---:|---:|",
    ]
    for name, doc in documents.items():
        samples = doc.get("generations", [])
        if samples:
            throughput = np.mean([s["tokens_per_second"] for s in samples])
            copying = [
                s["copied_word_8gram_rate"]
                for s in samples
                if s["copied_word_8gram_rate"] is not None
            ]
            lines.append(
                f"| {name} | {number(throughput)} | "
                f"{number(float(np.mean(copying)) if copying else None)} |"
            )
    lines += [
        "",
        "## Heuristic complete-span slices",
        "",
        "Character BPC within fully supported spans; overlapping phrases can share targets.",
        "",
        "| System | Partition | Slice | Supported spans | Span BPC | Exact span accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, doc in documents.items():
        results = doc.get("splits", {"earlier": doc["result"]} if "result" in doc else {})
        for split, measured in results.items():
            for label, item in measured.get("slices", {}).items():
                bpc = (
                    item["loss"] / item["characters"] / math.log(2) if item["characters"] else None
                )
                accuracy = (
                    item["correct_spans"] / item["supported_spans"]
                    if item["supported_spans"]
                    else None
                )
                lines.append(
                    f"| {name} | {split} | {label} | {item['supported_spans']} "
                    f"| {number(bpc)} | {number(accuracy)} |"
                )
    lines += [
        "",
        "## Interpretation",
        "",
        "The memory study compares uniform, speech-balanced, and context-diverse memory "
        "at equal entry counts. Validation chooses retrieval settings. Rare-word and "
        "member-name slices are heuristic matches; source removal measures an effect "
        "through retrieval only.",
        "",
        "The historical study uses one government transition. Continuing-speaker and "
        "matched-passage results are in the evaluation JSON files. Role labels cover only "
        "the three-party cohort identified by recorded affiliations. Some archive affiliations "
        "are outdated; dated membership verification is still needed. "
        "Input masks are sensitivity checks; they do not simulate "
        "how a politician would have spoken.",
        "",
        "Generation quality is unscored until the blinded review pack is completed. "
        "Low repetition can also occur in incoherent text. No positive improvement is "
        "inferred from aggregate scores alone.",
        "",
        f"Execution time: {status['elapsed_seconds']:.1f} seconds. "
        f"Peak process memory: {status['peak_memory_bytes'] / 1024**3:.2f} GiB.",
        "",
        "Missing artifacts: " + (", ".join(missing) or "none"),
        "",
        "## References",
        "",
    ]
    lines += [f"- [{title}]({url})" for title, url in REFERENCES]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
