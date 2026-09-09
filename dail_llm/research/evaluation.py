"""Shared baselines, historical cohorts, and explicit evaluation slices."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import Budget, binding, digest, read_json, records, write_json
from .data import words
from .memory import POLICIES, Memory, tune
from .modeling import evaluating, windows
from .training import load, score


class WittenBell:
    def __init__(self, rows, tokenizer, block_size):
        self.size = tokenizer.size
        self.counts = defaultdict(Counter)
        for w in windows(rows, tokenizer, block_size):
            for i, target in enumerate(w["y"][: w["length"]]):
                prefix = w["x"][: i + 1]
                for order in range(min(4, len(prefix)) + 1):
                    context = tuple(prefix[-order:]) if order else ()
                    self.counts[context][target] += 1
        self.totals = {key: sum(value.values()) for key, value in self.counts.items()}

    def probability(self, prefix, target):
        probability = 1 / self.size
        for order in range(min(4, len(prefix)) + 1):
            context = tuple(prefix[-order:]) if order else ()
            counts = self.counts.get(context)
            if counts:
                n, t = self.totals[context], len(counts)
                probability = (counts.get(target, 0) + t * probability) / (n + t)
        return probability

    def evaluate(self, rows, tokenizer, block_size, budget):
        per_speech = defaultdict(
            lambda: {
                "targets": 0,
                "supported": 0,
                "unknown": 0,
                "loss": 0.0,
                "supported_loss": 0.0,
                "correct": 0,
            }
        )
        targets = []
        started = time.perf_counter()
        for w in windows(rows, tokenizer, block_size):
            budget.check()
            stats = per_speech[w["row"]["speech_id"]]
            for i, target in enumerate(w["y"][: w["length"]]):
                prefix = w["x"][: i + 1]
                probabilities = [self.probability(prefix, t) for t in range(self.size)]
                loss = -math.log(max(probabilities[target], 1e-30))
                known = target != tokenizer.unknown
                correct = max(range(self.size), key=lambda t: probabilities[t]) == target
                stats["targets"] += 1
                stats["supported"] += known
                stats["unknown"] += not known
                stats["loss"] += loss
                stats["supported_loss"] += loss if known else 0
                stats["correct"] += known and correct
                targets.append(
                    {
                        "speech_id": w["row"]["speech_id"],
                        "offset": w["start"] + i + 1,
                        "loss": loss,
                        "supported": known,
                        "correct": correct,
                    }
                )
        aggregate = summarize(per_speech.values())
        aggregate["elapsed_seconds"] = time.perf_counter() - started
        return {"aggregate": aggregate, "per_speech": dict(per_speech), "target_records": targets}


def summarize(items):
    items = list(items)
    totals = {
        key: sum(row[key] for row in items)
        for key in ("targets", "supported", "unknown", "loss", "supported_loss", "correct")
    }
    totals.update(
        mapped_token_loss=totals["loss"] / max(1, totals["targets"]),
        supported_target_bpc=totals["supported_loss"] / max(1, totals["supported"]) / math.log(2),
        coverage=totals["supported"] / max(1, totals["targets"]),
        supported_accuracy=totals["correct"] / max(1, totals["supported"]),
    )
    if not totals["targets"]:
        totals["mapped_token_loss"] = totals["coverage"] = None
    if not totals["supported"]:
        totals["supported_target_bpc"] = totals["supported_accuracy"] = None
    return totals


def phrases(training):
    frequency = Counter()
    for row in training:
        tokens = words(row["text"])
        frequency.update(set(tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)))
    return {phrase for phrase, count in frequency.items() if count >= 5}


def slices(result, rows, training):
    frequency = Counter(word for row in training for word in words(row["text"]))
    recurring = phrases(training)
    names = {
        re.sub(
            r"^(?:Deputy|Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+", "", row["member_name"], flags=re.I
        ).strip()
        for row in rows + training
    }
    by_position = {(t["speech_id"], t["offset"]): t for t in result.pop("target_records")}
    summary = defaultdict(
        lambda: {"spans": 0, "supported_spans": 0, "correct_spans": 0, "loss": 0.0, "characters": 0}
    )
    for row in rows:
        sid, text = row["speech_id"], row["text"]
        spans = []
        tokens = list(re.finditer(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", text))
        for match in tokens:
            n = frequency[match.group().lower()]
            spans.append(
                (
                    "word_unseen" if n == 0 else "word_rare" if n <= 5 else "word_frequent",
                    match.start(),
                    match.end(),
                )
            )
        for name in sorted(names):
            if len(name) >= 4:
                for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I):
                    n = sum(len(re.findall(re.escape(name), r["text"], re.I)) for r in training)
                    spans.append(
                        (
                            "member_name_rare" if n <= 5 else "member_name_frequent",
                            match.start(),
                            match.end(),
                        )
                    )
        for i in range(len(tokens) - 4):
            phrase = tuple(t.group().lower() for t in tokens[i : i + 5])
            if phrase in recurring:
                spans.append(("recurring_phrase", tokens[i].start(), tokens[i + 4].end()))
        for label, start, end in spans:
            values = [by_position.get((sid, offset)) for offset in range(start, end)]
            item = summary[label]
            item["spans"] += 1
            if all(value and value["supported"] for value in values):
                item["supported_spans"] += 1
                item["correct_spans"] += all(value["correct"] for value in values)
                item["loss"] += sum(value["loss"] for value in values)
                item["characters"] += len(values)
    result["slices"] = dict(summary)
    result["slice_notes"] = (
        "Heuristic word/name matches; exact-span accuracy is teacher-forced. "
        "Overlapping phrase spans are not independent. "
        "Substantive-language labels require review."
    )


def matching(training, earlier, later):
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    vectorizer.fit([r["text"] for r in training])
    a, b = (
        vectorizer.transform([r["text"] for r in earlier]),
        vectorizer.transform([r["text"] for r in later]),
    )
    edges = []
    for i, first in enumerate(earlier):
        for j, second in enumerate(later):
            if (
                first["member_id"] == second["member_id"]
                and abs(len(first["text"]) - len(second["text"]))
                / max(len(first["text"]), len(second["text"]))
                <= 0.25
            ):
                similarity = float(a[i].multiply(b[j]).sum())
                edges.append((-similarity, first["speech_id"], second["speech_id"]))
    used_a, used_b, pairs = set(), set(), []
    for negative, first, second in sorted(edges):
        if first not in used_a and second not in used_b:
            pairs.append({"earlier": first, "later": second, "similarity": -negative})
            used_a.add(first)
            used_b.add(second)
    return {
        "pairs": pairs,
        "unmatched_earlier": len(earlier) - len(used_a),
        "unmatched_later": len(later) - len(used_b),
        "small_cohort": len(pairs) < 20,
        "method": (
            "Training-fitted TF-IDF, greedy maximum similarity, "
            "same speaker, at most 25% length difference"
        ),
    }


def perturb(rows, training, kind):
    recurring = phrases(training)
    names = {r["member_name"] for r in rows + training}
    output = []
    for row in rows:
        text = row["text"]
        positions = set()
        if kind == "names":
            for name in names:
                name = re.sub(r"^(?:Deputy|Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+", "", name, flags=re.I)
                if len(name) >= 4:
                    for match in re.finditer(re.escape(name), text, re.I):
                        positions.update(range(match.start(), match.end()))
        else:
            tokens = list(re.finditer(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", text))
            for i in range(len(tokens) - 4):
                if tuple(t.group().lower() for t in tokens[i : i + 5]) in recurring:
                    positions.update(range(tokens[i].start(), tokens[i + 4].end()))
        output.append({**row, "masked_positions": sorted(positions)})
    return output


def generations(model, tokenizer, rows, training, seed, memory=None, parameters=None, budget=None):
    rng = torch.Generator().manual_seed(seed + 20000)
    outputs = []
    train_grams = set()
    for row in training:
        tokens = words(row["text"])
        train_grams.update(tuple(tokens[i : i + 8]) for i in range(len(tokens) - 7))
    with evaluating(model):
        for row in sorted(rows, key=lambda r: r["speech_id"])[:6]:
            prefix = row["text"][:64]
            ids = tokenizer.encode(prefix)
            generated = []
            started = time.perf_counter()
            for _ in range(128):
                if budget:
                    budget.check()
                x = torch.tensor([ids[-model.block_size :]])
                logits, hidden = model.predict(x, torch.tensor([model.category(row)]))
                probs = logits[0, -1:].softmax(-1)
                if memory:
                    probs = memory.mix(hidden[0, -1:], probs, **parameters)
                probs[:, tokenizer.padding] = 0
                probs = probs.pow(1 / 0.8)
                next_id = int(torch.multinomial(probs, 1, generator=rng))
                ids.append(next_id)
                generated.append(tokenizer.label(next_id))
            elapsed = time.perf_counter() - started
            text = "".join(generated)
            tokens = words(text)
            grams = [tuple(tokens[i : i + 8]) for i in range(len(tokens) - 7)]
            copied = sum(gram in train_grams for gram in grams)
            tri = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
            outputs.append(
                {
                    "speech_id": row["speech_id"],
                    "prefix": prefix,
                    "continuation": text,
                    "generated_tokens": 128,
                    "tokens_per_second": 128 / max(elapsed, 1e-9),
                    "copied_word_8gram_rate": copied / len(grams) if grams else None,
                    "repeated_word_trigram_rate": 1 - len(set(tri)) / len(tri) if tri else None,
                    "review_label": "unreviewed continuation",
                }
            )
    return outputs


def evaluate(run: Path, seed: int, resume=False):
    config = read_json(run / "config.json")
    rows = records(run)
    subsets = {
        key: [r for r in rows if r["split"] == key]
        for key in ("train", "validation", "earlier", "later")
    }
    training = subsets["train"]
    folder = run / "evaluations"
    folder.mkdir(exist_ok=True)
    with Budget(run, f"evaluate/{seed}") as budget:
        cohort = matching(training, subsets["earlier"], subsets["later"])
        write_json(folder / "matching.json", {"binding": binding(run), **cohort})
        continuing = {r["member_id"] for r in subsets["earlier"]} & {
            r["member_id"] for r in subsets["later"]
        }
        for mode in ("none", "party", "role"):
            checkpoint = run / "models" / f"{mode}-{seed}" / "best.pt"
            destination = folder / f"{mode}-{seed}.json"
            if destination.exists():
                if not resume:
                    raise FileExistsError("Evaluation exists; use --resume")
                saved = read_json(destination)
                if saved["binding"] != binding(run) or saved["checkpoint_sha256"] != digest(
                    checkpoint
                ):
                    raise ValueError("Existing evaluation mismatch")
                continue
            model, tokenizer, state = load(run, checkpoint)
            output = {
                "schema_version": 2,
                "binding": binding(run),
                "seed": seed,
                "mode": mode,
                "checkpoint_sha256": digest(checkpoint),
                "training_step": state["step"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "splits": {},
            }
            for split in ("earlier", "later"):
                subset = subsets[split]
                result = score(
                    model, tokenizer, subset, config["block_size"], budget, collect_targets=True
                )
                slices(result, subset, training)
                result["continuing_speakers"] = summarize(
                    result["per_speech"][r["speech_id"]]
                    for r in subset
                    if r["member_id"] in continuing
                )
                result["mapped_role_cohort"] = summarize(
                    result["per_speech"][r["speech_id"]] for r in subset if r["role"] != "unmapped"
                )
                matched = {pair[split] for pair in cohort["pairs"]}
                result["matched"] = summarize(
                    result["per_speech"][r["speech_id"]]
                    for r in subset
                    if r["speech_id"] in matched
                )
                result["perturbations"] = {}
                for kind in ("names", "phrases"):
                    masked = perturb(subset, training, kind)
                    control = [
                        {**row, "excluded_targets": row["masked_positions"], "masked_positions": []}
                        for row in masked
                    ]
                    result["perturbations"][kind] = {
                        "masked_input": score(
                            model, tokenizer, masked, config["block_size"], budget
                        )["aggregate"],
                        "same_target_control": score(
                            model, tokenizer, control, config["block_size"], budget
                        )["aggregate"],
                        "masked_characters": sum(len(r["masked_positions"]) for r in masked),
                    }
                output["splits"][split] = result
            output["generations"] = generations(
                model, tokenizer, subsets["earlier"], training, seed, budget=budget
            )
            write_json(folder / f"{mode}-{seed}.json", output)
            print(f"Evaluated {mode}/{seed}", flush=True)
        base_path = run / "models" / f"none-{seed}" / "best.pt"
        model, tokenizer, _ = load(run, base_path)
        ngram_path = folder / "ngram.json"
        if not ngram_path.exists():
            baseline = WittenBell(training, tokenizer, config["block_size"])
            result = {
                "binding": binding(run),
                "method": "Witten–Bell, maximum order five",
                "splits": {},
            }
            for split in ("earlier", "later"):
                measured = baseline.evaluate(
                    subsets[split], tokenizer, config["block_size"], budget
                )
                slices(measured, subsets[split], training)
                result["splits"][split] = measured
            write_json(ngram_path, result)
            del baseline
        for policy in POLICIES:
            memory_path = run / "memory" / str(seed) / policy
            memory = Memory(run, base_path, tokenizer, memory_path)
            destination = folder / f"memory-{seed}-{policy}.json"
            if destination.exists():
                if not resume:
                    raise FileExistsError("Evaluation exists; use --resume")
                saved = read_json(destination)
                if saved["binding"] != binding(run) or saved["memory_manifest_sha256"] != digest(
                    memory_path / "manifest.json"
                ):
                    raise ValueError("Existing memory evaluation mismatch")
                continue
            tuned = tune(
                run,
                base_path,
                tokenizer,
                model,
                subsets["validation"],
                memory,
                config["block_size"],
                budget,
            )
            memory.seconds, memory.queries = 0.0, 0
            result = score(
                model,
                tokenizer,
                subsets["earlier"],
                config["block_size"],
                budget,
                memory,
                tuned["parameters"],
                collect_targets=True,
            )
            slices(result, subsets["earlier"], training)
            output = {
                "binding": binding(run),
                "seed": seed,
                "policy": policy,
                "checkpoint_sha256": digest(base_path),
                "memory_manifest_sha256": digest(memory_path / "manifest.json"),
                "tuning": tuned,
                "result": result,
                "retrieval": {
                    "seconds": memory.seconds,
                    "queries": memory.queries,
                    "ms_per_query": 1000 * memory.seconds / max(1, memory.queries),
                },
                "generations": generations(
                    model,
                    tokenizer,
                    subsets["earlier"],
                    training,
                    seed,
                    memory,
                    tuned["parameters"],
                    budget=budget,
                ),
            }
            write_json(folder / f"memory-{seed}-{policy}.json", output)
            print(f"Evaluated memory {policy}/{seed}", flush=True)
