"""Exact CPU speech memory with verifiable prediction-level interventions."""

from __future__ import annotations

import bisect
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .common import Budget, binding, digest, fingerprint, read_json, records, write_json
from .data import normalize
from .modeling import batch, evaluating, windows
from .training import load

POLICIES = ("uniform", "speech_balanced", "context_diverse")


def select(rows, count, seed, diverse=False):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    positions = {
        int(i): iter(rng.permutation(np.arange(1, len(rows[int(i)]["text"])))) for i in order
    }
    active, selected, contexts = list(order), [], Counter()
    while active and len(selected) < count:
        next_round = []
        for i in active:
            row = rows[int(i)]
            for offset in positions[int(i)]:
                offset = int(offset)
                key = normalize(row["text"][max(0, offset - 64) : offset]).casefold()
                if diverse and contexts[key] >= 4:
                    continue
                selected.append((row["speech_id"], offset))
                contexts[key] += 1
                next_round.append(i)
                break
            if len(selected) == count:
                break
        active = next_round
    return selected


def selections(rows, count, seed):
    lengths = np.cumsum([len(r["text"]) - 1 for r in rows]).tolist()
    count = min(count, lengths[-1])
    uniform = []
    for index in random.Random(seed).sample(range(lengths[-1]), count):
        i = bisect.bisect_right(lengths, index)
        uniform.append((rows[i]["speech_id"], index - (lengths[i - 1] if i else 0) + 1))
    result = {
        "uniform": uniform,
        "speech_balanced": select(rows, count, seed),
        "context_diverse": select(rows, count, seed, diverse=True),
    }
    achieved = min(map(len, result.values()))
    return {key: value[:achieved] for key, value in result.items()}


def build_memory(run: Path, seed: int, resume=False):
    config = read_json(run / "config.json")
    checkpoint = run / "models" / f"none-{seed}" / "best.pt"
    model, tokenizer, _ = load(run, checkpoint)
    rows = [r for r in records(run) if r["split"] == "train"]
    by_id = {r["speech_id"]: r for r in rows}
    folder = run / "memory" / str(seed)
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError("Memory already exists; use a new run or finish an incomplete stage")
    if resume:
        for existing in folder.glob("*/manifest.json"):
            Memory(run, checkpoint, tokenizer, existing.parent)
    folder.mkdir(parents=True, exist_ok=True)
    with Budget(run, f"memory/{seed}") as budget:
        choices = selections(rows, config["memory_entries"], seed)
        union = sorted(set().union(*map(set, choices.values())))
        union_ids = {pair: index for index, pair in enumerate(union)}
        vectors = np.lib.format.open_memmap(
            folder / "building.npy",
            mode="w+",
            dtype="float32",
            shape=(len(union), config["embed_dim"]),
        )
        items = windows(rows, tokenizer, config["block_size"])
        with evaluating(model):
            for start in range(0, len(items), 8):
                budget.check()
                group = items[start : start + 8]
                x, _, meta = batch(group, model)
                _, hidden = model.predict(x, meta)
                for w, representations in zip(group, hidden, strict=True):
                    for local in range(w["length"]):
                        pair = (w["row"]["speech_id"], w["start"] + local + 1)
                        if pair in union_ids:
                            vectors[union_ids[pair]] = representations[local].numpy()
        for policy, selected in choices.items():
            budget.check()
            target = folder / policy
            target.mkdir(exist_ok=resume)
            matrix = np.lib.format.open_memmap(
                target / "keys.npy",
                mode="w+",
                dtype="float32",
                shape=(len(selected), config["embed_dim"]),
            )
            entries = []
            for index, pair in enumerate(selected):
                sid, offset = pair
                row = by_id[sid]
                matrix[index] = vectors[union_ids[pair]]
                entries.append(
                    {
                        "speech_id": sid,
                        "date": row["date"],
                        "offset": offset,
                        "next_id": tokenizer.encode(row["text"][offset])[0],
                    }
                )
            matrix.flush()
            del matrix
            write_json(target / "entries.json", entries)
            write_json(
                target / "manifest.json",
                {
                    "schema_version": 2,
                    "binding": binding(run),
                    "checkpoint_sha256": digest(checkpoint),
                    "tokenizer_sha256": fingerprint(tokenizer.state()),
                    "policy": policy,
                    "entry_count": len(entries),
                    "requested_entries": config["memory_entries"],
                    "keys_sha256": digest(target / "keys.npy"),
                    "entries_sha256": digest(target / "entries.json"),
                    "extraction": {
                        "representation": "final_layer_norm",
                        "block_size": config["block_size"],
                        "seed": seed,
                        "context_limit": 64,
                        "context_cap": 4,
                    },
                },
            )
        del vectors
        (folder / "building.npy").unlink()


class Memory:
    def __init__(self, run, checkpoint, tokenizer, path, exclude=(), cutoff=None):
        manifest = read_json(path / "manifest.json")
        if (
            manifest["binding"] != binding(run)
            or manifest["checkpoint_sha256"] != digest(checkpoint)
            or manifest["tokenizer_sha256"] != fingerprint(tokenizer.state())
        ):
            raise ValueError("Memory/checkpoint/tokenizer/corpus mismatch")
        if (
            digest(path / "keys.npy") != manifest["keys_sha256"]
            or digest(path / "entries.json") != manifest["entries_sha256"]
        ):
            raise ValueError("Memory files do not match manifest")
        self.entries = read_json(path / "entries.json")
        source = {r["speech_id"]: r for r in records(run)}
        for entry in self.entries:
            row = source.get(entry["speech_id"])
            if (
                not row
                or row["split"] != "train"
                or row["date"] != entry["date"]
                or not 1 <= entry["offset"] < len(row["text"])
                or tokenizer.encode(row["text"][entry["offset"]])[0] != entry["next_id"]
            ):
                raise ValueError("Memory contains an ineligible source or target")
        self.keys = np.load(path / "keys.npy", mmap_mode="r")
        if len(self.keys) != len(self.entries):
            raise ValueError("Memory key and entry counts differ")
        self.allowed = np.array(
            [
                i
                for i, e in enumerate(self.entries)
                if e["speech_id"] not in exclude and (cutoff is None or e["date"] <= cutoff)
            ]
        )
        self.values = torch.tensor([e["next_id"] for e in self.entries])
        self.vocab_size = tokenizer.size
        self.seconds, self.queries = 0.0, 0

    def nearest(self, query, k):
        started = time.perf_counter()
        k = min(k, len(self.allowed))
        if not k:
            return torch.empty((len(query), 0)), torch.empty((len(query), 0), dtype=torch.long)
        distances = torch.full((len(query), k), float("inf"))
        indices = torch.zeros((len(query), k), dtype=torch.long)
        for start in range(0, len(self.allowed), 4096):
            ids = self.allowed[start : start + 4096]
            keys = torch.from_numpy(np.array(self.keys[ids], copy=True))
            d = (
                query.square().sum(-1)[:, None] + keys.square().sum(-1)[None] - 2 * query @ keys.T
            ).clamp_min(0)
            combined = torch.cat((distances, d), dim=1)
            positions = torch.argsort(combined, dim=1, stable=True)[:, :k]
            candidates = torch.cat((indices, torch.tensor(ids)[None].expand(len(query), -1)), dim=1)
            distances, indices = combined.gather(1, positions), candidates.gather(1, positions)
        self.seconds += time.perf_counter() - started
        self.queries += len(query)
        return distances, indices

    def mix(self, hidden, base, k=8, weight=0.25, temperature=10):
        if weight == 0 or not len(self.allowed):
            return base.clone()
        if not 0 <= weight <= 1 or temperature <= 0 or k < 1:
            raise ValueError("Invalid retrieval parameters")
        result = []
        for start in range(0, len(hidden), 32):
            d, ids = self.nearest(hidden[start : start + 32], k)
            distribution = torch.zeros((len(d), self.vocab_size))
            distribution.scatter_add_(1, self.values[ids], torch.softmax(-d / temperature, -1))
            result.append((1 - weight) * base[start : start + 32] + weight * distribution)
        return torch.cat(result)


def tune(run, checkpoint, tokenizer, model, validation, memory, block_size, budget):
    # Cache validation retrieval once; the grid only varies probability mixing.
    base_parts, target_parts, distance_parts, id_parts = [], [], [], []
    with evaluating(model):
        for w in windows(validation, tokenizer, block_size):
            budget.check()
            x, y, meta = batch([w], model)
            logits, hidden = model.predict(x, meta)
            for start in range(0, w["length"], 32):
                end = min(start + 32, w["length"])
                d, ids = memory.nearest(hidden[0, start:end], 32)
                base_parts.append(logits[0, start:end].softmax(-1))
                target_parts.append(y[0, start:end])
                distance_parts.append(d)
                id_parts.append(ids)
    base, targets = torch.cat(base_parts), torch.cat(target_parts)
    distances, indices = torch.cat(distance_parts), torch.cat(id_parts)
    trials = []
    for weight in (0, 0.25, 0.5):
        for k in (8, 32):
            for temperature in (1, 10, 100):
                budget.check()
                distribution = torch.zeros_like(base)
                if len(memory.allowed):
                    distribution.scatter_add_(
                        1,
                        memory.values[indices[:, :k]],
                        torch.softmax(-distances[:, :k] / temperature, -1),
                    )
                probabilities = (
                    base if not len(memory.allowed) else (1 - weight) * base + weight * distribution
                )
                loss = float(
                    -probabilities.gather(1, targets[:, None]).clamp_min(1e-30).log().mean()
                )
                trials.append(
                    {"k": k, "weight": weight, "temperature": temperature, "validation_loss": loss}
                )
    winner = min(
        trials,
        key=lambda t: (
            round(t["validation_loss"], 8),
            t["weight"] != 0,
            t["k"],
            t["weight"],
            t["temperature"],
        ),
    )
    return {"parameters": {k: winner[k] for k in ("k", "weight", "temperature")}, "trials": trials}


def inspect(run, seed, policy, prefix, exclude=()):
    checkpoint = run / "models" / f"none-{seed}" / "best.pt"
    model, tokenizer, _ = load(run, checkpoint)
    if not prefix or len(prefix) > model.block_size:
        raise ValueError(f"Prefix must contain 1–{model.block_size} characters")
    path = run / "memory" / str(seed) / policy
    results = read_json(run / "evaluations" / f"memory-{seed}-{policy}.json")
    if (
        results["checkpoint_sha256"] != digest(checkpoint)
        or results["binding"] != binding(run)
        or results["memory_manifest_sha256"] != digest(path / "manifest.json")
    ):
        raise ValueError("Inspection evaluation mismatch")
    parameters = results["tuning"]["parameters"]
    with Budget(run, f"inspect/{seed}/{policy}") as budget, evaluating(model):
        budget.check()
        x = torch.tensor([tokenizer.encode(prefix)])
        logits, hidden = model.predict(x)
        base = logits[0, -1:].softmax(-1)
        memory = Memory(run, checkpoint, tokenizer, path)
        modified = Memory(run, checkpoint, tokenizer, path, exclude=exclude)
        before = memory.mix(hidden[0, -1:], base, **parameters)
        after = modified.mix(hidden[0, -1:], base, **parameters)
        sources = {r["speech_id"]: r for r in records(run)}
        if any(sid not in sources for sid in exclude):
            raise ValueError("Excluded speech ID is not in this corpus")
        d, ids = memory.nearest(hidden[0, -1:], parameters["k"])
        contributions = torch.softmax(-d / parameters["temperature"], -1)[0]
        neighbours = []
        for index, contribution in zip(ids[0].tolist(), contributions.tolist(), strict=True):
            entry = memory.entries[index]
            row = sources[entry["speech_id"]]
            offset = entry["offset"]
            neighbours.append(
                {
                    **entry,
                    "memory_mass": contribution,
                    "mixture_mass": contribution * parameters["weight"],
                    "source_span": row["text"][max(0, offset - 64) : offset + 32],
                }
            )
        output = {
            "schema_version": 2,
            "binding": binding(run),
            "checkpoint_sha256": digest(checkpoint),
            "prefix": prefix,
            "excluded_speeches": list(exclude),
            "parameters": parameters,
            "neighbours_before": neighbours,
            "neighbours_after": [],
            "remaining_entries": len(modified.allowed),
            "distributions": [
                {
                    "character": tokenizer.label(i),
                    "base": float(base[0, i]),
                    "before": float(before[0, i]),
                    "after": float(after[0, i]),
                    "delta": float(after[0, i] - before[0, i]),
                }
                for i in range(tokenizer.size)
            ],
            "interpretation": (
                "Effect through retrieval for a fixed prefix; "
                "not training attribution or factual verification."
            ),
        }
        after_distances, after_ids = modified.nearest(hidden[0, -1:], parameters["k"])
        after_weights = torch.softmax(-after_distances / parameters["temperature"], -1)[0]
        for index, contribution in zip(after_ids[0].tolist(), after_weights.tolist(), strict=True):
            entry = modified.entries[index]
            row, offset = sources[entry["speech_id"]], entry["offset"]
            output["neighbours_after"].append(
                {
                    **entry,
                    "memory_mass": contribution,
                    "mixture_mass": contribution * parameters["weight"],
                    "source_span": row["text"][max(0, offset - 64) : offset + 32],
                }
            )
        write_json(
            run / "inspections" / f"{fingerprint([seed, policy, prefix, list(exclude)])}.json",
            output,
        )
    return output
