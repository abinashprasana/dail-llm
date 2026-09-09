"""Recheck completed run artifacts against independent numerical calculations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .common import Budget, binding, digest, fingerprint, read_json, records, write_json
from .data import partition
from .memory import POLICIES, Memory
from .modeling import ResearchTokenizer, batch, evaluating, windows
from .training import load


def verify(run: Path):
    config = read_json(run / "config.json")
    torch.set_num_threads(config.get("threads", 4))
    checks = {}

    def require(name, condition):
        if not condition:
            raise ValueError(f"Run verification failed: {name}")
        checks[name] = True

    with Budget(run, "verify") as budget:
        rows = records(run)
        require("unique_speech_ids", len({r["speech_id"] for r in rows}) == len(rows))
        require("fixed_date_partitions", all(partition(r["date"]) == r["split"] for r in rows))
        groups = {}
        for row in rows:
            require(
                "disjoint_debate_groups",
                groups.setdefault(row["debate_id"], row["split"]) == row["split"],
            )
        require(
            "no_retained_exact_duplicates", len({fingerprint(r["text"]) for r in rows}) == len(rows)
        )
        tokenizer = ResearchTokenizer.fit(rows)
        expected_characters = sorted({c for r in rows if r["split"] == "train" for c in r["text"]})
        require("training_only_vocabulary", tokenizer.characters == expected_characters)
        for row in rows:
            items = windows([row], tokenizer, config["block_size"])
            offsets = [w["start"] + i + 1 for w in items for i in range(w["length"])]
            require(
                "all_targets_once_with_speech_resets", offsets == list(range(1, len(row["text"])))
            )
        numerics, checkpoints, memories = {}, {}, {}
        for seed in config["seeds"]:
            for mode in ("none", "party", "role"):
                budget.check()
                checkpoint = run / "models" / f"{mode}-{seed}" / "best.pt"
                model, saved_tokenizer, state = load(run, checkpoint)
                require(f"tokenizer/{mode}/{seed}", saved_tokenizer.state() == tokenizer.state())
                latest = torch.load(
                    checkpoint.with_name("latest.pt"), weights_only=True, map_location="cpu"
                )
                require(f"exposure/{mode}/{seed}", latest["step"] == config["steps"])
                require(
                    f"checkpoint_selection/{mode}/{seed}",
                    state["best"] == min(h["validation_loss"] for h in latest["history"]),
                )
                doc = read_json(run / "evaluations" / f"{mode}-{seed}.json")
                require(
                    f"evaluation_checkpoint/{mode}/{seed}",
                    doc["checkpoint_sha256"] == digest(checkpoint),
                )
                for split in ("earlier", "later"):
                    subset = sorted(
                        [r for r in rows if r["split"] == split], key=lambda r: r["speech_id"]
                    )[:3]
                    worst = 0.0
                    with evaluating(model):
                        for row in subset:
                            loss, supported_loss, count, supported = 0.0, 0.0, 0, 0
                            for item in windows([row], tokenizer, config["block_size"]):
                                budget.check()
                                x, y, meta = batch([item], model)
                                logits, _ = model.predict(x, meta)
                                log_probs = logits[0].double().log_softmax(-1)
                                for i in range(item["length"]):
                                    target = int(y[0, i])
                                    value = -float(log_probs[i, target])
                                    loss += value
                                    count += 1
                                    if target != tokenizer.unknown:
                                        supported += 1
                                        supported_loss += value
                            saved = doc["splits"][split]["per_speech"][row["speech_id"]]
                            require(
                                f"target_counts/{mode}/{seed}/{split}",
                                saved["targets"] == count and saved["supported"] == supported,
                            )
                            delta = max(
                                abs(loss - saved["loss"]),
                                abs(supported_loss - saved["supported_loss"]),
                            ) / max(1, count)
                            require(f"independent_float64_loss/{mode}/{seed}/{split}", delta < 1e-5)
                            worst = max(worst, delta)
                    numerics[f"{mode}/{seed}/{split}"] = {
                        "sampled_speeches": len(subset),
                        "max_loss_difference_per_target": worst,
                    }
                checkpoints[f"{mode}/{seed}"] = digest(checkpoint)
            base_path = run / "models" / f"none-{seed}" / "best.pt"
            base_model, _, _ = load(run, base_path)
            source_rows = {r["speech_id"]: r for r in rows}
            sizes = []
            for policy in POLICIES:
                budget.check()
                path = run / "memory" / str(seed) / policy
                memory = Memory(run, base_path, tokenizer, path)
                sizes.append(len(memory.entries))
                sample_ids = sorted({0, len(memory.entries) // 2, len(memory.entries) - 1})
                with evaluating(base_model):
                    for index in sample_ids:
                        entry = memory.entries[index]
                        source_windows = windows(
                            [source_rows[entry["speech_id"]]], tokenizer, config["block_size"]
                        )
                        owner = source_windows[(entry["offset"] - 1) // config["block_size"]]
                        x, _, metadata = batch([owner], base_model)
                        _, hidden = base_model.predict(x, metadata)
                        expected_vector = hidden[0, entry["offset"] - owner["start"] - 1]
                        require(
                            f"extracted_representation/{seed}/{policy}/{index}",
                            torch.allclose(
                                expected_vector,
                                torch.tensor(np.array(memory.keys[index], copy=True)),
                                atol=1e-4,
                                rtol=1e-5,
                            ),
                        )
                query = torch.tensor(np.array(memory.keys[sample_ids], copy=True))
                distances, indices = memory.nearest(query, 8)
                keys = torch.tensor(np.array(memory.keys, copy=True))
                brute = (query[:, None] - keys[None]).square().sum(-1)
                # Duplicate vectors may tie; compare distance sets rather than arbitrary IDs.
                require(
                    f"brute_force_distances/{seed}/{policy}",
                    torch.allclose(
                        distances,
                        brute.sort(-1).values[:, : distances.shape[1]],
                        atol=1e-3,
                        rtol=1e-5,
                    ),
                )
                require(
                    f"retrieved_index_distances/{seed}/{policy}",
                    torch.allclose(distances, brute.gather(1, indices), atol=1e-3, rtol=1e-5),
                )
                sid = memory.entries[int(indices[0, 0])]["speech_id"]
                excluded = Memory(run, base_path, tokenizer, path, exclude=[sid])
                _, replacement = excluded.nearest(query, 8)
                require(
                    f"complete_source_exclusion/{seed}/{policy}",
                    all(
                        excluded.entries[i]["speech_id"] != sid
                        for i in replacement.flatten().tolist()
                    ),
                )
                memories[f"{seed}/{policy}"] = digest(path / "manifest.json")
            require(f"equal_memory_budgets/{seed}", len(set(sizes)) == 1)
        result = {
            "schema_version": 1,
            "status": "passed",
            "binding": binding(run),
            "checks": checks,
            "numerical_checks": numerics,
            "checkpoint_hashes": checkpoints,
            "memory_manifest_hashes": memories,
            "scope": (
                "All corpus target positions and memory eligibility; sampled float64 loss "
                "and brute-force search checks. Not human evaluation or "
                "membership-history verification."
            ),
        }
        require(
            "finite_numerical_errors",
            all(math.isfinite(v["max_loss_difference_per_target"]) for v in numerics.values()),
        )
        write_json(run / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify(arguments.run)
    print(f"Verified {len(result['checks'])} invariants; details in verification.json")
