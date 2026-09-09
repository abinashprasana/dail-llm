"""Seeded CPU training with exact same-environment continuation."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import Budget, BudgetExceeded, binding, read_json, records, save_torch, versions
from .modeling import ResearchModel, ResearchTokenizer, batch, evaluating, windows


def architecture(config):
    return {
        key: config[key] for key in ("block_size", "embed_dim", "n_layers", "n_heads", "dropout")
    }


def load(run: Path, checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["binding"] != binding(run):
        raise ValueError("Checkpoint configuration or corpus mismatch")
    tokenizer = ResearchTokenizer(state["tokenizer"]["characters"])
    model = ResearchModel(tokenizer.size, state["architecture"], state["mode"], state["categories"])
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, tokenizer, state


def score(
    model,
    tokenizer,
    rows,
    block_size,
    budget=None,
    memory=None,
    parameters=None,
    collect_targets=False,
):
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
    target_records = []
    items = windows(rows, tokenizer, block_size)
    started = time.perf_counter()
    with evaluating(model):
        for start in range(0, len(items), 8):
            if budget:
                budget.check()
            group = items[start : start + 8]
            x, y, meta = batch(group, model)
            logits, hidden = model.predict(x, meta)
            probabilities = torch.softmax(logits, -1)
            if memory:
                mask = y != -100
                mixed = memory.mix(hidden[mask], probabilities[mask], **parameters)
                probabilities[mask] = mixed
            for w, probs in zip(group, probabilities, strict=True):
                sid = w["row"]["speech_id"]
                stats = per_speech[sid]
                targets = torch.tensor(w["y"][: w["length"]])
                active = targets != -100
                offsets = active.nonzero().flatten().tolist()
                valid_probs = probs[: w["length"]][active]
                targets = targets[active]
                losses = -valid_probs.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-30).log()
                known = targets != tokenizer.unknown
                stats["targets"] += len(targets)
                stats["supported"] += int(known.sum())
                stats["unknown"] += int((~known).sum())
                stats["loss"] += float(losses.sum())
                stats["supported_loss"] += float(losses[known].sum())
                stats["correct"] += int(((valid_probs.argmax(-1) == targets) & known).sum())
                if collect_targets:
                    predicted = valid_probs.argmax(-1).tolist()
                    for i, loss in enumerate(losses.tolist()):
                        target_records.append(
                            {
                                "speech_id": sid,
                                "offset": w["start"] + offsets[i] + 1,
                                "loss": loss,
                                "supported": bool(known[i]),
                                "correct": predicted[i] == int(targets[i]),
                            }
                        )
    totals = {
        key: sum(s[key] for s in per_speech.values())
        for key in ("targets", "supported", "unknown", "loss", "supported_loss", "correct")
    }
    totals.update(
        mapped_token_loss=totals["loss"] / max(1, totals["targets"]),
        supported_target_bpc=totals["supported_loss"] / max(1, totals["supported"]) / math.log(2),
        coverage=totals["supported"] / max(1, totals["targets"]),
        supported_accuracy=totals["correct"] / max(1, totals["supported"]),
        elapsed_seconds=time.perf_counter() - started,
    )
    if not totals["targets"]:
        totals["mapped_token_loss"] = totals["coverage"] = None
    if not totals["supported"]:
        totals["supported_target_bpc"] = totals["supported_accuracy"] = None
    return {"aggregate": totals, "per_speech": dict(per_speech), "target_records": target_records}


def train(run: Path, mode: str, seed: int, resume=False, stop_after=None):
    config = read_json(run / "config.json")
    torch.set_num_threads(config.get("threads", 4))
    torch.use_deterministic_algorithms(True)
    rows = records(run)
    tokenizer = ResearchTokenizer.fit(rows)
    categories = sorted({r[mode] for r in rows if r["split"] == "train"}) if mode != "none" else []
    folder = run / "models" / f"{mode}-{seed}"
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError("Model directory exists; use --resume")
    folder.mkdir(parents=True, exist_ok=True)
    current_binding = binding(run)
    with Budget(run, f"train/{mode}/{seed}") as budget:
        torch.manual_seed(seed)
        model = ResearchModel(tokenizer.size, architecture(config), mode, categories)
        torch.manual_seed(seed + 30000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        sampler = torch.Generator().manual_seed(seed + 10000)
        step, best, history = 0, float("inf"), []
        if resume:
            state = torch.load(folder / "latest.pt", weights_only=True, map_location="cpu")
            if (
                state["binding"] != current_binding
                or state["mode"] != mode
                or state["seed"] != seed
            ):
                raise ValueError("Resume configuration mismatch")
            current_versions = {k: str(v) for k, v in versions().items()}
            if state["software"] != current_versions:
                raise ValueError("Resume software environment mismatch")
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            sampler.set_state(state["sampler_rng"])
            step, best, history = state["step"], state["best"], state["history"]
        training = windows(
            [r for r in rows if r["split"] == "train"], tokenizer, config["block_size"]
        )
        validation = [r for r in rows if r["split"] == "validation"]

        def snapshot(path):
            save_torch(
                path,
                {
                    "schema_version": 2,
                    "binding": current_binding,
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "tokenizer": tokenizer.state(),
                    "architecture": architecture(config),
                    "mode": mode,
                    "categories": categories,
                    "seed": seed,
                    "step": step,
                    "best": best,
                    "history": history,
                    "torch_rng": torch.get_rng_state(),
                    "sampler_rng": sampler.get_state(),
                    "software": {k: str(v) for k, v in versions().items()},
                },
            )

        try:
            model.train()
            for index in range(step, config["steps"]):
                budget.check()
                chosen = torch.randint(len(training), (config["batch_size"],), generator=sampler)
                x, y, meta = batch([training[int(i)] for i in chosen], model)
                logits, _ = model.predict(x, meta)
                loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer.size), y.flatten(), ignore_index=-100
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step = index + 1
                if step % 25 == 0 or step == config["steps"]:
                    measured = score(model, tokenizer, validation, config["block_size"], budget)
                    val = measured["aggregate"]["mapped_token_loss"]
                    history.append(
                        {"step": step, "train_loss": float(loss.detach()), "validation_loss": val}
                    )
                    if val < best:
                        best = val
                        snapshot(folder / "best.pt")
                    print(f"{mode}/{seed} step {step}: validation loss {val:.4f}", flush=True)
                if step % 10 == 0:
                    snapshot(folder / "latest.pt")
                if stop_after and step >= stop_after and step < config["steps"]:
                    raise BudgetExceeded("Requested interruption for resumability check")
            snapshot(folder / "latest.pt")
        except BudgetExceeded:
            snapshot(folder / "latest.pt")
            raise
    return folder / "best.pt"
