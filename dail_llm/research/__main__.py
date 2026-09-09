"""Offline research commands. The served application is never reconfigured."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import ROOT, BudgetExceeded, read_json
from .data import prepare
from .evaluation import evaluate
from .memory import POLICIES, build_memory, inspect
from .reporting import report
from .training import train


def main():
    parser = argparse.ArgumentParser(description="Dáil LLM offline research")
    parser.add_argument(
        "operation", choices=("prepare", "train", "build-memory", "evaluate", "inspect", "report")
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "experiments/pilot.toml")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "dataverse_files/Dail_debates_1919-2013.tab"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("none", "party", "role", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy", choices=POLICIES, default="context_diverse")
    parser.add_argument("--prefix")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    args.run = args.run.resolve()
    try:
        if args.operation == "prepare":
            prepare(args.run, args.profile, args.source)
            return
        config = read_json(args.run / "config.json")
        torch.set_num_threads(config.get("threads", 4))
        if args.seed is not None and args.seed not in config["seeds"]:
            raise ValueError("Seed is not part of this experiment profile")
        seeds = [args.seed] if args.seed is not None else config["seeds"]
        if args.operation == "train":
            for seed in seeds:
                for mode in ("none", "party", "role") if args.mode == "all" else (args.mode,):
                    exists = (args.run / "models" / f"{mode}-{seed}" / "latest.pt").exists()
                    train(args.run, mode, seed, resume=args.resume and exists)
        elif args.operation == "build-memory":
            for seed in seeds:
                build_memory(args.run, seed, resume=args.resume)
        elif args.operation == "evaluate":
            for seed in seeds:
                evaluate(args.run, seed, resume=args.resume)
        elif args.operation == "inspect":
            if args.seed is None or not args.prefix:
                raise ValueError("Inspection requires --seed and --prefix")
            print(
                json.dumps(
                    inspect(args.run, args.seed, args.policy, args.prefix, args.exclude),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            result = report(args.run)
            print(f"Report: {args.run / 'report.md'} ({result['status']})")
    except BudgetExceeded as error:
        parser.exit(2, f"Incomplete stage: {error}. Saved status and available checkpoints.\n")
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"Research command failed: {error}\n")


if __name__ == "__main__":
    main()
