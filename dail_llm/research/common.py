"""Artifact integrity, atomic writes, and cumulative execution budgets."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 2


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def save_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def peak_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError("Cannot measure the process memory budget")
        return int(counters.PeakWorkingSetSize)
    import resource

    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount if platform.system() == "Darwin" else amount * 1024)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, run: Path, stage: str):
        self.run, self.stage = run, stage
        self.config = read_json(run / "config.json")
        self.path = run / "status.json"
        self.state = (
            read_json(self.path)
            if self.path.exists()
            else {"elapsed_seconds": 0.0, "stages": {}, "peak_memory_bytes": 0}
        )
        self.started = time.monotonic()

    def check(self):
        peak = peak_bytes()
        self.state["peak_memory_bytes"] = max(self.state["peak_memory_bytes"], peak)
        if peak > self.config["memory_gib"] * 1024**3:
            raise BudgetExceeded("Process memory budget reached")
        seconds = self.config["max_seconds"]
        if seconds and self.state["elapsed_seconds"] + time.monotonic() - self.started >= seconds:
            raise BudgetExceeded("Aggregate execution budget reached")

    def __enter__(self):
        self.lock = self.run / ".execution.lock"
        try:
            with self.lock.open("x", encoding="ascii") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError as error:
            raise RuntimeError("Another operation owns this run's execution lock") from error
        if self.path.exists():
            self.state = read_json(self.path)
        self.state["stages"][self.stage] = {"status": "running"}
        try:
            write_json(self.path, self.state)
        except Exception:
            self.lock.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, kind, error, traceback):
        self.state["elapsed_seconds"] += time.monotonic() - self.started
        self.state["stages"][self.stage] = {
            "status": "complete" if error is None else "incomplete",
            "reason": None if error is None else str(error),
        }
        try:
            write_json(self.path, self.state)
        finally:
            self.lock.unlink(missing_ok=True)


def initialize(run: Path, profile: Path) -> dict:
    run = run.resolve()
    protected = [
        ROOT / p
        for p in (
            "frontend",
            "data",
            "dataverse_files",
            "dail_llm",
            "outputs/checkpoints",
            "outputs/plots",
        )
    ]
    if (
        run == ROOT
        or run == ROOT / "outputs"
        or any(run == p or run.is_relative_to(p) for p in protected)
    ):
        raise ValueError("Choose an isolated research run directory")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Run directory is not empty; choose a new run")
    config = tomllib.loads(profile.read_text(encoding="utf-8"))
    required = (
        "seeds",
        "steps",
        "batch_size",
        "train_chars",
        "eval_chars",
        "memory_entries",
        "max_seconds",
        "memory_gib",
        "block_size",
        "embed_dim",
        "n_layers",
        "n_heads",
    )
    for key in required:
        if key not in config:
            raise ValueError(f"Missing profile setting: {key}")
    if any(config[k] <= 0 for k in required if k not in ("seeds", "max_seconds")):
        raise ValueError("Profile sizes must be positive")
    if not config["seeds"] or config["max_seconds"] < 0:
        raise ValueError("Invalid seeds or execution budget")
    if config["embed_dim"] % config["n_heads"]:
        raise ValueError("Embedding dimension must divide into attention heads")
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", config)
    write_json(run / "environment.json", versions())
    return config


def records(run: Path) -> list[dict]:
    manifest = read_json(run / "manifest.json")
    if digest(run / "speeches.jsonl") != manifest["records_sha256"]:
        raise ValueError("Speech artifact does not match its manifest")
    return [
        json.loads(line)
        for line in (run / "speeches.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def binding(run: Path) -> dict:
    manifest = read_json(run / "manifest.json")
    return {
        "config": fingerprint(read_json(run / "config.json")),
        "corpus": manifest["records_sha256"],
        "manifest": digest(run / "manifest.json"),
        "implementation": fingerprint(
            {
                name: digest(Path(__file__).with_name(name))
                for name in ("data.py", "modeling.py", "training.py", "memory.py", "evaluation.py")
            }
        ),
    }
