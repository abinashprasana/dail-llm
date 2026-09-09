"""Audit the actual public release against its private source, without changing it."""

import argparse
from pathlib import Path

import torch

from dail_llm.research.common import digest, read_json
from dail_llm.research.publication import ResearchReader, public_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    original = {
        str(p): (digest(p), p.stat().st_mtime_ns) for p in args.run.rglob("*") if p.is_file()
    }
    reader = ResearchReader(args.run)
    summary = read_json(args.public / "summary.json")
    assert {k: v for k, v in summary.items() if k != "examples"} == public_summary(reader)
    inspections, removals = 0, 0
    for example in summary["examples"]:
        for policy, file in example["files"].items():
            assert digest(args.public / file["name"]) == file["sha256"]
            recording = read_json(args.public / file["name"])
            assert recording["original"] == reader.inspect(example["prefix"], policy)
            inspections += 1
            for sid, removal in recording["exclusions"].items():
                actual = reader.inspect(example["prefix"], policy, sid)
                assert removal["after"] == [d["after"] for d in actual["distributions"]]
                assert removal["remaining_entries"] == actual["remaining_entries"]
                assert removal["removed_entries"] == actual["removed_entries"]
                expected = [
                    [f"{n['speech_id']}:{n['offset']}", n["memory_mass"]]
                    for n in actual["neighbours_after"]
                ]
                assert removal["neighbours"] == expected
                assert all(n["speech_id"] != sid for n in actual["neighbours_after"])
                for entry in actual["neighbours_after"]:
                    key = f"{entry['speech_id']}:{entry['offset']}"
                    assert recording["source_entries"][key] == {
                        k: v for k, v in entry.items() if k not in ("memory_mass", "mixture_mass")
                    }
                removals += 1
    after = {str(p): (digest(p), p.stat().st_mtime_ns) for p in args.run.rglob("*") if p.is_file()}
    assert original == after, "Source run was modified"
    print(
        f"PASS: summary, {inspections} inspections, {removals} source removals; "
        "source run unchanged"
    )


if __name__ == "__main__":
    main()
