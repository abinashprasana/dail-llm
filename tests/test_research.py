"""Scientific invariants and a small offline end-to-end experiment."""

import pytest
import torch

from dail_llm.eval.metrics import calculate_language_model_metrics
from dail_llm.model.train import estimate_loss
from dail_llm.research.common import BudgetExceeded, digest, initialize, read_json, write_json
from dail_llm.research.data import COLUMNS, choose_groups, deduplicate, prepare, role
from dail_llm.research.evaluation import WittenBell, evaluate, matching
from dail_llm.research.memory import Memory, build_memory, inspect, selections
from dail_llm.research.modeling import ResearchModel, ResearchTokenizer, windows
from dail_llm.research.reporting import bootstrap, report
from dail_llm.research.training import load, score, train


@pytest.fixture
def prepared(tmp_path):
    profile = tmp_path / "profile.toml"
    profile.write_text("""name = "test"
seeds = [42]
steps = 2
batch_size = 2
train_chars = 20000
eval_chars = 10000
memory_entries = 16
max_seconds = 0
memory_gib = 4
block_size = 16
embed_dim = 16
n_layers = 1
n_heads = 2
threads = 1
dropout = 0.1
learning_rate = 0.0003
""")
    source = tmp_path / "source.tab"
    lines = ["\t".join(COLUMNS)]
    for index, day in enumerate(
        ("2008-03-04", "2009-02-02", "2010-02-03", "2010-08-04", "2011-05-05")
    ):
        for member, party in (("1", "Fianna Fáil"), ("2", "Fine Gael"), ("3", "Labour Party")):
            speech = f'Deputy Áine says "tax and services" today. Stage {index} member {member}.'
            values = [
                str(index * 3 + int(member)),
                member,
                member,
                "1",
                f"Debate {index}",
                day,
                "Deputy Áine",
                party,
                "Dublin",
                speech,
            ]
            lines.append("\t".join(values))
    lines.append("bad\trow")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    prepare(run, profile, source)
    return run, profile, source


def test_archive_quotes_unicode_partitions(prepared):
    run, _, source = prepared
    from dail_llm.research.common import records

    rows = records(run)
    assert any('"tax and services"' in r["text"] for r in rows)
    assert all("Áine" in r["text"] for r in rows)
    assert read_json(run / "manifest.json")["source_sha256"] == digest(source)
    assert read_json(run / "manifest.json")["counts"]["malformed"] == 1
    assert len({r["speech_id"] for r in rows}) == len(rows)
    by_group = {}
    for row in rows:
        assert by_group.setdefault(row["debate_id"], row["split"]) == row["split"]


def test_no_nonempty_run_reuse(prepared):
    run, profile, _ = prepared
    with pytest.raises(FileExistsError):
        initialize(run, profile)


def test_group_selection_is_deterministic_and_capped():
    groups = [("a", "2008-01", 10), ("b", "2008-02", 11), ("c", "2008-03", 100)]
    assert choose_groups(groups, 21) == choose_groups(list(reversed(groups)), 21)
    assert set(choose_groups(groups, 21)) == {"a", "b"}


def test_duplicates_keep_earliest():
    text = " ".join(f"word{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(70))
    rows = [
        {"speech_id": "2", "date": "2010-01-01", "text": text + " extra"},
        {"speech_id": "1", "date": "2008-01-01", "text": text},
        {"speech_id": "3", "date": "2011-01-01", "text": text},
    ]
    kept, removed = deduplicate(rows)
    assert [r["speech_id"] for r in kept] == ["1"]
    assert {r["reason"] for r in removed} == {"exact", "near"}


def test_tokenizer_windows_unknown_and_tails():
    tokenizer = ResearchTokenizer(["a", "b"])
    row = {"text": "ab☘abab", "speech_id": "1"}
    assert len(tokenizer.encode(row["text"])) == len(row["text"])
    assert tokenizer.encode("☘") == [0]
    items = windows([row, {**row, "speech_id": "2", "text": "ab"}], tokenizer, 4)
    offsets = [
        w["start"] + i + 1
        for w in items
        if w["row"]["speech_id"] == "1"
        for i in range(w["length"])
    ]
    assert offsets == list(range(1, 7))
    assert items[-1]["length"] == 1
    assert items[-1]["y"][1:] == [-100] * 3


@pytest.mark.parametrize("initial", [True, False])
def test_evaluation_restores_mode_on_success_and_error(initial):
    model = ResearchModel(4, dict(block_size=4, embed_dim=8, n_layers=1, n_heads=2, dropout=0.1))
    model.train(initial)
    calculate_language_model_metrics(model, torch.tensor([0, 1] * 10), 4, "cpu")
    assert model.training == initial
    with pytest.raises((IndexError, RuntimeError)):
        calculate_language_model_metrics(model, torch.tensor([99] * 10), 4, "cpu")
    assert model.training == initial
    estimate_loss(model, torch.tensor([0, 1] * 10), torch.tensor([0, 1] * 10), 1, 4, "cpu")
    assert model.training == initial


def test_exact_cpu_resume(prepared, tmp_path):
    run, profile, source = prepared
    other = tmp_path / "other"
    prepare(other, profile, source)
    train(run, "none", 42)
    with pytest.raises(BudgetExceeded):
        train(other, "none", 42, stop_after=1)
    train(other, "none", 42, resume=True)
    a = torch.load(run / "models/none-42/latest.pt", weights_only=True)
    b = torch.load(other / "models/none-42/latest.pt", weights_only=True)
    assert a["history"] == b["history"]
    assert all(torch.equal(value, b["model_state"][key]) for key, value in a["model_state"].items())
    write_json(other / "config.json", {**read_json(other / "config.json"), "steps": 3})
    with pytest.raises(ValueError, match="mismatch"):
        train(other, "none", 42, resume=True)


def test_memory_selection_equal_and_diverse():
    rows = [{"speech_id": str(i), "text": "a" * 150} for i in range(3)]
    policies = selections(rows, 300, 42)
    assert len({len(v) for v in policies.values()}) == 1
    assert policies == selections(rows, 300, 42)
    from collections import Counter

    counts = Counter(
        rows[int(sid)]["text"][max(0, offset - 64) : offset]
        for sid, offset in policies["context_diverse"]
    )
    assert max(counts.values()) <= 4


@pytest.mark.parametrize(
    ("party", "day", "expected"),
    [
        ("Fianna Fáil", "2010-01-01", "government"),
        ("Fianna Fáil", "2011-04-01", "opposition"),
        ("Fine Gael", "2011-04-01", "government"),
        ("Labour Party", "2008-04-01", "opposition"),
        ("The Labour Party", "2011-04-01", "government"),
        ("Fine Gael", "2011-03-09", "unmapped"),
        ("Independent", "2011-04-01", "unmapped"),
    ],
)
def test_roles(party, day, expected):
    assert role(party, day) == expected


def test_matching_is_one_to_one():
    a = [
        {"speech_id": str(i), "member_id": "a", "text": "tax and public services"} for i in range(3)
    ]
    b = [{"speech_id": "later", "member_id": "a", "text": "tax and public services"}]
    matched = matching(a, a, b)
    assert len(matched["pairs"]) == 1
    assert matched["unmatched_earlier"] == 2


def test_baseline_normalized():
    tokenizer = ResearchTokenizer(["a", "b", "c"])
    baseline = WittenBell([{"text": "abcabc", "speech_id": "1"}], tokenizer, 4)
    assert sum(baseline.probability([2, 3], t) for t in range(tokenizer.size)) == pytest.approx(1)


def test_end_to_end_and_source_exclusion(prepared, monkeypatch):
    run, _, _ = prepared
    for mode in ("none", "party", "role"):
        train(run, mode, 42)
    build_memory(run, 42)
    checkpoint = run / "models/none-42/best.pt"
    model, tokenizer, _ = load(run, checkpoint)
    path = run / "memory/42/uniform"
    memory = Memory(run, checkpoint, tokenizer, path)
    hidden = torch.zeros((1, 16))
    base = torch.ones((1, tokenizer.size)) / tokenizer.size
    assert torch.equal(memory.mix(hidden, base, weight=0), base)
    assert memory.mix(hidden, base).sum().item() == pytest.approx(1)
    _, before = memory.nearest(hidden, 32)
    removed_id = memory.entries[int(before[0, 0])]["speech_id"]
    excluded = Memory(run, checkpoint, tokenizer, path, exclude=[removed_id])
    _, after = excluded.nearest(hidden, 32)
    assert all(excluded.entries[i]["speech_id"] != removed_id for i in after[0].tolist())
    empty = Memory(run, checkpoint, tokenizer, path, cutoff="1900-01-01")
    assert torch.equal(empty.mix(hidden, base), base)
    evaluate(run, 42)
    output = inspect(run, 42, "uniform", "Deputy", exclude=[removed_id])
    assert output["remaining_entries"] < len(memory.entries)
    result = report(run)
    assert result["status"] == "complete"
    assert all(item["readability"] is None for item in read_json(run / "review-pack.json"))
    from dail_llm.research.verify import verify

    assert verify(run)["status"] == "passed"
    assert report(run)["verification"]["status"] == "passed"
    from dail_llm.research.publication import ResearchReader, export, private_bundle, public_summary

    reader = ResearchReader(run)
    expected_outputs = {}
    for policy in ("uniform", "speech_balanced", "context_diverse"):
        sid = reader.memories[policy].entries[0]["speech_id"]
        expected_outputs[policy] = (sid, inspect(run, 42, policy, "Deputy 🐈", [sid]))
    original_files = {p: (digest(p), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()}
    for policy, (sid, expected) in expected_outputs.items():
        actual = reader.inspect("Deputy 🐈", policy, sid)
        assert actual["unknown_characters"] == 1
        assert actual["distributions"] == expected["distributions"]
        assert actual["remaining_entries"] == expected["remaining_entries"]
        assert all(e["speech_id"] != sid for e in actual["neighbours_after"])
        assert [e["speech_id"] for e in actual["neighbours_after"]] == [
            e["speech_id"] for e in expected["neighbours_after"]
        ]
        assert reader.memories[policy].queries == 0
        for key in ("base", "before", "after"):
            assert sum(d[key] for d in actual["distributions"]) == pytest.approx(1)
    with pytest.raises(ValueError):
        reader.inspect("Deputy", "uniform", "missing-speech")
    release = run.parent / "public-release"
    exported = export(run, release)
    assert (
        exported["history"]["none"]["splits"]["earlier"]["aggregate"]
        == public_summary(reader)["history"]["none"]["splits"]["earlier"]["aggregate"]
    )
    assert len(exported["examples"]) <= 8
    for example in exported["examples"]:
        for file in example["files"].values():
            assert digest(release / file["name"]) == file["sha256"]
            assert (release / file["name"]).stat().st_size < 250_000
    with pytest.raises(FileExistsError):
        export(run, release)
    private = run.parent / "private-runtime"
    assert private_bundle(run, private)["release_id"] == reader.release_id
    assert not (private / "review-answer-key.json").exists()
    assert not (private / "models/none-42/latest.pt").exists()
    assert ResearchReader(private).inspect("Deputy", "uniform")["release_id"] == reader.release_id
    from fastapi.testclient import TestClient

    from dail_llm.api.app import create_app

    monkeypatch.setenv("DAIL_RESEARCH_RUN", str(private))
    monkeypatch.setenv("DAIL_RESEARCH_PUBLIC", str(release))
    with TestClient(create_app(False)) as client:
        assert client.get("/api/v1/research/capabilities").json()["live"] is True
        response = client.post(
            "/api/v1/research/inspect",
            json={"release_id": reader.release_id, "prefix": "Deputy", "policy": "uniform"},
        )
        assert response.status_code == 200
        assert (
            response.json()["distributions"] == reader.inspect("Deputy", "uniform")["distributions"]
        )
    assert original_files == {
        p: (digest(p), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()
    }
    import numpy as np

    reader.memories["uniform"].allowed = np.array([], dtype=np.int64)
    empty_result = reader.inspect("Deputy", "uniform")
    assert empty_result["neighbours_after"] == []
    assert all(d["after"] == d["base"] for d in empty_result["distributions"])
    entries = read_json(path / "entries.json")
    entries[0]["date"] = "2011-01-01"
    write_json(path / "entries.json", entries)
    manifest = read_json(path / "manifest.json")
    manifest["entries_sha256"] = digest(path / "entries.json")
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="ineligible"):
        Memory(run, checkpoint, tokenizer, path)
    with pytest.raises(ValueError, match="Memory changed"):
        report(run)
    with pytest.raises(ValueError, match="memory mismatch"):
        ResearchReader(run)


def test_bootstrap_uses_debate_groups():
    values = {"1": {"supported_loss": 1, "supported": 2}}
    assert bootstrap(values, values, [{"speech_id": "1", "debate_id": "d"}])["interval"] is None


def test_review_pack_preserves_available_strata():
    from dail_llm.research.reporting import review_candidates

    rows = [
        {
            "speech_id": str(i),
            "split": "train",
            "member_name": "Deputy Jane Smith",
            "text": "We must consider this matter carefully.",
        }
        for i in range(5)
    ]
    rows.append(
        {
            "speech_id": "test",
            "split": "earlier",
            "member_name": "Jane Smith",
            "text": "Jane Smith says we must consider this matter carefully.",
        }
    )
    docs = {
        "hidden-system": {
            "generations": [{"speech_id": "test", "prefix": "We", "continuation": " must consider"}]
        }
    }
    candidates = review_candidates(rows, docs)
    assert candidates == review_candidates(rows, docs)
    assert {sample["heuristic_stratum"] for _, sample in candidates} == {
        "ordinary continuation",
        "member-name span",
        "frequent phrase",
    }


def test_interrupted_preparation_is_reportable(tmp_path):
    write_json(tmp_path / "config.json", {"name": "cpu-pilot"})
    write_json(tmp_path / "status.json", {"stages": {"prepare": {"status": "incomplete"}}})
    assert report(tmp_path)["status"] == "incomplete"


def test_score_matches_manual_likelihood_and_coverage():
    tokenizer = ResearchTokenizer(["a", "b"])
    model = ResearchModel(4, dict(block_size=4, embed_dim=8, n_layers=1, n_heads=2, dropout=0))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = score(model, tokenizer, [{"text": "ab☘a", "speech_id": "1"}], 4)
    aggregate = result["aggregate"]
    assert aggregate["targets"] == 3
    assert aggregate["unknown"] == 1
    assert aggregate["coverage"] == pytest.approx(2 / 3)
    assert aggregate["supported_target_bpc"] == pytest.approx(2)
    unknown = score(model, tokenizer, [{"text": "a☘☘", "speech_id": "1"}], 4)["aggregate"]
    assert unknown["supported_target_bpc"] is None
    assert unknown["coverage"] == 0


def test_masked_input_and_control_use_identical_targets():
    tokenizer = ResearchTokenizer(["a", "b"])
    row = {"text": "abababa", "speech_id": "1", "masked_positions": [1, 2, 5]}
    masked = windows([row], tokenizer, 4)
    control = windows(
        [{**row, "excluded_targets": row["masked_positions"], "masked_positions": []}], tokenizer, 4
    )
    assert [w["y"] for w in masked] == [w["y"] for w in control]
    assert [w["x"] for w in masked] != [w["x"] for w in control]
    assert sum(t != -100 for w in masked for t in w["y"]) == 3


def test_exact_retrieval_matches_brute_force_and_recomputes():
    import numpy as np

    memory = Memory.__new__(Memory)
    rng = np.random.default_rng(17)
    memory.keys = rng.normal(size=(4200, 8)).astype("float32")
    memory.allowed = np.arange(4200)
    memory.seconds, memory.queries = 0, 0
    queries = torch.tensor(rng.normal(size=(3, 8)).astype("float32"))
    distances, indices = memory.nearest(queries, 8)
    brute = (queries[:, None, :] - torch.tensor(memory.keys)[None, :, :]).square().sum(-1)
    expected = torch.argsort(brute, dim=-1, stable=True)[:, :8]
    assert torch.equal(indices, expected)
    assert torch.allclose(distances, brute.gather(1, expected), atol=1e-5)
    removed = set(indices[0].tolist())
    memory.allowed = np.array([i for i in range(4200) if i not in removed])
    _, replacement = memory.nearest(queries, 8)
    assert not removed.intersection(replacement.flatten().tolist())
    assert replacement.shape == (3, 8)
