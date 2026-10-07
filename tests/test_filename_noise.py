import hashlib
import json
import time
from dataclasses import replace
from pathlib import PurePosixPath

import pytest

from fsbench.config import FSConfig
from fsbench.docs import ALL_FORMATS, Doc
from fsbench.evaluate import balance_warnings, evaluate_runs, keep_paired
from fsbench.generate import generate_env
from fsbench.layout import Entry, place_entries
from fsbench.plot import _x_value, plot_runs
from fsbench.render import render
from fsbench.runner import OracleAgent, run_agent
from fsbench.sweep import run_sweep
from fsbench.tools import FileTools, Tracer
from test_evaluate import ScriptedAgent


def indexed(manifest):
    return {(f["doc_id"], f["copy"]): f for f in manifest["files"]}


@pytest.mark.parametrize("task", ["retrieve", "reconcile", "conflict", "workflow"])
def test_noise_only_changes_filenames(make_env, task):
    base = FSConfig(distractors=30, stale_version_rate=1, duplicate_rate=1,
                    depth=0, max_entries_per_dir=7, mime_types=ALL_FORMATS)
    clean_env, clean = make_env(task, base)
    previous = set()
    for level in (0, .3, .6, .9, 1):
        env, m = make_env(task, replace(base, filename_noise=level))
        assert m["ground_truth"] == clean["ground_truth"]
        assert m["task"] == clean["task"]
        assert m["required_doc_ids"] == clean["required_doc_ids"]
        assert m["trap_doc_ids"] == clean["trap_doc_ids"]
        assert m["decoy_answers"] == clean["decoy_answers"]
        assert m["oracle"] == clean["oracle"]
        assert (env / "task.json").read_bytes() == (clean_env / "task.json").read_bytes()
        original, current = indexed(clean), indexed(m)
        assert original.keys() == current.keys()
        renamed = set()
        paths = [f["path"].casefold() for f in m["files"]]
        assert len(paths) == len(set(paths))
        for key, f in current.items():
            old = original[key]
            assert PurePosixPath(f["path"]).parent == PurePosixPath(old["path"]).parent
            assert PurePosixPath(f["path"]).suffix == "." + old["format"]
            assert f["role"] == old["role"]
            data = (env / "workspace" / f["path"]).read_bytes()
            assert data == (clean_env / "workspace" / old["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == f["sha256"]
            if f["path"] != old["path"]:
                renamed.add(key)
            assert f["noisy_name"] == (key in renamed)
        assert previous <= renamed
        previous = renamed
        if level == 0:
            assert not renamed
        if level == 1:
            assert len(renamed) == len(current)


def test_semantic_stages_and_world_seed():
    doc = Doc("acme-signed", "contract", "Contract", "acme_signed_contract_2026", ())
    def entry(noise, world=7, layout=99):
        e = Entry(doc, "required")
        place_entries([e], FSConfig(filename_noise=noise), layout, world_seed=world)
        return e
    assert entry(0).name == "acme_signed_contract_2026.txt"
    # Choose an identity selected at every nonzero stage, without changing the seed.
    for i in range(100):
        doc.doc_id = f"acme-{i}"
        if entry(.3).noisy_name:
            break
    low, mid, high = entry(.3), entry(.6), entry(.9)
    assert low.name.startswith("acme_contract_")
    assert mid.name.startswith("contract_")
    assert high.name.startswith("doc_")
    assert high.name == entry(.9).name
    assert high.name == entry(.9, layout=123).name
    assert entry(1, world=7).name != entry(1, world=8).name


def test_collision_resolution_is_order_independent():
    docs = [Doc(f"id-{i}", "contract", "Contract", "same_name", ()) for i in range(150)]
    def placed(level, reverse=False):
        entries = [Entry(d, "required", copy=c) for d in docs for c in (0, 1)]
        if reverse:
            entries.reverse()
        place_entries(entries, FSConfig(filename_noise=level, depth=0, max_entries_per_dir=8), 7)
        assert len({e.path.casefold() for e in entries}) == len(entries)
        return {e.key: e.path for e in entries}
    for level in (0, .3, .6, .9, 1):
        assert placed(level) == placed(level, reverse=True)


def test_forced_noisy_collisions_preserve_clean_paths(monkeypatch):
    monkeypatch.setattr("fsbench.layout._noisy_stem", lambda *args: "collision")
    docs = [Doc(f"id-{i}", "contract", "Contract", "collision", ()) for i in range(40)]
    def placed(level, reverse=False):
        es = [Entry(d, "required") for d in docs]
        if reverse:
            es.reverse()
        place_entries(es, FSConfig(filename_noise=level), 7)
        assert len({e.path.casefold() for e in es}) == len(es)
        return {e.key: e for e in es}
    clean = placed(0)
    previous = set()
    for level in (.3, .6, .9, 1):
        current = placed(level)
        assert {k: e.path for k, e in current.items()} == {
            k: e.path for k, e in placed(level, reverse=True).items()}
        changed = {k for k, e in current.items() if e.path != clean[k].path}
        assert changed == {k for k, e in current.items() if e.noisy_name}
        assert previous <= changed
        previous = changed


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_render_is_byte_stable_across_time(fmt):
    doc = Doc("test", "contract", "Contract", "contract", (), fields=[("Status", "Signed")])
    first = render(doc, fmt)
    time.sleep(2.1)  # Cross ZIP's two-second timestamp boundary.
    assert render(doc, fmt) == first


def test_diagnostics_and_metadata(make_env, tmp_path):
    env, m = make_env("conflict", FSConfig(filename_noise=.6))
    req = next(f["path"] for f in m["files"] if f["role"] == "required")
    calls = [("glob", {"pattern": "**/*"}), ("glob", {"pattern": "**/*"}),
             ("read_file", {"path": req})]
    metrics = run_agent(env, ScriptedAgent(calls, m["ground_truth"]), tmp_path / "run")
    assert metrics["candidate_files_seen"] == len(m["files"])
    assert metrics["steps_to_first_required_evidence"] == 3
    meta = json.loads((tmp_path / "run/meta.json").read_text())
    assert meta["filename_noise"] == .6
    assert meta["world_seed"] == 7
    assert meta["condition_label"] == "filename_noise=0.6"
    assert evaluate_runs(tmp_path / "run")[0]["filename_noise"] == .6
    none = run_agent(env, ScriptedAgent(calls[:1], {}), tmp_path / "none")
    assert none["steps_to_first_required_evidence"] is None
    assert none["candidate_files_seen"] == len(m["files"])
    direct = run_agent(env, OracleAgent(m), tmp_path / "direct")
    assert direct["steps_to_first_required_evidence"] == 1
    assert direct["candidate_files_seen"] == 0


def test_truncated_candidates_are_not_counted(tmp_path):
    for name in ("aaa.txt", "bbb.txt", "ccc.txt"):
        (tmp_path / name).write_text("test")
    tracer = Tracer()
    tool = FileTools(tmp_path, tracer, max_output_chars=10)
    tool.call("list_directory", {})
    assert tracer.events[-1]["seen"] == ["aaa.txt"]


def test_missing_entire_condition_is_detected_and_csv_is_paired(tmp_path, capsys):
    root = tmp_path / "envs"
    run_sweep(["conflict"], FSConfig(), {"filename_noise": [0, .3, .6, .9]}, root,
              replicates=2, seed_offset=1)
    for path in sorted(root.glob("*/manifest.json")):
        m = json.loads(path.read_text())
        if m["config"]["filename_noise"] == .9:
            continue
        run_agent(path.parent, OracleAgent(m), tmp_path / "runs" / path.parent.name)
    rows = evaluate_runs(tmp_path / "runs")
    assert balance_warnings(rows)
    assert keep_paired(rows) == []
    empty = tmp_path / "empty.csv"
    empty.write_text("stale results")
    assert evaluate_runs(tmp_path / "runs", empty, paired=True) == []
    assert len(empty.read_text().splitlines()) == 1
    # Add the last condition for only one seed.
    path = root / "conflict__filename_noise=0.9__r0"
    m = json.loads((path / "manifest.json").read_text())
    run_agent(path, OracleAgent(m), tmp_path / "runs" / path.name)
    out = tmp_path / "paired.csv"
    paired = evaluate_runs(tmp_path / "runs", out, paired=True)
    assert len(paired) == 4
    assert {r["world_seed"] for r in paired} == {1}
    assert len(out.read_text().splitlines()) == 5
    assert "missing" in capsys.readouterr().err


def test_manual_expected_conditions_and_mismatched_task_seed():
    rows = [{"condition": f"filename_noise={v}", "seed.world": 7, "seed.task": 7,
             "seed.layout": 7} for v in (0, .3, .6)]
    expected = {"filename_noise": [0, .3, .6, .9]}
    assert keep_paired(rows, expected) == []
    rows.append({**rows[0], "condition": "filename_noise=0.9", "seed.task": 8})
    assert keep_paired(rows, expected) == []


def test_manual_configs_pair_using_noise_metadata():
    rows = [{"condition": "custom settings", "filename_noise": v, "seed.world": 7}
            for v in (0, .3, .6, .9)]
    assert keep_paired(rows, {"filename_noise": [0, .3, .6, .9]}) == rows


def test_mime_diversity_pairing_uses_underlying_format_config():
    rows = [{"sweep.vars": f"mime_diversity={v}", "cfg.mime_types": formats, "seed.world": 7}
            for v, formats in [(1, "txt"), (2, "txt+pdf")]]
    assert keep_paired(rows, {"mime_diversity": [1, 2]}) == rows


def test_fractional_plot_values_and_output(tmp_path):
    pytest.importorskip("matplotlib")
    levels = [0, .3, .6, .9]
    rows = [{"cfg.filename_noise": v, "success": 1, "n_calls": 2, "total_tokens": 30,
             "cost_usd": .01, "wall_time_s": 1, "agent": "test", "toolset": "files"}
            for v in levels]
    assert [_x_value(r, "filename_noise") for r in rows] == levels
    outputs = plot_runs(rows, "filename_noise", tmp_path)
    assert {p.name for p in outputs} == {f"{m}_vs_filename_noise.png"
                                        for m in ("success", "n_calls", "total_tokens", "cost_usd", "wall_time_s")}
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in outputs)


def test_small_api_costs_remain_visible_in_summary(capsys):
    from fsbench.cli import _print_table
    from fsbench.evaluate import summarize

    summary = summarize([{"filename_noise": 0, "cost_usd": .000751},
                         {"filename_noise": 0, "cost_usd": .000750}], ("filename_noise",))
    assert summary[0]["cost_usd"] == .0007505
    _print_table([{"cost_usd": .000750}, {"cost_usd": .000917}])
    printed = capsys.readouterr().out
    assert "0.000750" in printed and "0.000917" in printed
