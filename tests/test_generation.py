import re
from dataclasses import replace

import pytest

from fsbench import PRESETS, FSConfig
from fsbench.builders import contract_doc, employee_doc, invoice_doc, ledger_doc
from fsbench.docs import ALL_FORMATS, money, plain_money
from fsbench.extract import extract_text
from fsbench.paths import long_path
from fsbench.render import render
from fsbench.tasks import TASK_TYPES
from fsbench.world import TODAY, build_world

TASKS = sorted(TASK_TYPES)

KNOB_VALUES = {
    "depth": [0, 3, 10],
    "filename_noise": [0.0, 0.9],
    "dirname_noise": [0.0, 0.9],
    "scatter": [0.0, 0.9],
    "distractors": [0, 120],
    "semantic_similarity": [0.0, 1.0],
    "duplicate_rate": [0.0, 0.5],
    "stale_version_rate": [0.0, 1.0],
    "mime_types": [("txt",), ALL_FORMATS],
    "max_entries_per_dir": [None, 5],
}


@pytest.mark.parametrize("task", TASKS)
def test_ground_truth_and_required_docs_are_invariant_across_conditions(make_env, task):
    _, base = make_env(task, FSConfig())
    for key, values in KNOB_VALUES.items():
        for v in values:
            _, m = make_env(task, replace(FSConfig(distractors=30), **{key: v}))
            assert m["ground_truth"] == base["ground_truth"], key
            assert m["required_doc_ids"] == base["required_doc_ids"], key
            assert m["task"]["prompt"] == base["task"]["prompt"], key
            present = {f["doc_id"] for f in m["files"]}
            assert set(m["required_doc_ids"]) <= present, key


@pytest.mark.parametrize("task", TASKS)
def test_doc_ids_are_unique_across_many_worlds(task):
    from itertools import islice

    for seed in range(40):
        t = TASK_TYPES[task].build(build_world(seed), seed)
        fixed = [d.doc_id for d in t.required + t.inherent_traps + t.stale_candidates]
        near = [d.doc_id for d in islice(t.near_pool(), 500)]
        assert len(set(fixed)) == len(fixed)
        assert len(set(near)) == len(near)
        assert not set(near) & set(fixed)


@pytest.mark.parametrize("task", TASKS)
def test_generation_is_deterministic(make_env, task):
    _, a = make_env(task, PRESETS["moderate"])
    _, b = make_env(task, PRESETS["moderate"])
    strip = lambda m: [(f["path"], f["doc_id"], f["role"], f["format"]) for f in m["files"]]
    assert strip(a) == strip(b)


def test_threshold_knobs_are_nested(make_env):
    low = FSConfig(distractors=150, filename_noise=0.3, scatter=0.3, mime_types=("txt",))
    _, m_low = make_env("reconcile", low)
    _, m_high = make_env("reconcile", replace(low, filename_noise=0.7, scatter=0.7))
    noisy = lambda m: {(f["doc_id"], f["copy"]) for f in m["files"] if f["noisy_name"]}
    scattered = lambda m: {(f["doc_id"], f["copy"]) for f in m["files"] if f["scattered"]}
    assert noisy(m_low) < noisy(m_high)
    assert scattered(m_low) < scattered(m_high)

    _, few = make_env("reconcile", FSConfig(distractors=20))
    _, many = make_env("reconcile", FSConfig(distractors=100))
    ids = lambda m: {f["doc_id"] for f in m["files"]}
    assert ids(few) < ids(many)


def test_distractor_mix_follows_semantic_similarity(make_env):
    _, unrelated = make_env("reconcile", FSConfig(distractors=80, semantic_similarity=0.0))
    _, similar = make_env("reconcile", FSConfig(distractors=80, semantic_similarity=1.0))
    assert unrelated["stats"]["role_counts"].get("near", 0) == 0
    assert similar["stats"]["role_counts"].get("generic", 0) == 0


@pytest.mark.parametrize("task", TASKS)
def test_layout_constraints(make_env, task):
    cfg = replace(PRESETS["horrible"], distractors=150, max_entries_per_dir=12, depth=6)
    env, m = make_env(task, cfg)
    paths = [f["path"] for f in m["files"]]
    assert len({p.casefold() for p in paths}) == len(paths)
    for f in m["files"]:
        depth = f["path"].count("/")
        assert depth in (6, 7), f["path"]  # +1 when an overfull folder was split
        assert long_path(env / "workspace").joinpath(*f["path"].split("/")).is_file()
    required = [f for f in m["files"] if f["role"] == "required"]
    assert all(f["doc_id"] in m["required_doc_ids"] for f in required)
    assert not set(m["trap_doc_ids"]) & set(m["required_doc_ids"])


def test_very_deep_trees_work_despite_windows_path_limits(make_env):
    env, m = make_env("conflict", FSConfig(depth=14, distractors=10, mime_types=("pdf", "docx")))
    assert m["stats"]["max_depth"] == 14


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_facts_survive_every_format(tmp_path, fmt):
    world = build_world(3)
    inv = world.invoices[0]
    c = world.contracts[world.clients[0]]
    e = world.employees[0]
    payments = world.payments[:25]
    cases = [
        (invoice_doc(world, inv), [inv.invoice_id, money(inv.total_cents), inv.issue_date.isoformat()]),
        (ledger_doc(world, payments, TODAY, doc_id="l", stem="l"),
         [p.payment_id for p in payments] + [plain_money(p.amount_cents) for p in payments]),
        (contract_doc(world, c, doc_id="c", stem="c", status="EXECUTED - in force", signed_on=c.effective_date),
         [c.contract_id, c.renewal_date.isoformat(), money(c.annual_value_cents), "EXECUTED"]),
        (employee_doc(world, e), [e.employee_id, e.title, e.manager, "CURRENT"]),
    ]
    for doc, facts in cases:
        path = tmp_path / f"{doc.kind}.{fmt}"
        path.write_bytes(render(doc, fmt))
        text = extract_text(path)
        for fact in facts:
            assert fact in text, (doc.kind, fmt, fact)


def _solve_reconcile_from_text(env, manifest, roles=("required",), ledger_id=None):
    """Recompute the answer purely from document text (txt format)."""
    ws = long_path(env / "workspace")
    totals, paid = {}, {}
    for f in manifest["files"]:
        if f["role"] not in roles and f["doc_id"] != ledger_id:
            continue
        text = ws.joinpath(*f["path"].split("/")).read_text(encoding="utf-8")
        if f["kind"] == "invoice":
            inv = re.search(r"Invoice number: (INV-\d+)", text).group(1)
            totals[inv] = round(float(re.search(r"Total due: \$([\d,]+\.\d\d)", text).group(1).replace(",", "")), 2)
        elif f["kind"] == "ledger" and (ledger_id is None or f["doc_id"] == ledger_id):
            paid = {}
            for inv, amount in re.findall(r"\| (INV-\d+)\s*\| ([\d.]+)", text):
                paid[inv] = paid.get(inv, 0) + float(amount)
    owed = {k: round(v - paid.get(k, 0), 2) for k, v in totals.items()}
    return round(sum(owed.values()), 2), sorted(k for k, v in owed.items() if v > 0.005)


@pytest.mark.parametrize("seed", range(8))
def test_reconcile_is_solvable_from_documents_and_stale_ledger_is_a_real_trap(make_env, seed):
    env, m = make_env("reconcile", FSConfig(stale_version_rate=1.0), seed=seed)
    balance, unpaid = _solve_reconcile_from_text(env, m)
    assert balance == m["ground_truth"]["outstanding_balance"]
    assert unpaid == m["ground_truth"]["unpaid_invoices"]
    assert unpaid, "every reconcile instance should have at least one unpaid invoice"

    stale = next(f["doc_id"] for f in m["files"] if f["doc_id"].startswith("ledger-stale"))
    stale_balance, _ = _solve_reconcile_from_text(env, m, roles=("required",), ledger_id=stale)
    assert stale_balance != balance


@pytest.mark.parametrize("seed", range(5))
def test_conflict_traps_disagree_with_executed_contract(make_env, seed):
    env, m = make_env("conflict", FSConfig(stale_version_rate=1.0), seed=seed)
    ws = long_path(env / "workspace")
    gt = m["ground_truth"]
    for f in m["files"]:
        text = ws.joinpath(*f["path"].split("/")).read_text(encoding="utf-8")
        has_both = gt["renewal_date"] in text and money(int(round(gt["annual_value"] * 100))) in text
        assert has_both == (f["role"] == "required"), f["path"]
