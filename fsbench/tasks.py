"""Task families, their ground truth, and their scorers.

Each builder returns a ``TaskInstance`` describing:

* ``required``          docs needed to answer (the answer is derived from these)
* ``inherent_traps``    conflicting versions that are always present (the point of the task)
* ``stale_candidates``  outdated versions included at ``stale_version_rate``
* ``near_pool``         an endless stream of semantically similar but irrelevant docs

None of the distractor or trap docs change the ground truth.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

from fsbench.builders import contract_doc, employee_doc, invoice_doc, ledger_doc, quote_doc
from fsbench.docs import Doc, plain_money, slug
from fsbench.rng import keyed_rng
from fsbench.world import (
    DEPARTMENTS,
    FIRST_NAMES,
    ITEM_CATALOG,
    LAST_NAMES,
    OFFICES,
    PAYMENT_METHODS,
    TODAY,
    Contract,
    Employee,
    Invoice,
    Payment,
    World,
    add_months,
)

WORKSPACE = "/workspace"


@dataclass
class TaskInstance:
    task_type: str
    level: int
    task_id: str
    prompt: str
    answer_schema: dict
    ground_truth: dict
    required: list[Doc]
    inherent_traps: list[Doc]
    stale_candidates: list[Doc]
    near_pool: Callable[[], Iterator[Doc]]
    output_path: str | None = None
    decoy_answers: dict[str, dict] = field(default_factory=dict)


def _full_prompt(question: str, schema: dict, extra: str = "") -> str:
    return (
        f"You have access to a company file share mounted at {WORKSPACE}.\n\n"
        f"{question}\n\n"
        f"{extra}"
        "When you are done, reply with only a JSON object of this form:\n"
        f"{json.dumps(schema, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Level 2: locate + retrieve
# ---------------------------------------------------------------------------


def _employee_answer(e: Employee, *, superseded: bool = False) -> dict:
    if superseded:
        return {"employee_id": e.employee_id, "job_title": e.prev_title, "manager": e.prev_manager}
    return {"employee_id": e.employee_id, "job_title": e.title, "manager": e.manager}


def _retrieve_confusable(world: World, target: Employee, task_seed: int):
    prng = keyed_rng("near", world.seed, task_seed, "retrieve")
    used_ids = {e.employee_id for e in world.employees}
    used_names = {e.name for e in world.employees}

    def new_id() -> str:
        while True:
            eid = f"E-{prng.randint(10000, 99999)}"
            if eid not in used_ids:
                used_ids.add(eid)
                return eid

    other_dept = prng.choice([d for d in DEPARTMENTS if d != target.department])
    twin = Employee(new_id(), target.first, target.last, other_dept,
                    DEPARTMENTS[other_dept][1], DEPARTMENTS[other_dept][0],
                    manager=prng.choice(world.employees).name, prev_manager="",
                    start_date=date(2014, 5, 12), office=prng.choice(OFFICES),
                    email=f"{target.first.lower()}.{target.last.lower()}2@{world.domain}")
    similar_last = target.last + ("g" if not target.last.endswith("g") else "s")
    lookalike = Employee(new_id(), target.first, similar_last, target.department,
                         target.prev_title, target.prev_title, manager=target.prev_manager,
                         start_date=target.start_date + timedelta(days=200), office=target.office,
                         email=f"{target.first.lower()}.{similar_last.lower()}@{world.domain}")
    used_names.add(lookalike.name)
    pairs = [
        (employee_doc(world, twin, status="Terminated"), _employee_answer(twin)),
        (employee_doc(world, lookalike), _employee_answer(lookalike)),
    ]
    return pairs, prng, used_ids, used_names


def build_retrieve(world: World, task_seed: int) -> TaskInstance:
    rng = keyed_rng("task", world.seed, task_seed, "retrieve")
    target = rng.choice(world.employees)
    schema = {
        "employee_id": "string, e.g. E-12345",
        "job_title": "string",
        "manager": "string, the manager's full name",
    }
    question = (
        f"Find the current HR record for {target.name} in the {target.department} department. "
        "Report their employee ID, current job title, and current manager."
    )
    stale = employee_doc(world, target, superseded=True)
    confusable, _, _, _ = _retrieve_confusable(world, target, task_seed)
    decoys = {stale.doc_id: _employee_answer(target, superseded=True)}
    decoys.update({d.doc_id: a for d, a in confusable})

    def near_pool() -> Iterator[Doc]:
        pairs, prng, used_ids, used_names = _retrieve_confusable(world, target, task_seed)
        for doc, _ in pairs:
            yield doc

        def new_id() -> str:
            while True:
                eid = f"E-{prng.randint(10000, 99999)}"
                if eid not in used_ids:
                    used_ids.add(eid)
                    return eid

        others = [e for e in world.employees if e is not target]
        prng.shuffle(others)
        for e in others:
            yield employee_doc(world, e)

        while True:
            first, last = prng.choice(FIRST_NAMES), prng.choice(LAST_NAMES)
            if f"{first} {last}" in used_names:
                continue
            used_names.add(f"{first} {last}")
            dept = prng.choice(list(DEPARTMENTS))
            titles = DEPARTMENTS[dept]
            e = Employee(new_id(), first, last, dept, titles[1], titles[0],
                         manager=prng.choice(world.employees).name,
                         start_date=date(2016, 1, 1) + timedelta(days=prng.randint(0, 3800)),
                         office=prng.choice(OFFICES), email=f"{first.lower()}.{last.lower()}@{world.domain}")
            yield employee_doc(world, e, status=prng.choice(["Active", "Terminated", "Contractor"]))

    return TaskInstance(
        task_type="retrieve",
        level=2,
        task_id=f"retrieve-w{world.seed}-t{task_seed}",
        prompt=_full_prompt(question, schema, "Use the employee's current record, not an outdated one.\n\n"),
        answer_schema=schema,
        ground_truth=_employee_answer(target),
        required=[employee_doc(world, target)],
        inherent_traps=[],
        stale_candidates=[stale],
        near_pool=near_pool,
        decoy_answers=_distinct_decoys("retrieve", decoys, _employee_answer(target)),
    )


# ---------------------------------------------------------------------------
# Level 3 / 5: cross-file reconciliation, and the same as a write workflow
# ---------------------------------------------------------------------------


def _pick_client_quarter(world: World, rng) -> tuple[str, int]:
    options = []
    for client in world.clients:
        for q in (1, 2, 3):
            invs = world.invoices_for(client, q)
            if not 2 <= len(invs) <= 6:
                continue
            outstanding = [i.total_cents - world.paid_cents(i.invoice_id) for i in invs]
            if any(o > 0 for o in outstanding) and any(o == 0 for o in outstanding):
                options.append((client, q))
    if not options:
        options = [(c, q) for c in world.clients for q in (1, 2, 3)
                   if world.invoices_for(c, q) and world.payments_for({i.invoice_id for i in world.invoices_for(c, q)})]
    return rng.choice(sorted(options))


def _draft_items(items, rng) -> list[tuple[str, int, int]]:
    desc, qty, unit = items[0]
    if qty > 1 and rng.random() < 0.5:
        changed = (desc, qty + rng.choice([-1, 1, 2]), unit)
    else:
        changed = (desc, qty, unit + rng.choice([-1, 1]) * rng.randint(5, 40) * 100)
    return [changed, *items[1:]]


def _synthetic_2025_invoice(world: World, k: int) -> Invoice:
    rng = keyed_rng("inv2025", world.seed, k)
    issue = date(2025, rng.randint(1, 12), rng.randint(1, 28))
    items = [(d, rng.randint(ql, qh), rng.randint(lo // 100, hi // 100) * 100)
             for d, lo, hi, ql, qh in rng.sample(ITEM_CATALOG, rng.randint(1, 2))]
    return Invoice(f"INV-0{k:03d}", rng.choice(world.clients), issue, issue + timedelta(days=30),
                   f"PO-{rng.randint(10000, 99999)}", items)


def _ledger_2025(world: World) -> Doc:
    rng = keyed_rng("ledger2025", world.seed)
    payments = []
    for k in range(60):
        inv = _synthetic_2025_invoice(world, k)
        if rng.random() < 0.8:
            payments.append(Payment(f"PAY-1{k:04d}", inv.issue_date + timedelta(days=rng.randint(10, 40)),
                                    inv.client, inv.invoice_id, inv.total_cents, rng.choice(PAYMENT_METHODS)))
    return ledger_doc(world, payments, date(2025, 12, 31), doc_id="ledger:2025",
                      stem="payment_ledger_2025", year=2025)


def _reconcile_parts(world: World, task_seed: int, stream: str):
    rng = keyed_rng("task", world.seed, task_seed, stream)
    client, q = _pick_client_quarter(world, rng)
    invoices = sorted(world.invoices_for(client, q), key=lambda i: i.issue_date)
    ids = {i.invoice_id for i in invoices}
    outstanding = {i.invoice_id: i.total_cents - world.paid_cents(i.invoice_id) for i in invoices}
    unpaid = sorted(k for k, v in outstanding.items() if v > 0)

    ledger = ledger_doc(world, world.payments, TODAY, doc_id="ledger:2026", stem="payment_ledger_2026")
    target_payments = world.payments_for(ids)
    stale_as_of = max(p.date for p in target_payments) - timedelta(days=1) if target_payments else date(2026, 6, 30)
    stale_ledger = ledger_doc(world, world.payments, stale_as_of, doc_id=f"ledger-stale:{stale_as_of}",
                              stem="payment_ledger_2026_old")
    draft_specs = [(i, _draft_items(i.items, rng)) for i in invoices]
    drafts = [invoice_doc(world, i, draft_items=items) for i, items in draft_specs]

    def near_pool() -> Iterator[Doc]:
        prng = keyed_rng("near", world.seed, task_seed, stream)
        hardest = [invoice_doc(world, i) for i in world.invoices
                   if i.invoice_id not in ids and (i.client == client or i.quarter == q)]
        q_start = date(2026, 3 * q - 2, 1)
        for k in range(3):
            qd = q_start + timedelta(days=prng.randint(0, 85))
            items = [(d, prng.randint(ql, qh), prng.randint(lo // 100, hi // 100) * 100)
                     for d, lo, hi, ql, qh in prng.sample(ITEM_CATALOG, 2)]
            hardest.append(quote_doc(world, client, f"Q-{prng.randint(3000, 3999)}{k}", qd, items))
        prng.shuffle(hardest)
        yield from hardest
        yield _ledger_2025(world)
        rest = [invoice_doc(world, i) for i in world.invoices
                if i.invoice_id not in ids and i.client != client and i.quarter != q]
        prng.shuffle(rest)
        yield from rest
        k = 0
        while True:
            yield invoice_doc(world, _synthetic_2025_invoice(world, k))
            k += 1

    return client, q, invoices, outstanding, unpaid, ledger, stale_ledger, drafts, draft_specs, stale_as_of, near_pool


def _outstanding_answer(outstanding: dict[str, int], *, workflow: bool) -> dict:
    unpaid = sorted(k for k, v in outstanding.items() if v > 0)
    if workflow:
        return {"rows": [{"invoice_id": k, "amount_due": plain_money(outstanding[k])} for k in unpaid]}
    return {"outstanding_balance": round(sum(outstanding.values()) / 100, 2), "unpaid_invoices": unpaid}


def _reconcile_decoys(world: World, invoices: list[Invoice], outstanding: dict[str, int],
                      stale_ledger: Doc, stale_as_of: date, draft_specs, *, workflow: bool) -> dict[str, dict]:
    decoys = {}
    stale_out = {i.invoice_id: i.total_cents - world.paid_cents(i.invoice_id, as_of=stale_as_of) for i in invoices}
    decoys[stale_ledger.doc_id] = _outstanding_answer(stale_out, workflow=workflow)
    for inv, items in draft_specs:
        alt = dict(outstanding)
        alt[inv.invoice_id] = sum(q * u for _, q, u in items) - world.paid_cents(inv.invoice_id)
        decoys[invoice_doc(world, inv, draft_items=items).doc_id] = _outstanding_answer(alt, workflow=workflow)
    return decoys


def _quarter_range(q: int) -> tuple[date, date]:
    start = date(2026, 3 * q - 2, 1)
    return start, add_months(start, 3) - timedelta(days=1)


def build_reconcile(world: World, task_seed: int) -> TaskInstance:
    client, q, invoices, outstanding, unpaid, ledger, stale_ledger, drafts, draft_specs, stale_as_of, near_pool = _reconcile_parts(
        world, task_seed, "reconcile")
    start, end = _quarter_range(q)
    schema = {
        "outstanding_balance": "number, total USD still owed across those invoices, e.g. 1234.56",
        "unpaid_invoices": ["invoice numbers that are not fully paid, e.g. INV-1234"],
    }
    question = (
        f"Our client {client} was invoiced several times in Q{q} 2026 (invoices issued {start} to {end}). "
        "Using the issued invoices and the most recent payment ledger, work out the total outstanding "
        "balance on those invoices and list every one that is not fully paid (including partially paid)."
    )
    return TaskInstance(
        task_type="reconcile",
        level=3,
        task_id=f"reconcile-w{world.seed}-t{task_seed}",
        prompt=_full_prompt(question, schema),
        answer_schema=schema,
        ground_truth={
            "outstanding_balance": round(sum(outstanding.values()) / 100, 2),
            "unpaid_invoices": unpaid,
            "client": client,
            "quarter": q,
        },
        required=[*(invoice_doc(world, i) for i in invoices), ledger],
        inherent_traps=[],
        stale_candidates=[stale_ledger, *drafts],
        near_pool=near_pool,
        decoy_answers=_distinct_decoys(
            "reconcile",
            _reconcile_decoys(world, invoices, outstanding, stale_ledger, stale_as_of, draft_specs, workflow=False),
            {"outstanding_balance": round(sum(outstanding.values()) / 100, 2), "unpaid_invoices": unpaid},
        ),
    )


def build_workflow(world: World, task_seed: int) -> TaskInstance:
    client, q, invoices, outstanding, unpaid, ledger, stale_ledger, drafts, draft_specs, stale_as_of, near_pool = _reconcile_parts(
        world, task_seed, "workflow")
    start, end = _quarter_range(q)
    out_rel = f"reports/unpaid_{slug(client)}_q{q}_2026.csv"
    schema = {"output_path": f"string, the path of the file you wrote (should be {WORKSPACE}/{out_rel})"}
    question = (
        f"Prepare a collections report for our client {client} covering invoices issued in Q{q} 2026 "
        f"({start} to {end}). Using the issued invoices and the most recent payment ledger, write a CSV file to "
        f"{WORKSPACE}/{out_rel} with the header `invoice_id,amount_due` and one row per invoice that is not "
        "fully paid, where amount_due is the remaining USD balance (e.g. 1234.56)."
    )
    return TaskInstance(
        task_type="workflow",
        level=5,
        task_id=f"workflow-w{world.seed}-t{task_seed}",
        prompt=_full_prompt(question, schema),
        answer_schema=schema,
        ground_truth={
            "output_path": out_rel,
            "rows": [{"invoice_id": k, "amount_due": plain_money(outstanding[k])} for k in unpaid],
            "client": client,
            "quarter": q,
        },
        required=[*(invoice_doc(world, i) for i in invoices), ledger],
        inherent_traps=[],
        stale_candidates=[stale_ledger, *drafts],
        near_pool=near_pool,
        output_path=out_rel,
        decoy_answers=_distinct_decoys(
            "workflow",
            _reconcile_decoys(world, invoices, outstanding, stale_ledger, stale_as_of, draft_specs, workflow=True),
            {"rows": [{"invoice_id": k, "amount_due": plain_money(outstanding[k])} for k in unpaid]},
        ),
    )


# ---------------------------------------------------------------------------
# Level 4: conflict resolution between versions
# ---------------------------------------------------------------------------

VENDOR_A = ["Apex", "Summit", "Pioneer", "Atlas", "Beacon", "Crest", "Meridian", "Harbor"]
VENDOR_B = ["Supply", "Logistics", "Software", "Staffing", "Facilities", "Media"]


def _contract_answer(c: Contract) -> dict:
    return {
        "contract_id": c.contract_id,
        "renewal_date": c.renewal_date.isoformat(),
        "annual_value": round(c.annual_value_cents / 100, 2),
    }


def _alt_contract(c: Contract, rng) -> Contract:
    term = rng.choice([t for t in (12, 24, 36) if t != c.term_months])
    delta = rng.choice([-1, 1]) * rng.randint(5, 25) * c.annual_value_cents // 100 // 100000 * 100000
    return replace(c, term_months=term, annual_value_cents=max(100000, c.annual_value_cents + (delta or 500000)))


def build_conflict(world: World, task_seed: int) -> TaskInstance:
    rng = keyed_rng("task", world.seed, task_seed, "conflict")
    client = rng.choice(world.clients)
    c = world.contracts[client]
    s = slug(client)
    signed_on = c.effective_date - timedelta(days=rng.randint(3, 20))
    executed = contract_doc(world, c, doc_id=f"contract:{c.contract_id}", stem=f"msa_{s}_signed",
                            status="EXECUTED - in force", signed_on=signed_on)
    c_draft = _alt_contract(c, rng)
    c_final = _alt_contract(c, rng)
    c_redline = _alt_contract(c, rng)
    draft_v1 = contract_doc(world, c_draft, doc_id=f"contract-draft1:{c.contract_id}",
                            stem=f"msa_{s}_draft_v1", status="DRAFT v1 - for discussion only", signed_on=None)
    final_unsigned = contract_doc(world, c_final, doc_id=f"contract-final:{c.contract_id}",
                                  stem=f"msa_{s}_final", status="FINAL DRAFT - pending signature",
                                  signed_on=None)
    redline = contract_doc(world, c_redline, doc_id=f"contract-redline:{c.contract_id}",
                           stem=f"msa_{s}_redline_v3", status="REDLINE v3 - superseded", signed_on=None)
    prior = world.prior_contracts[client]
    expired = contract_doc(world, prior, doc_id=f"contract-prior:{prior.contract_id}",
                           stem=f"msa_{s}_{prior.effective_date.year}",
                           status=f"EXPIRED - superseded by {c.contract_id}",
                           signed_on=prior.effective_date - timedelta(days=7))
    schema = {
        "contract_id": "string, the agreement number",
        "renewal_date": "string, YYYY-MM-DD",
        "annual_value": "number, annual contract value in USD",
    }
    question = (
        f"What are the agreement number, renewal date, and annual contract value of the executed "
        f"Master Services Agreement currently in force with {client}? Several versions of this agreement "
        "may exist; only the signed agreement that is currently in force counts."
    )

    def near_pool() -> Iterator[Doc]:
        prng = keyed_rng("near", world.seed, task_seed, "conflict")
        others = []
        for other in world.clients:
            if other == client:
                continue
            oc = world.contracts[other]
            others.append(contract_doc(world, oc, doc_id=f"contract:{oc.contract_id}",
                                       stem=f"msa_{slug(other)}_signed", status="EXECUTED - in force",
                                       signed_on=oc.effective_date - timedelta(days=5)))
            others.append(contract_doc(world, _alt_contract(oc, prng), doc_id=f"contract-draft1:{oc.contract_id}",
                                       stem=f"msa_{slug(other)}_draft_v1", status="DRAFT v1 - for discussion only",
                                       signed_on=None))
        prng.shuffle(others)
        yield from others
        k = 0
        while True:
            vendor = f"{prng.choice(VENDOR_A)} {prng.choice(VENDOR_B)} {k}"
            eff = date(2024, 1, 1) + timedelta(days=prng.randint(0, 900))
            vc = Contract(f"VSA-{eff.year}-{1000 + k}", vendor, eff, prng.choice([12, 24]),
                          prng.randint(10, 500) * 10000, "Net 30",
                          f"{prng.choice(FIRST_NAMES)} {prng.choice(LAST_NAMES)}", c.our_signatory)
            doc = contract_doc(world, vc, doc_id=f"vendor-contract:{vc.contract_id}",
                               stem=f"vendor_agreement_{slug(vendor)}", status="EXECUTED", signed_on=eff)
            doc.home = ("legal", "vendor_agreements")
            yield doc
            k += 1

    return TaskInstance(
        task_type="conflict",
        level=4,
        task_id=f"conflict-w{world.seed}-t{task_seed}",
        prompt=_full_prompt(question, schema),
        answer_schema=schema,
        ground_truth={
            "contract_id": c.contract_id,
            "renewal_date": c.renewal_date.isoformat(),
            "annual_value": round(c.annual_value_cents / 100, 2),
        },
        required=[executed],
        inherent_traps=[draft_v1, final_unsigned],
        stale_candidates=[redline, expired],
        near_pool=near_pool,
        decoy_answers=_distinct_decoys("conflict", {
            draft_v1.doc_id: _contract_answer(c_draft),
            final_unsigned.doc_id: _contract_answer(c_final),
            redline.doc_id: _contract_answer(c_redline),
            expired.doc_id: _contract_answer(prior),
            **{f"contract:{oc.contract_id}": _contract_answer(oc)
               for other, oc in world.contracts.items() if other != client},
        }, {
            "contract_id": c.contract_id,
            "renewal_date": c.renewal_date.isoformat(),
            "annual_value": round(c.annual_value_cents / 100, 2),
        }),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def parse_money(v) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        cleaned = re.sub(r"[,$\s]|USD", "", v, flags=re.I)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def parse_date(v) -> date | None:
    if not isinstance(v, str):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(v.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _money_eq(a, b) -> bool:
    pa, pb = parse_money(a), parse_money(b)
    return pa is not None and pb is not None and abs(pa - pb) <= 0.01


def _decoy_differs(task_type: str, decoy: dict, gt: dict) -> bool:
    if task_type == "workflow":
        return decoy.get("rows") != gt.get("rows")
    if task_type == "reconcile":
        return not (_money_eq(decoy.get("outstanding_balance"), gt.get("outstanding_balance"))
                    and decoy.get("unpaid_invoices") == gt.get("unpaid_invoices"))
    if task_type == "conflict":
        return not (_money_eq(decoy.get("annual_value"), gt.get("annual_value"))
                    and decoy.get("renewal_date") == gt.get("renewal_date")
                    and _norm(decoy.get("contract_id", "")) == _norm(gt.get("contract_id", "")))
    return any(_norm(decoy.get(k, "")) != _norm(gt.get(k, "")) for k in ("employee_id", "job_title", "manager"))


def _distinct_decoys(task_type: str, decoys: dict[str, dict], gt: dict) -> dict[str, dict]:
    return {k: v for k, v in decoys.items() if _decoy_differs(task_type, v, gt)}


def extract_decoy(task_type: str, doc: Doc) -> dict | None:
    """Best-effort decoy from a document's fields (used for near-miss files)."""
    f = dict(doc.fields)
    if task_type == "retrieve" and doc.kind == "employee":
        return {"employee_id": f.get("Employee ID"), "job_title": f.get("Job title"), "manager": f.get("Manager")}
    if task_type == "conflict" and doc.kind == "contract":
        return {
            "contract_id": f.get("Agreement number"),
            "renewal_date": f.get("Renewal date"),
            "annual_value": parse_money(f.get("Annual contract value")),
        }
    if task_type in ("reconcile", "workflow") and doc.kind == "invoice":
        return {"outstanding_balance": parse_money(f.get("Total due")), "invoice_id": f.get("Invoice number")}
    return None


def _fields_result(checks: dict[str, bool]) -> dict:
    return {"fields": checks, "score": sum(checks.values()) / len(checks), "success": all(checks.values())}


def score_retrieve(answer: dict, gt: dict, workspace: Path) -> dict:
    return _fields_result({k: _norm(answer.get(k, "")) == _norm(gt[k]) for k in ("employee_id", "job_title", "manager")})


def score_reconcile(answer: dict, gt: dict, workspace: Path) -> dict:
    balance_ok = _money_eq(answer.get("outstanding_balance"), gt["outstanding_balance"])
    raw = answer.get("unpaid_invoices") or []
    pred = {_norm(x).upper() for x in raw} if isinstance(raw, list) else set()
    true = {x.upper() for x in gt["unpaid_invoices"]}
    jaccard = len(pred & true) / len(pred | true) if pred | true else 1.0
    return {
        "fields": {"outstanding_balance": balance_ok, "unpaid_invoices": pred == true},
        "unpaid_jaccard": jaccard,
        "score": 0.5 * balance_ok + 0.5 * jaccard,
        "success": balance_ok and pred == true,
    }


def score_conflict(answer: dict, gt: dict, workspace: Path) -> dict:
    return _fields_result({
        "contract_id": _norm(answer.get("contract_id", "")) == _norm(gt["contract_id"]),
        "renewal_date": parse_date(answer.get("renewal_date")) == date.fromisoformat(gt["renewal_date"]),
        "annual_value": _money_eq(answer.get("annual_value"), gt["annual_value"]),
    })


def score_workflow(answer: dict, gt: dict, workspace: Path) -> dict:
    path = workspace / gt["output_path"]
    true = {(r["invoice_id"].upper(), round(float(r["amount_due"]), 2)) for r in gt["rows"]}
    if not path.is_file():
        return {"fields": {"file_exists": False, "header": False, "rows": False}, "row_f1": 0.0,
                "score": 0.0, "success": False}
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8", errors="replace")))
    header = [h.strip().lower() for h in (reader.fieldnames or [])]
    header_ok = "invoice_id" in header and "amount_due" in header
    pred = set()
    if header_ok:
        for row in reader:
            row = {(k or "").strip().lower(): v for k, v in row.items()}
            amount = parse_money(row.get("amount_due") or "")
            if row.get("invoice_id") and amount is not None:
                pred.add((row["invoice_id"].strip().upper(), round(amount, 2)))
    tp = len(pred & true)
    precision = tp / len(pred) if pred else (1.0 if not true else 0.0)
    recall = tp / len(true) if true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "fields": {"file_exists": True, "header": header_ok, "rows": pred == true},
        "row_f1": f1,
        "score": f1 if header_ok else 0.0,
        "success": header_ok and pred == true,
    }


@dataclass(frozen=True)
class TaskType:
    build: Callable[[World, int], TaskInstance]
    score: Callable[[dict, dict, Path], dict]


TASK_TYPES: dict[str, TaskType] = {
    "retrieve": TaskType(build_retrieve, score_retrieve),
    "reconcile": TaskType(build_reconcile, score_reconcile),
    "conflict": TaskType(build_conflict, score_conflict),
    "workflow": TaskType(build_workflow, score_workflow),
}
