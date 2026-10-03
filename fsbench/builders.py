"""Turn world entities into Docs."""

from __future__ import annotations

from datetime import date

from fsbench.docs import FORM_FORMATS, PROSE_FORMATS, TABULAR_FORMATS, Doc, Table, money, plain_money, slug
from fsbench.world import YEAR, Contract, Employee, Invoice, Payment, World


def invoice_doc(world: World, inv: Invoice, *, draft_items: list[tuple[str, int, int]] | None = None) -> Doc:
    items = draft_items if draft_items is not None else inv.items
    total = sum(q * u for _, q, u in items)
    is_draft = draft_items is not None
    stem = f"invoice_{slug(inv.client)}_{inv.issue_date:%Y-%m}_{inv.invoice_id}"
    return Doc(
        doc_id=f"invoice-draft:{inv.invoice_id}" if is_draft else f"invoice:{inv.invoice_id}",
        kind="invoice",
        title=f"INVOICE {inv.invoice_id}" + (" (DRAFT)" if is_draft else ""),
        stem=stem + ("_draft" if is_draft else ""),
        home=("finance", "invoices", slug(inv.client), str(inv.issue_date.year)),
        fields=[
            ("From", world.company),
            ("Bill to", inv.client),
            ("Invoice number", inv.invoice_id),
            ("Issue date", inv.issue_date.isoformat()),
            ("Due date", inv.due_date.isoformat()),
            ("PO number", inv.po_number),
            ("Status", "DRAFT - not sent to client; superseded by the issued invoice" if is_draft else "Issued"),
            ("Total due", money(total)),
        ],
        table=Table(
            ["Description", "Qty", "Unit price", "Amount"],
            [[d, str(q), money(u), money(q * u)] for d, q, u in items],
        ),
        paragraphs=[f"Payment terms: Net 30. Please reference {inv.invoice_id} on your remittance to {world.company}."],
        formats=FORM_FORMATS,
    )


def ledger_doc(
    world: World,
    payments: list[Payment],
    as_of: date,
    *,
    doc_id: str,
    stem: str,
    year: int = YEAR,
) -> Doc:
    rows = [
        [p.payment_id, p.date.isoformat(), p.client, p.invoice_id, plain_money(p.amount_cents), p.method]
        for p in sorted(payments, key=lambda p: (p.date, p.payment_id))
        if p.date <= as_of
    ]
    return Doc(
        doc_id=doc_id,
        kind="ledger",
        title=f"{world.company} - Accounts Receivable Payment Ledger {year}",
        stem=stem,
        home=("finance", "ledgers"),
        fields=[
            ("Entity", world.company),
            ("Fiscal year", str(year)),
            ("Ledger snapshot as of", as_of.isoformat()),
            ("Payments recorded", str(len(rows))),
        ],
        table=Table(["payment_id", "date", "client", "invoice_id", "amount", "method"], rows),
        paragraphs=["Each row is a customer payment applied to the invoice shown. Amounts are in USD."],
        formats=TABULAR_FORMATS,
    )


def employee_doc(world: World, e: Employee, *, superseded: bool = False, status: str = "Active") -> Doc:
    if superseded:
        title, manager, updated = e.prev_title, e.prev_manager, date(max(e.start_date.year, 2022), 3, 1)
        record = "SUPERSEDED - replaced by a newer HR record"
    else:
        title, manager, updated = e.title, e.manager, date(2026, 9, 15)
        record = "CURRENT"
    stem = f"{slug(e.last)}_{slug(e.first)}"
    return Doc(
        doc_id=f"employee-old:{e.employee_id}" if superseded else f"employee:{e.employee_id}",
        kind="employee",
        title=f"Employee Record - {e.name}",
        stem=stem + ("_old" if superseded else ""),
        home=("hr", "employees", slug(e.department)),
        fields=[
            ("Name", e.name),
            ("Employee ID", e.employee_id),
            ("Department", e.department),
            ("Job title", title),
            ("Manager", manager),
            ("Office", e.office),
            ("Email", e.email),
            ("Start date", e.start_date.isoformat()),
            ("Employment status", status),
            ("Record status", record),
            ("Last updated", updated.isoformat()),
        ],
        formats=FORM_FORMATS,
    )


CONTRACT_CLAUSES = [
    "1. Services. Provider will perform the data and analytics services described in each Statement of Work.",
    "2. Fees. Customer shall pay the Annual Contract Value stated above, invoiced monthly in arrears.",
    "3. Term and renewal. This Agreement begins on the Effective Date, continues for the Initial Term, "
    "and renews on the Renewal Date unless either party gives 60 days written notice.",
    "4. Confidentiality. Each party will protect the other party's confidential information with reasonable care.",
]


def contract_doc(
    world: World,
    c: Contract,
    *,
    doc_id: str,
    stem: str,
    status: str,
    signed_on: date | None,
) -> Doc:
    if signed_on:
        signatures = [
            ("Signed for customer", f"{c.client_signatory}, {signed_on.isoformat()}"),
            ("Signed for provider", f"{c.our_signatory}, {signed_on.isoformat()}"),
        ]
    else:
        signatures = [("Signed for customer", "(not signed)"), ("Signed for provider", "(not signed)")]
    return Doc(
        doc_id=doc_id,
        kind="contract",
        title=f"Master Services Agreement - {world.company} and {c.client}",
        stem=stem,
        home=("legal", "contracts", slug(c.client)),
        fields=[
            ("Agreement number", c.contract_id),
            ("Status", status),
            ("Customer", c.client),
            ("Effective date", c.effective_date.isoformat()),
            ("Initial term", f"{c.term_months} months"),
            ("Renewal date", c.renewal_date.isoformat()),
            ("Annual contract value", money(c.annual_value_cents)),
            ("Payment terms", c.payment_terms),
            *signatures,
        ],
        paragraphs=CONTRACT_CLAUSES,
        formats=PROSE_FORMATS,
    )


def quote_doc(world: World, client: str, quote_id: str, quote_date: date, items: list[tuple[str, int, int]]) -> Doc:
    total = sum(q * u for _, q, u in items)
    return Doc(
        doc_id=f"quote:{quote_id}",
        kind="quote",
        title=f"QUOTE {quote_id} - estimate, not an invoice",
        stem=f"quote_{slug(client)}_{quote_date:%Y-%m}_{quote_id}",
        home=("sales", "quotes", slug(client)),
        fields=[
            ("From", world.company),
            ("Prepared for", client),
            ("Quote number", quote_id),
            ("Quote date", quote_date.isoformat()),
            ("Valid for", "30 days"),
            ("Estimated total", money(total)),
        ],
        table=Table(
            ["Description", "Qty", "Unit price", "Amount"],
            [[d, str(q), money(u), money(q * u)] for d, q, u in items],
        ),
        paragraphs=["This quote is an estimate only and is not a request for payment."],
        formats=FORM_FORMATS,
    )
