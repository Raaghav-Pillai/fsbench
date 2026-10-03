"""Ground-truth world: the facts every filesystem condition is built from.

The world depends only on ``world_seed``. Tasks read their answers from it, and
the layout stage only decides where (and in what form) the facts land on disk,
so the ground truth is identical across every filesystem condition.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta

from fsbench.docs import slug
from fsbench.rng import keyed_rng

TODAY = date(2026, 10, 1)
YEAR = 2026

COMPANY_NAMES = [
    "Northwind Analytics",
    "Bluepeak Consulting",
    "Harbor & Finch Design",
    "Lumen Data Works",
    "Copperline Systems",
]

CLIENT_NAMES = [
    "Acme Corp",
    "Globex Industries",
    "Initech",
    "Umbrella Logistics",
    "Hooli Systems",
    "Stark Fabrication",
    "Wayne Analytics",
    "Cyberdyne Labs",
    "Soylent Foods",
    "Tyrell Robotics",
    "Massive Dynamic",
    "Wonka Confections",
]

FIRST_NAMES = [
    "Sarah", "James", "Priya", "Miguel", "Aisha", "Daniel", "Mei", "Oliver", "Fatima", "Lucas",
    "Hannah", "Kenji", "Elena", "Marcus", "Zoe", "Ravi", "Grace", "Tomas", "Leila", "Noah",
    "Chloe", "Arjun", "Ingrid", "Samuel", "Yuki", "Diego", "Amara", "Felix", "Nadia", "Owen",
]

LAST_NAMES = [
    "Chen", "Patel", "Garcia", "Okafor", "Nguyen", "Schmidt", "Kowalski", "Haddad", "Silva", "Murphy",
    "Tanaka", "Ivanova", "Johnson", "Rossi", "Kim", "Andersen", "Mendez", "Brooks", "Larsen", "Ahmed",
    "Fischer", "Novak", "Reyes", "Walsh", "Moreau", "Singh", "Costa", "Bauer", "Lindqvist", "Osei",
]

DEPARTMENTS = {
    "Engineering": ["Software Engineer I", "Software Engineer II", "Senior Software Engineer", "Staff Software Engineer"],
    "Sales": ["Sales Associate", "Account Executive", "Senior Account Executive", "Sales Director"],
    "Finance": ["Financial Analyst", "Senior Financial Analyst", "Finance Manager", "Controller"],
    "Marketing": ["Marketing Coordinator", "Marketing Specialist", "Senior Marketing Manager", "Head of Marketing"],
    "Operations": ["Operations Associate", "Operations Analyst", "Operations Manager", "Director of Operations"],
    "Support": ["Support Specialist", "Senior Support Specialist", "Support Team Lead", "Support Manager"],
}

OFFICES = ["Austin", "Chicago", "Denver", "Toronto", "Boston", "Remote"]

ITEM_CATALOG = [
    ("Consulting services (hours)", 9500, 21000, 4, 60),
    ("Data pipeline maintenance (monthly)", 150000, 600000, 1, 1),
    ("Dashboard development", 200000, 900000, 1, 1),
    ("Training workshop (seats)", 25000, 60000, 2, 20),
    ("Cloud hosting pass-through", 30000, 250000, 1, 1),
    ("Data quality audit", 300000, 1200000, 1, 1),
    ("Support retainer (monthly)", 80000, 300000, 1, 1),
]

PAYMENT_METHODS = ["ACH", "Wire", "Check", "Credit card"]


@dataclass
class Invoice:
    invoice_id: str
    client: str
    issue_date: date
    due_date: date
    po_number: str
    items: list[tuple[str, int, int]]

    @property
    def total_cents(self) -> int:
        return sum(qty * unit for _, qty, unit in self.items)

    @property
    def quarter(self) -> int:
        return (self.issue_date.month - 1) // 3 + 1


@dataclass
class Payment:
    payment_id: str
    date: date
    client: str
    invoice_id: str
    amount_cents: int
    method: str


@dataclass
class Employee:
    employee_id: str
    first: str
    last: str
    department: str
    title: str
    prev_title: str
    manager: str = ""
    prev_manager: str = ""
    start_date: date = date(2020, 1, 1)
    office: str = "Remote"
    email: str = ""

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


@dataclass
class Contract:
    contract_id: str
    client: str
    effective_date: date
    term_months: int
    annual_value_cents: int
    payment_terms: str
    client_signatory: str
    our_signatory: str

    @property
    def renewal_date(self) -> date:
        return add_months(self.effective_date, self.term_months)


@dataclass
class World:
    seed: int
    company: str
    domain: str
    clients: list[str]
    invoices: list[Invoice] = field(default_factory=list)
    payments: list[Payment] = field(default_factory=list)
    employees: list[Employee] = field(default_factory=list)
    contracts: dict[str, Contract] = field(default_factory=dict)
    prior_contracts: dict[str, Contract] = field(default_factory=dict)

    def invoices_for(self, client: str, quarter: int) -> list[Invoice]:
        return [i for i in self.invoices if i.client == client and i.quarter == quarter]

    def paid_cents(self, invoice_id: str, as_of: date | None = None) -> int:
        return sum(
            p.amount_cents
            for p in self.payments
            if p.invoice_id == invoice_id and (as_of is None or p.date <= as_of)
        )

    def payments_for(self, invoice_ids: set[str]) -> list[Payment]:
        return [p for p in self.payments if p.invoice_id in invoice_ids]


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def build_world(seed: int) -> World:
    rng = keyed_rng("world", seed)
    company = rng.choice(COMPANY_NAMES)
    world = World(
        seed=seed,
        company=company,
        domain=slug(company).replace("-", "") + ".com",
        clients=rng.sample(CLIENT_NAMES, 8),
    )
    _build_invoices(world, keyed_rng("world", seed, "invoices"))
    _build_payments(world, keyed_rng("world", seed, "payments"))
    _build_employees(world, keyed_rng("world", seed, "employees"))
    _build_contracts(world, keyed_rng("world", seed, "contracts"))
    return world


def _build_invoices(world: World, rng) -> None:
    drafts = []
    for client in world.clients:
        for month in range(1, 10):
            n = rng.choices([0, 1, 2], weights=[20, 55, 25])[0]
            for _ in range(n):
                issue = date(YEAR, month, rng.randint(1, 28))
                items = []
                for desc, lo, hi, qlo, qhi in rng.sample(ITEM_CATALOG, rng.randint(1, 3)):
                    unit = rng.randint(lo // 100, hi // 100) * 100 + rng.choice([0, 0, 50, 25, 75])
                    items.append((desc, rng.randint(qlo, qhi), unit))
                drafts.append((issue, client, items))
    drafts.sort(key=lambda d: (d[0], d[1]))
    base = rng.randint(1100, 4800)
    for i, (issue, client, items) in enumerate(drafts):
        world.invoices.append(
            Invoice(
                invoice_id=f"INV-{base + i}",
                client=client,
                issue_date=issue,
                due_date=issue + timedelta(days=30),
                po_number=f"PO-{rng.randint(10000, 99999)}",
                items=items,
            )
        )


def _build_payments(world: World, rng) -> None:
    raw: list[tuple[date, str, str, int, str]] = []
    for inv in world.invoices:
        total = inv.total_cents
        method = rng.choice(PAYMENT_METHODS)
        r = rng.random()
        if r < 0.55:
            parts = [(inv.issue_date + timedelta(days=rng.randint(7, 75)), total)]
        elif r < 0.70:
            first = (total * rng.randint(40, 60) // 100) // 100 * 100
            d1 = inv.issue_date + timedelta(days=rng.randint(7, 40))
            parts = [(d1, first), (d1 + timedelta(days=rng.randint(10, 40)), total - first)]
        elif r < 0.85:
            amount = (total * rng.randint(25, 75) // 100) // 100 * 100
            parts = [(inv.issue_date + timedelta(days=rng.randint(7, 60)), amount)]
        else:
            parts = []
        for d, amount in parts:
            if d < TODAY and amount > 0:
                raw.append((d, inv.client, inv.invoice_id, amount, method))
    raw.sort(key=lambda p: (p[0], p[2]))
    base = rng.randint(20000, 70000)
    for i, (d, client, inv_id, amount, method) in enumerate(raw):
        world.payments.append(Payment(f"PAY-{base + i}", d, client, inv_id, amount, method))


def _build_employees(world: World, rng) -> None:
    names = rng.sample([(f, l) for f in FIRST_NAMES for l in LAST_NAMES], 60)
    ids = rng.sample(range(10000, 99999), 60)
    for (first, last), eid in zip(names, ids):
        dept = rng.choice(list(DEPARTMENTS))
        level = rng.randint(1, 3)
        titles = DEPARTMENTS[dept]
        world.employees.append(
            Employee(
                employee_id=f"E-{eid}",
                first=first,
                last=last,
                department=dept,
                title=titles[level],
                prev_title=titles[level - 1],
                start_date=date(2016, 1, 1) + timedelta(days=rng.randint(0, 3800)),
                office=rng.choice(OFFICES),
                email=f"{first.lower()}.{last.lower()}@{world.domain}",
            )
        )
    for e in world.employees:
        peers = [p.name for p in world.employees if p is not e]
        same_dept = [p.name for p in world.employees if p is not e and p.department == e.department]
        e.manager = rng.choice(same_dept or peers)
        e.prev_manager = rng.choice([n for n in peers if n != e.manager])


def _build_contracts(world: World, rng) -> None:
    numbers = iter(rng.sample(range(100, 1000), 2 * len(world.clients)))
    for client in world.clients:
        term = rng.choice([12, 24, 36])
        latest = TODAY - timedelta(days=30)
        earliest = add_months(TODAY, -term) + timedelta(days=30)
        effective = earliest + timedelta(days=rng.randint(0, (latest - earliest).days))
        signer = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        ours = rng.choice(world.employees).name
        world.contracts[client] = Contract(
            contract_id=f"MSA-{effective.year}-{next(numbers)}",
            client=client,
            effective_date=effective,
            term_months=term,
            annual_value_cents=rng.randint(240, 4800) * 10000,
            payment_terms=rng.choice(["Net 30", "Net 45", "Net 60"]),
            client_signatory=signer,
            our_signatory=ours,
        )
        prior_term = rng.choice([12, 24])
        prior_start = add_months(effective, -prior_term)
        world.prior_contracts[client] = Contract(
            contract_id=f"MSA-{prior_start.year}-{next(numbers)}",
            client=client,
            effective_date=prior_start,
            term_months=prior_term,
            annual_value_cents=rng.randint(240, 4800) * 10000,
            payment_terms=rng.choice(["Net 30", "Net 45"]),
            client_signatory=signer,
            our_signatory=ours,
        )
