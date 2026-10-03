"""Generic (topically unrelated) distractor documents.

``generic_doc(world_seed, i)`` is a pure function of its arguments, so distractor
number ``i`` is the same document in every condition that includes it.
"""

from __future__ import annotations

from datetime import date, timedelta

from fsbench.docs import PROSE_FORMATS, TABULAR_FORMATS, Doc, Table, slug
from fsbench.rng import keyed_rng
from fsbench.world import FIRST_NAMES, LAST_NAMES

TOPICS = [
    "quarterly planning", "office move", "hiring pipeline", "security training", "vendor onboarding",
    "website refresh", "offsite agenda", "laptop refresh", "benefits enrollment", "data retention policy",
]

SENTENCES = [
    "The team agreed to revisit the timeline next week.",
    "Action item: circulate the updated draft before Friday.",
    "Budget questions were deferred to the next review.",
    "Several attendees raised concerns about scope creep.",
    "We will pilot the new process with one team first.",
    "The previous approach was retired after feedback.",
    "Follow-up owners were assigned for each open question.",
    "No decision was made; more data is needed.",
    "The proposal was approved with minor changes.",
    "Risks were logged in the shared tracker.",
]

RECIPES = [
    ("Lemon garlic pasta", ["spaghetti", "lemon", "garlic", "parmesan", "olive oil"]),
    ("Chickpea curry", ["chickpeas", "coconut milk", "onion", "curry paste", "spinach"]),
    ("Banana bread", ["bananas", "flour", "butter", "sugar", "eggs"]),
    ("Black bean tacos", ["black beans", "tortillas", "lime", "cilantro", "avocado"]),
    ("Tomato soup", ["tomatoes", "onion", "stock", "basil", "cream"]),
]

SERVICES = ["auth-api", "billing-worker", "etl-runner", "web-frontend", "search-indexer"]
LOG_MESSAGES = [
    "request completed", "cache miss", "retrying connection", "job finished", "slow query detected",
    "health check ok", "token refreshed", "queue depth high",
]
DEVICES = ["ThinkPad X1", "MacBook Pro 14", "Dell U2723 monitor", "Logitech MX Keys", "iPhone 15"]
CITIES = ["Lisbon", "Seattle", "Montreal", "Singapore", "Berlin", "Nashville"]


def _person(rng) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _day(rng) -> date:
    return date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))


def _meeting(i, rng) -> Doc:
    topic, d = rng.choice(TOPICS), _day(rng)
    return Doc(
        f"generic:{i}", "meeting_notes", f"Meeting notes - {topic} - {d}",
        f"meeting_notes_{d}_{slug(topic)}", ("admin", "meetings"),
        fields=[("Date", d.isoformat()), ("Attendees", ", ".join(_person(rng) for _ in range(rng.randint(2, 5))))],
        paragraphs=rng.sample(SENTENCES, rng.randint(3, 6)), formats=PROSE_FORMATS,
    )


def _recipe(i, rng) -> Doc:
    name, ingredients = rng.choice(RECIPES)
    return Doc(
        f"generic:{i}", "recipe", name, f"recipe_{slug(name)}", ("personal", "recipes"),
        fields=[("Serves", str(rng.randint(2, 6))), ("Prep time", f"{rng.randint(10, 60)} minutes")],
        paragraphs=["Ingredients: " + ", ".join(ingredients), "Combine, cook, and season to taste."],
        formats=PROSE_FORMATS,
    )


def _server_log(i, rng) -> Doc:
    svc, d = rng.choice(SERVICES), _day(rng)
    rows = [
        [f"{d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z",
         rng.choice(["INFO", "INFO", "WARN", "ERROR"]), svc, rng.choice(LOG_MESSAGES)]
        for _ in range(rng.randint(10, 40))
    ]
    rows.sort()
    return Doc(
        f"generic:{i}", "server_log", f"{svc} log export {d}", f"{svc}_{d}", ("engineering", "logs"),
        table=Table(["timestamp", "level", "service", "message"], rows),
        formats=("txt", "csv", "json"),
    )


def _inventory(i, rng) -> Doc:
    rows = [
        [f"AT-{rng.randint(1000, 9999)}", rng.choice(DEVICES), _person(rng), _day(rng).isoformat()]
        for _ in range(rng.randint(8, 25))
    ]
    return Doc(
        f"generic:{i}", "it_inventory", "IT asset inventory", f"it_assets_{rng.randint(1, 40):02d}",
        ("it", "inventory"), table=Table(["asset_tag", "device", "assigned_to", "purchased"], rows),
        formats=TABULAR_FORMATS,
    )


def _analytics(i, rng) -> Doc:
    start = _day(rng)
    rows = [
        [(start + timedelta(days=k)).isoformat(), str(rng.randint(200, 9000)), f"{rng.uniform(20, 70):.1f}%"]
        for k in range(rng.randint(7, 30))
    ]
    return Doc(
        f"generic:{i}", "web_analytics", f"Website traffic report from {start}", f"web_traffic_{start}",
        ("marketing", "analytics"), table=Table(["date", "sessions", "bounce_rate"], rows),
        formats=TABULAR_FORMATS,
    )


def _travel(i, rng) -> Doc:
    city, d = rng.choice(CITIES), _day(rng)
    return Doc(
        f"generic:{i}", "travel", f"Travel itinerary - {city}", f"itinerary_{slug(city)}_{d}", ("admin", "travel"),
        fields=[("Traveler", _person(rng)), ("Depart", d.isoformat()),
                ("Return", (d + timedelta(days=rng.randint(2, 6))).isoformat()),
                ("Hotel", f"{rng.choice(['Grand', 'Harbor', 'Central', 'Park'])} Hotel {city}")],
        formats=PROSE_FORMATS,
    )


def _project_update(i, rng) -> Doc:
    topic, d = rng.choice(TOPICS), _day(rng)
    return Doc(
        f"generic:{i}", "project_update", f"Project status: {topic}", f"status_{slug(topic)}_{d}",
        ("engineering", "projects"),
        fields=[("Owner", _person(rng)), ("Week of", d.isoformat()),
                ("Health", rng.choice(["Green", "Yellow", "Red"]))],
        paragraphs=rng.sample(SENTENCES, rng.randint(2, 4)), formats=PROSE_FORMATS,
    )


_MAKERS = [_meeting, _recipe, _server_log, _inventory, _analytics, _travel, _project_update]


def generic_doc(world_seed: int, i: int) -> Doc:
    rng = keyed_rng("generic", world_seed, i)
    doc = rng.choice(_MAKERS)(i, rng)
    doc.stem = f"{doc.stem}_{i}" if rng.random() < 0.3 else doc.stem
    return doc
