"""Keyed random streams.

Every random decision in layout generation draws from a stream keyed on
(seed, decision type, entity id) instead of one shared generator. Changing one
knob therefore never reshuffles unrelated decisions, and threshold decisions
(``u < rate``) are monotone: the set of noisy filenames at rate 0.6 is a superset
of the set at rate 0.3 for the same seed. This is what lets a sweep vary one
filesystem property while holding everything else fixed.
"""

from __future__ import annotations

import hashlib
import random


def keyed_rng(*keys: object) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, keys)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def keyed_uniform(*keys: object) -> float:
    return keyed_rng(*keys).random()
