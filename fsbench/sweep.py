"""Generate experiment matrices.

``ofat`` (one factor at a time) varies each knob alone from the base config;
``grid`` crosses every listed value (use it for interaction effects such as
depth x filename_noise). Replicate ``r`` uses seed ``seed_offset + r`` for the
world, task, and layout, so every condition within a replicate shares the same
facts, the same question, and the same random draws.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

from fsbench.config import FSConfig
from fsbench.generate import generate_env
from fsbench.paths import long_path


def conditions(base: FSConfig, vary: dict[str, list], mode: str = "ofat") -> list[tuple[dict, FSConfig]]:
    if mode == "ofat":
        out = []
        for key, values in vary.items():
            for v in values:
                out.append(({key: v}, base.with_value(key, v)))
        return out
    if mode == "grid":
        keys = list(vary)
        out = []
        for combo in itertools.product(*(vary[k] for k in keys)):
            cfg = base
            for k, v in zip(keys, combo):
                cfg = cfg.with_value(k, v)
            out.append((dict(zip(keys, combo)), cfg))
        return out
    raise ValueError(f"unknown sweep mode {mode!r}")


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return "+".join(map(str, v))
    return str(v)


def run_sweep(
    task_types: list[str],
    base: FSConfig,
    vary: dict[str, list],
    out_root: str | Path,
    *,
    replicates: int = 5,
    mode: str = "ofat",
    seed_offset: int = 0,
    progress=None,
) -> list[dict]:
    out_root = Path(out_root)
    long_path(out_root).mkdir(parents=True, exist_ok=True)
    conds = conditions(base, vary, mode)
    index = []
    total = len(task_types) * len(conds) * replicates
    for task in task_types:
        for cond, cfg in conds:
            for r in range(replicates):
                seed = seed_offset + r
                label = "__".join(f"{k}={_fmt(v)}" for k, v in cond.items())
                env_id = f"{task}__{label}__r{r}"
                m = generate_env(task, cfg, out_root / env_id, world_seed=seed, task_seed=seed,
                                 layout_seed=seed, env_id=env_id,
                                 sweep={"mode": mode, "vars": {k: _fmt(v) for k, v in cond.items()},
                                        "expected_values": {k: [_fmt(v) for v in vary[k]] for k in cond},
                                        "replicate": r})
                index.append({
                    "env_id": env_id, "task_type": task, "replicate": r, "seed": seed,
                    **{f"var.{k}": _fmt(v) for k, v in cond.items()},
                    "n_files": m["stats"]["n_files"], "n_dirs": m["stats"]["n_dirs"],
                    "max_depth": m["stats"]["max_depth"],
                })
                if progress:
                    progress(len(index), total, env_id)
    keys = list(dict.fromkeys(k for row in index for k in row))
    with open(long_path(out_root / "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(index)
    return index


def run_profiles(task_types, profiles, out_root, *, replicates=5, seed_offset=0, overrides=()):
    """Compare named profiles on paired worlds without adding a filesystem variable."""
    from fsbench.config import PRESETS
    out_root = Path(out_root)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    envs = []
    for task in task_types:
        for name in profiles:
            cfg = PRESETS[name]
            for key, value in overrides:
                cfg = cfg.with_value(key, value)
            for r in range(replicates):
                seed = seed_offset + r
                env = out_root / f"{task}__{name}__r{r}"
                generate_env(task, cfg, env, world_seed=seed, task_seed=seed, layout_seed=seed,
                    label=name, sweep={"mode": "profiles", "replicate": r})
                envs.append(env)
    from fsbench.difficulty import verify_composed
    verify_composed(envs)
    return envs
