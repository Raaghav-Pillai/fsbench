"""Line plots of success, calls, tokens, cost, and latency against one filesystem variable."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from fsbench.evaluate import mean_se

PLOT_METRICS = [
    ("success", "Success"),
    ("n_calls", "Tool calls"),
    ("total_tokens", "Total tokens"),
    ("cost_usd", "API cost (USD)"),
    ("wall_time_s", "Wall time (s)"),
    ("read_precision", "Read precision"),
    ("steps_to_first_required_evidence", "Steps to first required evidence"),
    ("candidate_files_seen", "Candidate files seen before required evidence"),
]


def _series_key(row: dict) -> str:
    parts = [row.get("agent") or "?", row.get("toolset") or "?"]
    if row.get("model"):
        parts[0] = row["model"]
    return " / ".join(parts)


def _x_value(row: dict, x: str):
    raw = row.get(f"cfg.{x}")
    if raw is None:
        vars_ = row.get("sweep.vars") or row.get("condition") or ""
        prefix = f"{x}="
        for part in str(vars_).split(";"):
            if part.startswith(prefix):
                raw = part[len(prefix):]
                break
    if raw is None:
        return None
    try:
        value = float(raw)
        return int(value) if value.is_integer() else value
    except (TypeError, ValueError):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw


def plot_runs(rows: list[dict], x: str, out_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install fsbench[plot]") from e

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        xv = _x_value(r, x)
        if xv is None:
            continue
        for metric, _ in PLOT_METRICS:
            if r.get(metric) is not None:
                grouped[(_series_key(r), metric, xv)].append(float(r[metric]))
    if not grouped:
        raise RuntimeError(f"no rows had a value for {x!r}")

    for metric, title in PLOT_METRICS:
        series: dict[str, list[tuple]] = defaultdict(list)
        for (sk, m, xv), vals in grouped.items():
            if m != metric:
                continue
            mu, se = mean_se(vals)
            series[sk].append((xv, mu, se))
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for sk, pts in sorted(series.items()):
            pts = sorted(pts, key=lambda p: (isinstance(p[0], str), p[0]))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            err = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=err, marker="o", capsize=3, label=sk)
        ax.set_xlabel(x)
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs {x}")
        ax.legend()
        fig.tight_layout()
        path = out / f"{metric}_vs_{x}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written
