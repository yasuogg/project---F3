"""Metrics aggregation over EpisodeRecords."""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Dict, Any
import math
from rlwa.utils.schemas import EpisodeRecord


def per_episode_metrics(ep: EpisodeRecord) -> dict:
    n = max(1, ep.n_steps)
    had_failure = ep.n_invalid > 0
    return {
        "task": ep.task,
        "seed": ep.seed,
        "success": float(ep.success),
        "n_steps": ep.n_steps,
        "invalid_rate": ep.n_invalid / n,
        "had_failure": int(had_failure),
        "recovered": int(ep.success and had_failure),
        "n_recovered_steps": ep.n_recovered,
        "total_reward": ep.total_reward,
    }


def aggregate(eps: Iterable[EpisodeRecord]) -> Dict[str, Any]:
    eps = list(eps)
    if not eps:
        return {}
    per = [per_episode_metrics(e) for e in eps]

    def avg(key):
        vs = [p[key] for p in per]
        return sum(vs) / len(vs)

    by_task = defaultdict(list)
    for p in per:
        by_task[p["task"]].append(p)
    task_table = {}
    for t, rows in by_task.items():
        s = [r["success"] for r in rows]
        task_table[t] = {
            "SR": sum(s) / len(s),
            "n": len(rows),
            "mean_steps": sum(r["n_steps"] for r in rows) / len(rows),
            "invalid_rate": sum(r["invalid_rate"] for r in rows) / len(rows),
        }

    # recovery rate: P(success | had_failure)
    had_fail = [p for p in per if p["had_failure"]]
    rr = (sum(p["success"] for p in had_fail) / len(had_fail)) if had_fail else float("nan")

    return {
        "n_episodes": len(per),
        "success_rate": avg("success"),
        "mean_steps": avg("n_steps"),
        "invalid_rate": avg("invalid_rate"),
        "recovery_rate": rr,
        "mean_reward": avg("total_reward"),
        "by_task": task_table,
        "per_episode": per,
    }


def bootstrap_ci(successes: list[float], n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    """95% bootstrap CI for the mean."""
    import random
    n = len(successes)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        sample = [successes[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def paired_bootstrap_pvalue(
    a: list[float], b: list[float], n_boot: int = 10_000
) -> tuple[float, float]:
    """Two-sided paired bootstrap p-value for H0: mean(a) == mean(b).
    Returns (mean_diff, p_value). `a` and `b` must be paired and same length.
    """
    import random
    assert len(a) == len(b), "paired_bootstrap requires equal length lists"
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    diffs = [ai - bi for ai, bi in zip(a, b)]
    obs = sum(diffs) / n
    # center the distribution under H0 by subtracting the observed mean
    centered = [d - obs for d in diffs]
    extreme = 0
    for _ in range(n_boot):
        boot_mean = sum(centered[random.randrange(n)] for _ in range(n)) / n
        if abs(boot_mean) >= abs(obs):
            extreme += 1
    return (obs, (extreme + 1) / (n_boot + 1))
