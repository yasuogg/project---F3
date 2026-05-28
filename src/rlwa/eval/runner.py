"""Generic eval runner: agent x tasks x seeds -> EpisodeRecord list.

True parallelism via ProcessPoolExecutor: each worker process owns its own
Playwright browser and Gemini client. No greenlet/thread conflicts.
"""
from __future__ import annotations
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, Future
from pathlib import Path
from typing import List, Callable
from tqdm import tqdm

from rlwa.envs import make_env
from rlwa.utils.schemas import EpisodeRecord
from rlwa.utils.logging import JsonlWriter, ok, warn


# ── per-process state ──────────────────────────────────────────────────────────

_proc_agent = None


def _proc_init(agent_factory: Callable) -> None:
    global _proc_agent
    _proc_agent = agent_factory()


def _proc_run(
    task: str, seed: int, max_steps: int, headless: bool, env_kwargs: dict
) -> EpisodeRecord:
    env = make_env(task, seed=seed, headless=headless, max_steps=max_steps, **env_kwargs)
    try:
        return _proc_agent.run_episode(env, task=task, seed=seed, max_steps=max_steps)
    finally:
        env.close()


# ── public API ─────────────────────────────────────────────────────────────────

# Generous per-episode budget: 25 steps × 15s/step (Gemini + browser) + headroom.
_EPISODE_TIMEOUT = 480  # seconds


def run_eval(
    agent,
    tasks: List[str],
    seeds: List[int],
    max_steps: int = 25,
    headless: bool = True,
    out_jsonl: str | Path | None = None,
    env_kwargs: dict | None = None,
    workers: int = 1,
    agent_factory: Callable | None = None,
) -> List[EpisodeRecord]:
    """
    Evaluate an agent across tasks x seeds.

    workers=1  — single process, uses `agent` directly.
    workers>1  — spawns N processes via ProcessPoolExecutor. Each process calls
                 agent_factory() once to build its own agent+planner+browser.
                 agent_factory must be picklable (top-level class / functools.partial).

    Concurrency is capped at min(workers, len(jobs)).
    Episodes that don't complete within _EPISODE_TIMEOUT seconds are logged and
    skipped — hung/OOM workers never block the entire run.
    """
    if workers > 1 and agent_factory is None:
        warn("workers>1 requires agent_factory; falling back to workers=1")
        workers = 1

    env_kwargs = env_kwargs or {}
    jobs = [(t, s) for t in tasks for s in seeds]
    effective = min(workers, len(jobs)) if workers > 1 else 1

    eps: List[EpisodeRecord] = []
    writer = JsonlWriter(out_jsonl, mode="w") if out_jsonl else None
    pbar = tqdm(total=len(jobs), desc=f"eval(x{effective})")

    if effective <= 1:
        for t, s in jobs:
            env = make_env(t, seed=s, headless=headless, max_steps=max_steps, **env_kwargs)
            try:
                ep = agent.run_episode(env, task=t, seed=s, max_steps=max_steps)
            finally:
                env.close()
            eps.append(ep)
            if writer:
                writer.write(ep)
            pbar.update(1)
            pbar.set_postfix(task=t.split(".")[-1], succ=int(ep.success))
    else:
        with ProcessPoolExecutor(
            max_workers=effective,
            initializer=_proc_init,
            initargs=(agent_factory,),
        ) as pool:
            # Submit all jobs and record when each one started.
            start_time: dict[Future, float] = {}
            futs: dict[Future, tuple[str, int]] = {}
            for t, s in jobs:
                f = pool.submit(_proc_run, t, s, max_steps, headless, env_kwargs)
                futs[f] = (t, s)
                start_time[f] = time.monotonic()

            pending: set[Future] = set(futs)

            # Poll loop: wait up to 5s for any completion, then expire overdue futures.
            while pending:
                done, pending = wait(pending, timeout=5.0, return_when=FIRST_COMPLETED)

                # Process futures that actually finished.
                for fut in done:
                    t, s = futs[fut]
                    try:
                        ep = fut.result()
                    except Exception as e:
                        pbar.write(f"! {t}/{s} failed: {e}")
                        pbar.update(1)
                        continue
                    eps.append(ep)
                    if writer:
                        writer.write(ep)
                    pbar.update(1)
                    pbar.set_postfix(task=t.split(".")[-1], succ=int(ep.success))

                # Expire futures that have been running too long (OOM / hung browser).
                now = time.monotonic()
                expired = {f for f in pending if now - start_time[f] > _EPISODE_TIMEOUT}
                for fut in expired:
                    t, s = futs[fut]
                    pbar.write(
                        f"! {t}/{s} timed out after {_EPISODE_TIMEOUT}s "
                        f"(hung/OOM worker) — skipping"
                    )
                    pbar.update(1)
                    pending.discard(fut)
                    fut.cancel()  # no-op if already running; marks future as cancelled

    pbar.close()
    if writer:
        writer.close()
        ok(f"Wrote {len(eps)} episodes -> {out_jsonl}")
    return eps
