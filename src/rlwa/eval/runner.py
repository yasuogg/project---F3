"""Generic eval runner: agent x tasks x seeds -> EpisodeRecord list.

True parallelism via ProcessPoolExecutor: each worker process owns its own
Playwright browser and Gemini client. No greenlet/thread conflicts.

Usage:
    # single-process (debug)
    run_eval(agent, tasks, seeds, workers=1)

    # multi-process (production)
    run_eval(None, tasks, seeds, workers=8, agent_factory=MyFactory())
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
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

# Seconds to wait for a single episode before declaring it hung.
# 25 steps × 15s/step (Gemini + browser) = 375s; 480s gives comfortable headroom.
_EPISODE_TIMEOUT = 480


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
    workers>1  — spawns N processes; each calls agent_factory() once to build
                 its own agent+planner+browser. `agent` is unused in this path.
                 agent_factory must be picklable (top-level class or functools.partial).

    Concurrency is capped at min(workers, len(jobs)) so we never launch more
    browsers than there are episodes. A per-episode timeout (_EPISODE_TIMEOUT s)
    prevents hung OOM-killed workers from blocking the run forever.
    """
    if workers > 1 and agent_factory is None:
        warn("workers>1 requires agent_factory; falling back to workers=1")
        workers = 1

    env_kwargs = env_kwargs or {}
    jobs = [(t, s) for t in tasks for s in seeds]

    # Never spawn more processes than there are jobs — wastes memory/browsers.
    effective_workers = min(workers, len(jobs)) if workers > 1 else 1

    eps: List[EpisodeRecord] = []
    writer = JsonlWriter(out_jsonl, mode="w") if out_jsonl else None
    pbar = tqdm(total=len(jobs), desc=f"eval(x{effective_workers})")

    if effective_workers <= 1:
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
            max_workers=effective_workers,
            initializer=_proc_init,
            initargs=(agent_factory,),
        ) as pool:
            futs: dict[Future, tuple[str, int]] = {
                pool.submit(_proc_run, t, s, max_steps, headless, env_kwargs): (t, s)
                for t, s in jobs
            }
            for fut in as_completed(futs):
                t, s = futs[fut]
                try:
                    ep = fut.result(timeout=_EPISODE_TIMEOUT)
                except TimeoutError:
                    pbar.write(
                        f"! {t}/{s} timed out after {_EPISODE_TIMEOUT}s "
                        f"(OOM or hung browser) — skipping"
                    )
                    pbar.update(1)
                    continue
                except Exception as e:
                    pbar.write(f"! {t}/{s} failed: {e}")
                    pbar.update(1)
                    continue
                eps.append(ep)
                if writer:
                    writer.write(ep)
                pbar.update(1)
                pbar.set_postfix(task=t.split(".")[-1], succ=int(ep.success))

    pbar.close()
    if writer:
        writer.close()
        ok(f"Wrote {len(eps)} episodes -> {out_jsonl}")
    return eps
