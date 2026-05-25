"""Generic eval runner: agent x tasks x seeds -> EpisodeRecord list. Parallel-capable."""
from __future__ import annotations
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from rlwa.envs import make_env
from rlwa.utils.schemas import EpisodeRecord
from rlwa.utils.logging import JsonlWriter, ok


def _run_one(agent, task, seed, max_steps, headless, env_kwargs):
    env = make_env(task, seed=seed, headless=headless, max_steps=max_steps, **env_kwargs)
    try:
        return agent.run_episode(env, task=task, seed=seed, max_steps=max_steps)
    finally:
        env.close()


def run_eval(
    agent,
    tasks: List[str],
    seeds: List[int],
    max_steps: int = 25,
    headless: bool = True,
    out_jsonl: str | Path | None = None,
    env_kwargs: dict | None = None,
    workers: int = 1,
) -> List[EpisodeRecord]:
    env_kwargs = env_kwargs or {}
    eps: List[EpisodeRecord] = []
    writer = JsonlWriter(out_jsonl, mode="w") if out_jsonl else None
    lock = Lock()
    jobs = [(t, s) for t in tasks for s in seeds]
    pbar = tqdm(total=len(jobs), desc=f"eval(x{workers})")

    if workers <= 1:
        for t, s in jobs:
            ep = _run_one(agent, t, s, max_steps, headless, env_kwargs)
            eps.append(ep)
            if writer: writer.write(ep)
            pbar.update(1); pbar.set_postfix(task=t.split(".")[-1], succ=int(ep.success))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, agent, t, s, max_steps, headless, env_kwargs): (t, s)
                    for t, s in jobs}
            for fut in as_completed(futs):
                t, s = futs[fut]
                try:
                    ep = fut.result()
                except Exception as e:
                    pbar.write(f"! {t}/{s} failed: {e}"); pbar.update(1); continue
                eps.append(ep)
                if writer:
                    with lock: writer.write(ep)
                pbar.update(1); pbar.set_postfix(task=t.split(".")[-1], succ=int(ep.success))
    pbar.close()
    if writer:
        writer.close()
        ok(f"Wrote {len(eps)} episodes -> {out_jsonl}")
    return eps
