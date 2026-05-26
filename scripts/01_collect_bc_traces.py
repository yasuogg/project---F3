"""Collect BC traces by running prompt-only across all tasks.

True parallelism via ProcessPoolExecutor: each worker process owns its own
Playwright browser and Gemini client. workers=32 actually means 32 browsers.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import OmegaConf
from tqdm import tqdm
import tyro

from rlwa.utils.logging import JsonlWriter, info, ok


# ── per-process state ──────────────────────────────────────────────────────────

_proc_agent = None


def _proc_init(agent_cfg: str, async_workers: int) -> None:
    global _proc_agent
    from omegaconf import OmegaConf
    from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
    from rlwa.agents import PromptOnlyAgent
    ac = OmegaConf.load(agent_cfg)
    planner = AsyncGeminiPlanner(GeminiPlanner(ac), max_workers=async_workers)
    _proc_agent = PromptOnlyAgent(ac, planner=planner)


def _proc_run(task: str, seed: int, max_steps: int, headless: bool):
    from rlwa.envs import make_env
    env = make_env(task, seed=seed, headless=headless, max_steps=max_steps)
    try:
        return _proc_agent.run_episode(env, task=task, seed=seed, max_steps=max_steps)
    finally:
        env.close()


# ── main ───────────────────────────────────────────────────────────────────────

def main(
    agent_cfg: str = "configs/agent/prompt_only.yaml",
    env_cfg: str = "configs/env/miniwob.yaml",
    train_cfg: str = "configs/train/ppo.yaml",
    episodes_per_task: int = 100,
    out: str = "data/traces/bc.jsonl",
    max_steps: int = 25,
    headless: bool = True,
    workers: int = 32,
    planner_workers: int = 96,
):
    ec = OmegaConf.load(env_cfg)
    tc = OmegaConf.load(train_cfg)
    episodes_per_task = episodes_per_task or int(tc.trace_collection.episodes_per_task)
    out = out or tc.trace_collection.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    jobs = [(task, seed) for task in ec.tasks for seed in range(episodes_per_task)]
    # planner threads split evenly across browser processes
    async_per_worker = max(1, planner_workers // max(workers, 1))

    info(f"Launching {len(jobs)} episodes across {workers} parallel workers "
         f"({async_per_worker} Gemini threads each)")

    n_total = 0
    n_succ = 0
    writer = JsonlWriter(out, mode="w")

    if workers <= 1:
        _proc_init(agent_cfg, planner_workers)
        for t, s in tqdm(jobs, desc="bc"):
            try:
                ep = _proc_run(t, s, max_steps, headless)
            except Exception as e:
                info(f"  ! {t}/{s} failed: {e}")
                continue
            n_total += 1
            if ep.success:
                n_succ += 1
                writer.write(ep)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_proc_init,
            initargs=(agent_cfg, async_per_worker),
        ) as pool:
            futs = {
                pool.submit(_proc_run, t, s, max_steps, headless): (t, s)
                for t, s in jobs
            }
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="bc"):
                t, s = futs[fut]
                try:
                    ep = fut.result()
                except Exception as e:
                    info(f"  ! {t}/{s} failed: {e}")
                    continue
                n_total += 1
                if ep.success:
                    n_succ += 1
                    writer.write(ep)
                if n_total % 20 == 0:
                    info(f"  {n_total}/{len(jobs)}  succ-so-far={n_succ}")

    writer.close()
    ok(f"Collected {n_succ}/{n_total} successful episodes -> {out}")


if __name__ == "__main__":
    tyro.cli(main)
