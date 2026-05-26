"""Collect BC traces by running prompt-only across all tasks. Parallel by default."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from omegaconf import OmegaConf
from tqdm import tqdm
import tyro

from rlwa.envs import make_env
from rlwa.agents import PromptOnlyAgent
from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
from rlwa.utils.logging import JsonlWriter, info, ok


def _run_one(task, seed, agent, max_steps, headless):
    env = make_env(task, seed=seed, headless=headless, max_steps=max_steps)
    try:
        return agent.run_episode(env, task=task, seed=seed, max_steps=max_steps)
    finally:
        env.close()


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
    ac = OmegaConf.load(agent_cfg)
    ec = OmegaConf.load(env_cfg)
    tc = OmegaConf.load(train_cfg)
    episodes_per_task = episodes_per_task or int(tc.trace_collection.episodes_per_task)
    out = out or tc.trace_collection.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    planner = AsyncGeminiPlanner(GeminiPlanner(ac), max_workers=planner_workers)
    agent = PromptOnlyAgent(ac, planner=planner)
    writer = JsonlWriter(out, mode="w")
    write_lock = Lock()

    jobs = [(task, seed) for task in ec.tasks for seed in range(episodes_per_task)]
    n_total = 0; n_succ = 0
    # Playwright sync API is single-threaded; AsyncGeminiPlanner handles parallelism.
    if workers > 1:
        info(f"workers={workers} requested but Playwright is sync; forcing workers=1"); workers = 1
    info(f"Launching {len(jobs)} episodes across {workers} parallel workers")

    if workers == 1:
        for t, s in tqdm(jobs, desc="bc"):
            try:
                ep = _run_one(t, s, agent, max_steps, headless)
            except Exception as e:
                info(f"  ! {t}/{s} failed: {e}"); continue
            n_total += 1
            if ep.success:
                n_succ += 1
                writer.write(ep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, t, s, agent, max_steps, headless): (t, s) for t, s in jobs}
            for fut in as_completed(futs):
                t, s = futs[fut]
                try:
                    ep = fut.result()
                except Exception as e:
                    info(f"  ! {t}/{s} failed: {e}"); continue
                n_total += 1
                if ep.success:
                    n_succ += 1
                    with write_lock:
                        writer.write(ep)
                if n_total % 20 == 0:
                    info(f"  {n_total}/{len(jobs)}  succ-so-far={n_succ}")
    writer.close(); planner.close()
    ok(f"Collected {n_succ}/{n_total} successful episodes -> {out}")


if __name__ == "__main__":
    tyro.cli(main)
