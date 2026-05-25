"""Robustness eval: same agents on CSS-jittered tasks."""
import json
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf
import tyro

from rlwa.envs import make_env, DEFAULT_TASKS
from rlwa.agents import PromptOnlyAgent, RLRefinerAgent
from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
from rlwa.eval.perturb import CSSJitterWrapper
from rlwa.eval.metrics import aggregate
from rlwa.utils.schemas import EpisodeRecord
from rlwa.utils.logging import JsonlWriter, info, ok
from tqdm import tqdm


def main(
    agent: Literal["prompt_only", "rl_refiner"] = "rl_refiner",
    seeds: int = 3,
    max_steps: int = 25,
    out_dir: str = "data/eval",
    headless: bool = True,
    ckpt: str | None = None,
    agent_cfg: str = "configs/agent/rl_refiner.yaml",
    prompt_cfg: str = "configs/agent/prompt_only.yaml",
    workers: int = 16,
    planner_workers: int = 64,
):
    if agent == "prompt_only":
        cfg = OmegaConf.load(prompt_cfg)
        planner = AsyncGeminiPlanner(GeminiPlanner(cfg), max_workers=planner_workers)
        a = PromptOnlyAgent(cfg, planner=planner)
    else:
        cfg = OmegaConf.load(agent_cfg)
        a = RLRefinerAgent(cfg, ckpt_path=ckpt)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{agent}_perturb.jsonl"
    writer = JsonlWriter(jsonl, mode="w")
    eps: list[EpisodeRecord] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    write_lock = Lock()

    def _run(task, seed):
        env = make_env(task, seed=seed, headless=headless, max_steps=max_steps)
        env = CSSJitterWrapper(env, seed=seed)
        try:
            return a.run_episode(env, task=task, seed=seed, max_steps=max_steps)
        finally:
            env.close()

    jobs = [(t, s) for t in DEFAULT_TASKS for s in range(seeds)]
    pbar = tqdm(total=len(jobs), desc=f"perturb(x{workers})")
    if workers > 1:
        pbar.write(f"workers={workers} but Playwright is sync; forcing workers=1"); workers = 1
    if workers == 1:
        for t, s in jobs:
            try: ep = _run(t, s)
            except Exception as e: pbar.write(f"! {t}/{s}: {e}"); pbar.update(1); continue
            eps.append(ep); writer.write(ep); pbar.update(1); pbar.set_postfix(succ=int(ep.success))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, t, s): (t, s) for t, s in jobs}
        for fut in as_completed(futs):
            t, s = futs[fut]
            try: ep = fut.result()
            except Exception as e: pbar.write(f"! {t}/{s}: {e}"); pbar.update(1); continue
            eps.append(ep)
            with write_lock: writer.write(ep)
            pbar.update(1); pbar.set_postfix(succ=int(ep.success))
    pbar.close(); writer.close()

    agg = aggregate(eps)
    summary_path = out / f"{agent}_perturb_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in agg.items() if k != "per_episode"}, f, indent=2)
    ok(f"Perturbed SR={agg['success_rate']:.3f}  steps={agg['mean_steps']:.1f}  "
       f"invalid={agg['invalid_rate']:.3f}  recovery={agg['recovery_rate']}")


if __name__ == "__main__":
    tyro.cli(main)
