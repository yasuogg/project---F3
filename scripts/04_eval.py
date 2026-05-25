"""Final evaluation: prompt-only vs BC-only vs RL-refiner across all tasks/seeds."""
import json
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf
import tyro

from rlwa.envs import DEFAULT_TASKS, HELD_OUT_TASKS
from rlwa.agents import PromptOnlyAgent, RLRefinerAgent, ReflectivePromptAgent, RandomRerankerAgent
from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
from rlwa.eval.runner import run_eval
from rlwa.eval.metrics import aggregate, bootstrap_ci
from rlwa.utils.logging import info, ok


def main(
    agent: Literal["prompt_only", "reflective", "rl_refiner", "bc_only", "random"] = "prompt_only",
    seeds: int = 5,
    max_steps: int = 25,
    held_out: bool = False,
    ckpt: str | None = None,
    out_dir: str = "data/eval",
    headless: bool = True,
    agent_cfg: str = "configs/agent/rl_refiner.yaml",
    prompt_cfg: str = "configs/agent/prompt_only.yaml",
    workers: int = 32,
    planner_workers: int = 64,
):
    tasks = HELD_OUT_TASKS if held_out else DEFAULT_TASKS

    if agent == "prompt_only":
        cfg = OmegaConf.load(prompt_cfg)
        planner = AsyncGeminiPlanner(GeminiPlanner(cfg), max_workers=planner_workers)
        a = PromptOnlyAgent(cfg, planner=planner)
    elif agent == "reflective":
        cfg = OmegaConf.load(prompt_cfg)
        planner = AsyncGeminiPlanner(GeminiPlanner(cfg), max_workers=planner_workers)
        a = ReflectivePromptAgent(cfg, planner=planner)
    elif agent == "random":
        cfg = OmegaConf.load(prompt_cfg)
        planner = AsyncGeminiPlanner(GeminiPlanner(cfg), max_workers=planner_workers)
        a = RandomRerankerAgent(cfg, planner=planner, seed=0)
    else:
        cfg = OmegaConf.load(agent_cfg)
        if agent == "bc_only":
            ckpt = ckpt or cfg.get("bc", {}).get("out", "data/checkpoints/bc.pt")
        else:
            ckpt = ckpt or cfg.get("checkpoint", "data/checkpoints/ppo_best.pt")
        a = RLRefinerAgent(cfg, ckpt_path=ckpt)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{agent}{'_heldout' if held_out else ''}.jsonl"
    seeds_list = list(range(seeds))
    info(f"Evaluating {agent} on {len(tasks)} tasks x {seeds} seeds = {len(tasks)*seeds} episodes")

    eps = run_eval(a, tasks=tasks, seeds=seeds_list, max_steps=max_steps,
                   headless=headless, out_jsonl=jsonl, workers=workers)
    agg = aggregate(eps)
    succ = [p["success"] for p in agg["per_episode"]]
    lo, hi = bootstrap_ci(succ)
    agg["sr_ci95"] = [lo, hi]

    summary_path = out / f"{agent}{'_heldout' if held_out else ''}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in agg.items() if k != "per_episode"}, f, indent=2)
    ok(f"SR={agg['success_rate']:.3f}  CI95=[{lo:.3f}, {hi:.3f}]  "
       f"steps={agg['mean_steps']:.1f}  invalid={agg['invalid_rate']:.3f}  "
       f"recovery={agg['recovery_rate']}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    tyro.cli(main)
