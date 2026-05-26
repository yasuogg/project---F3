"""Final evaluation: prompt-only vs BC-only vs RL-refiner across all tasks/seeds."""
import json
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf
import tyro

from rlwa.envs import DEFAULT_TASKS, HELD_OUT_TASKS
from rlwa.eval.runner import run_eval
from rlwa.eval.metrics import aggregate, bootstrap_ci
from rlwa.utils.logging import info, ok


class _AgentFactory:
    """Picklable factory: reconstructs agent + planner inside each worker process.

    Must be a top-level class (not a lambda or closure) so ProcessPoolExecutor
    can pickle it across the process boundary.
    """

    def __init__(
        self,
        agent_type: str,
        cfg_path: str,
        ckpt_path: str | None,
        planner_workers: int,
    ):
        self.agent_type = agent_type
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.planner_workers = planner_workers

    def __call__(self):
        from omegaconf import OmegaConf
        from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
        from rlwa.agents import (
            PromptOnlyAgent, RLRefinerAgent,
            ReflectivePromptAgent, RandomRerankerAgent,
        )
        cfg = OmegaConf.load(self.cfg_path)
        if self.agent_type in ("prompt_only", "reflective", "random"):
            planner = AsyncGeminiPlanner(GeminiPlanner(cfg), max_workers=self.planner_workers)
            if self.agent_type == "prompt_only":
                return PromptOnlyAgent(cfg, planner=planner)
            if self.agent_type == "reflective":
                return ReflectivePromptAgent(cfg, planner=planner)
            return RandomRerankerAgent(cfg, planner=planner, seed=0)
        return RLRefinerAgent(cfg, ckpt_path=self.ckpt_path)


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
    seeds_list = list(range(seeds))

    is_planner_agent = agent in ("prompt_only", "reflective", "random")
    cfg_path = prompt_cfg if is_planner_agent else agent_cfg

    if agent in ("bc_only", "rl_refiner"):
        cfg = OmegaConf.load(cfg_path)
        if agent == "bc_only":
            ckpt = ckpt or cfg.get("bc", {}).get("out", "data/checkpoints/bc.pt")
        else:
            ckpt = ckpt or cfg.get("checkpoint", "data/checkpoints/ppo_best.pt")

    # planner_workers are split evenly across browser processes
    async_per_worker = max(1, planner_workers // max(workers, 1))
    factory = _AgentFactory(agent, cfg_path, ckpt, async_per_worker)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{agent}{'_heldout' if held_out else ''}.jsonl"

    info(f"Evaluating {agent} on {len(tasks)} tasks x {seeds} seeds = {len(tasks)*seeds} episodes")
    info(f"workers={workers}  planner_threads_per_worker={async_per_worker}")

    # For workers=1 we need an agent instance; for workers>1 the factory handles it.
    a = factory() if workers <= 1 else None
    eps = run_eval(
        a, tasks=tasks, seeds=seeds_list,
        max_steps=max_steps, headless=headless, out_jsonl=jsonl,
        workers=workers, agent_factory=factory,
    )

    agg = aggregate(eps)
    succ = [p["success"] for p in agg["per_episode"]]
    lo, hi = bootstrap_ci(succ)
    agg["sr_ci95"] = [lo, hi]

    summary_path = out / f"{agent}{'_heldout' if held_out else ''}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in agg.items() if k != "per_episode"}, f, indent=2)
    ok(
        f"SR={agg['success_rate']:.3f}  CI95=[{lo:.3f}, {hi:.3f}]  "
        f"steps={agg['mean_steps']:.1f}  invalid={agg['invalid_rate']:.3f}  "
        f"recovery={agg['recovery_rate']}"
    )
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    tyro.cli(main)
