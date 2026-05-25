"""Record 1 trajectory per task per agent for the viva slides / report.

Produces data/traj/<task>__seed0/index.html with screenshots + actions.
"""
from pathlib import Path
from omegaconf import OmegaConf
import tyro

from rlwa.envs import make_env, DEFAULT_TASKS
from rlwa.agents import PromptOnlyAgent, RLRefinerAgent
from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
from rlwa.eval.recorder import RecordingAgent
from rlwa.utils.logging import info, ok


def main(
    seed: int = 0,
    max_steps: int = 25,
    out_dir: str = "data/traj",
    prompt_cfg: str = "configs/agent/prompt_only.yaml",
    refiner_cfg: str = "configs/agent/rl_refiner.yaml",
    ckpt: str = "data/checkpoints/ppo_best.pt",
    headless: bool = True,
    tasks: list[str] | None = None,
):
    tasks = tasks or DEFAULT_TASKS
    pc = OmegaConf.load(prompt_cfg)
    rc = OmegaConf.load(refiner_cfg)
    planner = AsyncGeminiPlanner(GeminiPlanner(pc), max_workers=16)
    prompt_agent = PromptOnlyAgent(pc, planner=planner)
    rl_agent = RLRefinerAgent(rc, ckpt_path=ckpt)

    for task in tasks:
        for label, agent in [("prompt_only", prompt_agent), ("rl_refiner", rl_agent)]:
            env = make_env(task, seed=seed, headless=headless, max_steps=max_steps)
            try:
                rec = RecordingAgent(agent, out_dir=f"{out_dir}/{label}", label=label)
                ep = rec.run_episode(env, task=task, seed=seed, max_steps=max_steps)
            finally:
                env.close()
            info(f"{label}/{task}: success={ep.success}, steps={ep.n_steps}")
    ok(f"Trajectories saved under {out_dir}/{{prompt_only,rl_refiner}}/")


if __name__ == "__main__":
    tyro.cli(main)
