"""Quick smoke test: load BrowserGym + Gemini, run 1 episode."""
import os
import sys
from omegaconf import OmegaConf
import tyro
from rlwa.envs import make_env
from rlwa.agents import PromptOnlyAgent
from rlwa.utils.logging import info, ok, err


def main(
    task: str = "miniwob.click-button",
    seed: int = 0,
    max_steps: int = 15,
    headless: bool = True,
    config: str = "configs/agent/prompt_only.yaml",
):
    if not os.environ.get("GEMINI_API_KEY"):
        err("GEMINI_API_KEY not set"); sys.exit(2)
    cfg = OmegaConf.load(config)
    info(f"Smoke test: task={task} seed={seed}")
    env = make_env(task, seed=seed, headless=headless, max_steps=max_steps)
    try:
        agent = PromptOnlyAgent(cfg)
        ep = agent.run_episode(env, task=task, seed=seed, max_steps=max_steps)
    finally:
        env.close()
    ok(f"Done: success={ep.success}  steps={ep.n_steps}  reward={ep.total_reward:.2f}  "
       f"invalid={ep.n_invalid}  recovered={ep.n_recovered}")


if __name__ == "__main__":
    tyro.cli(main)
