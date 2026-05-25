"""Train the RL refiner with PPO on the 8-task MiniWoB suite."""
import os
import random
from omegaconf import OmegaConf
import torch
import tyro

from rlwa.envs import make_env
from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
from rlwa.rl.ppo import PPOTrainer
from rlwa.utils.logging import info


def main(
    agent_cfg: str = "configs/agent/rl_refiner.yaml",
    env_cfg: str = "configs/env/miniwob.yaml",
    train_cfg: str = "configs/train/ppo.yaml",
    use_wandb: bool = False,
    wandb_project: str = "rlwa",
    bc_init: str | None = None,
    device: str = "cuda",
    planner_workers: int = 48,
    save_dir: str | None = None,
):
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set")

    ac = OmegaConf.load(agent_cfg)
    ec = OmegaConf.load(env_cfg)
    tc = OmegaConf.load(train_cfg)
    cfg = OmegaConf.merge(ac, ec, tc)
    if save_dir:
        cfg.train.save_dir = save_dir

    seed = int(cfg.train.seed)
    random.seed(seed); torch.manual_seed(seed)

    tasks = list(cfg.tasks)
    N = int(cfg.train.num_envs)
    # round-robin tasks across envs
    env_fns = []
    for i in range(N):
        t = tasks[i % len(tasks)]
        env_fns.append(lambda task=t, s=seed + i: make_env(
            task, seed=s, headless=cfg.env.headless, max_steps=int(cfg.env.max_steps),
            reward_kwargs={
                "progress_w": float(cfg.reward.progress_w),
                "step_w": float(cfg.reward.step_w),
                "recover_w": float(cfg.reward.recover_w),
                "invalid_w": float(cfg.reward.invalid_w),
            },
        ))

    planner = GeminiPlanner(cfg)
    planner = AsyncGeminiPlanner(planner, max_workers=planner_workers)
    trainer = PPOTrainer(cfg, env_fns=env_fns, planner=planner, device=device)

    # optional BC init
    init = bc_init or cfg.train.get("bc_init")
    if init and os.path.exists(init):
        sd = torch.load(init, map_location=device)
        trainer.policy.load_state_dict(sd["policy"], strict=False)
        info(f"Loaded BC init from {init}")

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project=wandb_project, config=OmegaConf.to_container(cfg, resolve=True))

    try:
        trainer.train(wandb_run=run)
    finally:
        for env in trainer.envs:
            try: env.close()
            except Exception: pass
        if run is not None:
            run.finish()


if __name__ == "__main__":
    tyro.cli(main)
