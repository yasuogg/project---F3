"""Run ablation sweeps: -KL, -Recovery, -MetaActions, -BCInit.

Launches separate PPO runs with surgical config overrides and tags checkpoints
so 04_eval.py can re-evaluate each one.
"""
import os
import random
import subprocess
import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch
import tyro

from rlwa.envs import make_env
from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
from rlwa.rl.ppo import PPOTrainer
from rlwa.utils.logging import info, ok


ABLATIONS = {
    "full":         {},
    "no_kl":        {"train.kl_to_vlm": 0.0, "train.kl_anneal_end": 0.0},
    "no_recovery":  {"reward.recover_w": 0.0},
    "no_meta":      {"policy.n_meta_actions": 0},
    "no_bc_init":   {"train.bc_init": None},
}


def _apply_overrides(cfg, overrides: dict):
    for dotted, val in overrides.items():
        OmegaConf.update(cfg, dotted, val, merge=True)
    return cfg


def main(
    only: str | None = None,        # comma-separated ablation names to run
    total_steps: int = 40_000,      # shorter than full PPO to fit in 3 days
    agent_cfg: str = "configs/agent/rl_refiner.yaml",
    env_cfg: str = "configs/env/miniwob.yaml",
    train_cfg: str = "configs/train/ppo.yaml",
    save_root: str = "data/checkpoints/ablations",
    device: str = "cuda",
):
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set")

    selected = ABLATIONS if only is None else {k: v for k, v in ABLATIONS.items() if k in only.split(",")}
    Path(save_root).mkdir(parents=True, exist_ok=True)

    for name, overrides in selected.items():
        info(f"=== ABLATION: {name}  overrides={overrides} ===")
        ac = OmegaConf.load(agent_cfg); ec = OmegaConf.load(env_cfg); tc = OmegaConf.load(train_cfg)
        cfg = OmegaConf.merge(ac, ec, tc)
        cfg.train.total_env_steps = int(total_steps)
        cfg.train.save_dir = f"{save_root}/{name}"
        _apply_overrides(cfg, overrides)

        seed = int(cfg.train.seed); random.seed(seed); torch.manual_seed(seed)
        tasks = list(cfg.tasks); N = int(cfg.train.num_envs)
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
        planner = AsyncGeminiPlanner(planner, max_workers=48)
        trainer = PPOTrainer(cfg, env_fns=env_fns, planner=planner, device=device)

        init = cfg.train.get("bc_init")
        if init and os.path.exists(init):
            sd = torch.load(init, map_location=device)
            trainer.policy.load_state_dict(sd["policy"], strict=False)
            info(f"BC init loaded: {init}")

        try:
            trainer.train()
        finally:
            for env in trainer.envs:
                try: env.close()
                except Exception: pass
        ok(f"Ablation {name} done -> {cfg.train.save_dir}/ppo_best.pt")


if __name__ == "__main__":
    tyro.cli(main)
