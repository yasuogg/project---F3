"""Reflective prompt agent: free win baseline.

On each invalid action, re-queries the planner with an explicit critique block
forced ON (regardless of the regular recovery flag), and tries the *new* top-1.
"""
from __future__ import annotations
from typing import Optional
import gymnasium as gym

from rlwa.planners import GeminiPlanner
from rlwa.agents.action_space import build_observation
from rlwa.utils.schemas import EpisodeRecord, StepRecord


CRITIQUE_PREFIX = (
    "Your previous action failed. Carefully re-read the AX-tree, look at the marks "
    "again, and propose a *different* action that targets a visible interactive "
    "element. Avoid repeating the failed bid."
)


class ReflectivePromptAgent:
    def __init__(self, cfg, planner: Optional[GeminiPlanner] = None):
        self.cfg = cfg
        self.planner = planner or GeminiPlanner(cfg)

    def run_episode(self, env: gym.Env, task: str, seed: int, max_steps: int = 25) -> EpisodeRecord:
        raw, _ = env.reset(seed=seed)
        ep = EpisodeRecord(task=task, seed=seed)
        history: list[dict] = []
        last_error: str | None = None

        for step in range(max_steps):
            obs = build_observation(raw)
            err_for_planner = last_error
            # if previous step failed, force-prepend an explicit critique
            if last_error:
                err_for_planner = f"{CRITIQUE_PREFIX}\nPrevious error: {last_error}"

            cands = self.planner.propose(
                goal=obs["goal"], som_image=obs["som_image"],
                axtree=obs["axtree"], mark_bids=obs["mark_bids"],
                history=history, last_error=err_for_planner,
            )

            # if last step failed, skip any candidate that points at the same bid
            chosen_idx = 0
            if last_error and history:
                prev_action = history[-1]["action"]
                for i, c in enumerate(cands):
                    if c.bid and c.bid not in prev_action:
                        chosen_idx = i; break
            chosen = cands[chosen_idx]
            action_str = chosen.to_browsergym_action()

            try:
                raw, reward, terminated, truncated, info = env.step(action_str)
                err_msg = info.get("last_action_error") or info.get("action_error")
            except Exception as e:
                reward, terminated, truncated, info = 0.0, False, False, {"action_error": str(e)}
                err_msg = str(e)

            failed = bool(err_msg)
            success = bool(info.get("env_reward", reward) > 0.5) or bool(info.get("success"))

            ep.steps.append(StepRecord(
                task=task, seed=seed, step=step, goal=obs["goal"],
                axtree_snippet=obs["axtree"][:600],
                candidates=cands, chosen_idx=chosen_idx, action_str=action_str,
                reward=float(reward), done=bool(terminated or truncated),
                success=success, error=err_msg,
            ))
            ep.total_reward += float(reward); ep.n_steps += 1
            if failed: ep.n_invalid += 1
            if history and history[-1].get("failed") and not failed:
                ep.n_recovered += 1

            history.append({"step": step, "action": action_str,
                            "reward": float(reward), "failed": failed})
            last_error = err_msg

            if terminated or truncated:
                ep.success = success or ep.success
                break
        return ep
