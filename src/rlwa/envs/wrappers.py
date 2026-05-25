"""BrowserGym env factory + reward-shaping wrapper."""
from __future__ import annotations
import hashlib
from typing import Optional
import gymnasium as gym

# Importing browsergym.miniwob registers MiniWoB tasks with gym
import browsergym.miniwob  # noqa: F401


class RewardShapeWrapper(gym.Wrapper):
    """Adds dense progress/recovery/invalid-action shaping on top of env reward."""

    def __init__(
        self,
        env: gym.Env,
        progress_w: float = 0.05,
        step_w: float = -0.01,
        recover_w: float = 0.2,
        invalid_w: float = -0.1,
    ):
        super().__init__(env)
        self.progress_w = progress_w
        self.step_w = step_w
        self.recover_w = recover_w
        self.invalid_w = invalid_w
        self._seen_states: set[str] = set()
        self._last_action_failed: bool = False

    def reset(self, **kwargs):
        self._seen_states.clear()
        self._last_action_failed = False
        return self.env.reset(**kwargs)

    @staticmethod
    def _state_hash(obs: dict) -> str:
        ax = obs.get("axtree_txt", "") or obs.get("axtree", "") or ""
        url = obs.get("url", "") or ""
        return hashlib.md5((ax + "||" + url).encode("utf-8")).hexdigest()[:12]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # detect invalid / failed action via info or last error
        action_failed = bool(info.get("action_error") or info.get("last_action_error"))

        shaped = float(reward) + self.step_w
        if action_failed:
            shaped += self.invalid_w
        elif self._last_action_failed:
            shaped += self.recover_w

        sh = self._state_hash(obs)
        if sh not in self._seen_states and not action_failed:
            shaped += self.progress_w
            self._seen_states.add(sh)

        self._last_action_failed = action_failed
        info["action_failed"] = action_failed
        info["shaped_reward"] = shaped
        info["env_reward"] = float(reward)
        return obs, shaped, terminated, truncated, info


def make_env(
    task_name: str,
    seed: int = 0,
    headless: bool = True,
    max_steps: int = 25,
    viewport: tuple[int, int] = (1280, 720),
    reward_kwargs: Optional[dict] = None,
    shape_reward: bool = True,
) -> gym.Env:
    """Build a single BrowserGym env, configured for our action space."""
    env = gym.make(
        task_name,
        headless=headless,
        action_mapping=None,  # use high-level python actions
        viewport={"width": viewport[0], "height": viewport[1]},
        slow_mo=0,
        wait_for_user_message=False,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    if shape_reward:
        env = RewardShapeWrapper(env, **(reward_kwargs or {}))
    env.reset(seed=seed)
    return env
