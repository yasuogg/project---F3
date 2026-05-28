"""BrowserGym env factory + reward-shaping wrapper."""
from __future__ import annotations
import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional
import gymnasium as gym


# browsergym emits "Overriding the task's viewport/slow_mo ..." via logging.warning
# on every env construction. With 800 episodes × N workers this floods the console.
# Logger-level filters only catch records emitted by *that* logger (not children
# whose records merely propagate), so we attach to handlers instead — covering
# lastResort and any handlers already configured on root / browsergym.
class _DropTaskOverrideMsg(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Overriding the task's")


_drop_filter = _DropTaskOverrideMsg()
if logging.lastResort is not None:
    logging.lastResort.addFilter(_drop_filter)
for _lname in ("", "browsergym"):
    for _h in logging.getLogger(_lname).handlers:
        _h.addFilter(_drop_filter)


def _ensure_miniwob_url():
    """Make sure MINIWOB_URL points at miniwob html files; clone them if missing."""
    if os.environ.get("MINIWOB_URL"):
        return
    cache = Path(os.environ.get("RLWA_CACHE", str(Path.home() / ".cache" / "rlwa")))
    miniwob_dir = cache / "miniwob-plusplus"
    html_dir = miniwob_dir / "miniwob" / "html" / "miniwob"
    if not html_dir.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Farama-Foundation/miniwob-plusplus.git",
             str(miniwob_dir)],
            check=True,
        )
    os.environ["MINIWOB_URL"] = f"file://{html_dir}/"


_ensure_miniwob_url()

# Importing browsergym.miniwob registers MiniWoB tasks with gym
import browsergym.miniwob  # noqa: F401, E402


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
    # browsergym 0.13+ registers ids as 'browsergym/miniwob.<task>'; accept either
    if not task_name.startswith("browsergym/"):
        candidates = [f"browsergym/{task_name}", task_name]
    else:
        candidates = [task_name]
    last_err: Exception | None = None
    env = None
    for tid in candidates:
        try:
            env = gym.make(
                tid,
                headless=headless,
                action_mapping=None,  # use high-level python actions
                viewport={"width": viewport[0], "height": viewport[1]},
                slow_mo=0,
                wait_for_user_message=False,
            )
            break
        except Exception as e:
            last_err = e
    if env is None:
        raise RuntimeError(f"Could not register env for {task_name!r}; tried {candidates}") from last_err
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    if shape_reward:
        env = RewardShapeWrapper(env, **(reward_kwargs or {}))
    return env
