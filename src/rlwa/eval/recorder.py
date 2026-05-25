"""Trajectory recorder: save screenshots + actions per step for replay.

Drop-in wrapper around any agent: records to data/traj/<task>_<seed>/ as
step_NN.png + step_NN.json, plus an index.html for viva playback.
"""
from __future__ import annotations
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Optional


def _img_to_b64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RecordingAgent:
    """Wrap any agent (PromptOnlyAgent, RLRefinerAgent, ...) and dump steps.

    Usage:
        rec = RecordingAgent(agent, out_dir="data/traj")
        ep = rec.run_episode(env, task, seed, max_steps)
    """
    def __init__(self, agent, out_dir: str, label: str = ""):
        self.agent = agent
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label = label

    def run_episode(self, env, task: str, seed: int, max_steps: int = 25):
        # we monkey-patch env.step to snapshot screenshots; safer to subclass agent
        # but BrowserGym envs expose raw obs through reset/step. Easiest: patch.
        ep_dir = self.out_dir / f"{task.replace('/', '_')}__seed{seed}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        steps_meta = []

        orig_step = env.step
        orig_reset = env.reset
        step_counter = {"n": 0}

        def _snapshot(raw, action_str=None, reward=None, error=None):
            from rlwa.agents.action_space import build_observation
            try:
                obs = build_observation(raw)
                im = obs.get("som_image")
            except Exception:
                im = None
            n = step_counter["n"]
            data = {
                "step": n, "task": task, "seed": seed, "label": self.label,
                "action": action_str, "reward": reward, "error": error,
                "goal": obs.get("goal") if im else None,
            }
            if im is not None:
                data["screenshot_b64"] = _img_to_b64(im)
            steps_meta.append(data)
            step_counter["n"] += 1

        def patched_reset(**kw):
            r = orig_reset(**kw)
            _snapshot(r[0], action_str="<reset>", reward=0.0)
            return r

        def patched_step(action_str):
            out = orig_step(action_str)
            raw, reward, term, trunc, info = out
            err = info.get("last_action_error") or info.get("action_error")
            _snapshot(raw, action_str=action_str, reward=float(reward), error=err)
            return out

        env.reset = patched_reset
        env.step = patched_step
        try:
            ep = self.agent.run_episode(env, task=task, seed=seed, max_steps=max_steps)
        finally:
            env.reset = orig_reset
            env.step = orig_step

        # write JSON + HTML viewer
        (ep_dir / "trajectory.json").write_text(
            json.dumps({"task": task, "seed": seed, "label": self.label,
                        "success": ep.success, "n_steps": ep.n_steps,
                        "total_reward": ep.total_reward, "steps": steps_meta}, indent=2)
        )
        _write_html(ep_dir, task, seed, ep, steps_meta, self.label)
        return ep


def _write_html(ep_dir: Path, task: str, seed: int, ep, steps: list, label: str):
    rows = []
    for s in steps:
        img_html = (f'<img src="data:image/png;base64,{s["screenshot_b64"]}" '
                    'style="max-width:600px;border:1px solid #ccc">'
                    if "screenshot_b64" in s else "(no screenshot)")
        err = f'<span style="color:red">{s["error"]}</span>' if s.get("error") else ""
        rows.append(f"""
        <tr>
          <td style="vertical-align:top;padding:8px;width:80px">step {s['step']}</td>
          <td style="vertical-align:top;padding:8px">{img_html}</td>
          <td style="vertical-align:top;padding:8px">
            <div><b>action:</b> <code>{(s.get('action') or '')[:200]}</code></div>
            <div><b>reward:</b> {s.get('reward')}</div>
            <div>{err}</div>
          </td>
        </tr>""")
    title = f"{label or 'agent'}  |  {task}  seed={seed}  success={ep.success}"
    html = f"""<!doctype html><html><head>
    <meta charset="utf-8"><title>{title}</title></head>
    <body style="font-family:sans-serif">
    <h2>{title}</h2>
    <p>Total reward = {ep.total_reward:.3f}, steps = {ep.n_steps}</p>
    <table>{''.join(rows)}</table></body></html>"""
    (ep_dir / "index.html").write_text(html, encoding="utf-8")
