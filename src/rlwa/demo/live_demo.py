"""Gradio side-by-side demo: prompt-only vs RL-refiner.

Run:  python -m rlwa.demo.live_demo
"""
from __future__ import annotations
import os
from pathlib import Path
from omegaconf import OmegaConf
import gradio as gr

from rlwa.envs import make_env, DEFAULT_TASKS
from rlwa.agents import PromptOnlyAgent, RLRefinerAgent


def _build_storyboard(ep):
    lines = []
    for s in ep.steps:
        ok = "✓" if not s.error else f"✗ ({s.error[:40]})"
        lines.append(f"[{s.step}] {ok}  reward={s.reward:+.2f}  -> {s.action_str}")
    return "\n".join(lines) + f"\n\n→ success={ep.success}  steps={ep.n_steps}  total_R={ep.total_reward:.2f}"


def _run_one(agent, task, seed, headless=True, max_steps=20):
    env = make_env(task, seed=seed, headless=headless, max_steps=max_steps)
    try:
        ep = agent.run_episode(env, task=task, seed=seed, max_steps=max_steps)
    finally:
        env.close()
    return _build_storyboard(ep), ep.success


def main(
    prompt_cfg: str = "configs/agent/prompt_only.yaml",
    refiner_cfg: str = "configs/agent/rl_refiner.yaml",
    ckpt: str = "data/checkpoints/ppo_best.pt",
    port: int = 7860,
):
    pc = OmegaConf.load(prompt_cfg)
    rc = OmegaConf.load(refiner_cfg)
    prompt_agent = PromptOnlyAgent(pc)
    rl_agent = RLRefinerAgent(rc, ckpt_path=ckpt if os.path.exists(ckpt) else None)

    def run(task, seed):
        po_text, po_succ = _run_one(prompt_agent, task, int(seed))
        rl_text, rl_succ = _run_one(rl_agent, task, int(seed))
        return po_text, rl_text, f"prompt-only: {po_succ} | rl-refiner: {rl_succ}"

    with gr.Blocks(title="RL-Augmented Vision Web Agent") as demo:
        gr.Markdown("# RL-Augmented Vision Web Agent — side-by-side")
        with gr.Row():
            task = gr.Dropdown(DEFAULT_TASKS, value=DEFAULT_TASKS[0], label="Task")
            seed = gr.Number(value=0, label="Seed", precision=0)
            go = gr.Button("Run")
        with gr.Row():
            po_out = gr.Textbox(label="Prompt-only trajectory", lines=18)
            rl_out = gr.Textbox(label="RL-Refiner trajectory", lines=18)
        verdict = gr.Textbox(label="Verdict")
        go.click(run, [task, seed], [po_out, rl_out, verdict])
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
