# RL-Augmented Vision Web Agents (RLWA)

Compact RL refinement on top of a frozen Gemini VLM planner, evaluated on
BrowserGym + MiniWoB++. Goal: show that a small PPO "action refiner" improves
task success, recovery, and robustness vs prompt-only planning.

## Contributions / What is novel

1. **KL-to-VLM annealed regularizer** — anchors the PPO policy to the VLM prior
   early in training (KL coef 0.10 → 0.02 over 40k steps), preventing reward
   hacking on shaped rewards while still letting the refiner override the
   planner once it has learned where the planner is unreliable.
2. **Factored action space with learned meta-actions** — instead of only
   re-ranking the K candidates, the policy can emit `ABSTAIN` (re-query the
   VLM with a recovery prompt) or `REPORT_INFEASIBLE`. Both are learned via
   PPO, not hard-coded heuristics.
3. **Recovery-shaped reward** — explicit +0.2 bonus when the agent recovers
   from a previously failed action, +0.05 progress (state-hash novelty),
   −0.01 step penalty, −0.1 invalid-action penalty. Decomposes success vs.
   recovery in evaluation.
4. **Top-K oracle headroom analysis** — `scripts/07_oracle_topk.py` computes
   `SR_oracle = max_k SR(force_idx=k)` per task, upper-bounding the success
   rate achievable by *any* pure re-ranker over the VLM's candidates. The gap
   between RL-refiner and this oracle quantifies how much of the remaining
   error is the planner's fault, not the refiner's.
5. **Reproducible Colab + Gemini-only pipeline** — no proprietary models, no
   manual labeling. PPO auto-resumes from `ppo_last.pt` on Colab disconnect;
   `scripts/10_fill_report.py` injects results into the report; pytest smoke
   tests validate schemas, metrics, and checkpoints.

## Sanity baselines (reported alongside the main results)

* `--agent random` — random re-ranker over the K VLM candidates.
* `--agent prompt_only` — direct VLM top-1 (no refiner).
* `--agent reflective` — VLM with a critique-loop prompt after failures.
* `--agent bc_only` — BC-pretrained refiner without PPO.
* `--agent rl_refiner` — BC + PPO refiner (full system).
* `oracle_topk` — upper bound across all re-rank positions.

## Quickstart (Colab Pro+ A100)

1. Push this repo to GitHub.
2. Open `notebooks/colab_run.ipynb` in Colab.
3. Set Colab secrets: `GEMINI_API_KEY` (required), `WANDB_API_KEY` (optional).
4. Run all cells.

## Quickstart (local)

```bash
# Linux / WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
playwright install chromium
playwright install-deps   # Linux only

export GEMINI_API_KEY=...
python scripts/00_smoke_test.py
```

## Pipeline

```
1. Baseline      python scripts/04_eval.py --agent prompt_only
2. BC traces     python scripts/01_collect_bc_traces.py
3. BC pretrain   python scripts/02_train_bc.py
4. PPO train     python scripts/03_train_ppo.py        # auto-resumes from ppo_last.pt
5. Final eval    python scripts/04_eval.py --agent rl_refiner
6. Reflective    python scripts/04_eval.py --agent reflective
7. Random base.  python scripts/04_eval.py --agent random
8. Robustness    python scripts/05_perturb_eval.py
9. Ablations     python scripts/06_ablations.py
10. Oracle top-K python scripts/07_oracle_topk.py
11. Plots        python scripts/08_plots.py
12. Trajectory   python scripts/09_record_trajectories.py
13. Report       python scripts/10_fill_report.py
```

## Tests

```bash
pytest                 # smoke tests for schemas, metrics, checkpoint round-trip
```


## Layout

```
src/rlwa/
  envs/         BrowserGym wrappers + 8-task MiniWoB registry
  obs/          Set-of-Mark renderer, AX-tree pruner, frozen encoders
  planners/     Gemini planner + prompt templates
  agents/       prompt-only + RL-refiner agents
  rl/           PPO, BC, buffer, reward shaping
  eval/         runner, metrics, perturbations
  demo/         Gradio side-by-side demo
scripts/        CLI entry points
configs/        Hydra YAML configs
notebooks/      Colab end-to-end notebook
paper/          short report template
```

## Scope (3-day version)

- 8 MiniWoB tasks, 5 seeds
- Planner: `gemini-2.0-flash`
- Refiner: SigLIP-base + MiniLM (frozen) + MLP fuse + PPO head (~5M trainable)
- Conditions: prompt-only, BC-only, RL-refiner (+ no-BC ablation)
- 1 robustness perturbation (CSS jitter)

See `paper/report_template.md` for the report skeleton.
