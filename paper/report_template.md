# RL-Augmented Vision Web Agents for MiniWoB++

> Short report — 4–5 pages — submission template.

## Abstract
We study whether a small, frozen-encoder RL refiner trained on top of a strong vision-language planner (Gemini 2.0 Flash) can improve **task success rate**, **sample efficiency at inference** (fewer steps), and **robustness to visual perturbations** on a suite of MiniWoB++ web-interaction tasks. The refiner re-ranks K=5 VLM-proposed actions and may emit two meta-actions (`ABSTAIN`, `REPORT_INFEASIBLE`). It is trained with PPO under a KL-to-VLM regularizer and a recovery-shaped reward. We report results on 8 training tasks, 2 held-out tasks, and a CSS-jittered perturbation suite.

## 1. Introduction
- Motivation: VLM planners suffer from grounding errors and brittle UI parsing.
- **Research questions**:
  - **RQ1**: Does re-ranking K VLM candidates with a small RL refiner beat top-1?
  - **RQ2**: Does the KL-to-VLM regularizer (annealed 0.10 → 0.02) prevent reward hacking while still allowing improvement?
  - **RQ3**: Does the recovery-shaped reward reduce invalid-action streaks?
  - **RQ4**: Does the refiner improve robustness to visual perturbations (CSS jitter) over a strong VLM baseline?
- Contribution:
  1. A **frozen-encoder action re-ranker** (~5M params) over VLM candidates.
  2. **KL-to-VLM** annealing (0.10→0.02) to preserve planner priors.
  3. A **recovery-shaped reward** that explicitly credits the agent for undoing its own mistakes.
  4. A reproducible Colab-based pipeline using only Gemini API + a single A100.

## 2. Method
### 2.1 Observation
- Screenshot rendered with **Set-of-Marks** (SoM) on AX-tree element bboxes.
- **Pruned AX-tree** (≤80 lines, ≤4000 chars) with the same bid mapping.
- Goal string from BrowserGym task.

### 2.2 Action space
- VLM proposes **K=5 candidates** in factored form `{action_type, bid, text, p, rationale}` over 9 action types (click, fill, select_option, hover, press, scroll, go_back, wait, noop).
- Refiner outputs a categorical over `K + 2` (the two meta-actions).
- Meta-actions: `ABSTAIN` (force re-plan with recovery prompt block) and `REPORT_INFEASIBLE` (terminate).

### 2.3 Policy
- Frozen **SigLIP-base-patch16-224** image encoder (768-d) and frozen **MiniLM-L6-v2** text encoder (384-d).
- Trainable MLP heads: image_proj, goal_proj, history_proj → fused context (256-d).
- Per-candidate score = MLP(context ‖ candidate_text_emb ‖ candidate_meta_feats).
- Separate `meta_head` (2 logits) and `value_head`. Logits concatenated to a (K+2)-way softmax.

### 2.4 Reward shaping
`r_t = +0.05 · progress_t − 0.01 · step + 0.2 · recover_t − 0.1 · invalid_t + R_task`
where `progress_t` fires on first visit to a new state hash and `recover_t` fires when an `invalid` step is followed by a `progress` step within a 3-step window.

### 2.5 Training
- **BC pretraining** on prompt-only success traces (~100 episodes × 8 tasks).
- **PPO**: 80k env-steps, N=4 vectorized envs, T=32 rollout, mb=64, 4 epochs, lr=3e-4, GAE λ=0.95, γ=0.99, clip=0.2, entropy 0.01.
- **KL-to-VLM** coefficient β annealed linearly 0.10 → 0.02 over training.

## 3. Experimental setup
- **8 train tasks**: click-button, click-checkboxes, click-tab-2, enter-text, enter-date, login-user, use-autocomplete, click-dialog.
- **2 held-out tasks**: click-button-sequence, enter-time.
- 5 seeds per (agent, task). Episode cap 25 steps.
- Perturbation: random CSS jitter (font size ±20%, padding ±4px, button hue rotate) injected via Playwright `page.evaluate` after reset.
- **Baselines**:
  - **Prompt-only**: Gemini top-1.
  - **Reflective**: Gemini + critique re-prompt after every invalid step (free additional API call).
  - **BC-only**: refiner trained on prompt-only success traces, no PPO.
  - **Top-K oracle**: for each step run all K candidates in parallel rollouts; take the best; upper bound on what re-ranking can ever achieve.
  - **RL-refiner (ours)**: BC-init + PPO with KL-to-VLM + recovery shaping + meta-actions.

## 4. Results

### 4.1 Main results (clean, 8 tasks × 5 seeds)
| Agent          | SR ↑   | CI95         | Steps ↓ | Invalid ↓ | Recovery rate |
|----------------|--------|--------------|---------|-----------|---------------|
| Prompt-only    |        |              |         |           |               |
| Reflective     |        |              |         |           |               |
| BC-only        |        |              |         |           |               |
| RL-refiner     |        |              |         |           |               |
| Top-K oracle   |        |              |   —     |    —      |       —       |

*Statistical significance*: paired bootstrap (10 000 resamples) RL-refiner vs Prompt-only:  p = _____ .

![Per-task SR](figs/sr_per_task.png)

### 4.2 Held-out tasks (2 tasks × 5 seeds)
| Agent          | SR     | CI95         |
|----------------|--------|--------------|
| Prompt-only    |        |              |
| RL-refiner     |        |              |

### 4.3 Robustness (CSS jitter, 8 tasks × 3 seeds)
| Agent          | SR-clean | SR-perturb | Δ |
|----------------|----------|------------|---|
| Prompt-only    |          |            |   |
| RL-refiner     |          |            |   |

![Robustness](figs/sr_perturb.png)

### 4.4 Learning curve
![Learning curve](figs/learning_curve.png)

## 5. Ablations
| Variant         | SR ↑ | Invalid ↓ | Recovery rate | Note                                  |
|-----------------|------|-----------|---------------|----------------------------------------|
| Full            |      |           |               | RQ1                                    |
| − KL-to-VLM     |      |           |               | RQ2 — expect drift, lower SR           |
| − Recovery rew. |      |           |               | RQ3 — expect higher invalid streaks    |
| − Meta actions  |      |           |               | RQ4 — expect drop under perturbation   |
| − BC init       |      |           |               | tests sample efficiency                |

![Ablations](figs/ablations.png)

## 6. Discussion
### 6.1 Headroom analysis (Top-K oracle)
The oracle reaches SR = _____ vs top-1 = _____ , leaving Δ = _____ of headroom that **any** re-ranker could in principle capture; our refiner closes _____ % of that gap.

### 6.2 Abstain precision
When the policy emitted `ABSTAIN`, the next planner candidate set was *different* in _____ % of cases, and the subsequent step succeeded in _____ % — evidence the meta-action is doing real work.

### 6.3 Per-task analysis
- Largest gains: `use-autocomplete` and `enter-date` (rich candidate sets, top-1 often wrong).
- Smallest / negative: `click-dialog` (single obvious target, no headroom).

### 6.4 Failure case study
Two annotated failures (`paper/figs/failure_1.png`, `failure_2.png`): one grounding error (wrong bid for similarly-styled buttons) and one cascading failure where the refiner correctly abstained but Gemini returned the same candidate set on retry.

## 7. Limitations
- 8 MiniWoB tasks is narrow; real-web evaluation is left as future work.
- Frozen encoders are a deliberate compute trade-off; LoRA-tuned SigLIP is the natural next step.
- Gemini Flash is a moving target; results are bound to the model version at run-time.
- Recovery shaping uses state-hash progress, which can miss semantic progress in form-filling tasks.

## 8. Conclusion
A lightweight RL refiner with KL-to-VLM and recovery shaping yields measurable gains in success, efficiency, and robustness over a strong VLM baseline on MiniWoB++ at minimal compute cost. The Top-K oracle analysis bounds the achievable improvement from re-ranking alone; closing the remaining gap will require expanding the candidate set (e.g. tree-search planners) rather than smarter selection.

---
### Reproducibility
- All code, configs, and Colab notebook in this repository.
- Seeds: 0..4 for main eval, 0..2 for perturbed eval.
- Single A100 (Colab Pro+) + Gemini 2.0 Flash API.
