"""RL-refined agent: VLM proposes top-K, PPO policy picks final action."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.distributions import Categorical
import gymnasium as gym

from rlwa.planners import GeminiPlanner
from rlwa.obs.encoders import FrozenEncoders
from rlwa.rl.policy import RefinerPolicy
from rlwa.rl.featurize import N_META_FEAT, candidate_text, candidate_meta_vec, history_to_text
from rlwa.agents.action_space import build_observation, META_ACTIONS
from rlwa.utils.schemas import EpisodeRecord, StepRecord, ActionCandidate


class RLRefinerAgent:
    def __init__(self, cfg, ckpt_path: str | None = None, planner: GeminiPlanner | None = None,
                 device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.planner = planner or GeminiPlanner(cfg)
        self.K = int(cfg.policy.n_candidates)
        self.n_meta = int(cfg.policy.n_meta_actions)

        self.encoders = FrozenEncoders(
            image_model=cfg.encoder.image_model,
            text_model=cfg.encoder.text_model,
            device=device,
        )
        self.policy = RefinerPolicy(
            img_dim=self.encoders.img_dim, txt_dim=self.encoders.txt_dim,
            n_candidates=self.K, n_meta_actions=self.n_meta,
            n_meta_feat=N_META_FEAT, hidden=int(cfg.encoder.hidden_dim),
            dropout=float(cfg.policy.dropout),
        ).to(device).eval()

        ckpt_path = ckpt_path or cfg.get("checkpoint")
        if ckpt_path and Path(ckpt_path).exists():
            sd = torch.load(ckpt_path, map_location=device)
            self.policy.load_state_dict(sd["policy"])

    @torch.no_grad()
    def _choose(self, obs, history, last_error) -> tuple[int, list[ActionCandidate]]:
        cands = self.planner.propose(
            goal=obs["goal"], som_image=obs["som_image"], axtree=obs["axtree"],
            mark_bids=obs["mark_bids"], history=history, last_error=last_error,
        )
        if len(cands) < self.K:
            cands = cands + [ActionCandidate(action_type="noop", p=0.0) for _ in range(self.K - len(cands))]
        cands = cands[: self.K]

        img_emb = self.encoders.encode_image(obs["som_image"]).unsqueeze(0)
        goal_emb = self.encoders.encode_texts([obs["goal"]])
        hist_emb = self.encoders.encode_texts([history_to_text(history)])
        cand_emb = self.encoders.encode_texts([candidate_text(c) for c in cands]).unsqueeze(0)
        cand_prior = torch.tensor([[c.p for c in cands]], device=self.device).float()
        cand_meta = torch.stack([candidate_meta_vec(c, device=self.device) for c in cands]).unsqueeze(0)
        cand_mask = torch.ones(1, self.K, device=self.device)

        logits, _ = self.policy(img_emb, goal_emb, hist_emb, cand_emb, cand_prior, cand_meta, cand_mask)
        action_id = int(Categorical(logits=logits).probs.argmax(-1).item())  # greedy at eval
        return action_id, cands

    def run_episode(self, env: gym.Env, task: str, seed: int, max_steps: int = 25) -> EpisodeRecord:
        raw, _ = env.reset(seed=seed)
        ep = EpisodeRecord(task=task, seed=seed)
        history: list[dict] = []
        last_error: str | None = None

        for step in range(max_steps):
            obs = build_observation(raw)
            action_id, cands = self._choose(obs, history, last_error)
            if action_id < self.K:
                chosen = cands[action_id]
                action_str = chosen.to_browsergym_action()
                chosen_idx = action_id
            else:
                meta = META_ACTIONS[action_id - self.K]
                if meta == "ABSTAIN":
                    action_str = "noop()"
                    last_error = (last_error or "") + " [abstain]"
                else:
                    action_str = 'report_infeasible("agent gave up")'
                chosen_idx = action_id

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
            if history and history[-1].get("failed") and not failed: ep.n_recovered += 1

            history.append({"step": step, "action": action_str,
                            "reward": float(reward), "failed": failed})
            last_error = err_msg

            if terminated or truncated:
                ep.success = success or ep.success
                break
        return ep
