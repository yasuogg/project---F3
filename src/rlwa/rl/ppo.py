"""PPO training loop (CleanRL-flavored, single-file)."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, List
import time
import math
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rlwa.rl.policy import RefinerPolicy
from rlwa.rl.buffer import RolloutBuffer
from rlwa.rl.featurize import N_META_FEAT, candidate_text, candidate_meta_vec, history_to_text
from rlwa.obs.encoders import FrozenEncoders
from rlwa.agents.action_space import build_observation, META_ACTIONS
from rlwa.utils.schemas import ActionCandidate
from rlwa.utils.logging import info, ok, warn

import gymnasium as gym


class PPOTrainer:
    def __init__(self, cfg, env_fns: List[Callable[[], gym.Env]], planner, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.planner = planner
        self.K = int(cfg.policy.n_candidates)
        self.n_meta = int(cfg.policy.n_meta_actions)

        self.encoders = FrozenEncoders(
            image_model=cfg.encoder.image_model,
            text_model=cfg.encoder.text_model,
            device=device,
        )
        self.policy = RefinerPolicy(
            img_dim=self.encoders.img_dim,
            txt_dim=self.encoders.txt_dim,
            n_candidates=self.K,
            n_meta_actions=self.n_meta,
            n_meta_feat=N_META_FEAT,
            hidden=int(cfg.encoder.hidden_dim),
            dropout=float(cfg.policy.dropout),
        ).to(device)

        self.envs = [fn() for fn in env_fns]
        self.N = len(self.envs)
        self.T = int(cfg.train.rollout_len)
        self.total_steps = int(cfg.train.total_env_steps)
        self.gamma = float(cfg.train.gamma)
        self.lam = float(cfg.train.lam)
        self.clip_eps = float(cfg.train.clip_eps)
        self.ent_coef = float(cfg.train.ent_coef)
        self.vf_coef = float(cfg.train.vf_coef)
        self.max_grad = float(cfg.train.max_grad_norm)
        self.epochs = int(cfg.train.ppo_epochs)
        self.mb = int(cfg.train.minibatch_size)
        self.kl_start = float(cfg.train.kl_to_vlm)
        self.kl_end = float(cfg.train.kl_anneal_end)
        self.kl_anneal_steps = int(cfg.train.kl_anneal_steps)
        self.save_dir = Path(cfg.train.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(cfg.train.save_every)

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=float(cfg.train.lr))

        # auto-resume support
        self.global_step = 0
        self._resume()

        # per-env state
        self._raw_obs = []
        self._histories: list[list[dict]] = [[] for _ in self.envs]
        self._last_err: list[str | None] = [None] * self.N
        self._cands_cache: list[list[ActionCandidate]] = [[] for _ in self.envs]
        for env in self.envs:
            raw, _ = env.reset()
            self._raw_obs.append(raw)

    def _resume(self):
        last = self.save_dir / "ppo_last.pt"
        if last.exists():
            try:
                sd = torch.load(last, map_location=self.device)
                self.policy.load_state_dict(sd["policy"], strict=False)
                if "optimizer" in sd:
                    self.opt.load_state_dict(sd["optimizer"])
                self.global_step = int(sd.get("global_step", 0))
                info(f"Resumed from {last} @ step {self.global_step}")
            except Exception as e:
                warn(f"Resume failed ({e}); starting from scratch")

    # ---------- featurization ----------
    @torch.no_grad()
    def _featurize(self, env_idx: int, cands_override=None):
        raw = self._raw_obs[env_idx]
        obs = build_observation(raw)
        if cands_override is not None:
            cands = cands_override
        else:
            cands = self.planner.propose(
                goal=obs["goal"],
                som_image=obs["som_image"],
                axtree=obs["axtree"],
                mark_bids=obs["mark_bids"],
                history=self._histories[env_idx],
                last_error=self._last_err[env_idx],
            )
        # pad / truncate to exactly K
        if len(cands) < self.K:
            cands = cands + [ActionCandidate(action_type="noop", p=0.0) for _ in range(self.K - len(cands))]
        cands = cands[: self.K]
        self._cands_cache[env_idx] = cands

        img_emb = self.encoders.encode_image(obs["som_image"])                            # (img_dim,)
        goal_emb = self.encoders.encode_texts([obs["goal"]])[0]                            # (txt_dim,)
        hist_emb = self.encoders.encode_texts([history_to_text(self._histories[env_idx])])[0]
        cand_texts = [candidate_text(c) for c in cands]
        cand_emb = self.encoders.encode_texts(cand_texts)                                   # (K, txt_dim)
        cand_prior = torch.tensor([c.p for c in cands], device=self.device).float()
        cand_meta = torch.stack([candidate_meta_vec(c, device=self.device) for c in cands])
        cand_mask = torch.tensor([1.0 if c.action_type != "noop" or i == 0 else 0.0
                                  for i, c in enumerate(cands)], device=self.device).float()
        # VLM prior over (K + n_meta): meta-actions get tiny uniform mass
        meta_floor = 0.02
        vlm_full = torch.cat([cand_prior, torch.full((self.n_meta,), meta_floor, device=self.device)])
        vlm_full = vlm_full / vlm_full.sum().clamp_min(1e-6)
        return obs, cands, img_emb, goal_emb, hist_emb, cand_emb, cand_prior, cand_meta, cand_mask, vlm_full

    def _batch_propose(self):
        """Concurrently call planner.propose for all N envs (if available)."""
        reqs = []
        obs_cache = []
        for i in range(self.N):
            o = build_observation(self._raw_obs[i])
            obs_cache.append(o)
            reqs.append(dict(
                goal=o["goal"], som_image=o["som_image"], axtree=o["axtree"],
                mark_bids=o["mark_bids"], history=self._histories[i],
                last_error=self._last_err[i],
            ))
        if hasattr(self.planner, "propose_batch"):
            cands_list = self.planner.propose_batch(reqs)
        else:
            cands_list = [self.planner.propose(**r) for r in reqs]
        return cands_list

    # ---------- one env-step ----------
    def _apply_action(self, env_idx: int, action_id: int):
        env = self.envs[env_idx]
        cands = self._cands_cache[env_idx]
        info_extra: dict = {}
        if action_id < self.K:
            chosen = cands[action_id]
            action_str = chosen.to_browsergym_action()
        else:
            meta = META_ACTIONS[action_id - self.K]
            if meta == "ABSTAIN":
                # re-query planner with recovery prompt forced ON
                prev_err = self._last_err[env_idx]
                self._last_err[env_idx] = prev_err or "abstain: please reconsider"
                # re-featurize will happen at next step; here we just no-op once
                action_str = 'noop()'
                info_extra["abstain"] = True
            else:
                action_str = 'report_infeasible("agent gave up")'

        try:
            raw, reward, terminated, truncated, info = env.step(action_str)
            err_msg = info.get("last_action_error") or info.get("action_error")
        except Exception as e:
            raw = self._raw_obs[env_idx]
            reward, terminated, truncated, info = 0.0, False, False, {"action_error": str(e)}
            err_msg = str(e)

        failed = bool(err_msg)
        self._histories[env_idx].append({
            "step": len(self._histories[env_idx]),
            "action": action_str,
            "reward": float(reward),
            "failed": failed,
        })
        self._last_err[env_idx] = err_msg

        done = bool(terminated or truncated)
        if done:
            raw, _ = env.reset()
            self._histories[env_idx] = []
            self._last_err[env_idx] = None

        self._raw_obs[env_idx] = raw
        return float(reward), done, info

    # ---------- main loop ----------
    def train(self, wandb_run=None):
        buf = RolloutBuffer(
            T=self.T, N=self.N, K=self.K, n_meta=self.n_meta,
            img_dim=self.encoders.img_dim, txt_dim=self.encoders.txt_dim,
            n_meta_feat=N_META_FEAT, device=self.device,
        )
        n_updates = math.ceil(self.total_steps / (self.T * self.N))
        info(f"Starting PPO: {n_updates} updates, T={self.T}, N={self.N}, total≈{self.T*self.N*n_updates} steps")
        # open jsonl log for plotting
        log_path = self.save_dir / "ppo_log.jsonl"
        log_f = open(log_path, "a", buffering=1)

        ep_returns = [0.0] * self.N
        ep_successes: list[float] = []
        ep_rew_window: list[float] = []
        t0 = time.time()

        for upd in range(n_updates):
            buf.reset()
            self.policy.eval()
            for t in range(self.T):
                # batch-call planner across N envs (concurrent if AsyncGeminiPlanner)
                cands_batch = self._batch_propose()
                # featurize each env
                img_b, goal_b, hist_b = [], [], []
                cemb_b, cprior_b, cmeta_b, cmask_b, vlmprior_b = [], [], [], [], []
                for i in range(self.N):
                    _, _, im, go, hi, ce, cp, cm, cms, vlm = self._featurize(i, cands_override=cands_batch[i])
                    img_b.append(im); goal_b.append(go); hist_b.append(hi)
                    cemb_b.append(ce); cprior_b.append(cp); cmeta_b.append(cm)
                    cmask_b.append(cms); vlmprior_b.append(vlm)
                img_b = torch.stack(img_b); goal_b = torch.stack(goal_b); hist_b = torch.stack(hist_b)
                cemb_b = torch.stack(cemb_b); cprior_b = torch.stack(cprior_b)
                cmeta_b = torch.stack(cmeta_b); cmask_b = torch.stack(cmask_b)
                vlmprior_b = torch.stack(vlmprior_b)

                with torch.no_grad():
                    logits, value = self.policy(img_b, goal_b, hist_b, cemb_b,
                                                 cprior_b, cmeta_b, cmask_b)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)

                rewards = torch.zeros(self.N, device=self.device)
                dones = torch.zeros(self.N, device=self.device)
                for i in range(self.N):
                    r, done, _ = self._apply_action(i, int(action[i].item()))
                    rewards[i] = r
                    dones[i] = float(done)
                    ep_returns[i] += r
                    if done:
                        ep_rew_window.append(ep_returns[i])
                        ep_successes.append(1.0 if ep_returns[i] > 0.5 else 0.0)
                        ep_returns[i] = 0.0

                buf.add(img=img_b, goal=goal_b, hist=hist_b,
                        cand_emb=cemb_b, cand_prior=cprior_b, cand_meta=cmeta_b,
                        cand_mask=cmask_b, action=action, logprob=logp, value=value,
                        reward=rewards, done=dones, vlm_prior=vlmprior_b)
                self.global_step += self.N

            # last value for GAE
            cands_batch = self._batch_propose()
            img_b, goal_b, hist_b = [], [], []
            cemb_b, cprior_b, cmeta_b, cmask_b = [], [], [], []
            for i in range(self.N):
                _, _, im, go, hi, ce, cp, cm, cms, _ = self._featurize(i, cands_override=cands_batch[i])
                img_b.append(im); goal_b.append(go); hist_b.append(hi)
                cemb_b.append(ce); cprior_b.append(cp); cmeta_b.append(cm); cmask_b.append(cms)
            with torch.no_grad():
                _, last_v = self.policy(
                    torch.stack(img_b), torch.stack(goal_b), torch.stack(hist_b),
                    torch.stack(cemb_b), torch.stack(cprior_b),
                    torch.stack(cmeta_b), torch.stack(cmask_b),
                )

            adv, ret = buf.compute_gae(last_v, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # current KL coef (linear anneal)
            frac = min(1.0, self.global_step / max(1, self.kl_anneal_steps))
            kl_coef = self.kl_start + (self.kl_end - self.kl_start) * frac

            self.policy.train()
            total_loss_log = 0.0
            for _ in range(self.epochs):
                for mb in buf.iter_minibatches(adv, ret, self.mb):
                    logits, value = self.policy(
                        mb["img"], mb["goal"], mb["hist"],
                        mb["cand_emb"], mb["cand_prior"], mb["cand_meta"], mb["cand_mask"],
                    )
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(mb["actions"])
                    ratio = torch.exp(new_logp - mb["logprobs"])
                    s1 = ratio * mb["adv"]
                    s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb["adv"]
                    loss_pi = -torch.min(s1, s2).mean()
                    loss_v = 0.5 * (value - mb["ret"]).pow(2).mean()
                    loss_ent = -dist.entropy().mean()
                    # KL(policy || VLM prior)
                    log_pi = F.log_softmax(logits, dim=-1)
                    log_q = torch.log(mb["vlm_prior"].clamp_min(1e-6))
                    kl = (log_pi.exp() * (log_pi - log_q)).sum(dim=-1).mean()

                    loss = loss_pi + self.vf_coef * loss_v + self.ent_coef * loss_ent + kl_coef * kl
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                    self.opt.step()
                    total_loss_log = float(loss.item())

            sr = (sum(ep_successes[-50:]) / max(1, len(ep_successes[-50:]))) if ep_successes else 0.0
            mean_r = (sum(ep_rew_window[-50:]) / max(1, len(ep_rew_window[-50:]))) if ep_rew_window else 0.0
            elapsed = time.time() - t0
            sps = self.global_step / max(1.0, elapsed)
            info(f"upd {upd+1}/{n_updates}  step={self.global_step}  loss={total_loss_log:.3f}  "
                 f"SR50={sr:.2f}  R50={mean_r:.2f}  kl_c={kl_coef:.3f}  sps={sps:.1f}")
            import json as _json
            log_f.write(_json.dumps({"upd": upd + 1, "step": int(self.global_step),
                                      "loss": float(total_loss_log), "sr_50": float(sr),
                                      "mean_r_50": float(mean_r), "kl_coef": float(kl_coef),
                                      "sps": float(sps)}) + "\n")
            if wandb_run is not None:
                wandb_run.log({
                    "loss": total_loss_log, "sr_50": sr, "mean_r_50": mean_r,
                    "kl_coef": kl_coef, "global_step": self.global_step, "sps": sps,
                })

            if (upd + 1) * self.T * self.N % self.save_every < self.T * self.N:
                self.save(self.save_dir / "ppo_last.pt")

        self.save(self.save_dir / "ppo_best.pt")
        log_f.close()
        ok(f"PPO done. Checkpoint -> {self.save_dir/'ppo_best.pt'}")

    def save(self, path: Path):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.opt.state_dict(),
            "global_step": int(self.global_step),
            "cfg": dict(self.cfg),
        }, path)
