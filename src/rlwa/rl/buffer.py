"""Rollout buffer for PPO with GAE."""
from __future__ import annotations
import torch


class RolloutBuffer:
    """
    Stores T*N transitions. All tensors live on the chosen device.
    Action space = K candidates + n_meta meta-actions.
    """
    def __init__(self, T: int, N: int, K: int, n_meta: int,
                 img_dim: int, txt_dim: int, n_meta_feat: int, device: str = "cuda"):
        self.T, self.N, self.K, self.n_meta = T, N, K, n_meta
        self.device = device

        z = lambda *s: torch.zeros(*s, device=device)
        self.img       = z(T, N, img_dim)
        self.goal      = z(T, N, txt_dim)
        self.hist      = z(T, N, txt_dim)
        self.cand_emb  = z(T, N, K, txt_dim)
        self.cand_prior= z(T, N, K)
        self.cand_meta = z(T, N, K, n_meta_feat)
        self.cand_mask = z(T, N, K)

        self.actions   = torch.zeros(T, N, dtype=torch.long, device=device)
        self.logprobs  = z(T, N)
        self.values    = z(T, N)
        self.rewards   = z(T, N)
        self.dones     = z(T, N)
        self.vlm_prior = z(T, N, K + n_meta)   # logged for KL-to-VLM

        self.ptr = 0

    def add(self, *, img, goal, hist, cand_emb, cand_prior, cand_meta, cand_mask,
            action, logprob, value, reward, done, vlm_prior):
        t = self.ptr
        self.img[t] = img
        self.goal[t] = goal
        self.hist[t] = hist
        self.cand_emb[t] = cand_emb
        self.cand_prior[t] = cand_prior
        self.cand_meta[t] = cand_meta
        self.cand_mask[t] = cand_mask
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.vlm_prior[t] = vlm_prior
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        adv = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            adv[t] = last_gae
        ret = adv + self.values
        return adv, ret

    def iter_minibatches(self, advantages, returns, mb_size: int):
        T, N = self.T, self.N
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        data = {
            "img":        flat(self.img),
            "goal":       flat(self.goal),
            "hist":       flat(self.hist),
            "cand_emb":   flat(self.cand_emb),
            "cand_prior": flat(self.cand_prior),
            "cand_meta":  flat(self.cand_meta),
            "cand_mask":  flat(self.cand_mask),
            "actions":    flat(self.actions),
            "logprobs":   flat(self.logprobs),
            "values":     flat(self.values),
            "vlm_prior":  flat(self.vlm_prior),
            "adv":        flat(advantages),
            "ret":        flat(returns),
        }
        n = T * N
        idx = torch.randperm(n, device=self.device)
        for s in range(0, n, mb_size):
            sl = idx[s:s + mb_size]
            yield {k: v[sl] for k, v in data.items()}

    def reset(self):
        self.ptr = 0
