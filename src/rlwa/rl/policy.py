"""Refiner policy: takes fused state -> logits over K candidates + meta-actions."""
from __future__ import annotations
import torch
import torch.nn as nn


class RefinerPolicy(nn.Module):
    """
    State = [image_emb, goal_emb, hist_emb, K candidate_embs+meta]
    Output = logits over (K + n_meta) actions, and a scalar value.

    Inputs to forward():
        img_emb  : (B, img_dim)        SigLIP screenshot embedding
        goal_emb : (B, txt_dim)
        hist_emb : (B, txt_dim)        pooled history embedding
        cand_emb : (B, K, txt_dim)     per-candidate text embedding
        cand_prior: (B, K)             VLM-self-reported probs (log added as feature)
        cand_meta : (B, K, n_meta_feat) [one-hot action_type ...]
    """
    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        n_candidates: int = 5,
        n_meta_actions: int = 2,
        n_meta_feat: int = 8,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = n_candidates
        self.n_meta = n_meta_actions

        self.img_proj  = nn.Linear(img_dim, hidden)
        self.goal_proj = nn.Linear(txt_dim, hidden)
        self.hist_proj = nn.Linear(txt_dim, hidden)
        self.cand_proj = nn.Linear(txt_dim + n_meta_feat + 1, hidden)  # +1 for prior

        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # cross-feature: concat context with each candidate, score
        self.cand_score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # meta actions get their own logits from context
        self.meta_head = nn.Linear(hidden, n_meta_actions)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _context(self, img_emb, goal_emb, hist_emb):
        h = self.img_proj(img_emb) + self.goal_proj(goal_emb) + self.hist_proj(hist_emb)
        return self.fuse(h)  # (B, hidden)

    def forward(
        self,
        img_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        hist_emb: torch.Tensor,
        cand_emb: torch.Tensor,        # (B, K, txt_dim)
        cand_prior: torch.Tensor,      # (B, K)
        cand_meta: torch.Tensor,       # (B, K, n_meta_feat)
        cand_mask: torch.Tensor | None = None,  # (B, K) 1 if valid
    ):
        B = img_emb.shape[0]
        ctx = self._context(img_emb, goal_emb, hist_emb)  # (B, H)

        log_prior = torch.log(cand_prior.clamp_min(1e-6)).unsqueeze(-1)  # (B,K,1)
        cand_in = torch.cat([cand_emb, cand_meta, log_prior], dim=-1)    # (B,K, txt+meta+1)
        cand_h = self.cand_proj(cand_in)                                  # (B,K,H)

        ctx_exp = ctx.unsqueeze(1).expand(-1, self.K, -1)                # (B,K,H)
        joint = torch.cat([ctx_exp, cand_h], dim=-1)                     # (B,K,2H)
        cand_logits = self.cand_score(joint).squeeze(-1)                 # (B,K)

        meta_logits = self.meta_head(ctx)                                # (B, n_meta)
        logits = torch.cat([cand_logits, meta_logits], dim=-1)           # (B, K+n_meta)

        if cand_mask is not None:
            full_mask = torch.cat(
                [cand_mask, torch.ones(B, self.n_meta, device=logits.device)], dim=-1
            )
            logits = logits.masked_fill(full_mask < 0.5, float("-inf"))

        value = self.value_head(ctx).squeeze(-1)                         # (B,)
        return logits, value
