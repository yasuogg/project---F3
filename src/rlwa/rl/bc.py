"""Behavior cloning pretrain on collected prompt-only success traces.

Each training example: (state features, expert action index in K+meta)
Where the "expert" = the action the prompt-only agent took in a SUCCESSFUL episode
(index = 0 because prompt-only picks top-1 candidate). We treat that as the gold
label for the refiner policy head.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import json
import torch
import torch.nn.functional as F

from rlwa.rl.policy import RefinerPolicy
from rlwa.rl.featurize import N_META_FEAT, candidate_text, candidate_meta_vec, history_to_text
from rlwa.obs.encoders import FrozenEncoders
from rlwa.utils.logging import info, ok


def _load_traces(path: str | Path, success_only: bool = True) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if success_only and not ep.get("success"):
                continue
            out.append(ep)
    return out


def train_bc(cfg, device: str = "cuda"):
    enc = FrozenEncoders(
        image_model=cfg.encoder.image_model,
        text_model=cfg.encoder.text_model,
        device=device,
    )
    K = int(cfg.policy.n_candidates)
    n_meta = int(cfg.policy.n_meta_actions)
    policy = RefinerPolicy(
        img_dim=enc.img_dim, txt_dim=enc.txt_dim,
        n_candidates=K, n_meta_actions=n_meta,
        n_meta_feat=N_META_FEAT, hidden=int(cfg.encoder.hidden_dim),
        dropout=float(cfg.policy.dropout),
    ).to(device)

    opt = torch.optim.AdamW(policy.parameters(), lr=float(cfg.bc.lr))
    info(f"Loading traces from {cfg.bc.traces}")
    eps = _load_traces(cfg.bc.traces)
    info(f"{len(eps)} successful episodes")

    # build (text-only) examples; we re-encode on the fly to save disk
    examples: list[dict] = []
    for ep in eps:
        for st in ep["steps"]:
            cands = st["candidates"]
            if len(cands) < K:
                cands = cands + [{"action_type": "noop", "p": 0.0, "rationale": "pad",
                                  "bid": None, "text": None}] * (K - len(cands))
            cands = cands[:K]
            examples.append({
                "goal": st["goal"],
                "axtree": st["axtree_snippet"],
                "cands": cands,
                "label": int(st["chosen_idx"]),
            })
    info(f"{len(examples)} step examples for BC")

    bs = int(cfg.bc.batch_size)
    n_epochs = int(cfg.bc.epochs)

    def collate(batch):
        # text-only features (no image) — use goal embedding twice as fallback
        goals = [b["goal"] for b in batch]
        hists = [b["axtree"][:300] for b in batch]
        goal_emb = enc.encode_texts(goals)
        hist_emb = enc.encode_texts(hists)
        # use goal_emb as "image" placeholder (BC doesn't have screenshots)
        img_emb = goal_emb.clone()

        cand_texts = []
        for b in batch:
            for c in b["cands"]:
                cand_texts.append(candidate_text(__obj_from_dict(c)))
        cand_emb_all = enc.encode_texts(cand_texts).view(len(batch), K, -1)

        cand_prior = torch.tensor([[c["p"] for c in b["cands"]] for b in batch],
                                  device=device).float()
        cand_meta = torch.stack([
            torch.stack([candidate_meta_vec(__obj_from_dict(c), device=device)
                         for c in b["cands"]])
            for b in batch
        ])
        cand_mask = torch.ones(len(batch), K, device=device)
        labels = torch.tensor([b["label"] for b in batch], device=device, dtype=torch.long)
        return img_emb, goal_emb, hist_emb, cand_emb_all, cand_prior, cand_meta, cand_mask, labels

    import random
    for epoch in range(n_epochs):
        random.shuffle(examples)
        total = 0.0; correct = 0; seen = 0
        for s in range(0, len(examples), bs):
            batch = examples[s:s + bs]
            img, go, hi, ce, cp, cm, cms, y = collate(batch)
            logits, _ = policy(img, go, hi, ce, cp, cm, cms)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * len(batch)
            correct += int((logits.argmax(-1) == y).sum().item())
            seen += len(batch)
        info(f"BC epoch {epoch+1}: loss={total/max(1,seen):.4f}  acc={correct/max(1,seen):.3f}")

    out = Path(cfg.bc.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict()}, out)
    ok(f"BC checkpoint -> {out}")
    return out


# small helper: pydantic-free dict -> ActionCandidate-like
class _Dummy:
    def __init__(self, d): self.__dict__.update(d)

def __obj_from_dict(d: dict):
    return _Dummy({
        "action_type": d.get("action_type", "noop"),
        "bid": d.get("bid"),
        "text": d.get("text"),
        "p": float(d.get("p", 0.0)),
        "rationale": d.get("rationale", ""),
    })
