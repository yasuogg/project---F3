"""Helpers to featurize a candidate (one-hot type + meta + prior)."""
from __future__ import annotations
import torch
from typing import List
from rlwa.utils.schemas import ActionCandidate

ACTION_TYPES = [
    "click", "fill", "select_option", "press", "scroll",
    "hover", "report_done", "report_infeasible",
]
N_META_FEAT = len(ACTION_TYPES)


def candidate_text(c: ActionCandidate) -> str:
    """Render a short text rep of a candidate for the text encoder."""
    parts = [c.action_type]
    if c.bid:
        parts.append(f"bid={c.bid}")
    if c.text:
        parts.append(f'"{c.text[:60]}"')
    if c.rationale:
        parts.append(f"// {c.rationale[:80]}")
    return " ".join(parts)


def candidate_meta_vec(c: ActionCandidate, device: str = "cpu") -> torch.Tensor:
    """One-hot action_type vector of length N_META_FEAT."""
    v = torch.zeros(N_META_FEAT, device=device)
    if c.action_type in ACTION_TYPES:
        v[ACTION_TYPES.index(c.action_type)] = 1.0
    return v


def history_to_text(history: List[dict], k: int = 5) -> str:
    if not history:
        return "no prior actions"
    parts = []
    for h in history[-k:]:
        ok = "OK" if not h.get("failed") else "FAIL"
        parts.append(f"step{h['step']}:{ok}:{h['action']}")
    return " | ".join(parts)
