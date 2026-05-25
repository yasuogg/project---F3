"""Observation builder + helpers shared by all agents."""
from __future__ import annotations
from typing import Any, List, Tuple
import numpy as np
from PIL import Image

from rlwa.obs.som import render_som
from rlwa.obs.axtree_prune import prune_axtree, extract_bids


META_ACTIONS = ["ABSTAIN", "REPORT_INFEASIBLE"]


def _axtree_to_text(raw_obs: dict) -> str:
    """BrowserGym >=0.13 returns `axtree_object` (a dict); older paths returned
    `axtree_txt`. Try the text fields first, then flatten the object."""
    txt = raw_obs.get("axtree_txt") or raw_obs.get("axtree") or ""
    if txt:
        return txt
    obj = raw_obs.get("axtree_object")
    if not obj:
        return ""
    try:
        from browsergym.utils.obs import flatten_axtree_to_str  # type: ignore
        return flatten_axtree_to_str(obj) or ""
    except Exception:
        # last-ditch: render nodes ourselves
        try:
            nodes = obj.get("nodes", []) if isinstance(obj, dict) else []
            lines = []
            for n in nodes:
                role = (n.get("role") or {}).get("value", "")
                name = (n.get("name") or {}).get("value", "")
                bid = (n.get("browsergym_id") or n.get("bid") or "")
                if role in {"none", "generic", ""} and not name:
                    continue
                tag = f"[{bid}] " if bid else ""
                lines.append(f"  {tag}{role} {name!r}".rstrip())
            return "\n".join(lines)
        except Exception:
            return ""


def build_observation(raw_obs: dict, max_marks: int = 30) -> dict:
    """Convert a raw BrowserGym obs into our composite observation."""
    screenshot = raw_obs.get("screenshot")
    if screenshot is None or not isinstance(screenshot, np.ndarray):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)

    axtree_txt = _axtree_to_text(raw_obs)
    bids_in_tree = extract_bids(axtree_txt)

    # element properties (browsergym exposes these per bid)
    elements = (
        raw_obs.get("extra_element_properties")
        or raw_obs.get("dom_element_properties")
        or {}
    )

    som_img, mark_bids = render_som(screenshot, elements, max_marks=max_marks)
    pruned = prune_axtree(axtree_txt, keep_bids=mark_bids or bids_in_tree)

    # goal may be plain string ('goal') or a list of message dicts ('goal_object')
    goal = raw_obs.get("goal")
    if not goal:
        go = raw_obs.get("goal_object") or []
        if isinstance(go, list):
            parts = []
            for m in go:
                if isinstance(m, dict):
                    parts.append(str(m.get("text") or m.get("content") or ""))
                else:
                    parts.append(str(m))
            goal = " ".join(p for p in parts if p)
        else:
            goal = str(go)

    return {
        "screenshot": screenshot,
        "som_image": som_img,
        "axtree": pruned,
        "mark_bids": mark_bids,
        "goal": goal or "",
        "url": raw_obs.get("url", ""),
    }
