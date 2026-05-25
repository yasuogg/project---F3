"""Observation builder + helpers shared by all agents."""
from __future__ import annotations
from typing import Any, List, Tuple
import numpy as np
from PIL import Image

from rlwa.obs.som import render_som
from rlwa.obs.axtree_prune import prune_axtree, extract_bids


META_ACTIONS = ["ABSTAIN", "REPORT_INFEASIBLE"]


def build_observation(raw_obs: dict, max_marks: int = 30) -> dict:
    """Convert a raw BrowserGym obs into our composite observation."""
    screenshot = raw_obs.get("screenshot")
    if screenshot is None or not isinstance(screenshot, np.ndarray):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)

    axtree_txt = raw_obs.get("axtree_txt") or raw_obs.get("axtree") or ""
    bids_in_tree = extract_bids(axtree_txt)

    # element properties (browsergym exposes these per bid)
    elements = (
        raw_obs.get("extra_element_properties")
        or raw_obs.get("dom_element_properties")
        or {}
    )

    som_img, mark_bids = render_som(screenshot, elements, max_marks=max_marks)
    pruned = prune_axtree(axtree_txt, keep_bids=mark_bids or bids_in_tree)

    return {
        "screenshot": screenshot,
        "som_image": som_img,
        "axtree": pruned,
        "mark_bids": mark_bids,
        "goal": raw_obs.get("goal", "") or "",
        "url": raw_obs.get("url", ""),
    }
