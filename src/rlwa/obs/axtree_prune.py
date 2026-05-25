"""Accessibility-tree pruning + bid extraction."""
from __future__ import annotations
import re
from typing import List, Tuple

# matches BrowserGym ax-tree lines like:
#   [42] button "Submit" focused
_BID_RE = re.compile(r"\[(\d+)\]")


def extract_bids(axtree_txt: str) -> List[str]:
    """Return all bids appearing in the ax-tree text in order."""
    return _BID_RE.findall(axtree_txt or "")


def prune_axtree(
    axtree_txt: str,
    keep_bids: List[str] | None = None,
    max_lines: int = 80,
    max_chars: int = 4000,
) -> str:
    """Keep only interactable/relevant lines, cap length."""
    if not axtree_txt:
        return ""
    lines = axtree_txt.splitlines()
    kept: List[str] = []
    keep_set = set(keep_bids) if keep_bids else None

    INTERACTIVE = ("button", "link", "textbox", "checkbox", "radio",
                   "combobox", "menuitem", "tab", "listitem", "searchbox",
                   "switch", "option", "image", "dialog", "heading",
                   "StaticText")
    for ln in lines:
        ln_s = ln.strip()
        if not ln_s:
            continue
        # always keep lines with a bid
        if "[" in ln_s and "]" in ln_s:
            if keep_set is not None:
                m = _BID_RE.search(ln_s)
                if m and m.group(1) not in keep_set:
                    continue
            kept.append(ln)
            continue
        # keep semantically useful types
        if any(t in ln_s for t in INTERACTIVE):
            kept.append(ln)

    kept = kept[:max_lines]
    out = "\n".join(kept)
    if len(out) > max_chars:
        out = out[: max_chars - 20] + "\n... [truncated]"
    return out
