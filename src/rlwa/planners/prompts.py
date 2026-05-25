"""Prompt templates for the VLM web agent."""

SYSTEM_PROMPT = """You are a precise web agent. You see (1) a screenshot with numbered colored \
bounding boxes (Set-of-Mark), where mark N corresponds to the Nth interactable element \
listed in the accessibility tree below, and (2) a pruned accessibility tree where each \
interactable element is prefixed with [bid].

GOAL: {goal}

ACCESSIBILITY TREE (pruned):
{axtree}

MARK -> BID MAPPING:
{mark_map}

RECENT HISTORY (last {hist_len} steps):
{history}
{recovery_block}
AVAILABLE ACTIONS:
  click(bid)
  fill(bid, "text")
  select_option(bid, "value")
  press("key")             # key in {{Enter, Tab, ArrowDown, ArrowUp, Escape}}
  scroll(0, 200)           # only dy supported here
  hover(bid)
  report_done("answer")    # call when goal is achieved
  report_infeasible("why") # call only after exhausting options

Return ONLY a JSON object (no markdown) with this exact schema:
{{
  "thought": "1-2 sentences of reasoning",
  "candidates": [
    {{"action_type": "click", "bid": "42", "text": null, "p": 0.55, "rationale": "..."}},
    ... up to {top_k} candidates, p values summing to ~1.0
  ]
}}

Rules:
- bid MUST be one of the bids shown in the MARK -> BID MAPPING (use the bid string, not the mark number).
- For non-targeted actions (press, scroll, report_*) set bid = null.
- For fill / select_option / press / report_*, set text to the string argument.
- Sort candidates by descending p.
"""

RECOVERY_BLOCK = """
⚠ PREVIOUS ACTION FAILED: {last_error}
Diagnose: was the bid wrong? page still loading? need a different element?
Propose a CORRECTIVE action as your top candidate.
"""


def build_user_prompt(
    goal: str,
    axtree: str,
    mark_bids: list[str],
    history: list[dict],
    top_k: int,
    last_error: str | None,
) -> str:
    mark_map = "\n".join(f"  {i+1} -> bid {b}" for i, b in enumerate(mark_bids)) or "  (none)"
    if history:
        hist_lines = []
        for h in history[-5:]:
            ok = "✓" if not h.get("failed") else "✗"
            hist_lines.append(f"  [{h['step']}] {ok} {h['action']}  reward={h.get('reward', 0):+.2f}")
        hist_txt = "\n".join(hist_lines)
    else:
        hist_txt = "  (no prior steps)"
    rec = RECOVERY_BLOCK.format(last_error=last_error) if last_error else ""
    return SYSTEM_PROMPT.format(
        goal=goal,
        axtree=axtree or "(empty)",
        mark_map=mark_map,
        hist_len=min(5, len(history)),
        history=hist_txt,
        recovery_block=rec,
        top_k=top_k,
    )
