"""Shared data schemas (pydantic)."""
from __future__ import annotations
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field


class ActionCandidate(BaseModel):
    """One action proposal from the VLM planner."""
    action_type: Literal[
        "click", "fill", "select_option", "press", "scroll",
        "hover", "report_done", "report_infeasible", "noop"
    ]
    bid: Optional[str] = None       # target element id (None for global actions)
    text: Optional[str] = None      # for fill / select_option / press / report_*
    p: float = 0.0                  # VLM-self-reported probability
    rationale: str = ""             # short reasoning string

    def to_browsergym_action(self) -> str:
        """Render as a BrowserGym high_level action string."""
        a = self.action_type
        if a == "click":
            return f'click("{self.bid}")'
        if a == "fill":
            return f'fill("{self.bid}", {self.text!r})'
        if a == "select_option":
            return f'select_option("{self.bid}", {self.text!r})'
        if a == "press":
            return f'keyboard_press({self.text!r})'
        if a == "scroll":
            return f"scroll(0, 200)"
        if a == "hover":
            return f'hover("{self.bid}")'
        if a == "report_done":
            return f'send_msg_to_user({(self.text or "done")!r})'
        if a == "report_infeasible":
            return f'report_infeasible({(self.text or "infeasible")!r})'
        return "noop()"


class StepRecord(BaseModel):
    """One env step, stored for trajectory replay + BC."""
    task: str
    seed: int
    step: int
    goal: str
    screenshot_path: Optional[str] = None
    axtree_snippet: str = ""
    candidates: List[ActionCandidate] = Field(default_factory=list)
    chosen_idx: int = 0
    action_str: str = ""
    reward: float = 0.0
    done: bool = False
    success: bool = False
    error: Optional[str] = None
    extra: dict = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    task: str
    seed: int
    steps: List[StepRecord] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    n_steps: int = 0
    n_invalid: int = 0
    n_recovered: int = 0      # successful step immediately after a failed one
