"""Abstract planner interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from rlwa.utils.schemas import ActionCandidate


class Planner(ABC):
    @abstractmethod
    def propose(
        self,
        goal: str,
        som_image: Image.Image,
        axtree: str,
        mark_bids: List[str],
        history: List[dict],
        last_error: str | None = None,
    ) -> List[ActionCandidate]:
        ...
