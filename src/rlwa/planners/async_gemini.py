"""Async/threaded wrapper around GeminiPlanner for high-concurrency calls.

Usage:
    planner = GeminiPlanner(cfg)
    async_planner = AsyncGeminiPlanner(planner, max_workers=32)
    results = async_planner.propose_batch([
        dict(goal=..., som_image=..., axtree=..., mark_bids=..., history=..., last_error=...),
        ...
    ])

The underlying SDK is thread-safe and releases the GIL during HTTP I/O,
so a ThreadPoolExecutor gives near-linear scaling up to the API's effective
concurrency limit (Tier 3: ~130 in flight at ~2s latency).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from PIL import Image

from rlwa.planners.gemini_planner import GeminiPlanner
from rlwa.planners.base import Planner
from rlwa.utils.schemas import ActionCandidate


class AsyncGeminiPlanner(Planner):
    def __init__(self, base: GeminiPlanner, max_workers: int = 32):
        self.base = base
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gemini")
        self.max_workers = max_workers

    # passthroughs
    @property
    def cfg(self): return self.base.cfg
    @property
    def top_k(self): return self.base.top_k

    def propose(self, *args, **kwargs) -> List[ActionCandidate]:
        return self.base.propose(*args, **kwargs)

    def propose_batch(self, requests: List[dict]) -> List[List[ActionCandidate]]:
        """Run N propose() calls concurrently. Each request is a kwargs dict."""
        futures = [self.executor.submit(self.base.propose, **r) for r in requests]
        return [f.result() for f in futures]

    def close(self):
        self.executor.shutdown(wait=False)
