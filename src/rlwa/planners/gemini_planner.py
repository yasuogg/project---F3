"""Gemini-based vision planner.

Uses google.genai with JSON response_mime_type for structured output.
"""
from __future__ import annotations
import io
import os
import json
import time
from typing import List
from PIL import Image

from rlwa.planners.base import Planner
from rlwa.planners.prompts import build_user_prompt
from rlwa.utils.schemas import ActionCandidate
from rlwa.utils.logging import warn


class GeminiPlanner(Planner):
    def __init__(self, cfg):
        from google import genai
        from google.genai import types
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY env var not set")
        self.cfg = cfg
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_name = cfg.planner.model
        self._build_config()
        self.top_k = int(cfg.planner.top_k_candidates)
        self.max_retries = int(cfg.planner.max_retries)
        self.backoff = float(cfg.planner.retry_backoff_s)
        self.enable_recovery = bool(cfg.planner.get("enable_recovery_prompt", True))

    def _build_config(self):
        self.gen_config = self.types.GenerateContentConfig(
            temperature=float(self.cfg.planner.temperature),
            top_p=float(self.cfg.planner.top_p),
            max_output_tokens=int(self.cfg.planner.max_output_tokens),
            response_mime_type="application/json",
        )

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name

    def propose(
        self,
        goal: str,
        som_image: Image.Image,
        axtree: str,
        mark_bids: List[str],
        history: List[dict],
        last_error: str | None = None,
    ) -> List[ActionCandidate]:
        prompt = build_user_prompt(
            goal=goal,
            axtree=axtree,
            mark_bids=mark_bids,
            history=history,
            top_k=self.top_k,
            last_error=last_error if self.enable_recovery else None,
        )
        buf = io.BytesIO()
        som_image.save(buf, format="PNG")
        image_part = self.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image_part],
                    config=self.gen_config,
                )
                text = (resp.text or "").strip()
                data = self._parse(text)
                cands = self._to_candidates(data, mark_bids)
                if cands:
                    return cands
            except Exception as e:
                last_exc = e
                warn(f"Gemini attempt {attempt+1}/{self.max_retries} failed: {e}")
                time.sleep(self.backoff * (2 ** attempt))
        warn(f"Planner gave up: {last_exc}")
        # safe fallback: do nothing
        return [ActionCandidate(action_type="noop", p=1.0, rationale="fallback")]

    @staticmethod
    def _parse(text: str) -> dict:
        # strip stray markdown fences if model ignores schema
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.startswith("json"):
                t = t[4:]
        return json.loads(t)

    def _to_candidates(self, data: dict, mark_bids: List[str]) -> List[ActionCandidate]:
        bid_set = set(mark_bids)
        cands: List[ActionCandidate] = []
        for c in (data.get("candidates") or [])[: self.top_k]:
            try:
                cand = ActionCandidate(**c)
            except Exception:
                continue
            # if planner returned a mark number instead of bid, try to map it
            if cand.bid and cand.bid.isdigit() and cand.bid not in bid_set:
                idx = int(cand.bid) - 1
                if 0 <= idx < len(mark_bids):
                    cand.bid = mark_bids[idx]
            # validate bid for targeted actions
            if cand.action_type in {"click", "fill", "select_option", "hover"} and cand.bid not in bid_set:
                continue
            cands.append(cand)
        # pad with noop if empty so downstream code never breaks
        while len(cands) < self.top_k:
            cands.append(ActionCandidate(action_type="noop", p=0.0, rationale="pad"))
        # normalize p
        s = sum(max(c.p, 0.0) for c in cands) or 1.0
        for c in cands:
            c.p = max(c.p, 0.0) / s
        return cands
