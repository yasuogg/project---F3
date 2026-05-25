"""Auto-fill report_template.md with numbers from data/eval/*_summary.json.

Reads:
  data/eval/prompt_only_summary.json
  data/eval/reflective_summary.json
  data/eval/bc_only_summary.json
  data/eval/rl_refiner_summary.json
  data/eval/random_summary.json
  data/eval/perturb_*_summary.json
  data/eval/oracle_topk_summary.json
  data/eval/ablations/*_summary.json

Substitutes placeholders of the form {{prompt_only.sr}} etc.
Writes paper/report.md.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import tyro

from rlwa.utils.logging import info, ok, warn


def _load_summaries(eval_dir: Path) -> dict:
    out = {}
    for p in eval_dir.glob("*_summary.json"):
        key = p.stem.replace("_summary", "")
        try:
            out[key] = json.loads(p.read_text())
        except Exception as e:
            warn(f"failed to read {p}: {e}")
    abl_dir = eval_dir / "ablations"
    if abl_dir.is_dir():
        for p in abl_dir.glob("*_summary.json"):
            key = "ablation_" + p.stem.replace("_summary", "")
            out[key] = json.loads(p.read_text())
    return out


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_.\-]+)\s*\}\}")


def _resolve(key: str, data: dict) -> str:
    """key like 'rl_refiner.sr' or 'rl_refiner.per_task.click-test'."""
    parts = key.split(".")
    cur = data
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return "—"
    if isinstance(cur, float):
        return f"{cur:.3f}"
    return str(cur)


def main(
    template: str = "paper/report_template.md",
    out: str = "paper/report.md",
    eval_dir: str = "data/eval",
):
    tmpl_path = Path(template); out_path = Path(out)
    if not tmpl_path.exists():
        warn(f"template {tmpl_path} missing"); return
    data = _load_summaries(Path(eval_dir))
    info(f"loaded summaries: {list(data.keys())}")

    text = tmpl_path.read_text(encoding="utf-8")
    def _sub(m):
        return _resolve(m.group(1), data)
    new = _PLACEHOLDER.sub(_sub, text)
    out_path.write_text(new, encoding="utf-8")
    ok(f"wrote {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
