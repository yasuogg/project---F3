"""Plot results for the report:
  - per-task SR bars (prompt_only vs reflective vs rl_refiner)  -> figs/sr_per_task.png
  - perturbation Δ                                              -> figs/sr_perturb.png
  - PPO learning curve (from ppo.log if present)                -> figs/learning_curve.png
  - ablation comparison                                         -> figs/ablations.png
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import defaultdict
import tyro

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlwa.utils.logging import read_jsonl, info, ok


def _load_eval(path: Path):
    if not path.exists(): return []
    return list(read_jsonl(path))


def _per_task_sr(episodes):
    by = defaultdict(list)
    for ep in episodes:
        by[ep["task"]].append(1.0 if ep.get("success") else 0.0)
    return {t: sum(v)/len(v) for t, v in by.items()}


def plot_per_task(eval_dir: Path, fig_dir: Path):
    runs = {
        "prompt_only": _load_eval(eval_dir / "prompt_only.jsonl"),
        "reflective":  _load_eval(eval_dir / "reflective.jsonl"),
        "bc_only":     _load_eval(eval_dir / "bc_only.jsonl"),
        "rl_refiner":  _load_eval(eval_dir / "rl_refiner.jsonl"),
    }
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        info("plot_per_task: no eval JSONL found"); return

    tasks = sorted({t for eps in runs.values() for t in _per_task_sr(eps)})
    x = list(range(len(tasks)))
    width = 0.8 / max(1, len(runs))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, eps) in enumerate(runs.items()):
        sr = _per_task_sr(eps)
        ax.bar([xx + i*width for xx in x], [sr.get(t, 0.0) for t in tasks],
               width=width, label=name)
    ax.set_xticks([xx + width*(len(runs)-1)/2 for xx in x])
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("Success Rate"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Per-task Success Rate")
    fig.tight_layout()
    out = fig_dir / "sr_per_task.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    ok(f"-> {out}")


def plot_perturb(eval_dir: Path, fig_dir: Path):
    pairs = []
    for name in ["prompt_only", "rl_refiner"]:
        clean = _load_eval(eval_dir / f"{name}.jsonl")
        pert  = _load_eval(eval_dir / f"{name}_perturb.jsonl")
        if not clean or not pert: continue
        sc = sum(1 for ep in clean if ep.get("success")) / max(1, len(clean))
        sp = sum(1 for ep in pert  if ep.get("success")) / max(1, len(pert))
        pairs.append((name, sc, sp))
    if not pairs:
        info("plot_perturb: no perturb data found"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(range(len(pairs)))
    ax.bar([x - 0.18 for x in xs], [p[1] for p in pairs], 0.35, label="clean")
    ax.bar([x + 0.18 for x in xs], [p[2] for p in pairs], 0.35, label="CSS-jitter")
    ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylabel("Success Rate"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Robustness to CSS perturbation")
    fig.tight_layout()
    out = fig_dir / "sr_perturb.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    ok(f"-> {out}")


_LOG_RE = re.compile(r"step=(\d+).*?SR50=([0-9.]+).*?R50=([0-9.\-]+).*?kl_c=([0-9.]+)")

def plot_learning_curve(log_path: Path, fig_dir: Path):
    # prefer JSONL log written by PPOTrainer
    jsonl = log_path.with_suffix(".jsonl") if log_path.suffix != ".jsonl" else log_path
    if jsonl.exists():
        import json as _json
        steps, srs, rs, kls = [], [], [], []
        for line in jsonl.read_text().splitlines():
            try:
                d = _json.loads(line)
            except Exception: continue
            steps.append(d["step"]); srs.append(d["sr_50"])
            rs.append(d["mean_r_50"]); kls.append(d["kl_coef"])
    elif log_path.exists():
        steps, srs, rs, kls = [], [], [], []
        for line in log_path.read_text(errors="ignore").splitlines():
            m = _LOG_RE.search(line)
            if m:
                steps.append(int(m.group(1))); srs.append(float(m.group(2)))
                rs.append(float(m.group(3))); kls.append(float(m.group(4)))
    else:
        info(f"plot_learning_curve: no log at {log_path}"); return
    if not steps: info("learning_curve: no matches"); return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(steps, srs, label="SR (50-ep window)"); a1.set_xlabel("env step"); a1.set_ylabel("SR"); a1.set_ylim(0,1)
    a1.plot(steps, rs,  label="mean R", alpha=0.6); a1.legend(); a1.set_title("PPO learning curve")
    a2.plot(steps, kls); a2.set_xlabel("env step"); a2.set_ylabel("KL coef"); a2.set_title("KL-to-VLM annealing")
    fig.tight_layout()
    out = fig_dir / "learning_curve.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    ok(f"-> {out}")


def plot_ablations(eval_dir: Path, fig_dir: Path):
    files = sorted((eval_dir / "ablations").glob("*_summary.json")) if (eval_dir / "ablations").exists() else []
    if not files:
        info("plot_ablations: no ablation summaries"); return
    names, srs = [], []
    for f in files:
        d = json.loads(f.read_text())
        names.append(f.stem.replace("_summary", ""))
        srs.append(float(d.get("success_rate", 0.0)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, srs, color=["#4C72B0" if n == "full" else "#DD8452" for n in names])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Success Rate"); ax.set_title("Ablations")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out = fig_dir / "ablations.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    ok(f"-> {out}")


def main(
    eval_dir: str = "data/eval",
    log_path: str = "data/checkpoints/ppo.log",
    fig_dir: str = "paper/figs",
):
    ed = Path(eval_dir); fd = Path(fig_dir); fd.mkdir(parents=True, exist_ok=True)
    plot_per_task(ed, fd)
    plot_perturb(ed, fd)
    plot_learning_curve(Path(log_path), fd)
    plot_ablations(ed, fd)


if __name__ == "__main__":
    tyro.cli(main)
