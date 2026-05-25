"""Top-K oracle upper bound.

For each (task, seed), run the prompt-only agent K times where the i-th run
forces selection of the i-th candidate at every step. SR_oracle = max over i.

This bounds how much *re-ranking* alone (without changing the candidate set)
can possibly improve over top-1.
"""
import json
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import tyro

from rlwa.envs import make_env, DEFAULT_TASKS
from rlwa.planners import GeminiPlanner, AsyncGeminiPlanner
from rlwa.agents.action_space import build_observation
from rlwa.utils.logging import JsonlWriter, info, ok


class _ForcedAgent:
    """Always pick candidate index = `force_idx` (clipped)."""
    def __init__(self, planner: GeminiPlanner, force_idx: int):
        self.planner = planner
        self.force_idx = force_idx

    def run_episode(self, env, task, seed, max_steps):
        raw, _ = env.reset(seed=seed)
        history = []; last_error = None
        total_r = 0.0; success = False
        for step in range(max_steps):
            obs = build_observation(raw)
            cands = self.planner.propose(
                goal=obs["goal"], som_image=obs["som_image"],
                axtree=obs["axtree"], mark_bids=obs["mark_bids"],
                history=history, last_error=last_error,
            )
            idx = min(self.force_idx, len(cands) - 1)
            action_str = cands[idx].to_browsergym_action()
            try:
                raw, reward, term, trunc, info = env.step(action_str)
                err = info.get("last_action_error") or info.get("action_error")
            except Exception as e:
                reward, term, trunc, info, err = 0.0, False, False, {}, str(e)
            total_r += float(reward)
            history.append({"step": step, "action": action_str,
                            "reward": float(reward), "failed": bool(err)})
            last_error = err
            if info.get("success") or reward > 0.5:
                success = True
            if term or trunc:
                break
        return success, total_r


def main(
    seeds: int = 3,
    max_steps: int = 25,
    K: int = 5,
    agent_cfg: str = "configs/agent/prompt_only.yaml",
    headless: bool = True,
    out: str = "data/eval/oracle_topk.jsonl",
    workers: int = 32,
    planner_workers: int = 96,
):
    cfg = OmegaConf.load(agent_cfg)
    planner = AsyncGeminiPlanner(GeminiPlanner(cfg), max_workers=planner_workers)
    writer = JsonlWriter(out, mode="w")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    lock = Lock()

    jobs = [(task, seed, k) for task in DEFAULT_TASKS
                              for seed in range(seeds)
                              for k in range(K)]
    raw_results: dict[tuple[str, int, int], tuple[bool, float]] = {}
    pbar = tqdm(total=len(jobs), desc=f"oracle(x{workers})")

    def _run(task, seed, k):
        env = make_env(task, seed=seed, headless=headless, max_steps=max_steps)
        try:
            agent = _ForcedAgent(planner, force_idx=k)
            return agent.run_episode(env, task, seed, max_steps)
        finally:
            env.close()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, t, s, k): (t, s, k) for t, s, k in jobs}
        for fut in as_completed(futs):
            t, s, k = futs[fut]
            try:
                success, R = fut.result()
            except Exception as e:
                pbar.write(f"! {t}/{s}/{k}: {e}"); pbar.update(1); continue
            raw_results[(t, s, k)] = (success, R)
            with lock:
                writer.write({"task": t, "seed": s, "force_idx": k,
                              "success": bool(success), "return": float(R)})
            pbar.update(1)
    pbar.close(); writer.close()

    per_task: dict[str, list[float]] = {t: [] for t in DEFAULT_TASKS}
    top1_per_task: dict[str, list[float]] = {t: [] for t in DEFAULT_TASKS}
    for task in DEFAULT_TASKS:
        for seed in range(seeds):
            any_success = False; top1 = False
            for k in range(K):
                s, _ = raw_results.get((task, seed, k), (False, 0.0))
                any_success = any_success or s
                if k == 0: top1 = s
            per_task[task].append(1.0 if any_success else 0.0)
            top1_per_task[task].append(1.0 if top1 else 0.0)

    summary = {
        "K": K,
        "per_task_oracle_sr": {t: sum(v)/len(v) for t, v in per_task.items()},
        "per_task_top1_sr":   {t: sum(v)/len(v) for t, v in top1_per_task.items()},
    }
    summary["mean_oracle_sr"] = sum(summary["per_task_oracle_sr"].values()) / len(summary["per_task_oracle_sr"])
    summary["mean_top1_sr"]   = sum(summary["per_task_top1_sr"].values())  / len(summary["per_task_top1_sr"])
    summary["headroom"] = summary["mean_oracle_sr"] - summary["mean_top1_sr"]

    out_summary = Path(out).with_suffix(".summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    ok(f"Oracle SR={summary['mean_oracle_sr']:.3f}  Top1 SR={summary['mean_top1_sr']:.3f}  "
       f"Headroom={summary['headroom']:.3f}")
    info(f"Summary -> {out_summary}")


if __name__ == "__main__":
    tyro.cli(main)
