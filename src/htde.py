"""HTDE - Hierarchical Task Decomposition Engine: transparent rule-based adaptive depth (NOT an RL agent).
Candidate depths: D1 coder only | D2 planner+coder | D3 planner+coder+reviewer(+repair on failure).
Start at D1 and expand D(k-1)->D(k) only if   dQ_k / dE_k  >  theta / max(alpha, epsilon)
with dQ_k = Qhat(D_k) - Qhat(D_k-1) and dE_k = Ehat(D_k) - Ehat(D_k-1) (gCO2eq at the CURRENT carbon intensity).
Qhat/Ehat come from the pilot calibration (results/processed/htde_calibration.json), stratified by task-complexity bin;
theta is selected by the calibration, never hand-set (a numeric htde.theta in the YAML overrides it, for experiments only)."""
import json
from pathlib import Path

import numpy as np

from src.config import path as cfg_path

BINS = ["simple", "medium", "hard"]


class HTDECalibrationError(RuntimeError):
    pass


def calibration_path(cfg: dict) -> Path:
    return cfg_path(cfg, "processed") / "htde_calibration.json"


def bin_edges(tasks: list[dict], n_bins: int = 3) -> list[float]:
    """Tercile edges of prompt-token estimate over the FROZEN task set (so every run bins identically)."""
    x = np.array([t["prompt_tokens_est"] for t in tasks], dtype=float)
    return [float(np.percentile(x, 100 * i / n_bins)) for i in range(1, n_bins)]


def assign_bin(prompt_tokens: float, edges: list[float]) -> str:
    return BINS[sum(prompt_tokens > e for e in edges)]


def features(task: dict) -> dict:
    return {"prompt_tokens_est": task["prompt_tokens_est"], "n_tests": task["n_tests"],
            "difficulty": task.get("difficulty", "") or None, "est_tokens_required": task["prompt_tokens_est"] + 350}


def decide_depth(table: dict, bin_name: str, intensity: float, alpha: float, theta: float, eps: float,
                 comm_fraction: float = 0.0, remaining_g: float | None = None) -> tuple[int, list]:
    """Pure decision rule (shared by the live HTDE and the theta-selection simulation)."""
    cell = table[bin_name]
    g = lambda d: cell[str(d)]["e_wh"] * intensity * 1e-3 * (1 + comm_fraction)
    depth, steps = 1, []
    for k in (2, 3):
        dq = cell[str(k)]["q"] - cell[str(k - 1)]["q"]
        de = g(k) - g(k - 1)
        thr = theta / max(alpha, eps)
        ratio = dq / max(de, 1e-12)
        over_budget = remaining_g is not None and g(k) > remaining_g
        ok = dq > 0 and ratio > thr and not over_budget
        steps.append({"k": k, "dQ": round(dq, 5), "dE_g": round(de, 6), "ratio": round(ratio, 4),
                      "threshold": round(thr, 4), "over_budget": over_budget, "expand": ok})
        if not ok:
            break
        depth = k
    return depth, steps


class HTDE:
    def __init__(self, cfg: dict, provider: str, edges: list[float]):
        p = calibration_path(cfg)
        if not p.exists():
            raise HTDECalibrationError(f"{p} not found. Run the pilot first: `python -m src.main pilot --provider {provider}`")
        self.cal = json.loads(p.read_text(encoding="utf-8"))
        if self.cal["provider"] != provider:
            raise HTDECalibrationError(f"Calibration was made with provider '{self.cal['provider']}', not '{provider}'. Re-run the pilot.")
        self.cfg = cfg
        self.eps = cfg["htde"]["epsilon"]
        self.theta = cfg["htde"]["theta"] if cfg["htde"]["theta"] is not None else self.cal["theta_selected"]
        self.comm = cfg["accounting"]["comm_overhead_fraction"]
        self.edges = edges
        self.table = self.cal["table"]

    def decide(self, task: dict, alpha: float, intensity: float, remaining_g: float | None = None) -> tuple[int, dict]:
        f = features(task)
        b = assign_bin(f["prompt_tokens_est"], self.edges)
        depth, steps = decide_depth(self.table, b, intensity, alpha, self.theta, self.eps, self.comm, remaining_g)
        return depth, {"bin": b, "features": f, "theta": self.theta, "alpha": alpha, "steps": steps}


def fit_calibration(cfg: dict, dev_by_depth: dict, edges: list[float], provider: str, dev_tasks: list[dict]) -> dict:
    """dev_by_depth[d] = {task_id: task_result_dict} from running ALL dev tasks at fixed depth d (d=1,2,3)."""
    ptok = {t["task_id"]: t["prompt_tokens_est"] for t in dev_tasks}
    per_task = {}
    for tid in ptok:
        per_task[tid] = {d: dev_by_depth[d][tid] for d in (1, 2, 3)}
    # table[bin][depth] = mean Q, mean energy (Wh), mean latency; empty cells fall back to the pooled mean
    def cell(rows):
        return {"q": float(np.mean([r["final_test_pass_rate"] for r in rows])),
                "e_wh": float(np.mean([r["inference_energy_wh"] for r in rows])),
                "lat": float(np.mean([r["latency_seconds"] for r in rows])), "n": len(rows)}
    table = {}
    for b in BINS:
        ids = [t for t in ptok if assign_bin(ptok[t], edges) == b]
        table[b] = {}
        for d in (1, 2, 3):
            rows = [per_task[t][d] for t in ids] or [per_task[t][d] for t in ptok]
            table[b][str(d)] = cell(rows)
            if not ids:
                table[b][str(d)]["n"] = 0
    # theta selection: min mean emissions subject to quality >= floor x always-D3 quality
    floor = cfg["htde"]["calibration_quality_retention_floor"]
    q3 = float(np.mean([per_task[t][3]["final_test_pass_rate"] for t in ptok]))
    results = []
    # Candidate thetas: the configured grid PLUS values just below/above every observed dQ/dE ratio, so the sweep always
    # covers every distinct decision boundary regardless of the units/scale of the emissions (a fixed grid can miss them).
    ratios = []
    for b in BINS:
        for k in (2, 3):
            dq = table[b][str(k)]["q"] - table[b][str(k - 1)]["q"]
            de = (table[b][str(k)]["e_wh"] - table[b][str(k - 1)]["e_wh"]) * float(np.mean([r[1]["carbon_intensity_gco2_kwh"] for r in per_task.values()])) * 1e-3
            if dq > 0 and de > 0:
                ratios.append(dq / de)
    grid = sorted({*map(float, cfg["htde"]["theta_grid"]), *[r * f for r in ratios for f in (0.9, 1.1)]})
    for theta in grid:
        qs, es, depths = [], [], []
        for t in ptok:
            b = assign_bin(ptok[t], edges)
            inten = per_task[t][1]["carbon_intensity_gco2_kwh"]
            d, _ = decide_depth(table, b, inten, 1.0, theta, cfg["htde"]["epsilon"], cfg["accounting"]["comm_overhead_fraction"])
            qs.append(per_task[t][d]["final_test_pass_rate"])
            es.append(per_task[t][d]["total_emissions_gco2eq"])
            depths.append(d)
        results.append({"theta": theta, "mean_quality": float(np.mean(qs)), "mean_emissions_g": float(np.mean(es)),
                        "mean_depth": float(np.mean(depths)), "feasible": float(np.mean(qs)) >= floor * q3})
    feas = [r for r in results if r["feasible"]]
    best = min(feas, key=lambda r: r["mean_emissions_g"]) if feas else max(results, key=lambda r: r["mean_quality"])
    return {"provider": provider, "n_dev_tasks": len(ptok), "bin_edges_prompt_tokens": edges, "table": table,
            "always_d3_quality": q3, "quality_retention_floor": floor, "theta_sweep": results, "theta_grid_used": grid,
            "theta_selected": best["theta"], "theta_selection_feasible": bool(feas),
            "epsilon": cfg["htde"]["epsilon"],
            "note": "Dev tasks are disjoint from the frozen 200. Pilot results are for calibration only (PDF)."}
