"""Summarise the pilot (20 dev tasks x depths 1/2/3) so you can paste ONE block of text.

  python scripts/pilot_report.py

Reads results/raw/*calib_d*  (the pilot runs) and results/processed/htde_calibration.json.
Also writes results/processed/pilot_report.txt. Prints facts and a latency-SLO *suggestion*; the decision stays yours."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C


def load(d: Path, name: str) -> pd.DataFrame:
    f = d / name
    if not f.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])


def main():
    cfg = C.load_all()
    raw = C.path(cfg, "raw")
    runs = {}
    for d in sorted(raw.glob("*calib_d*")):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("phase") != "pilot":
            continue
        depth = int(m["system"][-1])
        prefer = runs.get(depth)
        if prefer is None or (prefer[0].get("is_mock") and not m.get("is_mock")):
            runs[depth] = (m, d)
    if not runs:
        sys.exit("No pilot runs found in results/raw. Run: python -m src.main pilot --provider ollama")
    out = []
    P = out.append
    any_m = next(iter(runs.values()))[0]
    P("=== PILOT REPORT ===")
    P(f"provider={any_m['provider']}  mock={any_m['is_mock']}  carbon_profile={any_m['profile']} ({any_m['carbon_source']})")
    P("models: " + ", ".join(f"{t}={v['name']}" for t, v in any_m["models"].items()))
    P(f"energy Wh/token: {any_m['energy_per_token_wh']}  source: {sorted(set(any_m['energy_source'].values()))}")
    slo = cfg["experiment"]["latency_slo_seconds"]
    P(f"configured latency SLO: {slo}s")
    P("")
    P("depth | tasks | all-tests-pass | mean pass-rate | mean g/task | mean calls | latency p50 / p90 / max (s) | within SLO")
    d3_lat = None
    tier_calls = {}
    for depth in sorted(runs):
        m, d = runs[depth]
        t, c = load(d, "task_results.jsonl"), load(d, "llm_calls.jsonl")
        if t.empty:
            P(f"  {depth}  | (no finished tasks yet)")
            continue
        lat = t["latency_seconds"].to_numpy(float)
        P(f"  D{depth}  | {len(t):>3}   | {t['task_success'].mean():>6.0%}         | {t['final_test_pass_rate'].mean():>6.3f}        | "
          f"{t['total_emissions_gco2eq'].mean():.4f}      | {t['agent_calls'].mean():.2f}       | "
          f"{np.percentile(lat, 50):6.1f} / {np.percentile(lat, 90):6.1f} / {lat.max():6.1f}        | {(lat <= slo).mean():.0%}")
        if depth == 3:
            d3_lat = lat
        cc = c[c.total_tokens > 0]
        tier_calls[depth] = {r: int((cc.model_tier == r).sum()) for r in "SML"}
    P("")
    P("LLM calls by tier per depth (cache hits included): " + json.dumps(tier_calls))
    for depth in (1,):
        if depth in runs:
            t = load(runs[depth][1], "task_results.jsonl")
            if not t.empty:
                t["coder_tier"] = t["tier_counts"].map(lambda x: max(x, key=x.get) if x else "?")
                g = t.groupby("coder_tier").agg(n=("task_id", "count"), pass_rate=("final_test_pass_rate", "mean"), all_pass=("task_success", "mean"))
                P("")
                P("Depth-1 coder quality by the tier CAAR picked (small n; indicative only):")
                P(g.round(3).to_string())
    if d3_lat is not None and len(d3_lat):
        p90 = float(np.percentile(d3_lat, 90))
        P("")
        P(f"SLO suggestion (NOT a decision): p90 of D3 latency = {p90:.0f}s -> e.g. latency_slo_seconds: {int(math.ceil(p90 / 30.0) * 30)} "
          f"(rounded up to 30s). Record the reason in the paper.")
    cal = C.path(cfg, "processed") / "htde_calibration.json"
    P("")
    if cal.exists():
        j = json.loads(cal.read_text(encoding="utf-8"))
        P(f"HTDE calibration: provider={j['provider']} dev_tasks={j['n_dev_tasks']} theta_selected={j['theta_selected']} "
          f"(constraint satisfiable: {j['theta_selection_feasible']}), always-D3 quality={j['always_d3_quality']:.3f}, floor={j['quality_retention_floor']}")
        P("table[bin][depth] = mean pass-rate / mean energy Wh / n:")
        for b, cell in j["table"].items():
            P(f"  {b:<7}" + "  ".join(f"D{d}: q={v['q']:.2f} e={v['e_wh']:.3f}Wh n={v['n']}" for d, v in cell.items()))
        P("theta sweep (theta | quality | emissions g | mean depth | feasible):")
        for r in j["theta_sweep"]:
            P(f"  {r['theta']:>10.4g} | {r['mean_quality']:.3f} | {r['mean_emissions_g']:.4f} | {r['mean_depth']:.2f} | {'yes' if r['feasible'] else 'no'}")
    else:
        P("HTDE calibration file not found yet (pilot not finished).")
    text = "\n".join(out)
    print(text)
    dest = C.path(cfg, "processed")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "pilot_report.txt").write_text(text, encoding="utf-8")
    print("\n(saved to results/processed/pilot_report.txt)")


if __name__ == "__main__":
    main()