"""EcoAgent-CG v1 command line.

  python -m src.main info
  python -m src.main pilot      --provider ollama            # 20 dev tasks: verifies pipeline + calibrates HTDE theta
  python -m src.main run        --system ecoagent --scale smoke --provider ollama
  python -m src.main experiment --name main --scale reduced  # also: budget | grid | ablation | all
  python -m src.main analyze                                 # raw logs -> processed CSVs, stats, figures
  python -m src.main ask "Write a function that merges two sorted lists" --assert "assert f([1],[2])==[1,2]"
Use --provider mock to exercise everything offline (results are flagged and never publishable)."""
import argparse
import json
import shutil
import sys
import time

from src import config as C
from src.carbon.energy import EnergyModel
from src.carbon.intensity import CarbonContext, ensure_schedule, schedule_rows
from src.carbon.ledger import RunLogger
from src.caar import CAARRouter, CostAwareRouter, FixedTierRouter
from src.cbc import CarbonBudgetController
from src.htde import HTDE, HTDECalibrationError, bin_edges, calibration_path, fit_calibration
from src.model_pool import ModelPool
from src.session_manager import Components, SessionManager
from src.tasks import custom_task, load_dev, load_frozen, subset


# ---------------------------------------------------------------------------- system construction
def spec_for(cfg: dict, name: str) -> dict:
    b = cfg["baselines"]
    if name in b["systems"]:
        return dict(b["systems"][name])
    if name in b["ablation"]:
        return dict(b["ablation"][name])
    if name.startswith("calib_d"):        # pilot: fixed depth, carbon-aware routing, no budget
        return {"label": f"calibration depth {name[-1]}", "topology": f"fixed_d{name[-1]}", "router": "caar",
                "caar_carbon": True, "budget_aware": False, "cbc": False}
    raise SystemExit(f"unknown system '{name}'. Known: {list(b['systems']) + list(b['ablation'])}")


def make_router(cfg, spec, energy):
    r = spec["router"]
    if r == "fixed_L":
        return FixedTierRouter("L")
    if r == "cascade":
        return FixedTierRouter("S")
    if r == "cost_aware":
        return CostAwareRouter(cfg)
    if r == "caar":
        return CAARRouter(cfg, energy, budget_aware=bool(spec.get("budget_aware")))
    raise ValueError(r)


def run_id_for(system, seed, experiment, label, scale, is_mock):
    return f"{'mock_' if is_mock else ''}{system}_seed{seed}_{experiment}-{label}-{scale}"


def run_system(cfg, system, seed, scale, profile, provider, experiment="main", label="base", budget_per_task_g=None,
               resume=True, phase="main", tasks=None, edges=None, quiet=False, run_id=None):
    spec = spec_for(cfg, system)
    is_mock = provider == "mock"
    ctx = CarbonContext(cfg, profile)
    if ctx.problems:
        raise SystemExit(f"Carbon profile '{profile}' is unusable:\n  - " + "\n  - ".join(ctx.problems) +
                         "\nRe-run `python scripts/fetch_carbon_data.py` (it now skips such regions) or import another trace.")
    if not ctx.trace.is_real and not quiet:
        print(f"  ! carbon profile '{profile}' is SYNTHETIC ({ctx.trace.source}); outputs will be stamped PROVISIONAL", file=sys.stderr)
    frozen = load_frozen(cfg)
    edges = edges or bin_edges(frozen, cfg["htde"]["n_complexity_bins"])
    tasks = tasks if tasks is not None else subset(cfg, frozen, scale)
    order = C.shuffled([t["task_id"] for t in tasks], seed)
    by_id = {t["task_id"]: t for t in tasks}
    ensure_schedule(cfg, schedule_rows(len(tasks), seed, profile, order, ctx))

    pool, energy = ModelPool(cfg, provider), EnergyModel(cfg)
    htde = HTDE(cfg, provider, edges) if spec["topology"] == "htde" else None
    cbc = CarbonBudgetController(cfg) if spec.get("cbc") else None
    run_id = run_id or run_id_for(system, seed, experiment, label, scale, is_mock)
    meta = {"run_id": run_id, "system": system, "seed": seed, "experiment": experiment, "label": label, "scale": scale,
            "profile": profile, "provider": provider, "is_mock": is_mock, "phase": phase, "n_tasks": len(tasks),
            "carbon_source": ctx.trace.source, "carbon_is_real": ctx.trace.is_real, "carbon_thresholds": ctx.thresholds,
            "budget_per_task_gco2eq": budget_per_task_g, "config_hash": C.config_hash(cfg), "models": C.tier_models(cfg),
            "energy_per_token_wh": {t: energy.e_per_token(t) for t in "SML"}, "energy_source": {t: energy.source(t) for t in "SML"},
            "spec": spec, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    logger = RunLogger(C.path(cfg, "raw"), run_id, meta, resume=resume)
    comp = Components(cfg, spec, pool, energy, make_router(cfg, spec, energy), cbc, htde, edges, logger, meta)
    per_task = budget_per_task_g if budget_per_task_g is not None else cfg["experiment"]["carbon_budget_gco2eq"]
    n_sess = int(cfg["session"]["session_tasks"])
    done = logger.completed_task_ids()
    results, sm = [], None
    for pos, tid in enumerate(order, start=1):
        if (pos - 1) % n_sess == 0:
            sm = SessionManager(comp, per_task * n_sess, min(n_sess, len(order) - pos + 1))
        if tid in done:
            sm.tasks_remaining -= 1
            continue
        res = sm.run_task(by_id[tid], ctx.at(pos), pos)
        results.append(res)
        if not quiet:
            q = res["final_test_pass_rate"]
            print(f"[{system} s{seed} {pos}/{len(order)}] {tid:<14} pass={q if q is None else round(q, 2)} "
                  f"g={res['total_emissions_gco2eq']:.4f} calls={res['agent_calls']} depth={res['agent_graph_depth']} "
                  f"tiers={res['tier_counts']} {'[mock]' if is_mock else ''}", flush=True)
    return results, run_id


# ---------------------------------------------------------------------------- commands
def cmd_pilot(cfg, a):
    dev = load_dev(cfg)
    frozen = load_frozen(cfg)
    edges = bin_edges(frozen, cfg["htde"]["n_complexity_bins"])
    by_depth = {}
    for d in (1, 2, 3):
        rid = run_id_for(f"calib_d{d}", cfg["experiment"]["seed"], "pilot", "dev", "pilot", a.provider == "mock")
        shutil.rmtree(C.path(cfg, "raw") / rid, ignore_errors=True)      # pilot data is calibration-only and may be redone
        res, _ = run_system(cfg, f"calib_d{d}", cfg["experiment"]["seed"], "pilot", a.profile, a.provider, "pilot", "dev",
                            resume=False, phase="pilot", tasks=dev, edges=edges, run_id=rid)
        by_depth[d] = {r["task_id"]: r for r in res}
    cal = fit_calibration(cfg, by_depth, edges, a.provider, dev)
    cal["profile"] = a.profile
    p = calibration_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print("\nHTDE calibration written to", p)
    print("selected theta =", cal["theta_selected"], "(constraint satisfiable)" if cal["theta_selection_feasible"] else "(NO theta met the quality floor; picked best-quality)")
    for r in cal["theta_sweep"]:
        print("  theta", r["theta"], "Q=%.3f E=%.4f g depth=%.2f %s" % (r["mean_quality"], r["mean_emissions_g"], r["mean_depth"], "OK" if r["feasible"] else ""))


def cmd_run(cfg, a):
    seeds = a.seeds or cfg["matrix"]["seeds"][: cfg["scales"][a.scale]["seeds"]]
    for s in seeds:
        run_system(cfg, a.system, s, a.scale, a.profile, a.provider, a.experiment, a.label, a.budget, resume=a.resume)


def b1_mean_emissions(cfg, profile, provider, scale):
    from src.analysis.aggregate_results import load_runs
    tasks, _, _ = load_runs(cfg, include_mock=(provider == "mock"))
    d = tasks[(tasks.system == "b1") & (tasks.experiment == "main") & (tasks.profile == profile) & (tasks.scale == scale)]
    if d.empty:
        raise SystemExit("Budget sensitivity needs B1 results first: run `experiment --name main` (at least system b1) for this scale/profile.")
    return float(d.total_emissions_gco2eq.mean())


def cmd_experiment(cfg, a):
    m = cfg["matrix"]
    seeds = m["seeds"][: cfg["scales"][a.scale]["seeds"]]
    names = ["main", "budget", "grid", "ablation"] if a.name == "all" else [a.name]
    for name in names:
        jobs = []
        if name == "main":
            jobs = [(s, seed, m["main"]["profile"], "base", None) for s in m["main"]["systems"] for seed in seeds]
        elif name == "budget":
            b1 = b1_mean_emissions(cfg, m["budget_sensitivity"]["profile"], a.provider, a.scale)
            print(f"B1 mean emissions/task = {b1:.5f} g -> budgets: " + ", ".join(f"{k}={v * b1:.5f}" for k, v in m["budget_sensitivity"]["levels"].items()))
            jobs = [(m["budget_sensitivity"]["system"], seed, m["budget_sensitivity"]["profile"], lab, frac * b1)
                    for lab, frac in m["budget_sensitivity"]["levels"].items() for seed in seeds]
        elif name == "grid":
            jobs = [(s, seed, prof, prof, None) for s in m["grid_sensitivity"]["systems"] for prof in m["grid_sensitivity"]["profiles"] for seed in seeds]
        elif name == "ablation":
            ab_seeds = seeds[: m["ablation"]["n_seeds"]]
            jobs = [(s, seed, m["main"]["profile"], "base", None) for s in m["ablation"]["configs"] for seed in ab_seeds]
        print(f"\n=== experiment '{name}' scale={a.scale}: {len(jobs)} runs ===")
        for system, seed, profile, label, budget in jobs:
            run_system(cfg, system, seed, a.scale, profile, a.provider, name, label, budget, resume=True)


def cmd_analyze(cfg, a):
    from src.analysis import aggregate_results, plots, statistical_tests
    out = aggregate_results.run_all(cfg, include_mock=a.include_mock)
    statistical_tests.run_all(cfg, include_mock=a.include_mock)
    plots.run_all(cfg, include_mock=a.include_mock)
    print("files written:", out.get("files_written"))
    for w in (out.get("provenance") or {}).get("warnings", []):
        print("  WARNING:", w)


def cmd_ask(cfg, a):
    if a.provider == "mock":
        raise SystemExit("`ask` needs a real provider (the mock cannot solve free-form tasks).")
    task = custom_task(a.prompt, a.assert_ or [])
    profile = a.profile
    ctx = CarbonContext(cfg, profile)
    spec = spec_for(cfg, "ecoagent")
    frozen = load_frozen(cfg)
    edges = bin_edges(frozen, cfg["htde"]["n_complexity_bins"])
    try:
        htde = HTDE(cfg, a.provider, edges)
    except HTDECalibrationError:
        print("(no HTDE calibration for this provider yet -> using fixed depth 2; run the pilot to enable adaptive depth)")
        spec["topology"], htde = "fixed_d2", None
    pool, energy = ModelPool(cfg, a.provider), EnergyModel(cfg)
    rid = f"adhoc_{int(time.time())}"
    meta = {"run_id": rid, "system": "ecoagent", "seed": 0, "experiment": "adhoc", "profile": profile, "provider": a.provider,
            "is_mock": False, "phase": "adhoc", "carbon_source": ctx.trace.source, "scale": "adhoc"}
    logger = RunLogger(C.path(cfg, "raw"), rid, meta)
    comp = Components(cfg, spec, pool, energy, make_router(cfg, spec, energy), CarbonBudgetController(cfg), htde, edges, logger, meta)
    sm = SessionManager(comp, cfg["experiment"]["carbon_budget_gco2eq"], 1)
    res = sm.run_task(task, ctx.at(a.hour), a.hour)
    print(f"\ncarbon c(t)={res['carbon_intensity_gco2_kwh']} gCO2/kWh ({res['carbon_class']})  topology={res['selected_topology']}  tiers={res['tier_counts']}")
    print(f"tokens={res['total_tokens']}  emissions={res['total_emissions_gco2eq']:.5f} gCO2eq  latency={res['latency_seconds']:.1f}s")
    print("tests:", res["test_feedback"] if res["total_tests"] else "none supplied -> output UNVERIFIED (pass --assert to verify)")
    print("\n" + res["final_code"])


def cmd_info(cfg, a):
    from src.carbon.intensity import CarbonTrace
    print("config hash:", C.config_hash(cfg))
    print("tiers:", json.dumps({t: m["name"] for t, m in C.tier_models(cfg).items()}))
    e = EnergyModel(cfg)
    print("energy Wh/token:", {t: e.e_per_token(t) for t in "SML"}, "| source:", {t: e.source(t) for t in "SML"})
    for p in ("low", "medium", "high"):
        try:
            c = CarbonContext(cfg, p)
            print(f"carbon {p:<6} source={c.trace.source}  mean={c.trace.values.mean():.1f}  thresholds(low,high)={tuple(round(x, 1) for x in c.thresholds)}")
            for pr in c.problems:
                print(f"   !! PROBLEM with '{p}' profile: {pr}")
        except FileNotFoundError as ex:
            print("carbon", p, "MISSING")
    print("frozen tasks:", len(load_frozen(cfg)), "| dev tasks:", len(load_dev(cfg)))
    print("HTDE calibration:", "present" if calibration_path(cfg).exists() else "missing (run `pilot`)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", default=None, help="ollama | openai_compatible | mock (default from configs/models.yaml)")
    common.add_argument("--profile", default="medium", choices=["low", "medium", "high"])
    p = sub.add_parser("info", parents=[common]); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("pilot", parents=[common]); p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("run", parents=[common])
    p.add_argument("--system", required=True); p.add_argument("--scale", default="smoke", choices=["smoke", "reduced", "full"])
    p.add_argument("--seeds", nargs="*", type=int); p.add_argument("--experiment", default="main"); p.add_argument("--label", default="base")
    p.add_argument("--budget", type=float, default=None, help="per-task carbon budget (gCO2eq)"); p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("experiment", parents=[common])
    p.add_argument("--name", required=True, choices=["main", "budget", "grid", "ablation", "all"])
    p.add_argument("--scale", default="reduced", choices=["smoke", "reduced", "full"]); p.set_defaults(fn=cmd_experiment)
    p = sub.add_parser("analyze", parents=[common]); p.add_argument("--include-mock", action="store_true"); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("ask", parents=[common]); p.add_argument("prompt")
    p.add_argument("--assert", dest="assert_", action="append", help="python assert line(s) used to verify the answer")
    p.add_argument("--hour", type=int, default=13, help="which replayed hour (1-based) of the carbon trace to use"); p.set_defaults(fn=cmd_ask)
    a = ap.parse_args(argv)
    cfg = C.load_all()
    a.provider = a.provider or cfg["models"]["provider_default"]
    a.fn(cfg, a)


if __name__ == "__main__":
    main()
