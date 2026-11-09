"""Raw JSONL logs -> processed CSVs. Every number in the paper comes from here (never from console output).
Statistical convention: metrics are first averaged over seeds per (system, task); CIs/tests are then over TASKS (n = #tasks),
because seeds of a temperature-0 pipeline are not independent samples of the task distribution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C

METRICS = ["task_success", "final_test_pass_rate", "total_emissions_gco2eq", "total_tokens", "agent_calls",
           "agent_graph_depth", "retry_count", "latency_seconds", "slo_compliant", "api_cost_usd"]


SCALE_RANK = {"smoke": 0, "reduced": 1, "full": 2}


def load_runs(cfg: dict, include_mock: bool = False, scale: str | None = "auto"):
    """Load main-phase runs. `scale="auto"` keeps ONLY the largest scale present (so a smoke test can never be mixed into
    reduced/full results); pass an explicit scale to force one, or None to load everything. Excluded scales are reported
    in provenance.json (via metas.attrs["excluded_scales"])."""
    raw = C.path(cfg, "raw")
    metas = []
    for d in sorted(raw.glob("*/")):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("phase") != "main" or (meta.get("is_mock") and not include_mock):
            continue
        meta["_dir"] = str(d)
        metas.append(meta)
    present = {m.get("scale") for m in metas}
    chosen, excluded = None, []
    if scale == "auto" and present:
        chosen = max(present, key=lambda x: SCALE_RANK.get(x, -1))
    elif scale not in (None, "auto"):
        chosen = scale
    if chosen is not None:
        excluded = sorted(x for x in present if x != chosen)
        metas = [m for m in metas if m.get("scale") == chosen]
    tasks, calls = [], []
    for meta in metas:
        d = Path(meta["_dir"])
        for fname, store in (("task_results.jsonl", tasks), ("llm_calls.jsonl", calls)):
            f = d / fname
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        r.update(experiment=meta.get("experiment"), label=meta.get("label"), scale=meta.get("scale"))
                        store.append(r)
    t, c = pd.DataFrame(tasks), pd.DataFrame(calls)
    m = pd.DataFrame([{k: v for k, v in x.items() if k != "_dir"} for x in metas])
    m.attrs["excluded_scales"] = excluded
    m.attrs["chosen_scale"] = chosen
    for col in ("experiment", "label", "scale", "system", "profile"):          # keep filters safe on empty frames
        for df in (t, c):
            if col not in df.columns:
                df[col] = pd.Series(dtype=object)
    return t, c, m


def ci95(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    h = stats.sem(x) * stats.t.ppf(0.975, n - 1)
    return float(x.mean() - h), float(x.mean() + h)


def per_task_means(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = [m for m in METRICS if m in df.columns]
    return df.groupby(keys + ["task_id"], dropna=False)[cols].mean().reset_index()


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean over per-task (seed-averaged) values for every metric, per group."""
    pt = per_task_means(df, keys)
    out = pt.groupby(keys, dropna=False)[[m for m in METRICS if m in pt.columns]].mean().reset_index()
    out["n_tasks"] = pt.groupby(keys, dropna=False).task_id.nunique().to_numpy()
    runs = df.groupby(keys, dropna=False).run_id.nunique().reset_index(name="n_runs")
    out = out.merge(runs, on=keys, how="left")
    out["emissions_per_task_gco2eq"] = out["total_emissions_gco2eq"]
    out["total_emissions_sum_gco2eq"] = out["emissions_per_task_gco2eq"] * out["n_tasks"]
    out["carbon_quality_efficiency"] = out["final_test_pass_rate"] / out["emissions_per_task_gco2eq"].replace(0, np.nan)
    return out


def add_retention(out: pd.DataFrame, ref_quality: float) -> pd.DataFrame:
    out["quality_retention_pct"] = out["final_test_pass_rate"] / ref_quality * 100 if ref_quality else np.nan
    return out


def with_ci(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pt = per_task_means(df, keys)
    rows = []
    for k, g in pt.groupby(keys, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        for m in [m for m in METRICS if m in g.columns]:
            x = g[m].dropna().to_numpy()
            lo, hi = ci95(x)
            rows.append({**dict(zip(keys, k)), "metric": m, "mean": x.mean() if len(x) else np.nan,
                         "sd": x.std(ddof=1) if len(x) > 1 else np.nan, "ci95_low": lo, "ci95_high": hi,
                         "median": np.median(x) if len(x) else np.nan, "n": len(x)})
    return pd.DataFrame(rows)


def _save(df: pd.DataFrame, cfg: dict, name: str, written: list):
    proc = C.path(cfg, "processed")
    proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc / name, index=False)
    tab = C.ROOT / cfg["paths"]["paper"] / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab / name, index=False)
    (tab / (name.replace(".csv", ".md"))).write_text(df.round(5).to_markdown(index=False) if hasattr(df, "to_markdown") and _has_tabulate() else df.round(5).to_string(index=False), encoding="utf-8")
    written.append(name)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


def provenance(cfg: dict, metas: pd.DataFrame) -> dict:
    if metas.empty:
        return {"runs": 0, "provisional": True, "warnings": ["no runs found"]}
    warns = []
    prov = False
    if (~metas["carbon_is_real"].astype(bool)).any():
        prov = True
        warns.append("Some runs used SYNTHETIC carbon traces -> every output is PROVISIONAL.")
    if metas["is_mock"].astype(bool).any():
        prov = True
        warns.append("Some runs used the MOCK LLM -> not publishable.")
    model_sets = {json.dumps(m, sort_keys=True) for m in metas["models"]}
    if len(model_sets) > 1:
        warns.append("MODEL CONFIG CHANGED between runs (models must be frozen for all experiments).")
    if metas["config_hash"].nunique() > 1:
        warns.append(f"{metas['config_hash'].nunique()} different config hashes across runs (check that only intended settings changed).")
    es = {json.dumps(e, sort_keys=True) for e in metas["energy_source"]}
    if any("assumed" in e for e in es):
        warns.append("Energy per token is an ASSUMED estimate (not CodeCarbon-measured) -> state this as a limitation.")
    ex = metas.attrs.get("excluded_scales") or []
    if ex:
        warns.append(f"Analysis uses scale '{metas.attrs.get('chosen_scale')}' only; runs at scale(s) {ex} were excluded.")
    return {"runs": int(len(metas)), "analysed_scale": metas.attrs.get("chosen_scale"), "providers": sorted(metas["provider"].unique().tolist()),
            "carbon_sources": sorted(metas["carbon_source"].unique().tolist()), "provisional": prov,
            "carbon_thresholds_low_high": {p: g["carbon_thresholds"].iloc[0] for p, g in metas.groupby("profile")},
            "models": metas["models"].iloc[0], "energy_per_token_wh": metas["energy_per_token_wh"].iloc[0],
            "energy_source": metas["energy_source"].iloc[0], "warnings": warns}


def run_all(cfg: dict, include_mock: bool = False) -> dict:
    t, c, metas = load_runs(cfg, include_mock)
    written: list = []
    prov = provenance(cfg, metas)
    proc = C.path(cfg, "processed")
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    if t.empty:
        return {"status": "no runs to analyse", "provenance": prov}

    main = t[t.experiment == "main"]
    if not main.empty:
        s = summarize(main, ["system"])
        b1q = s.loc[s.system == "b1", "final_test_pass_rate"]
        s = add_retention(s, float(b1q.iloc[0]) if len(b1q) else float("nan"))
        _save(s, cfg, "main_results.csv", written)
        _save(with_ci(main, ["system"]), cfg, "main_results_with_ci.csv", written)

    ab = t[t.experiment == "ablation"]
    if not ab.empty:
        flags = {k: (v["topology"] in ("htde",), v["router"] == "caar", bool(v.get("cbc"))) for k, v in cfg["baselines"]["ablation"].items()}
        seeds = sorted(ab.seed.unique())
        none = main[(main.system == "b1") & (main.seed.isin(seeds))].assign(system="abl_none")
        s = summarize(pd.concat([ab, none]), ["system"])
        s["HTDE"] = s.system.map(lambda x: flags.get(x, (False, False, False))[0])
        s["CAAR"] = s.system.map(lambda x: flags.get(x, (False, False, False))[1])
        s["CBC"] = s.system.map(lambda x: flags.get(x, (False, False, False))[2])
        ref = s.loc[s.system == "abl_none", "final_test_pass_rate"]
        s = add_retention(s, float(ref.iloc[0]) if len(ref) else float("nan"))
        _save(s, cfg, "ablation_results.csv", written)

    gr = t[t.experiment == "grid"]
    if not gr.empty:
        s = summarize(gr, ["system", "profile"])
        th = metas.groupby("profile")["carbon_thresholds"].first().to_dict()
        s["tercile_thresholds_low_high"] = s.profile.map(lambda p: th.get(p))
        _save(s, cfg, "carbon_sensitivity.csv", written)

    bs = t[t.experiment == "budget"]
    if not bs.empty:
        s = summarize(bs, ["label"])
        bud = metas[metas.experiment == "budget"].groupby("label")["budget_per_task_gco2eq"].first()
        s["budget_per_task_gco2eq"] = s.label.map(bud)
        order = {"tight": 0, "medium": 1, "relaxed": 2}
        s = s.sort_values("label", key=lambda x: x.map(order))
        _save(s, cfg, "budget_sensitivity.csv", written)

    if not c.empty:
        cc = c[c.total_tokens > 0]
        tier = cc.groupby(["experiment", "system", "label", "agent_role", "model_tier"]).size().reset_index(name="n_calls")
        tot = tier.groupby(["experiment", "system", "label", "agent_role"]).n_calls.transform("sum")
        tier["pct_of_role_calls"] = tier.n_calls / tot * 100
        allr = cc.groupby(["experiment", "system", "label", "model_tier"]).size().reset_index(name="n_calls")
        allr["agent_role"] = "all"
        allr["pct_of_role_calls"] = allr.n_calls / allr.groupby(["experiment", "system", "label"]).n_calls.transform("sum") * 100
        _save(pd.concat([allr, tier], ignore_index=True), cfg, "tier_distribution.csv", written)

        rows = []
        for (exp, sysn), g in t.groupby(["experiment", "system"]):
            r = g[g.retry_count > 0]
            nr = g[g.retry_count == 0]
            calls_g = cc[(cc.experiment == exp) & (cc.system == sysn)]
            retry_g = calls_g[calls_g.retry_number > 0].emissions_gco2eq.sum() + calls_g[calls_g.retry_number > 0].communication_overhead_gco2eq.sum()
            allg = calls_g.emissions_gco2eq.sum() + calls_g.communication_overhead_gco2eq.sum()
            rows.append({"experiment": exp, "system": sysn, "tasks": len(g), "tasks_with_retry_pct": 100 * len(r) / len(g),
                         "mean_emissions_with_retry": r.total_emissions_gco2eq.mean() if len(r) else np.nan,
                         "mean_emissions_without_retry": nr.total_emissions_gco2eq.mean() if len(nr) else np.nan,
                         "retry_share_of_total_emissions_pct": 100 * retry_g / allg if allg else np.nan,
                         "retry_calls": int((calls_g.retry_number > 0).sum()), "total_calls": int(len(calls_g))})
        _save(pd.DataFrame(rows), cfg, "retry_analysis.csv", written)

    if not main.empty:
        rows = []
        for sysn, g in main.groupby("system"):
            for f in [0.0] + list(cfg["accounting"]["comm_sensitivity_fractions"]):
                tot = g.inference_emissions_gco2eq * (1 + f)
                rows.append({"system": sysn, "comm_overhead_fraction": f, "emissions_per_task_gco2eq": tot.mean(),
                             "total_emissions_gco2eq": tot.sum()})
        _save(pd.DataFrame(rows), cfg, "comm_overhead_sensitivity.csv", written)

    return {"status": "ok", "files_written": written, "provenance": prov}