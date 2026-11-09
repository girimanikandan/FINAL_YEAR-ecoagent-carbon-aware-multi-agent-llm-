"""Publication-style figures generated only from results/processed/*.csv. Stamped PROVISIONAL when the provenance says so."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

ORDER = ["b1", "b2", "b3", "b4", "ecoagent"]
COL = {"b1": "#8d99ae", "b2": "#6c757d", "b3": "#4361ee", "b4": "#f4a261", "ecoagent": "#2d6a4f"}


def _prov(cfg) -> bool:
    p = C.path(cfg, "processed") / "provenance.json"
    return bool(json.loads(p.read_text()).get("provisional", True)) if p.exists() else True


def _finish(fig, cfg, name, provisional):
    if provisional:
        fig.text(0.5, 0.5, "PROVISIONAL\nsynthetic carbon / mock LLM", ha="center", va="center", fontsize=28,
                 color="red", alpha=0.12, rotation=25)
    fig.tight_layout()
    for d in (C.path(cfg, "figures"), C.ROOT / cfg["paths"]["paper"] / "figures"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=200)
    plt.close(fig)


def _read(cfg, name):
    p = C.path(cfg, "processed") / name
    return pd.read_csv(p) if p.exists() else None


def run_all(cfg: dict, include_mock: bool = False) -> None:
    prov = _prov(cfg)
    main, ci = _read(cfg, "main_results.csv"), _read(cfg, "main_results_with_ci.csv")
    if main is not None and ci is not None:
        main = main.set_index("system").reindex([s for s in ORDER if s in set(main.system)]).reset_index()
        for metric, ylabel, fname in (("total_emissions_gco2eq", "Emissions per task (gCO2eq)", "fig_main_emissions.png"),
                                      ("final_test_pass_rate", "Mean test pass rate", "fig_main_quality.png")):
            c = ci[ci.metric == metric].set_index("system").reindex(main.system)
            fig, ax = plt.subplots(figsize=(6, 4))
            err = np.vstack([c["mean"] - c["ci95_low"], c["ci95_high"] - c["mean"]])
            ax.bar(main.system, c["mean"], yerr=err, capsize=4, color=[COL[s] for s in main.system])
            ax.set_ylabel(ylabel); ax.set_title(ylabel + " (95% CI over tasks)")
            _finish(fig, cfg, fname, prov)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for _, r in main.iterrows():
            ax.scatter(r.emissions_per_task_gco2eq, r.final_test_pass_rate, s=90, color=COL[r.system])
            ax.annotate(r.system, (r.emissions_per_task_gco2eq, r.final_test_pass_rate), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("Emissions per task (gCO2eq)"); ax.set_ylabel("Mean test pass rate"); ax.set_title("Quality vs emissions")
        _finish(fig, cfg, "fig_pareto.png", prov)
    td = _read(cfg, "tier_distribution.csv")
    if td is not None:
        d = td[(td.experiment == "main") & (td.agent_role == "all")]
        if not d.empty:
            pv = d.pivot_table(index="system", columns="model_tier", values="pct_of_role_calls", fill_value=0)
            pv = pv.reindex([s for s in ORDER if s in pv.index])
            fig, ax = plt.subplots(figsize=(6, 4))
            pv.plot(kind="bar", stacked=True, ax=ax, color={"S": "#95d5b2", "M": "#40916c", "L": "#1b4332"})
            ax.set_ylabel("% of LLM calls"); ax.set_title("Model-tier distribution"); ax.legend(title="tier")
            _finish(fig, cfg, "fig_tier_distribution.png", prov)
    cs = _read(cfg, "carbon_sensitivity.csv")
    if cs is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        pv = cs.pivot_table(index="profile", columns="system", values="emissions_per_task_gco2eq").reindex(["low", "medium", "high"])
        pv.plot(kind="bar", ax=ax, color=[COL.get(c) for c in pv.columns]); ax.set_ylabel("Emissions per task (gCO2eq)")
        ax.set_title("Grid-carbon sensitivity")
        _finish(fig, cfg, "fig_carbon_sensitivity.png", prov)
    bs = _read(cfg, "budget_sensitivity.csv")
    if bs is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(bs.label, bs.emissions_per_task_gco2eq, color="#2d6a4f"); ax.set_ylabel("Emissions per task (gCO2eq)")
        ax2 = ax.twinx(); ax2.plot(bs.label, bs.final_test_pass_rate, "o-", color="#d00000"); ax2.set_ylabel("Mean test pass rate")
        ax.set_title("Carbon-budget sensitivity (EcoAgent)")
        _finish(fig, cfg, "fig_budget_sensitivity.png", prov)
    ab = _read(cfg, "ablation_results.csv")
    if ab is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(ab))
        ax.bar(x, ab.emissions_per_task_gco2eq, color="#52b788"); ax.set_xticks(x); ax.set_xticklabels(ab.system, rotation=30, ha="right")
        ax.set_ylabel("Emissions per task (gCO2eq)")
        ax2 = ax.twinx(); ax2.plot(x, ab.final_test_pass_rate, "o-", color="#d00000"); ax2.set_ylabel("Mean test pass rate")
        ax.set_title("Ablation")
        _finish(fig, cfg, "fig_ablation.png", prov)
