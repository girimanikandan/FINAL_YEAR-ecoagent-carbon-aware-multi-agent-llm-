"""EcoAgent vs each baseline on per-task (seed-averaged) results. Paired tests: Wilcoxon signed-rank when the paired
differences are non-normal (Shapiro-Wilk), paired t-test otherwise. Effect sizes: Cliff's delta (and Cohen's d_z).
Holm-Bonferroni across all comparisons. Also the headline % change in mean with a paired-bootstrap 95% CI."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src import config as C
from src.analysis.aggregate_results import load_runs, per_task_means, _save

TEST_METRICS = ["total_emissions_gco2eq", "final_test_pass_rate", "total_tokens", "agent_calls", "latency_seconds"]
BASELINES = ["b1", "b2", "b3", "b4"]


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    if len(d) < 3 or np.allclose(d, 0):
        return {"test": "none (identical)", "statistic": np.nan, "p": 1.0, "normal": None}
    normal = bool(stats.shapiro(d).pvalue > 0.05) if len(d) <= 5000 else False
    if normal:
        r = stats.ttest_rel(a, b)
        return {"test": "paired t-test", "statistic": float(r.statistic), "p": float(r.pvalue), "normal": True}
    r = stats.wilcoxon(a, b, zero_method="wilcox")
    return {"test": "Wilcoxon signed-rank", "statistic": float(r.statistic), "p": float(r.pvalue), "normal": False}


def boot_pct_change(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """% change of mean(a) relative to mean(b), paired bootstrap over tasks."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    pct = (a[idx].mean(1) - b[idx].mean(1)) / b[idx].mean(1) * 100
    base = (a.mean() - b.mean()) / b.mean() * 100
    return float(base), float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))


def run_all(cfg: dict, include_mock: bool = False) -> None:
    t, _, _ = load_runs(cfg, include_mock)
    main = t[t.experiment == "main"] if not t.empty else t
    if main.empty or "ecoagent" not in set(main.system):
        return
    pt = per_task_means(main, ["system"])
    rows = []
    for base in BASELINES:
        if base not in set(pt.system):
            continue
        e = pt[pt.system == "ecoagent"].set_index("task_id")
        b = pt[pt.system == base].set_index("task_id")
        ids = e.index.intersection(b.index)
        for m in TEST_METRICS:
            a, bb = e.loc[ids, m].to_numpy(float), b.loc[ids, m].to_numpy(float)
            res = paired_test(a, bb)
            pct, lo, hi = boot_pct_change(a, bb) if bb.mean() else (np.nan, np.nan, np.nan)
            rows.append({"comparison": f"ecoagent vs {base}", "metric": m, "n_tasks": len(ids), "mean_ecoagent": a.mean(),
                         f"mean_baseline": bb.mean(), "pct_change_vs_baseline": pct, "pct_ci95_low": lo, "pct_ci95_high": hi,
                         "cliffs_delta": cliffs_delta(a, bb), "cohens_dz": float((a - bb).mean() / (a - bb).std(ddof=1)) if (a - bb).std(ddof=1) > 0 else 0.0,
                         **res})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["p_holm"] = multipletests(df["p"].to_numpy(), method="holm")[1]
    df["significant_holm_0.05"] = df["p_holm"] < 0.05
    _save(df, cfg, "statistical_tests.csv", [])
