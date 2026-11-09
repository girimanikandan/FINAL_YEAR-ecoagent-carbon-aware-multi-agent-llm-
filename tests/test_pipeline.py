"""Run with:  python -m pytest tests -q      (all offline; uses the mock LLM and temp directories)."""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.analysis.statistical_tests import cliffs_delta, paired_test
from src.carbon.energy import EnergyModel
from src.carbon.intensity import CarbonContext, classify, schedule_rows
from src.caar import CAARRouter, CostAwareRouter
from src.cbc import CarbonBudgetController
from src.evaluation.test_runner import build_plan, run_tests
from src.htde import decide_depth
from src.main import cmd_pilot, run_system
from src.tasks import custom_task, load_dev, load_frozen, subset

PDF_LOG_KEYS = {"run_id", "system", "task_id", "seed", "session_id", "agent_role", "model_tier", "model_name", "prompt_tokens",
                "completion_tokens", "total_tokens", "carbon_intensity_gco2_kwh", "energy_wh", "emissions_gco2eq",
                "communication_overhead_gco2eq", "latency_seconds", "api_cost_usd", "retry_number", "status", "timestamp_utc"}
PDF_TASK_KEYS = {"run_id", "system", "task_id", "seed", "final_test_pass_rate", "task_success", "total_emissions_gco2eq",
                 "total_tokens", "agent_calls", "agent_graph_depth", "retry_count", "latency_seconds", "slo_seconds",
                 "slo_compliant", "carbon_budget_gco2eq", "remaining_budget_gco2eq", "alpha", "selected_topology", "timestamp_utc"}


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    t = tmp_path_factory.mktemp("eco")
    c = C.load_all()
    c["paths"].update(raw=str(t / "raw"), processed=str(t / "processed"), cache=str(t / "cache"))
    c["data"]["schedule_csv"] = str(t / "schedule.csv")
    return c


@pytest.fixture(scope="module")
def calibrated(cfg):
    class A: provider, profile = "mock", "medium"
    cmd_pilot(cfg, A)
    return cfg


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ data + harness
def test_frozen_dataset_shape(cfg):
    fr = load_frozen(cfg)
    assert len(fr) == 200
    assert sum(t["source"] == "humaneval" for t in fr) == 100 and sum(t["source"] == "mbpp" for t in fr) == 100
    assert len({t["task_id"] for t in fr}) == 200
    dev = {t["task_id"] for t in load_dev(cfg)}
    assert len(dev) == 20 and not dev & {t["task_id"] for t in fr}, "dev tasks must be disjoint from the frozen set"


def test_reference_solutions_pass_harness(cfg):
    """Validates our reimplemented harness: every frozen/dev task's reference solution must pass 100%."""
    for t in load_frozen(cfg) + load_dev(cfg):
        r = run_tests(t, t["reference_code"], cfg["sandbox"])
        assert r.success, (t["task_id"], r.feedback)
        assert r.total == t["n_tests"] > 0


def test_wrong_code_fails_and_partial_counts(cfg):
    t = load_frozen(cfg)[0]
    assert run_tests(t, "def x():\n    return 1", cfg["sandbox"]).pass_rate == 0.0
    ct = custom_task("add", ["assert add(1,2)==3", "assert add(2,2)==4", "assert add(0,0)==1"])
    r = run_tests(ct, "def add(a,b):\n    return a+b", cfg["sandbox"])
    assert (r.passed, r.total) == (2, 3) and not r.success


def test_timeout_and_syntax_error_are_failures(cfg):
    sb = dict(cfg["sandbox"], timeout_seconds=2, cpu_seconds=2)
    ct = custom_task("loop", ["assert f()==1"])
    r = run_tests(ct, "def f():\n    while True: pass", sb)
    assert r.timed_out and r.passed == 0
    r = run_tests(ct, "def f(:\n  pass", sb)
    assert r.passed == 0 and "failed to load" in r.feedback.lower()


def test_sandbox_blocks_network_and_hides_secrets(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    ct = custom_task("env", ["assert __import__('os').environ.get('ANTHROPIC_API_KEY') is None"])
    assert run_tests(ct, "x = 1", cfg["sandbox"]).success
    net = custom_task("net", ["assert True"])
    r = run_tests(net, "import socket\nsocket.socket()", cfg["sandbox"])
    assert r.passed == 0, "network access must fail inside the sandbox"


def test_humaneval_assert_units_are_counted_individually(cfg):
    t = [x for x in load_frozen(cfg) if x["source"] == "humaneval"][0]
    plan = build_plan(t)
    assert len(plan["units"]) >= 1 and plan["entry_point"] == t["entry_point"]


# ------------------------------------------------------------------ carbon / energy
def test_energy_formula(cfg):
    e = EnergyModel(cfg)
    wh = e.energy_wh("M", 1000)
    assert wh == pytest.approx(e.e_per_token("M") * 1000)
    assert e.emissions_g(wh, 250.0) == pytest.approx(wh * 250.0 * 1e-3)
    assert e.emissions_g(2.41, 231.4) == pytest.approx(0.557674, rel=1e-4)      # PDF example record


def test_schedule_is_deterministic_and_hour1_is_task1(cfg):
    ctx = CarbonContext(cfg, "medium")
    ids = [t["task_id"] for t in load_frozen(cfg)][:10]
    o1, o2 = C.shuffled(ids, 42), C.shuffled(ids, 42)
    assert o1 == o2 and C.shuffled(ids, 43) != o1
    rows = schedule_rows(10, 42, "medium", o1, ctx)
    assert rows[0]["hour_index"] == 1 and rows[0]["carbon_intensity_gco2_kwh"] == ctx.trace.values[0]
    assert len(ctx.trace.values) == 720
    wrap = ctx.at(721)
    assert wrap["intensity"] == ctx.trace.values[0]                          # cyclic


def test_terciles_split_trace_roughly_in_thirds(cfg):
    ctx = CarbonContext(cfg, "medium")
    cls = [classify(v, ctx.thresholds) for v in ctx.trace.values]
    for k in ("low", "medium", "high"):
        assert 0.28 < cls.count(k) / len(cls) < 0.38


# ------------------------------------------------------------------ CBC / CAAR / HTDE rules
def test_cbc_policy_exactly_as_specified(cfg):
    c = CarbonBudgetController(cfg)
    assert [c.alpha(f, "medium") for f in (0.9, 0.51, 0.5, 0.25, 0.24, 0.10, 0.09, 0.0)] == [1.0, 1.0, 0.7, 0.7, 0.4, 0.4, 0.2, 0.2]
    assert c.alpha(0.9, "high") == pytest.approx(0.8) and c.alpha(0.3, "high") == pytest.approx(0.5)
    a = c.alpha(0.05, "high")
    assert 0 < a <= 0.2                                                      # stays inside (0, 1]
    assert c.apply_ceiling("L", 0.4, "M") == "M" and c.apply_ceiling("L", 0.2, "M") == "M" and c.apply_ceiling("L", 0.2, "S") == "S"
    assert not c.allow_repair(0.0) and c.allow_repair(0.01)


def test_caar_rules(cfg):
    e = EnergyModel(cfg)
    r = CAARRouter(cfg, e, budget_aware=True)
    base = {"est_prompt_tokens": 400, "intensity": 100.0, "remaining_fraction": 0.9, "failed_before": False}
    assert r.route("coder", {**base, "carbon_class": "low"}).tier == "L"             # clean grid: highest acceptable
    assert r.route("coder", {**base, "carbon_class": "high"}).tier == "M"            # dirty grid: lowest acceptable
    assert r.route("coder", {**base, "carbon_class": "low", "remaining_fraction": 0.2}).tier == "M"   # <25% budget forces lowest
    assert r.route("planner", {**base, "carbon_class": "low"}).tier == "M"           # planner acceptable = S/M
    assert r.route("reviewer", {**base, "carbon_class": "low", "failed_before": True}).tier == "L"    # widened after failure
    nb = CAARRouter(cfg, e, budget_aware=False)
    assert nb.route("coder", {**base, "carbon_class": "low", "remaining_fraction": 0.2}).tier == "L"  # B3 ignores budget
    assert CostAwareRouter(cfg).route("coder", {**base, "carbon_class": "low"}).tier == "M"           # B4 never uses carbon


def test_htde_expansion_rule():
    table = {"simple": {"1": {"q": .5, "e_wh": 1.0}, "2": {"q": .8, "e_wh": 2.0}, "3": {"q": .85, "e_wh": 6.0}}}
    # dQ2=.3, dE2=1Wh*100g/kWh*1e-3=.1g -> ratio 3 ; dQ3=.05, dE3=.4g -> ratio .125
    d, steps = decide_depth(table, "simple", 100.0, alpha=1.0, theta=1.0, eps=0.01)
    assert d == 2 and steps[0]["expand"] and not steps[1]["expand"]
    assert decide_depth(table, "simple", 100.0, 1.0, theta=5.0, eps=0.01)[0] == 1          # higher theta -> stays shallow
    assert decide_depth(table, "simple", 100.0, 0.2, theta=1.0, eps=0.01)[0] == 1          # low alpha raises the bar (theta/alpha=5)
    assert decide_depth(table, "simple", 100.0, 1.0, theta=0.01, eps=0.01)[0] == 3
    assert decide_depth(table, "simple", 100.0, 1.0, 0.01, 0.01, remaining_g=0.25)[0] == 2   # D2 costs .2g (ok), D3 .6g blocked
    assert decide_depth(table, "simple", 100.0, 1.0, 0.01, 0.01, remaining_g=0.15)[0] == 1   # even D2 (.2g) no longer affordable


# ------------------------------------------------------------------ end-to-end behaviour (mock LLM)
ALL_SYSTEMS = ["b1", "b2", "b3", "b4", "ecoagent", "abl_no_cbc", "abl_no_htde", "abl_no_caar", "abl_htde_only", "abl_caar_only", "abl_cbc_only"]


@pytest.mark.parametrize("system", ALL_SYSTEMS)
def test_every_system_runs_and_logs_to_schema(calibrated, system):
    res, rid = run_system(calibrated, system, 42, "smoke", "medium", "mock", "main", f"t-{system}", quiet=True, resume=False)
    assert len(res) == 6
    d = C.path(calibrated, "raw") / rid
    calls, tasks = read_jsonl(d / "llm_calls.jsonl"), read_jsonl(d / "task_results.jsonl")
    assert PDF_LOG_KEYS <= set(calls[0]) and PDF_TASK_KEYS <= set(tasks[0])
    for t in tasks:                                                          # task totals are exactly the sum of its calls
        cs = [c for c in calls if c["task_id"] == t["task_id"] and c["total_tokens"] > 0]
        assert t["agent_calls"] == len(cs) and t["total_tokens"] == sum(c["total_tokens"] for c in cs)
        assert t["total_emissions_gco2eq"] == pytest.approx(sum(c["emissions_gco2eq"] + c["communication_overhead_gco2eq"] for c in cs))
        assert t["agent_calls"] <= (12 if system == "b2" else 6)
    for c in calls:
        assert c["emissions_gco2eq"] == pytest.approx(c["energy_wh"] * c["carbon_intensity_gco2_kwh"] * 1e-3)
        assert c["total_tokens"] == c["prompt_tokens"] + c["completion_tokens"]


def test_b1_always_large_and_b4_never_uses_carbon(calibrated):
    _, rid = run_system(calibrated, "b1", 42, "smoke", "medium", "mock", "main", "t-b1x", quiet=True, resume=False)
    assert {c["model_tier"] for c in read_jsonl(C.path(calibrated, "raw") / rid / "llm_calls.jsonl")} == {"L"}
    _, rid = run_system(calibrated, "b4", 42, "smoke", "medium", "mock", "main", "t-b4x", quiet=True, resume=False)
    _, rid2 = run_system(calibrated, "b4", 42, "smoke", "high", "mock", "main", "t-b4y", quiet=True, resume=False)
    tiers = lambda r: [(c["task_id"], c["agent_role"], c["model_tier"]) for c in read_jsonl(C.path(calibrated, "raw") / r / "llm_calls.jsonl")]
    assert tiers(rid) == tiers(rid2), "B4 decisions must not depend on the carbon trace"


def test_b2_cascade_counts_every_retry(calibrated):
    res, rid = run_system(calibrated, "b2", 42, "smoke", "medium", "mock", "main", "t-b2x", quiet=True, resume=False)
    calls = read_jsonl(C.path(calibrated, "raw") / rid / "llm_calls.jsonl")
    esc = [r for r in res if r["retry_count"] > 0]
    assert esc, "mock tier S fails often enough that the cascade must escalate on some task"
    for r in esc:
        tiers = [c["model_tier"] for c in calls if c["task_id"] == r["task_id"]]
        assert "S" in tiers and len(tiers) == r["agent_calls"]                # failed tier-S attempt is still charged


def test_repair_is_limited_to_one_round_and_is_test_driven(calibrated):
    res, rid = run_system(calibrated, "b1", 42, "smoke", "medium", "mock", "main", "t-rep", quiet=True, resume=False)
    calls = read_jsonl(C.path(calibrated, "raw") / rid / "llm_calls.jsonl")
    for r in res:
        roles = [c["agent_role"] for c in calls if c["task_id"] == r["task_id"]]
        assert roles.count("repair") <= 1 and roles.count("reviewer") <= 1
        assert ("repair" in roles) == (r["retry_count"] == 1)


def test_cbc_actually_constrains_when_budget_is_tiny(calibrated):
    _, rid = run_system(calibrated, "abl_cbc_only", 42, "smoke", "medium", "mock", "main", "t-tiny", budget_per_task_g=1e-6, quiet=True, resume=False)
    tasks = read_jsonl(C.path(calibrated, "raw") / rid / "task_results.jsonl")
    calls = read_jsonl(C.path(calibrated, "raw") / rid / "llm_calls.jsonl")
    assert all(c["agent_role"] != "repair" for c in calls), "no repair once the budget is exhausted"
    assert all(t["alpha"] is not None and t["alpha"] <= 0.2 for t in tasks)


def test_raw_logs_immutable_and_resume_skips_done(calibrated):
    run_system(calibrated, "b3", 7, "smoke", "medium", "mock", "main", "t-res", quiet=True, resume=False)
    with pytest.raises(FileExistsError):
        run_system(calibrated, "b3", 7, "smoke", "medium", "mock", "main", "t-res", quiet=True, resume=False)
    res, rid = run_system(calibrated, "b3", 7, "smoke", "medium", "mock", "main", "t-res", quiet=True, resume=True)
    assert res == []                                                         # nothing re-run
    assert len(read_jsonl(C.path(calibrated, "raw") / rid / "task_results.jsonl")) == 6


def test_htde_refuses_stale_or_foreign_calibration(cfg):
    from src.htde import HTDE, HTDECalibrationError
    with pytest.raises(HTDECalibrationError):
        HTDE(cfg, "ollama", [100, 200])                                      # calibration must come from the same provider


# ------------------------------------------------------------------ statistics
def test_stats_helpers():
    a, b = np.array([1., 2, 3, 4]), np.array([5., 6, 7, 8])
    assert cliffs_delta(a, b) == -1.0 and cliffs_delta(b, a) == 1.0 and cliffs_delta(a, a) == 0.0
    r = paired_test(np.arange(30, dtype=float), np.arange(30, dtype=float) + 1 + np.random.default_rng(0).normal(0, .1, 30))
    assert r["p"] < 0.001
    assert paired_test(a, a)["p"] == 1.0


def test_analysis_flags_provisional_results(calibrated):
    from src.analysis.aggregate_results import run_all
    out = run_all(calibrated, include_mock=True)
    assert out["provenance"]["provisional"] is True
    assert any(("SYNTHETIC" in w) or ("MOCK" in w) for w in out["provenance"]["warnings"])
