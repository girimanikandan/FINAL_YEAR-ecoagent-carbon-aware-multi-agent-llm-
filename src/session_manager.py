"""Session Manager (layer 1): owns the session carbon ledger, SLO clock and per-task execution.
For one task it: decides topology (fixed / HTDE / complexity / cascade) -> for each agent call routes a tier (CAAR or a
baseline router, then the CBC ceiling) -> calls the model -> logs EVERY call (retries included) -> runs the hidden tests
-> optional single repair round -> writes the task result. Nothing is computed from console output."""
import time
import uuid
from dataclasses import dataclass

from src.agents.coder import Coder
from src.agents.planner import Planner
from src.agents.repair import Repair
from src.agents.reviewer import Reviewer
from src.carbon.ledger import SessionLedger, utc_now
from src.caar import FixedTierRouter
from src.evaluation.test_runner import TestResult, run_tests
from src.htde import assign_bin

TIERS = ["S", "M", "L"]


@dataclass
class Components:
    cfg: dict
    spec: dict
    pool: object
    energy: object
    router: object
    cbc: object | None
    htde: object | None
    bin_edges: list
    logger: object
    meta: dict                 # run_id, system, seed, profile, provider, carbon_source, is_mock


class SessionManager:
    def __init__(self, comp: Components, budget_g: float, tasks_in_session: int):
        self.c = comp
        self.cfg = comp.cfg
        self.spec = comp.spec
        self.ledger = SessionLedger(budget_g)
        self.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self.tasks_remaining = tasks_in_session
        self.slo = float(self.cfg["experiment"]["latency_slo_seconds"])
        self.agents = {"planner": Planner(self.cfg), "coder": Coder(self.cfg), "reviewer": Reviewer(self.cfg), "repair": Repair(self.cfg)}
        self.max_calls = int(self.spec.get("max_agent_calls", self.cfg["htde"]["max_agent_calls"]))
        self.max_repairs = int(self.cfg["htde"]["max_repair_rounds"])

    # ------------------------------------------------------------------ public
    def run_task(self, task: dict, carbon: dict, position: int) -> dict:
        self._reset(task, carbon, position)
        wall0 = time.perf_counter()
        topo = self.spec["topology"]
        alpha = self._alpha() if self.c.cbc else None
        htde_trace, depth = None, None
        if topo.startswith("fixed_d"):
            depth = int(topo[-1])
        elif topo == "htde":
            remaining = self.ledger.remaining_g if self.spec.get("budget_aware") or self.spec.get("cbc") else None
            depth, htde_trace = self.c.htde.decide(task, alpha if alpha is not None else 1.0, carbon["intensity"], remaining)
        elif topo == "complexity":
            b = assign_bin(task["prompt_tokens_est"], self.c.bin_edges)
            depth = self.cfg["baselines"]["b4"]["depth_by_bin"][b]
            htde_trace = {"bin": b, "rule": "complexity only (no carbon)"}
        if topo == "cascade":
            code, tr = self._cascade()
        else:
            code, tr = self._run_depth(depth, self.c.router)
        res = self._task_record(task, tr, depth, htde_trace, time.perf_counter() - wall0, code)
        self.tasks_remaining -= 1
        self.c.logger.log_task(res)
        return res

    # ------------------------------------------------------------------ per-task state
    def _reset(self, task, carbon, position):
        self.task, self.carbon, self.position = task, carbon, position
        self.records, self.roles = [], []
        self.pass_index, self.repairs_done, self.failed_before = 0, 0, False
        self.test_seconds, self.last_alpha = 0.0, None

    def _alpha(self) -> float:
        a = self.c.cbc.alpha(self.ledger.remaining_fraction, self.carbon["carbon_class"])
        self.last_alpha = a
        return a

    # ------------------------------------------------------------------ topologies
    def _run_depth(self, depth: int, router, allow_repair: bool = True):
        desc = self.task["description"]
        plan = ""
        if depth >= 2:
            r = self._agent_call("planner", self.agents["planner"].prompt(desc), router)
            plan = self.agents["planner"].parse(r.text) if r else ""
        r = self._agent_call("coder", self.agents["coder"].prompt(desc, plan), router)
        code = self.agents["coder"].parse(r.text) if r else ""
        tr = self._test(code)
        if depth >= 3 and allow_repair and self._should_repair(tr):
            self.pass_index += 1
            self.repairs_done += 1
            r = self._agent_call("reviewer", self.agents["reviewer"].prompt(desc, code, tr.feedback), router)
            review = self.agents["reviewer"].parse(r.text) if r else ""
            r = self._agent_call("repair", self.agents["repair"].prompt(desc, code, review, tr.feedback), router)
            fixed = self.agents["repair"].parse(r.text) if r else ""
            if fixed:
                code = fixed
            tr = self._test(code)
        return code, tr

    def _cascade(self):
        """B2: try S, then M, then L (whole fixed topology each time) until pass rate >= tau_q. All calls are counted."""
        tau = self.cfg["experiment"]["quality_threshold"]
        code, tr = "", TestResult()
        for level, tier in enumerate(TIERS):
            if level > 0:
                self.pass_index += 1
            code, tr = self._run_depth(3, FixedTierRouter(tier))
            if tr.total == 0 or tr.pass_rate >= tau:
                break
        return code, tr

    def _should_repair(self, tr: TestResult) -> bool:
        if tr.total == 0 or tr.success or self.repairs_done >= self.max_repairs:
            return False
        if self.c.cbc and not self.c.cbc.allow_repair(self.ledger.remaining_g):
            return False
        return True

    def _test(self, code: str) -> TestResult:
        tr = run_tests(self.task, code, self.cfg["sandbox"])
        self.test_seconds += tr.seconds
        if tr.total > 0 and not tr.success:
            self.failed_before = True
        return tr

    # ------------------------------------------------------------------ one agent call (routing + retries + logging)
    def _agent_call(self, role: str, prompt: str, router):
        cfg, spec = self.cfg, self.spec
        frac = self.ledger.remaining_fraction
        ctx = {"carbon_class": self.carbon["carbon_class"], "intensity": self.carbon["intensity"],
               "remaining_fraction": frac, "failed_before": self.failed_before,
               "est_prompt_tokens": max(1, len(prompt) // 4)}
        route = router.route(role, ctx)
        tier, reason, alpha = route.tier, route.reason, None
        if self.c.cbc:
            alpha = self._alpha()
            floor_tier = cfg["caar"]["role_tiers"][role][0]
            capped = self.c.cbc.apply_ceiling(tier, alpha, floor_tier)
            if capped != tier:
                reason += f"; CBC ceiling alpha={alpha:.2f} -> {capped}"
                tier = capped
        result = None
        for attempt in range(1 + int(cfg["llm"]["max_call_retries"])):
            if len(self.records) >= self.max_calls:
                return None
            result = self.c.pool.generate(tier, prompt, role, {"task": self.task})
            self._log_call(role, tier, result, reason, alpha, attempt, route.est_emissions_g)
            if result.status == "success":
                break
        return result if result and result.status == "success" else None

    def _log_call(self, role, tier, r, reason, alpha, attempt, est):
        info = self.c.pool.info(tier)
        processed = r.total_tokens > 0
        wh = self.c.energy.energy_wh(tier, r.total_tokens) if processed else 0.0
        g = self.c.energy.emissions_g(wh, self.carbon["intensity"])
        comm = self.c.energy.comm_overhead_g(g)
        cost = (r.prompt_tokens * info["cost_per_1k_input_usd"] + r.completion_tokens * info["cost_per_1k_output_usd"]) / 1000
        m = self.c.meta
        rec = {"run_id": m["run_id"], "system": m["system"], "task_id": self.task["task_id"], "seed": m["seed"],
               "session_id": self.session_id, "agent_role": role, "model_tier": tier,
               "model_name": info["name"] if not self.c.pool.is_mock else f"mock:{info['name']}",
               "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "total_tokens": r.total_tokens,
               "carbon_intensity_gco2_kwh": self.carbon["intensity"], "energy_wh": wh, "emissions_gco2eq": g,
               "communication_overhead_gco2eq": comm, "latency_seconds": r.latency_seconds, "api_cost_usd": cost,
               "retry_number": self.pass_index, "status": r.status, "timestamp_utc": utc_now(),
               # extras (explainability / provenance)
               "call_attempt": attempt, "routing_reason": reason, "alpha": alpha, "carbon_class": self.carbon["carbon_class"],
               "est_emissions_by_tier": est, "cache_hit": r.cache_hit, "token_source": r.token_source,
               "latency_simulated": r.latency_simulated, "energy_source": self.c.energy.source(tier),
               "provider": m["provider"], "is_mock": m["is_mock"], "profile": m["profile"], "error": r.error or None}
        self.c.logger.log_call(rec)
        self.records.append(rec)
        if processed:
            self.roles.append(role)
        self.ledger.add(g + comm)

    # ------------------------------------------------------------------ task result
    def _task_record(self, task, tr: TestResult, depth, htde_trace, wall, code: str = "") -> dict:
        m = self.c.meta
        recs = [r for r in self.records if r["total_tokens"] > 0]
        inf = sum(r["emissions_gco2eq"] for r in recs)
        comm = sum(r["communication_overhead_gco2eq"] for r in recs)
        stages = {"planner": "planner", "coder": "coder", "repair": "coder", "reviewer": "reviewer"}
        executed = []
        for r in self.roles:
            if r not in executed:
                executed.append(r)
        latency = sum(r["latency_seconds"] for r in recs) + self.test_seconds
        tiers = {t: sum(1 for r in recs if r["model_tier"] == t) for t in TIERS}
        verified = tr.total > 0
        return {
            "run_id": m["run_id"], "system": m["system"], "task_id": task["task_id"], "source": task["source"],
            "seed": m["seed"], "profile": m["profile"], "position": self.position,
            "hour_index": self.carbon["hour_index"], "carbon_intensity_gco2_kwh": self.carbon["intensity"],
            "carbon_class": self.carbon["carbon_class"],
            "final_test_pass_rate": tr.pass_rate if verified else None, "task_success": int(tr.success) if verified else None,
            "passed_tests": tr.passed, "total_tests": tr.total,
            "total_emissions_gco2eq": inf + comm, "inference_emissions_gco2eq": inf, "communication_overhead_gco2eq": comm,
            "inference_energy_wh": sum(r["energy_wh"] for r in recs),
            "total_tokens": sum(r["total_tokens"] for r in recs), "agent_calls": len(recs),
            "agent_graph_depth": len({stages[r] for r in executed}), "retry_count": self.pass_index,
            "latency_seconds": latency, "wall_clock_seconds": wall, "slo_seconds": self.slo,
            "slo_compliant": int(latency <= self.slo),
            "carbon_budget_gco2eq": self.ledger.budget_g, "remaining_budget_gco2eq": self.ledger.remaining_g,
            "alpha": self.last_alpha, "selected_depth": depth, "selected_topology": "+".join(executed),
            "htde_trace": htde_trace, "api_cost_usd": sum(r["api_cost_usd"] for r in recs),
            "tier_counts": tiers, "cache_hits": sum(1 for r in recs if r["cache_hit"]),
            "provider": m["provider"], "is_mock": m["is_mock"], "carbon_source": m["carbon_source"],
            "sanitized_dropped": tr.dropped, "final_code": (code or "")[:6000], "test_feedback": tr.feedback[:600],
            "timestamp_utc": utc_now(),
        }
