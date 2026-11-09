"""CBC - Carbon Budget Controller (rule-based, session-level). NOT reinforcement learning.

Output: scalar aggressiveness alpha in (0, 1] (smaller = more conservative).
  alpha = 1.0 if remaining budget > 50%
          0.7 if remaining in [25%, 50%]
          0.4 if remaining in [10%, 25%)
          0.2 if remaining < 10%
  then alpha -= 0.2 when current carbon intensity is in the high-carbon tercile (floored at cbc.alpha_floor so alpha stays > 0).
Consumers: HTDE (expansion threshold theta / max(alpha, eps)) and CAAR (force lowest tier when < 25% budget).
Direct actuation (so "CBC only" is meaningful; see README 'Interpretation decisions'):
  - tier ceiling: alpha <= 0.4 -> at most tier M; alpha <= 0.2 -> at most tier S (never below the role's lowest acceptable tier)
  - repair is skipped once the budget is exhausted."""
TIERS = ["S", "M", "L"]


class CarbonBudgetController:
    def __init__(self, cfg: dict):
        c = cfg["cbc"]
        self.lv = c["budget_levels"]
        self.av = c["alpha_values"]
        self.penalty = c["high_carbon_penalty"]
        self.floor = c["alpha_floor"]
        self.ceil_m = c["tier_ceiling"]["m_at_or_below_alpha"]
        self.ceil_s = c["tier_ceiling"]["s_at_or_below_alpha"]

    def alpha(self, remaining_fraction: float, carbon_class: str) -> float:
        f = remaining_fraction
        if f > self.lv["high"]:
            a = self.av["high"]
        elif f >= self.lv["medium"]:
            a = self.av["medium"]
        elif f >= self.lv["low"]:
            a = self.av["low"]
        else:
            a = self.av["critical"]
        if carbon_class == "high":
            a = max(self.floor, round(a - self.penalty, 10))
        return a

    def apply_ceiling(self, tier: str, alpha: float, floor_tier: str) -> str:
        ceiling = "S" if alpha <= self.ceil_s else ("M" if alpha <= self.ceil_m else "L")
        capped = TIERS[min(TIERS.index(tier), TIERS.index(ceiling))]
        return TIERS[max(TIERS.index(capped), TIERS.index(floor_tier))]

    @staticmethod
    def allow_repair(remaining_g: float) -> bool:
        return remaining_g > 0

    def state(self, ledger, carbon_class: str, intensity: float, tasks_remaining: int, elapsed_s: float, slo_s: float) -> dict:
        return {"budget_gco2eq": ledger.budget_g, "spent_gco2eq": ledger.spent_g, "remaining_gco2eq": ledger.remaining_g,
                "remaining_fraction": ledger.remaining_fraction, "carbon_intensity": intensity, "carbon_class": carbon_class,
                "completed_agent_calls": ledger.n_calls, "tasks_remaining": tasks_remaining,
                "elapsed_s": elapsed_s, "latency_slo_s": slo_s}
