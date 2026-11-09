"""CAAR - Carbon-Aware Agent Router. For every agent call choose tier S, M or L (two stages):
 1) capability filter: role -> acceptable tiers (configs: caar.role_tiers; widened by one tier after a failed test)
 2) carbon rule among acceptable tiers (precedence, first match wins):
      a) remaining carbon budget < 25%  (budget-aware systems only)  -> lowest acceptable tier
      b) clean grid (bottom tercile)                                   -> highest acceptable tier
      c) otherwise (medium/high carbon)                                -> lowest estimated emissions (= lowest tier)
Not used in v1 (PDF lists them as inputs but gives no rule): remaining latency budget, quality threshold, task complexity.
Also defined here: the carbon-oblivious routers used by baselines/ablations."""
from dataclasses import dataclass, field

TIERS = ["S", "M", "L"]


@dataclass
class Route:
    tier: str
    reason: str
    est_emissions_g: dict = field(default_factory=dict)


class FixedTierRouter:
    """Always the same tier (B1 / 'CAAR off' use L; B2's cascade forces S, then M, then L)."""

    def __init__(self, tier: str, cfg: dict | None = None, energy=None):
        self.tier = tier

    def route(self, role: str, ctx: dict) -> Route:
        return Route(self.tier, f"fixed tier {self.tier}")


class CostAwareRouter:
    """Carbon-oblivious cost/quality routing (B4): cheapest acceptable tier; strongest acceptable after a failure."""

    def __init__(self, cfg: dict, energy=None):
        self.role_tiers = cfg["caar"]["role_tiers"]

    def route(self, role: str, ctx: dict) -> Route:
        acc = self.role_tiers[role]
        if ctx.get("failed_before"):
            return Route(acc[-1], "cost-aware: escalate to strongest acceptable after failure")
        return Route(acc[0], "cost-aware: cheapest acceptable tier")


class CAARRouter:
    def __init__(self, cfg: dict, energy, budget_aware: bool):
        self.role_tiers = cfg["caar"]["role_tiers"]
        self.force_small = cfg["caar"]["low_budget_force_small_threshold"]
        self.est_out = cfg["caar"]["est_completion_tokens"]
        self.energy = energy
        self.budget_aware = budget_aware

    def acceptable(self, role: str, failed_before: bool) -> list:
        acc = list(self.role_tiers[role])
        if failed_before:
            top = TIERS.index(acc[-1])
            if top + 1 < len(TIERS):
                acc.append(TIERS[top + 1])
        return acc

    def route(self, role: str, ctx: dict) -> Route:
        acc = self.acceptable(role, ctx.get("failed_before", False))
        n_tok = int(ctx["est_prompt_tokens"] + self.est_out[role])
        est = {t: round(self.energy.estimate_g(t, n_tok, ctx["intensity"]), 6) for t in acc}
        frac = ctx.get("remaining_fraction")
        if self.budget_aware and frac is not None and frac < self.force_small:
            return Route(acc[0], f"budget {frac:.0%} < {self.force_small:.0%}: force lowest acceptable", est)
        if ctx["carbon_class"] == "low":
            return Route(acc[-1], "clean grid: allow highest acceptable tier", est)
        tier = min(acc, key=lambda t: est[t])
        return Route(tier, f"{ctx['carbon_class']} carbon: lowest-emission acceptable tier", est)
