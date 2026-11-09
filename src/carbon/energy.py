"""Energy and emissions accounting.   E_i(t) = e_i * n_tok * c(t) * 1e-3   [gCO2eq]
e_i in Wh/token, n_tok = input + output tokens, c(t) in gCO2eq/kWh. Wh -> kWh is the 1e-3."""


class EnergyModel:
    def __init__(self, cfg: dict):
        m = cfg["models"]
        self._assumed = m["energy_per_token_wh_assumed"]
        self._assumed_src = m["energy_source_assumed"]
        self._measured = cfg.get("energy_measured") or {}
        self.comm_fraction = cfg["accounting"]["comm_overhead_fraction"]

    def e_per_token(self, tier: str) -> float:
        if tier in self._measured:
            return float(self._measured[tier]["wh_per_token"])
        return float(self._assumed[tier])

    def source(self, tier: str) -> str:
        return self._measured[tier].get("method", "measured") if tier in self._measured else self._assumed_src

    def energy_wh(self, tier: str, total_tokens: int) -> float:
        return self.e_per_token(tier) * total_tokens

    @staticmethod
    def emissions_g(energy_wh: float, intensity_gco2_kwh: float) -> float:
        return energy_wh * intensity_gco2_kwh * 1e-3

    def comm_overhead_g(self, inference_emissions_g: float) -> float:
        return self.comm_fraction * inference_emissions_g

    def estimate_g(self, tier: str, est_total_tokens: int, intensity: float) -> float:
        return self.emissions_g(self.energy_wh(tier, est_total_tokens), intensity)
