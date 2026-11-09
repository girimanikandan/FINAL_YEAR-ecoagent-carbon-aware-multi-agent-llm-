"""Replayed carbon-intensity traces (never live API calls during experiments) and the deterministic task->hour schedule."""
import numpy as np
import pandas as pd

from src.config import ROOT, data_path

SYNTHETIC_SOURCES = {"synthetic_placeholder"}


class CarbonTrace:
    def __init__(self, profile: str, df: pd.DataFrame):
        self.profile = profile
        self.df = df.reset_index(drop=True)
        self.values = self.df["carbon_intensity_gco2_kwh"].astype(float).to_numpy()
        self.source = str(self.df["source"].iloc[0]) if "source" in self.df.columns else "unknown"

    @classmethod
    def load(cls, cfg: dict, profile: str) -> "CarbonTrace":
        p = ROOT / cfg["data"]["carbon"][profile]
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run `python scripts/fetch_carbon_data.py` (real data) or "
                f"`python scripts/make_placeholder_carbon.py` (synthetic, test-only).")
        return cls(profile, pd.read_csv(p))

    @property
    def is_real(self) -> bool:
        return self.source not in SYNTHETIC_SOURCES

    def hour(self, hour_index: int) -> tuple[str, float]:
        """hour_index is 1-based ("Task 1 uses hour 1"); wraps cyclically."""
        i = (hour_index - 1) % len(self.values)
        return str(self.df["datetime"].iloc[i]), float(self.values[i])

    def thresholds(self, cfg: dict, pooled: np.ndarray | None = None) -> tuple[float, float]:
        vals = pooled if pooled is not None else self.values
        lo = float(np.percentile(vals, cfg["caar"]["low_carbon_percentile"]))
        hi = float(np.percentile(vals, cfg["caar"]["high_carbon_percentile"]))
        return lo, hi


def classify(value: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    if value <= lo:
        return "low"
    if value > hi:
        return "high"
    return "medium"


class CarbonContext:
    """Loads one profile and exposes thresholds under the configured scope."""

    def __init__(self, cfg: dict, profile: str):
        self.cfg = cfg
        self.profile = profile
        self.trace = CarbonTrace.load(cfg, profile)
        if cfg["caar"]["threshold_scope"] == "pooled":
            pooled = np.concatenate([CarbonTrace.load(cfg, p).values for p in ("low", "medium", "high")])
            self.thresholds = self.trace.thresholds(cfg, pooled)
        else:
            self.thresholds = self.trace.thresholds(cfg)
            self.problems = self._check()

    def _check(self) -> list:
        """A trace is unusable if the terciles collapse or most hours are ~0."""
        v, (lo, hi), out = self.trace.values, self.thresholds, []
        if hi - lo < 1e-6:
            out.append(f"tercile thresholds collapsed to ({lo:.2f}, {hi:.2f}): too many identical values")
        zero = float((v <= 0.5).mean())
        if zero > 0.05:
            out.append(f"{zero:.0%} of hours are ~0 gCO2/kWh -> emissions ~0, budgets never consumed, dE ~ 0 in HTDE")
        return out

    def at(self, hour_index: int) -> dict:
        dt, val = self.trace.hour(hour_index)
        return {"hour_index": hour_index, "datetime": dt, "intensity": val,
                "carbon_class": classify(val, self.thresholds)}


def schedule_rows(n_tasks: int, seed: int, profile: str, ordered_task_ids: list[str], ctx: CarbonContext) -> list[dict]:
    rows = []
    for pos, tid in enumerate(ordered_task_ids, start=1):
        c = ctx.at(pos)
        rows.append({"n_tasks": n_tasks, "seed": seed, "profile": profile, "position": pos, "task_id": tid,
                     "hour_index": c["hour_index"], "datetime": c["datetime"],
                     "carbon_intensity_gco2_kwh": c["intensity"], "carbon_class": c["carbon_class"]})
    return rows


def ensure_schedule(cfg: dict, rows: list[dict]) -> None:
    """Append any (n_tasks, seed, profile) block that is not yet in data/task_carbon_schedule.csv."""
    path = data_path(cfg, "schedule_csv")
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        key = ["n_tasks", "seed", "profile"]
        have = set(map(tuple, old[key].drop_duplicates().to_numpy().tolist()))
        k = tuple(new[key].iloc[0].tolist())
        if k in have:
            return
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
