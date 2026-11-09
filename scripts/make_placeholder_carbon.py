"""SYNTHETIC placeholder traces so the pipeline can be tested offline. NOT real data.
Every row is tagged source=synthetic_placeholder; the runner warns, and the analysis stamps every output PROVISIONAL.
Replace with `scripts/fetch_carbon_data.py` before producing any result you intend to report."""
import math
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {  # mean, diurnal amplitude, noise sd, floor
    "low":    (55, 25, 12, 15),
    "medium": (200, 70, 25, 80),
    "high":   (620, 90, 30, 420),
}
H = 30 * 24
for i, (name, (mean, amp, sd, floor)) in enumerate(PROFILES.items()):
    rng = random.Random(1000 + i)
    weather = 0.0
    rows = []
    for h in range(H):
        weather = 0.97 * weather + rng.gauss(0, sd * 0.35)          # slow "weather" drift
        diurnal = -amp * math.cos(2 * math.pi * ((h % 24) - 14) / 24)   # trough around 14:00 (solar)
        v = max(floor, mean + diurnal + weather + rng.gauss(0, sd * 0.3))
        rows.append({"datetime": f"2026-09-{1 + h // 24:02d}T{h % 24:02d}:00:00Z", "carbon_intensity_gco2_kwh": round(v, 2),
                     "source": "synthetic_placeholder"})
    pd.DataFrame(rows).to_csv(ROOT / "data" / f"carbon_intensity_{name}.csv", index=False)
    print(name, "mean", round(sum(r["carbon_intensity_gco2_kwh"] for r in rows) / H, 1))
