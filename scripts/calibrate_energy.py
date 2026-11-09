"""Measure energy per token for each tier on THIS machine with CodeCarbon (PDF: measured > estimated for local runs).

  pip install codecarbon
  python scripts/calibrate_energy.py --calls 8

Method: (1) measure idle machine power for --idle-seconds; (2) run --calls real coder prompts per tier (cache OFF) inside a
CodeCarbon tracker; (3) net energy = tracked energy - idle_power x duration; (4) Wh/token = net Wh / (prompt+completion tokens).
Writes configs/energy_measured.yaml, which then overrides the assumed values in configs/models.yaml automatically.
CAVEATS (state them in the paper): CodeCarbon measures the whole machine (CPU/GPU/RAM estimate), not just Ollama, so keep
the machine otherwise idle; on CPU-only laptops CodeCarbon uses TDP-based estimates, not a wattmeter reading.
STATUS: could not be run in the build environment (no Ollama / CodeCarbon hardware access)."""
import argparse
import datetime
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.model_pool import ModelPool
from src.tasks import load_dev


def tracked(fn):
    from codecarbon import EmissionsTracker
    tr = EmissionsTracker(project_name="ecoagent_energy_calibration", save_to_file=False, log_level="error", measure_power_secs=1)
    tr.start()
    t0 = time.perf_counter()
    out = fn()
    dur = time.perf_counter() - t0
    tr.stop()
    kwh = tr._total_energy.kWh
    return out, kwh * 1000.0, dur       # Wh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--idle-seconds", type=int, default=30)
    a = ap.parse_args()
    cfg = C.load_all({"llm.cache_enabled": False})
    pool = ModelPool(cfg, cfg["models"]["provider_default"])
    from src.agents.coder import Coder
    coder = Coder(cfg)
    prompts = [coder.prompt(t["description"]) for t in load_dev(cfg)[: a.calls]]
    _, idle_wh, idle_s = tracked(lambda: time.sleep(a.idle_seconds))
    idle_w = idle_wh / (idle_s / 3600)
    print(f"idle power ~ {idle_w:.1f} W")
    out = {}
    for tier in "SML":
        def work():
            tot = 0
            for p in prompts:
                r = pool.generate(tier, p, "coder", {})
                if r.status != "success":
                    raise SystemExit(f"tier {tier} call failed: {r.error}")
                tot += r.total_tokens
            return tot
        tokens, wh, dur = tracked(work)
        net = max(wh - idle_w * dur / 3600, 0.0)
        out[tier] = {"wh_per_token": net / tokens, "method": "codecarbon_idle_subtracted", "model": pool.info(tier)["name"],
                     "n_calls": len(prompts), "n_tokens": tokens, "gross_wh": wh, "net_wh": net, "idle_watts": idle_w,
                     "measured_on": datetime.datetime.now().isoformat(timespec="seconds")}
        print(tier, out[tier])
    (ROOT / "configs" / "energy_measured.yaml").write_text(yaml.safe_dump(out), encoding="utf-8")
    print("wrote configs/energy_measured.yaml")


if __name__ == "__main__":
    main()
