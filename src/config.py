"""Config loading. Every experimental setting lives in configs/*.yaml (nothing is hard-coded in Python)."""
import copy
import hashlib
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all(overrides: dict | None = None) -> dict:
    """Return one merged dict: experiment/session/htde/caar/cbc/accounting/sandbox/llm + models + baselines + matrix."""
    cfg = _yaml("ecoagent.yaml")
    cfg["models"] = _yaml("models.yaml")
    cfg["baselines"] = _yaml("baselines.yaml")
    exp = _yaml("experiment.yaml")
    cfg["matrix"] = exp["matrix"]
    cfg["scales"] = exp["scales"]
    cfg["data"] = exp["data"]
    cfg["paths"] = exp["paths"]
    measured = CONFIG_DIR / "energy_measured.yaml"
    cfg["energy_measured"] = _yaml("energy_measured.yaml") if measured.exists() else {}
    for k, v in (overrides or {}).items():       # dotted keys, e.g. {"cbc.alpha_floor": 0.1}
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return cfg


def path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def data_path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["data"][key]


def tier_models(cfg: dict) -> dict:
    """{tier: {name, params_b, provider, endpoint, ..., max_output_tokens}} for the active profile."""
    m = cfg["models"]
    prof = m["profiles"][m["active_profile"]]
    out = {}
    for tier in ("S", "M", "L"):
        d = dict(m["common"])
        d.update(prof[tier])
        d["max_output_tokens"] = m["max_output_tokens"][tier]
        out[tier] = d
    return out


def config_hash(cfg: dict) -> str:
    slim = {k: v for k, v in cfg.items() if k not in ("energy_measured",)}
    return hashlib.sha256(json.dumps(slim, sort_keys=True, default=str).encode()).hexdigest()[:12]


def prompt_template(cfg: dict, role: str) -> str:
    version = cfg["experiment"]["prompt_version"]
    with open(CONFIG_DIR / "prompts" / version / f"{role}.txt", "r", encoding="utf-8") as f:
        return f.read()


def shuffled(items: list, seed: int) -> list:
    """Deterministic seeded shuffle (the single source of truth for task order)."""
    out = list(items)
    random.Random(seed).shuffle(out)
    return out
