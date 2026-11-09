"""Task loading (frozen CSVs). The frozen order in the CSV is the single source of truth; scale subsets are
deterministic prefixes per source, so every system sees exactly the same tasks."""
import json

import pandas as pd

from src.config import ROOT


def _read(rel: str) -> list[dict]:
    df = pd.read_csv(ROOT / rel, keep_default_na=False)
    for c in ("n_tests", "prompt_tokens_est"):
        df[c] = df[c].astype(int)
    return df.to_dict("records")


def load_frozen(cfg: dict) -> list[dict]:
    return _read(cfg["data"]["humaneval_csv"]) + _read(cfg["data"]["mbpp_csv"])


def load_dev(cfg: dict) -> list[dict]:
    return _read(cfg["data"]["dev_csv"])


def subset(cfg: dict, tasks: list[dict], scale: str) -> list[dict]:
    per = cfg["scales"][scale]["tasks_per_source"]
    out = []
    for src in ("humaneval", "mbpp"):
        out += [t for t in tasks if t["source"] == src][:per]
    return out


def custom_task(description: str, asserts: list[str]) -> dict:
    return {"task_id": "custom", "source": "custom", "description": description, "prompt_stub": "", "entry_point": "",
            "test_code": json.dumps(asserts), "test_setup": "", "n_tests": len(asserts),
            "prompt_tokens_est": max(1, len(description) // 4), "difficulty": "", "reference_code": ""}
