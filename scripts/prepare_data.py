"""Freeze the benchmark subsets. Run ONCE; commit the resulting CSVs; never re-sample per system.

  python scripts/prepare_data.py            # uses data/raw_benchmarks/*, downloads them if missing

Steps: load HumanEval (164) + MBPP (974) -> validate every reference solution through OUR harness (so the
harness is verified, and broken tasks are excluded transparently) -> seeded sample of 100+100 frozen tasks plus a
disjoint 10+10 dev set for the pilot/HTDE calibration -> write CSVs + data/data_validation.json."""
import ast
import gzip
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import load_all
from src.evaluation.test_runner import run_tests, build_plan

RAW = ROOT / "data" / "raw_benchmarks"
URLS = {
    "he.jsonl.gz": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    "mbpp.jsonl": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if not (RAW / name).exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, RAW / name)


def load_humaneval():
    rows = []
    with gzip.open(RAW / "he.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            num = d["task_id"].split("/")[1]
            rows.append({"task_id": f"HumanEval_{num}", "source": "humaneval", "description": d["prompt"],
                         "prompt_stub": d["prompt"], "entry_point": d["entry_point"], "test_code": d["test"],
                         "test_setup": "", "reference_code": d["prompt"] + d["canonical_solution"],
                         "n_tests": None, "difficulty": ""})
    return rows


def mbpp_entry(code: str, tests: list[str]):
    try:
        defined = [n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:
        return None
    called = set()
    for t in tests:
        try:
            for n in ast.walk(ast.parse(t)):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    called.add(n.func.id)
        except SyntaxError:
            return None
    for name in defined:
        if name in called:
            return name
    return None


def load_mbpp():
    rows = []
    with open(RAW / "mbpp.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tests = d["test_list"]
            entry = mbpp_entry(d["code"], tests)
            if not entry or not tests:
                continue
            desc = f"{d['text']}\nYour code should pass this test:\n{tests[0]}"
            rows.append({"task_id": f"MBPP_{d['task_id']}", "source": "mbpp", "description": desc, "prompt_stub": "",
                         "entry_point": entry, "test_code": json.dumps(tests), "test_setup": d.get("test_setup_code", ""),
                         "reference_code": d["code"], "n_tests": len(tests), "difficulty": ""})
    return rows


def validate(rows, sandbox_cfg):
    ok, bad = [], []
    for r in rows:
        plan = build_plan(r)
        r["n_tests"] = len(plan["units"])
        res = run_tests(r, r["reference_code"], sandbox_cfg)
        (ok if res.success else bad).append(r if res.success else {"task_id": r["task_id"], "why": res.feedback[:200]})
    return ok, bad


def main():
    cfg = load_all()
    fetch()
    sb = dict(cfg["sandbox"], timeout_seconds=20, cpu_seconds=20)
    report = {}
    frozen, dev = [], []
    n, nd = cfg["data"]["n_per_source"], cfg["data"]["n_dev_per_source"]
    for name, loader in (("humaneval", load_humaneval), ("mbpp", load_mbpp)):
        rows = loader()
        ok, bad = validate(rows, sb)
        report[name] = {"loaded": len(rows), "reference_passes_harness": len(ok), "excluded": bad}
        rng = random.Random(cfg["data"]["freeze_seed"])
        ids = [r["task_id"] for r in ok]
        rng.shuffle(ids)
        pick, devp = set(ids[:n]), set(ids[n:n + nd])
        by = {r["task_id"]: r for r in ok}
        # keep a stable, human-readable order inside the CSV (by numeric id)
        key = lambda t: int(t.split("_")[1])
        frozen += [by[t] for t in sorted(pick, key=key)]
        dev += [by[t] for t in sorted(devp, key=key)]
        report[name]["frozen"] = len(pick)
        report[name]["dev"] = len(devp)
    cols = ["task_id", "source", "description", "prompt_stub", "entry_point", "test_code", "test_setup",
            "n_tests", "difficulty", "reference_code"]
    for rows, key in ((frozen, "humaneval_csv"), (dev, "dev_csv")):
        df = pd.DataFrame(rows)[cols]
        df["prompt_tokens_est"] = (df["description"].str.len() / 4).round().astype(int)
        if key == "humaneval_csv":
            df[df.source == "humaneval"].to_csv(ROOT / cfg["data"]["humaneval_csv"], index=False)
            df[df.source == "mbpp"].to_csv(ROOT / cfg["data"]["mbpp_csv"], index=False)
        else:
            df.to_csv(ROOT / cfg["data"]["dev_csv"], index=False)
    report["freeze_seed"] = cfg["data"]["freeze_seed"]
    (ROOT / "data" / "data_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "excluded"} if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
