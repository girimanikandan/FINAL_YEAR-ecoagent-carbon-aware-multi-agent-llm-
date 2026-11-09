"""Shared helpers for the Streamlit app. The UI only READS results/ and LAUNCHES the real CLI - it never computes a result itself."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

CFG = C.load_all()
RAW, PROC, FIG = C.path(CFG, "raw"), C.path(CFG, "processed"), C.path(CFG, "figures")
JOBS = ROOT / "results" / "jobs"


def page(title: str, icon: str = "🌱"):
    st.set_page_config(page_title=f"EcoAgent-CG · {title}", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")
    prov = PROC / "provenance.json"
    if prov.exists():
        p = json.loads(prov.read_text())
        if p.get("provisional"):
            st.warning("PROVISIONAL results: " + " | ".join(p.get("warnings", [])))


@st.cache_data(ttl=3)
def list_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(RAW.glob("*/")):
        mp = d / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            tf = d / "task_results.jsonl"
            m["tasks_done"] = sum(1 for _ in open(tf, encoding="utf-8")) if tf.exists() else 0
            rows.append({k: m.get(k) for k in ("run_id", "system", "seed", "experiment", "label", "scale", "profile", "provider",
                                               "phase", "is_mock", "carbon_source", "n_tasks", "tasks_done", "started")})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3)
def read_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])


def read_csv(name: str):
    p = PROC / name
    return pd.read_csv(p) if p.exists() else None


def launch(args: list[str], label: str) -> Path:
    JOBS.mkdir(parents=True, exist_ok=True)
    log = JOBS / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.log"
    f = open(log, "w", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
    proc = subprocess.Popen([sys.executable, "-m", "src.main", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    (log.with_suffix(".pid")).write_text(str(proc.pid))
    return log


def job_running(log: Path) -> bool:
    pid_file = log.with_suffix(".pid")
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text())
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
            return str(pid) in out
        os.kill(pid, 0)
        # zombie check (finished child not yet reaped)
        stat = Path(f"/proc/{pid}/stat")
        return not (stat.exists() and stat.read_text().split(")")[-1].split()[0] == "Z")
    except (OSError, ProcessLookupError):
        return False


def stop_job(log: Path) -> None:
    pid_file = log.with_suffix(".pid")
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 15)
        except OSError:
            pass
