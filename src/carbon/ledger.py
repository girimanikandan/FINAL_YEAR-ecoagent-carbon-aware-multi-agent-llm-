"""Logging. One JSONL record per LLM call (llm_calls.jsonl) and one per task (task_results.jsonl), append-only,
plus a SQLite index of runs. Every number in the paper must be derivable from these files by the analysis scripts."""
import json
import sqlite3
import time
from pathlib import Path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SessionLedger:
    """Running carbon ledger for one session (budget + spend)."""

    def __init__(self, budget_g: float):
        self.budget_g = float(budget_g)
        self.spent_g = 0.0
        self.n_calls = 0

    def add(self, grams: float) -> None:
        self.spent_g += grams
        self.n_calls += 1

    @property
    def remaining_g(self) -> float:
        return max(0.0, self.budget_g - self.spent_g)

    @property
    def remaining_fraction(self) -> float:
        return self.remaining_g / self.budget_g if self.budget_g > 0 else 0.0


class RunLogger:
    def __init__(self, raw_dir: Path, run_id: str, meta: dict, resume: bool = False):
        self.dir = Path(raw_dir) / run_id
        if self.dir.exists() and not resume:
            raise FileExistsError(f"{self.dir} exists. Raw logs are immutable; use --resume or a new --tag.")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.dir / "llm_calls.jsonl"
        self.tasks_path = self.dir / "task_results.jsonl"
        self.meta = dict(meta, run_id=run_id)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        self._index(raw_dir)

    def _index(self, raw_dir: Path) -> None:
        db = Path(raw_dir).parent / "index.sqlite"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, system TEXT, seed INTEGER, "
                    "experiment TEXT, profile TEXT, provider TEXT, phase TEXT, carbon_source TEXT, created TEXT)")
        m = self.meta
        con.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                    (m["run_id"], m.get("system"), m.get("seed"), m.get("experiment"), m.get("profile"),
                     m.get("provider"), m.get("phase"), m.get("carbon_source"), utc_now()))
        con.commit()
        con.close()

    @staticmethod
    def _append(path: Path, rec: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def log_call(self, rec: dict) -> None:
        self._append(self.calls_path, rec)

    def log_task(self, rec: dict) -> None:
        self._append(self.tasks_path, rec)

    def completed_task_ids(self) -> set:
        if not self.tasks_path.exists():
            return set()
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            return {json.loads(line)["task_id"] for line in f if line.strip()}
