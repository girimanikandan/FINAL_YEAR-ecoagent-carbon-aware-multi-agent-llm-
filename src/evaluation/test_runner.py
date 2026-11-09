"""Isolated execution of generated code against the benchmark's hidden unit tests.

Counts individual test units (each top-level assert / assert-containing statement) so we can report
passed/total, not only all-or-nothing. A fresh temp dir per run, scrubbed environment (no API keys), timeouts =
failure, output capped. `subprocess` mode applies rlimits (memory/CPU) on POSIX and a best-effort network block;
`docker` mode adds `--network none`, memory and CPU limits (recommended on Windows; untested in this build env).
Generated code is NEVER evaluated by an LLM here."""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HARNESS = r'''
import json, sys, traceback
payload = json.load(open(sys.argv[1], encoding="utf-8"))
out_path = sys.argv[2]
try:
    import resource
    m = payload["mem_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (m, m))
    resource.setrlimit(resource.RLIMIT_CPU, (payload["cpu_seconds"], payload["cpu_seconds"]))
except Exception:
    pass
import socket
def _blocked(*a, **k):
    raise RuntimeError("network disabled in sandbox")
socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
res = {"passed": 0, "total": len(payload["units"]), "failures": [], "load_error": None}
ns = {"__name__": "solution"}
def finish():
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f)
try:
    for s in payload["module_setup"]:
        exec(s, ns)
    exec(payload["solution"], ns)
    if payload["entry_point"]:
        ns["candidate"] = ns[payload["entry_point"]]
    for s in payload["check_setup"]:
        exec(s, ns)
except BaseException as e:
    res["load_error"] = (type(e).__name__ + ": " + str(e))[:300]
    finish(); sys.exit(0)
for u in payload["units"]:
    try:
        exec(u, ns)
        res["passed"] += 1
    except BaseException as e:
        if len(res["failures"]) < 5:
            res["failures"].append((u[:160] + " -> " + type(e).__name__ + ": " + str(e))[:300])
finish()
'''


@dataclass
class TestResult:
    __test__ = False                 # not a pytest class
    passed: int = 0
    total: int = 0
    timed_out: bool = False
    exit_code: int | None = 0
    feedback: str = ""
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    error: str = ""
    dropped: int = 0          # top-level demo/test statements removed by sanitize()

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def success(self) -> bool:
        return self.total > 0 and self.passed == self.total


def _has_assert(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Assert) for n in ast.walk(node))


def build_plan(task: dict) -> dict:
    """Split a task's tests into setup + countable units. Pure function (unit-tested)."""
    src = task["source"]
    entry = task.get("entry_point") or ""
    if src == "humaneval":
        prompt_tree = ast.parse(task["prompt_stub"])
        prefix = [ast.unparse(n) for n in prompt_tree.body
                  if not (isinstance(n, ast.FunctionDef) and n.name == entry)]
        tree = ast.parse(task["test_code"])
        module_setup, check_setup, units = [], [], []
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name == "check":
                for st in n.body:
                    (units if _has_assert(st) else check_setup).append(ast.unparse(st))
            else:
                module_setup.append(ast.unparse(n))
        return {"prefix": "\n\n".join(prefix), "module_setup": module_setup, "check_setup": check_setup,
                "units": units, "entry_point": entry, "bind_candidate": True}
    if src == "mbpp":
        units = json.loads(task["test_code"])
        setup = [task["test_setup"]] if isinstance(task.get("test_setup"), str) and task["test_setup"].strip() else []
        return {"prefix": "", "module_setup": setup, "check_setup": [], "units": units,
                "entry_point": "", "bind_candidate": False}
    if src == "custom":
        units = json.loads(task["test_code"]) if task.get("test_code") else []
        return {"prefix": "", "module_setup": [], "check_setup": [], "units": units, "entry_point": "", "bind_candidate": False}
    raise ValueError(f"unknown task source {src}")


def _env() -> dict:
    keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _calls_local(node: ast.AST, local_names: set) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in local_names for n in ast.walk(node))


def sanitize(code: str) -> tuple[str, int]:
    """Drop top-level statements that are demo/test code the model appended (asserts, print calls, bare calls to its own
    functions, `result = own_function(...)`, `if __name__ == "__main__"` blocks). Without this, a model's own wrong self-test
    crashes the file at load time and correct solutions score 0. Definitions, imports, constants and calls like
    `sys.setrecursionlimit(...)` are kept. Applied identically to every system. Returns (clean_code, n_dropped)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0                       # let the harness report the syntax error as a load failure
    local = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    kept, dropped = [], 0
    for n in tree.body:
        drop = False
        if isinstance(n, ast.Assert):
            drop = True
        elif isinstance(n, ast.Expr):
            v = n.value
            is_attr_call = isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and not _calls_local(v, local)
            drop = not is_attr_call           # print(...), bare strings/names, bare calls of own functions
        elif isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__":
            drop = True
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and _calls_local(n.value, local):
            drop = True                       # result = merge([1], [2])  (demo invocation)
        if drop:
            dropped += 1
        else:
            kept.append(n)
    return (ast.unparse(ast.Module(body=kept, type_ignores=[])) if kept else ""), dropped


def run_tests(task: dict, code: str, sandbox_cfg: dict) -> TestResult:
    _, n = sanitize(code or "")
    r = _run_tests(task, code, sandbox_cfg)
    r.dropped = n
    return r


def _run_tests(task: dict, code: str, sandbox_cfg: dict) -> TestResult:
    plan = build_plan(task)
    total = len(plan["units"])
    code, n_dropped = sanitize(code or "")
    if not (code or "").strip():
        return TestResult(0, total, feedback="No code was extracted from the model response.", exit_code=None)
    solution = (plan["prefix"] + "\n\n" if plan["prefix"] else "") + code
    payload = {"solution": solution, "module_setup": plan["module_setup"], "check_setup": plan["check_setup"],
               "units": plan["units"], "entry_point": plan["entry_point"] if plan["bind_candidate"] else "",
               "mem_bytes": int(sandbox_cfg["memory_mb"]) * 1024 * 1024, "cpu_seconds": int(sandbox_cfg["cpu_seconds"])}
    tmp = Path(tempfile.mkdtemp(prefix="ecoagent_run_"))
    try:
        (tmp / "harness.py").write_text(HARNESS, encoding="utf-8")
        (tmp / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        t0 = time.perf_counter()
        if sandbox_cfg["mode"] == "docker":
            cmd = ["docker", "run", "--rm", "--network", "none", "--memory", f"{sandbox_cfg['memory_mb']}m", "--cpus", "1",
                   "-v", f"{tmp}:/work", sandbox_cfg["docker_image"], "python", "-I", "/work/harness.py",
                   "/work/payload.json", "/work/result.json"]
        else:
            cmd = [sys.executable, "-I", "-B", "harness.py", "payload.json", "result.json"]
        timed_out, exit_code, so, se = False, 0, "", ""
        try:
            cp = subprocess.run(cmd, cwd=tmp, env=_env(), capture_output=True, timeout=sandbox_cfg["timeout_seconds"])
            exit_code = cp.returncode
            cap = int(sandbox_cfg["max_output_bytes"])
            so, se = cp.stdout[:cap].decode("utf-8", "replace"), cp.stderr[:cap].decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            timed_out, exit_code = True, None
        secs = time.perf_counter() - t0
        res_file = tmp / "result.json"
        if timed_out or not res_file.exists():
            fb = "Timed out (treated as test failure)." if timed_out else f"Crashed before reporting results. stderr: {se[-300:]}"
            return TestResult(0, total, timed_out, exit_code, fb, so, se, secs)
        r = json.loads(res_file.read_text(encoding="utf-8"))
        if r.get("load_error"):
            return TestResult(0, total, False, exit_code, f"Solution failed to load: {r['load_error']}", so, se, secs)
        fb = f"Passed {r['passed']}/{r['total']} tests."
        if r["failures"]:
            fb += " First failures:\n" + "\n".join(r["failures"])
        return TestResult(r["passed"], r["total"], False, exit_code, fb[:1500], so, se, secs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)