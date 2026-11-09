# EcoAgent-CG v1 — carbon-aware multi-agent code generation

Restructured from the Phase 2–4 prototype to satisfy the *Prototype goal* spec. One workload (code generation), objective
evaluation (hidden unit tests), five components (Session Manager, HTDE, CAAR, Model Tier Pool, CBC), baselines B1–B4,
ablation, sensitivity studies, statistics. The previous code is preserved untouched in `legacy/phase2_4/`.

> **Read this first.** The code is tested offline (`python -m pytest tests -q` → 31 passing, mock LLM).
> Partial real Ollama runs exist in `results/raw/`, but **there are no complete experiment results yet**.
> Before treating results as publishable:
> (1) fetch real carbon data (the shipped CSVs are flagged *synthetic placeholders*), (2) run with real Ollama models,
> and (3) report energy/token as an *assumed* estimate unless it has been calibrated with CodeCarbon.
> Analysis marks outputs **PROVISIONAL** when synthetic carbon data or the mock LLM is used. It warns, but does not
> automatically mark outputs provisional, when energy/token is assumed or run coverage is incomplete. Check both before
> using any result. See `CHECKLIST.md` for requirement-by-requirement status.

## Quick start (Windows PowerShell, from this folder)
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. models (laptop profile in configs/models.yaml; edit the names there if your hardware differs, BEFORE any experiment)
ollama pull qwen2.5-coder:1.5b-instruct ; ollama pull qwen2.5-coder:3b-instruct ; ollama pull qwen2.5-coder:7b-instruct

# 2. real carbon data (UK Carbon Intensity API; 30 days hourly; picks low/medium/high regions by mean intensity)
python scripts/fetch_carbon_data.py --start 2026-09-01
#    (need a genuinely fossil-heavy grid? import one:  --import high=my_trace.csv:SOURCE_LABEL)

# 3. check the setup, then the offline test-suite
python -m src.main info
python -m pytest tests -q

# 4. pilot (20 dev tasks x depths 1/2/3): verifies prompts/extraction/tests/logging and calibrates HTDE theta
python -m src.main pilot --provider ollama
python scripts/pilot_report.py

# 5. experiments (start small!)     scale: smoke (3+3 tasks) | reduced (50+50 x 3 seeds) | full (100+100 x 5 seeds)
python -m src.main experiment --name main     --scale reduced
python -m src.main experiment --name budget   --scale reduced     # needs B1 from `main` first
python -m src.main experiment --name grid     --scale reduced
python -m src.main experiment --name ablation --scale reduced

# 6. every number/table/figure is generated from the raw logs
python -m src.main analyze
streamlit run dashboard/app.py
```
Optional measured energy: `pip install codecarbon`, then `python scripts/calibrate_energy.py` (writes
`configs/energy_measured.yaml`, which overrides the assumed values automatically).
Dry-run everything offline: add `--provider mock` (flagged; never publishable; use `analyze --include-mock`).

**Resuming runs:** `experiment` commands resume existing run IDs and skip completed tasks; `run` resumes only when
passed `--resume`. The pilot is different: it deletes and recreates its three calibration run directories, so an
interrupted pilot restarts from the beginning. Raw experiment logs are append-only. Check each run's `task_results.jsonl`
against its expected task count before analyzing or reporting it; analysis does not verify that planned runs are complete.
Free-form questions: `python -m src.main ask "Write merge(a, b) for two sorted lists" --assert "assert merge([1],[2])==[1,2]"`
(without asserts the answer is reported as unverified — custom tasks have no hidden tests).

## Layout (PDF structure + a few additions, marked *)
```
configs/      models.yaml  ecoagent.yaml  baselines.yaml  experiment.yaml  prompts/v1/*.txt   (all settings + prompts live here)
data/         humaneval_tasks.csv  mbpp_tasks.csv (frozen 100+100)   dev_tasks.csv* (20, disjoint, for the pilot)
              carbon_intensity_{low,medium,high}.csv   task_carbon_schedule.csv   data_validation.json*   raw_benchmarks/*
src/          main.py  session_manager.py  htde.py  caar.py  cbc.py  model_pool.py  config.py*  tasks.py*
  agents/     planner.py  coder.py  reviewer.py  repair.py
  carbon/     intensity.py  energy.py  ledger.py
  evaluation/ test_runner.py   (sandboxed, counts every assert)
  analysis/   aggregate_results.py (also holds the metrics)  statistical_tests.py  plots.py
results/      raw/ (immutable JSONL per run)   processed/ (CSVs, provenance.json, htde_calibration.json)   figures/
tests/        test_pipeline.py        scripts/*   prepare_data / fetch_carbon_data / make_placeholder_carbon / calibrate_energy
dashboard/*   Streamlit app (reads results/, launches the real CLI)       legacy/phase2_4/*   old prototype, untouched
```
`evaluation/metrics.py` from the spec is realised as the aggregation functions in `analysis/aggregate_results.py`
(all 13 metrics are derived from raw logs there). `session_manager.py` also contains the per-task orchestration flow.

## How the components work (as implemented)
* **Flow** — Planner → Coder → run hidden tests → *if failed and repair budget remains* Reviewer → Coder-Repair → re-run tests
  (max 1 repair round, max 6 calls/task; B2 max 12). D1 = Coder; D2 = Planner+Coder; D3 = Planner+Coder+(Reviewer+Repair on failure).
* **HTDE** — start at D1; expand D(k-1)→D(k) iff `ΔQ/ΔE > θ / max(α, ε)`; Q̂/Ê come from the pilot calibration per complexity bin
  (prompt-token terciles of the frozen set); ΔE is in gCO2eq at the *current* intensity. θ is *selected by the pilot*
  (min emissions s.t. quality ≥ 0.9 × always-D3), never hand-set.
* **CAAR** — per call: role→acceptable tiers (Planner S/M, Coder M/L, Reviewer S/M, Repair M/L; widened by one tier after a failed test);
  then budget < 25% → lowest; clean grid (bottom tercile) → highest acceptable; otherwise lowest-emission acceptable.
  Thresholds = trace percentiles 33/67 (printed by `info`, stored in each run's `meta.json`).
* **CBC** — rule-based α exactly as specified (1.0 / 0.7 / 0.4 / 0.2 by remaining budget; −0.2 in the high-carbon tercile, floored at 0.05). Not RL.
* **Accounting** — `E = e_i · (prompt+completion tokens) · c(t) · 1e-3` per call; retries and failed-but-processed calls are separate logged calls;
  token counts are provider-reported (Ollama usage); `E_comm = 0` in main runs, 1/5/10% sensitivity is recomputed from the raw logs.
* **Reproducibility** — temperature 0, frozen task ids, deterministic seeded order, task *i* → trace hour *i* (cyclic; saved to
  `task_carbon_schedule.csv`), config hash + model registry recorded per run, LLM response cache (replays original latency/tokens; flagged `cache_hit`).

## Interpretation decisions — please confirm with your guide (each is configurable)
| # | Ambiguity in the spec | What the code does |
|---|---|---|
| 1 | Budget scope: "initialise ledger + CBC for each task" vs "session-level" | Per task (`session.session_tasks: 1`), matching the spec's procedure and its example record (5.0 − 1.84 = 3.16). Set >1 for multi-task sessions. |
| 2 | α ∈ (0,1] but 0.2 − 0.2 = 0 | Floored at `cbc.alpha_floor = 0.05`. |
| 3 | α only feeds HTDE/CAAR, so "CBC only" would equal B1 | Added direct actuation: tier ceiling (α ≤ 0.4 → ≤M, α ≤ 0.2 → S, never below the role's lowest acceptable tier) and no repair once the budget is exhausted. |
| 4 | "Allow escalation after failure" | Widens the acceptable set by one tier; the carbon rule still picks within it. |
| 5 | Carbon thresholds: per trace or pooled? | `caar.threshold_scope: per_profile` (hours classified relative to their own grid). `pooled` makes low/high grids differ in absolute terms. |
| 6 | HTDE θ selection criterion not specified | Min mean emissions s.t. quality ≥ 0.9 × always-D3 on the 20 dev tasks (θ grid + every observed ΔQ/ΔE boundary). |
| 7 | B2 cascade | Each rung (S→M→L) runs the whole fixed topology (incl. the single allowed repair) and stops when pass-rate ≥ τ_q = 0.8; every call counted. |
| 8 | Meaning of "without CAAR / HTDE / CBC" | Each removed component reverts to its B1 form (always-L, fixed D3, no budget logic); "None" = B1 runs. |
| 9 | CAAR inputs latency budget / quality threshold / task complexity have no rule | Not used in v1 (an HTDE latency guard exists, off by default: `htde.use_latency_guard`). |
| 10 | Repair uses benchmark tests as feedback (the spec's flow) | Implemented as specified; it is an execution-feedback setting and final success is measured on the same tests — state this in the paper. MBPP prompts also show the first test (standard practice). |

## Known limitations / caveats (put the relevant ones in the paper)
* Energy/token is **assumed** until CodeCarbon calibration; CodeCarbon measures the whole machine (idle-subtracted); CPU-only laptops get TDP-based estimates.
* Temperature 0 ⇒ seeds only change task order / trace-hour alignment, so seeds are not independent samples of model randomness. Statistics are computed over tasks (seed-averaged).
* `subprocess` sandbox: memory/CPU rlimits and the socket block are POSIX / best-effort. On Windows use `sandbox.mode: docker` (`--network none`, memory/CPU limits) for the isolation the spec requires.
* UK regions are not coal-heavy; import a real fossil-heavy trace for the "high" profile if the paper needs one.
* Small local models (1.5B–7B) pass far fewer tasks than frontier models; quality retention is relative to B1 on the *same* models.

## Compute
Main comparison = 5 systems × tasks × seeds task-runs, each 1–4 LLM calls (B2 up to 8); ablation adds 7 × 3 seeds; sensitivity adds 3 + 6 runs.
Measure seconds/task in the pilot before choosing the scale; the response cache removes duplicate prompts across seeds/systems.
The spec allows 100 tasks × 3 seeds as the minimum for the main result.

## Git
```powershell
git tag v0-phase4        # in the OLD repo first, to snapshot the old structure
git add . ; git commit -m "Restructure to EcoAgent-CG v1 spec" ; git push
```
