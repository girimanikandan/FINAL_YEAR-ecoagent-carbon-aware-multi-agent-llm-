import streamlit as st
from common import JOBS, PROC, ROOT, job_running, launch, list_runs, page, stop_job

page("Run Control", "🚀")
st.write("Launches the real CLI (`python -m src.main ...`) as a background process and tails its log. Raw logs are append-only; re-launching resumes unfinished runs.")

provider = st.selectbox("LLM provider", ["ollama", "mock"], help="`mock` is an offline test double: results are flagged PROVISIONAL and not publishable.")
profile = st.selectbox("Carbon profile (pilot / single runs)", ["medium", "low", "high"])

tab1, tab2, tab3, tab4 = st.tabs(["1 · Pilot (calibrate HTDE)", "2 · Experiments", "3 · Analyse", "Jobs"])
with tab1:
    st.write("Runs the 20 disjoint dev tasks at depths 1/2/3, fits the HTDE tables and selects θ. Required before any run that uses HTDE.")
    if (PROC / "htde_calibration.json").exists():
        st.success("A calibration file exists (re-running overwrites pilot data only).")
    if st.button("Run pilot", type="primary"):
        st.session_state["log"] = str(launch(["pilot", "--provider", provider, "--profile", profile], "pilot"))
with tab2:
    name = st.selectbox("Experiment", ["main", "budget", "grid", "ablation", "all"])
    scale = st.selectbox("Scale", ["smoke", "reduced", "full"], index=1,
                         help="smoke = 3+3 tasks · reduced = 50+50 tasks × 3 seeds (PDF minimum) · full = 100+100 tasks × 5 seeds")
    st.caption("`budget` needs the B1 runs of `main` at the same scale first.")
    if st.button("Launch experiment", type="primary"):
        st.session_state["log"] = str(launch(["experiment", "--name", name, "--scale", scale, "--provider", provider], f"exp_{name}_{scale}"))
with tab3:
    inc = st.checkbox("Include mock runs (testing only)")
    if st.button("Aggregate raw logs → CSVs, stats, figures", type="primary"):
        st.session_state["log"] = str(launch(["analyze"] + (["--include-mock"] if inc else []), "analyze"))
with tab4:
    logs = sorted(JOBS.glob("*.log"), reverse=True) if JOBS.exists() else []
    if not logs:
        st.info("No jobs launched yet.")
    else:
        sel = st.selectbox("Job log", logs, format_func=lambda p: p.name, index=0)
        running = job_running(sel)
        st.write("Status: " + ("🟢 running" if running else "⚪ finished / stopped"))
        c1, c2 = st.columns(2)
        if c1.button("Refresh"):
            list_runs.clear()
            st.rerun()
        if running and c2.button("Stop job"):
            stop_job(sel)
            st.rerun()
        st.code("\n".join(sel.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]) or "(empty)")
if "log" in st.session_state:
    st.info(f"Started: {st.session_state['log']} — see the *Jobs* tab.")
