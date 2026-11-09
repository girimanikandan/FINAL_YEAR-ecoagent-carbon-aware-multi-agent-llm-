import json
import streamlit as st
from common import RAW, list_runs, page, read_jsonl

page("Task Drilldown", "🔎")
runs = list_runs()
if runs.empty:
    st.info("No runs yet.")
    st.stop()
rid = st.selectbox("Run", runs.sort_values("started", ascending=False).run_id.tolist())
tasks, calls = read_jsonl(str(RAW / rid / "task_results.jsonl")), read_jsonl(str(RAW / rid / "llm_calls.jsonl"))
if tasks.empty:
    st.info("No tasks completed in this run yet.")
    st.stop()
only_fail = st.checkbox("Only tasks that did not fully pass")
view = tasks[tasks.task_success != 1] if only_fail else tasks
tid = st.selectbox("Task", view.task_id.tolist())
t = tasks[tasks.task_id == tid].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tests", f"{int(t.passed_tests)}/{int(t.total_tests)}")
c2.metric("Emissions", f"{t.total_emissions_gco2eq:.5f} g")
c3.metric("Carbon c(t)", f"{t.carbon_intensity_gco2_kwh:.0f} ({t.carbon_class})")
c4.metric("Topology", t.selected_topology or "-")
st.caption(f"alpha={t.alpha} · retries={int(t.retry_count)} · latency={t.latency_seconds:.1f}s · SLO ok={bool(t.slo_compliant)}")
tc = calls[calls.task_id == tid]
st.subheader("LLM calls")
st.dataframe(tc[["agent_role", "model_tier", "model_name", "prompt_tokens", "completion_tokens", "emissions_gco2eq", "latency_seconds",
                 "retry_number", "call_attempt", "alpha", "routing_reason", "status", "cache_hit"]], hide_index=True)
if isinstance(t.htde_trace, dict):
    with st.expander("HTDE decision trace"):
        st.json(t.htde_trace)
with st.expander("Final code", expanded=True):
    st.code(t.final_code or "(none)", language="python")
st.text(t.test_feedback or "")
