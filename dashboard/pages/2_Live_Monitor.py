import streamlit as st
from common import RAW, list_runs, page, read_jsonl

page("Live Monitor", "📡")
runs = list_runs()
if runs.empty:
    st.info("No runs yet. Launch one from **Run Control**.")
    st.stop()
runs = runs.sort_values("started", ascending=False)
rid = st.selectbox("Run", runs.run_id.tolist())
meta = runs[runs.run_id == rid].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Progress", f"{int(meta.tasks_done)} / {int(meta.n_tasks)} tasks")
c2.metric("System · seed", f"{meta.system} · {meta.seed}")
c3.metric("Provider · carbon", f"{meta.provider} · {'real' if 'synthetic' not in str(meta.carbon_source) else 'SYNTHETIC'}")
st.progress(min(1.0, meta.tasks_done / max(1, meta.n_tasks)))
tasks, calls = read_jsonl(str(RAW / rid / "task_results.jsonl")), read_jsonl(str(RAW / rid / "llm_calls.jsonl"))
if st.button("Refresh"):
    list_runs.clear(); read_jsonl.clear(); st.rerun()
if tasks.empty:
    st.info("No completed tasks yet.")
    st.stop()
tasks["cumulative_emissions_gco2eq"] = tasks.total_emissions_gco2eq.cumsum()
a, b = st.columns(2)
a.line_chart(tasks.set_index("position")["cumulative_emissions_gco2eq"], height=240)
a.caption("Cumulative emissions (gCO2eq)")
b.bar_chart(calls[calls.total_tokens > 0].groupby("model_tier").size().reindex(["S", "M", "L"]).fillna(0), height=240)
b.caption("LLM calls by tier so far")
st.dataframe(tasks[["position", "task_id", "final_test_pass_rate", "total_emissions_gco2eq", "agent_calls", "retry_count",
                    "selected_topology", "carbon_class", "alpha", "latency_seconds"]].tail(15), hide_index=True)
