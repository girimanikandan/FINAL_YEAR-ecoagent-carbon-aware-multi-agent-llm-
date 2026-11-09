import pandas as pd
import streamlit as st
from common import CFG, ROOT, page
from src.carbon.intensity import CarbonContext

page("Carbon Traces", "🌍")
prof = st.selectbox("Profile", ["low", "medium", "high"], index=1)
try:
    ctx = CarbonContext(CFG, prof)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
st.write(f"Source: **{ctx.trace.source}** · {'real data' if ctx.trace.is_real else 'SYNTHETIC placeholder — replace via scripts/fetch_carbon_data.py before reporting results'}")
lo, hi = ctx.thresholds
c1, c2, c3 = st.columns(3)
c1.metric("Low tercile ≤", f"{lo:.1f} gCO2/kWh"); c2.metric("High tercile >", f"{hi:.1f} gCO2/kWh"); c3.metric("Hours", len(ctx.trace.values))
st.caption(f"Threshold scope: {CFG['caar']['threshold_scope']} (configs/ecoagent.yaml). These exact values must be reported in the paper.")
st.line_chart(ctx.trace.df.set_index("datetime")["carbon_intensity_gco2_kwh"], height=280)
sched = ROOT / CFG["data"]["schedule_csv"]
if sched.exists():
    df = pd.read_csv(sched)
    st.subheader("Task → hour schedule (data/task_carbon_schedule.csv)")
    st.dataframe(df[df.profile == prof].head(200), hide_index=True)
