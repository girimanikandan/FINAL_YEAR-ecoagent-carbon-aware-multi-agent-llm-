import streamlit as st
from common import FIG, PROC, page, read_csv

page("Results", "📊")
files = [("Main results", "main_results.csv"), ("With 95% CI", "main_results_with_ci.csv"), ("Ablation", "ablation_results.csv"),
         ("Grid-carbon sensitivity", "carbon_sensitivity.csv"), ("Budget sensitivity", "budget_sensitivity.csv"),
         ("Tier distribution", "tier_distribution.csv"), ("Retry analysis", "retry_analysis.csv"),
         ("Comm-overhead sensitivity", "comm_overhead_sensitivity.csv"), ("Statistical tests", "statistical_tests.csv")]
if not any((PROC / f).exists() for _, f in files):
    st.info("Nothing analysed yet. Run **Run Control → Analyse**.")
    st.stop()
tabs = st.tabs([n for n, _ in files] + ["Figures", "Provenance"])
for tab, (name, f) in zip(tabs, files):
    with tab:
        df = read_csv(f)
        if df is None:
            st.caption(f"{f} not generated yet (needs the matching experiment).")
        else:
            st.dataframe(df.round(5), hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), file_name=f)
with tabs[-2]:
    figs = sorted(FIG.glob("*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for c, p in zip(cols, figs[i:i + 2]):
            c.image(str(p), caption=p.name)
    if not figs:
        st.caption("No figures yet.")
with tabs[-1]:
    p = PROC / "provenance.json"
    st.json(p.read_text() if p.exists() else "{}")
