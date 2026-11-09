import yaml
import streamlit as st
from common import CFG, ROOT, C, page

page("Configuration (read-only)", "⚙️")
st.write(f"Config hash: `{C.config_hash(CFG)}` — recorded in every run's meta.json. Edit the YAML files, not the code.")
t = st.tabs(["Models", "EcoAgent", "Baselines & ablation", "Experiment matrix", "Prompts"])
for tab, f in zip(t[:4], ["models.yaml", "ecoagent.yaml", "baselines.yaml", "experiment.yaml"]):
    with tab:
        st.code((ROOT / "configs" / f).read_text(encoding="utf-8"), language="yaml")
with t[4]:
    ver = CFG["experiment"]["prompt_version"]
    for role in ("planner", "coder", "reviewer", "repair"):
        st.markdown(f"**{role}** (prompts/{ver}/{role}.txt)")
        st.code((ROOT / "configs" / "prompts" / ver / f"{role}.txt").read_text(encoding="utf-8"))
