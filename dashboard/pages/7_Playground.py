import subprocess
import sys
import streamlit as st
from common import PROC, ROOT, page

page("Playground — ask EcoAgent anything", "💬")
st.write("Runs the real EcoAgent pipeline (HTDE → CAAR → CBC) on a free-form coding task. Without asserts the answer is **unverified** "
         "(no hidden tests exist for custom tasks), so no repair loop runs.")
prompt = st.text_area("Task", placeholder="Write a function merge(a, b) that merges two sorted lists.")
asserts = st.text_area("Optional verification asserts (one per line)", placeholder="assert merge([1,3],[2]) == [1,2,3]")
c1, c2 = st.columns(2)
hour = c1.slider("Replayed carbon hour", 1, 720, 13)
profile = c2.selectbox("Carbon profile", ["medium", "low", "high"])
if not (PROC / "htde_calibration.json").exists():
    st.caption("No HTDE calibration yet → fixed depth 2 will be used (run the pilot for adaptive depth).")
if st.button("Run", type="primary", disabled=not prompt.strip()):
    args = [sys.executable, "-m", "src.main", "ask", prompt, "--provider", "ollama", "--profile", profile, "--hour", str(hour)]
    for line in asserts.splitlines():
        if line.strip():
            args += ["--assert", line.strip()]
    with st.spinner("Running…"):
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    st.code(out.stdout or out.stderr)
