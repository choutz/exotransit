"""
app.py — ExoTransit: Exoplanet Transit Detection Pipeline

Streamlit wizard app. Takes a star name, runs the full pipeline
(fetch → BLS → MCMC → physics), and tells the story at each step.

Run with:
    streamlit run app.py
"""

import streamlit as st
from config import CONF

st.set_page_config(
    page_title="ExoTransit",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    background-color: #080e1a;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}

/* Star field background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 20% 50%, rgba(29, 78, 216, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.04) 0%, transparent 50%),
        #080e1a;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* Typography */
h1, h2, h3 {
    font-family: 'Instrument Serif', serif;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: #f1f5f9 !important;
}

/* Step headers */
.step-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.25rem;
}

/* Planet card */
.planet-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

/* Metric value */
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #f1f5f9;
    line-height: 1.1;
}

.metric-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.1rem;
}

.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.4rem;
}

/* Status badge */
.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-right: 0.4rem;
}
.badge-green  { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-yellow { background: rgba(234,179,8,0.15);  color: #facc15; border: 1px solid rgba(234,179,8,0.3); }
.badge-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-blue   { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }

/* Explanation box */
.explain-box {
    background: rgba(15, 23, 42, 0.8);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.explain-box strong {
    color: #e2e8f0;
    font-weight: 500;
}

/* Input styling */
.stTextInput > div > div > input {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */

/* Primary buttons — Analyze + all nav buttons: blue gradient */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%) !important;
    border: 1px solid rgba(37, 99, 235, 0.5) !important;
    border-radius: 8px !important;
}

[data-testid="stBaseButton-primary"] p {
    color: #ffffff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    white-space: nowrap !important;
}

[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(180deg, #60a5fa 0%, #3b82f6 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.6) !important;
    filter: none !important;
    opacity: 1 !important;
}

/* Secondary buttons — example chips: white/light, unchanged */
[data-testid="stBaseButton-secondary"],
[data-testid="stPopoverButton"] {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
}

[data-testid="stBaseButton-secondary"] p,
[data-testid="stPopoverButton"] p {
    color: #1e293b !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
}

/* Freeze hover on secondary — no visual change */
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stPopoverButton"]:hover {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    filter: none !important;
    opacity: 1 !important;
}

/* Progress / spinner */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* Divider */
hr { border-color: rgba(148, 163, 184, 0.1) !important; }

/* Plotly charts — remove white bg flash */
.js-plotly-plot { background: transparent !important; }

/* ── Expanders ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    background: transparent !important;
}

[data-testid="stExpander"] details {
    background: rgba(15, 23, 42, 0.5) !important;
}

[data-testid="stExpander"] summary {
    background: rgba(15, 23, 42, 0.5) !important;
    padding: 0.6rem 1rem !important;
}

[data-testid="stExpander"] summary:hover {
    background: rgba(30, 41, 59, 0.6) !important;
    cursor: pointer;
}

/* Content area — this is what goes white */
[data-testid="stExpander"] details > div,
[data-testid="stExpander"] details > section {
    background: rgba(8, 14, 26, 0.8) !important;
}

/* Header text — font-family intentionally omitted to avoid clobbering the arrow icon font */
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #94a3b8 !important;
}

/* Arrow icon */
[data-testid="stExpander"] summary svg {
    color: #94a3b8 !important;
    fill: #94a3b8 !important;
}

/* Prose inside expanders */
[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] ul,
[data-testid="stExpander"] ol {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

[data-testid="stExpander"] strong,
[data-testid="stExpander"] b {
    color: #e2e8f0 !important;
}

[data-testid="stExpander"] h1,
[data-testid="stExpander"] h2,
[data-testid="stExpander"] h3 {
    color: #e2e8f0 !important;
}

/* KaTeX equations rendered via st.latex() inside expanders */
[data-testid="stExpander"] .katex,
[data-testid="stExpander"] .katex * {
    color: #e2e8f0 !important;
}

/* Old class name fallbacks */
.streamlit-expanderHeader {
    background: rgba(15, 23, 42, 0.5) !important;
    color: #94a3b8 !important;
}

.streamlit-expanderContent {
    background: rgba(8, 14, 26, 0.8) !important;
}

/* Spinner text */
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state bootstrap ──────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        conf=CONF,
        step=0,
        target=None,
        lc=None,
        stellar=None,
        ld=None,
        all_bls=None,
        all_bls_mask_data=None,
        all_mcmc=None,
        all_physics=None,
        error=None,
        prefill_target=""
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

from exotransit.app.steps import (
    step0_search,
    step1_data,
    step2_detection,
    step3_fitting,
    step4_results,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2.5rem 0 1.5rem 0;">
    <div style="
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    ">Exoplanet Transit Detection</div>
    <h1 style="
        font-family: 'Instrument Serif', serif;
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 400;
        color: #f1f5f9;
        margin: 0;
        letter-spacing: -0.03em;
    ">ExoTransit</h1>
    <p style="
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-style: italic;
        font-family: 'Instrument Serif', serif;
    ">From raw photons to planet characterization</p>
</div>
""", unsafe_allow_html=True)

# ── Step indicator ────────────────────────────────────────────────────────────
_STEP_LABELS = ["Search", "Raw Data", "Detection", "MCMC Fitting", "Results"]

def _render_step_indicator(current: int):
    parts = []
    for i, label in enumerate(_STEP_LABELS):
        if i == 0:
            continue  # search step has no indicator slot
        idx = i  # steps 1–4
        if idx < current:
            color = "#4ade80"   # done — green
            weight = "400"
        elif idx == current:
            color = "#38bdf8"   # active — blue
            weight = "500"
        else:
            color = "#475569"   # future — dim
            weight = "400"
        parts.append(
            f'<span style="color:{color}; font-weight:{weight};">'
            f'{idx}. {label}</span>'
        )
    html = ' <span style="color:#334155; margin:0 0.4rem;">·</span> '.join(parts)
    st.markdown(
        f'<div style="text-align:center; font-family:\'DM Mono\',monospace; '
        f'font-size:0.72rem; letter-spacing:0.08em; margin-bottom:1.5rem;">'
        f'{html}</div>',
        unsafe_allow_html=True,
    )

# ── Navigation buttons ────────────────────────────────────────────────────────
def _nav_buttons(prev_step: int | None, next_step: int | None, next_label: str = "Continue →"):
    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
    cols = st.columns([1, 2, 1])
    if prev_step is not None:
        with cols[0]:
            if st.button("← Back", key=f"back_{prev_step}", type="primary", use_container_width=True):
                st.session_state.step = prev_step
                st.rerun()
    if next_step is not None:
        with cols[2]:
            if st.button(next_label, key=f"next_{next_step}", type="primary", use_container_width=True):
                st.session_state.step = next_step
                st.rerun()

# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("← Start over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Step-by-step pipeline ─────────────────────────────────────────────────────
else:
    current = st.session_state.step

    if current == 0:
        step0_search.render()

    elif current == 1:
        _render_step_indicator(1)
        step1_data.render()
        if st.session_state.lc is not None and st.session_state.stellar is not None:
            _nav_buttons(prev_step=0, next_step=2, next_label="Step 2: Transit Detection →")

    elif current == 2:
        _render_step_indicator(2)
        step2_detection.render()
        if st.session_state.all_bls is not None:
            _nav_buttons(prev_step=1, next_step=3, next_label="Step 3: MCMC Fitting →")

    elif current == 3:
        _render_step_indicator(3)
        step3_fitting.render()
        if st.session_state.all_mcmc is not None:
            _nav_buttons(prev_step=2, next_step=4, next_label="Step 4: Results →")

    elif current == 4:
        _render_step_indicator(4)
        step4_results.render()
        st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
        col_back, col_mid, col_new = st.columns([1, 3, 1])
        with col_back:
            if st.button("← Back", key="back_step4", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col_new:
            if st.button("New star →", key="new_star", type="primary", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
