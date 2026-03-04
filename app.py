"""
app.py — ExoTransit: Exoplanet Transit Detection Pipeline

Streamlit wizard app. Takes a star name, runs the full pipeline
(fetch → BLS → MCMC → physics), and tells the story at each step.

Run with:
    streamlit run app.py
"""

import streamlit as st

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
    color: #64748b;
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
    color: #94a3b8;
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

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.35) !important;
}

/* Progress / spinner */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* Divider */
hr { border-color: rgba(148, 163, 184, 0.1) !important; }

/* Plotly charts — remove white bg flash */
.js-plotly-plot { background: transparent !important; }

/* Expander */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #64748b !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state bootstrap ──────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        step=0,            # 0=search, 1=data, 2=detection, 3=fitting, 4=results
        target=None,
        lc=None,
        stellar=None,
        ld=None,
        all_bls=None,
        all_mcmc=None,
        all_physics=None,
        error=None,
        prefill_target=""
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Router ───────────────────────────────────────────────────────────────────
from exotransit.app.steps import (
    step0_search,
    step1_data,
    step2_detection,
    step3_fitting,
    step4_results,
)

STEPS = [
    step0_search.render,
    step1_data.render,
    step2_detection.render,
    step3_fitting.render,
    step4_results.render,
]

# ── Progress indicator ───────────────────────────────────────────────────────
STEP_LABELS = ["Search", "Data", "Detection", "Fitting", "Results"]

def _render_progress():
    current = st.session_state.step
    cols = st.columns(len(STEP_LABELS))
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS)):
        with col:
            if i < current:
                color = "#38bdf8"
                icon = "✓"
            elif i == current:
                color = "#f1f5f9"
                icon = str(i + 1)
            else:
                color = "#334155"
                icon = str(i + 1)

            st.markdown(f"""
            <div style="text-align:center; padding: 0.5rem 0;">
                <div style="
                    width: 28px; height: 28px;
                    border-radius: 50%;
                    background: {'rgba(56,189,248,0.15)' if i == current else 'transparent'};
                    border: 1.5px solid {color};
                    color: {color};
                    font-family: 'DM Mono', monospace;
                    font-size: 0.7rem;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 0.3rem;
                ">{icon}</div>
                <div style="
                    font-family: 'DM Mono', monospace;
                    font-size: 0.65rem;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    color: {color};
                ">{label}</div>
            </div>
            """, unsafe_allow_html=True)

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
        color: #64748b;
        margin-top: 0.5rem;
        font-style: italic;
        font-family: 'Instrument Serif', serif;
    ">From raw photons to planet characterization</p>
</div>
""", unsafe_allow_html=True)

# Only show progress bar after search
if st.session_state.step > 0:
    _render_progress()
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# ── Error display ────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("← Start over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Render current step ──────────────────────────────────────────────────────
else:
    step_fn = STEPS[min(st.session_state.step, len(STEPS) - 1)]
    step_fn()
