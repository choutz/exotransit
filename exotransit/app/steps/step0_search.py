"""
exotransit/app/steps/step0_search.py
"""
import streamlit as st

EXAMPLE_TARGETS = [
    ("Kepler-5",  "Single planet system and one of the first five planets discovered by the Kepler spacecraft"),
    ("Kepler-183", "A two planet system discovered in 2014 in the constellation Cygnus"),
    ("Kepler-18", "A three planet system where two outer planets are in a near-perfect 2:1 resonance"),
    ("Kepler-58", "A Sun-like star hosting four Neptune-sized planets"),
    ("Kepler-20",  "A five planet system where the planet sizes bizarrely alternate between large and small"),
]

_RESULT_KEYS = ["lc", "stellar", "ld", "all_bls", "all_bls_mask_data", "all_mcmc", "all_physics", "error"]

def _reset_results():
    for k in _RESULT_KEYS:
        st.session_state[k] = None

def render():
    st.markdown("""
    <style>
    .search-container {
        max-width: 560px;
        margin: 0 auto;
    }
    </style>
    <div class="search-container">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box" style="margin-bottom: 2rem;">
        <strong>What this does:</strong> Enter any star observed by Kepler.
        The pipeline fetches raw photometry from NASA's archive, searches for periodic
        dips caused by orbiting planets, fits a physical transit model to measure each
        planet's size and orbit, and propagates uncertainties using Bayesian inference.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="step-header">Target Star</div>', unsafe_allow_html=True)
    target = st.text_input(
        label="target",
        placeholder="e.g. Kepler-5",
        label_visibility="collapsed",
        value=st.session_state.get("prefill_target", ""),
    )

    # Example chips — render as HTML buttons that trigger rerun via st.button
    st.markdown("""
    <div style="
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 0.75rem 0 0.5rem 0;
    ">Try an example</div>
    """, unsafe_allow_html=True)

    # One row of buttons
    cols = st.columns(5)
    for i, (name, desc) in enumerate(EXAMPLE_TARGETS):
        with cols[i]:
            if st.button(name, key=f"chip_{i}", help=desc,
                        use_container_width=True):
                _reset_results()
                st.session_state.prefill_target = name
                st.session_state.target = name
                st.session_state.step = 1
                st.rerun()

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    if st.button("Analyze →", type="primary", use_container_width=True):
        t = (target or "").strip()
        if not t:
            st.warning("Please enter a star name.")
        else:
            _reset_results()
            st.session_state.target = t
            st.session_state.step = 1
            st.rerun()

    st.markdown("""
    <div style="
        margin-top: 2rem;
        font-size: 0.75rem;
        color: #94a3b8;
        text-align: center;
        font-family: 'DM Mono', monospace;
        line-height: 1.6;
    ">
        Data sourced from NASA MAST archive via lightkurve.<br>
        Stellar parameters from NASA Exoplanet Archive.<br>
        Limb darkening from Claret &amp; Bloemen (2011).
    </div>
    </div>
    """, unsafe_allow_html=True)