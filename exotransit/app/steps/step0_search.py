"""
exotransit/app/steps/step0_search.py
"""
import streamlit as st

EXAMPLE_TARGETS = [
    ("Kepler-5 b",  "Hot Jupiter, strong signal"),
    ("Kepler-11",   "6-planet system"),
    ("Kepler-22 b", "Super-Earth in habitable zone"),
    ("Kepler-7 b",  "Inflated hot Jupiter"),
    ("HD 209458",   "First transiting exoplanet"),
]

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
        placeholder="e.g. Kepler-5 b, Kepler-11, HD 209458",
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
        color: #475569;
        margin: 0.75rem 0 0.5rem 0;
    ">Try an example</div>
    """, unsafe_allow_html=True)

    # One row of buttons — no columns, just flow naturally
    cols = st.columns(5)
    for i, (name, desc) in enumerate(EXAMPLE_TARGETS):
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: rgba(30,41,59,0.6);
                border: 1px solid rgba(148,163,184,0.15);
                border-radius: 6px;
                padding: 0.4rem 0.3rem;
                text-align: center;
                font-family: 'DM Mono', monospace;
                font-size: 0.7rem;
                color: #94a3b8;
                margin-bottom: 0.25rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                title="{desc}";
            ">{name}</div>
            """, unsafe_allow_html=True)
            if st.button("↑", key=f"chip_{i}", help=f"{name} — {desc}",
                        use_container_width=True):
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
            st.session_state.target = t
            st.session_state.step = 1
            st.rerun()

    st.markdown("""
    <div style="
        margin-top: 2rem;
        font-size: 0.75rem;
        color: #334155;
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