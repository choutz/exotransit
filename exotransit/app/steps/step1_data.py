"""
exotransit/app/steps/step1_data.py

Step 1: Fetch light curve and stellar parameters.
Shows pipeline plot and explains what we're looking at.
"""

import streamlit as st
from exotransit.pipeline.light_curves import fetch_stitched_light_curve
from exotransit.physics.stars import query_stellar_params
from exotransit.physics.limb_darkening import get_limb_darkening
from exotransit.viz.plots import plot_light_curve_pipeline


def render():
    target = st.session_state.target

    st.markdown(f"""
    <div class="step-header">Step 1 — Raw Data</div>
    <h2 style="margin: 0 0 0.25rem 0; font-size: 2rem;">
        {target}
    </h2>
    <p style="color: #94a3b8; margin: 0 0 1.5rem 0; font-family: 'DM Mono', monospace; font-size: 0.8rem;">
        Fetching photometry from NASA MAST archive
    </p>
    """, unsafe_allow_html=True)

    # Run if not cached
    if st.session_state.lc is None:
        with st.spinner("Downloading light curve from NASA MAST…"):
            try:
                lc = fetch_stitched_light_curve(target, mission="Kepler", max_quarters=8)
                st.session_state.lc = lc
            except Exception as e:
                st.session_state.error = (
                    f"Could not fetch Kepler light curve for '{target}'. "
                    f"Check the star name and try again.\n\nDetails: {e}"
                )
                st.rerun()

        with st.spinner("Querying stellar parameters from NASA Exoplanet Archive…"):
            try:
                stellar = query_stellar_params(target)
                st.session_state.stellar = stellar
                ld = get_limb_darkening(
                    teff=stellar.teff,
                    logg=stellar.logg,
                    metallicity=stellar.metallicity,
                )
                st.session_state.ld = ld
            except Exception as e:
                st.session_state.error = (
                    f"Could not query stellar parameters for '{target}'. "
                    f"Details: {e}"
                )
                st.rerun()

    lc      = st.session_state.lc
    stellar = st.session_state.stellar
    ld      = st.session_state.ld

    # Stellar summary cards
    st.markdown('<div class="step-header">Host Star</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric(f"{stellar.radius:.3f}", "R☉", "Stellar radius")
    with c2:
        _metric(f"{stellar.mass:.3f}", "M☉", "Stellar mass")
    with c3:
        _metric(f"{stellar.teff:.0f}", "K", "Effective temperature")
    with c4:
        _metric(f"{ld.u1:.3f}, {ld.u2:.3f}", "", "Limb darkening u₁, u₂")

    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

    # Light curve plot
    fig = plot_light_curve_pipeline(lc)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    # Explanation
    st.markdown("""
    <div class="explain-box">
        <strong>Raw flux (top):</strong> Each point is one brightness measurement.
        The sharp jumps between segments are <em>quarter boundaries</em> — every 90 days
        Kepler rotated 90° to keep its solar panels facing the Sun, landing the star on
        different pixels with slightly different sensitivity. The flux level shifts but
        the underlying signal is the same.<br><br>
        <strong>Detrended &amp; normalized (bottom):</strong> After removing slow instrumental
        drifts with a Savitzky–Golay filter and normalizing each segment to 1.0, the transit
        dips become visible as tiny downward spikes — each one a planet passing in front of
        the star. The highlighted points are candidates flagged at &gt;3σ below the median.
    </div>
    """, unsafe_allow_html=True)

    # Data summary
    baseline = lc.time[-1] - lc.time[0]
    cadence  = float(__import__('numpy').median(__import__('numpy').diff(lc.time))) * 24 * 60

    c1, c2, c3 = st.columns(3)
    with c1:
        _metric(f"{len(lc.time):,}", "points", "Data points")
    with c2:
        _metric(f"{baseline:.0f}", "days", "Observation baseline")
    with c3:
        _metric(f"{cadence:.0f}", "min", "Cadence")

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Search for planets →", width='stretch', type="primary"):
            st.session_state.step = 2
            st.rerun()
    with col1:
        if st.button("← Start over", use_container_width=False):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


def _metric(value: str, unit: str, label: str, val_style: str = ""):
    st.markdown(f"""
    <div class="planet-card" style="padding: 1rem; height: 7rem; box-sizing: border-box;">
        <div class="metric-val" style="{val_style}">{value}</div>
        <div class="metric-unit">{unit}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
