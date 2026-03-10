"""
exotransit/app/steps/step1_data.py

Step 1: Fetch light curve and stellar parameters.
Shows pipeline plot and explains what we're looking at.
"""

import streamlit as st
from exotransit.pipeline.light_curves import fetch_light_curve
from exotransit.physics.stars import query_stellar_params
from exotransit.physics.limb_darkening import get_limb_darkening
from exotransit.viz.plots import plot_light_curve_pipeline


def render():
    target = st.session_state.target

    st.markdown(f"""
    <div class="step-header">Step 1: Raw Data</div>
    <h2 style="margin: 0 0 0.25rem 0; font-size: 2rem;">
        {target}
    </h2>
    <p style="color: #94a3b8; margin: 0 0 1.5rem 0; font-family: 'DM Mono', monospace; font-size: 0.8rem;">
        Fetching photometry from NASA MAST archive
    </p>
    """, unsafe_allow_html=True)

    # Run if not cached
    conf = st.session_state.conf
    if st.session_state.lc is None:
        with st.spinner("Downloading light curve from NASA MAST…"):
            try:
                lc = fetch_light_curve(target, mission="Kepler", max_quarters=conf.max_quarters)
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

        st.rerun()  # trigger detection section to appear

    lc      = st.session_state.lc
    stellar = st.session_state.stellar
    ld      = st.session_state.ld

    # Stellar summary cards
    st.markdown('<div class="step-header">Host Star</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric(f"{stellar.radius:.3f}", "R☉", "Radius (solar radii)")
    with c2:
        _metric(f"{stellar.mass:.3f}", "M☉", "Mass (solar masses)")
    with c3:
        _metric(f"{stellar.teff:.0f}", "K", "Effective temperature")
    with c4:
        _metric(f"{ld.u1:.3f}, {ld.u2:.3f}", "", "Limb darkening u₁, u₂")
        with st.popover("What is this?"):
            st.markdown(r"""
**Limb darkening**

Stars are not uniformly bright disks; they appear dimmer toward their edges
(the "limb") than at their center. This happens because the edge of the star shows
us cooler, higher-altitude gas, which emits less light.

The effect changes the shape of a transit: the brightness drop deepens as the
planet crosses the bright center, and shallows near the edge. Ignoring it would
give wrong planet radii.

We use the **quadratic limb darkening law**:

$$I(\mu) = 1 - u_1(1 - \mu) - u_2(1 - \mu)^2$$

where $\mu = \cos\theta$ is the angle from disk center ($\mu = 1$) to limb
($\mu = 0$), and $u_1$, $u_2$ are coefficients predicted from stellar atmosphere
models for this star's temperature, gravity, and metallicity
(Claret & Bloemen 2011).
""")

    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

    # Light curve plot
    fig = plot_light_curve_pipeline(lc)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    # Explanation
    st.markdown("""
    <div class="explain-box">
        <strong>Raw flux (top):</strong> Each point is one brightness measurement.
        The sharp jumps between segments are <em>quarter boundaries</em>: every 90 days
        Kepler rotated 90° to keep its solar panels facing the Sun, landing the star on
        different pixels with slightly different sensitivity. The flux level shifts but
        the underlying signal is the same.<br><br>
        <strong>Detrended &amp; normalized (bottom):</strong> After normalizing each segment to 1.0
        and removing slow instrumental drifts with a robust biweight filter, the transit
        dips become visible as tiny downward spikes, each one a planet passing in front of
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



def _metric(value: str, unit: str, label: str, val_style: str = ""):
    st.markdown(f"""
    <div class="planet-card" style="padding: 1rem; height: 7rem; box-sizing: border-box;">
        <div class="metric-val" style="{val_style}">{value}</div>
        <div class="metric-unit">{unit}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
