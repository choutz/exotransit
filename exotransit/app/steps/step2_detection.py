"""
exotransit/app/steps/step2_detection.py

Step 2: BLS planet detection. Runs find_all_planets, shows power
spectrum and phase fold for each detected planet.
"""

import streamlit as st
from exotransit.detection.search import find_all_planets
from exotransit.viz.plots import plot_bls_power_spectrum, plot_phase_fold
from exotransit.detection.search import run_bls
import lightkurve as lk
import numpy as np
from exotransit.pipeline.fetch import LightCurveData


def render():
    target = st.session_state.target
    lc     = st.session_state.lc

    st.markdown(f"""
    <div class="step-header">Step 2 — Transit Detection</div>
    <h2 style="margin: 0 0 0.25rem 0; font-size: 2rem;">Box Least Squares Search</h2>
    <p style="color: #64748b; margin: 0 0 1.5rem 0; font-family: 'DM Mono', monospace; font-size: 0.8rem;">
        Searching for periodic box-shaped dips across {len(lc.time):,} data points
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
        <strong>Box Least Squares (BLS)</strong> tests thousands of period and duration
        combinations. For each candidate period, it phase-folds the light curve — stacking
        every orbit on top of each other — and scores how well a rectangular dip fits the
        stacked data. A sharp spike in the power spectrum means the data has a repeating
        box-shaped dip at that period: the signature of a transiting planet.<br><br>
        After each detection, the transits at that period are masked out and the search
        runs again to find weaker signals buried underneath.
    </div>
    """, unsafe_allow_html=True)

    # Run detection if not cached
    if st.session_state.all_bls is None:
        progress = st.progress(0, text="Initializing BLS search…")

        try:
            results = []
            # Run iteratively so we can update progress
            lc_lk = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
            max_planets = 6

            for i in range(max_planets):
                progress.progress(
                    int((i / max_planets) * 90),
                    text=f"Searching for planet {i+1}…"
                )

                current_lc = LightCurveData(
                    time=np.asarray(lc_lk.time.value),
                    flux=np.asarray(lc_lk.flux.value),
                    flux_err=np.asarray(lc_lk.flux_err.value),
                    mission=lc.mission,
                    target_name=lc.target_name,
                    sector_or_quarter=lc.sector_or_quarter,
                    raw_time=lc.raw_time,
                    raw_flux=lc.raw_flux,
                )

                result = run_bls(
                    current_lc,
                    min_period=2,
                    max_period=12,
                    max_period_grid_points=100_000,
                )

                if not result.is_reliable:
                    break

                results.append(result)

                mask = lc_lk.create_transit_mask(
                    period=result.best_period,
                    transit_time=result.best_t0,
                    duration=result.best_duration * 1.5,
                )
                lc_lk = lc_lk[~mask]

            progress.progress(100, text=f"Found {len(results)} planet candidate(s)")
            st.session_state.all_bls = results

        except Exception as e:
            st.session_state.error = f"BLS detection failed: {e}"
            st.rerun()

    all_bls = st.session_state.all_bls

    if not all_bls:
        st.warning("No reliable transit signals found. Try a different target.")
        if st.button("← Start over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        return

    # Results summary
    n = len(all_bls)
    badge = "badge-green" if n >= 1 else "badge-yellow"
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <span class="badge {badge}">{n} planet{'s' if n != 1 else ''} detected</span>
    </div>
    """, unsafe_allow_html=True)

    # Per-planet plots
    for i, bls in enumerate(all_bls):
        st.markdown(f"""
        <div style="margin: 1.5rem 0 0.5rem 0;">
            <span class="badge badge-blue">Planet {i+1}</span>
            <span style="font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
                P = {bls.best_period:.4f} d &nbsp;·&nbsp;
                depth = {bls.transit_depth:.5f} &nbsp;·&nbsp;
                SDE = {bls.sde:.1f} &nbsp;·&nbsp;
                SNR = {bls.snr:.1f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_bls_power_spectrum(bls),
                width='stretch',
                config={"displayModeBar": False},
            )
        with c2:
            st.plotly_chart(
                plot_phase_fold(bls),
                width='stretch',
                config={"displayModeBar": False},
            )

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Fit transit models →", width='stretch', type="primary"):
            st.session_state.step = 3
            st.rerun()
    with col1:
        if st.button("← Back to data"):
            st.session_state.step = 1
            st.rerun()
