"""
exotransit/app/steps/step4_results.py

Step 4: Physical results. Planet cards, orrery, comparison chart.
The payoff — raw photons to planet characterization.
"""

import streamlit as st
from exotransit.physics.planets import derive_planet_physics
from exotransit.viz.plots import plot_orrery, plot_planet_comparison


def render():
    target   = st.session_state.target
    lc       = st.session_state.lc
    stellar  = st.session_state.stellar
    all_bls  = st.session_state.all_bls
    all_mcmc = st.session_state.all_mcmc

    st.markdown(f"""
    <div class="step-header">Step 4 — Results</div>
    <h2 style="margin: 0 0 0.25rem 0; font-size: 2rem;">Planet Characterization</h2>
    <p style="color: #94a3b8; margin: 0 0 1.5rem 0; font-family: 'DM Mono', monospace; font-size: 0.8rem;">
        Physical parameters derived from MCMC posteriors + stellar parameters
    </p>
    """, unsafe_allow_html=True)

    # Derive physics if not cached
    if st.session_state.all_physics is None:
        with st.spinner("Deriving physical parameters…"):
            try:
                all_physics = [
                    derive_planet_physics(mcmc, stellar)
                    for mcmc in all_mcmc
                ]
                st.session_state.all_physics = all_physics
            except Exception as e:
                st.session_state.error = f"Physics derivation failed: {e}"
                st.rerun()

    all_physics = st.session_state.all_physics

    # ── Planet cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">Detected Planets</div>', unsafe_allow_html=True)

    for i, (bls, mcmc, phys) in enumerate(zip(all_bls, all_mcmc, all_physics)):
        r_jup   = phys.radius_jupiter
        r_earth = phys.radius_earth
        a_au    = phys.semi_major_axis_au
        t_eq    = phys.equilibrium_temp
        insol   = phys.insolation

        # Temperature color
        t_val = t_eq[0]
        if t_val > 1500:
            temp_badge = "badge-red"
            temp_label = "Ultra-hot"
        elif t_val > 800:
            temp_badge = "badge-yellow"
            temp_label = "Hot"
        elif t_val > 300:
            temp_badge = "badge-green"
            temp_label = "Warm"
        else:
            temp_badge = "badge-blue"
            temp_label = "Cold"

        st.markdown(f"""
        <div class="planet-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                <div>
                    <span style="font-family: 'Instrument Serif', serif; font-size: 1.4rem; color: #f1f5f9;">
                        Planet {i+1}
                    </span>
                    <span style="font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #94a3b8; margin-left: 0.75rem;">
                        P = {bls.best_period:.4f} d
                    </span>
                </div>
                <div>
                    <span class="badge {temp_badge}">{temp_label}</span>
                    <span class="badge badge-blue">{'Converged' if mcmc.converged else 'Check notes'}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem;">
                <div>
                    <div class="metric-val">{r_jup[0]:.3f}</div>
                    <div class="metric-unit">R<sub>Jup</sub></div>
                    <div class="metric-label">Planet radius</div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#94a3b8; margin-top:0.2rem;">
                        +{r_jup[2]:.3f} / -{r_jup[1]:.3f}
                    </div>
                </div>
                <div>
                    <div class="metric-val">{r_earth[0]:.1f}</div>
                    <div class="metric-unit">R<sub>⊕</sub></div>
                    <div class="metric-label">Planet radius</div>
                </div>
                <div>
                    <div class="metric-val">{a_au[0]:.4f}</div>
                    <div class="metric-unit">AU</div>
                    <div class="metric-label">Semi-major axis</div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#94a3b8; margin-top:0.2rem;">
                        +{a_au[2]:.4f} / -{a_au[1]:.4f}
                    </div>
                </div>
                <div>
                    <div class="metric-val">{t_eq[0]:.0f}</div>
                    <div class="metric-unit">K</div>
                    <div class="metric-label">Equilibrium temp</div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#94a3b8; margin-top:0.2rem;">
                        ±{t_eq[2]:.0f} K
                    </div>
                </div>
                <div>
                    <div class="metric-val">{insol[0]:.0f}</div>
                    <div class="metric-unit">S<sub>⊕</sub></div>
                    <div class="metric-label">Insolation</div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#94a3b8; margin-top:0.2rem;">
                        albedo = {phys.albedo_assumed}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── System plots ──────────────────────────────────────────────────────────
    st.markdown("<div style='margin: 1.5rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="step-header">Orbital Architecture</div>', unsafe_allow_html=True)

    st.plotly_chart(
        plot_orrery(all_bls, all_physics, lc.target_name),
        width='stretch',
        config={"displayModeBar": False},
    )

    st.markdown('<div class="step-header">Solar System Comparison</div>', unsafe_allow_html=True)

    st.plotly_chart(
        plot_planet_comparison(all_bls, all_physics, lc.target_name),
        width='stretch',
        config={"displayModeBar": False},
    )

    # ── Methodology note ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="explain-box" style="margin-top: 2rem;">
        <strong>Methodology:</strong>
        Raw photometry fetched from NASA MAST via lightkurve.
        Detrended with Savitzky–Golay filter, normalized per quarter.
        Transit detection via Box Least Squares (Kovács et al. 2002).
        Transit model fitting via MCMC (emcee) with Mandel–Agol batman model.
        Limb darkening fixed to Claret &amp; Bloemen (2011) ATLAS9 values.
        Stellar parameters from NASA Exoplanet Archive composite table.
        Physical parameters derived via Kepler's 3rd law with uncertainty
        propagated through the full MCMC posterior.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    if st.button("Analyze another star →", type="primary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
