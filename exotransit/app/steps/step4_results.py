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
    <div class="step-header">Step 4: Results</div>
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
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#94a3b8; margin-top:0.2rem;">
                        +{r_earth[2]:.3f} / -{r_earth[1]:.3f}
                    </div>
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
        Raw photometry fetched from NASA MAST via lightkurve. Each quarter
        normalized independently to remove inter-quarter flux jumps, then
        stitched. Detrending runs in two passes: a rough biweight filter for
        BLS period search (Pass 1), then a refined biweight with all detected
        transit windows explicitly masked so the trend is estimated from pure
        stellar continuum (Pass 2). MCMC fits the Pass-2 light curve.
        Transit detection via Box Least Squares (Kovács et al. 2002).
        Transit model fitting via MCMC (emcee) with Mandel–Agol batman model.
        Limb darkening fixed to Claret &amp; Bloemen (2011) ATLAS9 values.
        Stellar parameters from NASA Exoplanet Archive composite table.
        Physical parameters derived via Kepler's 3rd law with uncertainty
        propagated through the full MCMC posterior.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("A note on uncertainty quantification"):
        st.markdown(r"""
**What the uncertainties represent (and what they don't)**

The error bars on planet radius and impact parameter come from the MCMC posterior:
the spread of transit model parameters consistent with the detrended light curve.
This correctly captures *photon noise* (the random scatter in individual
brightness measurements).

**What is not captured:** the detrended flux is treated as fixed truth.
The biweight filter removed the stellar variability trend before MCMC ever ran.
Any uncertainty in *where the trend was*, including whether the filter slightly
suppressed a transit, or misidentified a stellar oscillation as continuum —
does not appear in the posterior. The reported error bars are therefore a lower
bound on the true parameter uncertainty.

**How heavyweight research handles this:** professional pipelines model the
stellar variability and the planet transit *simultaneously* using a Gaussian
Process (GP):

$$\text{Expected flux} = \underbrace{M(t \mid t_0, r_p, b)}_{\text{planet model}} + \underbrace{\mathcal{GP}(\theta)}_{\text{stellar noise kernel}}$$

The MCMC then explores both the transit parameters and the GP hyperparameters at
once. This correctly propagates the uncertainty from stellar variability into the
planet radius posterior, and avoids any pre-filtering step.

This project does not implement GP detrending because (1) it requires $O(N^3)$
covariance matrix operations that are too slow to run in real time on Streamlit,
and (2) GP hyperparameter inference for an unknown star requires careful prior
selection and is not straightforward to automate. For a portfolio project
demonstrating the detection-and-characterization pipeline, the biweight
pre-filter is a reasonable and widely-used approximation.
""")

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
