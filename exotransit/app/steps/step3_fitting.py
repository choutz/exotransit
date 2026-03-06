"""
exotransit/app/steps/step3_fitting.py

Step 3: MCMC transit fitting. One fit per detected planet.
Shows phase fold with model overlay, spaghetti, posteriors, corner.
"""

import streamlit as st
from exotransit.mcmc.fit_mcmc import run_mcmc
from exotransit.viz.plots import (
    plot_phase_fold,
    plot_mcmc_spaghetti,
    plot_posterior_histograms,
    plot_corner,
)


def render():
    lc      = st.session_state.lc
    stellar = st.session_state.stellar
    ld      = st.session_state.ld
    all_bls = st.session_state.all_bls

    st.markdown("""
    <div class="step-header">Step 3 — Bayesian Transit Fitting</div>
    <h2 style="margin: 0 0 0.25rem 0; font-size: 2rem;">MCMC Parameter Estimation</h2>
    <p style="color: #94a3b8; margin: 0 0 1.5rem 0; font-family: 'DM Mono', monospace; font-size: 0.8rem;">
        Mapping the full posterior probability distribution over transit parameters
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
        <strong>Why not just fit a curve?</strong> A least-squares fit gives you one answer
        with symmetric error bars — but transit parameters are often correlated and their
        uncertainties are asymmetric. Markov Chain Monte Carlo (MCMC) instead maps the full
        <em>posterior distribution</em>: the complete probability landscape over all parameter
        combinations consistent with the data.<br><br>
        We run 32 parallel walkers for 5,000 steps each, sampling planet radius, impact
        parameter, and transit center time simultaneously. Limb darkening is fixed to
        theoretically predicted values from stellar atmosphere models (Claret &amp; Bloemen 2011)
        — the photometric cadence is insufficient to constrain it independently.
    </div>
    """, unsafe_allow_html=True)

    # Run MCMC if not cached
    if st.session_state.all_mcmc is None:
        all_mcmc = []

        for i, bls in enumerate(all_bls):
            with st.spinner(f"Running MCMC for planet {i+1} of {len(all_bls)} "
                           f"(P = {bls.best_period:.3f} d)…  ~2 min per planet"):
                try:
                    mcmc = run_mcmc(
                        lc, bls,
                        n_walkers=32,
                        n_steps=5000,
                        n_burnin=500,
                        u1=ld.u1,
                        u2=ld.u2,
                        stellar_mass=stellar.mass,
                        stellar_radius=stellar.radius,
                    )
                    all_mcmc.append(mcmc)
                except Exception as e:
                    st.session_state.error = f"MCMC failed for planet {i+1}: {e}"
                    st.rerun()

        st.session_state.all_mcmc = all_mcmc
        st.rerun()  # Fresh render so plots don't appear alongside the spinner

    all_mcmc = st.session_state.all_mcmc

    # Per-planet results
    for i, (bls, mcmc) in enumerate(zip(all_bls, all_mcmc)):

        converged_badge = "badge-green" if mcmc.converged else "badge-yellow"
        converged_text  = "Converged" if mcmc.converged else "Check notes"

        st.markdown(f"""
        <div style="margin: 2rem 0 0.75rem 0;">
            <span class="badge badge-blue">Planet {i+1}</span>
            <span class="badge {converged_badge}">{converged_text}</span>
            <span style="font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #94a3b8;">
                acceptance = {mcmc.acceptance_fraction:.3f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Phase fold with model — main result
        st.plotly_chart(
            plot_phase_fold(bls, mcmc),
            width='stretch',
            config={"displayModeBar": False},
        )

        # Posterior distributions
        st.plotly_chart(
            plot_posterior_histograms(mcmc),
            width='stretch',
            config={"displayModeBar": False},
        )

        # Advanced: spaghetti + corner in expander
        with st.expander(f"Advanced — Planet {i+1} posterior detail"):

            st.markdown("""
            <div style="
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.75rem;
                margin: 0.5rem 0 1.25rem 0;
            ">
                <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                            border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                                color: #38bdf8; margin-bottom: 0.3rem;">Posterior Spaghetti</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                        Each faint line is one random draw from the MCMC chain — a transit model
                        that is consistent with the data. The spread of lines shows the range of
                        plausible solutions. A tight bundle means the data strongly constrain the
                        parameters; a wide spread means genuine uncertainty. The solid line is the
                        posterior median.
                    </div>
                </div>
                <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                            border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                                color: #38bdf8; margin-bottom: 0.3rem;">Parameter Correlations</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                        The contour blobs show how pairs of parameters are correlated in the
                        posterior. A diagonal blob means the two parameters trade off against
                        each other — you can't pin one down without knowing the other. Diagonals
                        on the corner plot indicate the parameters are independent. The histograms
                        on the diagonal show each parameter's marginal distribution.
                    </div>
                </div>
                <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                            border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                                color: #38bdf8; margin-bottom: 0.3rem;">Planet Radius (R⊕)</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                        Derived from the radius ratio r<sub>p</sub> = R<sub>planet</sub> /
                        R<sub>star</sub> fitted by MCMC, multiplied by the stellar radius from
                        the NASA archive. Uncertainty propagates from both the MCMC posterior
                        and the archive stellar radius uncertainty.
                    </div>
                </div>
                <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                            border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                                color: #38bdf8; margin-bottom: 0.3rem;">Impact Parameter (b)</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                        The projected distance between the planet's path and the centre of the
                        stellar disk, in units of stellar radii. b = 0 is a central transit;
                        b = 1 is grazing. Higher b produces a shallower, shorter transit and
                        is correlated with r<sub>p</sub> — a grazing geometry can mimic a
                        smaller planet.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(
                    plot_mcmc_spaghetti(bls, mcmc),
                    width='stretch',
                    config={"displayModeBar": False},
                )
            with c2:
                st.plotly_chart(
                    plot_corner(mcmc),
                    width='stretch',
                    config={"displayModeBar": False},
                )

            if mcmc.convergence_notes:
                for note in mcmc.convergence_notes:
                    st.markdown(f"""
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.75rem;
                                color: #94a3b8; margin: 0.25rem 0;">
                        · {note}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("See results →", width='stretch', type="primary"):
            st.session_state.step = 4
            st.rerun()
    with col1:
        if st.button("← Back to detection"):
            st.session_state.step = 2
            st.rerun()
