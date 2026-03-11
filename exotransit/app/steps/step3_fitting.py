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
    conf = st.session_state.conf

    st.markdown("""
    <div class="step-header">Step 3: Bayesian Transit Fitting</div>
    <h2 style="margin: 0 0 0.25rem 0; font-size: 2rem;">MCMC Parameter Estimation</h2>
    <p style="color: #94a3b8; margin: 0 0 1.5rem 0; font-family: 'DM Mono', monospace; font-size: 0.8rem;">
        Mapping the full posterior probability distribution over transit parameters
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explain-box">
        <strong>BLS found the periods. Now we measure the planets.</strong>
        Rather than fitting a single best curve, MCMC maps the full space of transit shapes
        consistent with the data: {conf.mcmc.n_walkers} parallel "walkers" each take
        {conf.mcmc.n_steps:,} steps through parameter space, concentrating where the data
        actually supports the model. The resulting cloud of solutions
        <em>is</em> the uncertainty: wide spread = genuinely uncertain; tight cluster = well constrained.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("What is being fit, and how noise becomes uncertainty"):
        st.markdown("""
**From BLS box to physical transit model**

BLS finds the period using a rough rectangular box (fast, but not physically precise).
For the actual parameter measurement we fit the **Mandel–Agol transit model**: a
mathematically exact computation of how much starlight a spherical planet blocks at
every moment as it crosses a limb-darkened stellar disk.

The three free parameters are:
- **t₀**: transit center time
- **rp = Rp / R★**: planet-to-star radius ratio
- **b**: impact parameter (how centrally the planet crosses: 0 = central, 1 = grazing)

The transit depth (fractional brightness drop) is:
""")
        st.latex(r"\delta = \left(\frac{R_p}{R_\star}\right)^2 = r_p^2")
        st.markdown("""
Planet radius in physical units is then:
""")
        st.latex(r"R_p = r_p \times R_\star")
        st.markdown("""
Limb darkening (stars appear dimmer at their edges than their centers) is fixed
to theoretically predicted values from stellar atmosphere models (Claret & Bloemen 2011).

---
**Why not just least-squares fit?**

A least-squares fit finds the single combination that minimizes residuals. It assumes
symmetric Gaussian errors. Transit parameters are correlated: a larger planet on a
more grazing orbit (higher b) can produce the same light curve as a smaller central-transit
planet. MCMC makes no such assumptions; it discovers the full shape of the uncertainty.

---
**Monte Carlo: mapping the probability landscape**

For each candidate parameter set (t₀, rp, b), we evaluate the **log-likelihood** —
how probable is it that this model produced exactly the data we observed, given the
known measurement noise:
""")
        st.latex(
            r"\ln \mathcal{L} = -\frac{1}{2} \sum_i \frac{\left(d_i - m_i\right)^2}{\sigma_i^2}"
        )
        st.markdown(r"""
$d_i$ is the measured flux, $m_i$ is the model flux, $\sigma_i$ is Kepler's
photometric uncertainty for that cadence. Each data point downloaded from MAST comes including its own individual $\sigma_i$ value. 
Candidates with high $\mathcal{L}$ are kept; the rest are rejected. All kept samples together form the **posterior distribution**.

---
**Markov Chain: guided exploration**

Pure random sampling wastes time in low-likelihood regions. A Markov Chain concentrates
the sampling: each new candidate is proposed *near* the current position, weighted by
the likelihood ratio. The chain gravitates toward where the model fits and explores
that region thoroughly. We use emcee's affine-invariant ensemble sampler, in which
the walkers learn from each other's positions to stretch proposals in the directions
the posterior actually occupies.

---
**How telescope noise becomes parameter uncertainty**

$\sigma_i$ appears directly in the likelihood. A deep, clean transit produces a
tight posterior: the data strongly constrain the fit. A shallow transit in noisy
data produces a broad posterior. The MCMC reveals the uncertainty that was always
encoded in the noise floor of the detector. The asymmetric error bars you see on
planet radius, impact parameter, and orbital period are not formulas applied after
the fact; they are the actual spread of solutions the data cannot rule out.
""")

    # Run MCMC if not cached
    if st.session_state.all_mcmc is None:
        all_mcmc = []

        for i, bls in enumerate(all_bls):
            with st.spinner(f"Running MCMC for planet {i+1} of {len(all_bls)} "
                           f"(P = {bls.best_period:.3f} d)…  ~1 min per planet"):
                try:
                    mcmc = run_mcmc(
                        lc, bls,
                        n_walkers=conf.mcmc.n_walkers,
                        n_steps=conf.mcmc.n_steps,
                        n_burnin=conf.mcmc.n_burnin,
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
        with st.expander(f"Advanced: Planet {i+1} posterior detail"):

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
                        Each faint line is one random draw from the MCMC chain, a transit model
                        consistent with the data. The spread of lines shows the range of
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
                        each other: you can't pin one down without knowing the other. Diagonals
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
                        is correlated with r<sub>p</sub> (a grazing geometry can mimic a
                        smaller planet).
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(
                plot_mcmc_spaghetti(bls, mcmc),
                width='stretch',
                config={"displayModeBar": False},
            )
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

