"""
exotransit/app/steps/step2_detection.py

Step 2: BLS planet detection. Runs find_all_planets, shows power
spectrum and phase fold for each detected planet.
"""

import logging

import streamlit as st
from exotransit.detection.multi_planet import find_all_planets
from exotransit.viz.plots import plot_bls_power_spectrum, plot_phase_fold, plot_transit_mask
from config import CONF

logger = logging.getLogger(__name__)


def render():
    target = st.session_state.target
    lc     = st.session_state.lc

    st.markdown(f"""
    <div class="step-header">Step 2: Transit Detection</div>
    <h2 style="margin: 0 0 0.25rem 0; font-size: 2rem;">Box Least Squares Search</h2>
    <p style="color: #94a3b8; margin: 0 0 1.5rem 0; font-family: 'DM Mono', monospace; font-size: 0.8rem;">
        Searching for periodic box-shaped dips across {len(lc.time):,} data points
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
        <strong>Box Least Squares (BLS)</strong> scans hundreds of thousands of orbital period
        candidates. At each period, it phase-folds the light curve, stacking every
        orbit on top of the last, and scores how well a flat-bottomed rectangular dip fits
        the combined signal. A spike in the power spectrum flags a real repeating transit.<br><br>
        After each detection, those in-transit points are masked out and the search reruns,
        potentially revealing additional planets hiding underneath. Once all planets are
        identified, the light curve is re-detrended with all transit windows explicitly
        excluded, giving MCMC a cleaner baseline with no residual transit suppression.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How BLS works: the full picture"):
        st.markdown("""
**The sliding box**

Imagine your brightness record as a long strip of paper. You cut a small rectangular
notch out of cardboard (width set to a trial transit duration) and drag it along
the strip. Everywhere the notch fits a real dip, the residuals between the data and
the notch are small; everywhere it doesn't, they're large. The period that gives the
smallest total residual wins.

BLS does this for every combination of period, duration, and starting time across a
grid of up to 25,000 candidates. It uses a fast frequency-domain algorithm, but
conceptually it's a sliding match-filter hunting for box-shaped dips.
""")

        st.markdown("**Phase folding: stacking every orbit**")
        st.markdown(r"""
If the orbital period is $P$, every transit occurs at times $t_0,\ t_0+P,\ t_0+2P,\ \ldots$
Phase folding maps each data point to its position within one cycle:

$$\phi = \frac{(t - t_0) \bmod P}{P}$$

Points at the same orbital phase land on top of each other. Stack $N$ transits and
the signal grows by $\sqrt{N}$ while the noise averages down; a planet whose single
transit is invisible in the noise becomes clearly detectable in the stacked view.
The **phase-folded plots** below show this: potentially hundreds of individual transits
collapsed into one clean dip.
""")

        st.markdown("""
**The power spectrum and SDE**

After computing the best-fit box depth at every period, BLS returns a power spectrum:
a curve of "how good was the box fit at this period." A sharp, isolated spike means
one period dramatically outperforms all others, the hallmark of a real transit signal.

**SDE** (Signal Detection Efficiency) measures how many standard deviations the peak
rises above the surrounding noise floor. The Kepler pipeline requires SDE > 7 for a
credible detection; below that, noise fluctuations can mimic the same spike by chance.

**SNR** is independent: transit depth divided by its measurement uncertainty. A deep
transit in noisy data can have high SDE but low SNR, and vice versa.
""")

        st.markdown("""
**Multi-planet search by sequential masking**

The brightest signal (deepest, most repeated transit) always dominates the BLS
spectrum. To find planets underneath, once a planet is confirmed we replace its
in-transit data points with the median brightness, effectively erasing that signal.
The search then reruns on the cleaned light curve. Each pass can reveal a planet that
was previously hidden beneath a stronger neighbor.
""")

    # Run detection if not cached
    conf = st.session_state.conf
    if st.session_state.all_bls is None:
        _ORDINALS = ["one", "two", "three", "four", "five", "six", "seven"]

        status_text = st.empty()
        status_text.markdown(
            '<p style="font-family:\'DM Mono\',monospace; font-size:0.8rem; color:#94a3b8;">'
            '⟳ Scanning period grid…</p>',
            unsafe_allow_html=True,
        )

        def _on_progress(event, n_found, period):
            if event == "found":
                label = _ORDINALS[n_found - 1] if n_found <= len(_ORDINALS) else str(n_found)
                period_str = f" (P = {period:.3f} d)" if period else ""
                status_text.markdown(
                    f'<p style="font-family:\'DM Mono\',monospace; font-size:0.8rem; color:#4ade80;">'
                    f'✓ Found {label} planet{"s" if n_found != 1 else ""}{period_str}, checking for more…</p>',
                    unsafe_allow_html=True,
                )
            elif event == "duplicate":
                status_text.markdown(
                    '<p style="font-family:\'DM Mono\',monospace; font-size:0.8rem; color:#94a3b8;">'
                    '⟳ Duplicate period detected, masking and continuing…</p>',
                    unsafe_allow_html=True,
                )
            elif event == "redetrend":
                label = _ORDINALS[n_found - 1] if n_found <= len(_ORDINALS) else str(n_found)
                status_text.markdown(
                    f'<p style="font-family:\'DM Mono\',monospace; font-size:0.8rem; color:#94a3b8;">'
                    f'⟳ {n_found} planet{"s" if n_found != 1 else ""} found, refining detrend for MCMC…</p>',
                    unsafe_allow_html=True,
                )
            elif event == "done":
                status_text.empty()

        with st.spinner("Searching for planets…"):
            try:
                results, mask_data, refined_lc = find_all_planets(
                    lc,
                    max_planets=conf.max_planets,
                    min_period=conf.bls.min_period,
                    max_period=conf.bls.max_period,
                    max_period_grid_points=conf.bls.max_period_grid_points,
                    progress_callback=_on_progress,
                )
                status_text.empty()
                st.session_state.lc = refined_lc
                st.session_state.all_bls = results
                st.session_state.all_bls_mask_data = mask_data

            except Exception as e:
                st.session_state.error = f"BLS detection failed: {e}"
                st.rerun()

        st.rerun()

    all_bls = st.session_state.all_bls

    if not all_bls:
        st.warning("No reliable transit signals found. Try a different target using the search above.")
        return

    # Results summary
    n = len(all_bls)
    badge = "badge-green" if n >= 1 else "badge-yellow"
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <span class="badge {badge}">{n} planet{'s' if n != 1 else ''} detected</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 1rem 0 1.5rem 0;
    ">
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                    border-radius: 8px; padding: 0.75rem 1rem;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                        color: #38bdf8; margin-bottom: 0.3rem;">P: Orbital Period</div>
            <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                How long one orbit takes in days. The most fundamental property of the system,
                set by how far the planet is from its star.
            </div>
        </div>
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                    border-radius: 8px; padding: 0.75rem 1rem;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                        color: #38bdf8; margin-bottom: 0.3rem;">Depth: Transit Depth</div>
            <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                Fractional drop in brightness during transit. Proportional to
                (R<sub>planet</sub> / R<sub>star</sub>)². A depth of 0.01 means the planet
                blocks 1% of starlight.
            </div>
        </div>
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                    border-radius: 8px; padding: 0.75rem 1rem;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                        color: #38bdf8; margin-bottom: 0.3rem;">SDE: Signal Detection Efficiency</div>
            <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                How many standard deviations the peak in the BLS power spectrum stands above
                the noise floor. SDE &gt; 7 is the Kepler pipeline's minimum threshold for a
                credible detection.
            </div>
        </div>
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.12);
                    border-radius: 8px; padding: 0.75rem 1rem;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                        color: #38bdf8; margin-bottom: 0.3rem;">SNR: Signal-to-Noise Ratio</div>
            <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
                Transit depth divided by its measurement uncertainty. Independent of SDE:
                a deep transit can have low SNR if the photometric noise is large, and vice versa.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-planet plots
    all_mask_data = st.session_state.get("all_bls_mask_data") or []
    for i, bls in enumerate(all_bls):
        st.markdown(f"""
        <div style="margin: 1.5rem 0 0.5rem 0;">
            <span class="badge badge-blue">Planet {i+1}</span>
            <span style="font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
                P = {bls.best_period:.4f} d &nbsp;·&nbsp;
                depth = {bls.transit_depth * 100:.4f}% &nbsp;·&nbsp;
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

        if i < len(all_mask_data):
            md = all_mask_data[i]
            mask_window_h = bls.best_duration * (CONF.mask_width_factor/2) * 24
            with st.expander(f"Masking diagnostic: how planet {i+1} was isolated"):
                st.plotly_chart(
                    plot_transit_mask(
                        md["time"], md["flux"], md["mask"],
                        bls, planet_number=i + 1, target_name=lc.target_name,
                    ),
                    width='stretch',
                    config={"displayModeBar": False},
                )
                st.markdown(
                    f'<div style="font-size:0.8rem; color:#94a3b8; font-family:\'DM Mono\',monospace; margin-top:0.25rem;">'
                    f"Red points are replaced with the median flux before the next BLS pass. "
                    f"Mask window: ±{mask_window_h:.1f} h around each transit center."
                    f'</div>',
                    unsafe_allow_html=True,
                )

