"""
exotransit/viz/plots.py

All visualizations for the exotransit pipeline.
Plotly throughout — interactive in Streamlit, exportable as HTML/PNG.

Plot functions return plotly Figure objects so the caller decides
whether to show(), write_html(), or pass to st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from exotransit.pipeline.fetch import LightCurveData
from exotransit.detection.search import BLSResult
from exotransit.mcmc.fit import MCMCResult
from exotransit.physics.planets import PlanetPhysics
from exotransit.physics.stellar import StellarParams

# Consistent color palette across all plots
COLORS = {
    "raw": "#94a3b8",
    "trend": "#f59e0b",
    "clean": "#38bdf8",
    "transit": "#818cf8",
    "model": "#f43f5e",
    "residual": "#a3e635",
    "power": "#38bdf8",
    "peak": "#f43f5e",
    "alias": "#fb923c",
    "mcmc_sample": "#818cf8",
    "posterior": "#38bdf8",
    "hot": "#ef4444",
    "cold": "#3b82f6",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    font=dict(family="monospace", size=12),
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
)


def plot_light_curve_pipeline(lc: LightCurveData) -> go.Figure:
    """
    2-panel plot showing the full preprocessing pipeline:
      1. Raw flux
      2. Detrended and normalized flux

    Quarter/sector boundaries are marked as vertical lines.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Raw Flux", "Detrended & Normalized"),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
    )

    raw_time = np.asarray(lc.raw_time, dtype=float)
    raw_flux = np.asarray(lc.raw_flux, dtype=float)
    fig.add_trace(go.Scattergl(
        x=raw_time,
        y=raw_flux,
        mode="markers",
        marker=dict(size=1.5, color=COLORS["raw"], opacity=0.6),
        name="Raw flux",
        showlegend=True,
    ), row=1, col=1)

    # Quarter boundary lines — large time gaps (>5 days)
    time_gaps = np.where(np.diff(lc.time) > 5.0)[0]
    for idx in time_gaps:
        boundary_time = float(lc.time[idx])
        for row in [1, 2]:
            fig.add_vline(
                x=boundary_time,
                line=dict(color="#475569", width=1, dash="dot"),
                row=row, col=1,
            )

    fig.add_trace(go.Scattergl(
        x=lc.time,
        y=lc.flux,
        mode="markers",
        marker=dict(size=1.5, color=COLORS["clean"], opacity=0.6),
        name="Detrended flux",
        showlegend=True,
    ), row=2, col=1)

    # Mark transit candidates: >3-sigma dips
    median_flux = np.median(lc.flux)
    sigma = np.std(lc.flux)
    dip_mask = lc.flux < (median_flux - 3 * sigma)
    if dip_mask.sum() > 0:
        fig.add_trace(go.Scattergl(
            x=lc.time[dip_mask],
            y=lc.flux[dip_mask],
            mode="markers",
            marker=dict(size=3, color=COLORS["transit"], symbol="circle"),
            name="Transit candidates",
            showlegend=True,
        ), row=2, col=1)

    fig.update_xaxes(title_text="Time (BKJD days)", row=2, col=1)
    fig.update_yaxes(title_text="Flux", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Flux", row=2, col=1)
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{lc.target_name} — Light Curve Pipeline", x=0.5),
        height=600,
    )

    return fig


def plot_bls_power_spectrum(bls: BLSResult) -> go.Figure:
    """
    BLS power spectrum with detected period, harmonics, and SDE annotation.
    """
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=bls.periods,
        y=bls.power,
        mode="lines",
        line=dict(color=COLORS["power"], width=1),
        name="BLS power",
    ))

    peak_idx = np.argmax(bls.power)
    fig.add_trace(go.Scatter(
        x=[bls.periods[peak_idx]],
        y=[bls.power[peak_idx]],
        mode="markers+text",
        marker=dict(size=10, color=COLORS["peak"], symbol="circle"),
        text=[f"P = {bls.best_period:.4f} d"],
        textposition="top center",
        textfont=dict(color=COLORS["peak"]),
        name="Best period",
        showlegend=True,
    ))

    # Always mark common harmonics: P/2, 2P, 3P
    harmonics = {"P/2": bls.best_period * 0.5, "2P": bls.best_period * 2.0, "3P": bls.best_period * 3.0}
    for label, h_period in harmonics.items():
        if h_period < bls.periods.min() or h_period > bls.periods.max():
            continue
        h_idx = np.argmin(np.abs(bls.periods - h_period))
        fig.add_trace(go.Scatter(
            x=[bls.periods[h_idx]],
            y=[bls.power[h_idx]],
            mode="markers+text",
            marker=dict(size=8, color=COLORS["alias"], symbol="diamond"),
            text=[label],
            textposition="top center",
            textfont=dict(color=COLORS["alias"], size=10),
            name=f"Harmonic {label}",
            showlegend=True,
        ))

    # Detected aliases from BLS
    for alias in bls.aliases:
        alias_idx = np.argmin(np.abs(bls.periods - alias))
        fig.add_trace(go.Scatter(
            x=[bls.periods[alias_idx]],
            y=[bls.power[alias_idx]],
            mode="markers+text",
            marker=dict(size=8, color=COLORS["alias"], symbol="diamond-open"),
            text=[f"alias {alias:.2f}d"],
            textposition="top center",
            textfont=dict(color=COLORS["alias"], size=10),
            showlegend=False,
        ))

    fig.add_annotation(
        x=0.02, y=0.95,
        xref="paper", yref="paper",
        text=f"SDE = {bls.sde:.1f}   SNR = {bls.snr:.1f}",
        showarrow=False,
        font=dict(color="#e2e8f0", size=13),
        bgcolor="#1e293b",
        bordercolor="#475569",
    )

    fig.update_xaxes(
        title_text="Period (days)",
        type="log",
        range=[np.log10(bls.periods.min()), np.log10(bls.periods.max())],
    )
    fig.update_yaxes(title_text="BLS Power")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"BLS Power Spectrum — best period {bls.best_period:.4f} d", x=0.5),
        height=400,
        margin=dict(t=80),
    )

    return fig


def plot_phase_fold(bls: BLSResult, mcmc: MCMCResult | None = None) -> go.Figure:
    """
    Phase-folded light curve with binned points and optional batman model overlay.

    Raw folded points shown as faint gray. Binned points reveal transit shape.
    If MCMCResult provided, overlays best-fit batman model and residuals panel.
    Limb darkening fixed to mcmc.u1, mcmc.u2 (Claret values).
    """
    has_model = mcmc is not None
    rows = 2 if has_model else 1
    subplot_titles = (
        ["Phase-Folded Light Curve", "Residuals (Data − Model)"]
        if has_model else ["Phase-Folded Light Curve"]
    )

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3] if has_model else [1.0],
    )

    sort_idx = np.argsort(bls.folded_time)
    t = bls.folded_time[sort_idx]
    f = bls.folded_flux[sort_idx]

    # Raw folded points — drawn first so model renders on top
    fig.add_trace(go.Scattergl(
        x=t, y=f,
        mode="markers",
        marker=dict(size=2, color=COLORS["raw"], opacity=0.4),
        name="Folded data",
        showlegend=True,
    ), row=1, col=1)

    # Binned points — 30 bins, reveals transit shape clearly
    n_bins = 30
    bin_edges = np.linspace(t.min(), t.max(), n_bins + 1)
    bin_centers, bin_flux, bin_err = [], [], []
    for i in range(n_bins):
        mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
        if mask.sum() < 3:
            continue
        bin_centers.append(float(np.mean(t[mask])))
        bin_flux.append(float(np.mean(f[mask])))
        bin_err.append(float(np.std(f[mask]) / np.sqrt(mask.sum())))

    fig.add_trace(go.Scatter(
        x=bin_centers, y=bin_flux,
        error_y=dict(array=bin_err, color=COLORS["clean"], thickness=1.5),
        mode="markers",
        marker=dict(size=7, color=COLORS["clean"], line=dict(width=1, color="#0f172a")),
        name="Binned",
        showlegend=True,
    ), row=1, col=1)

    if has_model:
        from exotransit.mcmc.fit import _transit_model

        params = np.array([mcmc.t0_med, mcmc.rp_med, mcmc.b_med])
        t_model = np.linspace(t.min(), t.max(), 1000)
        f_model = _transit_model(
            params, t_model, mcmc.period, mcmc.u1, mcmc.u2,
            mcmc.stellar_mass, mcmc.stellar_radius,
            supersample=15,
        )

        # Glow effect — wide faint line underneath for visibility over dense data
        fig.add_trace(go.Scatter(
            x=t_model, y=f_model,
            mode="lines",
            line=dict(color=COLORS["model"], width=10),
            opacity=0.25,
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

        # Sharp model line on top
        fig.add_trace(go.Scatter(
            x=t_model, y=f_model,
            mode="lines",
            line=dict(color=COLORS["model"], width=2.5),
            name="MCMC model",
            showlegend=True,
        ), row=1, col=1)

        # Residuals
        f_model_at_data = _transit_model(
            params, t, mcmc.period, mcmc.u1, mcmc.u2,
            mcmc.stellar_mass, mcmc.stellar_radius,
            supersample=15,
        )
        residuals = f - f_model_at_data

        fig.add_trace(go.Scattergl(
            x=t, y=residuals,
            mode="markers",
            marker=dict(size=2, color=COLORS["residual"], opacity=0.4),
            name="Residuals",
            showlegend=True,
        ), row=2, col=1)

        fig.add_hline(y=0, line=dict(color="#475569", width=1), row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)

    fig.update_xaxes(title_text="Time from transit center (days)", row=rows, col=1)
    fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Phase-Folded Light Curve — P = {bls.best_period:.4f} d, "
                 f"depth = {bls.transit_depth:.5f}",
            x=0.5
        ),
        height=500 if has_model else 400,
    )

    return fig


def plot_mcmc_spaghetti(bls: BLSResult, mcmc: MCMCResult, n_samples: int = 100) -> go.Figure:
    """
    MCMC posterior spaghetti plot: n_samples random draws from the chain
    plotted as faint lines over the phase-folded data.

    Samples are [t0, rp, b] — u1/u2 are fixed to mcmc.u1, mcmc.u2.
    """
    from exotransit.mcmc.fit import _transit_model

    fig = go.Figure()

    sort_idx = np.argsort(bls.folded_time)
    t = bls.folded_time[sort_idx]
    f = bls.folded_flux[sort_idx]

    fig.add_trace(go.Scattergl(
        x=t, y=f,
        mode="markers",
        marker=dict(size=2, color=COLORS["raw"], opacity=0.2),
        name="Folded data",
        showlegend=True,
    ))

    t_model = np.linspace(t.min(), t.max(), 500)
    indices = np.random.choice(len(mcmc.samples), size=n_samples, replace=False)

    for i, idx in enumerate(indices):
        t0, rp, b = mcmc.samples[idx]
        try:
            f_model = _transit_model(
                np.array([t0, rp, b]), t_model, mcmc.period, mcmc.u1, mcmc.u2,
                mcmc.stellar_mass, mcmc.stellar_radius,
                supersample=15,
            )
            fig.add_trace(go.Scatter(
                x=t_model, y=f_model,
                mode="lines",
                line=dict(color=COLORS["mcmc_sample"], width=1),
                opacity=0.25,
                showlegend=(i == 0),
                name="Posterior samples" if i == 0 else None,
            ))
        except Exception:
            continue

    # Best fit on top
    params = np.array([mcmc.t0_med, mcmc.rp_med, mcmc.b_med])
    f_best = _transit_model(params, t_model, mcmc.period, mcmc.u1, mcmc.u2,
                            mcmc.stellar_mass, mcmc.stellar_radius,
                            supersample=15                            )
    fig.add_trace(go.Scatter(
        x=t_model, y=f_best,
        mode="lines",
        line=dict(color=COLORS["model"], width=2),
        name="Posterior median",
        showlegend=True,
    ))

    fig.update_xaxes(title_text="Time from transit center (days)")
    fig.update_yaxes(title_text="Normalized Flux")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"MCMC Posterior — {n_samples} samples", x=0.5),
        height=400,
    )

    return fig


def plot_corner(mcmc: MCMCResult) -> go.Figure:
    """
    Corner plot in physical units: planet radius (R_earth),
    orbital inclination (degrees), and transit center time (days).
    """
    # Use physically meaningful quantities instead of raw rp/b
    sample_arrays = [
        mcmc.samples[:, 0],          # t0 (days)
        mcmc.radius_earth_samples,   # planet radius (R_earth)
        mcmc.inclination_samples,    # inclination (degrees)
    ]
    names = ["Transit center (days)", "Planet radius (R⊕)", "Inclination (°)"]
    n = len(names)

    fig = make_subplots(
        rows=n, cols=n,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    for i in range(n):
        for j in range(n):
            if j > i:
                continue

            if i == j:
                fig.add_trace(go.Histogram(
                    x=sample_arrays[i],
                    nbinsx=50,
                    marker_color=COLORS["posterior"],
                    showlegend=False,
                    opacity=0.8,
                ), row=i + 1, col=j + 1)
            else:
                fig.add_trace(go.Histogram2dContour(
                    x=sample_arrays[j],
                    y=sample_arrays[i],
                    colorscale=[
                        [0.0, "#1e293b"],
                        [0.2, "#1e3a5f"],
                        [0.5, "#1d4ed8"],
                        [0.8, "#38bdf8"],
                        [1.0, "#e0f2fe"],
                    ],
                    showscale=False,
                    showlegend=False,
                    contours=dict(coloring="fill"),
                    line=dict(color="#475569", width=0.5),
                ), row=i + 1, col=j + 1)

            if i == n - 1:
                fig.update_xaxes(
                    title_text=names[j],
                    title_font=dict(size=10),
                    tickfont=dict(size=9),
                    nticks=4,
                    row=i + 1, col=j + 1,
                )
            else:
                fig.update_xaxes(showticklabels=False, row=i + 1, col=j + 1)

            if j == 0 and i != 0:
                fig.update_yaxes(
                    title_text=names[i],
                    title_font=dict(size=10),
                    tickfont=dict(size=9),
                    nticks=4,
                    row=i + 1, col=j + 1,
                )
            else:
                fig.update_yaxes(
                    showticklabels=(i == j),
                    row=i + 1, col=j + 1,
                )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="MCMC Parameter Correlations", x=0.5),
        height=900,
        width=900,
        margin=dict(l=100, r=60, t=100, b=100),
    )
    return fig


def plot_posterior_histograms(mcmc: MCMCResult) -> go.Figure:
    """
    Per-parameter posterior histograms in physical units.
    Shows planet radius (R_earth), inclination (degrees), transit center time.
    """
    sample_arrays = [
        mcmc.samples[:, 0],
        mcmc.radius_earth_samples,
        mcmc.inclination_samples,
    ]
    names     = ["Transit center (days)", "Planet radius (R⊕)", "Inclination (°)"]
    medians   = [mcmc.t0_med, mcmc.radius_earth_med, mcmc.inclination_med]
    errs_lo   = [mcmc.t0_err[0], mcmc.radius_earth_err[0], mcmc.inclination_err[0]]
    errs_hi   = [mcmc.t0_err[1], mcmc.radius_earth_err[1], mcmc.inclination_err[1]]
    n = len(names)

    fig = make_subplots(rows=1, cols=n, subplot_titles=names, horizontal_spacing=0.08)

    for i in range(n):
        fig.add_trace(go.Histogram(
            x=sample_arrays[i],
            nbinsx=50,
            marker_color=COLORS["posterior"],
            opacity=0.8,
            showlegend=False,
        ), row=1, col=i + 1)

        fig.add_vline(
            x=medians[i],
            line=dict(color=COLORS["model"], width=2),
            row=1, col=i + 1,
        )
        fig.add_vline(
            x=medians[i] - errs_lo[i],
            line=dict(color=COLORS["model"], width=1, dash="dash"),
            row=1, col=i + 1,
        )
        fig.add_vline(
            x=medians[i] + errs_hi[i],
            line=dict(color=COLORS["model"], width=1, dash="dash"),
            row=1, col=i + 1,
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="MCMC Posterior Distributions", x=0.5),
        height=350,
    )
    return fig


def plot_orrery(
    results: list[BLSResult],
    physics_list: list[PlanetPhysics],
    target_name: str,
) -> go.Figure:
    """
    Top-down orbital map scaled by semi-major axis.
    Planets colored by equilibrium temperature (blue=cold, red=hot).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=20, color="#fbbf24", symbol="circle"),
        name=target_name,
        showlegend=True,
    ))

    theta = np.linspace(0, 2 * np.pi, 300)
    t_min = min(p.equilibrium_temp[0] for p in physics_list)
    t_max = max(p.equilibrium_temp[0] for p in physics_list)

    for i, (bls, physics) in enumerate(zip(results, physics_list)):
        a = physics.semi_major_axis_au[0]
        t_eq = physics.equilibrium_temp[0]
        r_jup = physics.radius_jupiter[0]

        fig.add_trace(go.Scatter(
            x=a * np.cos(theta),
            y=a * np.sin(theta),
            mode="lines",
            line=dict(color="#334155", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

        t_norm = (t_eq - t_min) / (t_max - t_min + 1e-6)
        r = int(59 + t_norm * (239 - 59))
        g = int(130 - t_norm * 130)
        b_val = int(246 - t_norm * 246)
        color = f"rgb({r},{g},{b_val})"
        marker_size = max(8, min(30, r_jup * 20))

        fig.add_trace(go.Scatter(
            x=[a], y=[0],
            mode="markers+text",
            marker=dict(size=marker_size, color=color,
                        line=dict(width=1, color="#e2e8f0")),
            text=[f"P{i+1}"],
            textposition="top center",
            textfont=dict(size=10, color="#e2e8f0"),
            name=f"Planet {i+1}: {bls.best_period:.2f}d, {t_eq:.0f}K",
            showlegend=True,
        ))

    fig.add_annotation(
        x=0.01, y=0.01,
        xref="paper", yref="paper",
        text="Orbits shown as circular — eccentricity not constrained by transit photometry alone",
        showarrow=False,
        font=dict(color="#64748b", size=10),
        xanchor="left",
        yanchor="bottom",
    )
    fig.update_xaxes(title_text="AU", scaleanchor="y", zeroline=False)
    fig.update_yaxes(title_text="AU", zeroline=False)
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{target_name} — Orbital Architecture", x=0.5),
        height=500,
    )

    return fig


def plot_planet_comparison(
    results: list[BLSResult],
    physics_list: list[PlanetPhysics],
    target_name: str,
) -> go.Figure:
    """
    Bubble chart comparing detected planets to Solar System benchmarks.
    X-axis: insolation (Earth=1, log scale).
    Y-axis: radius (Earth radii).
    Color: equilibrium temperature.
    """
    references = [
        ("Earth",   1.0,    1.0,   255),
        ("Venus",   1.9,    0.95,  737),
        ("Neptune", 0.001,  3.9,   72),
        ("Jupiter", 0.037,  11.2,  110),
    ]

    fig = go.Figure()

    for name, insol, r_earth, t in references:
        fig.add_trace(go.Scatter(
            x=[insol], y=[r_earth],
            mode="markers+text",
            marker=dict(
                size=16,
                color=t,
                colorscale="RdBu_r",
                cmin=200, cmax=2500,
                showscale=False,
                line=dict(width=1.5, color="#94a3b8"),
            ),
            text=[name],
            textposition="top center",
            textfont=dict(size=11, color="#94a3b8"),
            name=name,
            showlegend=False,
        ))

    for i, (bls, physics) in enumerate(zip(results, physics_list)):
        r      = physics.radius_earth[0]
        insol  = physics.insolation[0]
        t_eq   = physics.equilibrium_temp[0]
        r_lo   = physics.radius_earth[1]
        r_hi   = physics.radius_earth[2]
        period = bls.best_period

        fig.add_trace(go.Scatter(
            x=[insol], y=[r],
            error_y=dict(array=[r_hi], arrayminus=[r_lo],
                         color=COLORS["clean"], thickness=2, width=6),
            mode="markers+text",
            marker=dict(
                size=max(12, min(35, r * 2.5)),
                color=t_eq,
                colorscale="RdBu_r",
                cmin=200, cmax=2500,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Equilibrium Temp (K)", side="right"),
                    thickness=15,
                    len=0.6,
                    x=1.02,
                    tickfont=dict(size=10),
                ),
                line=dict(width=1.5, color="#e2e8f0"),
            ),
            text=[f"P{i+1}"],
            textposition="top center",
            textfont=dict(size=11, color="#e2e8f0"),
            name=f"Planet {i+1}: {r:.1f} R⊕, {t_eq:.0f} K, P={period:.2f} d",
            showlegend=True,
        ))

    fig.add_vrect(
        x0=0.3, x1=1.7,
        fillcolor="#166534", opacity=0.15,
        line_width=0,
        annotation_text="Habitable Zone",
        annotation_position="top left",
        annotation_font=dict(color="#86efac", size=10),
    )

    # Pad axes so planets aren't squashed into corners
    all_insol = [p.insolation[0] for p in physics_list]
    all_radii = [p.radius_earth[0] for p in physics_list]
    x_min = min(0.0005, min(all_insol) * 0.1)
    x_max = max(3.0, max(all_insol) * 5)
    y_max = max(13.0, max(all_radii) * 1.4)

    fig.update_xaxes(
        title_text="Insolation relative to Earth (log scale)",
        type="log",
        range=[np.log10(x_min), np.log10(x_max)],
    )
    fig.update_yaxes(
        title_text="Planet Radius (R⊕)",
        range=[0, y_max],
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{target_name} — Planet Comparison", x=0.5),
        height=550,
        margin=dict(l=70, r=150, t=80, b=70),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    return fig