"""
exotransit/mcmc/fit.py

MCMC posterior sampling for transit parameter estimation.

BLS gives point estimates. This module maps the full posterior probability
distribution over transit parameters using emcee (ensemble MCMC sampler)
and batman (Mandel-Agol transit model). The result is thousands of samples
from P(params | data) — from which we read off medians, asymmetric
uncertainties, and parameter correlations with no assumptions about
Gaussian errors.

We fit 5 parameters on the phase-folded light curve, holding period fixed
at the BLS value. The fold already encodes the period information; sampling
it again from folded data is unconstrained and causes the chain to wander.
Parameters: t0, rp (radius ratio), b (impact parameter), u1, u2 (limb darkening).
"""

import logging
import numpy as np
import emcee
import batman
from dataclasses import dataclass
from exotransit.pipeline.fetch import LightCurveData
from exotransit.detection.search import BLSResult

logger = logging.getLogger(__name__)

PARAM_NAMES = ["t0", "rp", "b", "u1", "u2"]


@dataclass
class MCMCResult:
    """
    Full posterior from MCMC transit fitting.

    Attributes
    ----------
    period : float
        Orbital period fixed to BLS value (days).
    t0_med, rp_med, b_med, u1_med, u2_med : float
        Posterior medians.
    t0_err, rp_err, b_err, u1_err, u2_err : tuple[float, float]
        (lower, upper) 1-sigma uncertainties = (median - 16th pct, 84th pct - median).
        Asymmetric by design — we make no assumption of Gaussian posteriors.
    depth_med, depth_err : float, tuple
        Transit depth = rp² with uncertainty propagated through posterior samples.
    samples : np.ndarray
        Raw chain, shape (n_walkers * n_steps, 5). Kept for corner plots.
    param_names : list[str]
        Parameter names for samples columns: ["t0", "rp", "b", "u1", "u2"].
    acceptance_fraction : float
        Fraction of proposed steps accepted. Healthy range: 0.2–0.5.
    converged : bool
        Whether the chain passed autocorrelation convergence check.
    convergence_notes : list[str]
        Human-readable convergence diagnostics.
    """
    period: float
    t0_med: float
    rp_med: float
    b_med: float
    u1_med: float
    u2_med: float
    t0_err: tuple
    rp_err: tuple
    b_err: tuple
    u1_err: tuple
    u2_err: tuple
    depth_med: float
    depth_err: tuple
    samples: np.ndarray
    param_names: list
    acceptance_fraction: float
    converged: bool
    convergence_notes: list


def _transit_model(params: np.ndarray, time: np.ndarray, period: float) -> np.ndarray:
    """
    Mandel-Agol transit light curve via batman.
    params = [t0, rp, b, u1, u2]. Period is fixed, not sampled.

    Semi-major axis is derived from period assuming a solar-density star —
    good enough for posterior shape, not for publication-quality physics.
    """
    t0, rp, b, u1, u2 = params

    p = batman.TransitParams()
    p.per = period
    p.t0 = t0
    p.rp = rp
    p.b = b
    # Semi-major axis from Kepler's 3rd law assuming solar density (1 AU = 215 R_sun)
    p.a = (period / 365.25) ** (2/3) * 215.0
    p.inc = np.degrees(np.arccos(np.clip(b / p.a, -1, 1)))
    p.ecc = 0.0
    p.w = 90.0
    p.u = [u1, u2]
    p.limb_dark = "quadratic"

    return batman.TransitModel(p, time).light_curve(p)


def _log_prior(params: np.ndarray) -> float:
    """
    Uniform priors within physically meaningful bounds.
    Returns -inf for unphysical combinations.
    """
    t0, rp, b, u1, u2 = params

    if not (-0.5 < t0 < 0.5):     # t0 within ±0.5 days of fold center
        return -np.inf
    if not (0.0 < rp < 0.3):      # planet can't be >30% of stellar radius
        return -np.inf
    if not (0.0 <= b < 1.0):      # b=1 is grazing, b>1 is no transit
        return -np.inf
    if not (0.0 <= u1 <= 1.0):
        return -np.inf
    if not (0.0 <= u2 <= 1.0):
        return -np.inf

    return 0.0  # uniform — all valid combinations equally likely a priori


def _log_likelihood(params: np.ndarray, time: np.ndarray, flux: np.ndarray,
                    flux_err: np.ndarray, period: float) -> float:
    """
    Gaussian log likelihood assuming known per-point uncertainties.
    Standard for photon-noise-dominated photometry.
    """
    try:
        model_flux = _transit_model(params, time, period)
    except Exception:
        return -np.inf

    residuals = flux - model_flux
    return -0.5 * np.sum((residuals / flux_err) ** 2)


def _log_probability(params: np.ndarray, time: np.ndarray, flux: np.ndarray,
                     flux_err: np.ndarray, period: float) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = _log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(params, time, flux, flux_err, period)


def run_mcmc(
    lc: LightCurveData,
    bls: BLSResult,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burnin: int = 500,
) -> MCMCResult:
    """
    Run MCMC posterior sampling on a detected transit.

    Fits [t0, rp, b, u1, u2] on the phase-folded light curve with
    period held fixed at the BLS value. Using the folded curve reduces
    data points from ~60k to ~2k while preserving all transit information,
    making the sampler fast enough for interactive use.

    Parameters
    ----------
    lc : LightCurveData
        Full light curve from fetch pipeline.
    bls : BLSResult
        BLS detection providing starting parameters and fixed period.
    n_walkers : int
        Number of parallel MCMC walkers. Must be > 2 * n_params = 10.
        32 is a good default.
    n_steps : int
        Production steps per walker after burn-in.
    n_burnin : int
        Burn-in steps discarded before production. Walkers need time
        to find the high-probability region before we start recording.

    Returns
    -------
    MCMCResult
    """
    logger.info(f"Running MCMC for {lc.target_name}, period={bls.best_period:.4f}d (fixed)")

    # Phase-folded data: same transit information, far fewer points
    time = bls.folded_time
    flux = bls.folded_flux
    flux_err = bls.folded_flux_err
    period = bls.best_period

    # Initial parameter vector [t0, rp, b, u1, u2]
    rp_init = np.sqrt(max(bls.transit_depth, 1e-6))
    p0_center = np.array([
        0.0,      # t0: folded time is centered on 0
        rp_init,  # rp: sqrt(depth) from BLS
        0.2,      # b: start near central transit
        0.3,      # u1: typical solar-type star
        0.3,      # u2
    ])

    n_params = len(p0_center)

    # Tight Gaussian ball around starting point
    scatter = np.array([1e-4, 1e-4, 0.01, 0.01, 0.01])
    p0 = p0_center + scatter * np.random.randn(n_walkers, n_params)

    # Clip to valid ranges
    p0[:, 1] = np.clip(p0[:, 1], 1e-4, 0.29)   # rp
    p0[:, 2] = np.clip(p0[:, 2], 0.0, 0.99)     # b
    p0[:, 3] = np.clip(p0[:, 3], 0.0, 1.0)      # u1
    p0[:, 4] = np.clip(p0[:, 4], 0.0, 1.0)      # u2

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_params,
        _log_probability,
        args=(time, flux, flux_err, period),
    )

    # Burn-in: find high-probability region, discard results
    logger.info(f"Burn-in: {n_burnin} steps...")
    p0_burned, _, _ = sampler.run_mcmc(p0, n_burnin, progress=True)
    sampler.reset()

    # Production run
    logger.info(f"Production: {n_steps} steps...")
    sampler.run_mcmc(p0_burned, n_steps, progress=True)

    samples = sampler.get_chain(flat=True)
    acceptance = float(np.mean(sampler.acceptance_fraction))

    # Posteriors: median and 16th/84th percentiles (1-sigma equivalent)
    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)

    def err(i):
        return (float(p50[i] - p16[i]), float(p84[i] - p50[i]))

    # Depth = rp^2, propagated through all posterior samples
    depth_samples = samples[:, 1] ** 2
    depth_p16, depth_p50, depth_p84 = np.percentile(depth_samples, [16, 50, 84])

    # Convergence diagnostics
    notes = []
    converged = True

    if not (0.2 <= acceptance <= 0.5):
        notes.append(f"Acceptance fraction {acceptance:.2f} outside healthy range 0.2–0.5")
        converged = False

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        max_tau = float(np.max(tau))
        if n_steps < 50 * max_tau:
            notes.append(f"Chain may be too short: n_steps={n_steps} < 50×τ={max_tau:.0f}")
            converged = False
        else:
            notes.append(f"Converged: τ={max_tau:.0f}, n_steps/τ={n_steps/max_tau:.0f}")
    except Exception as e:
        notes.append(f"Could not estimate autocorrelation time: {e}")

    logger.info(
        f"MCMC done: rp={p50[1]:.4f}+{err(1)[1]:.4f}-{err(1)[0]:.4f}, "
        f"acceptance={acceptance:.3f}, converged={converged}"
    )

    return MCMCResult(
        period=period,
        t0_med=float(p50[0]),
        rp_med=float(p50[1]),
        b_med=float(p50[2]),
        u1_med=float(p50[3]),
        u2_med=float(p50[4]),
        t0_err=err(0),
        rp_err=err(1),
        b_err=err(2),
        u1_err=err(3),
        u2_err=err(4),
        depth_med=float(depth_p50),
        depth_err=(float(depth_p50 - depth_p16), float(depth_p84 - depth_p50)),
        samples=samples,
        param_names=PARAM_NAMES,
        acceptance_fraction=acceptance,
        converged=converged,
        convergence_notes=notes,
    )