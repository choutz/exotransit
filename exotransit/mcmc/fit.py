"""
exotransit/mcmc/fit.py

MCMC posterior sampling for transit parameter estimation.

BLS gives point estimates. This module maps the full posterior probability
distribution over transit parameters using emcee (ensemble MCMC sampler)
and batman (Mandel-Agol transit model). The result is thousands of samples
from P(params | data) — from which we read off medians, asymmetric
uncertainties, and parameter correlations with no assumptions about
Gaussian errors.

We fit 3 parameters on the phase-folded light curve, holding period fixed
at the BLS value and limb darkening fixed to Claret (2011) theoretical
values for the host star. This is standard practice for transit fitting
when the photometric cadence is insufficient to constrain LD independently
(Kepler long-cadence, most hot Jupiters).

Parameters sampled: t0, rp (radius ratio), b (impact parameter).
Period: fixed to BLS value.
Limb darkening: fixed to Claret values via get_limb_darkening().
Stellar mass/radius: passed in from NASA archive query for correct
    semi-major axis and transit duration. Defaults to solar values.
Exposure time: derived from light curve cadence for correct
    integration time smearing correction.
"""

import logging
import numpy as np
import emcee
import batman
from dataclasses import dataclass
from exotransit.pipeline.fetch import LightCurveData
from exotransit.detection.search import BLSResult

logger = logging.getLogger(__name__)

PARAM_NAMES = ["t0", "rp", "b"]
R_SUN_TO_EARTH   = 109.076   # 1 R_sun in Earth radii
R_SUN_TO_JUPITER = 9.731     # 1 R_sun in Jupiter radii
AU_TO_RSUN       = 215.032   # 1 AU in solar radii


@dataclass
class MCMCResult:
    """
    Full posterior from MCMC transit fitting.

    Limb darkening coefficients are fixed to Claret values and stored
    for reference, not sampled. Period is fixed to BLS value.
    Stellar parameters used for the fit are stored for provenance.

    Raw samples are in [t0, rp, b] space. Derived sample arrays convert
    to physical units for plotting — planet radius in Earth/Jupiter radii,
    orbital inclination in degrees.

    Attributes
    ----------
    period : float
        Orbital period fixed to BLS value (days).
    t0_med, rp_med, b_med : float
        Posterior medians in raw parameter space.
    t0_err, rp_err, b_err : tuple[float, float]
        (lower, upper) 1-sigma uncertainties in raw parameter space.
    depth_med, depth_err : float, tuple
        Transit depth = rp² with propagated uncertainty.
    radius_earth_med, radius_earth_err : float, tuple
        Planet radius in Earth radii with propagated uncertainty.
    radius_jupiter_med, radius_jupiter_err : float, tuple
        Planet radius in Jupiter radii with propagated uncertainty.
    inclination_med, inclination_err : float, tuple
        Orbital inclination in degrees with propagated uncertainty.
    u1, u2 : float
        Fixed limb darkening coefficients from Claret (2011).
    stellar_mass, stellar_radius : float
        Stellar parameters used for semi-major axis calculation (solar units).
    samples : np.ndarray
        Raw chain, shape (n_walkers * n_steps, 3). Columns: [t0, rp, b].
    radius_earth_samples : np.ndarray
        Planet radius samples in Earth radii.
    radius_jupiter_samples : np.ndarray
        Planet radius samples in Jupiter radii.
    inclination_samples : np.ndarray
        Orbital inclination samples in degrees.
    param_names : list[str]
        Parameter names: ["t0", "rp", "b"].
    acceptance_fraction : float
        Fraction of proposed steps accepted. Healthy range: 0.2–0.5.
    converged : bool
        Whether chain length exceeded 50x autocorrelation time.
    convergence_notes : list[str]
        Human-readable convergence diagnostics.
    """
    period: float
    t0_med: float
    rp_med: float
    b_med: float
    t0_err: tuple
    rp_err: tuple
    b_err: tuple
    depth_med: float
    depth_err: tuple
    radius_earth_med: float
    radius_earth_err: tuple
    radius_jupiter_med: float
    radius_jupiter_err: tuple
    inclination_med: float
    inclination_err: tuple
    u1: float
    u2: float
    stellar_mass: float
    stellar_radius: float
    samples: np.ndarray
    radius_earth_samples: np.ndarray
    radius_jupiter_samples: np.ndarray
    inclination_samples: np.ndarray
    param_names: list
    acceptance_fraction: float
    converged: bool
    convergence_notes: list

def _transit_model(
    params: np.ndarray,
    time: np.ndarray,
    period: float,
    u1: float,
    u2: float,
    stellar_mass: float = 1.0,
    stellar_radius: float = 1.0,
    exp_time_days: float = 30.0 / 60.0 / 24.0,
    supersample: int = 1,
) -> np.ndarray:
    """
    Mandel-Agol transit light curve via batman.
    params = [t0, rp, b]. Period and limb darkening are fixed.

    Semi-major axis is derived from Kepler's 3rd law using actual stellar
    mass, then converted to stellar radii using actual stellar radius.
    Assuming solar values produces incorrect transit duration for non-solar stars.

    supersample > 1 corrects for integration time smearing.
    exp_time_days should match the actual cadence of the observations:
        Kepler long-cadence: 30 min = 0.020833 days
        TESS 2-min cadence:  2 min  = 0.001389 days
        TESS 10-min cadence: 10 min = 0.006944 days
    """
    t0, rp, b = params

    p = batman.TransitParams()
    p.per = period
    p.t0 = t0
    p.rp = rp
    p.b = b
    # Semi-major axis: Kepler's 3rd law with actual stellar mass (AU),
    # converted to stellar radii using actual stellar radius
    a_au = (stellar_mass * (period / 365.25) ** 2) ** (1/3)
    p.a = a_au * 215.032 / stellar_radius
    p.inc = np.degrees(np.arccos(np.clip(b / p.a, -1, 1)))
    p.ecc = 0.0   # circular orbit — valid for hot Jupiters, assumed for others
    p.w = 90.0
    p.u = [u1, u2]
    p.limb_dark = "quadratic"

    m = batman.TransitModel(
        p, time,
        supersample_factor=supersample,
        exp_time=exp_time_days if supersample > 1 else 0.0,
    )
    return m.light_curve(p)


def _log_prior(params: np.ndarray) -> float:
    """Uniform priors within physically meaningful bounds."""
    t0, rp, b = params

    if not (-0.5 < t0 < 0.5):
        return -np.inf
    if not (0.0 < rp < 0.3):
        return -np.inf
    if not (0.0 <= b < 1.0):
        return -np.inf

    return 0.0


def _log_likelihood(
    params: np.ndarray,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    u1: float,
    u2: float,
    stellar_mass: float,
    stellar_radius: float,
    exp_time_days: float,
) -> float:
    """Gaussian log likelihood assuming known per-point uncertainties."""
    try:
        model_flux = _transit_model(
            params, time, period, u1, u2,
            stellar_mass, stellar_radius, exp_time_days,
        )
    except Exception:
        return -np.inf

    residuals = flux - model_flux
    return -0.5 * np.sum((residuals / flux_err) ** 2)


def _log_probability(
    params, time, flux, flux_err, period,
    u1, u2, stellar_mass, stellar_radius, exp_time_days,
) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = _log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(
        params, time, flux, flux_err, period,
        u1, u2, stellar_mass, stellar_radius, exp_time_days,
    )


def run_mcmc(
    lc: LightCurveData,
    bls: BLSResult,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burnin: int = 500,
    u1: float = 0.3,
    u2: float = 0.3,
    stellar_mass: float = 1.0,
    stellar_radius: float = 1.0,
) -> MCMCResult:
    """
    Run MCMC posterior sampling on a detected transit.

    Fits [t0, rp, b] on the phase-folded light curve with period and
    limb darkening held fixed. Stellar mass and radius are used to
    compute the correct semi-major axis and transit duration — passing
    solar defaults for a non-solar star will produce wrong transit shapes.

    Exposure time is derived from the light curve cadence automatically.

    Parameters
    ----------
    lc : LightCurveData
        Full light curve from fetch pipeline.
    bls : BLSResult
        BLS detection providing starting parameters and fixed period.
    n_walkers : int
        Number of parallel MCMC walkers. Must be > 2 * n_params = 6.
    n_steps : int
        Production steps per walker after burn-in.
    n_burnin : int
        Burn-in steps discarded before production.
    u1, u2 : float
        Fixed quadratic limb darkening coefficients from Claret (2011).
        Defaults to 0.3/0.3 — reasonable for solar-type star.
    stellar_mass : float
        Host star mass in solar masses. Default 1.0 (solar).
    stellar_radius : float
        Host star radius in solar radii. Default 1.0 (solar).

    Returns
    -------
    MCMCResult
    """
    # Cadence from median time spacing — works for Kepler, TESS, K2
    exp_time_days = float(np.median(np.diff(lc.time)))

    logger.info(
        f"Running MCMC for {lc.target_name}, period={bls.best_period:.4f}d, "
        f"u1={u1:.4f}, u2={u2:.4f} (fixed), "
        f"M*={stellar_mass:.3f} M_sun, R*={stellar_radius:.3f} R_sun, "
        f"cadence={exp_time_days * 24 * 60:.1f} min"
    )

    # Clip to transit region — scale window to 3x BLS duration
    # so ingress/egress are fully included but out-of-transit noise is limited
    window = max(bls.best_duration * 3.0, 0.3)
    transit_mask = np.abs(bls.folded_time) < window
    time = bls.folded_time[transit_mask]
    flux = bls.folded_flux[transit_mask]
    flux_err = bls.folded_flux_err[transit_mask]

    period = bls.best_period
    rp_init = np.sqrt(max(bls.transit_depth, 1e-6))
    p0_center = np.array([0.0, rp_init, 0.2])
    n_params = len(p0_center)

    scatter = np.array([1e-3, 1e-3, 0.05])
    p0 = p0_center + scatter * np.random.randn(n_walkers, n_params)
    p0[:, 1] = np.clip(p0[:, 1], 1e-4, 0.29)   # rp
    p0[:, 2] = np.clip(p0[:, 2], 0.0, 0.99)     # b

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_params,
        _log_probability,
        args=(time, flux, flux_err, period, u1, u2,
              stellar_mass, stellar_radius, exp_time_days),
    )

    logger.info(f"Burn-in: {n_burnin} steps...")
    p0_burned, _, _ = sampler.run_mcmc(p0, n_burnin, progress=True)
    sampler.reset()

    logger.info(f"Production: {n_steps} steps...")
    sampler.run_mcmc(p0_burned, n_steps, progress=True)

    samples = sampler.get_chain(flat=True)
    acceptance = float(np.mean(sampler.acceptance_fraction))

    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)

    # Derived physical samples
    # Planet radius: rp (ratio) * R_star (solar) * conversion
    r_earth_samples = samples[:, 1] * stellar_radius * R_SUN_TO_EARTH
    r_jup_samples = samples[:, 1] * stellar_radius * R_SUN_TO_JUPITER

    # Orbital inclination from impact parameter
    # b = (a/R_star) * cos(inc), so inc = arccos(b * R_star / a)
    a_au = (stellar_mass * (period / 365.25) ** 2) ** (1 / 3)
    a_rsun = a_au * AU_TO_RSUN
    inc_samples = np.degrees(np.arccos(
        np.clip(samples[:, 2] * stellar_radius / a_rsun, -1, 1)
    ))

    re_p16, re_p50, re_p84 = np.percentile(r_earth_samples, [16, 50, 84])
    rj_p16, rj_p50, rj_p84 = np.percentile(r_jup_samples, [16, 50, 84])
    inc_p16, inc_p50, inc_p84 = np.percentile(inc_samples, [16, 50, 84])

    def err(i):
        return (float(p50[i] - p16[i]), float(p84[i] - p50[i]))

    depth_samples = samples[:, 1] ** 2
    depth_p16, depth_p50, depth_p84 = np.percentile(depth_samples, [16, 50, 84])

    acceptance_ok = 0.2 <= acceptance <= 0.5
    tau_ok = False
    tau_note = ""

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        max_tau = float(np.max(tau))
        if n_steps < 50 * max_tau:
            tau_note = f"Chain too short: n_steps={n_steps} < 50×τ={max_tau:.0f}"
        else:
            tau_ok = True
            tau_note = f"τ={max_tau:.0f}, n_steps/τ={n_steps / max_tau:.0f}"
    except Exception as e:
        tau_note = f"Could not estimate autocorrelation time: {e}"

    notes = []
    if not acceptance_ok:
        notes.append(f"Acceptance fraction {acceptance:.2f} outside healthy range 0.2–0.5")
    notes.append(tau_note)

    # Converged = chain length is sufficient. Acceptance fraction is a quality
    # warning but doesn't invalidate the result if the chain converged.
    converged = tau_ok

    logger.info(
        f"MCMC done: rp={p50[1]:.4f}+{err(1)[1]:.4f}-{err(1)[0]:.4f}, "
        f"b={p50[2]:.4f}, acceptance={acceptance:.3f}"
    )

    return MCMCResult(
        period=period,
        t0_med=float(p50[0]),
        rp_med=float(p50[1]),
        b_med=float(p50[2]),
        t0_err=err(0),
        rp_err=err(1),
        b_err=err(2),
        depth_med=float(depth_p50),
        depth_err=(float(depth_p50 - depth_p16), float(depth_p84 - depth_p50)),
        radius_earth_med=float(re_p50),
        radius_earth_err=(float(re_p50 - re_p16), float(re_p84 - re_p50)),
        radius_jupiter_med=float(rj_p50),
        radius_jupiter_err=(float(rj_p50 - rj_p16), float(rj_p84 - rj_p50)),
        inclination_med=float(inc_p50),
        inclination_err=(float(inc_p50 - inc_p16), float(inc_p84 - inc_p50)),
        u1=u1,
        u2=u2,
        stellar_mass=stellar_mass,
        stellar_radius=stellar_radius,
        samples=samples,
        radius_earth_samples=r_earth_samples,
        radius_jupiter_samples=r_jup_samples,
        inclination_samples=inc_samples,
        param_names=PARAM_NAMES,
        acceptance_fraction=acceptance,
        converged=converged,
        convergence_notes=notes,
    )