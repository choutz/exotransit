import batman
import numpy as np


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
        Kepler short-cadence: 1 min = 0.000694 days
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


def _log_prior(params: np.ndarray, snr: float = 0.0) -> float:
    """
    Log prior. Hard bounds on all parameters plus a Gaussian soft prior on b.

    The soft prior penalizes grazing transits (high b) for strong detections.
    For high-SNR signals a grazing geometry is implausible — the transit shape
    would be V-shaped, not box-shaped. Width scales with SNR: tight for strong
    signals, loose for weak ones where grazing is genuinely uncertain.
    """
    t0, rp, b = params

    if not (-0.5 < t0 < 0.5):
        return -np.inf
    if not (1e-4 < rp < 0.3):
        return -np.inf
    if not (0.0 <= b < 1.0):
        return -np.inf

    # Soft Gaussian prior on b centered at 0 (central transit).
    # b_sigma widens for low-SNR detections where grazing is uncertain.
    b_sigma = max(0.3, 1.0 / (1.0 + snr / 20.0))
    lp = -0.5 * (b / b_sigma) ** 2

    return lp


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
    snr: float = 0.0,
) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = _log_prior(params, snr=snr)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(
        params, time, flux, flux_err, period,
        u1, u2, stellar_mass, stellar_radius, exp_time_days,
    )
