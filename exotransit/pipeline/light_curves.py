"""
exotransit/pipeline/light_curves.py

Fetches and preprocesses light curves from NASA MAST archive via lightkurve.
No data is stored locally — everything is queried at runtime.

Detrending uses a two-pass biweight filter:

  Pass 1 (rough) — applied once to the full stitched baseline.
      The biweight M-estimator naturally down-weights outliers (including
      transit dips), so no masking is required for the first pass. This
      produces a light curve clean enough for BLS period search.

  Pass 2 (refined) — called via redetrend_with_mask() after BLS has
      identified all planet periods. Transit points are explicitly set to
      NaN before the filter runs, so the trend is estimated purely from
      the stellar continuum. The refined light curve goes to MCMC.

This two-pass approach removes any residual depth suppression that can
occur when transit dips are partially down-weighted rather than fully
excluded during trend estimation.
"""

import logging
import numpy as np
import lightkurve as lk
from dataclasses import dataclass, field
from astropy.stats import biweight_location

from exotransit.pipeline.helpers import _extract_flux_err

logger = logging.getLogger(__name__)

# Biweight window in days. Must be long enough that a typical Kepler transit
# fills less than ~30% of the window. 0.75 days = 18 hours safely handles
# transit durations up to ~5-6 hours. Wider → better transit preservation
# but less sensitive to fast stellar variability.
_BIWEIGHT_WINDOW_DAYS = 0.75


@dataclass
class LightCurveData:
    """
    Container for a processed light curve.

    time : np.ndarray
        Days since reference epoch (BKJD/BTJD).
    flux : np.ndarray
        Detrended, normalized brightness. 1.0 = baseline.
    flux_err : np.ndarray
        Per-point photometric uncertainty.
    mission : str
        'Kepler' or 'K2'.
    target_name : str
        Resolved name of the target star.
    sector_or_quarter : int
        Number of quarters stitched. ~90 days per Kepler quarter.
    raw_time : np.ndarray
        Pre-processing time array, kept for before/after diagnostic plots.
    raw_flux : np.ndarray
        Pre-processing flux array, kept for before/after diagnostic plots.
    flux_normalized : np.ndarray or None
        Post-normalize, pre-detrend flux. Stored by fetch_light_curve so
        redetrend_with_mask() can re-run the biweight filter on the original
        data with transit points explicitly excluded.
    flux_err_normalized : np.ndarray or None
        Post-normalize, pre-detrend flux_err. Stored alongside flux_normalized.
    """
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    mission: str
    target_name: str
    sector_or_quarter: int
    raw_time: np.ndarray
    raw_flux: np.ndarray
    flux_normalized: np.ndarray | None = field(default=None)
    flux_err_normalized: np.ndarray | None = field(default=None)


def _biweight_trend(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float,
    transit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a rolling biweight trend estimate.

    For each point i, collects all flux values within ±window_days/2 in time
    and computes the biweight location, a robust weighted mean that
    down-weights outliers via Tukey's bisquare kernel.

    Parameters
    ----------
    time : np.ndarray
        Time array (sorted, days).
    flux : np.ndarray
        Flux array (same length as time).
    window_days : float
        Full width of the sliding window in days.
    transit_mask : np.ndarray or None
        Boolean array (True = transit point). When provided, transit points
        are excluded from every window — the trend is estimated purely from
        stellar continuum, so transit dips cannot influence the trend at all.

    Returns
    -------
    np.ndarray
        Trend estimate at each time point.
    """
    flux_for_trend = flux.copy()
    if transit_mask is not None:
        flux_for_trend[transit_mask] = np.nan

    half  = window_days / 2.0
    n     = len(time)
    trend = np.empty_like(flux)

    # Binary-search window boundaries: O(N log N) overall
    left  = np.searchsorted(time, time - half, side='left')
    right = np.searchsorted(time, time + half, side='right')

    for i in range(n):
        window_flux = flux_for_trend[left[i]:right[i]]
        finite      = window_flux[np.isfinite(window_flux)]
        if len(finite) < 3:
            trend[i] = np.nanmedian(finite) if len(finite) > 0 else 1.0
        else:
            trend[i] = biweight_location(finite, ignore_nan=True)

    return trend


def fetch_light_curve(
    target: str,
    mission: str = "Kepler",
    max_quarters: int = 8,
    exptime: str = "long",
) -> LightCurveData:
    """
    Download and stitch multiple Kepler quarters into one detrended light curve.

    Per-quarter: remove NaNs → sigma-clip (5σ) → normalize. Normalization
    per quarter removes inter-quarter flux jumps before stitching.

    After stitching: rough biweight detrend across the full baseline (Pass 1).
    The normalized pre-detrend flux is stored in LightCurveData.flux_normalized
    so that redetrend_with_mask() can run Pass 2 after BLS.

    Parameters
    ----------
    target : str
        Star name or catalog ID.
    mission : str
        'Kepler' or 'K2'.
    max_quarters : int
        Max quarters to download and stitch.
    exptime : str
        'long' (30-min Kepler cadence) or 'short' (1-min cadence).

    Returns
    -------
    LightCurveData
        Stitched, detrended, normalized light curve with flux_normalized stored.

    Raises
    ------
    ValueError
        If no quarters are found or all downloads fail.
    """
    search_result = lk.search_lightcurve(target, mission=mission, exptime=exptime)

    if len(search_result) == 0:
        raise ValueError(f"No light curves found for '{target}'.")

    n_to_download = min(max_quarters, len(search_result))
    light_curves, raw_time_list, raw_flux_list = [], [], []

    for i in range(n_to_download):
        try:
            lc = search_result[i].download()
            if lc is None:
                logger.warning(f"Quarter {i} returned None, skipping")
                continue

            raw_time_list.append(lc.time.value)
            raw_flux_list.append(lc.flux.value)

            # Normalize per quarter to remove inter-quarter flux jumps.
            # No detrending here. the biweight runs once on the full stitched
            # baseline, avoiding quarter-boundary edge effects.
            lc = lc.remove_nans().remove_outliers(sigma_upper=5.0, sigma_lower=np.inf)
            lc = lc.normalize()
            light_curves.append(lc)

        except Exception as e:
            logger.warning(f"Quarter {i} failed ({e}), skipping")

    if not light_curves:
        raise ValueError(f"All quarters failed to download for '{target}'.")

    stitched = lk.LightCurveCollection(light_curves).stitch()

    time_arr     = np.asarray(stitched.time.value, dtype=float)
    flux_arr     = np.asarray(stitched.flux.value, dtype=float)
    flux_err_arr = _extract_flux_err(stitched, flux_arr)

    sort_idx     = np.argsort(time_arr)
    time_arr     = time_arr[sort_idx]
    flux_arr     = flux_arr[sort_idx]
    flux_err_arr = flux_err_arr[sort_idx]

    # ── Pass 1: rough biweight detrend ────────────────────────────────────────
    logger.info(
        f"Pass-1 biweight detrend (window={_BIWEIGHT_WINDOW_DAYS}d) "
        f"on {len(time_arr)} points across {len(light_curves)} quarters"
    )
    trend = _biweight_trend(time_arr, flux_arr, _BIWEIGHT_WINDOW_DAYS)

    safe_trend   = np.where(np.abs(trend) > 1e-10, trend, 1.0)
    flux_normed  = flux_arr / safe_trend
    err_normed   = flux_err_arr / safe_trend

    finite_mask  = np.isfinite(flux_normed) & np.isfinite(err_normed)
    time_clean   = time_arr[finite_mask]
    flux_clean   = flux_normed[finite_mask]
    err_clean    = err_normed[finite_mask]

    # Store the pre-detrend (but post-normalize) flux for Pass 2.
    # After BLS, redetrend_with_mask() will re-run the biweight on this
    # with transit times explicitly excluded.
    flux_norm_stored    = flux_arr[finite_mask]
    flux_err_norm_stored = flux_err_arr[finite_mask]

    raw_time = np.concatenate(raw_time_list) if raw_time_list else time_clean
    raw_flux = np.concatenate(raw_flux_list) if raw_flux_list else flux_clean

    return LightCurveData(
        time=time_clean,
        flux=flux_clean,
        flux_err=err_clean,
        mission=mission,
        target_name=target,
        sector_or_quarter=len(light_curves),
        raw_time=raw_time,
        raw_flux=raw_flux,
        flux_normalized=flux_norm_stored,
        flux_err_normalized=flux_err_norm_stored,
    )


def redetrend_with_mask(lc: LightCurveData, bls_results: list) -> LightCurveData:
    """
    Pass 2: re-run biweight detrend with all detected transit times masked.

    Transit points are set to NaN before the biweight window runs, so the
    trend is estimated purely from the stellar continuum. Any residual depth
    suppression from Pass 1 (where transit dips received low but non-zero
    weight) is eliminated.

    Call this after BLS has found all planets and before running MCMC.

    Parameters
    ----------
    lc : LightCurveData
        Output from fetch_light_curve(). Must have flux_normalized set.
    bls_results : list of BLSResult
        Detected planets. Transit times are derived from best_period, best_t0,
        and best_duration for each result.

    Returns
    -------
    LightCurveData
        New light curve with refined detrending. flux_normalized and
        flux_err_normalized are preserved from the original.
    """
    # Use getattr so stale pickled LightCurveData objects (created before
    # flux_normalized was added) raise AttributeError → None gracefully.
    flux_norm     = getattr(lc, 'flux_normalized', None)
    flux_err_norm = getattr(lc, 'flux_err_normalized', None)

    if flux_norm is None or flux_err_norm is None:
        logger.warning(
            "redetrend_with_mask: flux_normalized not stored — "
            "returning original LC unchanged. "
            "Delete the .lc_cache/ pickle and re-fetch to enable Pass-2 re-detrending."
        )
        return lc

    if not bls_results:
        logger.info("redetrend_with_mask: no planets to mask — returning original LC")
        return lc

    # Build combined boolean mask over all detected planets.
    # Use 3× BLS duration as the mask window — wide enough to cover ingress/egress
    # and any BLS duration underestimate.
    combined_mask = np.zeros(len(lc.time), dtype=bool)
    lc_lk = lk.LightCurve(time=lc.time, flux=lc.flux)
    for bls in bls_results:
        planet_mask = lc_lk.create_transit_mask(
            period=bls.best_period,
            transit_time=bls.best_t0,
            duration=bls.best_duration * 3.0,
        )
        combined_mask |= np.asarray(planet_mask)

    n_masked = int(combined_mask.sum())
    logger.info(
        f"Pass-2 biweight detrend: masking {n_masked} transit points "
        f"({n_masked/len(lc.time)*100:.1f}% of baseline) across "
        f"{len(bls_results)} planet(s)"
    )

    trend = _biweight_trend(
        lc.time,
        flux_norm,
        _BIWEIGHT_WINDOW_DAYS,
        transit_mask=combined_mask,
    )

    safe_trend   = np.where(np.abs(trend) > 1e-10, trend, 1.0)
    flux_refined = flux_norm / safe_trend
    err_refined  = flux_err_norm / safe_trend

    finite_mask  = np.isfinite(flux_refined) & np.isfinite(err_refined)

    return LightCurveData(
        time=lc.time[finite_mask],
        flux=flux_refined[finite_mask],
        flux_err=err_refined[finite_mask],
        mission=lc.mission,
        target_name=lc.target_name,
        sector_or_quarter=lc.sector_or_quarter,
        raw_time=lc.raw_time,
        raw_flux=lc.raw_flux,
        flux_normalized=lc.flux_normalized[finite_mask],
        flux_err_normalized=lc.flux_err_normalized[finite_mask],
    )

