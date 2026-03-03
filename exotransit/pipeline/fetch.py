"""
exotransit/pipeline/fetch.py

Fetches and preprocesses light curves from NASA MAST archive via lightkurve.
No data is stored locally — everything is queried at runtime.
"""

import logging
import numpy as np
import lightkurve as lk
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LightCurveData:
    """
    Container for a processed light curve.

    time : np.ndarray
        Days since reference epoch (BKJD/BTJD), corrected for Earth's
        orbital position so timing is telescope-independent.
    flux : np.ndarray
        Normalized brightness. 1.0 = baseline, 0.99 = 1% transit dip.
    flux_err : np.ndarray
        Per-point photometric uncertainty — the basis of all uncertainty analysis.
    mission : str
        'Kepler', 'K2', or 'TESS'.
    target_name : str
        Resolved name of the target star.
    sector_or_quarter : int
        Observation window index. ~90 days (Kepler quarter) or ~27 days (TESS sector).
    raw_time : np.ndarray
        Pre-processing time array, kept for before/after diagnostic plots.
    raw_flux : np.ndarray
        Pre-processing flux array, kept for before/after diagnostic plots.
    """
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    mission: str
    target_name: str
    sector_or_quarter: int
    raw_time: np.ndarray
    raw_flux: np.ndarray


def _compute_window_points(time_array: np.ndarray) -> int:
    """Odd integer number of cadences spanning ~0.5 days. Savgol requires odd."""
    cadence_days = np.nanmedian(np.diff(time_array))
    window = int(0.5 / cadence_days)
    if window % 2 == 0:
        window += 1
    return max(window, 3)


def _extract_flux_err(lc, flux: np.ndarray) -> np.ndarray:
    """Return flux_err from lc, filling NaNs with median. Falls back to std(flux)."""
    if lc.flux_err is not None:
        flux_err = np.asarray(lc.flux_err.value, dtype=float)
        median_err = np.nanmedian(flux_err)
        return np.where(np.isnan(flux_err), median_err, flux_err)
    logger.warning("No flux_err in data, estimating from scatter.")
    return np.full_like(flux, np.std(flux))


def fetch_light_curve(
    target: str,
    mission: str = "any",
    exptime: str = "long",
    sector: int | None = None,
) -> LightCurveData:
    """
    Fetch and preprocess a single quarter/sector from NASA MAST.
    Note: use fetch_stitched_light_curve() for transit detection —
    single quarters rarely contain enough transits for reliable BLS.

    Parameters
    ----------
    target : str
        Star name or catalog ID e.g. "Kepler-90", "TIC 261136679", "TOI-700".
    mission : str
        'Kepler', 'K2', 'TESS', or 'any'.
    exptime : str
        'long' (30-min Kepler / 10-min TESS) or 'short' (1-min cadence).
    sector : int or None
        Specific quarter/sector index. None = first available.

    Returns
    -------
    LightCurveData

    Raises
    ------
    ValueError
        If no light curves are found or download fails.
    """
    search_result = lk.search_lightcurve(
        target,
        mission=mission if mission != "any" else None,
        exptime=exptime,
    )

    if len(search_result) == 0:
        raise ValueError(f"No light curves found for '{target}'.")

    idx = 0
    if sector is not None:
        for i, row in enumerate(search_result):
            if hasattr(row, 'mission') and str(sector) in str(search_result[i].mission):
                idx = i
                break

    lc_raw = search_result[idx].download()
    if lc_raw is None:
        raise ValueError(f"Download failed for '{target}'.")

    raw_time = np.asarray(lc_raw.time.value, dtype=float)
    raw_flux = np.asarray(lc_raw.flux.value, dtype=float)

    # Clean → flatten (detrend) → normalize
    lc_clean = lc_raw.remove_nans().remove_outliers(sigma=5.0)
    lc_flat = lc_clean.flatten(window_length=_compute_window_points(lc_clean.time.value))
    lc_norm = lc_flat.normalize()

    time = np.asarray(lc_norm.time.value, dtype=float)
    flux = np.asarray(lc_norm.flux.value, dtype=float)
    flux_err = _extract_flux_err(lc_norm, flux)

    mission_name = str(search_result[idx].mission[0]) if hasattr(search_result[idx], 'mission') else "Unknown"
    try:
        sector_num = int(search_result[idx].mission[0].split()[-1])
    except (ValueError, IndexError):
        sector_num = 0

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        mission=mission_name,
        target_name=target,
        sector_or_quarter=sector_num,
        raw_time=raw_time,
        raw_flux=raw_flux,
    )


def fetch_stitched_light_curve(
    target: str,
    mission: str = "Kepler",
    max_quarters: int = 8,
    exptime: str = "long",
) -> LightCurveData:
    """
    Download and stitch multiple quarters/sectors into one long baseline.

    Each quarter is detrended and normalized independently before stitching —
    this removes inter-quarter flux jumps caused by spacecraft rotations
    landing the star on different detector pixels.

    Parameters
    ----------
    target : str
        Star name or catalog ID.
    mission : str
        'Kepler', 'K2', or 'TESS'.
    max_quarters : int
        Max quarters/sectors to download. 8 (~2 years) is a good balance
        between baseline length and download time.
    exptime : str
        'long' or 'short' cadence.

    Returns
    -------
    LightCurveData
        Stitched light curve with full baseline.

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

            lc = lc.remove_nans().remove_outliers(sigma=5.0)
            lc = lc.flatten(window_length=_compute_window_points(lc.time.value))
            lc = lc.normalize()
            light_curves.append(lc)

        except Exception as e:
            logger.warning(f"Quarter {i} failed ({e}), skipping")

    if not light_curves:
        raise ValueError(f"All quarters failed to download for '{target}'.")

    stitched = lk.LightCurveCollection(light_curves).stitch()

    time = np.asarray(stitched.time.value, dtype=float)
    flux = np.asarray(stitched.flux.value, dtype=float)
    flux_err = _extract_flux_err(stitched, flux)

    raw_time = np.concatenate(raw_time_list) if raw_time_list else time
    raw_flux = np.concatenate(raw_flux_list) if raw_flux_list else flux

    sort_idx = np.argsort(time)

    return LightCurveData(
        time=time[sort_idx],
        flux=flux[sort_idx],
        flux_err=flux_err[sort_idx],
        mission=mission,
        target_name=target,
        sector_or_quarter=len(light_curves),
        raw_time=raw_time,
        raw_flux=raw_flux,
    )


def list_available_observations(target: str, mission: str = "any") -> list[dict]:
    """
    List available light curves for a target without downloading anything.
    Used to populate the sector/quarter selector in the UI.

    Returns list of dicts with keys: 'index', 'mission', 'year', 'exptime'.
    """
    search_result = lk.search_lightcurve(
        target,
        mission=mission if mission != "any" else None,
    )
    return [
        {
            "index": i,
            "mission": str(search_result[i].mission[0]),
            "year": str(search_result[i].year[0]),
            "exptime": str(search_result[i].exptime[0]),
        }
        for i in range(len(search_result))
    ]
