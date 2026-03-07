"""
exotransit/pipeline/light_curves.py

Fetches and preprocesses light curves from NASA MAST archive via lightkurve.
No data is stored locally — everything is queried at runtime.
"""

import logging
import numpy as np
import lightkurve as lk
from dataclasses import dataclass

from exotransit.pipeline.helpers import _compute_window_points, _extract_flux_err

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
        'Kepler' or 'K2'.
    target_name : str
        Resolved name of the target star.
    sector_or_quarter : int
        Observation window index. ~90 days per Kepler quarter.
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


def fetch_light_curve(
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
        'Kepler' or 'K2'.
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


