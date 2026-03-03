"""
exotransit/pipeline/fetch.py

Handles all data fetching and preprocessing from NASA MAST archive
via lightkurve. No data is stored locally — everything is queried
at runtime.
"""

import logging
import numpy as np
import lightkurve as lk
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LightCurveData:
    """
    Clean container for a processed light curve.

    Attributes
    ----------
    time : np.ndarray
        Time in BTJD (Barycentric TESS Julian Date) or BKJD (Kepler).
        These are just days elapsed since a reference date, with a
        correction for Earth's position around the Sun so that timing
        is consistent regardless of where Earth was when observed.
    flux : np.ndarray
        Normalized flux (brightness). 1.0 = baseline brightness.
        A transit dip to 0.99 means the planet blocked 1% of starlight.
    flux_err : np.ndarray
        Per-datapoint uncertainty on flux. This is real measured noise
        from the detector — the foundation of our uncertainty analysis.
    mission : str
        Which telescope collected this data: 'Kepler', 'K2', or 'TESS'.
    target_name : str
        The resolved name of the target star.
    sector_or_quarter : int
        TESS calls observation windows 'sectors', Kepler calls them
        'quarters'. Each is ~27 days (TESS) or ~90 days (Kepler).
    raw_time : np.ndarray
        Unprocessed time array, kept for diagnostic plots.
    raw_flux : np.ndarray
        Unprocessed flux array, kept for showing pipeline steps.
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
    mission: str = "any",
    exptime: str = "long",
    sector: int | None = None,
) -> LightCurveData:
    """
    Fetch and preprocess a light curve from NASA MAST archive.

    Parameters
    ----------
    target : str
        Star name or ID. Examples:
            "Kepler-90"      (famous 8-planet system)
            "TIC 261136679"  (TESS Input Catalog ID)
            "KIC 10593626"   (Kepler Input Catalog ID)
            "TOI-700"        (TESS Object of Interest)
    mission : str
        One of 'Kepler', 'K2', 'TESS', or 'any'.
    exptime : str
        'long' (30-min Kepler, 10-min TESS) or 'short' (1-min cadence).
        Long cadence is fine for most transit searches.
    sector : int or None
        Specific TESS sector or Kepler quarter. None = use first available.

    Returns
    -------
    LightCurveData
        Cleaned, normalized light curve ready for transit search.

    Raises
    ------
    ValueError
        If no light curves are found for the target.
    """
    logger.info(f"Searching MAST archive for target: {target}")

    # --- Step 1: Search the archive ---
    # lightkurve.search_lightcurve talks to NASA's MAST archive
    # and returns a table of available observations. Nothing is
    # downloaded yet — this is just a catalog query.
    search_result = lk.search_lightcurve(
        target,
        mission=mission if mission != "any" else None,
        exptime=exptime,
    )

    if len(search_result) == 0:
        raise ValueError(
            f"No light curves found for '{target}'. "
            f"Try a different target name or mission."
        )

    logger.info(f"Found {len(search_result)} light curve(s). Downloading first.")

    # --- Step 2: Download one sector/quarter ---
    # We grab just one to keep it fast and free.
    # The user can select different sectors in the app later.
    idx = 0
    if sector is not None:
        # Find the index matching the requested sector
        for i, row in enumerate(search_result):
            if hasattr(row, 'mission') and str(sector) in str(search_result[i].mission):
                idx = i
                break

    lc_raw = search_result[idx].download()

    if lc_raw is None:
        raise ValueError(f"Download failed for '{target}'.")

    # --- Step 3: Store raw arrays before any processing ---
    # We keep these so the app can show a "before/after" view
    # of the detrending pipeline. Transparency is the whole point.
    raw_time = np.asarray(lc_raw.time.value, dtype=float)
    raw_flux = np.asarray(lc_raw.flux.value, dtype=float)

    # --- Step 4: Remove NaNs and outliers ---
    # NaNs (Not a Number) appear when the detector was saturated,
    # the star was near the edge of the field, etc.
    # We also sigma-clip: remove points more than 5 standard
    # deviations from the median. These are almost always cosmic
    # rays or instrumental glitches, not real astrophysics.
    lc_clean = lc_raw.remove_nans().remove_outliers(sigma=5.0)

    # --- Step 5: Flatten (detrend) ---
    # This is the key preprocessing step. We fit a smooth spline
    # to the slow stellar variability and instrumental drift,
    # then divide it out. What remains should be flat at 1.0
    # except for transit dips.
    #
    # window_length must be an odd integer number of cadences.
    # We want to smooth over ~0.5 days to preserve transit signals
    # (most transits are 1-8 hours) while removing stellar variability.
    # We calculate how many data points fit in 0.5 days based on cadence.
    cadence_days = np.nanmedian(np.diff(lc_clean.time.value))
    window_points = int(0.5 / cadence_days)
    if window_points % 2 == 0:
        window_points += 1  # savgol filter requires odd window length
    window_points = max(window_points, 3)  # minimum sensible window
    lc_flat = lc_clean.flatten(window_length=window_points)

    # --- Step 6: Normalize ---
    # Ensure median flux is exactly 1.0. This makes transit depth
    # directly interpretable: a dip to 0.99 = 1% depth.
    lc_norm = lc_flat.normalize()

    # --- Step 7: Extract arrays ---
    time = np.asarray(lc_norm.time.value, dtype=float)
    flux = np.asarray(lc_norm.flux.value, dtype=float)

    # flux_err is the per-point photometric uncertainty.
    # lightkurve propagates this through the pipeline.
    # If it's missing for some reason, estimate from scatter.
    if lc_norm.flux_err is not None:
        flux_err = np.asarray(lc_norm.flux_err.value, dtype=float)
        # Replace any NaN errors with median error
        median_err = np.nanmedian(flux_err)
        flux_err = np.where(np.isnan(flux_err), median_err, flux_err)
    else:
        # Fallback: estimate uncertainty from the scatter
        # (standard deviation of residuals from a smoothed baseline)
        flux_err = np.full_like(flux, np.std(flux))
        logger.warning("No flux_err in data, estimating from scatter.")

    # --- Step 8: Determine mission metadata ---
    mission_name = str(search_result[idx].mission[0]) if hasattr(
        search_result[idx], 'mission'
    ) else "Unknown"

    sector_num = int(search_result[idx].mission[0].split()[-1]) if hasattr(
        search_result[idx], 'mission'
    ) else 0

    logger.info(
        f"Processed light curve: {len(time)} points, "
        f"mission={mission_name}, "
        f"flux_err median={np.median(flux_err):.6f}"
    )

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


def list_available_observations(target: str, mission: str = "any") -> list[dict]:
    """
    Return a list of available light curves for a target without
    downloading anything. Used to populate a sector/quarter selector
    in the UI.

    Parameters
    ----------
    target : str
        Star name or ID.
    mission : str
        One of 'Kepler', 'K2', 'TESS', or 'any'.

    Returns
    -------
    list of dict
        Each dict has keys: 'index', 'mission', 'year', 'exptime'
    """
    search_result = lk.search_lightcurve(
        target,
        mission=mission if mission != "any" else None,
    )

    observations = []
    for i in range(len(search_result)):
        observations.append({
            "index": i,
            "mission": str(search_result[i].mission[0]),
            "year": str(search_result[i].year[0]),
            "exptime": str(search_result[i].exptime[0]),
        })

    return observations