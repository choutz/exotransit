import numpy as np

import logging
logger = logging.getLogger(__name__)


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
