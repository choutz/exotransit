"""
exotransit/physics/limb_darkening.py

Quadratic limb darkening coefficients for the Kepler bandpass via
interpolation in the Claret & Bloemen (2011) ATLAS model grid.

Source: Full table downloaded from
    https://tapvizier.cds.unistra.fr/adql/?%20J/A+A/529/A75/table-af
Filtered to: Filt='Kp' (Kepler bandpass), Met='L' (LTE), Mod='A' (ATLAS9)
Saved as: exotransit/physics/claret2011_kepler.csv (9586 rows)
Reference: Claret, A. & Bloemen, S. (2011), A&A 529, A75

Coverage and limitations:
- Teff: 3500–50000 K (main sequence through hot stars)
- logg: 0.0–5.0 (giants through white dwarfs)
- Metallicity [Fe/H]: -5.0 to +1.0
- Kepler bandpass only — not valid for K2, ground-based, or other space-based filters
- ATLAS9 LTE models: reliable for FGK stars (Teff 4000–8000K, logg 3.5–5.0)
  Less reliable for M dwarfs (Teff < 4000K, where PHOENIX models are preferred)
  and hot stars (Teff > 8000K, where non-LTE effects become significant)
- Microturbulence velocity xi=2.0 km/s assumed throughout
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TABLE_PATH = Path(__file__).parent / "claret2011_kepler.csv"
_table: pd.DataFrame | None = None


@dataclass
class LimbDarkeningCoeffs:
    """
    Quadratic limb darkening coefficients u1, u2 for the Kepler bandpass.

    Attributes
    ----------
    u1, u2 : float
        Quadratic limb darkening coefficients.
        Intensity profile: I(mu) = 1 - u1*(1-mu) - u2*(1-mu)^2
        where mu = cos(angle from disk center).
    u1_sigma, u2_sigma : float
        Uncertainty from grid interpolation — reflects how far the
        stellar parameters are from the nearest tabulated grid points.
    teff_used, logg_used, z_used : float
        Stellar parameters actually used for the lookup.
    """
    u1: float
    u2: float
    u1_sigma: float
    u2_sigma: float
    teff_used: float
    logg_used: float
    z_used: float


def _load_table() -> pd.DataFrame:
    """Load Claret 2011 table once and cache in memory."""
    global _table
    if _table is None:
        if not _TABLE_PATH.exists():
            raise FileNotFoundError(
                f"Claret 2011 limb darkening table not found at {_TABLE_PATH}. "
                f"See exotransit/physics/README for instructions."
            )
        _table = pd.read_csv(_TABLE_PATH)
        logger.info(f"Loaded Claret 2011 LD table: {len(_table)} rows")
    return _table


def get_limb_darkening(
    teff: float,
    logg: float,
    metallicity: float = 0.0,
    n_neighbors: int = 8,
) -> LimbDarkeningCoeffs:
    """
    Interpolate quadratic limb darkening coefficients from Claret (2011).

    Uses inverse-distance weighted interpolation over the nearest
    n_neighbors grid points in (Teff, logg, Z) space. Distances are
    computed in normalized coordinates so all three parameters contribute
    equally regardless of their absolute scales.

    Parameters
    ----------
    teff : float
        Stellar effective temperature in Kelvin.
    logg : float
        Stellar surface gravity log10(g [cm/s²]).
    metallicity : float
        Stellar metallicity [Fe/H] in dex. Default 0.0 (solar).
    n_neighbors : int
        Number of nearest grid points to use for interpolation.

    Returns
    -------
    LimbDarkeningCoeffs
    """
    df = _load_table()

    teff_range = df["Teff"].max() - df["Teff"].min()
    logg_range = df["logg"].max() - df["logg"].min()
    z_range = df["Z"].max() - df["Z"].min() or 1.0

    dt = (df["Teff"].values - teff) / teff_range
    dl = (df["logg"].values - logg) / logg_range
    dz = (df["Z"].values - metallicity) / z_range

    distances = np.sqrt(dt**2 + dl**2 + dz**2)

    nearest_idx = np.argsort(distances)[:n_neighbors]
    nearest_dist = distances[nearest_idx]
    nearest_rows = df.iloc[nearest_idx]

    # epsilon prevents div-by-zero and floating point blow-up near exact matches
    weights = 1.0 / (nearest_dist + 1e-10)
    weights /= weights.sum()

    u1 = float(np.sum(weights * nearest_rows["a"].values))
    u2 = float(np.sum(weights * nearest_rows["b"].values))

    u1_sigma = float(np.sqrt(np.sum(weights * (nearest_rows["a"].values - u1) ** 2)))
    u2_sigma = float(np.sqrt(np.sum(weights * (nearest_rows["b"].values - u2) ** 2)))

    sigma_floor = 0.005

    u1_sigma = max(u1_sigma, sigma_floor)
    u2_sigma = max(u2_sigma, sigma_floor)

    logger.info(
        f"LD coefficients for Teff={teff:.0f}K, logg={logg:.2f}, [Fe/H]={metallicity:.2f}: "
        f"u1={u1:.4f}±{u1_sigma:.4f}, u2={u2:.4f}±{u2_sigma:.4f}"
    )

    return LimbDarkeningCoeffs(
        u1=u1,
        u2=u2,
        u1_sigma=u1_sigma,
        u2_sigma=u2_sigma,
        teff_used=teff,
        logg_used=logg,
        z_used=metallicity,
    )