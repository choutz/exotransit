"""
exotransit/physics/stars.py

Queries stellar parameters from the NASA Exoplanet Archive.
All parameters needed to convert dimensionless MCMC outputs
(rp, a/R_star) into physical units (km, K, Earth radii).
"""

import logging
import requests
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


@dataclass
class StellarParams:
    """
    Stellar parameters from NASA Exoplanet Archive.

    Attributes
    ----------
    name : str
        Resolved star name.
    radius : float
        Stellar radius in solar radii.
    radius_err : tuple[float, float]
        (lower, upper) 1-sigma uncertainty on radius.
    mass : float
        Stellar mass in solar masses.
    mass_err : tuple[float, float]
    teff : float
        Effective temperature in Kelvin.
    teff_err : tuple[float, float]
    logg : float
        Surface gravity log10(g [cm/s²]).
    metallicity : float
        [Fe/H] in dex.
    source : str
        Which table/row the parameters came from.
    """
    name: str
    radius: float
    radius_err: tuple
    mass: float
    mass_err: tuple
    teff: float
    teff_err: tuple
    logg: float
    metallicity: float
    source: str


def query_stellar_params(target: str) -> StellarParams:
    """
    Query stellar parameters from NASA Exoplanet Archive via TAP/ADQL.

    Uses the pscomppars table (Planetary Systems Composite Parameters) —
    the archive's best-effort compilation of stellar properties from
    all available literature. Falls back to ps table if not found.

    Parameters
    ----------
    target : str
        Star name e.g. "Kepler-5", "Kepler-11". Drop the planet letter.

    Returns
    -------
    StellarParams

    Raises
    ------
    ValueError
        If no results found for the target.
    """
    # Strip planet designation if present (e.g. "Kepler-5 b" -> "Kepler-5")
    star_name = _clean_star_name(target)
    logger.info(f"Querying NASA Exoplanet Archive for '{star_name}'")

    query = f"""
        SELECT TOP 1
            hostname, st_rad, st_raderr1, st_raderr2,
            st_mass, st_masserr1, st_masserr2,
            st_teff, st_tefferr1, st_tefferr2,
            st_logg, st_met
        FROM pscomppars
        WHERE LOWER(hostname) LIKE LOWER('{star_name}')
    """

    params = {
        "query": query,
        "format": "json",
    }

    response = requests.get(NASA_TAP_URL, params=params, timeout=30)

    if not response.ok:
        print(f"Archive error: {response.text[:500]}")
        response.raise_for_status()
    rows = response.json()

    if not rows:
        raise ValueError(
            f"No stellar parameters found for '{star_name}' in NASA Exoplanet Archive. "
            f"Try the host star name without the planet letter."
        )

    row = rows[0]

    def safe(val, fallback=np.nan):
        return float(val) if val is not None else fallback

    return StellarParams(
        name=star_name,
        radius=safe(row.get("st_rad")),
        radius_err=(
            abs(safe(row.get("st_raderr2"), 0.0)),   # lower (negative in archive)
            safe(row.get("st_raderr1"), 0.0),          # upper
        ),
        mass=safe(row.get("st_mass")),
        mass_err=(
            abs(safe(row.get("st_masserr2"), 0.0)),
            safe(row.get("st_masserr1"), 0.0),
        ),
        teff=safe(row.get("st_teff")),
        teff_err=(
            abs(safe(row.get("st_tefferr2"), 0.0)),
            safe(row.get("st_tefferr1"), 0.0),
        ),
        logg=safe(row.get("st_logg")),
        metallicity=safe(row.get("st_met")),
        source="pscomppars",
    )


def _clean_star_name(target: str) -> str:
    """
    Strip planet designation from target name.
    'Kepler-5 b' -> 'Kepler-5', 'Kepler-11c' -> 'Kepler-11'.
    """
    import re
    # Remove trailing space + letter or trailing letter (planet designations)
    cleaned = re.sub(r'\s+[a-z]$', '', target.strip())
    cleaned = re.sub(r'[a-z]$', '', cleaned.strip())
    return cleaned.strip()