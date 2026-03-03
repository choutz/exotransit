"""
exotransit/physics/planets.py

Derives physical planet parameters from MCMC posteriors and stellar params.

All calculations propagate uncertainty by sampling — we draw from the MCMC
posterior and stellar parameter uncertainties simultaneously, then read off
percentiles. No Gaussian error propagation assumptions.
"""

import logging
import numpy as np
from dataclasses import dataclass
from exotransit.mcmc.fit import MCMCResult
from exotransit.physics.stellar import StellarParams

logger = logging.getLogger(__name__)

# Physical constants
R_SUN_KM = 695_700.0       # km
R_EARTH_KM = 6_371.0       # km
R_JUPITER_KM = 69_911.0    # km
AU_TO_R_SUN = 215.032       # 1 AU in solar radii
STEFAN_BOLTZMANN = 5.67e-8  # W m^-2 K^-4


@dataclass
class PlanetPhysics:
    """
    Physical parameters derived from MCMC posteriors + stellar params.

    All values are (median, lower_err, upper_err) tuples unless noted.
    Uncertainties are 1-sigma from posterior sampling.

    Attributes
    ----------
    radius_earth : tuple
        Planet radius in Earth radii.
    radius_jupiter : tuple
        Planet radius in Jupiter radii.
    radius_km : tuple
        Planet radius in km.
    semi_major_axis_au : tuple
        Orbital semi-major axis in AU, from Kepler's 3rd law.
    equilibrium_temp : tuple
        Equilibrium temperature in Kelvin assuming albedo=0.3 (Bond albedo).
        T_eq = T_star * sqrt(R_star / 2a) * (1 - albedo)^0.25
    insolation : tuple
        Stellar flux received relative to Earth (S_earth = 1.0).
        Key habitability metric.
    stellar_params : StellarParams
        The stellar parameters used in the calculation.
    n_samples : int
        Number of posterior samples used.
    albedo_assumed : float
        Bond albedo assumed for equilibrium temperature.
    """
    radius_earth: tuple
    radius_jupiter: tuple
    radius_km: tuple
    semi_major_axis_au: tuple
    equilibrium_temp: tuple
    insolation: tuple
    stellar_params: StellarParams
    n_samples: int
    albedo_assumed: float


def derive_planet_physics(
    mcmc: MCMCResult,
    stellar: StellarParams,
    albedo: float = 0.3,
) -> PlanetPhysics:
    """
    Derive physical planet parameters from MCMC posteriors and stellar params.

    Uncertainty propagation: draw N samples from the MCMC posterior and
    from Gaussian approximations to the stellar parameter uncertainties.
    Compute derived quantities for each sample. Read off percentiles.
    This correctly handles non-Gaussian posteriors and parameter correlations.

    Parameters
    ----------
    mcmc : MCMCResult
        MCMC posterior samples from fit.run_mcmc().
    stellar : StellarParams
        Stellar parameters from stellar.query_stellar_params().
    albedo : float
        Bond albedo assumed for equilibrium temperature. Default 0.3
        (roughly Jupiter-like). Earth is 0.3, hot Jupiters ~0.1-0.4.

    Returns
    -------
    PlanetPhysics
    """
    n = len(mcmc.samples)
    rp_samples = mcmc.samples[:, 1]  # radius ratio samples from MCMC

    # Sample stellar radius with Gaussian uncertainty
    r_star_err = np.mean(stellar.radius_err) if not any(
        np.isnan(e) for e in stellar.radius_err
    ) else 0.05 * stellar.radius
    r_star_samples = stellar.radius + r_star_err * np.random.randn(n)
    r_star_samples = np.clip(r_star_samples, 0.1, 100.0)

    # Sample stellar mass similarly
    m_star_err = np.mean(stellar.mass_err) if not any(
        np.isnan(e) for e in stellar.mass_err
    ) else 0.05 * stellar.mass
    m_star_samples = stellar.mass + m_star_err * np.random.randn(n)
    m_star_samples = np.clip(m_star_samples, 0.1, 100.0)

    # Sample stellar temperature
    teff_err = np.mean(stellar.teff_err) if not any(
        np.isnan(e) for e in stellar.teff_err
    ) else 0.02 * stellar.teff
    teff_samples = stellar.teff + teff_err * np.random.randn(n)
    teff_samples = np.clip(teff_samples, 1000.0, 100_000.0)

    # --- Planet radius ---
    # R_planet = rp * R_star (both dimensionless ratio and solar radii)
    r_planet_solar = rp_samples * r_star_samples
    r_planet_km = r_planet_solar * R_SUN_KM

    # --- Semi-major axis from Kepler's 3rd law ---
    # a^3 = G*M_star * P^2 / 4pi^2
    # In convenient units: a [AU] = (M_star [M_sun] * P [years]^2)^(1/3)
    period_years = mcmc.period / 365.25
    a_au_samples = (m_star_samples * period_years ** 2) ** (1/3)

    # --- Equilibrium temperature ---
    # T_eq = T_star * sqrt(R_star / 2a) * (1 - albedo)^0.25
    # R_star and a must be in the same units
    a_r_sun = a_au_samples * AU_TO_R_SUN
    t_eq_samples = (
        teff_samples
        * np.sqrt(r_star_samples / (2 * a_r_sun))
        * (1 - albedo) ** 0.25
    )

    # --- Insolation relative to Earth ---
    # S/S_earth = (L_star/L_sun) / (a/1AU)^2
    # L_star/L_sun = (R_star/R_sun)^2 * (T_star/T_sun)^4
    T_SUN = 5778.0
    l_star_samples = r_star_samples ** 2 * (teff_samples / T_SUN) ** 4
    insolation_samples = l_star_samples / a_au_samples ** 2

    def stats(arr):
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        return (float(p50), float(p50 - p16), float(p84 - p50))

    logger.info(
        f"Planet physics: R={stats(r_planet_km / R_EARTH_KM)[0]:.2f} R_earth, "
        f"T_eq={stats(t_eq_samples)[0]:.0f} K, "
        f"S={stats(insolation_samples)[0]:.1f} S_earth"
    )

    return PlanetPhysics(
        radius_earth=stats(r_planet_km / R_EARTH_KM),
        radius_jupiter=stats(r_planet_km / R_JUPITER_KM),
        radius_km=stats(r_planet_km),
        semi_major_axis_au=stats(a_au_samples),
        equilibrium_temp=stats(t_eq_samples),
        insolation=stats(insolation_samples),
        stellar_params=stellar,
        n_samples=n,
        albedo_assumed=albedo,
    )