import logging
import numpy as np
from exotransit.detection.search import run_bls
from exotransit.mcmc.fit import run_mcmc
from tests.helpers import get_light_curve
from exotransit.physics.stellar import query_stellar_params
from exotransit.physics.planets import derive_planet_physics
from exotransit.physics.limb_darkening import get_limb_darkening

logging.disable(logging.INFO)


if __name__ == "__main__":
    lc = get_light_curve("Kepler-5 b", mission="Kepler", max_quarters=8)

    print("Running BLS...")
    bls = run_bls(lc, min_period=2.0, max_period=10.0)
    print(f"  BLS period:  {bls.best_period:.4f} days")
    print(f"  BLS depth:   {bls.transit_depth:.6f} ± {bls.depth_uncertainty:.6f}")
    print(f"  BLS rp est:  {np.sqrt(bls.transit_depth):.4f}")

    print("\nQuerying stellar parameters...")
    stellar = query_stellar_params("Kepler-5 b")
    print(f"  Star:    {stellar.name}")
    print(f"  R_star:  {stellar.radius:.3f} +{stellar.radius_err[1]:.3f} -{stellar.radius_err[0]:.3f} R_sun")
    print(f"  M_star:  {stellar.mass:.3f} +{stellar.mass_err[1]:.3f} -{stellar.mass_err[0]:.3f} M_sun")
    print(f"  T_eff:   {stellar.teff:.0f} +{stellar.teff_err[1]:.0f} -{stellar.teff_err[0]:.0f} K")

    print("\nLimb darkening...")
    ld = get_limb_darkening(
        teff=stellar.teff,
        logg=stellar.logg,
        metallicity=stellar.metallicity,
    )
    print(f"  u1={ld.u1:.4f}±{ld.u1_sigma:.4f}, u2={ld.u2:.4f}±{ld.u2_sigma:.4f}")

    print("\nRunning MCMC...")
    mcmc = run_mcmc(
        lc, bls,
        n_walkers=32, n_steps=5000, n_burnin=500,
        u1=ld.u1, u2=ld.u2,
        stellar_mass=stellar.mass,
        stellar_radius=stellar.radius,
    )

    print(f"\nMCMC Results:")
    print(f"  period:  {mcmc.period:.5f} days (fixed from BLS)")
    print(f"  t0:      {mcmc.t0_med:.5f} +{mcmc.t0_err[1]:.5f} -{mcmc.t0_err[0]:.5f}")
    print(f"  rp:      {mcmc.rp_med:.5f} +{mcmc.rp_err[1]:.5f} -{mcmc.rp_err[0]:.5f}")
    print(f"  b:       {mcmc.b_med:.4f} +{mcmc.b_err[1]:.4f} -{mcmc.b_err[0]:.4f}")
    # print(f"  u1:      {mcmc.u1_med:.4f} +{mcmc.u1_err[1]:.4f} -{mcmc.u1_err[0]:.4f}")
    # print(f"  u2:      {mcmc.u2_med:.4f} +{mcmc.u2_err[1]:.4f} -{mcmc.u2_err[0]:.4f}")
    print(f"  depth:   {mcmc.depth_med:.6f} +{mcmc.depth_err[1]:.6f} -{mcmc.depth_err[0]:.6f}")
    print(f"\n  Acceptance fraction: {mcmc.acceptance_fraction:.3f}")
    print(f"  Converged: {mcmc.converged}")
    print(f"  Notes: {mcmc.convergence_notes}")
    print(f"  Samples shape: {mcmc.samples.shape}")

    print("\nDeriving planet physics...")
    physics = derive_planet_physics(mcmc, stellar)

    def fmt(t):
        return f"{t[0]:.3f} +{t[2]:.3f} -{t[1]:.3f}"

    print(f"  Radius:      {fmt(physics.radius_earth)} R_earth")
    print(f"  Radius:      {fmt(physics.radius_jupiter)} R_jupiter")
    print(f"  Radius:      {fmt(physics.radius_km)} km")
    print(f"  Semi-major:  {fmt(physics.semi_major_axis_au)} AU")
    print(f"  T_eq:        {fmt(physics.equilibrium_temp)} K")
    print(f"  Insolation:  {fmt(physics.insolation)} S_earth")
    print(f"  (albedo={physics.albedo_assumed} assumed)")