import logging
import numpy as np
from exotransit.detection.search import run_bls
from exotransit.mcmc.fit import run_mcmc
from tests.helpers import get_light_curve

logging.disable(logging.INFO)


if __name__ == "__main__":
    # Use Kepler-5b — single clean hot Jupiter, strong signal, easy MCMC target
    lc = get_light_curve("Kepler-5 b", mission="Kepler", max_quarters=8)

    print("Running BLS...")
    bls = run_bls(lc, min_period=2.0, max_period=10.0)
    print(f"  BLS period:  {bls.best_period:.4f} days")
    print(f"  BLS depth:   {bls.transit_depth:.6f} ± {bls.depth_uncertainty:.6f}")
    print(f"  BLS rp est:  {np.sqrt(bls.transit_depth):.4f}")

    print("\nRunning MCMC...")
    mcmc = run_mcmc(lc, bls, n_walkers=32, n_steps=5000, n_burnin=500)

    print(f"\nMCMC Results:")
    print(f"  period:  {mcmc.period:.5f} days (fixed from BLS)")
    print(f"  t0:      {mcmc.t0_med:.5f} +{mcmc.t0_err[1]:.5f} -{mcmc.t0_err[0]:.5f}")
    print(f"  rp:      {mcmc.rp_med:.5f} +{mcmc.rp_err[1]:.5f} -{mcmc.rp_err[0]:.5f}")
    print(f"  b:       {mcmc.b_med:.4f} +{mcmc.b_err[1]:.4f} -{mcmc.b_err[0]:.4f}")
    print(f"  u1:      {mcmc.u1_med:.4f} +{mcmc.u1_err[1]:.4f} -{mcmc.u1_err[0]:.4f}")
    print(f"  u2:      {mcmc.u2_med:.4f} +{mcmc.u2_err[1]:.4f} -{mcmc.u2_err[0]:.4f}")
    print(f"  depth:   {mcmc.depth_med:.6f} +{mcmc.depth_err[1]:.6f} -{mcmc.depth_err[0]:.6f}")
    print(f"\n  Acceptance fraction: {mcmc.acceptance_fraction:.3f}")
    print(f"  Converged: {mcmc.converged}")
    print(f"  Notes: {mcmc.convergence_notes}")
    print(f"  Samples shape: {mcmc.samples.shape}")
