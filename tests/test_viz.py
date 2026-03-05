"""
tests/test_viz.py
Smoke tests for all visualization functions — verifies they run
without error and writes HTML outputs for visual inspection.
"""
import logging
import pickle
import numpy as np
from pathlib import Path
from tests.helpers import get_light_curve
from exotransit.detection.multi_planet import find_all_planets
from exotransit.mcmc.fit_mcmc import run_mcmc
from exotransit.physics.stars import query_stellar_params
from exotransit.physics.planets import derive_planet_physics
from exotransit.physics.limb_darkening import get_limb_darkening
from exotransit.viz.plots import (
    plot_light_curve_pipeline,
    plot_bls_power_spectrum,
    plot_phase_fold,
    plot_mcmc_spaghetti,
    plot_corner,
    plot_posterior_histograms,
    plot_orrery,
    plot_planet_comparison,
)

logging.disable(logging.INFO)
OUTPUT_DIR = Path(__file__).parent / "viz_output"
CACHE_DIR  = Path(__file__).parent / ".lc_cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

TARGET       = "Kepler-11"
MISSION      = "Kepler"
MAX_QUARTERS = 17


def load_cache(name):
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        print(f"  Loading '{name}' from cache...")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(name, obj):
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def save_plot(fig, name):
    path = OUTPUT_DIR / f"{name}.html"
    fig.write_html(str(path))
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    print("Loading light curve...")
    lc = get_light_curve(TARGET, mission=MISSION, max_quarters=MAX_QUARTERS)

    print("Stellar params...")
    stellar = load_cache(f"{TARGET}_stellar")
    if stellar is None:
        stellar = query_stellar_params(TARGET)
        save_cache(f"{TARGET}_stellar", stellar)
    print(f"  {stellar.name}: R*={stellar.radius:.3f} R_sun, "
          f"M*={stellar.mass:.3f} M_sun, Teff={stellar.teff:.0f} K")

    print("Limb darkening...")
    ld = get_limb_darkening(
        teff=stellar.teff,
        logg=stellar.logg,
        metallicity=stellar.metallicity,
    )
    print(f"  u1={ld.u1:.4f}±{ld.u1_sigma:.4f}, u2={ld.u2:.4f}±{ld.u2_sigma:.4f}")

    print("BLS (multi-planet)...")
    all_bls = load_cache(f"{TARGET}_bls")
    if all_bls is None:
        all_bls = find_all_planets(lc, max_planets=6, min_period=2.0, max_period=120.0)
        save_cache(f"{TARGET}_bls", all_bls)
    print(f"  Found {len(all_bls)} planet(s):")
    for i, b in enumerate(all_bls):
        print(f"    P{i+1}: {b.best_period:.4f}d, depth={b.transit_depth:.6f}, "
              f"SDE={b.sde:.1f}, SNR={b.snr:.1f}")

    print("MCMC (one per planet)...")
    all_mcmc = load_cache(f"{TARGET}_mcmc")
    if all_mcmc is None:
        all_mcmc = []
        for i, bls in enumerate(all_bls):
            print(f"  Planet {i+1} (P={bls.best_period:.4f}d)...")
            mcmc = run_mcmc(
                lc, bls,
                n_walkers=32, n_steps=5000, n_burnin=500,
                u1=ld.u1, u2=ld.u2,
                stellar_mass=stellar.mass,
                stellar_radius=stellar.radius,
            )
            print(f"    rp={mcmc.rp_med:.4f}, b={mcmc.b_med:.4f}, "
                  f"R={mcmc.radius_earth_med:.2f} R_earth, "
                  f"inc={mcmc.inclination_med:.2f}°, "
                  f"acc={mcmc.acceptance_fraction:.3f}, "
                  f"converged={mcmc.converged}")
            all_mcmc.append(mcmc)
        save_cache(f"{TARGET}_mcmc", all_mcmc)

    print("Planet physics...")
    all_physics = load_cache(f"{TARGET}_physics")
    if all_physics is None:
        all_physics = [derive_planet_physics(mcmc, stellar) for mcmc in all_mcmc]
        save_cache(f"{TARGET}_physics", all_physics)

    print("\nGenerating plots...")

    # Pipeline plot — one per target
    save_plot(plot_light_curve_pipeline(lc), "01_pipeline")

    # Per-planet plots
    for i, (bls, mcmc) in enumerate(zip(all_bls, all_mcmc)):
        n = i + 1
        save_plot(plot_bls_power_spectrum(bls),      f"02_power_spectrum_p{n}")
        save_plot(plot_phase_fold(bls, mcmc),         f"03_phase_fold_p{n}")
        save_plot(plot_mcmc_spaghetti(bls, mcmc),     f"04_spaghetti_p{n}")
        save_plot(plot_corner(mcmc),                  f"05_corner_p{n}")
        save_plot(plot_posterior_histograms(mcmc),    f"06_posteriors_p{n}")

    # System-level plots — all planets together
    save_plot(plot_orrery(all_bls, all_physics, lc.target_name),          "07_orrery")
    save_plot(plot_planet_comparison(all_bls, all_physics, lc.target_name), "08_comparison")

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Open any .html file in a browser to inspect.")