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
from exotransit.detection.search import run_bls
from exotransit.mcmc.fit import run_mcmc
from exotransit.physics.stellar import query_stellar_params
from exotransit.physics.planets import derive_planet_physics
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
CACHE_DIR = Path(__file__).parent / ".lc_cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

TARGET = "Kepler-5 b"
MISSION = "Kepler"
MAX_QUARTERS = 8


def load_cache(name):
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        print(f"  Loading {name} from cache...")
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

    print("BLS...")
    bls = load_cache("viz_bls")
    if bls is None:
        bls = run_bls(lc, min_period=2.0, max_period=10.0)
        save_cache("viz_bls", bls)

    print("MCMC...")
    mcmc = load_cache("viz_mcmc")
    if mcmc is None:
        mcmc = run_mcmc(lc, bls, n_walkers=32, n_steps=6000, n_burnin=500)
        save_cache("viz_mcmc", mcmc)

    print("Stellar params...")
    stellar = load_cache("viz_stellar")
    if stellar is None:
        stellar = query_stellar_params(TARGET)
        save_cache("viz_stellar", stellar)

    print("Planet physics...")
    physics = load_cache("viz_physics")
    if physics is None:
        physics = derive_planet_physics(mcmc, stellar)
        save_cache("viz_physics", physics)

    print("\nGenerating plots...")
    save_plot(plot_light_curve_pipeline(lc), "01_pipeline")
    save_plot(plot_bls_power_spectrum(bls), "02_power_spectrum")
    save_plot(plot_phase_fold(bls, mcmc), "03_phase_fold")
    save_plot(plot_mcmc_spaghetti(bls, mcmc), "04_spaghetti")
    save_plot(plot_corner(mcmc), "05_corner")
    save_plot(plot_posterior_histograms(mcmc), "06_posteriors")
    save_plot(plot_orrery([bls], [physics], lc.target_name), "07_orrery")
    save_plot(plot_planet_comparison([bls], [physics], lc.target_name), "08_comparison")

    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Open any .html file in a browser to inspect.")