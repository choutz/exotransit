import numpy as np
from exotransit.detection.multi_planet import find_all_planets
from tests.helpers import get_light_curve
from config import LOW

conf = LOW
"""
passed with MEDIUM conf: 
Kepler-7 perfect 1 planet
Kepler-144 got 3 / 2
kepler-9 gets 2/3 and has some uncertainty problems
kepler 15 worked but the posterior distributions  / uncertainties are horrible 
Kepler-11 found six

AFTER FIXES
kepler 7 still one planet but still wrong radius. seems very confident in the wrong radius. why?
Kepler-144 fixed  


"""


if __name__ == "__main__":

    # for tn in [7,11,80,9,15]:
    for tn in [144]:

        lc = get_light_curve(f"Kepler-{tn}", mission="Kepler", max_quarters=conf.max_quarters)
        print(f"  Target:   {lc.target_name}")
        print(f"  Points:   {len(lc.time)}")
        print(f"  Baseline: {lc.time[-1] - lc.time[0]:.1f} days")
        print(f"  Flux range: {lc.flux.min():.6f} – {lc.flux.max():.6f}")
        print(f"  Median uncertainty: {np.median(lc.flux_err):.6f}")

        print("\nSearching for multiple planets...")
        results, _, refined_lc = find_all_planets(
            lc,
            max_planets=conf.max_planets,
            min_period=conf.bls.min_period,
            max_period=conf.bls.max_period,
            max_period_grid_points=conf.bls.max_period_grid_points,
            debug_dir=r'/Users/charlie/Desktop/job search 2026/exotransit/tests/debug_masks',
        )
        print(f"Found {len(results)} planet candidate(s):")
        for i, result in enumerate(results):
            print(f"\n  Planet {i + 1}:")
            print(f"    Period:        {result.best_period:.4f} days")
            print(f"    Transit depth: {result.transit_depth:.6f} ± {result.depth_uncertainty:.6f}")
            print(f"    SDE:           {result.sde:.1f}")
            print(f"    SNR:           {result.snr:.1f}")
            print(f"    Reliable:      {result.is_reliable}")
            print(f"    Flags:         {result.reliability_flags}")
            print(f"    Aliases:       {result.aliases}")