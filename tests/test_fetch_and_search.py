from exotransit.pipeline.fetch import fetch_stitched_light_curve, list_available_observations
from exotransit.detection.search import run_bls
import numpy as np

if __name__ == "__main__":
    print("Testing list_available_observations...")
    obs = list_available_observations("Kepler-5 b", mission="Kepler")
    print(f"Found {len(obs)} observations:")
    for o in obs:
        print(f"  {o}")

    print("Testing fetch_stitched_light_curve...")
    lc = fetch_stitched_light_curve("Kepler-5 b", mission="Kepler", max_quarters=8)
    print(f"Success!")
    print(f"  Target:   {lc.target_name}")
    print(f"  Points:   {len(lc.time)}")
    print(f"  Baseline: {lc.time[-1] - lc.time[0]:.1f} days")
    print(f"  Flux range: {lc.flux.min():.6f} – {lc.flux.max():.6f}")
    print(f"  Median uncertainty: {np.median(lc.flux_err):.6f}")

    print("\nTesting BLS period search...")
    result = run_bls(lc, min_period=2.0, max_period=30.0)
    print(f"  Best period:   {result.best_period:.4f} days")
    print(f"  Transit depth: {result.transit_depth:.6f} ± {result.depth_uncertainty:.6f}")
    print(f"  SDE:           {result.sde:.1f}")
    print(f"  SNR:           {result.snr:.1f}")
    print(f"  Reliable:      {result.is_reliable}")
    print(f"  Flags:         {result.reliability_flags}")
    print(f"  Aliases:       {result.aliases}")
