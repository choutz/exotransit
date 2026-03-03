import pickle
import numpy as np
from pathlib import Path
from exotransit.pipeline.fetch import fetch_stitched_light_curve, list_available_observations
from exotransit.detection.search import run_bls, find_all_planets

CACHE_DIR = Path(__file__).parent / ".lc_cache"


def _cache_key(target: str, mission: str, max_quarters: int) -> str:
    return target.lower().replace(" ", "_") + f"_{mission.lower()}_{max_quarters}q"


def get_light_curve(target: str, mission: str, max_quarters: int):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(target, mission, max_quarters)}.pkl"

    if cache_file.exists():
        print(f"Loading '{target}' from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Querying available observations for {target}...")
    obs = list_available_observations(target, mission=mission)
    print(f"Found {len(obs)} observations:")
    for o in obs:
        print(f"  {o}")

    print("Fetching light curve from MAST...")
    lc = fetch_stitched_light_curve(target, mission=mission, max_quarters=max_quarters)
    with open(cache_file, "wb") as f:
        pickle.dump(lc, f)
    return lc


if __name__ == "__main__":
    lc = get_light_curve("Kepler-11", mission="Kepler", max_quarters=17)
    print(f"  Target:   {lc.target_name}")
    print(f"  Points:   {len(lc.time)}")
    print(f"  Baseline: {lc.time[-1] - lc.time[0]:.1f} days")
    print(f"  Flux range: {lc.flux.min():.6f} – {lc.flux.max():.6f}")
    print(f"  Median uncertainty: {np.median(lc.flux_err):.6f}")

    print("\nSearching for multiple planets...")
    results = find_all_planets(lc, max_planets=6, min_period=2.0, max_period=120.0)
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