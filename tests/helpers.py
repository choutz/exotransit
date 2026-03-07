"""
tests/helpers.py
Shared test utilities — not part of the exotransit package.
"""

import pickle
from pathlib import Path
from exotransit.pipeline.light_curves import fetch_light_curve
from exotransit.pipeline.observations import list_available_observations

CACHE_DIR = Path(__file__).parent / ".lc_cache"


def get_light_curve(target: str, mission: str, max_quarters: int, verbose: bool = True):
    """
    Fetch a stitched light curve, caching to disk to avoid re-downloading.
    Delete the cache file to force a fresh fetch.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = target.lower().replace(" ", "_") + f"_{mission.lower()}_{max_quarters}q"
    cache_file = CACHE_DIR / f"{key}.pkl"

    if cache_file.exists():
        print(f"Loading '{target}' from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # if verbose:
    #     obs = list_available_observations(target, mission=mission)
        # print(f"Found {len(obs)} observations:")
        # for o in obs:
        #     print(f"  {o}")

    # print("Fetching light curve from MAST...")
    lc = fetch_light_curve(target, mission=mission, max_quarters=max_quarters)
    with open(cache_file, "wb") as f:
        pickle.dump(lc, f)
    return lc