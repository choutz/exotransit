import lightkurve as lk


def list_available_observations(target: str, mission: str = "Kepler") -> list[dict]:
    """
    List available light curves for a target without downloading anything.
    Used to populate the sector/quarter selector in the UI.

    Returns list of dicts with keys: 'index', 'mission', 'year', 'exptime'.
    """
    search_result = lk.search_lightcurve(
        target,
        mission=mission if mission != "any" else None,
    )
    return [
        {
            "index": i,
            "mission": str(search_result[i].mission[0]),
            "year": str(search_result[i].year[0]),
            "exptime": str(search_result[i].exptime[0]),
        }
        for i in range(len(search_result))
    ]


if __name__ == '__main__':
    obs= list_available_observations("Kepler-5", mission="Kepler")
    for o in obs:
        print(o)