from exotransit.pipeline.fetch import fetch_light_curve, list_available_observations

if __name__ == "__main__":
    print("Testing list_available_observations...")
    obs = list_available_observations("Kepler-90", mission="Kepler")
    print(f"Found {len(obs)} observations:")
    for o in obs:
        print(f"  {o}")

    print("\nTesting fetch_light_curve...")
    lc = fetch_light_curve("Kepler-90", mission="Kepler")
    print(f"Success!")
    print(f"  Target:  {lc.target_name}")
    print(f"  Mission: {lc.mission}")
    print(f"  Points:  {len(lc.time)}")
    print(f"  Flux range: {lc.flux.min():.6f} – {lc.flux.max():.6f}")
    print(f"  Median uncertainty: {lc.flux_err.mean():.6f}")