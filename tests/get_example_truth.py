from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import re

OUTPUT_DIR = "/Users/charlie/Desktop/job search 2026/exotransit/tests"


def fetch_koi_table() -> pd.DataFrame:
    """Fetch and clean the KOI cumulative table from NASA Exoplanet Archive."""
    print("Fetching Kepler Master List from NASA...")
    koi_table = NasaExoplanetArchive.query_criteria(table="cumulative").to_pandas()

    cols = [
        'kepid', 'kepler_name',
        'koi_disposition',
        'koi_period',
        'koi_prad',
        'koi_ror',
        'koi_impact',
        'koi_depth',
        'koi_duration',
        'koi_incl',
        'koi_srad',
        'koi_smass',
        'koi_steff',
        'koi_slogg',
        'koi_smet',
    ]

    df = koi_table[cols].copy()

    def extract_star_num(name):
        if pd.isna(name) or name == "": return None
        match = re.search(r'Kepler-(\d+)', str(name))
        return int(match.group(1)) if match else None

    df['Kepler_#'] = df['kepler_name'].apply(extract_star_num)

    df = df.rename(columns={
        'kepler_name':     'Planet',
        'koi_disposition': 'Disposition',
        'koi_period':      'Period_d',
        'koi_prad':        'Radius_Rearth',
        'koi_ror':         'rp_over_Rstar',
        'koi_impact':      'ImpactParam_b',
        'koi_depth':       'Depths_ppm',
        'koi_duration':    'Durations_h',
        'koi_incl':        'Incl_deg',
        'koi_srad':        'Rstar_Rsun',
        'koi_smass':       'Mstar_Msun',
        'koi_steff':       'Teff_K',
        'koi_slogg':       'logg',
        'koi_smet':        'FeH',
    })

    return df


def save_full_table(path: str = f"{OUTPUT_DIR}/examples.csv") -> pd.DataFrame:
    """Save one row per planet (all columns) to CSV."""
    df = fetch_koi_table()
    df = df.sort_values(['Kepler_#', 'Period_d']).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Saved full table: {len(df)} planets → {path}")
    return df


def save_star_summary(path: str = f"{OUTPUT_DIR}/examples_by_star.csv") -> pd.DataFrame:
    """
    Save one row per star (confirmed planets only).
    Columns: Kepler_#, kepid, n_planets, radii_Rearth, periods_d
    Only includes stars where every planet is CONFIRMED.
    """
    df = fetch_koi_table()

    # Keep only stars where every planet is CONFIRMED
    confirmed_stars = (
        df.groupby('kepid')['Disposition']
        .apply(lambda x: (x == 'CONFIRMED').all())
    )
    confirmed_kepids = confirmed_stars[confirmed_stars].index
    df = df[df['kepid'].isin(confirmed_kepids)]

    df = df.sort_values(['Kepler_#', 'Period_d'])

    summary = (
        df.groupby(['kepid', 'Kepler_#'])
        .apply(lambda g: pd.Series({
            'n_planets':    len(g),
            'radii_Rearth': ', '.join(g['Radius_Rearth'].round(2).astype(str)),
            'periods_d':    ', '.join(g['Period_d'].round(4).astype(str)),
        }))
        .reset_index()
        .sort_values('Kepler_#')
        .reset_index(drop=True)
    )

    summary.to_csv(path, index=False)
    print(f"Saved star summary: {len(summary)} confirmed stars → {path}")
    return summary


if __name__ == "__main__":
    # save_full_table()
    save_star_summary(r'/Users/charlie/Desktop/job search 2026/exotransit/tests/summary.csv')
