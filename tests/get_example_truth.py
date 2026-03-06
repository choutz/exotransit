from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import re

print("Fetching Kepler Master List from NASA...")
koi_table = NasaExoplanetArchive.query_criteria(table="cumulative").to_pandas()

# One row per planet, only the columns we care about
cols = [
    'kepid', 'kepler_name',
    'koi_disposition',
    'koi_period',
    'koi_prad',          # planet radius (R_earth)
    'koi_ror',           # rp/R* — MCMC rp parameter
    'koi_impact',        # impact parameter b
    'koi_depth',         # transit depth (ppm)
    'koi_duration',      # transit duration (hours)
    'koi_incl',          # orbital inclination (deg)
    'koi_srad',          # stellar radius (R_sun)
    'koi_smass',         # stellar mass (M_sun)
    'koi_steff',         # stellar Teff (K)
    'koi_slogg',         # stellar logg
    'koi_smet',          # stellar [Fe/H]
]

df = koi_table[cols].copy()

# Extract star number from kepler_name for easy lookup
def extract_star_num(name):
    if pd.isna(name) or name == "": return None
    match = re.search(r'Kepler-(\d+)', str(name))
    return int(match.group(1)) if match else None

df['Kepler_#'] = df['kepler_name'].apply(extract_star_num)

# Keep only stars where EVERY planet is CONFIRMED
# i.e. drop any star that has even one non-CONFIRMED KOI
confirmed_stars = (
    df.groupby('kepid')['koi_disposition']
    .apply(lambda x: (x == 'CONFIRMED').all())
)
confirmed_kepids = confirmed_stars[confirmed_stars].index
df = df[df['kepid'].isin(confirmed_kepids)]

# Sort so planets from the same star are grouped, ordered by period
df = df.sort_values(['Kepler_#', 'koi_period']).reset_index(drop=True)

df = df.rename(columns={
    'kepler_name':   'Planet',
    'koi_disposition': 'Disposition',
    'koi_period':    'Period_d',
    'koi_prad':      'Radius_Rearth',
    'koi_ror':       'rp_over_Rstar',   # direct comparison to MCMC rp output
    'koi_impact':    'ImpactParam_b',   # direct comparison to MCMC b output
    'koi_depth':     'Depths_ppm',      # compare to BLS transit_depth * 1e6
    'koi_duration':  'Durations_h',     # compare to BLS best_duration * 24
    'koi_incl':      'Incl_deg',
    'koi_srad':      'Rstar_Rsun',
    'koi_smass':     'Mstar_Msun',
    'koi_steff':     'Teff_K',
    'koi_slogg':     'logg',
    'koi_smet':      'FeH',
})

print(f"{len(df)} confirmed planets across {df['kepid'].nunique()} fully-confirmed stars")
print(df.head(20).to_string())

df.to_csv(r"/Users/charlie/Desktop/job search 2026/exotransit/tests/examples.csv", index=False)
print("Saved to examples.csv")