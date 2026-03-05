# exotransit

A Python pipeline for detecting and characterizing exoplanet transits from Kepler and TESS photometry. Given a star name, it downloads the light curve, searches for periodic transit signals, fits a physical model using MCMC, and derives planet properties with full uncertainty propagation.

Validated on Kepler-5b (single planet, hot Jupiter), Kepler-7b, and Kepler-11 (six-planet system).

---

## What it does

### 1. Light curve ingestion

Raw photometry is downloaded from NASA MAST via [lightkurve](https://docs.lightkurve.org/). For Kepler targets, multiple quarters (~90 days each) are downloaded and stitched together. Each quarter is detrended and normalized independently before stitching — this removes inter-quarter flux jumps caused by the spacecraft rotating and landing the star on a different detector pixel.

Preprocessing per quarter: remove NaNs → sigma-clip outliers (5σ) → Savitzky-Golay detrend → normalize to unit flux.

### 2. Period search — Box Least Squares (BLS)

Transit signals are periodic and box-shaped: the planet blocks a fixed fraction of starlight for a fixed duration every orbit. BLS (Kovács, Zucker & Mazeh 2002) is the standard algorithm for finding this — it tests thousands of period/duration combinations and scores each by how well a box model fits the phase-folded data.

The period grid is spaced linearly in frequency rather than period, which gives uniform sampling of transit repetition rate. A sharp peak in the power spectrum at some period P means a repeating box-shaped dip every P days.

Two quality metrics are computed:
- **SDE** (Signal Detection Efficiency): how many standard deviations above the noise floor the peak stands. SDE > 7.1 is the Kepler pipeline's minimum threshold (Jenkins et al. 2010).
- **SNR**: signal-to-noise of the transit depth itself.

### 3. Reliability vetting

Detections are vetted against a set of flags modeled on NASA's Threshold Crossing Event (TCE) pipeline:

| Flag | Test |
|------|------|
| TCE-01/02 | SDE and SNR above 7.1σ floor |
| TCE-03 | Per-transit SNR > 1.5σ (signal should grow as √N transits) |
| TCE-04 | Duration/period ratio < 0.1 (eclipsing binary discriminator) |
| TCE-05/06 | ≥ 3 in-transit points, ≥ 70% of expected cadences covered |
| TCE-07 | Depth > 3σ above noise floor |
| TCE-08 | ≥ 3 complete transit windows in the baseline |
| TCE-09 | No strong alias periods (P/2, 2P, etc.) |
| TCE-10 | Odd/even transit depth symmetry (alternating depths → eclipsing binary) |
| TCE-13 | Depth < 3% (deeper signals are almost always grazing eclipsing binaries) |
| TCE-14 | Not below Kepler's 30 ppm reliable detection floor |

### 4. Multi-planet detection

After finding the first planet, its transits are masked out and BLS is run again on the residuals. This iterates up to a configurable maximum. If a new detection duplicates a period already found (within 5%), the masking failed and the search stops.

> **Note:** Multi-planet detection is an active area of work. Balancing the reliability thresholds across diverse systems (different planet sizes, periods, stellar types) is tricky — a threshold that correctly rejects noise for a hot Jupiter around a quiet star may reject a real small planet around a noisier one. Diagnostic mask plots are generated at each iteration (pass `debug_dir` to `find_all_planets`) to make the masking behavior inspectable.

### 5. Transit model fitting — MCMC

Once a candidate period is identified, a physical transit model is fit using MCMC ([emcee](https://emcee.readthedocs.io/) + [batman](https://lkreidberg.github.io/batman/)). The free parameters are:

- `t0` — transit center time (days)
- `rp` — planet-to-star radius ratio (Rp/R★)
- `b` — impact parameter (0 = central transit, 1 = grazing)

Limb darkening is fixed to quadratic coefficients interpolated from Claret (2011) Kepler tables using the star's Teff, log g, and metallicity from the NASA Exoplanet Archive. Fitting limb darkening freely requires higher photometric precision than most single-system fits can provide, and Claret tables are well-validated for Kepler targets.

The semi-major axis is derived from Kepler's 3rd law using the archive stellar mass rather than being left as a free parameter. This couples the transit duration to the orbital physics, which is the physically correct constraint.

For long-cadence Kepler data (30-minute integrations), the transit model is supersampled and integrated over each exposure window. Without this correction, the box-shaped ingress/egress in the model doesn't match the smoothed shape that 30-minute averaging produces, and `rp` is systematically underestimated.

### 6. Physical parameter derivation

MCMC posteriors are combined with stellar parameters from the NASA Exoplanet Archive to derive:

- **Planet radius** (Earth radii, Jupiter radii, km)
- **Semi-major axis** (AU, from Kepler's 3rd law)
- **Equilibrium temperature** (K, assuming Bond albedo = 0.3)
- **Insolation** (relative to Earth)

Uncertainty propagation is done by sampling: stellar radius, mass, and temperature uncertainties are drawn from Gaussians and combined with the MCMC posterior at each sample, then percentiles are read off the resulting distributions. This correctly handles non-Gaussian posteriors without Gaussian error propagation assumptions.

---

## Architecture

```
pipeline/
  light_curves.py      — download, detrend, stitch (MAST via lightkurve)
  observations.py      — list available quarters/sectors without downloading
  helpers.py           — Savitzky-Golay window sizing, flux_err extraction

detection/
  bls.py               — BLS period search, BLSResult dataclass
  result_evaluation.py — TCE-style reliability vetting
  multi_planet.py      — iterative multi-planet search with transit masking

mcmc/
  fit_mcmc.py          — emcee sampler, MCMCResult dataclass
  helpers.py           — batman transit model, log posterior

physics/
  stars.py             — NASA Exoplanet Archive TAP/ADQL queries
  planets.py           — physical parameter derivation with uncertainty sampling
  limb_darkening.py    — Claret (2011) LD table interpolation

viz/
  plots.py             — Plotly figures (light curve, BLS spectrum, phase fold,
                         MCMC spaghetti, corner plot, orrery, planet comparison)

app/
  steps/               — Streamlit UI
```

---

## Dependencies

```
lightkurve        — light curve download and preprocessing
emcee             — MCMC ensemble sampler
batman-package    — transit light curve models
astropy           — time/unit handling, BLS
numpy, scipy      — numerical core
plotly            — interactive visualizations
streamlit         — web app
requests          — NASA Exoplanet Archive queries
tqdm, matplotlib  — progress display, diagnostic plots
```

---

## Running

**Multi-planet search:**
```bash
python tests/test_fetch_and_search.py
```

**Full pipeline with MCMC:**
```bash
python tests/test_mcmc.py
```

**Streamlit app:**
```bash
streamlit run app.py
```

Configuration profiles (`MEDIUM` for Streamlit Cloud, `FULL` for local) are in `config.py`.

---

## References

- Kovács, Zucker & Mazeh (2002) — Box Least Squares algorithm
- Jenkins et al. (2010) — Kepler TCE pipeline and SDE threshold
- Mandel & Agol (2002) — Analytic transit light curve models (batman)
- Claret (2011) — Kepler quadratic limb darkening tables
- Foreman-Mackey et al. (2013) — emcee ensemble MCMC sampler