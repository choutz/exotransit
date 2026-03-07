# exotransit

A Python pipeline for detecting and characterizing exoplanet transits from Kepler photometry. Given a star name, it downloads the light curve, searches for periodic transit signals, fits a physical transit model using MCMC, and derives planet properties with full uncertainty propagation.

**Live app: [exotransit.streamlit.app](https://exotransit.streamlit.app/)**

Validated on stars with varying numbers of planets, including Kepler-97, Kepler-183, Kepler-203, Kepler-215, and Kepler-20.

---

## What it does

### 1. Light curve ingestion

Raw photometry is downloaded from NASA MAST via [lightkurve](https://lightkurve.github.io/lightkurve/). For Kepler targets, multiple quarters (~90 days each) are downloaded and stitched together. Each quarter is normalized independently before stitching — this removes inter-quarter flux jumps caused by the spacecraft rotating and landing the star on a different detector pixel.

Per-quarter preprocessing: remove NaNs → sigma-clip outliers (5σ) → normalize to unit flux.

After stitching, detrending runs in two passes:

**Pass 1 (rough):** A biweight filter runs across the full stitched baseline. The biweight is an M-estimator that down-weights outliers — including transit dips — so it does not absorb them into the trend. This produces a clean enough light curve for BLS period search, and preserves transit depth significantly better than Savitzky-Golay, which can attenuate signals whose duration approaches the filter window.

**Pass 2 (refined):** After BLS has identified all planet periods, the biweight runs again — but this time every point inside a transit window (3× the BLS-estimated duration, centered on each transit time) is explicitly set to NaN before the filter sees anything. The sliding window then has no transit flux to work with at those times: the trend at every point near a transit is estimated entirely from out-of-transit stellar continuum on either side. When the flux is divided by this trend, the full transit depth is recovered without any suppression.

This matters because even the biweight's down-weighting in Pass 1 is not the same as full exclusion. A transit dip receives low but non-zero weight in the biweight kernel, which slightly pulls the local trend estimate downward and partially fills in the dip. For small planets — where a few percent of depth suppression is a meaningful fraction of the signal — Pass 1 alone underestimates the true transit depth. The Pass 2 light curve is what transit model fitting receives.

### 2. Period search — Box Least Squares (BLS)

Transit signals are periodic and somewhat box-shaped: the planet blocks a fixed fraction of starlight for a fixed duration every orbit. Box Least Squares, or BLS (Kovács, Zucker & Mazeh 2002), is the standard algorithm for finding this — it tests thousands of period/duration combinations and scores each by how well a box model fits the phase-folded data.

**Period grid:** frequencies are spaced uniformly with step size df = min\_duration / baseline², chosen so a transit drifts by at most one duration when stepping between adjacent frequency grid points. This ensures uniform coverage of transit repetition rate rather than orbital period. The grid is capped at 200,000 points to stay within memory limits; periods are recovered as 1/frequency.

**Duration grid:** 12 values spaced geometrically from ~30 minutes to a maximum bounded by the minimum search period. Geometric spacing gives finer resolution at short durations where relative errors are larger.

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
| TCE-09 | No strong alias periods (P/2, P/3, 2P, 3P, etc.) |
| TCE-13 | Depth < 3% (deeper signals are almost always grazing eclipsing binaries) |
| TCE-14 | Not below Kepler's 30 ppm detection floor with marginal SNR |
| TCE-15 | Depth ≥ 60 ppm for long-cadence (30-min) data |

### 4. Multi-planet detection

After finding the first planet, its transit windows are median-filled and BLS runs again on the residuals. This iterates up to a configurable maximum. If a candidate period matches one already found (within 5%), it is skipped but the search continues — the search terminates only when a reliability flag trips on a new candidate, indicating no more credible signals remain.

> **Note:** Multi-planet detection is an active area of work. Balancing the reliability thresholds across diverse systems (different planet sizes, periods, stellar types) is tricky — a threshold that correctly rejects noise for a hot Jupiter around a quiet star may reject a real small planet around a noisier one. Diagnostic mask plots are generated at each iteration and are viewable via expandable sections in the web app, or written to disk by passing `debug_dir` to `find_all_planets`.

### 5. Transit model fitting — MCMC

Once all candidate periods are identified, a physical transit model is fit to each one. MCMC — Markov Chain Monte Carlo — is a statistical sampling method that maps out the full probability distribution over parameter values consistent with the data, rather than finding a single best-fit point. This gives asymmetric uncertainties and correctly captures parameter correlations (for example, between planet radius and impact parameter, which are degenerate for grazing transits).

The sampler is [emcee](https://emcee.readthedocs.io/) (Foreman-Mackey et al. 2013), an ensemble sampler that runs many parallel walkers through parameter space and collects samples from the posterior. The transit model is [batman](https://lkreidberg.github.io/batman/) (Mandel & Agol 2002). The free parameters are:

- `t0` — transit center time (days)
- `rp` — planet-to-star radius ratio (Rp/R★)
- `b` — impact parameter (0 = central transit, 1 = grazing)

Limb darkening is fixed to quadratic coefficients interpolated from Claret (2011) Kepler tables using the star's Teff, log g, and metallicity from the NASA Exoplanet Archive. The semi-major axis is derived from Kepler's 3rd law using the archive stellar mass rather than being left as a free parameter. This couples the transit duration to the orbital physics, which is the physically correct constraint.

**Exposure time smearing correction:** Kepler long-cadence data records the *average* brightness over a 30-minute window. During ingress and egress the stellar disk is only partially covered, so brightness is changing rapidly — but the detector averages over the whole 30 minutes regardless. A transit model evaluated at a single instant per cadence would predict a sharp ingress/egress profile, while the actual data shows it smoothed. The correction supersamples each 30-minute exposure into many sub-cadence time points, evaluates the model at each, then averages before comparing to the data — matching what the detector physically measures. Without this, the optimizer shrinks `rp` to compensate for the mismatch in ingress sharpness, producing a systematically underestimated planet radius.

### 6. Physical parameter derivation

MCMC posteriors are combined with stellar parameters from the NASA Exoplanet Archive to derive:

- **Planet radius** (Earth radii, Jupiter radii, km)
- **Semi-major axis** (AU, from Kepler's 3rd law)
- **Equilibrium temperature** (K, assuming Bond albedo = 0.3 — roughly Earth- or Jupiter-like; the exact choice shifts temperature by ≲20% and is not particularly meaningful without a real atmosphere model)
- **Insolation** (relative to Earth)

Uncertainty propagation is done by sampling: stellar radius, mass, and temperature are each drawn from Gaussians centered on the archive values with their reported uncertainties, then combined with the MCMC posterior samples point-by-point. Derived quantities are computed for each combination and percentiles of the resulting distributions give the reported medians and 1σ bounds. This correctly handles non-Gaussian posteriors and parameter correlations without Gaussian error propagation assumptions.

---

## Architecture

```
pipeline/
  light_curves.py      — download, stitch, biweight-detrend (MAST via lightkurve);
                         redetrend_with_mask() runs Pass 2 after BLS
  observations.py      — list available quarters/sectors without downloading
  helpers.py           — flux_err extraction

detection/
  bls.py               — BLS period search, BLSResult dataclass
  result_evaluation.py — TCE-style reliability vetting
  multi_planet.py      — iterative multi-planet search, transit masking,
                         Pass 2 redetrend; returns (planets, mask_data, refined_lc)

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
  steps/               — Streamlit UI (step0–step4)
```

---

## Dependencies

```
lightkurve        — light curve download, stitching, and preprocessing
emcee             — MCMC ensemble sampler
batman-package    — Mandel-Agol analytic transit light curve models
astropy           — time/unit handling, BLS implementation
numpy             — numerical core
pandas            — data handling (stellar parameter queries, LD tables)
plotly            — interactive visualizations
streamlit         — web app
requests          — NASA Exoplanet Archive TAP queries
tqdm              — progress bars in CLI pipeline runs
wotan             — detrending (lightkurve dependency)
```

---

## Running

```bash
streamlit run app.py
```

Configuration profiles (`MEDIUM` for Streamlit Cloud, `FULL` for local) are in `config.py`.

---

## Streamlit app

The pipeline is wrapped in a step-by-step Streamlit app for interactive exploration of Kepler targets.

**Step 1 — Raw Data**
Enter a Kepler star name (e.g. `Kepler-11`). The app downloads the stitched light curve from MAST and queries the NASA Exoplanet Archive for stellar parameters (radius, mass, Teff, log g, metallicity). It shows the raw and detrended flux side by side, with quarter boundaries and >3σ transit candidates highlighted.

**Step 2 — Transit Detection**
BLS runs iteratively. A status line updates after each planet is found — e.g. *"Found one planet (P = 4.886 d) — checking for more…"* — so it's clear the app is alive during long searches. After all planets are found, the light curve is re-detrended with transit windows explicitly excluded (Pass 2), ensuring transit model fitting in the next step works from uncontaminated stellar continuum. For each detection the app shows the BLS power spectrum (log period axis, aliases marked) and the phase-folded light curve, plus an expandable masking diagnostic showing which points were median-filled before the next BLS pass.

**Step 3 — MCMC Fitting**
For each detected planet, emcee runs a fit with limb darkening fixed to the Claret (2011) values retrieved in Step 1. Shows the phase fold with model overlay, MCMC spaghetti plot (posterior draws), corner plot in physical units, and per-parameter posterior histograms.

**Step 4 — Results**
Physical parameters (radius, semi-major axis, equilibrium temperature, insolation) with 1σ uncertainties. An orrery shows the orbital architecture to scale, and a bubble chart plots the planets against Solar System benchmarks with a habitable zone overlay.

The app is designed to run within Streamlit Community Cloud's constraints (1 GB RAM, 1 vCPU), which drives the `MEDIUM` config profile — a coarser period grid and fewer MCMC steps than the local `FULL` profile.

---

## Known limitations and open problems

The detection pipeline works well on clean, well-separated signals, but generalizing to arbitrary targets surfaces a range of astrophysical and algorithmic challenges:

**Transit masking in multi-planet systems**
After masking a detected planet's transits, BLS occasionally re-detects the same period on the next iteration, indicating the mask didn't fully suppress the signal. The mask window is currently 3× the BLS-estimated duration, but BLS duration estimates can underestimate true duration, especially for grazing or long-duration transits.

**Reliability threshold generalizability**
The TCE-style vetting thresholds were tuned against the Kepler pipeline's assumptions (quiet FGK stars, long-cadence photometry, ~4-year baselines). They don't generalize cleanly to:
- Small planets with shallow depths where per-transit SNR is marginal
- Short-baseline TESS observations where only 2–3 transit windows exist
- Active stars with coronal variability that inflates the noise floor

**Transit timing variations (TTVs)**
Planets in or near mean-motion resonance have transit times that shift by minutes to hours from orbit to orbit due to gravitational interactions. BLS assumes perfectly periodic transits and smears out TTV signals when phase-folding, reducing detection SNR. Handling TTVs properly requires a separate dynamical modeling step not currently implemented.

**Stellar activity**
Starspots, flares, and coronal mass ejections all produce flux variations that can alias into the BLS period grid or corrupt the transit depth estimate. The biweight filter removes slow trends but is not designed to handle short-duration flares or rotationally-modulated spot patterns.

**Eclipsing binary contamination**
Eclipsing binaries are a major source of false positives in transit surveys. The pipeline catches the most blatant cases via TCE-04 (duty cycle) and TCE-13 (depth > 3%), but anything more subtle — odd/even depth alternation, secondary eclipses at phase 0.5, background EBs blended inside the photometric aperture — requires centroid motion analysis, radial velocity follow-up, or multi-wavelength photometry to rule out. If the target is actually an eclipsing binary that passes the coarse cuts, the pipeline will fit it and give you numbers. The numbers will be wrong.

**Uncertainty underestimation**
The detrended light curve is treated as fixed truth by the transit model fitting. Any uncertainty in where the stellar trend was — whether the filter slightly suppressed a transit or misidentified a stellar oscillation as continuum — does not propagate into the parameter posteriors. The reported error bars represent photon noise only. Fully propagating detrending uncertainty requires simultaneous GP + transit modeling, which is computationally prohibitive for real-time use.

**Why real planet discoveries get their own papers**

The above list is not exhaustive. Real planet confirmation involves independent TCE vetting by multiple reviewers, centroid motion analysis to rule out background EBs, spectroscopic stellar characterization, dynamical modeling for multi-planet systems, and statistical false positive probability calculations. This pipeline runs the full detection and characterization sequence — period search, reliability vetting, transit model fitting, and physical parameter derivation — and produces results consistent with known systems. What it does not do is confirm planets: that requires the follow-up observations and analysis that real discovery papers are built around.

---

## References

- Kovács, Zucker & Mazeh (2002) — Box Least Squares algorithm
- Jenkins et al. (2010) — Kepler TCE pipeline and SDE threshold
- Mandel & Agol (2002) — Analytic transit light curve models (batman)
- Claret (2011) — Kepler quadratic limb darkening tables
- Foreman-Mackey et al. (2013) — emcee ensemble MCMC sampler
