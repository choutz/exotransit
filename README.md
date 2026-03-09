# exotransit

A transit detection pipeline built from scratch to find exoplanets and understand the full stack required to do it: signal processing on real photometry data, period search via Box Least Squares, Bayesian parameter estimation with Markov Chain Monte Carlo (MCMC), and physical characterization from first principles. Partly a skills refresh, partly an excuse to work on some cool signal processing and time series problems, and partly an exercise in debugging and testing the places where physics, statistics, and numerical methods interact badly. The goal was never to match or replace NASA tooling.

Given a Kepler star name, the pipeline downloads the light curve, searches for periodic transit signals, fits a physical transit model using MCMC, and derives planet properties with full uncertainty propagation.

**Live app: [exotransit.streamlit.app](https://exotransit.streamlit.app/)**

Validated on stars with varying numbers of planets, including Kepler-97, Kepler-183, Kepler-203, Kepler-215, and Kepler-20.

---

## What it does

### 1. Light curve ingestion

Raw photometry is downloaded from NASA MAST via [lightkurve](https://lightkurve.github.io/lightkurve/). For Kepler targets, multiple quarters (~90 days each) are downloaded and stitched together. Each quarter is normalized independently before stitching, which removes inter-quarter flux jumps caused by the spacecraft rotating and landing the star on a different detector pixel.

Per-quarter preprocessing: remove NaNs → sigma-clip outliers (5σ) → normalize to unit flux.

After stitching, detrending runs in two passes:

**Pass 1 (rough):** A biweight filter runs across the full stitched baseline. The biweight is an M-estimator that down-weights outliers (including transit dips) so it does not absorb them into the trend. This produces a clean enough light curve for BLS period search, and preserves transit depth significantly better than Savitzky-Golay, which can attenuate signals whose duration approaches the filter window.

**Pass 2 (refined):** After BLS has identified all planet periods, the biweight runs again, but this time every point inside a transit window (3× the BLS-estimated duration, centered on each transit time) is explicitly set to NaN before the filter sees anything. The sliding window then has no transit flux to work with at those times: the trend at every point near a transit is estimated entirely from out-of-transit stellar continuum on either side. When the flux is divided by this trend, the full transit depth is recovered without any suppression.

This matters because even the biweight's down-weighting in Pass 1 is not the same as full exclusion. A transit dip receives low but non-zero weight in the biweight kernel, which slightly pulls the local trend estimate downward and partially fills in the dip. For small planets, where a few percent of depth suppression is a meaningful fraction of the signal, Pass 1 alone underestimates the true transit depth. The Pass 2 light curve is what transit model fitting receives.

### 2. Period search — Box Least Squares (BLS)

Transit signals are periodic and box-shaped: the planet blocks a fixed fraction of starlight for a fixed duration every orbit. Box Least Squares, or BLS (Kovács, Zucker & Mazeh 2002), is the standard algorithm for finding this. It tests thousands of period/duration combinations and scores each by how well a box model fits the phase-folded data.

**Period grid:** frequencies are spaced uniformly with step size df = min\_duration / baseline², chosen so a transit drifts by at most one duration when stepping between adjacent frequency grid points. This ensures uniform coverage of transit repetition rate rather than orbital period. The grid is capped at 200,000 points to stay within memory limits; periods are recovered as 1/frequency.

**Duration grid:** 12 values spaced geometrically from ~30 minutes to a maximum bounded by the minimum search period. Geometric spacing gives finer resolution at short durations where relative errors are larger.

Two quality metrics are computed:
- **SDE** (Signal Detection Efficiency): how many standard deviations above the noise floor the peak stands. SDE > 7.1 is the Kepler pipeline's minimum threshold (Jenkins et al. 2010).
- **SNR**: signal-to-noise of the transit depth itself.

### 3. Reliability vetting

This is the part of the pipeline that changed the most during development, and the story of why it changed is worth telling in full — see [The vetting problem: NASA's approach vs. the right approach for this app](#the-vetting-problem-nasas-approach-vs-the-right-approach-for-this-app) below.

The current vetting uses a decision tree classifier trained on labeled BLS candidates from 248 Kepler targets. The tree takes BLS output features — SDE, SNR, transit duration, duty cycle, in-transit data points, expected transit windows, per-transit SNR, and transit point coverage ratio — and classifies each candidate as real or false positive. On the training data it achieves 97.7% precision and 92.9% recall (F1 = 0.952), eliminating 838 of 852 false positives while retaining 591 of 636 real planets.

**Why not the NASA TCE framework?** See the section below. The short answer is: the TCE pipeline was designed for a different problem, with different failure modes, and optimizes for a metric that is exactly wrong for an app where results go directly to a user.

### 4. Multi-planet detection

After finding the first planet, its transit windows are median-filled and BLS runs again on the residuals. This iterates up to a configurable maximum. If a candidate period matches one already found (within 5%), it is skipped but the search continues — the search terminates only when the decision tree rejects a new candidate, indicating no more credible signals remain. Diagnostic mask plots are generated at each iteration and are viewable via expandable sections in the web app, or written to disk by passing `debug_dir` to `find_all_planets`.

### 5. Transit model fitting — MCMC

Once all candidate periods are identified, a physical transit model is fit to each one. MCMC (Markov Chain Monte Carlo) is a statistical sampling method that maps out the full probability distribution over parameter values consistent with the data, rather than finding a single best-fit point. This gives asymmetric uncertainties and correctly captures parameter correlations (for example, between planet radius and impact parameter, which are degenerate for grazing transits).

The sampler is [emcee](https://emcee.readthedocs.io/) (Foreman-Mackey et al. 2013), an ensemble sampler that runs many parallel walkers through parameter space and collects samples from the posterior. The transit model is [batman](https://lkreidberg.github.io/batman/) (Mandel & Agol 2002). The free parameters are:

- `t0` — transit center time (days)
- `rp` — planet-to-star radius ratio (Rp/R★)
- `b` — impact parameter (0 = central transit, 1 = grazing)

Limb darkening is fixed to quadratic coefficients interpolated from Claret (2011) Kepler tables using the star's Teff, log g, and metallicity from the NASA Exoplanet Archive. The semi-major axis is derived from Kepler's 3rd law using the archive stellar mass rather than being left as a free parameter. This couples the transit duration to the orbital physics, which is the physically correct constraint.

**Exposure time smearing correction:** Kepler long-cadence data records the *average* brightness over a 30-minute window. During ingress and egress the stellar disk is only partially covered, so brightness is changing rapidly — but the detector averages over the whole 30 minutes regardless. A transit model evaluated at a single instant per cadence would predict a sharp ingress/egress profile, while the actual data shows it smoothed. The correction supersamples each 30-minute exposure into many sub-cadence time points, evaluates the model at each, then averages before comparing to the data, matching what the detector physically measures. Without this, the optimizer shrinks `rp` to compensate for the mismatch in ingress sharpness, producing a systematically underestimated planet radius.

### 6. Physical parameter derivation

MCMC posteriors are combined with stellar parameters from the NASA Exoplanet Archive to derive:

- **Planet radius** (Earth radii, Jupiter radii, km)
- **Semi-major axis** (AU, from Kepler's 3rd law)
- **Equilibrium temperature** (K, assuming Bond albedo = 0.3 — roughly Earth- or Jupiter-like; the exact choice shifts temperature by ≲20% and is not particularly meaningful without a real atmosphere model)
- **Insolation** (relative to Earth)

Uncertainty propagation is done by sampling: stellar radius, mass, and temperature are each drawn from Gaussians centered on the archive values with their reported uncertainties, then combined with the MCMC posterior samples point-by-point. Derived quantities are computed for each combination and percentiles of the resulting distributions give the reported medians and 1σ bounds. This correctly handles non-Gaussian posteriors and parameter correlations without Gaussian error propagation assumptions.

---

## The vetting problem: NASA's approach vs. the right approach for this app

This section is about the part of the pipeline that caused the most trouble, required the most iteration, and ultimately taught the most interesting lessons — both about astrophysics and about what it means to build something for actual users.

### The false positive problem

After the pipeline was working correctly for known single-planet systems, the first big test on a broader sample revealed a persistent and annoying problem: for roughly half of all Kepler targets, BLS would correctly find the real planet and then keep going — surfacing one, two, or three additional "detections" at long periods (typically 60–100 days) that were clearly not planets. The BLS power spectra for these candidates looked like pure noise: a flat wall of equally-tall spikes across the entire period range, with the "best period" just happening to be the tallest spike by a small margin. Signal Detection Efficiency of 8 or 9 — barely above the detection floor. Duration of half an hour, which is physically impossible for a real planet at those periods.

These are Kepler quarterly roll artifacts — instrumental systematics caused by the spacecraft rotating 90 degrees every ~90 days to keep its solar panels oriented correctly, landing each star on a slightly different detector pixel each time. BLS finds them because they repeat, not because they are transits.

The experience of seeing these in the app is immediately off-putting even if you know nothing about transit photometry. You look at the BLS power spectrum and something feels wrong. Everything is the same height. There is no isolated dominant peak. It does not look like a detection — it looks like the algorithm found the tallest spike in a histogram of noise. Which is exactly what happened.

This is an experience-breaking false positive. Not just a wrong answer — an answer that makes the whole app feel unreliable.

### NASA's solution: the TCE framework

NASA's Kepler pipeline uses a set of threshold-based vetting rules called the Threshold Crossing Event (TCE) pipeline (Jenkins et al. 2010). The logic is as follows: a BLS detection must pass a battery of physical plausibility checks before it is flagged as a planet candidate worthy of follow-up. The checks are:

| Flag | Test |
|------|------|
| TCE-01/02 | SDE and SNR above 7.1σ floor |
| TCE-03 | Per-transit SNR > threshold (signal should grow as √N transits) |
| TCE-04 | Duration/period ratio < 0.1 (eclipsing binary discriminator) |
| TCE-05/06 | ≥ 3 in-transit points, ≥ 70% of expected cadences covered |
| TCE-07 | Depth > 3σ above noise floor |
| TCE-08 | ≥ 3 complete transit windows in the baseline |
| TCE-09 | No strong alias periods (P/2, P/3, 2P, 3P, etc.) |
| TCE-13 | Depth < 3% (deeper signals are almost always grazing eclipsing binaries) |
| TCE-14 | Not below Kepler's 30 ppm detection floor with marginal SNR |
| TCE-15 | Depth ≥ 60 ppm for long-cadence (30-min) data |

This is a rigorous, well-motivated framework with decades of calibration behind it. The SDE floor of 7.1 comes from a detailed noise model of the full Kepler photometric pipeline. The depth checks are calibrated against known eclipsing binary populations. The alias rejection logic is designed to prevent harmonics of already-detected signals from generating spurious candidates.

This was the first vetting approach implemented in this pipeline. And it helped — but not enough. Even with the full TCE battery applied, the false positive rate on a 248-target validation run was over 57%. The noise artifacts at long periods were clearing the checks because the TCE thresholds were not calibrated for this pipeline's specific output, and because some of the checks (alias rejection, per-transit SNR floors) were simultaneously cutting real planets and admitting fake ones. The TCE framework was implemented faithfully, tested, tuned, and ultimately found to be insufficient for the problem at hand.

### Why NASA's approach is correct for NASA's problem

It is worth being precise about this, because "the TCE checks did not work well here" is easily misread as a criticism of the NASA pipeline. It is not.

NASA's TCE pipeline is designed to run on raw, unvetted Kepler photometry across 200,000 stars. Its job is to flag anything that might possibly be a planet for human follow-up. The design priority is **recall** — do not miss anything real. False positives are not a terminal failure; they get caught downstream by centroid analysis, spectroscopic confirmation, secondary eclipse checks, odd/even depth tests, community vetting, and ultimately by the requirement that a planet candidate receive peer-reviewed confirmation before appearing in a discovery paper. The TCE is stage one of a ten-stage process. Optimizing it for precision would mean missing real planets, which is the scientific sin.

The TCE thresholds were also calibrated against NASA's full noise model, which includes the spacecraft's actual systematic noise budget, detector characteristics, pixel-level sensitivity maps, and data quality flags that this pipeline does not have access to. The SDE floor of 7.1 is not an arbitrary number — it is the value at which false alarm probability drops below a specific threshold under a specific noise model. Applied to this pipeline's BLS output, the number does not mean the same thing.

### The sin is different for an app

The product requirement here is almost exactly inverted from NASA's. This is a fun, interactive app showing people how exoplanet detection works using real Kepler data. The user does not have a downstream vetting pipeline. They do not have spectroscopic follow-up resources. They are looking at the output of this pipeline directly, and if the pipeline shows them an 81-day "planet" with a BLS power spectrum that looks like a histogram of static, their experience of the whole thing is degraded — regardless of whether they know enough astrophysics to name the failure mode.

For this app, showing a false positive is not just a wrong answer. It is an experience-breaking wrong answer. The sin is precision, not recall. Missing a real small planet that was hard to detect is acceptable — nobody looking at the app for Kepler-55 was expecting all five sub-Neptunes to be findable in a few seconds. Showing an obvious noise artifact as a discovery is not acceptable.

### The empirical approach

Rather than continuing to tune the TCE thresholds by hand, an empirical approach was taken. The pipeline was run in permissive mode — six sequential BLS passes per target, maximum detection sensitivity — on Kepler targets 1 through 250. This took approximately 36 hours of continuous computation on a 2017 MacBook Pro and produced 1,488 labeled BLS candidates: 636 real planets (matched against NASA's published ephemerides) and 852 false positives.

Those 1,488 candidates were then used to train a decision tree classifier (scikit-learn, max depth 4, balanced class weights) on the BLS output features. The tree was evaluated on the same data it was trained on — this is in-sample performance, not cross-validated generalization — and the results were compared against both the TCE-only approach and a two-stage pipeline that combined TCE pre-filtering with a decision tree second stage.

### Results

**Feature distributions (real planets vs. false positives):**

| Feature | Real median | Real p25–p75 | FP median | FP p25–p75 |
|---------|-------------|--------------|-----------|------------|
| SDE | 68.32 | 46.0–84.2 | 8.05 | 7.3–9.8 |
| SNR | 33.69 | 24.8–50.2 | 12.77 | 9.5–19.9 |
| Transit depth (ppm) | 742.56 | 421.9–1155.3 | 468.38 | 278.2–797.7 |
| Depth uncertainty (ppm) | 21.00 | 11.3–34.0 | 32.12 | 16.6–66.6 |
| In-transit data points | 472.50 | 260.2–798.8 | 99.50 | 44.0–307.2 |
| Expected transit windows | 96.61 | 46.2–183.3 | 21.70 | 13.4–79.2 |
| Per-transit SNR | 3.54 | 2.6–5.3 | 2.19 | 1.5–3.4 |
| Transit duration (hours) | 2.69 | 2.1–4.4 | 1.30 | 0.6–2.7 |

The SDE separation is striking. Real planets have a median SDE of 68 — the BLS peak stands nearly 70 standard deviations above the noise floor. False positives cluster just above the detection floor at SDE 8. These populations barely overlap. This single feature captures most of what is knowable from BLS output alone, and it also explains why the experience of seeing a false positive in the app feels wrong even without knowing the formalism: a real transit produces a narrow, isolated, dominant spike in the BLS power spectrum. Noise produces a flat forest of equally-tall spikes and the "best period" is just the tallest one by a small margin.

**Single-feature threshold sweep (95% recall floor):**

| Feature | Threshold | Recall | Precision | F1 | FP cut |
|---------|-----------|--------|-----------|-----|--------|
| SDE ≥ | 9.1 | 95.1% | 69.1% | 0.801 | 582/852 |
| SNR ≥ | 14.3 | 95.1% | 62.8% | 0.756 | 493/852 |
| In-transit points ≥ | 101 | 95.3% | 59.1% | 0.730 | 433/852 |
| Expected windows ≥ | 19.3 | 95.3% | 57.1% | 0.714 | 396/852 |
| Duration ≥ | 1.0h | 96.5% | 52.3% | 0.679 | 293/852 |

**Decision tree (depth 4, trained on all 1,488 candidates):**

```
recall=92.9%  precision=97.7%  F1=0.952  FP cut=838/852  real kept=591/636
```

The tree eliminates 838 of 852 false positives while retaining 591 of 636 real planets. Feature importances confirm the SDE dominance: SDE accounts for 92.2% of the information gain across all splits. The remaining features (duty cycle, coverage ratio, in-transit points, duration, expected windows) collectively account for the remaining 7.8%.

The root split is SDE ≤ 19.17. Below that threshold, the tree uses a combination of in-transit data points, coverage ratio, duty cycle, and per-transit SNR to handle the harder cases where a real planet has a modest BLS peak. Above SDE 19.17, essentially everything is real — the tree uses minor geometry checks to clean up a small number of edge cases. The empirically determined threshold of 19.17 is notably much higher than the NASA TCE floor of 7.1, which makes sense: the TCE floor is set to catch everything that might be real, while the tree's split is set where the two populations actually stop overlapping on this pipeline's specific output.

**Was a two-stage pipeline (TCE pre-filter + decision tree) better?**

No. The tree was also tested in a two-stage configuration where TCE hard cuts eliminated candidates before the tree ran. All three TCE leniency levels produced slightly worse F1 than the tree alone:

| Config | Recall | Precision | F1 | FP cut | Real kept |
|--------|--------|-----------|-----|--------|-----------|
| Tree only (baseline) | 92.9% | 97.7% | 0.952 | 838/852 | 591/636 |
| TCE LOOSE → tree | 92.0% | 98.5% | 0.951 | 843/852 | 585/636 |
| TCE MEDIUM → tree | 91.5% | 99.0% | 0.951 | 846/852 | 582/636 |
| TCE TIGHT → tree | 82.2% | 99.6% | 0.901 | 850/852 | 523/636 |

The reason the two-stage approach does not help is that the tree already learned the TCE-relevant boundaries from the data. The root split at SDE=19.17 is functionally doing what TCE-01 attempts to do, but calibrated to the empirical distribution of this pipeline's output rather than to a theoretical noise model. Adding an explicit TCE pre-filter on top is redundant work — and any real planets the TCE rejects before the tree sees them are gone permanently. The tree alone is both simpler and better.

### What this means for generalization

The decision tree was fit and evaluated on the same 1,488 candidates from the same 248 targets. This is in-sample performance. The 97.7% precision figure is optimistic and will not fully generalize to unseen targets, noisier stars, or different instruments. TESS, for example, has a different systematic noise profile than Kepler, different cadence options, and much shorter per-sector baselines — a tree trained on Kepler would need to be retrained from scratch to work there.

This is fine. The explicit goal was to make the app work well and look credible for Kepler targets 1–250. That goal has been achieved. The pipeline is not being submitted to the Astrophysical Journal. What this exercise did produce — beyond a better app — is a clear empirical characterization of what information BLS output actually carries about candidate reliability on this specific instrument and pipeline. Almost everything useful is in SDE. A small amount is in geometry. Virtually nothing is in the features that TCE checks most carefully. Whether that finding generalizes is an interesting question for someone who wants to spend another 36 hours of laptop time.

---

## What I learned

**Sliding window detrending corrupts wide transits without masking.**
Any local smoother — Savitzky-Golay, biweight, or otherwise — that is blind to the transit signal will partially fill it in. The smoother sees a dip and pulls the local trend downward to compensate, attenuating the depth. The effect is worst for long-duration transits whose width approaches the filter window. The fix is iterative: detect with a rough detrend, mask the transit windows, redetrend from pure stellar continuum. Pass 2 in this pipeline implements exactly that — the biweight never sees in-transit flux when estimating the trend the MCMC fits against.

**Tight MCMC posteriors are not evidence of a correct answer.**
A corrupted likelihood surface converges confidently to the wrong place. If the detrending partially filled a transit before MCMC ran, the sampler faithfully finds the best fit to the altered data, and there is nothing in the posterior itself that signals the problem. The diagnostic is to evaluate the log-likelihood at the known-truth parameters and compare it to the converged value. If truth scores worse than the sampler's answer, the data was altered — not the sampler's fault.

**batman with quadratic limb darkening cannot reproduce published depths for some targets.**
Kepler-7b is the clearest example: the original discovery paper used a more complex limb darkening model, and the quadratic approximation cannot reproduce that transit shape exactly. The mismatch is invisible from the posterior — the sampler converges, uncertainties look reasonable, the answer is just wrong. The only way to detect it is to inject known parameters and check whether they are recovered, which is a different kind of test than just running the pipeline and looking at the output.

**Model mismatch is undetectable without injection tests.**
This is the general version of the above. If the model is wrong in a way that still allows convergence, there is no signal of failure in the posterior. Injection-recovery (simulate a transit with known parameters, run the full pipeline, check if you get them back) is the correct test. It separates "the sampler worked" from "the model was right."

**SDE is almost all of the information.**
The BLS Signal Detection Efficiency accounts for 92% of the decision tree's information gain in separating real planets from false positives. Real Kepler planet detections have median SDE around 68. False positives cluster at SDE 8. The TCE threshold of SDE > 7.1 catches almost nothing because it was designed as a recall-maximizing detection floor, not a precision-maximizing discriminator. The empirically calibrated split from the decision tree is SDE > 19.17 — more than twice as high, and the place where the two populations actually stop overlapping in this pipeline.

**The right metric depends on the product, not the science.**
NASA optimizes for recall because missing a real planet is the scientific failure mode. False positives get vetted downstream by humans with telescopes. An interactive app has no downstream vetting — it shows results directly to users, and a false positive that looks obviously wrong is an experience-breaking failure. The sin is different. Optimizing for precision, at the cost of missing some hard detections, produces a better product even if it would make a worse survey instrument.

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
  result_evaluation.py — decision tree reliability vetting
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

tests/
  threshold_optimization/
    fit_thresholds.py         — grid search + decision tree fitting on labeled candidates
    candidates_*.csv          — labeled BLS candidates from permissive validation runs
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
astroquery        — NASA Exoplanet Archive queries (used in test scripts)
scikit-learn      — decision tree classifier for reliability vetting
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

**Transit masking in multi-planet systems**
After masking a detected planet's transits, BLS occasionally re-detects the same period on the next iteration, indicating the mask didn't fully suppress the signal. The mask window is currently 3× the BLS-estimated duration, but BLS duration estimates can underestimate true duration, especially for grazing or long-duration transits.

**Reliability threshold generalizability**
The decision tree was fit on labeled candidates from Kepler targets 1–250 and evaluated on the same data (in-sample). It will not generalize without retraining to different instruments (TESS has a very different systematic noise profile), short-baseline observations with few transit windows, or active stars where the noise floor is elevated. For the intended use case — Kepler 1–250 in the app — this is fine.

**Radius underestimation for large planets**
Hot Jupiters and other large planets (depth > ~1%) have transits wide enough that even Pass 2 detrending leaves a small residual suppression — the biweight window near the transit edges still sees some in-transit continuum. The MCMC then fits a slightly shallower transit than the true depth, producing a systematically underestimated radius. The effect is small for Earth-to-Neptune-sized planets but becomes meaningful above ~10 R⊕.

**Transit timing variations (TTVs)**
Planets in or near mean-motion resonance have transit times that shift by minutes to hours from orbit to orbit due to gravitational interactions. BLS assumes perfectly periodic transits and smears out TTV signals when phase-folding, reducing detection SNR. Handling TTVs properly requires a separate dynamical modeling step not currently implemented.

**Stellar activity**
Starspots, flares, and coronal mass ejections all produce flux variations that can alias into the BLS period grid or corrupt the transit depth estimate. The biweight filter removes slow trends but is not designed to handle short-duration flares or rotationally-modulated spot patterns.

**Eclipsing binary contamination**
Eclipsing binaries are a major source of false positives in transit surveys. The decision tree catches most of the obvious cases via the duty cycle and depth features it learned to use, but anything subtle — odd/even depth alternation, secondary eclipses at phase 0.5, or background EBs blended inside the photometric aperture — requires centroid motion analysis, radial velocity follow-up, or multi-wavelength photometry to rule out. If the target is actually an eclipsing binary that passes the tree, the pipeline will fit it and give you numbers. The numbers will be wrong.

**Uncertainty underestimation**
The detrended light curve is treated as fixed truth by the transit model fitting. Any uncertainty in where the stellar trend was — whether the filter slightly suppressed a transit or misidentified a stellar oscillation as continuum — does not propagate into the parameter posteriors. The reported error bars represent photon noise only. Fully propagating detrending uncertainty requires simultaneous GP + transit modeling, which is computationally prohibitive for real-time use.

**Why real planet discoveries get their own papers**
The above list is not exhaustive. Real planet confirmation involves independent TCE vetting by multiple reviewers, centroid motion analysis to rule out background EBs, spectroscopic stellar characterization, dynamical modeling for multi-planet systems, and statistical false positive probability calculations. This pipeline runs the full detection and characterization sequence and produces results consistent with known systems. What it does not do is confirm planets: that requires the follow-up observations and analysis that real discovery papers are built around.

---

## References

- [Kovács, Zucker & Mazeh (2002) — Box Least Squares algorithm](https://arxiv.org/abs/astro-ph/0206099)
- [Jenkins et al. (2010) — Kepler TCE pipeline and SDE threshold](https://arxiv.org/abs/1001.0258)
- [Mandel & Agol (2002) — Analytic transit light curve models (batman)](https://arxiv.org/abs/astro-ph/0210099)
- [Claret (2011) — Kepler quadratic limb darkening tables](https://www.aanda.org/articles/aa/full_html/2011/05/aa16451-11/aa16451-11.html)
- [Foreman-Mackey et al. (2013) — emcee ensemble MCMC sampler](https://arxiv.org/abs/1202.3665)