# The vetting problem: NASA's approach vs. the right approach for this app

This document covers the part of the pipeline that caused the most trouble, required the most iteration, and ultimately taught the most interesting lessons, both about astrophysics and about what it means to build something for actual users.

---

## The false positive problem

After the pipeline was working correctly for known single-planet systems, the first big test on a broader sample revealed a persistent and annoying problem: for roughly half of all Kepler targets, Box Least Squares (BLS) would correctly find the real planet and then keep going, surfacing one, two, or three additional "detections" at long periods (typically 60–100 days) that were clearly not planets. The BLS power spectra for these candidates looked like pure noise: a flat wall of equally-tall spikes across the entire period range, with the "best period" just happening to be the tallest spike by a small margin. Signal Detection Efficiency (SDE) of 8 or 9, barely above the detection floor. Duration of half an hour, which is physically impossible for a real planet at those periods.

These are Kepler quarterly roll artifacts: instrumental systematics caused by the spacecraft rotating 90 degrees every ~90 days to keep its solar panels oriented correctly, landing each star on a slightly different detector pixel each time. BLS finds them because they repeat, not because they are transits.

The experience of seeing these in the app is immediately off-putting even if you know nothing about transit photometry. You look at the BLS power spectrum and something feels wrong. Everything is the same height. There is no isolated dominant peak. It does not look like a detection; it looks like the algorithm found the tallest spike in a histogram of noise. Which is exactly what happened.

This is an experience-breaking false positive. Not just a wrong answer, but an answer that makes the whole app feel unreliable.

---

## NASA's solution: the TCE framework

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

This was the first vetting approach implemented in this pipeline. And it helped, but not enough. Even with the full TCE battery applied, the false positive rate on a 250-target validation run was over 57%. The noise artifacts at long periods were clearing the checks because the TCE thresholds were not calibrated for this pipeline's specific output, and because some of the checks (alias rejection, per-transit SNR floors) were simultaneously cutting real planets and admitting fake ones. The TCE framework was implemented faithfully, tested, tuned, and ultimately found to be insufficient for the problem at hand.

---

## Why NASA's approach is correct for NASA's problem

It is worth being precise about this, because "the TCE checks did not work well here" is easily misread as a criticism of the NASA pipeline. It is not.

NASA's TCE pipeline is designed to run on raw, unvetted Kepler photometry across 200,000 stars. Its job is to flag anything that might possibly be a planet for human follow-up. The design priority is **recall**: do not miss anything real. False positives are not a terminal failure; they get caught downstream by centroid analysis, spectroscopic confirmation, secondary eclipse checks, odd/even depth tests, community vetting, and ultimately by the requirement that a planet candidate receive peer-reviewed confirmation before appearing in a discovery paper. The TCE is stage one of a ten-stage process. Optimizing it for precision would mean missing real planets, which is the scientific sin.

The TCE thresholds were also calibrated against NASA's full noise model, which includes the spacecraft's actual systematic noise budget, detector characteristics, pixel-level sensitivity maps, and data quality flags that this pipeline does not have access to. The SDE floor of 7.1 is not an arbitrary number; it is the value at which false alarm probability drops below a specific threshold under a specific noise model. Applied to this pipeline's BLS output, the number does not mean the same thing.

---

## The sin is different for an app

The product requirement here is almost exactly inverted from NASA's. This is a fun, interactive app showing people how exoplanet detection works using real Kepler data. The user does not have a downstream vetting pipeline. They do not have spectroscopic follow-up resources. They are looking at the output of this pipeline directly, and if the pipeline shows them an 81-day "planet" with a BLS power spectrum that looks like a histogram of static, their experience of the whole thing is degraded, regardless of whether they know enough astrophysics to name the failure mode.

For this app, showing a false positive is not just a wrong answer. It is an experience-breaking wrong answer. The sin is precision, not recall. Missing a real small planet that was hard to detect is acceptable; nobody looking at the app for Kepler-55 was expecting all five sub-Neptunes to be findable in a few seconds. Showing an obvious noise artifact as a discovery is not acceptable.

---

## The empirical approach

Rather than continuing to tune the TCE thresholds by hand, an empirical approach was taken. The pipeline was run in permissive mode (five sequential BLS passes per target, maximum detection sensitivity) on Kepler targets 1 through 250. With the BLS period grid set to 25,000 points and a parallelized implementation, the full permissive run completed in approximately 45 minutes, producing 1,250 labeled BLS candidates: 599 real planets (matched against NASA's published ephemerides) and 651 false positives.

Those 1,250 candidates were then used to train a decision tree classifier (scikit-learn, max depth 4, balanced class weights) on the BLS output features. The tree was evaluated on the same data it was trained on (in-sample performance, not cross-validated generalization), and the results were compared against both the TCE-only approach and a two-stage pipeline that combined TCE pre-filtering with a decision tree second stage.

---

## Results

**Feature distributions (real planets vs. false positives):**

| Feature | Real median | Real p25–p75 | FP median | FP p25–p75 |
|---------|-------------|--------------|-----------|------------|
| SDE | 47.36 | 29.0–57.8 | 7.13 | 6.4–9.1 |
| SNR | 28.56 | 21.3–42.2 | 13.91 | 9.4–22.7 |
| Transit depth (ppm) | 745.40 | 412.1–1155.9 | 474.98 | 301.3–807.3 |
| Depth uncertainty (ppm) | 23.62 | 12.3–39.2 | 30.54 | 15.9–69.1 |
| In-transit data points | 362.00 | 213.5–584.0 | 95.00 | 32.5–472.5 |
| Expected transit windows | 66.42 | 35.4–125.1 | 16.75 | 8.4–110.8 |
| Per-transit SNR | 3.61 | 2.6–5.4 | 2.52 | 1.7–4.1 |
| Transit duration (hours) | 2.88 | 2.4–3.8 | 1.92 | 1.0–3.8 |

The SDE separation is striking. Real planets have a median SDE of 47; the BLS peak stands nearly 50 standard deviations above the noise floor. False positives cluster just above the detection floor at SDE 7. These populations barely overlap. This single feature captures most of what is knowable from BLS output alone, and it also explains why the experience of seeing a false positive in the app feels wrong even without knowing the formalism: a real transit produces a narrow, isolated, dominant spike in the BLS power spectrum. Noise produces a flat forest of equally-tall spikes and the "best period" is just the tallest one by a small margin.

**Single-feature threshold sweep (95% recall floor):**

| Feature | Threshold | Recall | Precision | F1 | FP cut |
|---------|-----------|--------|-----------|-----|--------|
| SDE ≥ | 10.13 | 95.3% | 81.2% | 0.877 | 519/651 |
| In-transit points ≥ | 116 | 95.7% | 66.3% | 0.783 | 360/651 |
| SNR ≥ | 13.52 | 95.5% | 63.3% | 0.761 | 319/651 |
| Expected windows ≥ | 12.02 | 98.0% | 61.5% | 0.755 | 283/651 |
| Duration ≥ | 1.44h | 98.7% | 56.6% | 0.719 | 197/651 |

**Decision tree (depth 4, trained on all 1,250 candidates):**

```
recall=95.0%  precision=96.1%  F1=0.955  FP cut=628/651  real kept=569/599
```

The tree eliminates 628 of 651 false positives while retaining 569 of 599 real planets. Feature importances confirm the SDE dominance: SDE accounts for 86.0% of the information gain across all splits. The remaining features (duty cycle, coverage ratio, in-transit points, duration, expected windows, per-transit SNR) collectively account for the other 14%.

The root split is SDE ≤ 15.12. Below that threshold, the tree uses a combination of expected transit windows, duration, and duty cycle to handle the harder cases where a real planet has a modest BLS peak. Above SDE 15.12, essentially everything is real, and the tree uses minor geometry checks (in-transit points, coverage ratio) to clean up a small number of edge cases. The empirically determined threshold of 15.12 is notably much higher than the NASA TCE floor of 7.1, which makes sense: the TCE floor is set to catch everything that might be real, while the tree's split is set where the two populations actually stop overlapping on this pipeline's specific output.

On the full 250-target run using the decision tree, 193 targets (77%) produced fully clean results with no false positives and no missed planets. 30 targets had at least one false positive, and 27 had at least one missed or only partially detected planet. For a pipeline that runs in a few minutes in a live web app and prioritizes not showing users something wrong, this is a reasonable outcome.

**Was a two-stage pipeline (TCE pre-filter + decision tree) better?**

No. The tree was also tested in a two-stage configuration where TCE hard cuts eliminated candidates before the tree ran. All three TCE leniency levels produced slightly worse F1 than the tree alone:

| Config | Recall | Precision | F1 | FP cut | Real kept |
|--------|--------|-----------|-----|--------|-----------|
| Tree only (baseline) | 95.0% | 96.1% | 0.955 | 628/651 | 569/599 |
| TCE LOOSE → tree | 93.7% | 97.2% | 0.954 | 635/651 | 561/599 |
| TCE MEDIUM → tree | 92.7% | 97.2% | 0.949 | 635/651 | 555/599 |
| TCE TIGHT → tree | 81.5% | 98.4% | 0.891 | 643/651 | 488/599 |

The reason the two-stage approach does not help is that the tree already learned the TCE-relevant boundaries from the data. The root split at SDE = 15.12 is functionally doing what TCE-01 attempts to do, but calibrated to the empirical distribution of this pipeline's output rather than to a theoretical noise model. Adding an explicit TCE pre-filter on top is redundant work, and any real planets the TCE rejects before the tree sees them are gone permanently. The tree alone is both simpler and better.

---

## What this means for generalization

The decision tree was fit and evaluated on the same 1,250 candidates from the same 250 targets. This is in-sample performance. The 96.1% precision figure is optimistic and will not fully generalize to unseen targets, noisier stars, or different instruments. TESS, for example, has a different systematic noise profile than Kepler, different cadence options, and much shorter per-sector baselines, so a tree trained on Kepler would need to be retrained from scratch to work there.

This is fine. The explicit goal was to make the app work well and look credible for Kepler targets 1–250. That goal has been achieved. The pipeline is not being submitted to the Astrophysical Journal. Beyond a better app, this exercise produced a clear empirical characterization of what information BLS output actually carries about candidate reliability on this specific instrument and pipeline. Almost everything useful is in SDE. A small amount is in geometry. Virtually nothing is in the features that TCE checks most carefully. Whether that finding generalizes is an interesting question for a future validation effort on a different dataset.
