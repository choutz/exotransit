# Validation against known targets

The full pipeline was tested on five Kepler systems spanning one to five confirmed planets each. BLS detection and MCMC fitting were run and results compared against NASA's published ephemerides. All 15 planets across the five targets were successfully detected, including Kepler-20 e at 0.84 R⊕, smaller than Earth. Periods were recovered to better than 0.01% for all 15 planets.

**Planet-level radius recovery:**

| Planet | True R (R⊕) | Pipeline R (R⊕) | Error | Notes |
|--------|------------|----------------|-------|-------|
| Kepler-5 b | 14.92 | 15.37 | +3.0% | |
| Kepler-183 c | 2.90 | 2.47 | −14.9% | Stellar radius uncertainty dominates; see note |
| Kepler-183 b | 2.76 | 2.12 | −23.2% | Stellar radius uncertainty dominates; see note |
| Kepler-18 c | 4.91 | 5.41 | +10.2% | |
| Kepler-18 d | 5.94 | 6.60 | +11.1% | |
| Kepler-18 b | 2.03 | 2.15 | +6.1% | |
| Kepler-58 b | 2.33 | 2.10 | −10.0% | |
| Kepler-58 c | 2.94 | 2.30 | −21.6% | Truth b = 0.946, nearly grazing |
| Kepler-58 d | 2.92 | 2.43 | −16.9% | Truth b = 0.868, high impact parameter |
| Kepler-58 e | 1.61 | 1.38 | −14.0% | |
| Kepler-20 c | 2.99 | 2.90 | −3.1% | |
| Kepler-20 b | 1.78 | 1.81 | +1.5% | |
| Kepler-20 d | 2.59 | 2.37 | −8.5% | MCMC not converged; true depth ~78 ppm |
| Kepler-20 e | 0.84 | 0.92 | +9.6% | Sub-Earth |
| Kepler-20 f | 0.90 | 0.99 | +9.9% | MCMC not converged; true depth ~98 ppm |

**Kepler-183 stellar radius note.** The Kepler-183 radius errors are dominated by a stellar radius discrepancy rather than pipeline fitting error. The pipeline's rp/R* recovery for Kepler-183 c is +6.5% and for Kepler-183 b is −3.8%, consistent with the rest of the validation sample. The large apparent radius errors arise because two different NASA Exoplanet Archive tables return meaningfully different stellar radii for this star.

The ground truth values in examples.csv come from the KOI cumulative table (`koi_srad`), which gives R★ = 1.198 R☉ for Kepler-183. The pipeline's `query_stellar_params` function queries a different archive table and returns R★ = 0.958 R☉, matching the Rowe et al. 2014 value. The NASA Exoplanet Archive overview page for Kepler-183 lists stellar radii ranging from 0.87 to 1.068 R☉ across TICv8, Berger et al. 2018, Gaia DR2, Morton et al. 2016, and Rowe et al. 2014 - with the modern photometric/astrometric sources (TICv8, Berger, Gaia) clustering around 1.06 R☉. Neither value is definitively wrong; Kepler-183 is a star with significant published scatter in its stellar parameters. The Kepler-183 rows in the radius table should be read as reflecting this underlying stellar uncertainty, not pipeline fitting failure.

**Analysis of Persistent Impact Parameter Bias**

To mitigate the effects of "transit erosion", where a sliding-window biweight filter distorts the ingress and egress of a light curve, this pipeline employs a multi-pass masking strategy. By identifying transit locations via an initial Box Least Squares (BLS) search and masking these regions during the final detrending pass, the stellar baseline is reconstructed without "seeing" the planetary signal. This theoretically preserves the pristine U-shape of the transit, preventing the filter from flattening the shoulders or depressing the out-of-transit baseline.

Despite this robust masking and redetrending process, the impact parameter b is systematically recovered at higher-than-true values in many test cases. This indicates that the preference for a more V-shaped, grazing geometry is not merely a byproduct of detrending distortion, but is driven by a deeper interaction between the data and the model. 

When the MCMC sampler converges on a high-b solution despite a preserved light curve shape, it suggests a fundamental degeneracy where the distorted geometry is statistically preferred over the physical truth.

If the detrending is no longer the primary culprit, the source of the bias likely shifts to the following factors:

* Limb Darkening Mismatch: Quadratic limb darkening coefficients ($u_1, u_2$) define the curvature of the transit bottom. If the assumed stellar parameters are even slightly inaccurate, the MCMC sampler may inflate b as a "fudge factor" to match the observed curvature of the star's disk. A high impact parameter and strong limb darkening both produce similar "rounding" of the light curve; the sampler may be unable to distinguish between the two.
* Prior Volume Effects: In the high-dimensional space of an MCMC, there is a "volumetric" bias toward higher b values. In a uniform prior between 0 and 1, a larger portion of the parameter space corresponds to inclined orbits. Without a sufficiently high Signal-to-Noise Ratio (SNR) to "lock" the sampler into a flat-bottomed solution, the random walk will naturally tend to spend more time in the high-b regime. Using a longer baseline would increase SNR and the resolution of phase-folded light curves of real planets and potentially help mitigate this issue, but the web app and this testing use the LOW config, with only 8 quarters of data, in order to make the speed of the web app tolerable. 

Crucially, while b and the semi-major axis remain difficult to pin down, the radius ratio (rp/R★) remains highly accurate. This is because the transit depth is physically constrained by the total flux deficit at the center of the transit, which is largely unaffected by the "smearing" or "rounding" of the shoulders.

**Unconverged MCMC for near-threshold planets.** Kepler-20 d (true depth ~78 ppm) and Kepler-20 f (true depth ~98 ppm) had MCMC acceptance fractions of 0.17 and 0.19 respectively, both below the healthy range of 0.2-0.5, and autocorrelation times longer than the chain length. Both planets were found by BLS and passed vetting, and radius estimates are within 10% of truth, but the posteriors are not fully trustworthy. This is a known limitation for signals near the detection floor where the likelihood surface is nearly flat and the sampler struggles to explore it efficiently.

**BLS depth overestimation.** The BLS depth estimate is consistently 15-90% higher than the true transit depth. The bottom-30% flux estimator applied to the phase-folded light curve picks up noise in the transit minimum rather than just the true transit floor, and the phase-folded minimum is itself sensitive to scatter from imperfect phase alignment. This does not propagate into final planet sizes: MCMC uses the full unfolded light curve rather than the BLS depth estimate, and converges to accurate rp values independently. The practical consequence is that the MCMC initialization point for rp is always too large and the chain must burn in away from the BLS starting guess, adding a small amount of convergence time.
