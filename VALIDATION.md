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

**Impact parameter degeneracy.** The impact parameter `b` is systematically recovered incorrectly for most planets. The biweight detrending filter operates on a sliding window of several days. When the window passes over a transit, it partially fits through the transit signal, which has two effects: it raises the in-transit flux slightly and depresses the out-of-transit baseline slightly near the transit edges. This distorts the transit shape - specifically it flattens and narrows the ingress and egress shoulders relative to the flat transit bottom, making the transit appear more V-shaped than it truly is. The MCMC sampler, fitting the detrended light curve, converges to a higher impact parameter than the true value because a more inclined (higher-b) orbit produces this more-pointed transit shape. The sampler is not making an error; it is correctly fitting the distorted data. This is confirmed by a log-likelihood check: evaluating the likelihood at the published parameters gives a worse score than the converged answer, meaning the detrended data genuinely prefers the wrong geometry. Radius accuracy is preserved because rp² is approximately equal to transit depth. When a planet transits its star, the fractional drop in flux is:

```
depth = (area of planet disk) / (area of star disk) = π rp² / π R★² = (rp/R★)²
```

So `(rp/R★)²` is literally the transit depth, and rp is constrained by how deep the transit is. What changes with impact parameter is the shape of the transit: a high-b transit is V-shaped because the planet chord across the stellar disk is shorter, making ingress and egress take up a larger fraction of the total duration. But the depth at mid-transit stays approximately the same as long as the planet is fully on the disk.

When detrending distorts the ingress and egress shape and the sampler compensates by inflating b, it is adjusting the geometry to match the distorted shoulders - but the depth at the bottom of the transit is largely untouched by the detrending filter, and that depth is what pins rp. The sampler finds the wrong b but the right rp because the two parameters are constrained by different aspects of the light curve: b by the shape, rp by the depth. The geometry is wrong; the size is not.

This approximation breaks down slightly in the presence of limb darkening, which makes the stellar surface brighter at the center than the edges. A grazing transit at high b samples a darker part of the star, complicating the depth-to-radius relationship. This is why rp² = depth is approximate rather than exact, and why small radius errors persist even on well-behaved targets.

**Impact parameter recovery at high inclination.** The two planets with the highest published impact parameters in the sample - Kepler-18 b (b = 0.741) and Kepler-58 d (b = 0.868) - have the best b recovery. High-b transits are genuinely V-shaped, so the detrending-induced V-shape distortion is less damaging: the data already looks like what the model expects. Low-b transits have a flat, U-shaped bottom that gets distorted into something intermediate, and the sampler compensates by inflating b. This is a hypothesis supported by the validation sample rather than a proven result; other factors including transit depth, duration, and per-transit SNR also influence how severely the detrending distorts the transit shape.

**Unconverged MCMC for near-threshold planets.** Kepler-20 d (true depth ~78 ppm) and Kepler-20 f (true depth ~98 ppm) had MCMC acceptance fractions of 0.17 and 0.19 respectively, both below the healthy range of 0.2-0.5, and autocorrelation times longer than the chain length. Both planets were found by BLS and passed vetting, and radius estimates are within 10% of truth, but the posteriors are not fully trustworthy. This is a known limitation for signals near the detection floor where the likelihood surface is nearly flat and the sampler struggles to explore it efficiently.

**BLS depth overestimation.** The BLS depth estimate is consistently 15-90% higher than the true transit depth. The bottom-30% flux estimator applied to the phase-folded light curve picks up noise in the transit minimum rather than just the true transit floor, and the phase-folded minimum is itself sensitive to scatter from imperfect phase alignment. This does not propagate into final planet sizes: MCMC uses the full unfolded light curve rather than the BLS depth estimate, and converges to accurate rp values independently. The practical consequence is that the MCMC initialization point for rp is always too large and the chain must burn in away from the BLS starting guess, adding a small amount of convergence time.
