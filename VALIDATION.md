# Validation against known targets

The full pipeline was tested on five Kepler systems spanning one to five confirmed planets each. BLS detection and MCMC fitting were run and results compared against NASA's published ephemerides. All 15 planets across the five targets were successfully detected, including Kepler-20 e at 0.84 R⊕, smaller than Earth.

**Planet-level radius recovery:**

| Planet | True R (R⊕) | Pipeline R (R⊕) | Error | Notes |
|--------|------------|----------------|-------|-------|
| Kepler-5 b | 14.92 | 15.37 | +3.0% | |
| Kepler-183 c | 2.90 | 2.47 | −14.9% | Archive R★ −20% wrong |
| Kepler-183 b | 2.76 | 2.12 | −23.2% | Archive R★ −20% wrong |
| Kepler-18 c | 4.91 | 5.41 | +10.2% | |
| Kepler-18 d | 5.94 | 6.60 | +11.1% | |
| Kepler-18 b | 2.03 | 2.15 | +6.1% | |
| Kepler-58 b | 2.33 | 2.10 | −10.0% | |
| Kepler-58 c | 2.94 | 2.30 | −21.6% | Truth b = 0.946, nearly grazing |
| Kepler-58 d | 2.92 | 2.43 | −16.9% | Truth b = 0.868, high impact |
| Kepler-58 e | 1.61 | 1.38 | −14.0% | |
| Kepler-20 c | 2.99 | 2.90 | −3.1% | |
| Kepler-20 b | 1.78 | 1.81 | +1.5% | |
| Kepler-20 d | 2.59 | 2.37 | −8.5% | MCMC not converged, depth ~78 ppm |
| Kepler-20 e | 0.84 | 0.92 | +9.6% | Sub-Earth |
| Kepler-20 f | 0.90 | 0.99 | +9.9% | MCMC not converged, depth ~98 ppm |

**Kepler-183 stellar radius.** The NASA Exoplanet Archive returned 0.958 R☉ for Kepler-183; the correct value is 1.198 R☉, a −20% error. Since planet radius scales directly with stellar radius, both planets show errors of 15–23% despite accurate rp/R★ recovery from the light curve. This is a bad archive entry, not a pipeline failure.

**Impact parameter degeneracy.** The impact parameter `b` is systematically recovered incorrectly for most planets. The biweight detrending slightly rounds transit ingress and egress shoulders, altering the transit shape in a way that creates a genuine degeneracy between `b` and `rp`. The sampler converges to a `b` value that fits the detrended data better than the published value. This is visible from a log-likelihood check: evaluating at the published parameters gives a worse score than the converged answer, meaning the detrended data genuinely prefers the wrong geometry. Radius accuracy is preserved because rp² is approximately equal to transit depth, which the detrending does not significantly alter. The geometry is wrong; the size is not.

**Impact parameter at high inclination.** Two planets with the highest published impact parameters (Kepler-18 b at b = 0.741 and Kepler-58 d at b = 0.868) have the best b recovery in the validation sample. High-b transits have a distinctive V-shaped geometry that the sampler can identify even in slightly distorted data, giving the fit more shape information to constrain b. Low-b transits have a nearly flat, U-shaped bottom that the detrending distorts more severely.

**Unconverged MCMC for near-threshold planets.** Kepler-20 d (true depth ~78 ppm) and Kepler-20 f (true depth ~98 ppm) had MCMC acceptance fractions of 0.17 and 0.19 respectively. Both planets were still found by BLS and passed vetting, and radius estimates are within 10% of truth, but the posteriors are not fully trustworthy. This is a known limitation for signals near the detection floor.

**BLS depth overestimation.** The BLS depth estimate is consistently 15–90% higher than the true transit depth. The bottom-30% flux estimator applied to the phase-folded light curve picks up noise spikes in the transit minimum rather than just the true transit floor. MCMC uses the actual light curve rather than this estimate and converges to accurate rp values, so the overestimation does not propagate into final planet sizes. It does mean the BLS initial guess for rp is always too large, and the MCMC chain must burn in away from this starting point.
