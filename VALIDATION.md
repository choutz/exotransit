# Validation against known targets

The full pipeline was tested on five Kepler systems spanning one to five confirmed planets each. BLS detection and MCMC fitting were run and results compared against NASA's published ephemerides. All 15 planets across the five targets were successfully detected, including Kepler-20 e at 0.84 R⊕, smaller than Earth. Periods were recovered to better than 0.01% for all 15 planets.

## Radius recovery

| Planet | True R (R⊕) | Pipeline R (R⊕) | Error | Notes |
|--------|------------|----------------|-------|-------|
| Kepler-5 b | 14.92 | 15.37 | +3.0% | |
| Kepler-183 c | 2.90 | 2.47 | −14.9% | Stellar radius uncertainty; see note |
| Kepler-183 b | 2.76 | 2.12 | −23.2% | Stellar radius uncertainty; see note |
| Kepler-18 c | 4.91 | 5.41 | +10.2% | |
| Kepler-18 d | 5.94 | 6.60 | +11.1% | |
| Kepler-18 b | 2.03 | 2.15 | +6.1% | |
| Kepler-58 b | 2.33 | 2.10 | −10.0% | |
| Kepler-58 c | 2.94 | 2.30 | −21.6% | Near-grazing orbit (literature b ≈ 0.95) |
| Kepler-58 d | 2.92 | 2.43 | −16.9% | High impact parameter (literature b ≈ 0.87) |
| Kepler-58 e | 1.61 | 1.38 | −14.0% | |
| Kepler-20 c | 2.99 | 2.90 | −3.1% | |
| Kepler-20 b | 1.78 | 1.81 | +1.5% | |
| Kepler-20 d | 2.59 | 2.37 | −8.5% | Near detection floor (~78 ppm depth) |
| Kepler-20 e | 0.84 | 0.92 | +9.6% | Sub-Earth |
| Kepler-20 f | 0.90 | 0.99 | +9.9% | Near detection floor (~98 ppm depth) |

Radius errors are mostly in the 3–15% range. The two outliers (Kepler-58 c and d) are nearly grazing transits where the true transit shape deviates furthest from the central-transit approximation, and radius and impact parameter become more degenerate. The Kepler-183 rows reflect stellar parameter uncertainty, not fitting error (see note below).

## Period recovery

All 15 planets recovered to better than 0.01% of the published period. BLS period search is the strongest part of the pipeline.

## Impact parameter

Impact parameter (b) is not reliably recovered, and this is expected.

The validation "truth" values for b come from the KOI cumulative table, which uses an automated trapezoidal fitting pipeline optimized for planet discovery throughput rather than precise geometric characterization. That pipeline is known to systematically push b toward zero, preferring face-on solutions. For Kepler-5b, the KOI table reports b = 0.075, while the detailed independent fit in Koch et al. (2010) gives b ≈ 0.27–0.28. The pipeline's MCMC returns b ≈ 0.33 for that system, which is roughly 20% off from the literature value, not the 339% error that comparing against the KOI table would suggest.

Beyond the reference data issue, b is genuinely the hardest transit parameter to recover:

- **b–rp degeneracy.** A more grazing transit (higher b) looks similar to a smaller planet transiting near center; both produce a V-shaped transit bottom. At moderate SNR, the posterior over b is wide.
- **Limb darkening.** Fixed limb darkening coefficients (Claret 2011 theoretical values) can't absorb small errors in stellar parameters, so the sampler may adjust b to compensate.
- **30-minute cadence smearing.** Kepler long-cadence data blurs ingress and egress over the integration window. The pipeline applies batman's supersampling correction, but smearing fundamentally reduces the geometric information in the light curve.

The saving grace is that transit depth (and therefore planet radius) is set by the flux deficit at transit center, which is much less sensitive to b than the ingress/egress shape. This is why radius recovery is good even when b is uncertain.

## Notes

**Kepler-183 stellar radius.** The Kepler-183 radius errors are dominated by a stellar radius discrepancy in the NASA archive, not pipeline fitting error. The pipeline's rp/R* ratio for Kepler-183 c is +6.5% and for Kepler-183 b is −3.8%, consistent with the rest of the sample. The apparent planet radius errors arise because the KOI table gives R★ = 1.198 R☉ for this star, while the pipeline's archive query returns R★ = 0.958 R☉ (matching Rowe et al. 2014). Published values for Kepler-183 range from 0.87 to 1.068 R☉ across different catalogs; the star has genuine uncertainty in its stellar parameters.

**Near-threshold planets.** Kepler-20 d and f have MCMC acceptance fractions below the healthy 0.2–0.5 range (0.17 and 0.19) and autocorrelation times longer than the chain length. Radius estimates are within 10% but the posteriors are not fully trustworthy. This is expected behavior for signals near the detection floor where the likelihood surface is nearly flat.

**BLS depth overestimation.** The BLS depth estimate runs 15–90% above true transit depth. This does not propagate into final planet sizes; MCMC fits the full transit independently and converges to accurate rp values. The practical effect is that the MCMC chain initializes from an inflated rp and must burn in toward the correct value.
