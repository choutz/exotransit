"""
exotransit/detection/search.py

BLS (Box Least Squares) period search for exoplanet transits.

BLS was introduced by Kovacs, Zucker & Mazeh (2002) and remains
the standard algorithm for transit detection in Kepler/TESS data.
It searches for periodic, box-shaped dips in a light curve by
testing thousands of period/duration combinations and scoring
each by how well a box model fits the data.

We wrap lightkurve's BLS implementation and add:
- Full power spectrum output for visualization
- SDE (Signal Detection Efficiency) calculation
- Alias detection (harmonics that can fool the algorithm)
- Honest uncertainty flags
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from exotransit.pipeline.fetch import LightCurveData
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

logger = logging.getLogger(__name__)


@dataclass
class BLSResult:
    """
    Full results from a BLS period search.

    We store everything — not just the best answer, but the full
    picture of what the algorithm found and how confident we are.

    Attributes
    ----------
    best_period : float
        Most likely orbital period in days.
    best_duration : float
        Most likely transit duration in days.
    best_t0 : float
        Time of first transit center (BJD).
    transit_depth : float
        Fractional flux decrease during transit.
        e.g. 0.001 means the planet blocks 0.1% of starlight.
        For reference: Jupiter blocks ~1%, Earth blocks ~0.01%.
    depth_uncertainty : float
        1-sigma uncertainty on transit depth. A depth of 0.001 ± 0.0005
        is very different from 0.001 ± 0.00001.
    sde : float
        Signal Detection Efficiency. How many standard deviations
        above the noise floor the best period peak stands.
        Convention: SDE > 7 = candidate, > 9 = strong candidate.
    snr : float
        Signal-to-noise ratio of the transit detection.
    periods : np.ndarray
        All periods tested (days). Used to plot the full power spectrum.
    power : np.ndarray
        BLS power at each tested period. The full spectrum, not just
        the peak. Shows aliases, harmonics, and ambiguity visually.
    folded_time : np.ndarray
        Phase-folded time array for plotting the transit shape.
        "Phase folding" means collapsing all transits on top of each
        other by dividing time by the period. If the period is right,
        all transits stack and the signal gets much cleaner.
    folded_flux : np.ndarray
        Phase-folded flux values.
    folded_flux_err : np.ndarray
        Phase-folded flux uncertainties — propagated through folding.
    aliases : list of float
        Detected alias periods (P/2, P/3, 2P etc.) that also show
        peaks in the power spectrum. These are a known artifact of
        periodic signals and can be mistaken for additional planets.
    is_reliable : bool
        Our honest assessment of whether this detection is trustworthy.
    reliability_flags : list of str
        Human-readable list of any concerns about the detection.
        e.g. ["SDE below threshold", "Strong alias at P/2"]
    """
    best_period: float
    best_duration: float
    best_t0: float
    transit_depth: float
    depth_uncertainty: float
    sde: float
    snr: float
    periods: np.ndarray
    power: np.ndarray
    folded_time: np.ndarray
    folded_flux: np.ndarray
    folded_flux_err: np.ndarray
    aliases: list = field(default_factory=list)
    is_reliable: bool = True
    reliability_flags: list = field(default_factory=list)


def run_bls(
    lc: LightCurveData,
    min_period: float = 0.5,
    max_period: float = 30.0,
) -> BLSResult:
    """
    Run BLS period search on a preprocessed light curve.

    Uses astropy's BoxLeastSquares directly for full control over
    the period and duration grids, avoiding the combinatorial explosion
    that occurs with lightkurve's wrapper on long baselines.

    Period grid: 100,000 linearly-spaced frequencies converted to periods.
    Linear frequency spacing is standard for BLS — it ensures uniform
    sampling of transit repetition rate rather than orbital period.

    Duration grid: 5 values from 1.2 to 12 hours. Missing the exact
    duration only weakens the signal slightly; missing the period loses
    it entirely. So we search periods finely and durations coarsely.

    Parameters
    ----------
    lc : LightCurveData
        Output from fetch.fetch_light_curve() or fetch_stitched_light_curve().
    min_period : float
        Minimum orbital period to search in days.
    max_period : float
        Maximum orbital period to search in days. Default 30 days —
        with 90 days of data you need at least 3 transits, so
        max reliable period ≈ baseline / 3.

    Returns
    -------
    BLSResult
        Full results including power spectrum and reliability assessment.
    """
    logger.info(
        f"Running BLS on {lc.target_name}: "
        f"period range {min_period}–{max_period} days, "
        f"{len(lc.time)} data points"
    )

    # Build lightkurve LightCurve for phase folding later
    lc_lk = lk.LightCurve(
        time=lc.time,
        flux=lc.flux,
        flux_err=lc.flux_err,
    )

    # --- Period grid ---
    # Sample linearly in frequency (1/period), then convert to period.
    # This is the correct sampling strategy for BLS: uniform in frequency
    # means we spend equal effort on each transit-rate hypothesis.
    # 100,000 points gives fine resolution without combinatorial explosion.
    period_grid = 1.0 / np.linspace(1.0 / max_period, 1.0 / min_period, 100_000)

    # --- Duration grid ---
    # 5 durations from 1.2 to 12 hours covers the vast majority of
    # real exoplanet transits. Coarse is fine here — a wrong duration
    # just weakens the peak, it doesn't move it to the wrong period.
    durations = np.linspace(0.05, 0.5, 5)  # days (0.05d = 1.2hr, 0.5d = 12hr)

    # --- Run BLS directly via astropy ---
    model = BoxLeastSquares(lc_lk.time, lc_lk.flux, lc_lk.flux_err)
    pg_results = model.power(period_grid, durations)

    # --- Extract best solution ---
    best_idx = np.argmax(pg_results.power)
    best_period = pg_results.period[best_idx].value
    best_duration = pg_results.duration[best_idx].value
    best_t0 = float(pg_results.transit_time[best_idx].value)

    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(pg_results.period, pg_results.power, color='black', lw=0.5)
    # plt.axvline(best_period, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_period:.2f}d')
    # plt.xlabel("Period (days)")
    # plt.ylabel("BLS Power")
    # plt.title(f"BLS Power Spectrum: {lc.target_name}")
    # plt.legend()
    # plt.show()


    periods = np.asarray(pg_results.period)
    power = np.asarray(pg_results.power)

    # --- Phase fold at best period ---
    # Collapses all transits on top of each other by dividing time
    # by the period. Every transit stacks at phase 0, dramatically
    # boosting SNR over any individual transit.
    folded = lc_lk.fold(period=best_period, epoch_time=best_t0)
    folded_time = np.asarray(folded.time.value)
    folded_flux = np.asarray(folded.flux.value)
    folded_flux_err = np.asarray(folded.flux_err.value)

    # --- Transit depth and uncertainty ---
    # Mask to in-transit points: within half a transit duration of phase 0.
    half_dur = best_duration / (2 * best_period)
    in_transit = np.abs(folded_time) < half_dur

    if in_transit.sum() < 3:
        transit_depth = 0.0
        depth_uncertainty = np.inf
    else:
        in_flux = folded_flux[in_transit]
        in_err = folded_flux_err[in_transit]
        transit_depth = float(1.0 - np.mean(in_flux))
        # Propagate per-point errors through the mean
        depth_uncertainty = float(np.sqrt(np.sum(in_err ** 2)) / in_transit.sum())

    # --- SDE (Signal Detection Efficiency) ---
    # How many standard deviations above the noise floor is our peak?
    # Convention: SDE > 7 = candidate worth examining.
    power_mean = np.mean(power)
    power_std = np.std(power)
    sde = float((power.max() - power_mean) / power_std) if power_std > 0 else 0.0

    # --- SNR ---
    snr = float(transit_depth / depth_uncertainty) if np.isfinite(depth_uncertainty) and depth_uncertainty > 0 else 0.0

    # --- Aliases and reliability assessment ---
    aliases = _find_aliases(periods, power, best_period)
    is_reliable, flags = _assess_reliability(
        sde=sde,
        snr=snr,
        transit_depth=transit_depth,
        depth_uncertainty=depth_uncertainty,
        best_period=best_period,
        best_duration=best_duration,
        n_transit_points=int(in_transit.sum()),
        aliases=aliases,
        lc=lc,
    )

    logger.info(
        f"BLS result: period={best_period:.4f}d, depth={transit_depth:.6f}, "
        f"SDE={sde:.1f}, SNR={snr:.1f}, reliable={is_reliable}"
    )

    return BLSResult(
        best_period=best_period,
        best_duration=best_duration,
        best_t0=best_t0,
        transit_depth=transit_depth,
        depth_uncertainty=depth_uncertainty,
        sde=sde,
        snr=snr,
        periods=periods,
        power=power,
        folded_time=folded_time,
        folded_flux=folded_flux,
        folded_flux_err=folded_flux_err,
        aliases=aliases,
        is_reliable=is_reliable,
        reliability_flags=flags,
    )

def _find_aliases(
    periods: np.ndarray,
    power: np.ndarray,
    best_period: float,
    alias_threshold: float = 0.7,
) -> list[float]:
    """
    Find alias periods in the BLS power spectrum.

    An alias is a spurious peak caused by integer harmonics of the
    true period. We check P/2, P/3, 2P, 3P and flag any that have
    power above alias_threshold * peak_power.

    Parameters
    ----------
    periods : np.ndarray
        Period grid that was searched.
    power : np.ndarray
        BLS power at each period.
    best_period : float
        The best period found.
    alias_threshold : float
        Fraction of peak power above which we call something an alias.

    Returns
    -------
    list of float
        Alias periods found.
    """
    peak_power = power.max()
    alias_ratios = [0.5, 1/3, 2.0, 3.0, 0.25, 4.0]
    aliases = []

    for ratio in alias_ratios:
        alias_period = best_period * ratio
        if alias_period < periods.min() or alias_period > periods.max():
            continue

        # Find the closest period in our grid to the alias period
        idx = np.argmin(np.abs(periods - alias_period))
        # Check in a small window around that index
        window = slice(max(0, idx - 10), min(len(power), idx + 10))
        local_max_power = power[window].max()

        if local_max_power > alias_threshold * peak_power:
            aliases.append(round(float(alias_period), 4))

    return aliases


def _assess_reliability(
    sde: float,
    snr: float,
    transit_depth: float,
    depth_uncertainty: float,
    best_period: float,
    best_duration: float,
    n_transit_points: int,
    aliases: list,
    lc: LightCurveData,
) -> tuple[bool, list[str]]:
    """
    Honest reliability assessment of a BLS detection.

    Returns a boolean and a list of human-readable flags explaining
    any concerns. We err toward flagging rather than overclaiming.
    """
    flags = []

    # SDE threshold: the conventional minimum for a credible detection
    if sde < 7.0:
        flags.append(f"SDE={sde:.1f} is below the detection threshold of 7.0")

    # SNR threshold
    if snr < 5.0:
        flags.append(f"SNR={snr:.1f} is low — depth is not well-constrained")

    # Depth sanity check: transits should be between 10ppm and 5%
    # Smaller than 10ppm is below Kepler's noise floor
    # Larger than 5% is almost certainly an eclipsing binary, not a planet
    if transit_depth < 1e-5:
        flags.append("Transit depth < 10ppm — likely noise, not a real transit")
    if transit_depth > 0.05:
        flags.append(
            f"Transit depth={transit_depth:.1%} > 5% — "
            "may be an eclipsing binary rather than a planet"
        )

    # Duration sanity check
    # Transit duration > 0.5 * period is physically impossible
    if best_duration > 0.5 * best_period:
        flags.append("Transit duration > half the period — unphysical, likely a false positive")

    # Minimum in-transit points
    if n_transit_points < 5:
        flags.append(
            f"Only {n_transit_points} in-transit data points — "
            "depth measurement is poorly constrained"
        )

    # Alias warning
    if aliases:
        flags.append(
            f"Strong aliases detected at periods: {aliases} days — "
            "verify this is the true period, not a harmonic"
        )

    # Edge period warning: if best period is near the search boundary,
    # the algorithm may be running out of search space
    if best_period < 0.6:
        flags.append("Period near minimum search boundary — may be truncated")

    if not np.isfinite(depth_uncertainty):
        flags.append("Could not measure depth uncertainty — insufficient in-transit points")

    if transit_depth <= 0:
        flags.append(
            "Transit depth is negative — this is a brightening, not a transit. Almost certainly noise or a systematic artifact.")

    # Baseline coverage: we need at least 3 transits for confidence
    baseline_days = lc.time[-1] - lc.time[0]
    n_expected_transits = baseline_days / best_period
    if n_expected_transits < 3:
        flags.append(
            f"Only ~{n_expected_transits:.0f} expected transits in this dataset — "
            "need at least 3 for a reliable detection"
        )

    is_reliable = len(flags) == 0
    return is_reliable, flags




