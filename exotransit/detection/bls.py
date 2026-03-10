"""
BLS (Box Least Squares) period search for exoplanet transits.

BLS was introduced by Kovacs, Zucker & Mazeh (2002) and remains
the standard algorithm for transit detection in Kepler photometry.
It searches for periodic, box-shaped dips in a light curve by
testing thousands of period/duration combinations and scoring
each by how well a box model fits the data.

We wrap lightkurve's BLS implementation and add:
- Full power spectrum output for visualization
- SDE (Signal Detection Efficiency) calculation
- Alias detection (harmonics that can fool the algorithm)
- Honest uncertainty flags
"""



from dataclasses import dataclass, field

import lightkurve as lk
import numpy as np
from astropy.timeseries import BoxLeastSquares

from exotransit.detection.result_evaluation import assess_reliability
from exotransit.pipeline.light_curves import LightCurveData
import logging

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
        other by dividing time offset from t0 by the period. If the period is right,
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
        Our assessment of whether this detection is trustworthy.
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
    max_period_grid_points: int = 200_000,
) -> BLSResult:
    """
    Run BLS (Box Least Squares) period search on a preprocessed light curve.

    BLS works by trying every combination of period and duration in a grid,
    and for each combination phase-folding the light curve and fitting a
    box-shaped dip. The "power" at each period is basically how much
    better a box model fits the data compared to a flat line. A sharp peak
    in the power spectrum at some period P means the data has a repeating
    box-shaped dip every P days, the signature of a transiting planet.

    Period grid: configurable number of linearly-spaced frequencies converted to periods.
    Linear frequency spacing ensures uniform sampling of transit repetition
    rate rather than orbital period.

    Duration grid: 5 values from 30 minutes to upper limit bounded by min_period constraint.
    Missing the exact duration only weakens the signal slightly; missing the period loses
    it entirely. So we search periods finely and durations coarsely.

    Parameters
    ----------
    lc : LightCurveData
        Output from fetch.fetch_light_curve().
    min_period : float
        Minimum orbital period to search in days.
    max_period : float
        Maximum orbital period to search in days. Default 30 days —
        with 90 days of data you need at least 3 transits, so
        max reliable period ≈ baseline / 3.
    max_period_grid_points : int
        Hard cap on period grid size. Prevents OOM on memory-constrained
        hosts like Streamlit Community Cloud (1GB RAM limit).

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

    lc_lk = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)

    # 1. Calculate the baseline (total time span)
    # This is the most critical factor for period resolution.
    baseline = np.ptp(lc_lk.time.value)  # .value ensures we are working with floats

    # 2. Define Duration Grid (Geometric Spacing)
    # We use geomspace because a 5-minute error on a 30-minute transit
    # is much more significant than a 5-minute error on a 5-hour transit.
    dur_min = 0.02  # ~30 mins
    dur_max = min(0.5, 0.15 * min_period)  # Max duration shouldn't exceed min period constraints

    # 12 to 15 steps is usually the "sweet spot" for accuracy vs speed
    durations = np.geomspace(dur_min, dur_max, 12)

    # 3. Calculate Period Grid (Frequency Spacing)
    # The frequency step (df) should be small enough so the transit doesn't
    # shift by more than a fraction of its duration over the whole baseline.
    # Logic: df = frequency_factor * min_duration / baseline^2
    frequency_factor = 1.0
    df = (frequency_factor * dur_min) / (baseline ** 2)

    f_min = 1.0 / max_period
    f_max = 1.0 / min_period

    # Determine number of points based on the required frequency resolution
    n_points = int(np.ceil((f_max - f_min) / df))

    # Safety cap to prevent memory errors on extremely long baselines (e.g., years of data)
    n_points = min(n_points, max_period_grid_points)

    # Generate the grid uniform in frequency, then convert to periods
    period_grid = 1.0 / np.linspace(f_max, f_min, n_points)

    # 4. Run the Model
    model = BoxLeastSquares(lc_lk.time, lc_lk.flux, lc_lk.flux_err)
    pg_results = model.power(period_grid, durations, oversample=10)


    periods = np.asarray(pg_results.period)
    power = np.asarray(pg_results.power)

    best_idx = np.argmax(pg_results.power)
    best_period = pg_results.period[best_idx].value
    best_duration = pg_results.duration[best_idx].value
    best_t0 = float(pg_results.transit_time[best_idx].value)

    # Phase fold: stack all transits at phase 0 to boost SNR
    folded = lc_lk.fold(period=best_period, epoch_time=best_t0)
    folded_time = np.asarray(folded.time.value)
    folded_flux = np.asarray(folded.flux.value)
    folded_flux_err = np.asarray(folded.flux_err.value)

    # folded_time is in days centered on 0. half_dur is a day threshold, not a phase fraction.
    cadence_days = np.median(np.diff(lc.time))
    half_dur = max(
        best_duration / 2,
        1.5 * cadence_days,  # minimum 1.5 cadence lengths to catch sparse transits
    )
    in_transit = np.abs(folded_time) < half_dur

    if in_transit.sum() < 3:
        transit_depth = 0.0
        depth_uncertainty = np.inf
    else:
        in_flux = folded_flux[in_transit]
        in_err = folded_flux_err[in_transit]

        # Use bottom 30% of in-transit points to estimate depth.
        # np.mean(in_flux) dilutes depth by including ingress/egress —
        # for a 30-min cadence with ~10 in-transit points this causes
        # systematic 2-5x underestimation of true transit depth.
        n_bottom = max(1, int(len(in_flux) * 0.3))
        bottom_idx = np.argsort(in_flux)[:n_bottom]
        bottom_flux = in_flux[bottom_idx]
        bottom_err = in_err[bottom_idx]

        transit_depth = float(1.0 - np.mean(bottom_flux))
        depth_uncertainty = float(np.sqrt(np.sum(bottom_err ** 2)) / n_bottom)


    power_std = np.std(power)

    # SDE: sigma above noise floor.
    sde = float((power.max() - np.mean(power)) / power_std) if power_std > 0 else 0.0
    snr = float(transit_depth / depth_uncertainty) if np.isfinite(depth_uncertainty) and depth_uncertainty > 0 else 0.0

    aliases = _find_aliases(periods, power, best_period)
    is_reliable, flags = assess_reliability(
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
    if not is_reliable:
        logger.warning(
            f"[{lc.target_name}] BLS candidate REJECTED — "
            f"period={best_period:.4f}d, depth={transit_depth*1e6:.1f}ppm, "
            f"SDE={sde:.2f}, SNR={snr:.2f}, "
            f"in_transit_pts={int(in_transit.sum())}, "
            f"n_transits_expected={((lc.time.max()-lc.time.min())/best_period):.1f}"
        )
        for flag in flags:
            logger.warning(f"  [{lc.target_name}]  {flag}")

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
