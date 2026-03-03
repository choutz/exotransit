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
from tqdm import tqdm

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
    max_period_grid_points: int = 200_000,
) -> BLSResult:
    """
    Run BLS (Box Least Squares) period search on a preprocessed light curve.

    BLS works by trying every combination of period and duration in a grid,
    and for each combination phase-folding the light curve and fitting a
    box-shaped dip. The "power" at each period is basically how much
    better a box model fits the data compared to a flat line. A sharp peak
    in the power spectrum at some period P means the data has a repeating
    box-shaped dip every P days — the signature of a transiting planet.

    Period grid: 100,000 linearly-spaced frequencies converted to periods.
    Linear frequency spacing ensures uniform sampling of transit repetition
    rate rather than orbital period.

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

    # Linear in frequency = uniform sampling of transit repetition rate
    n_points = min(int(100_000 * (max_period / 30.0)), max_period_grid_points)
    period_grid = 1.0 / np.linspace(1.0 / max_period, 1.0 / min_period, n_points)
    durations = np.linspace(0.05, 0.5, 5)  # days: 30 mins to 12hr

    model = BoxLeastSquares(lc_lk.time, lc_lk.flux, lc_lk.flux_err)
    pg_results = model.power(period_grid, durations)
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
        transit_depth = float(1.0 - np.mean(in_flux))
        depth_uncertainty = float(np.sqrt(np.sum(in_err ** 2)) / in_transit.sum())

    power_std = np.std(power)

    # SDE: sigma above noise floor. Convention: > 7 = candidate worth examining
    sde = float((power.max() - np.mean(power)) / power_std) if power_std > 0 else 0.0
    snr = float(transit_depth / depth_uncertainty) if np.isfinite(depth_uncertainty) and depth_uncertainty > 0 else 0.0

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


def find_all_planets(
    lc: LightCurveData,
    max_planets: int = 5,
    min_period: float = 0.5,
    max_period: float = 30.0,
    max_period_grid_points: int = 100_000,
) -> list[BLSResult]:
    """
    Iteratively search for multiple planets via period masking.

    After each detection, the transits at that period are masked out
    of the light curve before re-running BLS. This lets the algorithm
    find weaker signals that were previously buried under a stronger one.
    Stops when BLS returns an unreliable result or max_planets is reached.

    max_period_grid_points caps memory usage — important for deployment
    on memory-constrained environments like Streamlit Community Cloud (1GB).

    Parameters
    ----------
    lc : LightCurveData
        Preprocessed light curve.
    max_planets : int
        Maximum number of planets to search for.
    min_period, max_period : float
        Period search bounds in days, passed through to run_bls.
    max_period_grid_points : int
        Hard cap on period grid size. Prevents OOM on low-memory hosts.

    Returns
    -------
    list[BLSResult]
        One entry per detected planet, in order of detection strength.
    """
    lc_lk = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
    results = []

    with tqdm(total=max_planets, desc="Searching for planets", unit="planet") as pbar:
        for i in range(max_planets):
            pbar.set_description(f"Searching for planet {i + 1}")

            current_lc = LightCurveData(
                time=np.asarray(lc_lk.time.value),
                flux=np.asarray(lc_lk.flux.value),
                flux_err=np.asarray(lc_lk.flux_err.value),
                mission=lc.mission,
                target_name=lc.target_name,
                sector_or_quarter=lc.sector_or_quarter,
                raw_time=lc.raw_time,
                raw_flux=lc.raw_flux,
            )

            result = run_bls(
                current_lc,
                min_period=min_period,
                max_period=max_period,
                max_period_grid_points=max_period_grid_points,
            )

            if not result.is_reliable:
                pbar.set_description(f"Stopping after {i} reliable planet(s)")
                pbar.update(max_planets - i)
                break

            results.append(result)
            pbar.set_postfix(period=f"{result.best_period:.2f}d", sde=f"{result.sde:.1f}")
            pbar.update(1)
            logger.info(f"Planet {i + 1} at {result.best_period:.4f}d, masking and re-searching...")

            mask = lc_lk.create_transit_mask(
                period=result.best_period,
                transit_time=result.best_t0,
                duration=result.best_duration * 1.5,
            )
            lc_lk = lc_lk[~mask]

    return results



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
    Vetting logic based on NASA's threshold crossing event (TCE) standards.
    """
    flags = []

    # Check signal strength: NASA's Kepler pipeline uses a 7.1 sigma floor
    if sde < 7.0:
        flags.append("low signal detection efficiency")
    if snr < 7.1:
        flags.append("low signal to noise ratio")

    # Check data coverage: Confirming a period requires seeing it repeat
    total_baseline = lc.time.max() - lc.time.min()
    if (total_baseline / best_period) < 2.5:
        flags.append("insufficient number of transits")

    # Check for eclipsing binaries: Planets rarely block >10% of their orbit
    if (best_duration / best_period) > 0.1:
        flags.append("physically implausible transit duration")

    # Check noise floor: Signal must be clear of the local flux scatter
    if transit_depth < (3 * depth_uncertainty):
        flags.append("depth buried in noise floor")

    # Coverage/sampling check: cadence-aware, works across Kepler/TESS/K2
    cadence_days = np.median(np.diff(lc.time))
    expected_points = best_duration / cadence_days
    coverage_ratio = n_transit_points / expected_points if expected_points > 0 else 0
    if n_transit_points < 3:
        flags.append("critically low point count in transit")
    elif coverage_ratio < 0.7:
        flags.append(f"poor transit coverage: {int(coverage_ratio * 100)}% of expected points captured")

    if aliases:
        flags.append(f"strong aliases at {aliases} days — verify true period")

    if transit_depth <= 0:
        flags.append("negative depth — brightening event, not a transit")

    # Reliability is true only if the candidate clears all hurdles
    is_reliable = len(flags) == 0

    return is_reliable, flags