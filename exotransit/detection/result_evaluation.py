import numpy as np

from exotransit.pipeline.light_curves import LightCurveData


def assess_reliability(
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
    Vetting logic modeled on NASA's Kepler TCE (Threshold Crossing Event)
    disposition pipeline (Jenkins et al. 2010, Batalha et al. 2010).

    A candidate must survive ALL of the following tests:

    SIGNAL STRENGTH TESTS
    ─────────────────────
    TCE-01  SDE floor            > 7.1σ  (Kepler pipeline minimum)
    TCE-02  SNR floor            > 7.1σ  (independent of SDE)
    TCE-03  MES (Multiple Event  requires ≥ 3 independent transit events
            Statistic) minimum   each contributing signal, not just data coverage

    GEOMETRIC PLAUSIBILITY TESTS
    ────────────────────────────
    TCE-04  Duration/period      duty cycle < 0.1  (EB discriminator)
    TCE-05  Minimum transit      ≥ 3 in-transit points (cadence check)
            point coverage
    TCE-06  Transit coverage     ≥ 70% of expected points (gap check)
    TCE-07  Depth floor          depth > 3σ above noise

    PERIODICITY TESTS
    ─────────────────
    TCE-08  Minimum transit      baseline / period ≥ 3.0 (raised from 2.5)
            repetitions          NASA requires 3 complete transit events
                                 for a TCE; 2.5 allowed partial-transit edge cases
    TCE-09  Alias contamination  strong alias present → flag

    FALSE POSITIVE DISCRIMINATORS
    ──────────────────────────────
    TCE-13  Depth ratio          depth > 0.03 (3%) suggests grazing EB,
                                 not a planet
    TCE-14  Very low depth       depth < 30 ppm with SNR < 10 is below
            with marginal SNR   Kepler's reliable detection floor

    NOTE: Eclipsing binary discrimination beyond TCE-04/13 (odd/even depth,
    secondary eclipse checks) is not implemented. Reliable EB rejection from
    photometry alone requires centroid motion analysis, spectroscopic follow-up,
    or space-based resolution — all out of scope here.
    """
    flags = []

    # ── TCE-01/02: Signal strength ────────────────────────────────────────────
    SDE_THRESHOLD = 12   # Determined empirically from testing
    SNR_THRESHOLD = 12   # Determined empirically from testing
    if sde < SDE_THRESHOLD:
        flags.append(f"TCE-01: SDE={sde:.1f} below {SDE_THRESHOLD} threshold (Jenkins+2010)")
    if snr < SNR_THRESHOLD:
        flags.append(f"TCE-02: SNR={snr:.1f} below {SNR_THRESHOLD} threshold")

    # ── TCE-08: Transit repetitions (raised to 3.0 from 2.5) ─────────────────
    # NASA's Kepler TCE definition requires ≥ 3 independent transit events.
    # 2.5 allowed edge cases where a partial transit straddles the baseline
    # boundary — but that's exactly the kind of marginal case that produces
    # false positives. Require 3.0 to be conservative.
    total_baseline = lc.time.max() - lc.time.min()
    n_expected_transits = total_baseline / best_period
    MIN_TRANSITS = 3.0
    if n_expected_transits < MIN_TRANSITS:
        flags.append(
            f"TCE-08: Only {n_expected_transits:.1f} transit windows in baseline "
            f"(need ≥ {MIN_TRANSITS})"
        )

    # ── TCE-03: Multiple Event Statistic — each transit must contribute ────────
    # A real planet produces a signal that grows as √N_transits.
    # If SNR / √(n_transits) is too low, the "detection" is driven by
    # noise piling up, not by a real repeating event.
    # Per-transit SNR < 1.5σ means individual transits are undetectable —
    # a red flag for noise artifacts.
    if n_expected_transits > 0:
        per_transit_snr = snr / np.sqrt(max(n_expected_transits, 1))
        if per_transit_snr < 1.5:
            flags.append(
                f"TCE-03: Per-transit SNR={per_transit_snr:.2f} < 1.5 — "
                f"signal may be noise accumulation, not real transits"
            )

    # ── TCE-04: Geometric plausibility — duration/period ratio ────────────────
    if (best_duration / best_period) > 0.1:
        flags.append(
            f"TCE-04: Duration/period={best_duration / best_period:.3f} > 0.1 "
            f"— physically implausible for a planet (eclipsing binary?)"
        )

    # TCE-05/06: In-transit data coverage
    cadence_days = np.median(np.diff(lc.time))
    expected_points = best_duration / cadence_days

    # BLS duration estimates are coarse (only 5 grid points).
    # Never penalize a detection for having fewer points than
    # 2 cadences worth — that's a BLS resolution limit, not a
    # real coverage problem.
    expected_points = max(expected_points, 2.0)

    coverage_ratio = n_transit_points / expected_points if expected_points > 0 else 0
    if n_transit_points < 3:
        flags.append("TCE-05: <3 in-transit points — cadence insufficient to resolve transit")
    elif coverage_ratio < 0.7:
        flags.append(
            f"TCE-06: Transit coverage {int(coverage_ratio * 100)}% "
            f"(need ≥ 70% of {expected_points:.1f} expected points)"
        )


    # ── TCE-07: Depth above noise floor ───────────────────────────────────────
    if transit_depth <= 0:
        flags.append("TCE-07a: Non-positive depth — brightening event, not a transit")
    elif transit_depth < (3.0 * depth_uncertainty):
        flags.append(
            f"TCE-07b: Depth {transit_depth:.6f} < 3σ={3 * depth_uncertainty:.6f} — "
            f"buried in noise floor"
        )

    # ── TCE-13: Very deep transit (eclipsing binary discriminator) ────────────
    # Planets transit at < 3% depth. Deeper signals are almost always
    # grazing eclipsing binaries or background contamination.
    if transit_depth > 0.03:
        flags.append(
            f"TCE-13: Transit depth {transit_depth * 100:.2f}% > 3% — "
            f"likely eclipsing binary, not a planet"
        )

    # ── TCE-14: Very shallow depth with marginal SNR ──────────────────────────
    # Below Kepler's reliable detection floor: depth < 30 ppm AND SNR < 10.
    # This combination is where instrumental systematics dominate.
    if transit_depth < 30e-6 and snr < 10.0:
        flags.append(
            f"TCE-14: Depth={transit_depth * 1e6:.1f} ppm < 30 ppm with SNR={snr:.1f} < 10 "
            f"— below reliable Kepler detection floor"
        )

    # ── TCE-09: Alias contamination ───────────────────────────────────────────
    if aliases:
        flags.append(
            f"TCE-09: Strong aliases at {[round(a, 4) for a in aliases]} days — "
            f"verify this is not a harmonic of a stronger signal"
        )

    # ── TCE-15: Absolute depth floor for long-cadence data ────────────────────
    # Kepler 30-min cadence noise floor is ~50-100 ppm for typical targets.
    # Detections below 60 ppm are unreliable without independent confirmation.
    cadence_hours = np.median(np.diff(lc.time)) * 24
    if cadence_hours > 0.6 and transit_depth < 60e-6:
        flags.append(
            f"TCE-15: Depth={transit_depth * 1e6:.1f} ppm below 60 ppm long-cadence "
            f"noise floor — unreliable without independent confirmation"
        )

    is_reliable = len(flags) == 0
    return is_reliable, flags
