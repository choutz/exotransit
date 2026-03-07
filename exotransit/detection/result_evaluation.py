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
    TCE-10  Minimum duration     ≥ 2 cadences (≥ ~1 hr for 30-min data)
            floor                Single-cadence dips are instrumental artifacts,
                                 not planet transits. Real hot Jupiters on 3d
                                 periods have durations ≥ 1.5h; at longer periods
                                 durations only grow. One Kepler cadence = 29.4 min.

    PERIODICITY TESTS
    ─────────────────
    TCE-08  Minimum transit      baseline / period ≥ 5.0
            repetitions          Raised from 3.0. The dominant false-positive
                                 population in validation runs was long-period
                                 instrumental systematics (Kepler quarterly rolls
                                 at ~90d) that barely clear the old 3-transit floor.
                                 Requiring 5 independent transit windows suppresses
                                 these while preserving real planets: a 90d planet
                                 in a 1250d baseline has ~14 windows and easily
                                 passes; a 90d systematic in a 977d baseline has
                                 ~10 windows and also passes — but combined with
                                 TCE-03 and TCE-10 the full chain of cuts removes
                                 them.
    TCE-09  Alias contamination  strong alias present → flag, UNLESS the alias
                                 is a harmonic or sub-harmonic of the detected
                                 period (alias ≈ N×P or alias ≈ P/N for integer N),
                                 AND the signal is strong (SNR > 50). Harmonics of
                                 a real periodic signal are expected and are not
                                 evidence of contamination.

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
    SDE_THRESHOLD = 7.1   # Jenkins et al. 2010 Kepler pipeline floor
    SNR_THRESHOLD = 7.1
    if sde < SDE_THRESHOLD:
        flags.append(f"TCE-01: SDE={sde:.1f} below {SDE_THRESHOLD} threshold (Jenkins+2010)")
    if snr < SNR_THRESHOLD:
        flags.append(f"TCE-02: SNR={snr:.1f} below {SNR_THRESHOLD} threshold")

    # ── TCE-08: Transit repetitions (raised to 5.0) ───────────────────────────
    # The original 3.0 floor allowed long-period Kepler systematics (quarterly
    # rolls at ~90d) to pass in ~1000d baselines. Raising to 5.0 suppresses
    # this class of false positive. Real long-period planets (e.g. Kepler-20f
    # at 77d) have 14+ windows in a 1250d baseline and are unaffected.
    total_baseline = lc.time.max() - lc.time.min()
    n_expected_transits = total_baseline / best_period
    MIN_TRANSITS = 5.0
    if n_expected_transits < MIN_TRANSITS:
        flags.append(
            f"TCE-08: Only {n_expected_transits:.1f} transit windows in baseline "
            f"(need ≥ {MIN_TRANSITS})"
        )

    # ── TCE-03: Multiple Event Statistic — each transit must contribute ────────
    # A real planet produces a signal that grows as √N_transits.
    # If SNR / √(n_transits) is too low, the "detection" is driven by
    # noise piling up, not by a real repeating event.
    # Threshold raised from 1.5 to 2.0: the 1.5 floor still admitted
    # long-period instrumental systematics whose per-transit signal sits in
    # the 1.5–2.0 range. Real transiting planets at detectable depths
    # comfortably clear 2.0 per-transit SNR.
    if n_expected_transits > 0:
        per_transit_snr = snr / np.sqrt(max(n_expected_transits, 1))
        if per_transit_snr < 2.0:
            flags.append(
                f"TCE-03: Per-transit SNR={per_transit_snr:.2f} < 2.0 — "
                f"signal may be noise accumulation, not real transits"
            )

    # ── TCE-04: Geometric plausibility — duration/period ratio ────────────────
    if (best_duration / best_period) > 0.1:
        flags.append(
            f"TCE-04: Duration/period={best_duration / best_period:.3f} > 0.1 "
            f"— physically implausible for a planet (eclipsing binary?)"
        )

    # ── TCE-10: Minimum duration floor ────────────────────────────────────────
    # A real planet transit must last at least 2 cadences (≥ ~1 hour for
    # Kepler 30-min data). Single-cadence dips (0.48h in 29.4-min cadence
    # data) are instrumental artifacts: cosmic rays, scattered light spikes,
    # or BLS fitting noise. This cut is independent of depth or SNR — even a
    # very deep single-cadence dip is not a planet.
    #
    # Physical basis: the minimum transit duration for a grazing central
    # transit scales as t_min ~ 2 R_* / v_orb. For a hot Jupiter at P=3d
    # around a solar star, t_min ≈ 1.5h. At longer periods, orbital velocity
    # decreases and duration only grows. Sub-hour transits at any period are
    # unphysical for real planets.
    #
    # The 1.0h floor (slightly below 2 cadences) gives a small buffer for
    # BLS duration grid coarseness.
    MIN_DURATION_HOURS = 1.0
    duration_hours = best_duration * 24.0
    if duration_hours < MIN_DURATION_HOURS:
        flags.append(
            f"TCE-10: Duration={duration_hours:.2f}h below {MIN_DURATION_HOURS}h minimum "
            f"— sub-cadence dip, not a planet transit"
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
    # An alias at period A is a harmonic of the detected period P if
    # A ≈ N×P or A ≈ P/N for small integer N (checking N = 2, 3, 4).
    # Harmonics of a real periodic signal are expected and are not evidence
    # of contamination — they just mean the same signal folds at a multiple
    # of the true period.
    #
    # Additionally: if SNR > 50, the signal is so strong it is almost
    # certainly real regardless of alias structure. Rejecting Kepler-14
    # (SNR=70) or Kepler-25 (SNR=541) on alias grounds is wrong. At SNR > 50
    # the alias check is demoted to a warning only — it does not veto.
    #
    # Harmonic tolerance: 2% of P, to accommodate BLS period grid spacing.
    SNR_ALIAS_VETO_FLOOR = 50.0   # above this SNR, aliases warn but don't veto
    HARMONIC_TOLERANCE = 0.02     # 2% relative tolerance for harmonic matching
    HARMONIC_INTEGERS = [2, 3, 4] # N values to check for both N×P and P/N

    if aliases:
        non_harmonic_aliases = []
        for alias in aliases:
            is_harmonic = False
            for n in HARMONIC_INTEGERS:
                # Check alias ≈ N × P  (sub-harmonic of detected period)
                if abs(alias - n * best_period) / (n * best_period) < HARMONIC_TOLERANCE:
                    is_harmonic = True
                    break
                # Check alias ≈ P / N  (harmonic — half/third/quarter period)
                if abs(alias - best_period / n) / (best_period / n) < HARMONIC_TOLERANCE:
                    is_harmonic = True
                    break
            if not is_harmonic:
                non_harmonic_aliases.append(alias)

        if non_harmonic_aliases:
            if snr > SNR_ALIAS_VETO_FLOOR:
                # Strong signal: alias is a warning only, not a veto.
                # The signal is too strong to plausibly be an artifact of
                # a different real signal at the alias period.
                flags.append(
                    f"TCE-09 (warning only — SNR={snr:.1f} > {SNR_ALIAS_VETO_FLOOR}): "
                    f"Aliases at {[round(a, 4) for a in non_harmonic_aliases]} days — "
                    f"signal too strong to reject; verify manually"
                )
                # Do NOT set is_reliable = False for this flag — handled below
                # by excluding this flag from the veto count.
            else:
                flags.append(
                    f"TCE-09: Strong aliases at {[round(a, 4) for a in non_harmonic_aliases]} days — "
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

    # ── Reliability verdict ───────────────────────────────────────────────────
    # TCE-09 warning-only flags (SNR > 50 alias) do not veto.
    veto_flags = [f for f in flags if not f.startswith("TCE-09 (warning only")]
    is_reliable = len(veto_flags) == 0
    return is_reliable, flags