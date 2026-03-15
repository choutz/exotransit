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
    Reliability vetting via a decision tree classifier trained on 1,250 labeled
    BLS candidates from 250 Kepler targets (Kepler 1–250 permissive validation run,
    fast BLS config).

    Raw sklearn tree:

        |--- sde <= 15.12
        |   |--- sde <= 9.91
        |   |   |--- duty_cycle <= 0.01
        |   |   |   |--- depth_ppm <= 163.19  → class: 0
        |   |   |   |--- depth_ppm >  163.19  → class: 0
        |   |   |--- duty_cycle >  0.01
        |   |   |   |--- duty_cycle <= 0.01   → class: 1   ← impossible branch
        |   |   |   |--- duty_cycle >  0.01   → class: 0
        |   |--- sde >  9.91
        |   |   |--- n_transits_expected <= 13.75
        |   |   |   |--- n_transit_pts <= 120.50  → class: 0
        |   |   |   |--- n_transit_pts >  120.50  → class: 0
        |   |   |--- n_transits_expected >  13.75
        |   |   |   |--- duration_h <= 7.92   → class: 1
        |   |   |   |--- duration_h >  7.92   → class: 0
        |--- sde >  15.12
        |   |--- n_transit_pts <= 65.00        → class: 0
        |   |--- n_transit_pts >  65.00
        |   |   |--- coverage_ratio <= 277.04
        |   |   |   |--- duration_h <= 10.80  → class: 1
        |   |   |   |--- duration_h >  10.80  → class: 1
        |   |   |--- coverage_ratio >  277.04
        |   |   |   |--- per_transit_snr <= 2.66  → class: 1
        |   |   |   |--- per_transit_snr >  2.66  → class: 0

    Dead branches collapsed (class preserved):

    if sde <= 15.12:
        if sde <= 9.91:
            → FALSE POSITIVE
            (all leaves under sde ≤ 9.91 are class 0, including the
             impossible duty_cycle > 0.01 → duty_cycle ≤ 0.01 re-split)
        else:                                     # 9.91 < sde <= 15.12
            if n_transits_expected > 13.75 and duration_h <= 7.92:
                → REAL
            else:
                → FALSE POSITIVE
                (n_transits_expected ≤ 13.75 has both leaves class 0)
    else:                                         # sde > 15.12
        if n_transit_pts <= 65:
            → FALSE POSITIVE
        else:                                     # n_transit_pts > 65
            if coverage_ratio <= 277.04:
                → REAL
                (both duration_h leaves are class 1 — collapsed)
            else:                                 # coverage_ratio > 277.04
                if per_transit_snr <= 2.66:
                    → REAL
                else:
                    → FALSE POSITIVE
    """
    flags = []

    # ── Derived features (match training feature set exactly) ─────────────────
    total_baseline      = lc.time.max() - lc.time.min()
    cadence_days        = float(np.median(np.diff(lc.time)))
    n_transits_expected = total_baseline / best_period
    per_transit_snr     = snr / np.sqrt(max(n_transits_expected, 1))
    duty_cycle          = best_duration / best_period
    duration_h          = best_duration * 24.0
    expected_points     = max(best_duration / cadence_days, 2.0)
    coverage_ratio      = n_transit_points / expected_points if expected_points > 0 else 0.0

    # ── Decision tree ──────────────────────────────────────────────────────────
    # Implements the sklearn tree with dead branches collapsed.

    if sde <= 15.12:

        if sde <= 9.91:
            # All leaves under this branch are class 0 — every combination of
            # duty_cycle and depth produces a false positive. Below SDE=9.91
            # the signal is too weak to trust regardless of other features.
            flags.append(
                f"TREE: sde={sde:.2f} ≤ 9.91 — signal too weak to be reliable"
            )

        else:
            # 9.91 < sde <= 15.12 — moderate signal. Real only if there are
            # enough transit windows AND the duration is physically plausible.
            # n_transits_expected ≤ 13.75 has both leaves class 0 (collapsed).
            if n_transits_expected > 13.75 and duration_h <= 7.92:
                pass  # → class 1 (real)
            else:
                flags.append(
                    f"TREE: sde={sde:.2f} in (9.91, 15.12], "
                    f"failed n_transits_expected > 13.75 and duration_h ≤ 7.92 "
                    f"(n_transits={n_transits_expected:.1f}, duration_h={duration_h:.2f})"
                )

    else:
        # sde > 15.12 — strong signal. The tree focuses on data coverage quality
        # to catch the remaining false positives.

        if n_transit_points <= 65:
            # Too few in-transit data points for a confident detection even at
            # high SDE — likely a short-duration systematic or sparse cadence
            # hitting a few outlier points.
            flags.append(
                f"TREE: sde={sde:.2f} > 15.12 but n_transit_pts={n_transit_points} ≤ 65 "
                f"— insufficient in-transit coverage"
            )

        else:
            # n_transit_pts > 65: good coverage. coverage_ratio distinguishes
            # normal transits from anomalously wide ones.
            if coverage_ratio <= 277.04:
                # Normal coverage ratio — real regardless of duration.
                # Both duration_h leaves (≤ 10.80 and > 10.80) are class 1.
                pass  # → class 1 (real)
            else:
                # coverage_ratio > 277.04: anomalously high — far more in-transit
                # points than the duration implies. Per-transit SNR distinguishes
                # genuine strong signals from duration overestimates.
                if per_transit_snr <= 2.66:
                    pass  # → class 1 (real)
                else:
                    flags.append(
                        f"TREE: sde={sde:.2f} > 15.12, n_transit_pts={n_transit_points} > 65, "
                        f"coverage_ratio={coverage_ratio:.1f} > 277.04, "
                        f"per_transit_snr={per_transit_snr:.2f} > 2.66 "
                        f"— anomalous coverage with high per-transit SNR, likely systematic"
                    )

    # ── Hard physical limits (applied after tree, as absolute vetoes) ──────────
    # These are not in the tree because they were filtered before training —
    # no candidate with a non-positive depth or a duty cycle > 0.1 appeared
    # in the labeled dataset. They remain as sanity checks.

    if transit_depth <= 0:
        flags.append(
            "HARD: Non-positive transit depth — brightening event, not a transit"
        )

    if duty_cycle > 0.1:
        flags.append(
            f"HARD: Duty cycle={duty_cycle:.3f} > 0.1 — "
            f"physically implausible for a planet (eclipsing binary?)"
        )

    if transit_depth > 0.03:
        flags.append(
            f"HARD: Transit depth {transit_depth * 100:.2f}% > 3% — "
            f"likely eclipsing binary, not a planet"
        )

    is_reliable = len(flags) == 0
    return is_reliable, flags
