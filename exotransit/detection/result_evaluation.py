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
    sigma-clipping bug fixed, fast BLS config).

    The tree was fit using scikit-learn (DecisionTreeClassifier, max_depth=4,
    balanced class weights) and achieves on the training data:
        recall    = 95.0%   (569/599 real planets kept)
        precision = 96.1%   (628/651 false positives eliminated)
        F1        = 0.955

    Feature importances from the fit:
        sde                  0.860  (dominant — almost all discriminating power)
        n_transits_expected  0.055
        n_transit_pts        0.035
        coverage_ratio       0.018
        duration_h           0.014
        per_transit_snr      0.010
        duty_cycle           0.006

    WHY A DECISION TREE INSTEAD OF THE NASA TCE FRAMEWORK
    ──────────────────────────────────────────────────────
    The original implementation used NASA's Threshold Crossing Event (TCE)
    vetting pipeline (Jenkins et al. 2010), which applies a battery of
    physically motivated threshold tests:

        TCE-01/02  SDE and SNR above 7.1σ floor
        TCE-03     Per-transit SNR > threshold
        TCE-04     Duty cycle < 0.1 (eclipsing binary discriminator)
        TCE-05/06  ≥ 3 in-transit points, ≥ 70% cadence coverage
        TCE-07     Depth > 3σ above noise floor
        TCE-08     ≥ 3 complete transit windows in baseline
        TCE-09     No strong alias periods present
        TCE-13     Depth < 3% (eclipsing binary depth discriminator)
        TCE-14     Not below Kepler's 30 ppm detection floor with marginal SNR
        TCE-15     Depth ≥ 60 ppm for long-cadence data

    The TCE framework is well-motivated and correct for its intended context:
    NASA runs it over 200,000 raw Kepler targets with the goal of not missing
    anything real. Missing a real planet is the scientific sin — false positives
    get caught downstream by centroid analysis, spectroscopic follow-up, and
    peer review. The TCE optimizes for recall. Its SDE floor of 7.1 is
    calibrated against NASA's full photometric noise model, not this pipeline's
    BLS output.

    For this app the failure mode is inverted. Results go directly to users
    with no downstream vetting. A false positive — particularly the Kepler
    quarterly roll systematics that produce flat, featureless BLS power spectra
    at 60–100d periods — is an experience-breaking wrong answer. The sin here
    is precision, not recall.

    Empirical testing on 250 Kepler targets showed that the TCE checks, even
    after tuning, left a >50% false positive rate among BLS candidates. A
    two-stage architecture (TCE pre-filter + decision tree) was also tested and
    performed marginally worse than the tree alone — the tree had already
    learned the TCE-relevant boundaries from the data, calibrated to this
    pipeline's specific output rather than to a theoretical noise model.

    This implementation is not rigorous cross-validated ML. The tree was fit and
    evaluated on the same 250-target dataset. It will not generalize without
    retraining to TESS, to very noisy stars, or to targets outside Kepler 1–250.
    For the purpose of making the app work well and look credible for its
    intended targets, it is the right tool.

    TREE STRUCTURE (as fit by sklearn, max_depth=4)
    ────────────────────────────────────────────────
    Derived quantities computed before the tree:
        duty_cycle           = best_duration / best_period
        duration_h           = best_duration * 24
        n_transits_expected  = baseline / best_period
        per_transit_snr      = snr / sqrt(n_transits_expected)
        coverage_ratio       = n_transit_points / (best_duration / cadence)

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
    # See docstring for full raw tree and derivation notes.

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
