"""
tests/threshold_optimization/run_permissive_test.py

Runs the BLS pipeline on all Kepler targets with reliability vetting
completely disabled. Every BLS candidate — real planet or noise peak —
is logged with its full feature vector, then labeled real/FP by
period-matching against NASA truth data.

The output CSV is the training dataset for fit_thresholds.py, which
uses it to find empirically optimal vetting thresholds.

Why permissive mode?
    Normal mode stops when a candidate fails vetting. This means we only
    observe candidates that the CURRENT thresholds allow through — we
    never see the FP population that the thresholds correctly suppress,
    so we cannot measure whether different thresholds would do better.
    Permissive mode shows us everything: the full joint distribution of
    (features, label) across the real and false positive populations.

Masking:
    Even in permissive mode we median-fill detected signals after each
    iteration, otherwise BLS re-finds the same dominant peak forever.
    We mask anything with SDE > 3 (basically everything above pure noise).
    We stop a target's search when SDE drops below 3 or we reach max_iters.

Output:
    tests/threshold_optimization/candidates_YYYYMMDD_HHMMSS.csv
    tests/threshold_optimization/permissive_test_YYYYMMDD_HHMMSS.log

Run from project root:
    caffeinate -i python tests/threshold_optimization/run_permissive_test.py
"""

import sys
import logging
import numpy as np
import pandas as pd
import lightkurve as lk
from pathlib import Path
from datetime import datetime, timedelta
import time

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MEDIUM
from tests.helpers import get_light_curve
from exotransit.detection.bls import run_bls
from exotransit.pipeline.light_curves import LightCurveData
from exotransit.physics.stars import query_stellar_params

# ── Logging ────────────────────────────────────────────────────────────────────
OUT_DIR   = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = OUT_DIR / f"permissive_test_{timestamp}.log"
CSV_FILE  = OUT_DIR / f"candidates_{timestamp}.csv"

fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
formatter = logging.Formatter(fmt)

root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers.clear()

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(formatter)
root.addHandler(_sh)

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(formatter)
root.addHandler(_fh)

logger = logging.getLogger("permissive_test")

# Suppress noisy sub-loggers
for _mod in ["lightkurve", "lightkurve.utils", "exotransit.detection.bls",
             "exotransit.detection.multi_planet", "exotransit.pipeline.light_curves",
             "exotransit.physics.stars", "exotransit.physics.limb_darkening"]:
    logging.getLogger(_mod).setLevel(logging.ERROR)


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(result, lc: LightCurveData) -> dict:
    """
    Compute the full feature vector for a BLS candidate.
    Replicates the quantities computed inside assess_reliability so the
    optimizer has the same information the vetting logic uses.
    """
    baseline      = lc.time.max() - lc.time.min()
    cadence_days  = float(np.median(np.diff(lc.time)))

    n_transits_expected = baseline / result.best_period
    per_transit_snr     = result.snr / np.sqrt(max(n_transits_expected, 1))
    duty_cycle          = result.best_duration / result.best_period
    duration_h          = result.best_duration * 24.0

    # n_transit_points: recompute from folded data (same method as bls.py)
    try:
        lc_lk  = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
        folded = lc_lk.fold(period=result.best_period, epoch_time=result.best_t0)
        half_dur = max(result.best_duration / 2, 1.5 * cadence_days)
        n_transit_pts = int(np.sum(np.abs(folded.time.value) < half_dur))
    except Exception:
        n_transit_pts = 0

    # Coverage ratio (TCE-06)
    expected_pts   = max(result.best_duration / cadence_days, 2.0)
    coverage_ratio = n_transit_pts / expected_pts if expected_pts > 0 else 0.0

    return {
        "sde":                 result.sde,
        "snr":                 result.snr,
        "depth_ppm":           result.transit_depth * 1e6,
        "depth_unc_ppm":       result.depth_uncertainty * 1e6,
        "n_transit_pts":       n_transit_pts,
        "n_transits_expected": n_transits_expected,
        "per_transit_snr":     per_transit_snr,
        "duty_cycle":          duty_cycle,
        "duration_h":          duration_h,
        "coverage_ratio":      coverage_ratio,
        "has_aliases":         int(len(result.aliases) > 0),
        "n_aliases":           len(result.aliases),
        "period_d":            result.best_period,
        "baseline_d":          baseline,
    }


# ── Per-target permissive search ───────────────────────────────────────────────

def run_target_permissive(
    target_name: str,
    truth_periods: list[float],
    truth_n: int,
    conf,
) -> list[dict]:
    """
    Run BLS iteratively with no reliability gate. Returns a list of
    candidate feature dicts, each with is_real labeled.
    """
    logger.info(f"{'='*60}")
    logger.info(f"TARGET: {target_name}  (truth: {truth_n} planet(s), periods: {[f'{p:.3f}' for p in truth_periods]})")

    # ── Light curve ────────────────────────────────────────────────────────────
    try:
        lc = get_light_curve(target_name, mission="Kepler", max_quarters=conf.max_quarters)
        logger.info(f"  LC: {len(lc.time):,} pts  baseline {lc.time[-1]-lc.time[0]:.1f} d")
    except Exception as e:
        logger.error(f"  LC FETCH FAILED: {e}")
        return []

    # ── Permissive BLS loop ────────────────────────────────────────────────────
    lc_lk    = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
    # max_iters = max(truth_n + 15, 20)
    max_iters = 6
    candidates = []

    for i in range(max_iters):
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

        try:
            result = run_bls(
                current_lc,
                min_period=conf.bls.min_period,
                max_period=conf.bls.max_period,
                max_period_grid_points=conf.bls.max_period_grid_points,
            )
        except Exception as e:
            logger.warning(f"  iter {i}: BLS failed: {e}")
            break

        # Stop if the signal is pure noise
        if result.sde < 3.0:
            logger.info(f"  iter {i}: SDE={result.sde:.2f} < 3.0 — stopping (noise floor)")
            break

        # Label this candidate before masking
        is_real = int(any(
            abs(result.best_period - tp) / tp < 0.05
            for tp in truth_periods
        ))

        features = extract_features(result, current_lc)
        row = {
            "target":     target_name,
            "iteration":  i,
            "is_real":    is_real,
            **features,
        }
        candidates.append(row)

        label = "REAL" if is_real else "FP"
        logger.info(
            f"  iter {i}: P={result.best_period:.4f}d  SDE={result.sde:.2f}  "
            f"SNR={result.snr:.2f}  depth={result.transit_depth*1e6:.1f}ppm  [{label}]"
        )

        # Mask this signal (SDE > 3 = mask it, otherwise pure noise, stop)
        try:
            mask = lc_lk.create_transit_mask(
                period=result.best_period,
                transit_time=result.best_t0,
                duration=result.best_duration * 3.0,
            )
            lc_lk.flux.value[mask] = np.nanmedian(lc_lk.flux.value)
        except Exception as e:
            logger.warning(f"  iter {i}: masking failed: {e}")
            break

    n_real = sum(1 for c in candidates if c["is_real"])
    n_fp   = len(candidates) - n_real
    logger.info(f"  Done: {len(candidates)} candidates ({n_real} real, {n_fp} FP)")
    return candidates


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_values(val) -> list[float]:
    if isinstance(val, (int, float)):
        return [float(val)]
    return [float(x.strip()) for x in str(val).split(",")]


def main():
    conf = MEDIUM
    summary_path = PROJECT_ROOT / "tests" / "summary.csv"

    logger.info("=" * 72)
    logger.info("PERMISSIVE BLS TEST — threshold optimization data collection")
    logger.info(f"  Config:     MEDIUM  ({conf.max_quarters} quarters, "
                f"{conf.bls.max_period_grid_points:,} BLS points)")
    logger.info(f"  Truth data: {summary_path}")
    logger.info(f"  Output CSV: {CSV_FILE}")
    logger.info(f"  Log file:   {LOG_FILE}")
    logger.info("=" * 72)

    df = pd.read_csv(summary_path)
    df["periods_list"] = df["periods_d"].apply(parse_values)
    df["radii_list"]   = df["radii_Rearth"].apply(parse_values)
    df["min_period"]   = df["periods_list"].apply(min)
    df["max_period"]   = df["periods_list"].apply(max)

    df_run = df[(df["min_period"] > 2) & (df["max_period"] < 100)].copy().reset_index(drop=True)
    df_run = df_run.iloc[:250]

    logger.info(f"Systems in truth data:                 {len(df)}")
    logger.info(f"Systems with all periods in (2, 100)d: {len(df_run)}")
    logger.info("")

    all_candidates = []
    n_targets      = len(df_run)
    t_start        = time.monotonic()
    target_times   = []  # rolling window of per-target durations for ETA

    for idx, (_, row) in enumerate(df_run.iterrows()):
        target   = f"Kepler-{int(row['Kepler_#'])}"
        t_target = time.monotonic()

        candidates = run_target_permissive(
            target_name=target,
            truth_periods=row["periods_list"],
            truth_n=int(row["n_planets"]),
            conf=conf,
        )
        all_candidates.extend(candidates)

        elapsed_target = time.monotonic() - t_target
        target_times.append(elapsed_target)
        if len(target_times) > 20:          # rolling window — adapts to recent pace
            target_times.pop(0)

        done     = idx + 1
        remaining = n_targets - done
        elapsed_total = time.monotonic() - t_start
        avg_per_target = np.mean(target_times)
        eta_s    = avg_per_target * remaining
        eta_str  = str(timedelta(seconds=int(eta_s)))
        elapsed_str = str(timedelta(seconds=int(elapsed_total)))

        # ASCII progress bar (40 chars wide)
        pct  = done / n_targets
        bar  = ("█" * int(pct * 40)).ljust(40)
        logger.info(
            f"  [{bar}] {done}/{n_targets} ({pct*100:.0f}%)  "
            f"elapsed {elapsed_str}  ETA {eta_str}  "
            f"({avg_per_target:.0f}s/target)"
        )

        # Write incrementally so a crash doesn't lose everything
        if all_candidates:
            pd.DataFrame(all_candidates).to_csv(CSV_FILE, index=False)

    if not all_candidates:
        logger.error("No candidates collected — check errors above.")
        return

    out_df = pd.DataFrame(all_candidates)
    out_df.to_csv(CSV_FILE, index=False)

    n_real = out_df["is_real"].sum()
    n_fp   = len(out_df) - n_real

    logger.info("")
    logger.info("=" * 72)
    logger.info("DONE")
    logger.info(f"  Total candidates logged: {len(out_df)}")
    logger.info(f"  Real planets:            {n_real}")
    logger.info(f"  False positives:         {n_fp}")
    logger.info(f"  CSV written to:          {CSV_FILE}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
