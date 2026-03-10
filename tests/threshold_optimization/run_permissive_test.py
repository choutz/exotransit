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

Parallelism:
    Targets are processed in parallel using ProcessPoolExecutor.
    Each worker runs the full LC fetch + BLS loop for one target and
    returns its candidate list. Logging is routed through a shared
    multiprocessing queue so all output goes to the same log file.

    Default: 4 workers. Tune with --workers N.
    Note: more than ~6 workers may hit MAST download rate limits.

Output:
    tests/threshold_optimization/candidates_YYYYMMDD_HHMMSS.csv
    tests/threshold_optimization/permissive_test_YYYYMMDD_HHMMSS.log

Run from project root:
    caffeinate -i python tests/threshold_optimization/run_permissive_test.py
    caffeinate -i python tests/threshold_optimization/run_permissive_test.py --workers 6
"""

import sys
import os
import time
import logging
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import lightkurve as lk
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import QueueHandler, QueueListener

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MEDIUM
from tests.helpers import get_light_curve, CACHE_DIR
from exotransit.detection.bls import run_bls
from exotransit.pipeline.light_curves import LightCurveData

# ── Logging ────────────────────────────────────────────────────────────────────
OUT_DIR   = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = OUT_DIR / f"permissive_test_{timestamp}.log"
CSV_FILE  = OUT_DIR / f"candidates_{timestamp}.csv"

_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"

_NOISY_LOGGERS = [
    "lightkurve", "lightkurve.utils",
    "exotransit.detection.bls", "exotransit.detection.multi_planet",
    "exotransit.pipeline.light_curves",
    "exotransit.physics.stars", "exotransit.physics.limb_darkening",
]


def _setup_main_logging(log_queue: multiprocessing.Queue):
    """Configure root logger in the main process, backed by a QueueListener."""
    formatter = logging.Formatter(_LOG_FMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)

    # QueueListener drains the queue from worker processes and forwards
    # records to the main-process handlers above.
    listener = QueueListener(log_queue, sh, fh, respect_handler_level=True)
    listener.start()

    for mod in _NOISY_LOGGERS:
        logging.getLogger(mod).setLevel(logging.ERROR)

    return listener


def _worker_logging_init(log_queue: multiprocessing.Queue):
    """Called once in each worker process — routes all logging through the queue."""
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))
    root.setLevel(logging.INFO)
    for mod in _NOISY_LOGGERS:
        logging.getLogger(mod).setLevel(logging.ERROR)


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(result, lc: LightCurveData) -> dict:
    baseline      = lc.time.max() - lc.time.min()
    cadence_days  = float(np.median(np.diff(lc.time)))

    n_transits_expected = baseline / result.best_period
    per_transit_snr     = result.snr / np.sqrt(max(n_transits_expected, 1))
    duty_cycle          = result.best_duration / result.best_period
    duration_h          = result.best_duration * 24.0

    try:
        lc_lk  = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
        folded = lc_lk.fold(period=result.best_period, epoch_time=result.best_t0)
        half_dur = max(result.best_duration / 2, 1.5 * cadence_days)
        n_transit_pts = int(np.sum(np.abs(folded.time.value) < half_dur))
    except Exception:
        n_transit_pts = 0

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


# ── Per-target permissive search (runs in worker process) ──────────────────────

def run_target_permissive(
    target_name: str,
    truth_periods: list,
    truth_n: int,
    conf,
) -> list:
    """
    Run BLS iteratively with no reliability gate. Returns a list of
    candidate feature dicts, each with is_real labeled.
    Safe to call in a subprocess — all logging goes through QueueHandler.
    """
    logger = logging.getLogger("permissive_test.worker")
    logger.info(f"TARGET: {target_name}  "
                f"(truth: {truth_n} planet(s), periods: {[f'{p:.3f}' for p in truth_periods]})")

    try:
        lc = get_light_curve(target_name, mission="Kepler", max_quarters=conf.max_quarters)
        logger.info(f"  {target_name}: LC {len(lc.time):,} pts  "
                    f"baseline {lc.time[-1]-lc.time[0]:.1f} d")
    except Exception as e:
        logger.error(f"  {target_name}: LC FETCH FAILED: {e}")
        return []

    lc_lk     = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
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
            logger.warning(f"  {target_name} iter {i}: BLS failed: {e}")
            break

        if result.sde < 3.0:
            logger.info(f"  {target_name} iter {i}: SDE={result.sde:.2f} < 3 — noise floor, stopping")
            break

        is_real = int(any(
            abs(result.best_period - tp) / tp < 0.05
            for tp in truth_periods
        ))

        features = extract_features(result, current_lc)
        candidates.append({
            "target":    target_name,
            "iteration": i,
            "is_real":   is_real,
            **features,
        })

        label = "REAL" if is_real else "FP"
        logger.info(
            f"  {target_name} iter {i}: P={result.best_period:.4f}d  "
            f"SDE={result.sde:.2f}  SNR={result.snr:.2f}  "
            f"depth={result.transit_depth*1e6:.1f}ppm  [{label}]"
        )

        try:
            mask = lc_lk.create_transit_mask(
                period=result.best_period,
                transit_time=result.best_t0,
                duration=result.best_duration * 3.0,
            )
            lc_lk.flux.value[mask] = np.nanmedian(lc_lk.flux.value)
        except Exception as e:
            logger.warning(f"  {target_name} iter {i}: masking failed: {e}")
            break

    n_real = sum(1 for c in candidates if c["is_real"])
    logger.info(f"  {target_name}: done — {len(candidates)} candidates "
                f"({n_real} real, {len(candidates)-n_real} FP)")
    return candidates


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_values(val) -> list:
    if isinstance(val, (int, float)):
        return [float(val)]
    return [float(x.strip()) for x in str(val).split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int,
        default=min(4, os.cpu_count() or 4),
        help="Parallel worker processes (default: min(4, cpu_count)). "
             "More than ~6 may hit MAST rate limits.",
    )
    args = parser.parse_args()
    n_workers = max(1, args.workers)

    conf = MEDIUM
    summary_path = PROJECT_ROOT / "tests" / "summary.csv"

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_queue = multiprocessing.Queue()
    listener  = _setup_main_logging(log_queue)
    logger    = logging.getLogger("permissive_test")

    logger.info("=" * 72)
    logger.info("PERMISSIVE BLS TEST — threshold optimization data collection")
    logger.info(f"  Config:     MEDIUM  ({conf.max_quarters} quarters, "
                f"{conf.bls.max_period_grid_points:,} BLS points)")
    logger.info(f"  Workers:    {n_workers} (BLS phase only — downloads are sequential)")
    logger.info(f"  LC cache:   {CACHE_DIR}")
    logger.info(f"  Truth data: {summary_path}")
    logger.info(f"  Output CSV: {CSV_FILE}")
    logger.info(f"  Log file:   {LOG_FILE}")
    logger.info("=" * 72)

    # ── Load targets ──────────────────────────────────────────────────────────
    df = pd.read_csv(summary_path)
    df["periods_list"] = df["periods_d"].apply(parse_values)
    df["min_period"]   = df["periods_list"].apply(min)
    df["max_period"]   = df["periods_list"].apply(max)

    df_run = df[(df["min_period"] > 2) & (df["max_period"] < 100)].copy().reset_index(drop=True)
    df_run = df_run.iloc[:250]

    logger.info(f"Systems in truth data:                 {len(df)}")
    logger.info(f"Systems with all periods in (2, 100)d: {len(df_run)}")
    logger.info("")

    targets = [
        (f"Kepler-{int(row['Kepler_#'])}", row["periods_list"], int(row["n_planets"]))
        for _, row in df_run.iterrows()
    ]
    n_targets = len(targets)

    # ── Phase 1: sequential prefetch ───────────────────────────────────────────
    # Download all light curves one at a time before launching workers.
    # This prevents parallel workers from hammering MAST simultaneously.
    # get_light_curve already caches to disk, so workers will read from
    # cache and make zero network calls.
    logger.info("Phase 1: prefetching light curves sequentially…")
    failed_targets = set()
    for i, (target_name, _, _) in enumerate(targets):
        for attempt in range(3):
            try:
                get_light_curve(target_name, mission="Kepler", max_quarters=conf.max_quarters)
                logger.info(f"  [{i+1}/{n_targets}] {target_name} — cached")
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                if attempt < 2:
                    logger.warning(f"  [{i+1}/{n_targets}] {target_name} fetch failed "
                                   f"(attempt {attempt+1}/3), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"  [{i+1}/{n_targets}] {target_name} — "
                                 f"all 3 attempts failed, skipping: {e}")
                    failed_targets.add(target_name)
        else:
            continue
        time.sleep(0.5)   # polite pause between downloads

    targets = [(n, p, k) for n, p, k in targets if n not in failed_targets]
    n_targets = len(targets)
    logger.info(f"Phase 1 done — {n_targets} targets cached, "
                f"{len(failed_targets)} skipped")
    logger.info("")

    # ── Phase 2: parallel BLS ──────────────────────────────────────────────────
    all_candidates  = []
    completed       = 0
    completion_times = []          # wall-clock seconds per completed target
    t_start         = datetime.now()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_logging_init,
        initargs=(log_queue,),
    ) as executor:
        future_to_target = {
            executor.submit(run_target_permissive, name, periods, n, conf): name
            for name, periods, n in targets
        }

        for future in as_completed(future_to_target):
            target_name = future_to_target[future]
            t_done = datetime.now()

            try:
                candidates = future.result()
            except Exception as e:
                logger.error(f"  {target_name}: unhandled exception: {e}")
                candidates = []

            all_candidates.extend(candidates)
            completed += 1

            # Track per-target wall time (approximate — parallelism means
            # we measure wall time / n_workers on average)
            elapsed_total = (t_done - t_start).total_seconds()
            completion_times.append(elapsed_total / completed)
            window = completion_times[-20:]   # rolling 20-target window
            avg_s  = np.mean(window)
            eta_s  = avg_s * (n_targets - completed)

            pct = completed / n_targets
            bar = ("█" * int(pct * 40)).ljust(40)
            logger.info(
                f"  [{bar}] {completed}/{n_targets} ({pct*100:.0f}%)  "
                f"elapsed {str(timedelta(seconds=int(elapsed_total)))}  "
                f"ETA {str(timedelta(seconds=int(eta_s)))}  "
                f"(~{avg_s:.0f}s/target wall)"
            )

            # Incremental CSV write — survive a crash
            if all_candidates:
                pd.DataFrame(all_candidates).to_csv(CSV_FILE, index=False)

    listener.stop()

    if not all_candidates:
        logging.error("No candidates collected — check errors above.")
        return

    out_df = pd.DataFrame(all_candidates)
    out_df.to_csv(CSV_FILE, index=False)

    n_real = int(out_df["is_real"].sum())
    n_fp   = len(out_df) - n_real
    total_s = (datetime.now() - t_start).total_seconds()

    logger.info("")
    logger.info("=" * 72)
    logger.info("DONE")
    logger.info(f"  Total wall time:         {str(timedelta(seconds=int(total_s)))}")
    logger.info(f"  Total candidates logged: {len(out_df)}")
    logger.info(f"  Real planets:            {n_real}")
    logger.info(f"  False positives:         {n_fp}")
    logger.info(f"  CSV written to:          {CSV_FILE}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
