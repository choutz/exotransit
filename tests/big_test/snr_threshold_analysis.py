"""
tests/big_test/snr_threshold_analysis.py

Reads a big_test log and shows the trade-off of raising the SNR threshold:
how many false positives go away vs how many real planet detections get cut.

Usage:
    python tests/big_test/snr_threshold_analysis.py                  # latest log
    python tests/big_test/snr_threshold_analysis.py big_test_XYZ.log  # specific log
    python tests/big_test/snr_threshold_analysis.py big_test_XYZ.log --threshold 15  # single value
"""

import sys
import argparse
from pathlib import Path

LOG_DIR = Path(__file__).parent
sys.path.insert(0, str(LOG_DIR))

from parse_big_test_results import parse, LOG_DIR


def classify_detections(targets: list[dict]) -> tuple[list[float], list[float]]:
    """
    Returns (real_snrs, fp_snrs):
      real_snrs — SNR of every detected planet that matched a truth period
      fp_snrs   — SNR of every detected planet that did NOT match any truth period
    """
    real_snrs = []
    fp_snrs   = []

    for t in targets:
        detected = t["detected_planets"]   # [{period, snr, sde, ...}, ...]
        matched  = t["period_matches"]     # [{truth_p, det_p, pct_err}, ...]
        fp_periods = t["false_positives"]  # [period, ...]

        # Build set of detected periods that were matched to truth
        matched_det_periods = {m["det_p"] for m in matched}

        for dp in detected:
            p   = dp["period"]
            snr = dp["snr"]

            # Is this period in the matched set? (0.1% tolerance)
            is_real = any(abs(p - mp) / mp < 0.001 for mp in matched_det_periods)

            # Alternatively confirm via fp_periods list
            is_fp = any(abs(p - fp) / fp < 0.001 for fp in fp_periods) if fp_periods else not is_real

            if is_real:
                real_snrs.append(snr)
            else:
                fp_snrs.append(snr)

    return real_snrs, fp_snrs


def analyse(real_snrs: list[float], fp_snrs: list[float], thresholds: list[float]):
    n_real = len(real_snrs)
    n_fp   = len(fp_snrs)
    n_total = n_real + n_fp

    print()
    print(f"  Detected planets total : {n_total}")
    print(f"  Real (matched to truth): {n_real}")
    print(f"  False positives        : {n_fp}")
    print()
    print(f"  {'SNR ≥':>8}  {'FP cut':>8}  {'FP cut %':>9}  {'Real cut':>9}  {'Real cut %':>11}  {'Remaining FP':>13}  {'Remaining real':>15}")
    print("  " + "-" * 90)

    for thr in thresholds:
        fp_cut   = sum(1 for s in fp_snrs   if s < thr)
        real_cut = sum(1 for s in real_snrs if s < thr)
        fp_left  = n_fp   - fp_cut
        real_left = n_real - real_cut

        fp_pct   = fp_cut   / n_fp   * 100 if n_fp   else 0.0
        real_pct = real_cut / n_real * 100 if n_real else 0.0

        # Color coding: green if fp% >> real%, red if real% is high
        if fp_pct > 0 and real_pct == 0:
            color = "\033[92m"   # green — cuts FPs only
        elif real_pct > fp_pct:
            color = "\033[91m"   # red — hurts real more than it helps
        elif fp_pct >= 2 * real_pct:
            color = "\033[92m"   # green — FP reduction >> real cost
        else:
            color = "\033[93m"   # yellow — marginal trade-off
        reset = "\033[0m"

        row = (
            f"  {thr:>8.1f}  {fp_cut:>8d}  {fp_pct:>8.1f}%  "
            f"{real_cut:>9d}  {real_pct:>10.1f}%  "
            f"{fp_left:>13d}  {real_left:>15d}"
        )
        print(f"{color}{row}{reset}")

    print()
    print("  Green = FP reduction dominates. Yellow = marginal. Red = cuts more real planets than FPs.")
    print()


def snr_distribution(real_snrs: list[float], fp_snrs: list[float]):
    """Print a simple ASCII histogram of SNR distributions side by side."""
    import math

    all_snrs = real_snrs + fp_snrs
    if not all_snrs:
        return

    lo  = 0.0
    hi  = min(max(all_snrs), 100.0)   # cap display at 100 for readability
    bins = 20
    width = (hi - lo) / bins

    print("  SNR distribution (capped at 100):")
    print(f"  {'Bin':>12}  {'Real':>6}  {'FP':>6}")
    print("  " + "-" * 40)

    for i in range(bins):
        bin_lo = lo + i * width
        bin_hi = bin_lo + width
        r_count = sum(1 for s in real_snrs if bin_lo <= s < bin_hi)
        f_count = sum(1 for s in fp_snrs   if bin_lo <= s < bin_hi)
        bar_r = "█" * r_count
        bar_f = "░" * f_count
        print(f"  {bin_lo:>5.1f}–{bin_hi:<5.1f}  {r_count:>6}  {f_count:>6}  \033[92m{bar_r}\033[0m\033[93m{bar_f}\033[0m")

    # Overflow bin for SNR > 100
    r_over = sum(1 for s in real_snrs if s >= 100)
    f_over = sum(1 for s in fp_snrs   if s >= 100)
    if r_over or f_over:
        print(f"  {'> 100':>12}  {r_over:>6}  {f_over:>6}")

    print()
    print("  \033[92m█\033[0m = real planet  \033[93m░\033[0m = false positive")
    print()


def main():
    parser = argparse.ArgumentParser(description="SNR threshold trade-off analysis for big_test logs")
    parser.add_argument("log", nargs="?", help="Log file name or path (default: latest)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Single SNR threshold to evaluate (default: scan 7–50)")
    parser.add_argument("--no-histogram", action="store_true",
                        help="Skip the SNR distribution histogram")
    args = parser.parse_args()

    # Resolve log path
    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = LOG_DIR / log_path
    else:
        logs = sorted(LOG_DIR.glob("big_test_*.log"))
        if not logs:
            print("No big_test_*.log files found in", LOG_DIR)
            sys.exit(1)
        log_path = logs[-1]

    print(f"\nParsing: {log_path}")

    targets = parse(log_path)
    if not targets:
        print("No target results found in log.")
        sys.exit(1)

    real_snrs, fp_snrs = classify_detections(targets)

    if not args.no_histogram:
        snr_distribution(real_snrs, fp_snrs)

    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        # Default: scan from current floor (7.1) up to 50 in steps
        thresholds = [7.1, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    analyse(real_snrs, fp_snrs, thresholds)


if __name__ == "__main__":
    main()
