"""
tests/big_test/optimize_thresholds.py

Reads a big_test log and finds the optimal SNR + SDE threshold pair to
minimize false positives while maximizing real planet detections.

For every (SDE, SNR) combination on a grid, computes:
  - How many real planets are retained
  - How many false positives are retained
  - Precision, recall, F1

Prints:
  1. The Pareto frontier — combinations where raising either threshold
     further would cost a real planet without gaining a FP reduction
  2. Best by F1
  3. Best "FPs eliminated with zero real planets lost"
  4. ASCII grid heatmap of F1 over the threshold space

Usage:
    python tests/big_test/optimize_thresholds.py                  # latest log
    python tests/big_test/optimize_thresholds.py big_test_XYZ.log
"""

import sys
import argparse
from pathlib import Path

LOG_DIR = Path(__file__).parent
sys.path.insert(0, str(LOG_DIR))

from parse_big_test_results import parse


# ── Extract (sde, snr, is_real) for every detected planet ─────────────────────

def extract_detections(targets: list[dict]) -> list[dict]:
    """
    Returns a flat list of every detected planet with:
      {sde, snr, is_real, target, period}
    """
    rows = []
    for t in targets:
        detected   = t["detected_planets"]   # [{period, sde, snr, ...}]
        fp_periods = set(t["false_positives"])
        matched_det_periods = {m["det_p"] for m in t["period_matches"]}

        for dp in detected:
            p   = dp["period"]
            is_real = any(abs(p - mp) / mp < 0.001 for mp in matched_det_periods)
            rows.append({
                "target":  t["target"],
                "period":  p,
                "sde":     dp["sde"],
                "snr":     dp["snr"],
                "is_real": is_real,
            })
    return rows


# ── Grid scan ──────────────────────────────────────────────────────────────────

def scan_grid(
    detections: list[dict],
    sde_values: list[float],
    snr_values: list[float],
) -> list[dict]:
    """
    For each (sde_floor, snr_floor) pair, compute TP/FP/FN and derived metrics.
    A detection passes if BOTH sde >= sde_floor AND snr >= snr_floor.
    """
    total_real = sum(1 for d in detections if d["is_real"])
    total_fp   = sum(1 for d in detections if not d["is_real"])

    results = []
    for sde_thr in sde_values:
        for snr_thr in snr_values:
            tp = sum(1 for d in detections if d["is_real"]     and d["sde"] >= sde_thr and d["snr"] >= snr_thr)
            fp = sum(1 for d in detections if not d["is_real"] and d["sde"] >= sde_thr and d["snr"] >= snr_thr)
            fn = total_real - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            fp_eliminated   = total_fp   - fp
            real_eliminated = total_real - tp

            results.append({
                "sde_thr":         sde_thr,
                "snr_thr":         snr_thr,
                "tp":              tp,
                "fp":              fp,
                "fn":              fn,
                "precision":       precision,
                "recall":          recall,
                "f1":              f1,
                "fp_eliminated":   fp_eliminated,
                "real_eliminated": real_eliminated,
                "total_real":      total_real,
                "total_fp":        total_fp,
            })
    return results


# ── Pareto frontier ────────────────────────────────────────────────────────────

def pareto_frontier(results: list[dict]) -> list[dict]:
    """
    A point is Pareto-optimal if no other point has both:
      - higher recall (more real planets retained)
      - higher precision (fewer FPs retained)
    Returns only non-dominated points, sorted by recall descending.
    """
    frontier = []
    for r in results:
        dominated = any(
            other["recall"] >= r["recall"] and other["precision"] >= r["precision"]
            and (other["recall"] > r["recall"] or other["precision"] > r["precision"])
            for other in results
        )
        if not dominated:
            frontier.append(r)
    return sorted(frontier, key=lambda x: (-x["recall"], x["sde_thr"], x["snr_thr"]))


# ── ASCII heatmap ──────────────────────────────────────────────────────────────

def ascii_heatmap(results: list[dict], sde_values: list[float], snr_values: list[float]):
    """Print an ASCII grid of F1 scores over (SDE, SNR) space."""
    from math import isnan

    # Build lookup
    grid = {(r["sde_thr"], r["snr_thr"]): r["f1"] for r in results}

    SHADES = " ░▒▓█"

    def shade(f1):
        if f1 <= 0:     return SHADES[0]
        elif f1 < 0.4:  return SHADES[1]
        elif f1 < 0.6:  return SHADES[2]
        elif f1 < 0.8:  return SHADES[3]
        else:           return SHADES[4]

    # Header
    snr_labels = [f"{s:>4.0f}" for s in snr_values[::2]]
    print(f"\n  F1 score heatmap  (SDE→ rows, SNR→ columns)   {SHADES[1]}<0.4  {SHADES[2]}0.4–0.6  {SHADES[3]}0.6–0.8  {SHADES[4]}≥0.8")
    print(f"  {'SDE\\SNR':>8}  " + "".join(f"{s:>5.0f}" for s in snr_values))
    print("  " + "-" * (10 + 5 * len(snr_values)))

    for sde in sde_values:
        row = "".join(f"  {shade(grid.get((sde, snr), 0)):>4}" for snr in snr_values)
        print(f"  {sde:>8.1f} {row}")
    print()


# ── Printing ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def fmt_row(r, highlight=False):
    c = GREEN if highlight else ""
    return (
        f"{c}  SDE≥{r['sde_thr']:>5.1f}  SNR≥{r['snr_thr']:>5.1f}  "
        f"recall={r['recall']*100:>5.1f}%  precision={r['precision']*100:>5.1f}%  "
        f"F1={r['f1']:.3f}  "
        f"real kept={r['tp']:>3}/{r['total_real']:<3}  "
        f"FP kept={r['fp']:>3}/{r['total_fp']:<3}  "
        f"FP cut={r['fp_eliminated']:>3}{RESET}"
    )


def print_results(results: list[dict], detections: list[dict], sde_values, snr_values):
    total_real = results[0]["total_real"]
    total_fp   = results[0]["total_fp"]

    print(f"\n  Detected planets: {len(detections)}  |  Real: {total_real}  |  False positives: {total_fp}")

    # ── Best F1 ───────────────────────────────────────────────────────────────
    best_f1 = max(results, key=lambda r: r["f1"])
    print(f"\n  {CYAN}── Best F1 ──────────────────────────────────────────────────────{RESET}")
    print(fmt_row(best_f1, highlight=True))

    # ── Most FPs eliminated with zero real planets lost ────────────────────────
    lossless = [r for r in results if r["real_eliminated"] == 0]
    if lossless:
        best_lossless = max(lossless, key=lambda r: r["fp_eliminated"])
        print(f"\n  {CYAN}── Most FPs cut with no real planets lost ───────────────────────{RESET}")
        print(fmt_row(best_lossless, highlight=True))
    else:
        print(f"\n  {YELLOW}  No threshold combination cuts FPs without also losing real planets.{RESET}")

    # ── Pareto frontier ────────────────────────────────────────────────────────
    frontier = pareto_frontier(results)
    print(f"\n  {CYAN}── Pareto frontier ({len(frontier)} points) ───────────────────────────────{RESET}")
    print(f"  {'SDE≥':>7}  {'SNR≥':>7}  {'Recall':>8}  {'Precision':>10}  {'F1':>6}  {'Real kept':>10}  {'FP kept':>8}  {'FP cut':>7}")
    print("  " + "-" * 78)
    for r in frontier:
        is_best = (r["sde_thr"] == best_f1["sde_thr"] and r["snr_thr"] == best_f1["snr_thr"])
        color = GREEN if is_best else ""
        print(
            f"{color}  {r['sde_thr']:>7.1f}  {r['snr_thr']:>7.1f}  "
            f"{r['recall']*100:>7.1f}%  {r['precision']*100:>9.1f}%  "
            f"{r['f1']:>6.3f}  "
            f"{r['tp']:>4}/{r['total_real']:<4}  "
            f"{r['fp']:>4}/{r['total_fp']:<4}  "
            f"{r['fp_eliminated']:>6}{RESET}"
        )

    # ── Heatmap ────────────────────────────────────────────────────────────────
    ascii_heatmap(results, sde_values, snr_values)

    # ── Per-detection listing for detections near the boundary ────────────────
    # Show real planets with low SDE or SNR (most at risk from raising thresholds)
    at_risk = sorted(
        [d for d in detections if d["is_real"]],
        key=lambda d: min(d["sde"], d["snr"])
    )[:10]
    print(f"  {CYAN}── Real planets most at risk (lowest min(SDE, SNR)) ─────────────{RESET}")
    print(f"  {'Target':<22}  {'Period':>8}  {'SDE':>7}  {'SNR':>7}")
    print("  " + "-" * 52)
    for d in at_risk:
        print(f"  {d['target']:<22}  {d['period']:>8.4f}  {d['sde']:>7.2f}  {d['snr']:>7.2f}")

    at_risk_fp = sorted(
        [d for d in detections if not d["is_real"]],
        key=lambda d: -min(d["sde"], d["snr"])
    )[:10]
    print(f"\n  {CYAN}── Hardest FPs to cut (highest min(SDE, SNR)) ───────────────────{RESET}")
    print(f"  {'Target':<22}  {'Period':>8}  {'SDE':>7}  {'SNR':>7}")
    print("  " + "-" * 52)
    for d in at_risk_fp:
        print(f"  {d['target']:<22}  {d['period']:>8.4f}  {d['sde']:>7.2f}  {d['snr']:>7.2f}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", help="Log file (default: latest)")
    parser.add_argument("--sde-max", type=float, default=25.0)
    parser.add_argument("--snr-max", type=float, default=40.0)
    parser.add_argument("--step",    type=float, default=1.0,
                        help="Grid step size (default 1.0; use 0.5 for finer resolution)")
    args = parser.parse_args()

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

    detections = extract_detections(targets)
    if not detections:
        print("No detected planets found in log.")
        sys.exit(1)

    step = args.step
    sde_values = [round(7.0 + i * step, 2) for i in range(int((args.sde_max - 7.0) / step) + 1)]
    snr_values = [round(7.0 + i * step, 2) for i in range(int((args.snr_max - 7.0) / step) + 1)]

    results = scan_grid(detections, sde_values, snr_values)
    print_results(results, detections, sde_values, snr_values)


if __name__ == "__main__":
    main()
