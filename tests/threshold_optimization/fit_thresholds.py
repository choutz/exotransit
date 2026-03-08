"""
tests/threshold_optimization/fit_thresholds.py

Reads the labeled candidate CSV from run_permissive_test.py and finds
empirically optimal vetting thresholds.

Analysis steps:
  1. Class balance and feature distributions
  2. Per-feature precision-recall sweep — best single threshold for each feature
  3. Feature importance ranking
  4. Pairwise grid scan over the top feature pairs (Pareto frontier)
  5. Decision tree fit (sklearn) — produces interpretable IF/AND threshold rules
  6. Final recommendation

Usage:
    python tests/threshold_optimization/fit_thresholds.py                        # latest CSV
    python tests/threshold_optimization/fit_thresholds.py candidates_XYZ.csv
    python tests/threshold_optimization/fit_thresholds.py candidates_XYZ.csv --recall-floor 0.98
    python tests/threshold_optimization/fit_thresholds.py candidates_XYZ.csv --no-tree
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

OUT_DIR = Path(__file__).parent

# ── Feature definitions ────────────────────────────────────────────────────────
# (column_name, threshold_direction, human_label)
# direction ">=" means "pass if value >= threshold" (higher = more real)
# direction "<=" means "pass if value <= threshold" (lower = more real)
FEATURES = [
    ("sde",                 ">=", "SDE (Signal Detection Efficiency)"),
    ("snr",                 ">=", "SNR (Signal-to-Noise Ratio)"),
    ("depth_ppm",           ">=", "Transit depth (ppm)"),
    ("depth_unc_ppm",       "<=", "Depth uncertainty (ppm)"),
    ("n_transit_pts",       ">=", "In-transit data points"),
    ("n_transits_expected", ">=", "Expected transit windows"),
    ("per_transit_snr",     ">=", "Per-transit SNR"),
    ("duty_cycle",          "<=", "Duration/period ratio (duty cycle)"),
    ("duration_h",          ">=", "Transit duration (hours)"),
    ("coverage_ratio",      ">=", "Transit point coverage ratio"),
    ("has_aliases",         "<=", "Has alias periods (0=no, 1=yes)"),
]

FEATURE_COLS = [f[0] for f in FEATURES]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, int(tp), int(fp), int(fn)


def apply_threshold(col_vals, direction, thr):
    if direction == ">=":
        return (col_vals >= thr).astype(int)
    else:
        return (col_vals <= thr).astype(int)


# ── 1. Per-feature sweep ───────────────────────────────────────────────────────

def per_feature_sweep(df: pd.DataFrame, y: np.ndarray, recall_floor: float) -> list[dict]:
    """
    For each feature, find the threshold that maximises F1 subject to
    recall >= recall_floor. Returns a list of best-threshold dicts,
    sorted by F1 descending.
    """
    print(f"\n{CYAN}{'='*72}")
    print(f"  PER-FEATURE THRESHOLD SWEEP  (recall floor: {recall_floor:.0%})")
    print(f"{'='*72}{RESET}")
    print(f"  {'Feature':<36} {'Dir':>4}  {'Threshold':>10}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'FP cut':>7}")
    print("  " + "-" * 72)

    results = []
    for col, direction, label in FEATURES:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        # Candidate thresholds: percentiles of observed distribution
        thresholds = np.unique(np.nanpercentile(vals[np.isfinite(vals)],
                                                np.linspace(0, 100, 200)))
        best = None
        for thr in thresholds:
            pred = apply_threshold(vals, direction, thr)
            p, r, f1, tp, fp, fn = metrics(y, pred)
            if r < recall_floor:
                continue
            total_fp = int(np.sum(y == 0))
            fp_cut   = total_fp - fp
            if best is None or f1 > best["f1"]:
                best = dict(col=col, label=label, direction=direction,
                            threshold=thr, precision=p, recall=r, f1=f1,
                            tp=tp, fp=fp, fn=fn, fp_cut=fp_cut)

        if best is None:
            print(f"  {label:<36} {direction:>4}  {'—':>10}  (no threshold achieves recall ≥ {recall_floor:.0%})")
        else:
            color = GREEN if best["f1"] > 0.85 else (YELLOW if best["f1"] > 0.70 else RED)
            print(
                f"  {color}{label:<36} {direction:>4}  {best['threshold']:>10.3f}"
                f"  {best['recall']*100:>6.1f}%  {best['precision']*100:>6.1f}%"
                f"  {best['f1']:>7.3f}  {best['fp_cut']:>7}{RESET}"
            )
            results.append(best)

    return sorted(results, key=lambda x: -x["f1"])


# ── 2. Pairwise grid scan ──────────────────────────────────────────────────────

def pairwise_scan(df: pd.DataFrame, y: np.ndarray,
                  top_features: list[dict], recall_floor: float, n_pairs: int = 3):
    """
    Grid scan over the top N feature pairs. Finds the Pareto frontier.
    """
    print(f"\n{CYAN}{'='*72}")
    print(f"  PAIRWISE GRID SCAN (top {n_pairs} features, recall floor: {recall_floor:.0%})")
    print(f"{'='*72}{RESET}")

    top_cols = [f["col"] for f in top_features[:min(n_pairs + 1, len(top_features))]]

    for col_a, col_b in combinations(top_cols, 2):
        spec_a = next(f for f in FEATURES if f[0] == col_a)
        spec_b = next(f for f in FEATURES if f[0] == col_b)

        vals_a = df[col_a].values.astype(float)
        vals_b = df[col_b].values.astype(float)
        thrs_a = np.unique(np.nanpercentile(vals_a[np.isfinite(vals_a)], np.linspace(0, 100, 60)))
        thrs_b = np.unique(np.nanpercentile(vals_b[np.isfinite(vals_b)], np.linspace(0, 100, 60)))

        best = None
        for ta in thrs_a:
            for tb in thrs_b:
                pred = (apply_threshold(vals_a, spec_a[1], ta) &
                        apply_threshold(vals_b, spec_b[1], tb))
                p, r, f1, tp, fp, fn = metrics(y, pred)
                if r < recall_floor:
                    continue
                total_fp = int(np.sum(y == 0))
                if best is None or f1 > best["f1"]:
                    best = dict(col_a=col_a, col_b=col_b,
                                ta=ta, tb=tb,
                                precision=p, recall=r, f1=f1,
                                tp=tp, fp=fp, fn=fn,
                                fp_cut=total_fp - fp)

        if best:
            print(f"\n  {spec_a[2]}  {spec_a[1]}  {best['ta']:.3f}")
            print(f"  {spec_b[2]}  {spec_b[1]}  {best['tb']:.3f}")
            print(f"  → recall={best['recall']*100:.1f}%  precision={best['precision']*100:.1f}%"
                  f"  F1={best['f1']:.3f}  FP cut={best['fp_cut']}/{int(np.sum(y==0))}")


# ── 3. Decision tree ───────────────────────────────────────────────────────────

def decision_tree(df: pd.DataFrame, y: np.ndarray,
                  recall_floor: float, max_depth: int = 4):
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
        print(f"\n{YELLOW}  sklearn not installed — skipping decision tree.{RESET}")
        print(f"  Install with: pip install scikit-learn")
        return

    print(f"\n{CYAN}{'='*72}")
    print(f"  DECISION TREE  (max_depth={max_depth}, recall floor: {recall_floor:.0%})")
    print(f"{'='*72}{RESET}")

    available = [col for col in FEATURE_COLS if col in df.columns]
    X = df[available].fillna(0).values

    # Weight classes so the tree is penalized more for missing real planets
    classes   = np.array([0, 1])
    weights   = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {0: weights[0], 1: weights[1]}

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42,
    )
    clf.fit(X, y)

    pred = clf.predict(X)
    p, r, f1, tp, fp, fn = metrics(y, pred)
    total_fp = int(np.sum(y == 0))
    print(f"  Training performance: recall={r*100:.1f}%  precision={p*100:.1f}%  F1={f1:.3f}  FP cut={total_fp-fp}/{total_fp}")

    if r < recall_floor:
        print(f"  {YELLOW}Warning: recall {r*100:.1f}% is below the {recall_floor:.0%} floor.{RESET}")
        print(f"  Try --depth 3 for a shallower tree, or --recall-floor for a looser constraint.")

    print(f"\n  Tree rules (feature >= threshold at each split):\n")
    tree_text = export_text(clf, feature_names=available)
    for line in tree_text.split("\n"):
        print(f"    {line}")

    # Feature importances
    importances = sorted(
        zip(available, clf.feature_importances_),
        key=lambda x: -x[1]
    )
    print(f"\n  Feature importances:")
    for col, imp in importances:
        if imp < 0.001:
            continue
        bar = "█" * int(imp * 40)
        print(f"    {col:<24}  {imp:.3f}  {bar}")


# ── 4. Distribution summary ────────────────────────────────────────────────────

def distribution_summary(df: pd.DataFrame, y: np.ndarray):
    print(f"\n{CYAN}{'='*72}")
    print(f"  FEATURE DISTRIBUTIONS  (real vs false positive)")
    print(f"{'='*72}{RESET}")
    print(f"  {'Feature':<28}  {'Real median':>12}  {'Real p25–p75':>16}  {'FP median':>12}  {'FP p25–p75':>16}")
    print("  " + "-" * 88)

    real = df[y == 1]
    fp   = df[y == 0]

    for col, _, label in FEATURES:
        if col not in df.columns:
            continue
        rv = real[col].dropna()
        fv = fp[col].dropna()
        if len(rv) == 0 or len(fv) == 0:
            continue
        r_med = np.median(rv)
        f_med = np.median(fv)
        r_q   = f"{np.percentile(rv, 25):.1f}–{np.percentile(rv, 75):.1f}"
        f_q   = f"{np.percentile(fv, 25):.1f}–{np.percentile(fv, 75):.1f}"
        # Color by separation: green = distributions well separated
        sep = abs(r_med - f_med) / (np.std(rv) + np.std(fv) + 1e-9)
        color = GREEN if sep > 1.0 else (YELLOW if sep > 0.3 else "")
        print(f"  {color}{label:<28}  {r_med:>12.2f}  {r_q:>16}  {f_med:>12.2f}  {f_q:>16}{RESET}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", help="CSV file (default: latest candidates_*.csv)")
    parser.add_argument("--recall-floor", type=float, default=0.99,
                        help="Minimum recall required (default: 0.99)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Max depth for decision tree (default: 4)")
    parser.add_argument("--no-tree", action="store_true",
                        help="Skip decision tree fitting")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = OUT_DIR / csv_path
    else:
        csvs = sorted(OUT_DIR.glob("candidates_*.csv"))
        if not csvs:
            print("No candidates_*.csv files found in", OUT_DIR)
            sys.exit(1)
        csv_path = csvs[-1]

    print(f"\n{BOLD}Parsing: {csv_path}{RESET}")
    df = pd.read_csv(csv_path)

    if "is_real" not in df.columns:
        print("CSV missing 'is_real' column — was it produced by run_permissive_test.py?")
        sys.exit(1)

    y = df["is_real"].values.astype(int)
    n_real = int(y.sum())
    n_fp   = int((y == 0).sum())
    n_total = len(df)

    print(f"\n  Total candidates: {n_total}")
    print(f"  Real planets:     {GREEN}{n_real}  ({n_real/n_total*100:.1f}%){RESET}")
    print(f"  False positives:  {YELLOW}{n_fp}  ({n_fp/n_total*100:.1f}%){RESET}")
    print(f"  Targets covered:  {df['target'].nunique()}")
    print(f"  Recall floor:     {args.recall_floor:.0%}")

    # ── Distributions ──────────────────────────────────────────────────────────
    distribution_summary(df, y)

    # ── Per-feature sweep ──────────────────────────────────────────────────────
    ranked = per_feature_sweep(df, y, args.recall_floor)

    # ── Pairwise scan ──────────────────────────────────────────────────────────
    if len(ranked) >= 2:
        pairwise_scan(df, y, ranked, args.recall_floor, n_pairs=4)

    # ── Decision tree ──────────────────────────────────────────────────────────
    if not args.no_tree:
        decision_tree(df, y, args.recall_floor, max_depth=args.depth)

    # ── Quick summary of top recommendations ───────────────────────────────────
    print(f"\n{CYAN}{'='*72}")
    print(f"  TOP SINGLE-FEATURE THRESHOLDS")
    print(f"{'='*72}{RESET}")
    for r in ranked[:5]:
        print(f"  {r['label']:<36} {r['direction']} {r['threshold']:.3f}"
              f"  →  recall={r['recall']*100:.1f}%  precision={r['precision']*100:.1f}%  F1={r['f1']:.3f}")
    print()


if __name__ == "__main__":
    main()
