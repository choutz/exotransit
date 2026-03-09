"""
tests/threshold_optimization/fit_thresholds.py

Reads the labeled candidate CSV from run_permissive_test.py and finds
empirically optimal vetting thresholds — including a two-stage pipeline
that uses TCE hard cuts as a loose pre-filter followed by a decision tree
as a finer second stage.

Analysis steps:
  1. Class balance and feature distributions
  2. Per-feature precision-recall sweep — best single threshold for each feature
  3. Pairwise grid scan over the top feature pairs (Pareto frontier)
  4. Decision tree fit (sklearn) — produces interpretable IF/AND threshold rules
  5. TWO-STAGE ANALYSIS — TCE pre-filter → decision tree on survivors
     5a. Grid search over TCE threshold leniency levels
     5b. Decision tree fit on TCE survivors only
     5c. Combined pipeline performance vs single-stage baseline

Usage:
    python tests/threshold_optimization/fit_thresholds.py                        # latest CSV
    python tests/threshold_optimization/fit_thresholds.py candidates_XYZ.csv
    python tests/threshold_optimization/fit_thresholds.py candidates_XYZ.csv --recall-floor 0.98
    python tests/threshold_optimization/fit_thresholds.py candidates_XYZ.csv --no-tree
    python tests/threshold_optimization/fit_thresholds.py candidates_XYZ.csv --two-stage-only
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

OUT_DIR = Path(__file__).parent

# ── Feature definitions ────────────────────────────────────────────────────────
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


# ── TCE filter configurations ─────────────────────────────────────────────────
# Each entry is a dict of {column: (direction, threshold)}.
# "LOOSE" = minimal pre-filter to remove only the most obvious junk.
# "MEDIUM" = the current assess_reliability defaults.
# "TIGHT" = aggressive, high-precision, lower recall.
#
# These are applied as hard AND gates — a candidate must pass ALL checks
# in the config to survive to the decision tree stage.
#
# Rationale for LOOSE defaults:
#   TCE-01/02: Keep SDE/SNR floors at the Jenkins+2010 minimum (7.1).
#              The decision tree will tighten this further.
#   TCE-03:    Loosened to 1.2 (was 2.0) — 2.0 cut 75 real planets on
#              248-target validation. Let the tree handle marginal signals.
#   TCE-04:    Duty cycle cap kept at 0.1 — this is a hard physical limit.
#   TCE-08:    Loosened to 3 windows (was 5) — real long-period planets
#              with few transits should survive to the tree.
#   TCE-10:    Loosened to 0.5h (was 1.0h) — 1.0h cut 22 real planets.
#   TCE-13:    Kept at 3% depth cap — eclipsing binary discriminator.
#   TCE-15:    Kept at 60 ppm long-cadence floor.
TCE_CONFIGS = {
    "LOOSE": {
        "sde":                 (">=", 7.1),
        "snr":                 (">=", 7.1),
        "per_transit_snr":     (">=", 1.2),
        "duty_cycle":          ("<=", 0.10),
        "n_transits_expected": (">=", 3.0),
        "duration_h":          (">=", 0.5),
        "depth_ppm":           ("<=", 30000),   # TCE-13: < 3%
        # TCE-15: depth floor — only meaningful for long-cadence; applied separately
    },
    "MEDIUM": {
        "sde":                 (">=", 7.1),
        "snr":                 (">=", 7.1),
        "per_transit_snr":     (">=", 1.5),
        "duty_cycle":          ("<=", 0.10),
        "n_transits_expected": (">=", 4.0),
        "duration_h":          (">=", 0.75),
        "depth_ppm":           ("<=", 30000),
    },
    "TIGHT": {
        "sde":                 (">=", 7.1),
        "snr":                 (">=", 7.1),
        "per_transit_snr":     (">=", 2.0),
        "duty_cycle":          ("<=", 0.10),
        "n_transits_expected": (">=", 5.0),
        "duration_h":          (">=", 1.0),
        "depth_ppm":           ("<=", 30000),
    },
}


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


def apply_tce_config(df, config):
    """Apply a TCE config dict to a DataFrame, return boolean mask of survivors."""
    mask = np.ones(len(df), dtype=bool)
    for col, (direction, thr) in config.items():
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if direction == ">=":
            mask &= vals >= thr
        else:
            mask &= vals <= thr
    # TCE-15: long-cadence depth floor (60 ppm) — applied if depth_ppm present
    if "depth_ppm" in df.columns:
        mask &= df["depth_ppm"].values >= 60
    return mask


# ── 1. Per-feature sweep ───────────────────────────────────────────────────────

def per_feature_sweep(df, y, recall_floor):
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

def pairwise_scan(df, y, top_features, recall_floor, n_pairs=3):
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

def fit_decision_tree(df, y, recall_floor, max_depth=4, label=""):
    """
    Fit and report a decision tree. Returns the fitted classifier and the
    column list used, so the caller can apply it to held-out data.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
        print(f"\n{YELLOW}  sklearn not installed — skipping decision tree.{RESET}")
        print(f"  Install with: pip install scikit-learn")
        return None, None

    available = [col for col in FEATURE_COLS if col in df.columns]
    X = df[available].fillna(0).values

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y)
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

    tag = f" [{label}]" if label else ""
    print(f"  Training performance{tag}: recall={r*100:.1f}%  precision={p*100:.1f}%  "
          f"F1={f1:.3f}  FP cut={total_fp-fp}/{total_fp}")

    if r < recall_floor:
        print(f"  {YELLOW}Warning: recall {r*100:.1f}% is below the {recall_floor:.0%} floor.{RESET}")
        print(f"  Try --depth 3 for a shallower tree, or --recall-floor for a looser constraint.")

    print(f"\n  Tree rules (feature >= threshold at each split):\n")
    tree_text = export_text(clf, feature_names=available)
    for line in tree_text.split("\n"):
        print(f"    {line}")

    importances = sorted(zip(available, clf.feature_importances_), key=lambda x: -x[1])
    print(f"\n  Feature importances:")
    for col, imp in importances:
        if imp < 0.001:
            continue
        bar = "█" * int(imp * 40)
        print(f"    {col:<24}  {imp:.3f}  {bar}")

    return clf, available


# ── 4. Two-stage analysis ──────────────────────────────────────────────────────

def two_stage_analysis(df, y, recall_floor, max_depth=4):
    """
    Two-stage pipeline: TCE hard cuts → decision tree on survivors.

    For each TCE config (LOOSE / MEDIUM / TIGHT):
      1. Apply TCE filter, report what survives and at what recall/precision.
      2. Fit a decision tree on TCE survivors only.
      3. Apply that tree back to the survivors and report combined performance
         relative to the full dataset (so recall is computed against all 1488
         candidates, not just survivors).
      4. Compare to the single-stage baseline (tree on raw candidates).
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
        print(f"\n{YELLOW}  sklearn not installed — skipping two-stage analysis.{RESET}")
        return

    print(f"\n{CYAN}{'='*72}")
    print(f"  TWO-STAGE ANALYSIS: TCE pre-filter → decision tree")
    print(f"  Recall floor: {recall_floor:.0%}  |  Tree max_depth: {max_depth}")
    print(f"{'='*72}{RESET}")

    n_total_real = int(y.sum())
    n_total_fp   = int((y == 0).sum())

    # ── Single-stage baseline: tree on all candidates ─────────────────────────
    print(f"\n  {BOLD}── Baseline: single-stage decision tree (no TCE pre-filter) ──{RESET}")
    available = [col for col in FEATURE_COLS if col in df.columns]
    X_all = df[available].fillna(0).values
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y)

    clf_base = DecisionTreeClassifier(max_depth=max_depth,
                                      class_weight={0: weights[0], 1: weights[1]},
                                      random_state=42)
    clf_base.fit(X_all, y)
    pred_base = clf_base.predict(X_all)
    p_base, r_base, f1_base, tp_base, fp_base, fn_base = metrics(y, pred_base)
    print(f"  recall={r_base*100:.1f}%  precision={p_base*100:.1f}%  F1={f1_base:.3f}"
          f"  FP cut={n_total_fp - fp_base}/{n_total_fp}  real kept={tp_base}/{n_total_real}")

    # ── Per-TCE-config analysis ────────────────────────────────────────────────
    best_two_stage = None

    for config_name, config in TCE_CONFIGS.items():
        print(f"\n  {BOLD}── TCE config: {config_name} ──{RESET}")

        # Step 1: Apply TCE filter
        tce_mask = apply_tce_config(df, config)
        df_survivors = df[tce_mask].copy()
        y_survivors  = y[tce_mask]

        n_surv       = int(tce_mask.sum())
        n_surv_real  = int(y_survivors.sum())
        n_surv_fp    = int((y_survivors == 0).sum())
        tce_recall   = n_surv_real / n_total_real if n_total_real > 0 else 0
        tce_precision = n_surv_real / n_surv if n_surv > 0 else 0
        fp_removed_by_tce = n_total_fp - n_surv_fp

        print(f"  TCE survivors: {n_surv}/{len(df)}  "
              f"({n_surv_real} real, {n_surv_fp} FP)")
        print(f"  TCE stage:     recall={tce_recall*100:.1f}%  "
              f"precision={tce_precision*100:.1f}%  "
              f"FP removed={fp_removed_by_tce}/{n_total_fp}")

        if n_surv_real == 0 or n_surv_fp == 0:
            print(f"  {YELLOW}  Skipping tree — no class balance in survivors.{RESET}")
            continue

        # Step 2: Fit decision tree on survivors only
        X_surv = df_survivors[available].fillna(0).values
        w_surv = compute_class_weight("balanced", classes=classes, y=y_survivors)

        clf_stage2 = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight={0: w_surv[0], 1: w_surv[1]},
            random_state=42,
        )
        clf_stage2.fit(X_surv, y_survivors)

        # Step 3: Apply tree to survivors, compute combined pipeline metrics
        # against the full dataset (candidates rejected by TCE count as FN if real)
        pred_survivors = clf_stage2.predict(X_surv)

        # Combined prediction over full dataset:
        #   TCE rejects → predict 0
        #   TCE survivors → use tree prediction
        pred_combined = np.zeros(len(df), dtype=int)
        surv_indices  = np.where(tce_mask)[0]
        pred_combined[surv_indices] = pred_survivors

        p_comb, r_comb, f1_comb, tp_comb, fp_comb, fn_comb = metrics(y, pred_combined)
        fp_cut_comb = n_total_fp - fp_comb

        print(f"  Combined pipeline: recall={r_comb*100:.1f}%  precision={p_comb*100:.1f}%  "
              f"F1={f1_comb:.3f}  FP cut={fp_cut_comb}/{n_total_fp}  "
              f"real kept={tp_comb}/{n_total_real}")

        # Flag improvement vs baseline
        if f1_comb > f1_base:
            delta_f1  = f1_comb - f1_base
            delta_prec = (p_comb - p_base) * 100
            delta_rec  = (r_comb - r_base) * 100
            print(f"  {GREEN}  ▲ vs baseline: F1 +{delta_f1:.3f}  "
                  f"precision {delta_prec:+.1f}pp  recall {delta_rec:+.1f}pp{RESET}")
        else:
            delta_f1 = f1_comb - f1_base
            print(f"  {YELLOW}  ▼ vs baseline: F1 {delta_f1:+.3f}{RESET}")

        # Track best two-stage config
        if best_two_stage is None or f1_comb > best_two_stage["f1"]:
            best_two_stage = dict(
                config_name=config_name, config=config,
                clf=clf_stage2, available=available,
                f1=f1_comb, recall=r_comb, precision=p_comb,
                fp_cut=fp_cut_comb, tp=tp_comb,
                tce_recall=tce_recall, tce_fp_removed=fp_removed_by_tce,
            )

    # ── Best two-stage config: print tree rules ────────────────────────────────
    if best_two_stage:
        print(f"\n  {BOLD}{'='*60}")
        print(f"  BEST TWO-STAGE CONFIG: TCE={best_two_stage['config_name']}")
        print(f"  recall={best_two_stage['recall']*100:.1f}%  "
              f"precision={best_two_stage['precision']*100:.1f}%  "
              f"F1={best_two_stage['f1']:.3f}  "
              f"FP cut={best_two_stage['fp_cut']}/{n_total_fp}  "
              f"real kept={best_two_stage['tp']}/{n_total_real}")
        print(f"  {'='*60}{RESET}")

        print(f"\n  TCE thresholds ({best_two_stage['config_name']}):")
        for col, (direction, thr) in best_two_stage["config"].items():
            label = next((f[2] for f in FEATURES if f[0] == col), col)
            print(f"    {label:<36} {direction} {thr}")
        print(f"    (+ TCE-15: depth_ppm >= 60)")

        print(f"\n  Stage-2 decision tree rules:\n")
        tree_text = export_text(best_two_stage["clf"],
                                feature_names=best_two_stage["available"])
        for line in tree_text.split("\n"):
            print(f"    {line}")

        importances = sorted(
            zip(best_two_stage["available"], best_two_stage["clf"].feature_importances_),
            key=lambda x: -x[1],
        )
        print(f"\n  Stage-2 feature importances:")
        for col, imp in importances:
            if imp < 0.001:
                continue
            bar = "█" * int(imp * 40)
            print(f"    {col:<24}  {imp:.3f}  {bar}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n  {CYAN}{'─'*72}")
    print(f"  SUMMARY: single-stage vs two-stage")
    print(f"  {'─'*72}{RESET}")
    print(f"  {'Config':<30}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'FP cut':>10}  {'Real kept':>10}")
    print(f"  {'─'*72}")
    print(f"  {'Baseline (tree only)':<30}  {r_base*100:>6.1f}%  {p_base*100:>6.1f}%  "
          f"{f1_base:>7.3f}  {n_total_fp - fp_base:>7}/{n_total_fp}  "
          f"{tp_base:>7}/{n_total_real}")

    for config_name, config in TCE_CONFIGS.items():
        tce_mask      = apply_tce_config(df, config)
        df_surv       = df[tce_mask].copy()
        y_surv        = y[tce_mask]
        if y_surv.sum() == 0 or (y_surv == 0).sum() == 0:
            continue
        X_surv = df_surv[available].fillna(0).values
        w_surv = compute_class_weight("balanced", classes=classes, y=y_surv)
        clf_s  = DecisionTreeClassifier(max_depth=max_depth,
                                        class_weight={0: w_surv[0], 1: w_surv[1]},
                                        random_state=42)
        clf_s.fit(X_surv, y_surv)
        pred_c          = np.zeros(len(df), dtype=int)
        pred_c[np.where(tce_mask)[0]] = clf_s.predict(X_surv)
        p_c, r_c, f1_c, tp_c, fp_c, fn_c = metrics(y, pred_c)
        marker = f"{GREEN}◀ best{RESET}" if (best_two_stage and
                                              config_name == best_two_stage["config_name"]) else ""
        print(f"  {f'TCE {config_name} → tree':<30}  {r_c*100:>6.1f}%  {p_c*100:>6.1f}%  "
              f"{f1_c:>7.3f}  {n_total_fp - fp_c:>7}/{n_total_fp}  "
              f"{tp_c:>7}/{n_total_real}  {marker}")
    print()


# ── Distribution summary ───────────────────────────────────────────────────────

def distribution_summary(df, y):
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
        sep   = abs(r_med - f_med) / (np.std(rv) + np.std(fv) + 1e-9)
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
                        help="Skip single-stage decision tree fitting")
    parser.add_argument("--two-stage-only", action="store_true",
                        help="Skip single-stage analysis, run only the two-stage comparison")
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
    n_real  = int(y.sum())
    n_fp    = int((y == 0).sum())
    n_total = len(df)

    print(f"\n  Total candidates: {n_total}")
    print(f"  Real planets:     {GREEN}{n_real}  ({n_real/n_total*100:.1f}%){RESET}")
    print(f"  False positives:  {YELLOW}{n_fp}  ({n_fp/n_total*100:.1f}%){RESET}")
    print(f"  Targets covered:  {df['target'].nunique()}")
    print(f"  Recall floor:     {args.recall_floor:.0%}")

    distribution_summary(df, y)

    if not args.two_stage_only:
        ranked = per_feature_sweep(df, y, args.recall_floor)

        if len(ranked) >= 2:
            pairwise_scan(df, y, ranked, args.recall_floor, n_pairs=4)

        if not args.no_tree:
            print(f"\n{CYAN}{'='*72}")
            print(f"  DECISION TREE  (max_depth={args.depth}, recall floor: {args.recall_floor:.0%})")
            print(f"{'='*72}{RESET}")
            fit_decision_tree(df, y, args.recall_floor, max_depth=args.depth,
                              label="single-stage, all candidates")

        print(f"\n{CYAN}{'='*72}")
        print(f"  TOP SINGLE-FEATURE THRESHOLDS")
        print(f"{'='*72}{RESET}")
        for r in ranked[:5]:
            print(f"  {r['label']:<36} {r['direction']} {r['threshold']:.3f}"
                  f"  →  recall={r['recall']*100:.1f}%  precision={r['precision']*100:.1f}%  F1={r['f1']:.3f}")

    # ── Two-stage analysis (always runs unless --no-tree) ─────────────────────
    if not args.no_tree:
        two_stage_analysis(df, y, args.recall_floor, max_depth=args.depth)

    print()


if __name__ == "__main__":
    main()