"""
tests/big_test/parse_big_test_results.py

Parse a big_test log file and print a clean summary table.

Usage:
    python tests/big_test/parse_big_test_results.py                  # latest log
    python tests/big_test/parse_big_test_results.py big_test_XYZ.log  # specific log
"""

import re
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent

# ── Log line stripping ─────────────────────────────────────────────────────────
# Lines look like: "2024-01-01 12:00:00  INFO      <message>"
_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s+\w+\s+")

def strip_prefix(line: str) -> str:
    return _PREFIX.sub("", line).rstrip()


# ── Patterns ───────────────────────────────────────────────────────────────────
RE_TARGET   = re.compile(r"^TARGET:\s+(.+)")
RE_TRUTH_N  = re.compile(r"Truth:\s+(\d+) planet")
RE_FOUND_N  = re.compile(r"Found (\d+) / (\d+) planet")
RE_OVERALL  = re.compile(r"OVERALL:\s+(.+)")
RE_PLANET_DET = re.compile(
    r"Planet \d+: P=([\d.]+)d\s+depth=([\d.]+)ppm\s+dur=([\d.]+)h\s+SDE=([\d.]+)\s+SNR=([\d.]+)"
)
RE_PERIOD_MATCH = re.compile(
    r"truth ([\d.]+)d → detected ([\d.]+)d\s+\(([+\-\d.]+)%\)"
)
RE_PERIOD_MISS  = re.compile(r"truth ([\d.]+)d → NOT DETECTED")
RE_FP_LINE      = re.compile(r"FP: P=([\d.]+)d")
RE_MCMC_PLANET  = re.compile(
    r"Planet \d+: P=([\d.]+)d\s+rp=([\d.]+) \+([\d.]+)/-([\d.]+)\s+"
    r"b=([\d.]+)\s+R=([\d.]+) \+([\d.]+)/-([\d.]+) R⊕\s+"
    r"accept=([\d.]+)\s+converged=(\w+)"
)
RE_VERDICT_LINE = re.compile(
    r"P=([\d.]+)d \(([+\-\d.]+)%\)\s+"
    r"truth R=([\d.]+) R⊕\s+"
    r"detected R=([\d.]+) \+([\d.]+)/-([\d.]+) R⊕.*→ (.+)"
)
RE_STOP_REASON  = re.compile(r"STOP REASON:\s+(.+)")
RE_STOP_CAND    = re.compile(
    r"Stopping candidate: P=([\d.]+)d.*SDE=([\d.]+)\s+SNR=([\d.]+)"
)
RE_MCMC_FAILED  = re.compile(r"Planet \d+ MCMC FAILED")
RE_CONVERGENCE  = re.compile(r"CONVERGENCE:\s+(.+)")
RE_SUMMARY_PASS = re.compile(r"PASS \(all planets\):\s+(\d+)")
RE_SUMMARY_N    = re.compile(r"Targets run:\s+(\d+)")


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse(log_path: Path) -> list[dict]:
    targets = []
    current = None

    with open(log_path, encoding="utf-8") as f:
        for raw in f:
            line = strip_prefix(raw)

            m = RE_TARGET.match(line)
            if m:
                if current:
                    targets.append(current)
                current = {
                    "target": m.group(1).strip(),
                    "truth_n": None,
                    "found_n": None,
                    "overall": None,
                    "detected_planets": [],   # list of {period, depth_ppm, dur_h, sde, snr}
                    "period_matches": [],     # list of {truth_p, det_p, pct_err}
                    "missed_periods": [],
                    "false_positives": [],    # list of periods
                    "mcmc_planets": [],       # list of {period, r_med, r_lo, r_hi, rp, b, accept, converged}
                    "planet_verdicts": [],    # list of {truth_p, truth_r, det_r, interval, pct_err, verdict}
                    "stop_reasons": [],
                    "stop_candidate": None,
                    "convergence_notes": [],
                    "mcmc_failures": 0,
                }
                continue

            if current is None:
                continue

            if m := RE_TRUTH_N.search(line):
                current["truth_n"] = int(m.group(1))

            elif m := RE_FOUND_N.search(line):
                current["found_n"] = int(m.group(1))

            elif m := RE_OVERALL.match(line):
                current["overall"] = m.group(1).strip()

            elif m := RE_PLANET_DET.search(line):
                current["detected_planets"].append({
                    "period": float(m.group(1)),
                    "depth_ppm": float(m.group(2)),
                    "dur_h": float(m.group(3)),
                    "sde": float(m.group(4)),
                    "snr": float(m.group(5)),
                })

            elif m := RE_PERIOD_MATCH.search(line):
                current["period_matches"].append({
                    "truth_p": float(m.group(1)),
                    "det_p": float(m.group(2)),
                    "pct_err": float(m.group(3)),
                })

            elif m := RE_PERIOD_MISS.search(line):
                current["missed_periods"].append(float(m.group(1)))

            elif m := RE_FP_LINE.search(line):
                current["false_positives"].append(float(m.group(1)))

            elif m := RE_MCMC_PLANET.search(line):
                current["mcmc_planets"].append({
                    "period": float(m.group(1)),
                    "rp": float(m.group(2)),
                    "r_med": float(m.group(6)),
                    "r_hi": float(m.group(7)),
                    "r_lo": float(m.group(8)),
                    "accept": float(m.group(9)),
                    "converged": m.group(10) == "True",
                })

            elif m := RE_VERDICT_LINE.search(line):
                current["planet_verdicts"].append({
                    "truth_p": float(m.group(1)),
                    "period_pct_err": float(m.group(2)),
                    "truth_r": float(m.group(3)),
                    "det_r": float(m.group(4)),
                    "det_r_hi": float(m.group(5)),
                    "det_r_lo": float(m.group(6)),
                    "verdict": m.group(7).strip(),
                })

            elif m := RE_STOP_REASON.search(line):
                current["stop_reasons"].append(m.group(1).strip())

            elif m := RE_STOP_CAND.search(line):
                current["stop_candidate"] = {
                    "period": float(m.group(1)),
                    "sde": float(m.group(2)),
                    "snr": float(m.group(3)),
                }

            elif m := RE_CONVERGENCE.search(line):
                current["convergence_notes"].append(m.group(1).strip())

            elif RE_MCMC_FAILED.search(line):
                current["mcmc_failures"] += 1

    if current:
        targets.append(current)

    return targets


# ── Formatting helpers ─────────────────────────────────────────────────────────

VERDICT_COLOR = {
    "PASS":             "\033[92m",   # green
    "FALSE_POSITIVE":   "\033[93m",   # yellow
    "MISSED":           "\033[91m",   # red
    "RADIUS_FAIL":      "\033[91m",   # red
    "FAIL_RADIUS":      "\033[91m",
    "MCMC_FAILED":      "\033[91m",
    "ERROR":            "\033[91m",
    "LC_FETCH_FAILED":  "\033[91m",
    "STELLAR_FETCH_FAILED": "\033[91m",
    "DETECTION_FAILED": "\033[91m",
}
RESET = "\033[0m"

def color(text: str, verdict: str) -> str:
    for key, code in VERDICT_COLOR.items():
        if key in verdict.upper():
            return f"{code}{text}{RESET}"
    return text


def print_summary(targets: list[dict]):
    n = len(targets)
    passed = sum(1 for t in targets if t["overall"] == "PASS")
    fp     = sum(1 for t in targets if t["overall"] and "FALSE_POSITIVE" in t["overall"])
    missed = sum(1 for t in targets if t["overall"] and "MISSED" in t["overall"])
    r_fail = sum(1 for t in targets if t["overall"] and "RADIUS_FAIL" in t["overall"])
    errors = sum(1 for t in targets if t["overall"] in (
        None, "ERROR", "LC_FETCH_FAILED", "STELLAR_FETCH_FAILED",
        "DETECTION_FAILED", "MCMC_FAILED",
    ))

    print()
    print("=" * 78)
    print(f"  {'TARGET':<22} {'FOUND':>5} {'TRUTH':>5}  {'VERDICT'}")
    print("=" * 78)
    for t in targets:
        overall = t["overall"] or "ERROR"
        found   = t["found_n"] if t["found_n"] is not None else "?"
        truth   = t["truth_n"] if t["truth_n"] is not None else "?"
        row = f"  {t['target']:<22} {str(found):>5} {str(truth):>5}  {overall}"
        print(color(row, overall))
    print("=" * 78)
    print(f"  Targets:        {n}")
    print(color(f"  PASS:           {passed}  ({passed/n*100:.0f}%)", "PASS"))
    if fp:
        print(color(f"  False positives:{fp}", "FALSE_POSITIVE"))
    if missed:
        print(color(f"  Missed:         {missed}", "MISSED"))
    if r_fail:
        print(color(f"  Radius fail:    {r_fail}", "RADIUS_FAIL"))
    if errors:
        print(color(f"  Errors:         {errors}", "ERROR"))
    print()


def print_detail(targets: list[dict]):
    for t in targets:
        overall = t["overall"] or "ERROR"
        if overall == "PASS":
            continue  # only expand failures

        print()
        print(color(f"  ── {t['target']}  [{overall}] ──", overall))

        # Period matching
        for pm in t["period_matches"]:
            print(f"     Period {pm['truth_p']:.4f}d → detected {pm['det_p']:.4f}d  ({pm['pct_err']:+.2f}%)")
        for mp in t["missed_periods"]:
            print(color(f"     Period {mp:.4f}d → NOT DETECTED", "MISSED"))

        # False positives
        for fp_p in t["false_positives"]:
            print(color(f"     FALSE POSITIVE: P={fp_p:.4f}d", "FALSE_POSITIVE"))

        # Planet-level verdicts
        for pv in t["planet_verdicts"]:
            v = pv["verdict"]
            det_r  = pv["det_r"]
            det_lo = pv["det_r_lo"]
            det_hi = pv["det_r_hi"]
            interval = f"[{det_r-det_lo:.2f}, {det_r+det_hi:.2f}]"
            line = (
                f"     P={pv['truth_p']:.4f}d  "
                f"truth R={pv['truth_r']:.2f}  "
                f"det R={det_r:.2f} +{det_hi:.2f}/-{det_lo:.2f}  "
                f"{interval}  → {v}"
            )
            print(color(line, v))

        # Stop reason if pipeline stopped early
        if t["stop_candidate"]:
            sc = t["stop_candidate"]
            print(f"     Stopping candidate: P={sc['period']:.4f}d  SDE={sc['sde']:.2f}  SNR={sc['snr']:.2f}")
        for sr in t["stop_reasons"]:
            print(color(f"     Stop reason: {sr}", "MISSED"))

        # Convergence issues
        for note in t["convergence_notes"]:
            print(color(f"     Convergence: {note}", "RADIUS_FAIL"))

        if t["mcmc_failures"]:
            print(color(f"     MCMC failures: {t['mcmc_failures']}", "ERROR"))


# ── Radius accuracy table (for passing targets) ────────────────────────────────

def print_radius_table(targets: list[dict]):
    rows = []
    for t in targets:
        for pv in t["planet_verdicts"]:
            det_r  = pv["det_r"]
            det_lo = pv["det_r_lo"]
            det_hi = pv["det_r_hi"]
            err_pct = (det_r - pv["truth_r"]) / pv["truth_r"] * 100
            rows.append({
                "target":   t["target"],
                "period":   pv["truth_p"],
                "truth_r":  pv["truth_r"],
                "det_r":    det_r,
                "det_lo":   det_lo,
                "det_hi":   det_hi,
                "err_pct":  err_pct,
                "verdict":  pv["verdict"],
            })

    if not rows:
        return

    print()
    print("  RADIUS ACCURACY (all matched planets)")
    print(f"  {'TARGET':<22} {'P (d)':>8} {'Truth R⊕':>9} {'Det R⊕':>8} {'Interval':>16} {'Err%':>7}  Verdict")
    print("  " + "-" * 76)
    for r in rows:
        interval = f"[{r['det_r']-r['det_lo']:.2f}, {r['det_r']+r['det_hi']:.2f}]"
        row = (
            f"  {r['target']:<22} {r['period']:>8.4f} {r['truth_r']:>9.2f} "
            f"{r['det_r']:>8.2f} {interval:>16} {r['err_pct']:>+7.1f}%  {r['verdict']}"
        )
        print(color(row, r["verdict"]))
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
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

    print_summary(targets)
    print_detail(targets)
    print_radius_table(targets)


if __name__ == "__main__":
    main()
