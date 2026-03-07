"""
tests/big_test/run_big_test.py

Batch validation of the full detection + characterization pipeline against
NASA truth data (tests/summary.csv). Runs on all confirmed Kepler systems
where every planet has a period between 2 and 100 days.

Passing grade for each planet:
  - Period detected within 5% of truth
  - Truth radius falls within the MCMC 1-sigma uncertainty interval
    i.e. (detected_median - lo_err) <= truth_radius <= (detected_median + hi_err)

When the pipeline finds too many planets (false positive), logs exactly what
BLS stats made that candidate pass reliability vetting.

When the pipeline stops short (missed planets), logs exactly which reliability
flag tripped on the stopping candidate.

Run from the project root:
    python tests/big_test/run_big_test.py

Output: tests/big_test/big_test_YYYYMMDD_HHMMSS.log
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MEDIUM
from tests.helpers import get_light_curve
from exotransit.detection.multi_planet import find_all_planets
from exotransit.detection.bls import run_bls
from exotransit.pipeline.light_curves import LightCurveData
from exotransit.mcmc.fit_mcmc import run_mcmc
from exotransit.physics.stars import query_stellar_params
from exotransit.physics.planets import derive_planet_physics
from exotransit.physics.limb_darkening import get_limb_darkening

import lightkurve as lk

# ── Logging setup ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"big_test_{timestamp}.log"

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

logger = logging.getLogger("big_test")

# Capture BLS rejection warnings from pipeline loggers
for _mod in [
    "exotransit.detection.bls",
    "exotransit.detection.multi_planet",
    "exotransit.pipeline.light_curves",
]:
    logging.getLogger(_mod).setLevel(logging.WARNING)

# Suppress lightkurve's cadence quality mask noise
logging.getLogger("lightkurve").setLevel(logging.ERROR)
logging.getLogger("lightkurve.utils").setLevel(logging.ERROR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_values(val) -> list[float]:
    if isinstance(val, (int, float)):
        return [float(val)]
    return [float(x.strip()) for x in str(val).split(",")]


def match_periods(truth_periods, detected_periods, tol=0.05):
    """
    Greedily match detected periods to truth periods within fractional tolerance.
    Returns list of (truth_p, detected_p | None, pct_error | None).
    """
    used = set()
    matches = []
    for tp in truth_periods:
        best_idx, best_err = None, float("inf")
        for j, dp in enumerate(detected_periods):
            if j in used:
                continue
            err = abs(dp - tp) / tp
            if err < tol and err < best_err:
                best_idx, best_err = j, err
        if best_idx is not None:
            used.add(best_idx)
            matches.append((tp, detected_periods[best_idx], best_err * 100))
        else:
            matches.append((tp, None, None))
    return matches


def radius_passes(truth_r, det_med, det_lo, det_hi) -> bool:
    """Truth radius falls within the MCMC 1-sigma interval."""
    return (det_med - det_lo) <= truth_r <= (det_med + det_hi)


def probe_next_candidate(refined_lc, found_planets, conf):
    """
    Run one more BLS pass on the median-filled LC (as find_all_planets would)
    to surface the candidate that triggered the stop, with its flags.
    Returns the BLSResult or None.
    """
    try:
        lc_lk = lk.LightCurve(
            time=refined_lc.time,
            flux=refined_lc.flux,
            flux_err=refined_lc.flux_err,
        )
        # Median-fill all found planet transits
        for bls in found_planets:
            mask = lc_lk.create_transit_mask(
                period=bls.best_period,
                transit_time=bls.best_t0,
                duration=bls.best_duration * 3.0,
            )
            lc_lk.flux.value[mask] = np.nanmedian(lc_lk.flux.value)

        probe_lc = LightCurveData(
            time=np.asarray(lc_lk.time.value),
            flux=np.asarray(lc_lk.flux.value),
            flux_err=np.asarray(lc_lk.flux_err.value),
            mission=refined_lc.mission,
            target_name=refined_lc.target_name,
            sector_or_quarter=refined_lc.sector_or_quarter,
            raw_time=refined_lc.raw_time,
            raw_flux=refined_lc.raw_flux,
        )
        return run_bls(
            probe_lc,
            min_period=conf.bls.min_period,
            max_period=conf.bls.max_period,
            max_period_grid_points=conf.bls.max_period_grid_points,
        )
    except Exception as e:
        logger.warning(f"  Could not probe next candidate: {e}")
        return None


# ── Per-target pipeline ───────────────────────────────────────────────────────

def run_target(target_name, truth_n, truth_periods, truth_radii, conf):
    divider = "=" * 72
    logger.info(divider)
    logger.info(f"TARGET: {target_name}")
    logger.info(f"  Truth: {truth_n} planet(s)")
    logger.info(f"  Truth periods (d): {[f'{p:.4f}' for p in truth_periods]}")
    logger.info(f"  Truth radii  (R⊕): {[f'{r:.2f}' for r in truth_radii]}")

    result = {
        "target":        target_name,
        "truth_n":       truth_n,
        "truth_periods": truth_periods,
        "truth_radii":   truth_radii,
        "found_n":       0,
        "verdict":       "ERROR",
        "planet_results": [],
    }

    # ── 1. Light curve ────────────────────────────────────────────────────────
    try:
        lc = get_light_curve(target_name, mission="Kepler", max_quarters=conf.max_quarters)
        logger.info(f"  LC:   {len(lc.time):,} pts  baseline {lc.time[-1]-lc.time[0]:.1f} d  "
                    f"cadence {np.median(np.diff(lc.time))*24*60:.0f} min")
    except Exception as e:
        logger.error(f"  LC FETCH FAILED: {e}")
        result["verdict"] = "LC_FETCH_FAILED"
        return result

    # ── 2. Stellar params & limb darkening ───────────────────────────────────
    try:
        stellar = query_stellar_params(target_name)
        ld = get_limb_darkening(
            teff=stellar.teff, logg=stellar.logg, metallicity=stellar.metallicity
        )
        logger.info(f"  Star: R*={stellar.radius:.3f} R☉  M*={stellar.mass:.3f} M☉  "
                    f"Teff={stellar.teff:.0f} K  logg={stellar.logg:.2f}")
        logger.info(f"  LD:   u1={ld.u1:.4f}  u2={ld.u2:.4f}")
    except Exception as e:
        logger.error(f"  STELLAR FETCH FAILED: {e}")
        result["verdict"] = "STELLAR_FETCH_FAILED"
        return result

    # ── 3. Detection ──────────────────────────────────────────────────────────
    logger.info("  --- DETECTION ---")
    try:
        planets, _, refined_lc = find_all_planets(
            lc,
            max_planets=conf.max_planets,
            min_period=conf.bls.min_period,
            max_period=conf.bls.max_period,
            max_period_grid_points=conf.bls.max_period_grid_points,
        )
    except Exception as e:
        logger.error(f"  DETECTION FAILED: {e}")
        result["verdict"] = "DETECTION_FAILED"
        return result

    n_found = len(planets)
    result["found_n"] = n_found
    found_periods = [p.best_period for p in planets]
    logger.info(f"  Found {n_found} / {truth_n} planet(s)")

    for i, bls in enumerate(planets):
        logger.info(
            f"    Planet {i+1}: P={bls.best_period:.4f}d  "
            f"depth={bls.transit_depth*1e6:.1f}ppm  dur={bls.best_duration*24:.2f}h  "
            f"SDE={bls.sde:.2f}  SNR={bls.snr:.2f}"
            + (f"  aliases={bls.aliases}" if bls.aliases else "")
        )

    # ── 3a. If stopped early — probe what the stopping candidate looked like ──
    if n_found < truth_n:
        logger.warning(f"  STOPPED EARLY: {n_found}/{truth_n} planets found.")
        logger.warning("  Probing next BLS candidate to show what triggered the stop:")
        next_bls = probe_next_candidate(refined_lc, planets, conf)
        if next_bls is not None:
            logger.warning(
                f"    Stopping candidate: P={next_bls.best_period:.4f}d  "
                f"depth={next_bls.transit_depth*1e6:.1f}ppm  dur={next_bls.best_duration*24:.2f}h  "
                f"SDE={next_bls.sde:.2f}  SNR={next_bls.snr:.2f}  "
                f"reliable={next_bls.is_reliable}"
            )
            if next_bls.reliability_flags:
                for flag in next_bls.reliability_flags:
                    logger.warning(f"    STOP REASON: {flag}")
            else:
                logger.warning("    (Candidate passed reliability — pipeline may have hit max_planets)")

    # ── 3b. Period matching ───────────────────────────────────────────────────
    period_matches = match_periods(truth_periods, found_periods)
    logger.info("  --- PERIOD MATCHING (5% tolerance) ---")
    for tp, dp, err in period_matches:
        if dp is not None:
            logger.info(f"    truth {tp:.4f}d → detected {dp:.4f}d  ({err:+.2f}%)")
        else:
            logger.warning(f"    truth {tp:.4f}d → NOT DETECTED")

    # ── 3c. If too many found — explain the false positives ───────────────────
    matched_detected = {dp for _, dp, _ in period_matches if dp is not None}
    false_positive_bls = [
        bls for bls in planets
        if not any(abs(bls.best_period - dp) / dp < 0.001
                   for dp in matched_detected if dp is not None)
    ]
    if false_positive_bls:
        logger.warning(f"  FALSE POSITIVE(S): {len(false_positive_bls)} detected planet(s) not in truth.")
        logger.warning("  These passed all reliability flags — here is why they looked real:")
        for bls in false_positive_bls:
            logger.warning(
                f"    FP: P={bls.best_period:.4f}d  "
                f"depth={bls.transit_depth*1e6:.1f}ppm  dur={bls.best_duration*24:.2f}h  "
                f"SDE={bls.sde:.2f}  SNR={bls.snr:.2f}"
                + (f"  aliases={bls.aliases}" if bls.aliases else "  no aliases flagged")
            )

    # ── 4. MCMC + physics ─────────────────────────────────────────────────────
    logger.info("  --- MCMC + RADIUS ---")
    planet_results = []
    for i, bls in enumerate(planets):
        pr = {"period": bls.best_period, "sde": bls.sde, "snr": bls.snr}
        try:
            mcmc = run_mcmc(
                refined_lc, bls,
                n_walkers=conf.mcmc.n_walkers,
                n_steps=conf.mcmc.n_steps,
                n_burnin=conf.mcmc.n_burnin,
                u1=ld.u1, u2=ld.u2,
                stellar_mass=stellar.mass,
                stellar_radius=stellar.radius,
            )
            phys = derive_planet_physics(mcmc, stellar)
            r_med, r_lo, r_hi = phys.radius_earth
            pr.update({
                "rp_med": mcmc.rp_med, "rp_err": mcmc.rp_err,
                "b_med": mcmc.b_med,
                "r_earth": (r_med, r_lo, r_hi),
                "converged": mcmc.converged,
                "acceptance": mcmc.acceptance_fraction,
            })
            logger.info(
                f"    Planet {i+1}: P={bls.best_period:.4f}d  "
                f"rp={mcmc.rp_med:.4f} +{mcmc.rp_err[1]:.4f}/-{mcmc.rp_err[0]:.4f}  "
                f"b={mcmc.b_med:.3f}  "
                f"R={r_med:.2f} +{r_hi:.2f}/-{r_lo:.2f} R⊕  "
                f"accept={mcmc.acceptance_fraction:.3f}  converged={mcmc.converged}"
            )
            if not mcmc.converged:
                for note in mcmc.convergence_notes:
                    logger.warning(f"      CONVERGENCE: {note}")
        except Exception as e:
            logger.error(f"    Planet {i+1} MCMC FAILED: {e}")
            pr["r_earth"] = None
        planet_results.append(pr)

    result["planet_results"] = planet_results

    # ── 5. Verdict per planet ─────────────────────────────────────────────────
    logger.info("  --- VERDICT ---")
    planet_verdicts = []
    for (tp, dp, period_err), truth_r in zip(period_matches, truth_radii):
        if dp is None:
            logger.warning(f"    P={tp:.4f}d  R={truth_r:.2f} R⊕  → MISSED")
            planet_verdicts.append("MISSED")
            continue
        idx = next(
            (i for i, bls in enumerate(planets) if abs(bls.best_period - dp) / dp < 0.001),
            None,
        )
        if idx is None or planet_results[idx].get("r_earth") is None:
            logger.warning(f"    P={tp:.4f}d  R={truth_r:.2f} R⊕  → PERIOD MATCHED / MCMC FAILED")
            planet_verdicts.append("MCMC_FAILED")
            continue

        r_med, r_lo, r_hi = planet_results[idx]["r_earth"]
        passes = radius_passes(truth_r, r_med, r_lo, r_hi)
        r_err_pct = (r_med - truth_r) / truth_r * 100
        verdict_str = "PASS" if passes else "FAIL (radius outside 1σ)"
        logger.info(
            f"    P={tp:.4f}d ({period_err:+.2f}%)  "
            f"truth R={truth_r:.2f} R⊕  "
            f"detected R={r_med:.2f} +{r_hi:.2f}/-{r_lo:.2f} R⊕  "
            f"[{r_med-r_lo:.2f}, {r_med+r_hi:.2f}]  "
            f"err={r_err_pct:+.1f}%  → {verdict_str}"
        )
        planet_verdicts.append("PASS" if passes else "FAIL_RADIUS")

    # Extra detected planets not in truth
    for _ in false_positive_bls:
        planet_verdicts.append("FALSE_POSITIVE")

    # Overall verdict
    if any(v == "FALSE_POSITIVE" for v in planet_verdicts):
        verdict = f"FALSE_POSITIVE ({n_found} found, {truth_n} truth)"
    elif any(v == "MISSED" for v in planet_verdicts):
        verdict = f"MISSED ({n_found}/{truth_n} planets)"
    elif any(v == "FAIL_RADIUS" for v in planet_verdicts):
        n_fail = sum(1 for v in planet_verdicts if v == "FAIL_RADIUS")
        verdict = f"RADIUS_FAIL ({n_fail} planet(s) outside 1σ)"
    elif any(v == "MCMC_FAILED" for v in planet_verdicts):
        verdict = "MCMC_FAILED"
    else:
        verdict = "PASS"

    result["verdict"] = verdict
    logger.info(f"  OVERALL: {verdict}")
    logger.info("")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    conf = MEDIUM
    summary_path = PROJECT_ROOT / "tests" / "summary.csv"

    logger.info("=" * 72)
    logger.info("EXOTRANSIT BIG TEST")
    logger.info(f"  Config:     MEDIUM  ({conf.max_quarters} quarters, "
                f"{conf.bls.max_period_grid_points:,} BLS points, "
                f"{conf.mcmc.n_steps:,} MCMC steps)")
    logger.info(f"  Truth data: {summary_path}")
    logger.info(f"  Log file:   {LOG_FILE}")
    logger.info("=" * 72)

    df = pd.read_csv(summary_path)
    df["periods_list"] = df["periods_d"].apply(parse_values)
    df["radii_list"]   = df["radii_Rearth"].apply(parse_values)
    df["min_period"]   = df["periods_list"].apply(min)
    df["max_period"]   = df["periods_list"].apply(max)

    df_run = df[(df["min_period"] > 2) & (df["max_period"] < 100)].copy().reset_index(drop=True)

    logger.info(f"Total systems in truth data:             {len(df)}")
    logger.info(f"Systems with all periods in (2, 100) d: {len(df_run)}")
    logger.info("")

    results = []
    for _, row in df_run.iterrows():
        target = f"Kepler-{int(row['Kepler_#'])}"
        r = run_target(
            target_name=target,
            truth_n=int(row["n_planets"]),
            truth_periods=row["periods_list"],
            truth_radii=row["radii_list"],
            conf=conf,
        )
        results.append(r)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 72)

    verdicts = [r.get("verdict", "ERROR") for r in results]
    n         = len(results)
    passed    = sum(1 for v in verdicts if v == "PASS")
    fp        = sum(1 for v in verdicts if "FALSE_POSITIVE" in v)
    missed    = sum(1 for v in verdicts if "MISSED" in v)
    r_fail    = sum(1 for v in verdicts if "RADIUS_FAIL" in v)
    errors    = sum(1 for v in verdicts if v in (
        "ERROR", "LC_FETCH_FAILED", "STELLAR_FETCH_FAILED", "DETECTION_FAILED", "MCMC_FAILED"
    ))

    logger.info(f"  Targets run:           {n}")
    logger.info(f"  PASS (all planets):    {passed}  ({passed/n*100:.0f}%)")
    logger.info(f"  False positives:       {fp}")
    logger.info(f"  Missed planets:        {missed}")
    logger.info(f"  Radius outside 1σ:     {r_fail}")
    logger.info(f"  Pipeline errors:       {errors}")
    logger.info("")
    logger.info("  Per-target:")
    for r in results:
        logger.info(
            f"    {r.get('target','?'):20s}  "
            f"found {r.get('found_n',0)}/{r.get('truth_n','?')}  "
            f"{r.get('verdict', r.get('status','ERROR'))}"
        )

    logger.info("")
    logger.info(f"Full log: {LOG_FILE}")


if __name__ == "__main__":
    main()
