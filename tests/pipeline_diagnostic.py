"""
tests/pipeline_diagnostic.py

Diagnostic pipeline runner: BLS + MCMC with verbose per-step logging,
compared against published values from examples.csv.

Run from the repo root:
    python -m tests.test_pipeline_diagnostic

Output goes entirely to the console. No assertions — this is for diagnosis, not CI.
"""

import logging
import sys
import numpy as np
import pandas as pd
import lightkurve as lk
from pathlib import Path

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MEDIUM
from tests.helpers import get_light_curve
from exotransit.detection.bls import run_bls
from exotransit.mcmc.fit_mcmc import run_mcmc
from exotransit.physics.stars import query_stellar_params
from exotransit.physics.limb_darkening import get_limb_darkening

# ── configuration ──────────────────────────────────────────────────────────────
TARGETS = [7, 144, 9, 15, 11]
conf = MEDIUM
TRUTH_CSV = Path(__file__).parent / "examples.csv"
PERIOD_MATCH_TOL = 0.20   # 20% tolerance for matching detected period to truth
R_SUN_TO_EARTH = 109.076  # 1 R_sun in Earth radii

# ── logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # suppress noisy library logs
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("diagnostic")
log.setLevel(logging.DEBUG)

# Route diagnostic logger to stdout at DEBUG; keep everything else at WARNING
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)
log.propagate = False  # don't double-print through root handler


# ── helpers ────────────────────────────────────────────────────────────────────

def _sep(char="─", width=72):
    log.info(char * width)


def _header(text):
    _sep("═")
    log.info(f"  {text}")
    _sep("═")


def _section(text):
    _sep()
    log.info(f"  {text}")
    _sep()


def _match_truth(found_period, truth_rows):
    """Return the closest truth row within PERIOD_MATCH_TOL, or None."""
    if truth_rows.empty:
        return None
    diffs = np.abs(truth_rows["Period_d"].values - found_period) / truth_rows["Period_d"].values
    idx = np.argmin(diffs)
    if diffs[idx] <= PERIOD_MATCH_TOL:
        return truth_rows.iloc[idx]
    return None


def _pct_err(measured, truth):
    """Fractional error: (measured - truth) / truth, as percent string."""
    if truth == 0 or np.isnan(truth):
        return "n/a"
    err = (measured - truth) / abs(truth)
    sign = "+" if err >= 0 else ""
    return f"{sign}{err * 100:.1f}%"


def _flag(condition, label):
    return f"  *** {label}" if condition else ""


# ── main diagnostic function ───────────────────────────────────────────────────

def run_diagnostic(kepler_num: int, truth_df: pd.DataFrame):
    target = f"Kepler-{kepler_num}"
    truth_rows = truth_df[truth_df["Kepler_#"] == float(kepler_num)].sort_values("Period_d")

    _header(f"TARGET: {target}  ({len(truth_rows)} confirmed planets in truth)")

    if truth_rows.empty:
        log.warning(f"  No truth rows found for Kepler-{kepler_num} in examples.csv")

    # ── fetch light curve ──────────────────────────────────────────────────────
    log.info(f"\n  Fetching light curve (max_quarters={conf.max_quarters})...")
    lc = get_light_curve(target, mission="Kepler", max_quarters=conf.max_quarters)
    baseline = lc.time[-1] - lc.time[0]
    cadence_min = float(np.median(np.diff(lc.time))) * 24 * 60

    log.info(f"  Points:    {len(lc.time):,}")
    log.info(f"  Baseline:  {baseline:.1f} days")
    log.info(f"  Cadence:   {cadence_min:.1f} min")
    log.info(f"  Flux min:  {lc.flux.min():.6f}   max: {lc.flux.max():.6f}")
    log.info(f"  Median err:{np.median(lc.flux_err):.6f}")

    # ── fetch stellar params and limb darkening ────────────────────────────────
    log.info(f"\n  Fetching stellar params from NASA Exoplanet Archive...")
    stellar = query_stellar_params(target)
    ld = get_limb_darkening(teff=stellar.teff, logg=stellar.logg, metallicity=stellar.metallicity)

    log.info(f"  R_star:  {stellar.radius:.4f} R_sun")
    log.info(f"  M_star:  {stellar.mass:.4f} M_sun")
    log.info(f"  Teff:    {stellar.teff:.0f} K")
    log.info(f"  u1={ld.u1:.4f}  u2={ld.u2:.4f}")

    if not truth_rows.empty:
        t_rstar = truth_rows["Rstar_Rsun"].iloc[0]
        log.info(f"  Truth R_star from CSV: {t_rstar:.4f} R_sun  "
                 f"(archive query returned {stellar.radius:.4f}, "
                 f"diff={_pct_err(stellar.radius, t_rstar)})")

    # ── BLS masking loop ───────────────────────────────────────────────────────
    log.info(f"\n  Running BLS (max_planets={conf.max_planets})...")
    lc_work = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
    found_bls = []

    for planet_i in range(conf.max_planets):
        from exotransit.pipeline.light_curves import LightCurveData
        current_lc = LightCurveData(
            time=np.asarray(lc_work.time.value),
            flux=np.asarray(lc_work.flux.value),
            flux_err=np.asarray(lc_work.flux_err.value),
            mission=lc.mission,
            target_name=lc.target_name,
            sector_or_quarter=lc.sector_or_quarter,
            raw_time=lc.raw_time,
            raw_flux=lc.raw_flux,
        )
        bls = run_bls(
            current_lc,
            min_period=conf.bls.min_period,
            max_period=conf.bls.max_period,
            max_period_grid_points=conf.bls.max_period_grid_points,
        )

        if not bls.is_reliable:
            log.info(f"\n  BLS stopped at planet {planet_i + 1}: not reliable")
            for flag in bls.reliability_flags:
                log.info(f"    {flag}")
            break

        # duplicate check
        is_dup = any(
            abs(bls.best_period - ex.best_period) / ex.best_period < 0.05
            for ex in found_bls
        )

        # mask and fill regardless
        mask = lc_work.create_transit_mask(
            period=bls.best_period,
            transit_time=bls.best_t0,
            duration=bls.best_duration * 3.0,
        )
        lc_work.flux.value[mask] = np.nanmedian(lc_work.flux.value)

        if is_dup:
            log.info(f"  Skipping duplicate at {bls.best_period:.4f}d")
            continue

        found_bls.append(bls)
        _diagnose_planet(planet_i, bls, stellar, ld, truth_rows, lc)

    # ── summary of missed truth planets ───────────────────────────────────────
    _section(f"{target} — DETECTION SUMMARY")
    log.info(f"  Found {len(found_bls)} planet(s) | Truth has {len(truth_rows)} confirmed")

    for _, trow in truth_rows.iterrows():
        matched = any(
            abs(b.best_period - trow["Period_d"]) / trow["Period_d"] <= PERIOD_MATCH_TOL
            for b in found_bls
        )
        status = "FOUND" if matched else "*** MISSED ***"
        log.info(f"  {status:15s}  {trow['Planet']:20s}  P={trow['Period_d']:.4f}d  "
                 f"R={trow['Radius_Rearth']:.2f} R_earth")

    for b in found_bls:
        t = _match_truth(b.best_period, truth_rows)
        if t is None:
            log.info(f"  *** FALSE POSITIVE ***  P={b.best_period:.4f}d  (no truth match within {PERIOD_MATCH_TOL*100:.0f}%)")


def _diagnose_planet(planet_i, bls, stellar, ld, truth_rows, lc):
    """Full diagnostic output for one BLS detection, including MCMC."""

    truth = _match_truth(bls.best_period, truth_rows)
    label = truth["Planet"] if truth is not None else "NO TRUTH MATCH"

    _section(f"PLANET {planet_i + 1} — {label}  (P={bls.best_period:.4f}d)")

    # ══ §1 BLS OUTPUT ══════════════════════════════════════════════════════════
    log.info("  [§1 BLS OUTPUT]")
    log.info(f"    Period:        {bls.best_period:.5f} d")
    log.info(f"    Duration:      {bls.best_duration * 24:.3f} h")
    log.info(f"    Depth:         {bls.transit_depth * 1e6:.1f} ppm  ({bls.transit_depth:.6f})")
    log.info(f"    SDE:           {bls.sde:.2f}")
    log.info(f"    SNR:           {bls.snr:.2f}")

    baseline = lc.time[-1] - lc.time[0]
    expected_transits = baseline / bls.best_period
    cadence_days = float(np.median(np.diff(lc.time)))
    expected_pts_per_transit = bls.best_duration / cadence_days
    # in-transit points: re-derive from folded flux
    in_transit_mask_bls = np.abs(bls.folded_time) < bls.best_duration / 2
    n_in_transit = int(in_transit_mask_bls.sum())

    log.info(f"    In-transit pts:{n_in_transit}  (expected ~{expected_pts_per_transit:.1f}/transit x {expected_transits:.1f} transits)")
    log.info(f"    Expected transits in baseline: {expected_transits:.2f}")
    log.info(f"    Aliases:       {bls.aliases if bls.aliases else 'none'}")

    if truth is not None:
        t_period = truth["Period_d"]
        t_depth  = truth["Depth_ppm"] / 1e6
        t_dur_h  = truth["Duration_h"]
        log.info(f"\n    vs truth [{label}]:")
        log.info(f"      Period:   {bls.best_period:.5f} d   truth={t_period:.5f} d   err={_pct_err(bls.best_period, t_period)}")
        log.info(f"      Depth:    {bls.transit_depth*1e6:.1f} ppm   truth={truth['Depth_ppm']:.1f} ppm   err={_pct_err(bls.transit_depth, t_depth)}")
        log.info(f"      Duration: {bls.best_duration*24:.3f} h     truth={t_dur_h:.3f} h     err={_pct_err(bls.best_duration*24, t_dur_h)}")
    else:
        log.info("    *** NO TRUTH MATCH within tolerance — possible false positive ***")

    # ══ §2 FOLDED FLUX ANALYSIS ════════════════════════════════════════════════
    log.info("\n  [§2 FOLDED FLUX ANALYSIS]")

    folded_t = bls.folded_time
    folded_f = bls.folded_flux

    # Use same half-duration window as bls.py uses for depth measurement
    half_dur = max(bls.best_duration / 2, 1.5 * cadence_days)
    in_win  = np.abs(folded_t) < half_dur
    out_win = ~in_win

    n_in  = int(in_win.sum())
    n_out = int(out_win.sum())

    baseline_median = float(np.median(folded_f[out_win])) if n_out > 0 else 1.0
    baseline_std    = float(np.std(folded_f[out_win]))    if n_out > 0 else np.nan
    transit_min     = float(np.min(folded_f[in_win]))     if n_in  > 0 else np.nan

    implied_depth_fold = baseline_median - transit_min if n_in > 0 else np.nan

    rp_init_from_bls  = float(np.sqrt(max(bls.transit_depth, 1e-6)))
    rp_init_from_fold = float(np.sqrt(max(implied_depth_fold, 1e-6))) if n_in > 0 else np.nan

    depth_disagree = (
        n_in > 0 and not np.isnan(implied_depth_fold)
        and abs(bls.transit_depth - implied_depth_fold) / max(implied_depth_fold, 1e-9) > 0.05
    )

    log.info(f"    Transit window:   ±{half_dur*24:.2f} h  ({n_in} in-transit, {n_out} out-of-transit pts)")
    log.info(f"    Baseline median:  {baseline_median:.6f}  (should be ~1.0{_flag(abs(baseline_median - 1.0) > 0.002, 'BASELINE OFFSET > 200ppm')})")
    log.info(f"    Baseline std:     {baseline_std*1e6:.1f} ppm  (noise level)")
    log.info(f"    Transit minimum:  {transit_min:.6f}")
    log.info(f"    Implied depth (baseline - min):  {implied_depth_fold*1e6:.1f} ppm")
    log.info(f"    bls.transit_depth:               {bls.transit_depth*1e6:.1f} ppm{_flag(depth_disagree, 'DISAGREE >5%')}")

    if depth_disagree:
        log.info(f"    *** DEPTH MISMATCH: folded flux says {implied_depth_fold*1e6:.1f} ppm, "
                 f"bls.transit_depth says {bls.transit_depth*1e6:.1f} ppm ***")
        log.info(f"        -> Problem is in how depth is computed from the folded flux, NOT upstream in detrending")

    log.info(f"\n    rp_init (from bls.transit_depth):  {rp_init_from_bls:.5f}")
    log.info(f"    rp_init (from folded flux min):    {rp_init_from_fold:.5f}{_flag(abs(rp_init_from_bls - rp_init_from_fold) > 0.005, 'DIFFER > 0.005')}")

    if truth is not None:
        t_rp = truth["rp_over_Rstar"]
        t_depth_ppm = truth["Depth_ppm"]
        log.info(f"\n    Truth rp/R*:     {t_rp:.5f}")
        log.info(f"    Truth depth:     {t_depth_ppm:.1f} ppm  ({t_depth_ppm/1e6:.6f})")
        log.info(f"    Folded implies:  {implied_depth_fold*1e6:.1f} ppm  (err vs truth: {_pct_err(implied_depth_fold, t_depth_ppm/1e6)})")
        log.info(f"    BLS depth gives: {bls.transit_depth*1e6:.1f} ppm  (err vs truth: {_pct_err(bls.transit_depth, t_depth_ppm/1e6)})")

        if implied_depth_fold < (t_depth_ppm / 1e6) * 0.7:
            log.info(f"    *** FOLDED FLUX DEPTH ALREADY SUPPRESSED — problem is UPSTREAM (SavGol detrending or data) ***")
        elif bls.transit_depth < (t_depth_ppm / 1e6) * 0.7 and implied_depth_fold >= (t_depth_ppm / 1e6) * 0.7:
            log.info(f"    *** FOLDED FLUX LOOKS OK but BLS depth is low — problem in depth CALCULATION from folded flux ***")

    # ══ §3 MCMC INITIALIZATION ═════════════════════════════════════════════════
    log.info("\n  [§3 MCMC INITIALIZATION]")

    rp_init = float(np.sqrt(max(bls.transit_depth, 1e-6)))
    p0_center = np.array([0.0, rp_init, 0.2])
    scatter   = np.array([1e-3, rp_init * 0.5, 0.3])
    n_walkers = conf.mcmc.n_walkers

    rng = np.random.default_rng(seed=0)
    p0 = p0_center + scatter * rng.standard_normal((n_walkers, 3))
    p0[:, 1] = np.clip(p0[:, 1], 1e-4, 0.29)
    p0[:, 2] = np.clip(p0[:, 2], 0.0,  0.95)

    for pi, pname in enumerate(["t0", "rp", "b"]):
        center_val = p0_center[pi]
        col = p0[:, pi]
        log.info(f"    {pname}: center={center_val:.5f}  spread min={col.min():.5f}  max={col.max():.5f}  std={col.std():.5f}")

    if truth is not None:
        t_rp = truth["rp_over_Rstar"]
        t_b  = truth["ImpactParam_b"]
        in_range_rp = p0[:, 1].min() <= t_rp <= p0[:, 1].max()
        log.info(f"\n    Truth rp={t_rp:.5f}  b={t_b:.4f}")
        log.info(f"    rp truth in walker range [{p0[:,1].min():.5f}, {p0[:,1].max():.5f}]: "
                 f"{'YES' if in_range_rp else '*** NO — walkers miss the truth entirely ***'}")
        if not in_range_rp:
            log.info(f"    *** INITIALIZATION PROBLEM: walkers start far from truth rp ***")

    # ══ §4 MCMC OUTPUT ═════════════════════════════════════════════════════════
    log.info(f"\n  [§4 MCMC OUTPUT]  (n_walkers={conf.mcmc.n_walkers}, "
             f"n_steps={conf.mcmc.n_steps}, n_burnin={conf.mcmc.n_burnin})")

    mcmc = run_mcmc(
        lc, bls,
        n_walkers=conf.mcmc.n_walkers,
        n_steps=conf.mcmc.n_steps,
        n_burnin=conf.mcmc.n_burnin,
        u1=ld.u1,
        u2=ld.u2,
        stellar_mass=stellar.mass,
        stellar_radius=stellar.radius,
    )

    log.info(f"    rp:   {mcmc.rp_med:.5f}  -{mcmc.rp_err[0]:.5f}  +{mcmc.rp_err[1]:.5f}")
    log.info(f"    b:    {mcmc.b_med:.4f}   -{mcmc.b_err[0]:.4f}   +{mcmc.b_err[1]:.4f}")
    log.info(f"    t0:   {mcmc.t0_med:.5f}  -{mcmc.t0_err[0]:.5f}  +{mcmc.t0_err[1]:.5f}")
    log.info(f"    depth (rp^2): {mcmc.depth_med*1e6:.1f} ppm")
    log.info(f"    Acceptance fraction: {mcmc.acceptance_fraction:.4f}"
             f"{_flag(mcmc.acceptance_fraction > 0.5, 'HIGH — walkers may be stuck / not exploring')}"
             f"{_flag(mcmc.acceptance_fraction < 0.15, 'LOW — possible stuck chain / poor initialization')}")
    log.info(f"    Converged: {mcmc.converged}")
    for note in mcmc.convergence_notes:
        if note:
            log.info(f"    Convergence note: {note}")

    # ══ §5 DERIVED VS PUBLISHED ════════════════════════════════════════════════
    log.info("\n  [§5 DERIVED VS PUBLISHED]")

    r_earth_computed = mcmc.rp_med * stellar.radius * R_SUN_TO_EARTH
    r_earth_lo = (mcmc.rp_med - mcmc.rp_err[0]) * stellar.radius * R_SUN_TO_EARTH
    r_earth_hi = (mcmc.rp_med + mcmc.rp_err[1]) * stellar.radius * R_SUN_TO_EARTH

    log.info(f"    Stellar radius used: {stellar.radius:.4f} R_sun")
    log.info(f"    rp_mcmc × R_star × 109.076 = {r_earth_computed:.3f} R_earth  "
             f"[{r_earth_lo:.3f}, {r_earth_hi:.3f}]")
    log.info(f"    Period (BLS):  {bls.best_period:.5f} d")

    if truth is not None:
        t_r_earth = truth["Radius_Rearth"]
        t_period  = truth["Period_d"]
        t_rp      = truth["rp_over_Rstar"]
        t_b       = truth["ImpactParam_b"]
        t_depth   = truth["Depth_ppm"] / 1e6

        log.info(f"\n    Published ({label}):")
        log.info(f"      R_planet: {t_r_earth:.3f} R_earth  ->  computed: {r_earth_computed:.3f}  err: {_pct_err(r_earth_computed, t_r_earth)}")
        log.info(f"      Period:   {t_period:.5f} d          ->  BLS:      {bls.best_period:.5f}  err: {_pct_err(bls.best_period, t_period)}")
        log.info(f"      rp/R*:    {t_rp:.5f}               ->  MCMC:     {mcmc.rp_med:.5f}  err: {_pct_err(mcmc.rp_med, t_rp)}")
        log.info(f"      b:        {t_b:.4f}                ->  MCMC:     {mcmc.b_med:.4f}  err: {_pct_err(mcmc.b_med, t_b)}")
    else:
        log.info(f"    R_planet (computed): {r_earth_computed:.3f} R_earth  (no truth to compare)")

    # ══ §6 KEY 3-WAY DEPTH DIAGNOSTIC ═════════════════════════════════════════
    log.info("\n  [§6 KEY DEPTH DIAGNOSTIC — where is depth lost?]")

    rp_mcmc_sq = mcmc.rp_med ** 2

    log.info(f"    A) BLS transit_depth:           {bls.transit_depth*1e6:8.1f} ppm")
    log.info(f"    B) 1 - min(folded flux):        {(1.0 - transit_min)*1e6:8.1f} ppm  (naive, ignores baseline offset)")
    log.info(f"    C) baseline_med - min(fold):    {implied_depth_fold*1e6:8.1f} ppm  (corrected for baseline)")
    log.info(f"    D) rp_mcmc^2:                   {rp_mcmc_sq*1e6:8.1f} ppm")

    if truth is not None:
        t_depth_ppm = truth["Depth_ppm"]
        log.info(f"    E) Truth depth:                 {t_depth_ppm:8.1f} ppm")
        log.info("")

        frac_a = _pct_err(bls.transit_depth,   t_depth_ppm / 1e6)
        frac_c = _pct_err(implied_depth_fold,   t_depth_ppm / 1e6)
        frac_d = _pct_err(rp_mcmc_sq,           t_depth_ppm / 1e6)

        log.info(f"    Error vs truth:  A={frac_a}  C={frac_c}  D={frac_d}")
        log.info("")

        # Verdict
        truth_depth = t_depth_ppm / 1e6
        c_ok  = not np.isnan(implied_depth_fold) and implied_depth_fold >= truth_depth * 0.75
        a_ok  = bls.transit_depth >= truth_depth * 0.75
        d_ok  = rp_mcmc_sq >= truth_depth * 0.75

        if not c_ok:
            log.info("    VERDICT: C is low -> depth loss is UPSTREAM of BLS (SavGol detrending / data preprocessing)")
        elif not a_ok and c_ok:
            log.info("    VERDICT: C is OK but A is low -> depth loss in BLS depth CALCULATION from folded flux")
        elif not d_ok and a_ok:
            log.info("    VERDICT: A is OK but D is low -> MCMC is not finding the true depth (initialization / prior / convergence)")
        else:
            log.info("    VERDICT: All depths broadly consistent with truth (within 25%)")

        # Check A vs C agreement
        if not np.isnan(implied_depth_fold) and implied_depth_fold > 1e-9:
            ac_diff = abs(bls.transit_depth - implied_depth_fold) / implied_depth_fold
            if ac_diff > 0.05:
                log.info(f"    NOTE: A and C disagree by {ac_diff*100:.1f}% — BLS depth calc inconsistency")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    truth_df = pd.read_csv(TRUTH_CSV)

    for kepler_num in TARGETS:
        try:
            run_diagnostic(kepler_num, truth_df)
        except Exception as e:
            log.error(f"\n*** DIAGNOSTIC FAILED for Kepler-{kepler_num}: {e} ***\n")
            import traceback
            traceback.print_exc()

    _header("DIAGNOSTIC COMPLETE")
