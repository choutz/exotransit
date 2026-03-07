"""
tests/pipeline_diagnostic.py

Diagnostic pipeline runner: BLS + MCMC with verbose per-step logging,
compared against published values from examples.csv.

Run from the repo root:
    python -m tests.pipeline_diagnostic

Output goes entirely to the console. No assertions — this is for diagnosis, not CI.
"""

import logging
import sys
import numpy as np
import pandas as pd
import lightkurve as lk
import emcee
from pathlib import Path

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FULL, MEDIUM
from tests.helpers import get_light_curve
from exotransit.detection.bls import run_bls
from exotransit.mcmc.fit_mcmc import run_mcmc
from exotransit.mcmc.helpers import _transit_model, _log_likelihood, _log_probability
from exotransit.pipeline.light_curves import LightCurveData
from exotransit.physics.stars import query_stellar_params
from exotransit.physics.limb_darkening import get_limb_darkening

# ── configuration ──────────────────────────────────────────────────────────────
TARGETS = [7, 144, 9, 15, 11]
conf = MEDIUM
TRUTH_CSV = Path(__file__).parent / "examples.csv"
PERIOD_MATCH_TOL = 0.20   # 20% tolerance for matching detected period to truth
R_SUN_TO_EARTH   = 109.076
AU_TO_RSUN       = 215.032

# ── logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # suppress noisy library logs
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("diagnostic")
log.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)
log.propagate = False


# ── formatting helpers ─────────────────────────────────────────────────────────

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
    if truth_rows.empty:
        return None
    diffs = np.abs(truth_rows["Period_d"].values - found_period) / truth_rows["Period_d"].values
    idx = np.argmin(diffs)
    return truth_rows.iloc[idx] if diffs[idx] <= PERIOD_MATCH_TOL else None

def _pct_err(measured, truth):
    if truth == 0 or (isinstance(truth, float) and np.isnan(truth)):
        return "n/a"
    err = (measured - truth) / abs(truth)
    sign = "+" if err >= 0 else ""
    return f"{sign}{err * 100:.1f}%"

def _flag(condition, label):
    return f"  *** {label}" if condition else ""


# ── per-target entry point ─────────────────────────────────────────────────────

def run_diagnostic(kepler_num: int, truth_df: pd.DataFrame):
    target = f"Kepler-{kepler_num}"
    truth_rows = truth_df[truth_df["Kepler_#"] == float(kepler_num)].sort_values("Period_d")

    _header(f"TARGET: {target}  ({len(truth_rows)} confirmed planets in truth)")

    if truth_rows.empty:
        log.warning(f"  No truth rows found for Kepler-{kepler_num} in examples.csv")

    # ── light curve ────────────────────────────────────────────────────────────
    log.info(f"\n  Fetching light curve (max_quarters={conf.max_quarters})...")
    lc = get_light_curve(target, mission="Kepler", max_quarters=conf.max_quarters)
    baseline = lc.time[-1] - lc.time[0]
    cadence_min = float(np.median(np.diff(lc.time))) * 24 * 60

    log.info(f"  Points:     {len(lc.time):,}")
    log.info(f"  Baseline:   {baseline:.1f} days")
    log.info(f"  Cadence:    {cadence_min:.1f} min")
    log.info(f"  Flux min:   {lc.flux.min():.6f}   max: {lc.flux.max():.6f}")
    log.info(f"  Median err: {np.median(lc.flux_err):.6f}")

    # ── stellar params ─────────────────────────────────────────────────────────
    log.info(f"\n  Fetching stellar params from NASA Exoplanet Archive...")
    stellar = query_stellar_params(target)
    ld = get_limb_darkening(teff=stellar.teff, logg=stellar.logg, metallicity=stellar.metallicity)

    log.info(f"  R_star: {stellar.radius:.4f} R_sun  |  M_star: {stellar.mass:.4f} M_sun  |  Teff: {stellar.teff:.0f} K")
    log.info(f"  u1={ld.u1:.4f}  u2={ld.u2:.4f}")

    if not truth_rows.empty:
        t_rstar = truth_rows["Rstar_Rsun"].iloc[0]
        log.info(f"  Truth R_star (CSV): {t_rstar:.4f} R_sun  (archive returned {stellar.radius:.4f}, diff={_pct_err(stellar.radius, t_rstar)})")

    # ── BLS masking loop ───────────────────────────────────────────────────────
    log.info(f"\n  Running BLS (max_planets={conf.max_planets})...")
    lc_work = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
    found_bls = []

    for planet_i in range(conf.max_planets):
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

        is_dup = any(
            abs(bls.best_period - ex.best_period) / ex.best_period < 0.05
            for ex in found_bls
        )

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

    # ── summary ────────────────────────────────────────────────────────────────
    _section(f"{target} — DETECTION SUMMARY")
    log.info(f"  Found {len(found_bls)} planet(s) | Truth has {len(truth_rows)} confirmed")

    for _, trow in truth_rows.iterrows():
        matched = any(
            abs(b.best_period - trow["Period_d"]) / trow["Period_d"] <= PERIOD_MATCH_TOL
            for b in found_bls
        )
        status = "FOUND" if matched else "*** MISSED ***"
        log.info(f"  {status:15s}  {trow['Planet']:20s}  P={trow['Period_d']:.4f}d  R={trow['Radius_Rearth']:.2f} R_earth")

    for b in found_bls:
        if _match_truth(b.best_period, truth_rows) is None:
            log.info(f"  *** FALSE POSITIVE ***  P={b.best_period:.4f}d  (no truth match within {PERIOD_MATCH_TOL*100:.0f}%)")


# ── per-planet diagnostic ──────────────────────────────────────────────────────

def _diagnose_planet(planet_i, bls, stellar, ld, truth_rows, lc):

    truth = _match_truth(bls.best_period, truth_rows)
    label = truth["Planet"] if truth is not None else "NO TRUTH MATCH"

    _section(f"PLANET {planet_i + 1} — {label}  (P={bls.best_period:.4f}d)")

    # shared geometry used in multiple sections
    cadence_days  = float(np.median(np.diff(lc.time)))
    exp_time_days = cadence_days
    window        = max(bls.best_duration * 3.0, 0.3)
    transit_mask_mcmc = np.abs(bls.folded_time) < window
    time_mcmc     = bls.folded_time[transit_mask_mcmc]
    flux_mcmc     = bls.folded_flux[transit_mask_mcmc]
    flux_err_mcmc = bls.folded_flux_err[transit_mask_mcmc]

    # orbital geometry
    a_au         = (stellar.mass * (bls.best_period / 365.25) ** 2) ** (1 / 3)
    a_rsun       = a_au * AU_TO_RSUN
    a_over_rstar = a_rsun / stellar.radius   # = batman p.a

    # ══ §1 BLS OUTPUT ══════════════════════════════════════════════════════════
    log.info("  [§1 BLS OUTPUT]")
    log.info(f"    Period:        {bls.best_period:.5f} d")
    log.info(f"    Duration:      {bls.best_duration * 24:.3f} h")
    log.info(f"    Depth:         {bls.transit_depth * 1e6:.1f} ppm  ({bls.transit_depth:.6f})")
    log.info(f"    SDE:           {bls.sde:.2f}")
    log.info(f"    SNR:           {bls.snr:.2f}")

    baseline          = lc.time[-1] - lc.time[0]
    expected_transits = baseline / bls.best_period
    expected_pts_per  = bls.best_duration / cadence_days
    in_bls_mask       = np.abs(bls.folded_time) < bls.best_duration / 2
    n_in_transit      = int(in_bls_mask.sum())

    log.info(f"    In-transit pts: {n_in_transit}  (expected ~{expected_pts_per:.1f}/transit × {expected_transits:.1f} transits)")
    log.info(f"    Expected transits in baseline: {expected_transits:.2f}")
    log.info(f"    Aliases:        {bls.aliases if bls.aliases else 'none'}")

    if truth is not None:
        t_period = truth["Period_d"]
        t_depth  = truth["Depth_ppm"] / 1e6
        t_dur_h  = truth["Duration_h"]
        log.info(f"\n    vs truth [{label}]:")
        log.info(f"      Period:   {bls.best_period:.5f} d   truth={t_period:.5f} d   err={_pct_err(bls.best_period, t_period)}")
        log.info(f"      Depth:    {bls.transit_depth*1e6:.1f} ppm  truth={truth['Depth_ppm']:.1f} ppm  err={_pct_err(bls.transit_depth, t_depth)}")
        log.info(f"      Duration: {bls.best_duration*24:.3f} h    truth={t_dur_h:.3f} h    err={_pct_err(bls.best_duration*24, t_dur_h)}")
    else:
        log.info("    *** NO TRUTH MATCH within tolerance — possible false positive ***")

    # ══ §2 FOLDED FLUX ANALYSIS ════════════════════════════════════════════════
    log.info("\n  [§2 FOLDED FLUX ANALYSIS]")

    folded_t = bls.folded_time
    folded_f = bls.folded_flux

    half_dur = max(bls.best_duration / 2, 1.5 * cadence_days)
    in_win   = np.abs(folded_t) < half_dur
    out_win  = ~in_win

    n_in  = int(in_win.sum())
    n_out = int(out_win.sum())

    baseline_median  = float(np.median(folded_f[out_win])) if n_out > 0 else 1.0
    baseline_std     = float(np.std(folded_f[out_win]))    if n_out > 0 else np.nan
    transit_min      = float(np.min(folded_f[in_win]))     if n_in  > 0 else np.nan
    implied_depth    = baseline_median - transit_min        if n_in  > 0 else np.nan

    rp_init_from_bls  = float(np.sqrt(max(bls.transit_depth, 1e-6)))
    rp_init_from_fold = float(np.sqrt(max(implied_depth, 1e-6))) if n_in > 0 else np.nan

    depth_disagree = (
        n_in > 0 and not np.isnan(implied_depth)
        and abs(bls.transit_depth - implied_depth) / max(implied_depth, 1e-9) > 0.05
    )

    log.info(f"    Transit window:  ±{half_dur*24:.2f} h  ({n_in} in-transit, {n_out} out-of-transit pts)")
    log.info(f"    Baseline median: {baseline_median:.6f}  (should be ~1.0{_flag(abs(baseline_median - 1.0) > 0.002, 'OFFSET > 200ppm')})")
    log.info(f"    Baseline std:    {baseline_std*1e6:.1f} ppm  (noise level)")
    log.info(f"    Transit minimum: {transit_min:.6f}")
    log.info(f"    Implied depth (baseline - min): {implied_depth*1e6:.1f} ppm")
    log.info(f"    bls.transit_depth:              {bls.transit_depth*1e6:.1f} ppm{_flag(depth_disagree, 'DISAGREE >5%')}")

    if depth_disagree:
        log.info(f"    *** DEPTH MISMATCH: folded says {implied_depth*1e6:.1f} ppm, BLS says {bls.transit_depth*1e6:.1f} ppm")
        log.info(f"        -> Problem is in depth computation from folded flux, not upstream in detrending")

    log.info(f"\n    rp_init (from bls.transit_depth): {rp_init_from_bls:.5f}")
    log.info(f"    rp_init (from folded flux min):   {rp_init_from_fold:.5f}"
             f"{_flag(abs(rp_init_from_bls - rp_init_from_fold) > 0.005, 'DIFFER > 0.005 — init wrong from the start')}")

    if truth is not None:
        t_rp        = truth["rp_over_Rstar"]
        t_depth_ppm = truth["Depth_ppm"]
        log.info(f"\n    Truth rp/R*:    {t_rp:.5f}  (rp^2 = {t_rp**2*1e6:.1f} ppm)")
        log.info(f"    Truth depth:    {t_depth_ppm:.1f} ppm")
        log.info(f"    Folded implies: {implied_depth*1e6:.1f} ppm  (err vs truth: {_pct_err(implied_depth, t_depth_ppm/1e6)})")
        log.info(f"    BLS depth:      {bls.transit_depth*1e6:.1f} ppm  (err vs truth: {_pct_err(bls.transit_depth, t_depth_ppm/1e6)})")

        if implied_depth < (t_depth_ppm / 1e6) * 0.7:
            log.info(f"    *** FOLDED FLUX ALREADY SUPPRESSED — problem UPSTREAM (SavGol or data) ***")
        elif bls.transit_depth < (t_depth_ppm / 1e6) * 0.7 and implied_depth >= (t_depth_ppm / 1e6) * 0.7:
            log.info(f"    *** FOLDED FLUX OK BUT BLS DEPTH LOW — problem in depth calculation ***")

    # ══ §3 MCMC INITIALIZATION ═════════════════════════════════════════════════
    log.info("\n  [§3 MCMC INITIALIZATION]")

    rp_init   = float(np.sqrt(max(bls.transit_depth, 1e-6)))
    p0_center = np.array([0.0, rp_init, 0.2])
    scatter   = np.array([1e-3, rp_init * 0.5, 0.3])
    n_walkers = conf.mcmc.n_walkers

    rng = np.random.default_rng(seed=0)
    p0_display = p0_center + scatter * rng.standard_normal((n_walkers, 3))
    p0_display[:, 1] = np.clip(p0_display[:, 1], 1e-4, 0.29)
    p0_display[:, 2] = np.clip(p0_display[:, 2], 0.0,  0.95)

    for pi, pname in enumerate(["t0", "rp", "b"]):
        col = p0_display[:, pi]
        log.info(f"    {pname}: center={p0_center[pi]:.5f}  "
                 f"min={col.min():.5f}  max={col.max():.5f}  std={col.std():.5f}")

    # Orbital geometry
    b_max_physical   = a_over_rstar - 1.0 - rp_init
    frac_b_gt_a      = float(np.mean(p0_display[:, 2] >= a_over_rstar))
    frac_b_gt_1      = float(np.mean(p0_display[:, 2] >= 1.0))

    log.info(f"\n    Orbital geometry:")
    log.info(f"      a/R_star (batman p.a):         {a_over_rstar:.4f}")
    log.info(f"      b_max for full transit:         {b_max_physical:.4f}  (= a/R* - 1 - rp)")
    log.info(f"      Walkers with b >= a/R* (geom. invalid): {frac_b_gt_a:.1%}")
    log.info(f"      Walkers with b >= 1.0 (grazing):        {frac_b_gt_1:.1%}")

    if truth is not None:
        t_rp_val = truth["rp_over_Rstar"]
        t_b_val  = truth["ImpactParam_b"]
        in_range = p0_display[:, 1].min() <= t_rp_val <= p0_display[:, 1].max()
        log.info(f"\n    Truth rp={t_rp_val:.5f}  b={t_b_val:.4f}")
        log.info(f"    rp truth in walker init range "
                 f"[{p0_display[:,1].min():.5f}, {p0_display[:,1].max():.5f}]: "
                 f"{'yes' if in_range else '*** NO — walkers start below true rp ***'}")

        # Batman at truth params to verify stellar params are consistent
        try:
            truth_params  = np.array([0.0, t_rp_val, t_b_val])
            truth_model   = _transit_model(
                truth_params, time_mcmc, bls.best_period,
                ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
            )
            batman_depth  = 1.0 - truth_model.min()
            t_depth_ppm   = truth["Depth_ppm"]
            log.info(f"\n    Batman at truth params [t0=0, rp={t_rp_val:.5f}, b={t_b_val:.4f}]:")
            log.info(f"      model_flux.min()  = {truth_model.min():.6f}")
            log.info(f"      implied depth     = {batman_depth*1e6:.1f} ppm")
            log.info(f"      truth depth (CSV) = {t_depth_ppm:.1f} ppm")
            log.info(f"      batman vs truth:    {_pct_err(batman_depth, t_depth_ppm/1e6)}")
            if batman_depth < (t_depth_ppm / 1e6) * 0.5:
                log.info(f"      *** BATMAN CANNOT REPRODUCE TRUTH DEPTH — stellar params or MCMC window wrong ***")
        except Exception as e:
            log.warning(f"    Batman at truth failed: {e}")

    # ══ §3b BURN-IN DIAGNOSTIC ═════════════════════════════════════════════════
    log.info(f"\n  [§3b BURN-IN STATE]  (running {conf.mcmc.n_burnin} burn-in steps...)")

    p0_burned   = None
    mid_model   = None
    try:
        p0_fresh = p0_center + scatter * np.random.randn(n_walkers, 3)
        p0_fresh[:, 1] = np.clip(p0_fresh[:, 1], 1e-4, 0.29)
        p0_fresh[:, 2] = np.clip(p0_fresh[:, 2], 0.0,  0.95)

        sampler_diag = emcee.EnsembleSampler(
            n_walkers, 3, _log_probability,
            args=(time_mcmc, flux_mcmc, flux_err_mcmc, bls.best_period,
                  ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days, bls.snr),
        )
        p0_burned, _, _ = sampler_diag.run_mcmc(p0_fresh, conf.mcmc.n_burnin, progress=True)

        median_walker = np.median(p0_burned, axis=0)
        t0_mid, rp_mid, b_mid = median_walker

        mid_model   = _transit_model(
            median_walker, time_mcmc, bls.best_period,
            ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
        )
        data_min    = float(flux_mcmc.min())
        mid_min     = float(mid_model.min())

        log.info(f"    Median walker after burn-in:  t0={t0_mid:.5f}  rp={rp_mid:.5f}  b={b_mid:.4f}")
        log.info(f"    model_flux.min() at median walker: {mid_min:.6f}  "
                 f"(depth={1-mid_min:.6f} = {(1-mid_min)*1e6:.1f} ppm)")
        log.info(f"    data_flux.min() in MCMC window:    {data_min:.6f}  "
                 f"(depth={1-data_min:.6f} = {(1-data_min)*1e6:.1f} ppm)")

        if truth is not None:
            t_rp_val = truth["rp_over_Rstar"]
            t_b_val  = truth["ImpactParam_b"]

            lnL_truth_burnin = _log_likelihood(
                np.array([0.0, t_rp_val, t_b_val]),
                time_mcmc, flux_mcmc, flux_err_mcmc, bls.best_period,
                ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
            )
            lnL_burned = _log_likelihood(
                median_walker,
                time_mcmc, flux_mcmc, flux_err_mcmc, bls.best_period,
                ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
            )

            log.info(f"\n    lnL at truth values:          {lnL_truth_burnin:.2f}")
            log.info(f"    lnL at burn-in median walker: {lnL_burned:.2f}")
            delta = lnL_truth_burnin - lnL_burned
            if lnL_truth_burnin < lnL_burned:
                log.info(f"    *** lnL(truth) < lnL(burned median) by {-delta:.1f} "
                         f"— LIKELIHOOD PREFERS WRONG ANSWER (data problem) ***")
            else:
                log.info(f"    lnL(truth) > lnL(burned median) by {delta:.1f} "
                         f"— sampler hasn't found the true maximum yet (expected at burn-in stage)")

    except Exception as e:
        log.warning(f"    Burn-in diagnostic failed: {e}")
        import traceback; traceback.print_exc()

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

    log.info(f"    rp:           {mcmc.rp_med:.5f}  -{mcmc.rp_err[0]:.5f}  +{mcmc.rp_err[1]:.5f}")
    log.info(f"    b:            {mcmc.b_med:.4f}   -{mcmc.b_err[0]:.4f}   +{mcmc.b_err[1]:.4f}")
    log.info(f"    t0:           {mcmc.t0_med:.5f}  -{mcmc.t0_err[0]:.5f}  +{mcmc.t0_err[1]:.5f}")
    log.info(f"    depth (rp^2): {mcmc.depth_med*1e6:.1f} ppm")
    log.info(f"    Acceptance fraction: {mcmc.acceptance_fraction:.4f}"
             f"{_flag(mcmc.acceptance_fraction > 0.5, 'HIGH — walkers may be stuck / not exploring')}"
             f"{_flag(mcmc.acceptance_fraction < 0.15, 'LOW — possible stuck chain / poor init')}")
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
    log.info(f"    rp_mcmc × R_star × 109.076 = {r_earth_computed:.3f} R_earth  [{r_earth_lo:.3f}, {r_earth_hi:.3f}]")
    log.info(f"    Period (BLS):  {bls.best_period:.5f} d")

    if truth is not None:
        t_r_earth = truth["Radius_Rearth"]
        t_period  = truth["Period_d"]
        t_rp      = truth["rp_over_Rstar"]
        t_b       = truth["ImpactParam_b"]
        log.info(f"\n    Published ({label}):")
        log.info(f"      R_planet: {t_r_earth:.3f} R_earth  ->  computed: {r_earth_computed:.3f}  err: {_pct_err(r_earth_computed, t_r_earth)}")
        log.info(f"      Period:   {t_period:.5f} d          ->  BLS:      {bls.best_period:.5f}  err: {_pct_err(bls.best_period, t_period)}")
        log.info(f"      rp/R*:    {t_rp:.5f}               ->  MCMC:     {mcmc.rp_med:.5f}  err: {_pct_err(mcmc.rp_med, t_rp)}")
        log.info(f"      b:        {t_b:.4f}                ->  MCMC:     {mcmc.b_med:.4f}  err: {_pct_err(mcmc.b_med, t_b)}")
    else:
        log.info(f"    R_planet (computed): {r_earth_computed:.3f} R_earth  (no truth to compare)")

    # ══ §6 DEPTH DIAGNOSTIC + POST-CONVERGENCE BATMAN ══════════════════════════
    log.info("\n  [§6 KEY DEPTH DIAGNOSTIC — where is depth lost?]")

    rp_mcmc_sq = mcmc.rp_med ** 2

    log.info(f"    A) BLS transit_depth:           {bls.transit_depth*1e6:8.1f} ppm")
    log.info(f"    B) 1 - min(folded flux):        {(1.0 - transit_min)*1e6:8.1f} ppm  (naive, ignores baseline offset)")
    log.info(f"    C) baseline_med - min(fold):    {implied_depth*1e6:8.1f} ppm  (corrected)")
    log.info(f"    D) rp_mcmc^2:                   {rp_mcmc_sq*1e6:8.1f} ppm")

    if truth is not None:
        t_depth_ppm = truth["Depth_ppm"]
        log.info(f"    E) Truth depth:                 {t_depth_ppm:8.1f} ppm")
        log.info(f"    Error vs truth:  A={_pct_err(bls.transit_depth, t_depth_ppm/1e6)}  "
                 f"C={_pct_err(implied_depth, t_depth_ppm/1e6)}  "
                 f"D={_pct_err(rp_mcmc_sq, t_depth_ppm/1e6)}")
        log.info("")

        truth_depth = t_depth_ppm / 1e6
        c_ok = not np.isnan(implied_depth) and implied_depth >= truth_depth * 0.75
        a_ok = bls.transit_depth >= truth_depth * 0.75
        d_ok = rp_mcmc_sq >= truth_depth * 0.75

        if not c_ok:
            log.info("    VERDICT: C low  -> depth loss UPSTREAM of BLS (SavGol detrending)")
        elif not a_ok:
            log.info("    VERDICT: C ok, A low -> depth loss in BLS depth calculation from folded flux")
        elif not d_ok:
            log.info("    VERDICT: A ok, D low -> MCMC not finding true depth (init / prior / convergence)")
        else:
            log.info("    VERDICT: All depths within 25% of truth")

        if not np.isnan(implied_depth) and implied_depth > 1e-9:
            ac_diff = abs(bls.transit_depth - implied_depth) / implied_depth
            if ac_diff > 0.05:
                log.info(f"    NOTE: A and C disagree by {ac_diff*100:.1f}% — BLS depth calc inconsistency")

    # ── §6b: post-convergence batman check ────────────────────────────────────
    log.info("\n  [§6b POST-CONVERGENCE BATMAN CHECK]")
    try:
        converged_params  = np.array([mcmc.t0_med, mcmc.rp_med, mcmc.b_med])
        converged_model   = _transit_model(
            converged_params, time_mcmc, bls.best_period,
            ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
        )
        batman_min        = float(converged_model.min())
        batman_depth      = 1.0 - batman_min
        rp_sq_check       = mcmc.rp_med ** 2
        rp_sq_vs_batman   = abs(rp_sq_check - batman_depth) / max(rp_sq_check, 1e-9) * 100

        log.info(f"    Converged params: t0={mcmc.t0_med:.5f}  rp={mcmc.rp_med:.5f}  b={mcmc.b_med:.4f}")
        log.info(f"    rp^2:                     {rp_sq_check*1e6:8.1f} ppm")
        log.info(f"    1 - batman_model.min():   {batman_depth*1e6:8.1f} ppm")
        log.info(f"    rp^2 vs batman: diff={rp_sq_vs_batman:.1f}%"
                 f"{_flag(rp_sq_vs_batman > 2.0, 'BATMAN NOT MATCHING rp^2 — geometry issue (grazing?)')}")

        if truth is not None:
            t_rp_val  = truth["rp_over_Rstar"]
            t_b_val   = truth["ImpactParam_b"]
            t_depth_ppm = truth["Depth_ppm"]

            lnL_truth_final = _log_likelihood(
                np.array([0.0, t_rp_val, t_b_val]),
                time_mcmc, flux_mcmc, flux_err_mcmc, bls.best_period,
                ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
            )
            lnL_converged = _log_likelihood(
                converged_params,
                time_mcmc, flux_mcmc, flux_err_mcmc, bls.best_period,
                ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
            )

            log.info(f"\n    lnL at truth values [rp={t_rp_val:.5f}, b={t_b_val:.4f}]: {lnL_truth_final:.2f}")
            log.info(f"    lnL at converged    [rp={mcmc.rp_med:.5f}, b={mcmc.b_med:.4f}]: {lnL_converged:.2f}")
            delta = lnL_truth_final - lnL_converged
            if lnL_truth_final < lnL_converged:
                log.info(f"    *** lnL(truth) < lnL(converged) by {-delta:.1f} "
                         f"— DATA PROBLEM: likelihood genuinely prefers the wrong params ***")
                log.info(f"        The detrended data no longer supports the published solution.")
                log.info(f"        SavGol has altered the flux enough that the true rp/b is a worse fit.")
            else:
                log.info(f"    lnL(truth) > lnL(converged) by {delta:.1f} "
                         f"— SAMPLER PROBLEM: true optimum exists but chain didn't find it")

        # Binned folded flux vs converged model
        log.info(f"\n    Binned folded flux vs batman model  (MCMC window ±{window*24:.1f} h):")
        n_bins     = 15
        bin_edges  = np.linspace(time_mcmc.min(), time_mcmc.max(), n_bins + 1)
        bin_centers= 0.5 * (bin_edges[:-1] + bin_edges[1:])
        model_bins = _transit_model(
            converged_params, bin_centers, bls.best_period,
            ld.u1, ld.u2, stellar.mass, stellar.radius, exp_time_days,
        )
        log.info(f"    {'phase(h)':>8}  {'data_med':>10}  {'model':>10}  {'n_pts':>6}  {'resid(ppm)':>12}")
        for bi in range(n_bins):
            in_bin   = (time_mcmc >= bin_edges[bi]) & (time_mcmc < bin_edges[bi + 1])
            if in_bin.sum() == 0:
                continue
            data_med = float(np.median(flux_mcmc[in_bin]))
            mod_val  = float(model_bins[bi])
            resid    = (data_med - mod_val) * 1e6
            ctr_h    = float(bin_centers[bi]) * 24
            log.info(f"    {ctr_h:>8.2f}  {data_med:>10.6f}  {mod_val:>10.6f}  "
                     f"{in_bin.sum():>6}  {resid:>+10.1f}")

    except Exception as e:
        log.warning(f"    Post-convergence batman check failed: {e}")
        import traceback; traceback.print_exc()


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
