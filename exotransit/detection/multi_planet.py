import lightkurve as lk
import numpy as np
from pathlib import Path
from tqdm import tqdm

from exotransit.detection.bls import BLSResult, run_bls
from exotransit.pipeline.light_curves import LightCurveData
import logging

logger = logging.getLogger(__name__)


def find_all_planets(
    lc: LightCurveData,
    max_planets: int = 5,
    min_period: float = 2.0,
    max_period: float = 30.0,
    max_period_grid_points: int = 100_000,
    debug_dir: str | None = None,
) -> list[BLSResult]:
    # Use a copy so we don't mutate the original input
    lc_lk = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
    unique_results = []

    if debug_dir is not None:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    with tqdm(total=max_planets, desc="Searching for planets", unit="planet") as pbar:
        for i in range(max_planets):
            pbar.set_description(f"Searching for planet {i + 1}")

            # current_lc now uses the modified (filled) lc_lk flux
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

            result = run_bls(
                current_lc,
                min_period=min_period,
                max_period=max_period,
                max_period_grid_points=max_period_grid_points,
            )

            if not result.is_reliable:
                pbar.set_description(f"Stopping at {len(unique_results)} planets")
                print(result.reliability_flags)
                pbar.update(max_planets - i)
                break

            # Create the mask
            mask = lc_lk.create_transit_mask(
                period=result.best_period,
                transit_time=result.best_t0,
                duration=result.best_duration * 3.0
            )

            # Check for ghosts/duplicates
            is_dup = False
            for existing in unique_results:
                period_diff = abs(result.best_period - existing.best_period) / existing.best_period
                if period_diff < 0.05:
                    logger.info(f"Skipping duplicate/TTV ghost at {result.best_period:.4f}d")
                    is_dup = True
                    break

            # Save plot before we overwrite the flux
            if debug_dir is not None:
                _save_mask_debug_plot(
                    time=np.asarray(lc_lk.time.value),
                    flux=np.asarray(lc_lk.flux.value),
                    mask=np.asarray(mask),
                    result=result,
                    planet_number=i + 1,
                    debug_dir=debug_dir,
                    target_name=lc.target_name,
                )

            # MEDIAN FILL: Flatten the transits instead of deleting them
            # This keeps the time array continuous and the BLS algorithm happy
            lc_lk.flux.value[mask] = np.nanmedian(lc_lk.flux.value)

            if is_dup:
                continue

            # Store and update UI
            unique_results.append(result)
            pbar.set_postfix(period=f"{result.best_period:.2f}d", sde=f"{result.sde:.1f}")
            pbar.update(1)
            logger.info(f"Planet added: {result.best_period:.4f}d. Masked and continuing...")

    return unique_results

def _save_mask_debug_plot(
    time: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    result: BLSResult,
    planet_number: int,
    debug_dir: str,
    target_name: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = ~mask
    n_masked = mask.sum()
    n_kept = kept.sum()

    # Phase-fold the full light curve at the detected period to show what was found
    phase = ((time - result.best_t0) % result.best_period) / result.best_period
    phase[phase > 0.5] -= 1.0  # center on 0

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"{target_name} — Planet {planet_number} mask  "
        f"(P={result.best_period:.4f}d, dur={result.best_duration*24:.2f}h, "
        f"depth={result.transit_depth:.5f}, SDE={result.sde:.1f})",
        fontsize=11,
    )

    # ── Panel 1: full light curve, masked points highlighted ─────────────────
    ax = axes[0]
    ax.scatter(time[kept],  flux[kept],  s=1.5, c="#94a3b8", alpha=0.5, label=f"kept ({n_kept})")
    ax.scatter(time[mask],  flux[mask],  s=6,   c="#f43f5e", alpha=0.9, label=f"masked ({n_masked})")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title("Light curve — masked points in red")
    ax.legend(markerscale=3, fontsize=8)

    # ── Panel 2: phase fold at detected period, masked points highlighted ────
    ax = axes[1]
    ax.scatter(phase[kept], flux[kept], s=1.5, c="#94a3b8", alpha=0.4, label="kept")
    ax.scatter(phase[mask], flux[mask], s=8,   c="#f43f5e", alpha=0.9, label="masked")
    half_dur_phase = (result.best_duration * 1.5) / result.best_period
    ax.axvspan(-half_dur_phase, half_dur_phase, color="#f43f5e", alpha=0.1, label="mask window")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux")
    ax.set_title(f"Phase fold at P={result.best_period:.4f}d — is the dip centered and fully masked?")
    ax.legend(markerscale=3, fontsize=8)

    # ── Panel 3: what BLS will see next — remaining light curve ──────────────
    ax = axes[2]
    ax.scatter(time[kept], flux[kept], s=1.5, c="#38bdf8", alpha=0.5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title(f"Remaining light curve ({n_kept} points) — input to next BLS iteration")

    fig.tight_layout()
    out_path = Path(debug_dir) / f"{target_name}_planet_{planet_number:02d}_mask.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved mask debug plot: {out_path}")
