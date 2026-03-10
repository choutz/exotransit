"""
exotransit/config.py

Pipeline configuration — single source of truth for all tunable parameters.

Two profiles:
  MEDIUM — for Streamlit app, balances speed with accuracy within
           Streamlit Community Cloud limits (1GB RAM, 1 vCPU)
  FULL   — for local testing/validation, publication-quality results

Import and use like:
    from exotransit.config import MEDIUM, FULL
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BLSConfig:
    min_period: float            # days
    max_period: float            # days
    max_period_grid_points: int


@dataclass(frozen=True)
class MCMCConfig:
    n_walkers: int
    n_steps: int
    n_burnin: int


@dataclass(frozen=True)
class PipelineConfig:
    bls: BLSConfig
    mcmc: MCMCConfig
    max_planets: int
    max_quarters: int

LOW = PipelineConfig(
    bls=BLSConfig(
        max_period_grid_points=10_000,
        min_period=2,
        max_period=100
    ),
    mcmc=MCMCConfig(
        n_walkers=32,
        n_steps=4000,
        n_burnin=500,
    ),
    max_planets=5,
    max_quarters=4,
)

MEDIUM = PipelineConfig(
    bls=BLSConfig(
        max_period_grid_points=100_000,
        min_period=2,
        max_period=100
    ),
    mcmc=MCMCConfig(
        n_walkers=32,
        n_steps=4000,
        n_burnin=500,
    ),
    max_planets=6,
    max_quarters=12,
)

FULL = PipelineConfig(
    bls=BLSConfig(
        max_period_grid_points=200_000,
        min_period=1,
        max_period=100
    ),
    mcmc=MCMCConfig(
        n_walkers=32,
        n_steps=6000,
        n_burnin=500,
    ),
    max_planets=6,
    max_quarters=17,
)