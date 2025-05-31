"""Shared pytest fixtures for sepflows test suite."""

from __future__ import annotations

import numpy as np
import pytest

from sepflows.absorption import AmineAbsorber
from sepflows.asu import CryogenicASU
from sepflows.config import SepConfig
from sepflows.distillation import RigorousColumn, ShortcutColumn
from sepflows.flash import FlashDrum

# ── Config fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def default_config() -> SepConfig:
    """Default solver configuration."""
    return SepConfig()


@pytest.fixture
def tight_config() -> SepConfig:
    """High-precision solver configuration for regression tests."""
    return SepConfig(max_iterations=500, convergence_tol=1e-10)


@pytest.fixture
def loose_config() -> SepConfig:
    """Fast/loose config for speed-sensitive integration tests."""
    return SepConfig(max_iterations=50, convergence_tol=1e-5)


# ── Component definitions ──────────────────────────────────────────────────────


@pytest.fixture
def methanol_water_components() -> list[str]:
    """Binary methanol/water system."""
    return ["methanol", "water"]


@pytest.fixture
def crude_meoh_components() -> list[str]:
    """Full crude methanol oxygenate mixture (CENG50009 basis)."""
    return ["dme", "methyl_formate", "methanol", "ethanol", "water", "1_propanol", "1_butanol"]


@pytest.fixture
def air_components() -> list[str]:
    """Air components for ASU."""
    return ["nitrogen", "oxygen", "argon"]


# ── Feed compositions ──────────────────────────────────────────────────────────


@pytest.fixture
def z_methanol_water() -> np.ndarray:
    """Feed for binary methanol/water column."""
    return np.array([0.60, 0.40])


@pytest.fixture
def z_crude_meoh() -> np.ndarray:
    """CENG50009 crude methanol feed composition (approx, sums to 1)."""
    return np.array([0.02, 0.03, 0.65, 0.04, 0.22, 0.02, 0.02])


@pytest.fixture
def z_air() -> np.ndarray:
    """Dry air composition."""
    return np.array([0.7812, 0.2096, 0.0092])


# ── Pre-built model instances ──────────────────────────────────────────────────


@pytest.fixture
def flash_drum_meoh_water(default_config: SepConfig) -> FlashDrum:
    """Flash drum for methanol/water at 320 K, 2 bar."""
    return FlashDrum(
        components=["methanol", "water"],
        temperature_k=320.0,
        pressure_pa=2.0e5,
        config=default_config,
    )


@pytest.fixture
def flash_drum_crude(crude_meoh_components: list[str], default_config: SepConfig) -> FlashDrum:
    """Flash drum for full crude methanol mixture at 280 K, 20 bar."""
    return FlashDrum(
        components=crude_meoh_components,
        temperature_k=280.0,
        pressure_pa=20.0e5,
        config=default_config,
    )


@pytest.fixture
def shortcut_meoh_water() -> ShortcutColumn:
    """Shortcut column: MeOH (LK) / H₂O (HK), matching CENG50009 B3."""
    return ShortcutColumn(
        light_key="methanol",
        heavy_key="water",
        recovery_lk=0.995,
        recovery_hk=0.995,
        pressure_pa=101_325.0,
        feed_temperature_k=337.0,
    )


@pytest.fixture
def rigorous_meoh_water() -> RigorousColumn:
    """Rigorous column for binary MeOH/H₂O (20 stages, feed at stage 10)."""
    return RigorousColumn(
        components=["methanol", "water"],
        n_stages=20,
        feed_stage=10,
        reflux_ratio=2.72,
        distillate_to_feed=0.55,
        pressure_pa=101_325.0,
        feed_temperature_k=337.0,
    )


@pytest.fixture
def amine_absorber_default() -> AmineAbsorber:
    """Default 90% CO₂ removal unit."""
    return AmineAbsorber(
        removal_target=0.90,
        inlet_co2_mol_h=5_000.0,
        inlet_total_mol_h=50_000.0,
    )


@pytest.fixture
def cryogenic_asu_default() -> CryogenicASU:
    """Default ASU for both N₂ and O₂ production."""
    return CryogenicASU(
        n2_demand_mol_h=10_000.0,
        o2_demand_mol_h=5_000.0,
    )
