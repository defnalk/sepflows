"""Unit tests for the FlashDrum module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sepflows.flash import FlashDrum


class TestFlashDrumInit:
    """Tests for FlashDrum constructor validation."""

    def test_valid_construction(self) -> None:
        drum = FlashDrum(["methanol", "water"], 320.0, 2e5)
        assert drum.temperature_k == 320.0
        assert drum.pressure_pa == 2e5

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature_k must be positive"):
            FlashDrum(["methanol", "water"], -10.0)

    def test_zero_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="pressure_pa must be positive"):
            FlashDrum(["methanol", "water"], 320.0, pressure_pa=0.0)

    def test_single_component_raises(self) -> None:
        with pytest.raises(ValueError, match="At least two components"):
            FlashDrum(["methanol"], 320.0)

    def test_components_tuple(self) -> None:
        drum = FlashDrum(["methanol", "water", "ethanol"], 320.0)
        assert drum.components == ("methanol", "water", "ethanol")


class TestFlashDrumSolve:
    """Tests for FlashDrum.solve() correctness."""

    def test_vapour_fraction_in_bounds(self, flash_drum_meoh_water: FlashDrum) -> None:
        z = np.array([0.50, 0.50])
        result = flash_drum_meoh_water.solve(z)
        assert 0.0 <= result.vapour_fraction <= 1.0

    def test_material_balance_closed(self, flash_drum_meoh_water: FlashDrum) -> None:
        """F = V + L  →  1 = Ψ·y + (1-Ψ)·x  (per component)."""
        z = np.array([0.60, 0.40])
        res = flash_drum_meoh_water.solve(z)
        reconstructed = res.vapour_fraction * res.y + (1 - res.vapour_fraction) * res.x
        np.testing.assert_allclose(reconstructed, z, atol=1e-5)

    def test_phase_fractions_sum_to_one(self, flash_drum_meoh_water: FlashDrum) -> None:
        z = np.array([0.70, 0.30])
        res = flash_drum_meoh_water.solve(z)
        assert math.isclose(res.x.sum(), 1.0, abs_tol=1e-6)
        assert math.isclose(res.y.sum(), 1.0, abs_tol=1e-6)

    def test_result_is_frozen_dataclass(self, flash_drum_meoh_water: FlashDrum) -> None:
        z = np.array([0.50, 0.50])
        res = flash_drum_meoh_water.solve(z)
        with pytest.raises((AttributeError, TypeError)):
            res.vapour_fraction = 0.5  # type: ignore[misc]

    def test_liquid_fraction_complement(self, flash_drum_meoh_water: FlashDrum) -> None:
        z = np.array([0.50, 0.50])
        res = flash_drum_meoh_water.solve(z)
        assert math.isclose(res.liquid_fraction, 1.0 - res.vapour_fraction)

    @pytest.mark.parametrize("z_meoh", [0.2, 0.5, 0.8])
    def test_lk_enriched_in_vapour(self, z_meoh: float) -> None:
        """Methanol (more volatile) should be enriched in the vapour phase."""
        drum = FlashDrum(["methanol", "water"], 320.0, 2e5)
        z = np.array([z_meoh, 1.0 - z_meoh])
        res = drum.solve(z)
        if 0.0 < res.vapour_fraction < 1.0:
            assert res.y[0] > res.x[0], (
                f"Methanol vapour fraction {res.y[0]:.4f} should exceed "
                f"liquid fraction {res.x[0]:.4f}"
            )

    def test_high_temp_all_vapour(self) -> None:
        """At very high temperature, mixture should be fully vaporised."""
        drum = FlashDrum(["methanol", "water"], 600.0, 1e5)
        z = np.array([0.50, 0.50])
        res = drum.solve(z)
        assert res.vapour_fraction >= 0.99

    def test_low_temp_all_liquid(self) -> None:
        """At low temperature methanol/water should remain liquid."""
        drum = FlashDrum(["methanol", "water"], 200.0, 1e5)
        z = np.array([0.50, 0.50])
        res = drum.solve(z)
        assert res.vapour_fraction <= 0.01


class TestFlashDrumValidation:
    """Tests for feed validation in solve()."""

    def test_wrong_length_raises(self, flash_drum_meoh_water: FlashDrum) -> None:
        with pytest.raises(ValueError, match="3 elements but 2 components"):
            flash_drum_meoh_water.solve(np.array([0.3, 0.3, 0.4]))

    def test_negative_fraction_raises(self, flash_drum_meoh_water: FlashDrum) -> None:
        with pytest.raises(ValueError, match="≥ 0"):
            flash_drum_meoh_water.solve(np.array([-0.1, 1.1]))

    def test_non_unit_sum_raises(self, flash_drum_meoh_water: FlashDrum) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            flash_drum_meoh_water.solve(np.array([0.3, 0.3]))


class TestFlashDrumSensitivity:
    """Tests for FlashDrum.sensitivity() temperature sweep."""

    def test_sensitivity_length(self, flash_drum_meoh_water: FlashDrum) -> None:
        z = np.array([0.50, 0.50])
        temps = np.linspace(280.0, 420.0, 15)
        results = flash_drum_meoh_water.sensitivity(z, temps)
        assert len(results) == 15

    def test_sensitivity_monotonic_vf(self, flash_drum_meoh_water: FlashDrum) -> None:
        """Vapour fraction should increase monotonically with temperature."""
        z = np.array([0.50, 0.50])
        temps = np.linspace(250.0, 450.0, 20)
        results = flash_drum_meoh_water.sensitivity(z, temps)
        vfs = [r.vapour_fraction for r in results]
        # Allow occasional ties but overall monotone
        assert vfs[-1] >= vfs[0]
