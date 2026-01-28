"""Unit tests for thermodynamic utility functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sepflows.utils.thermodynamics import (
    antoine_pressure,
    bubble_point_temperature,
    dew_point_temperature,
    k_values_raoult,
    rachford_rice,
    relative_volatility,
)
from sepflows.constants import P_ATM


class TestAntoinePressure:
    """Tests for antoine_pressure()."""

    def test_methanol_at_normal_bp(self) -> None:
        """Methanol Tb ≈ 337.85 K → P_sat ≈ 1 atm."""
        p = antoine_pressure("methanol", 337.85)
        assert math.isclose(p, P_ATM, rel_tol=0.02), f"P_sat={p:.0f} Pa, expected ~{P_ATM:.0f} Pa"

    def test_water_at_normal_bp(self) -> None:
        """Water Tb ≈ 373.15 K → P_sat ≈ 1 atm."""
        p = antoine_pressure("water", 373.15)
        assert math.isclose(p, P_ATM, rel_tol=0.02), f"P_sat={p:.0f} Pa"

    def test_increases_with_temperature(self) -> None:
        p_low = antoine_pressure("methanol", 300.0)
        p_high = antoine_pressure("methanol", 400.0)
        assert p_high > p_low

    def test_unknown_component_raises(self) -> None:
        with pytest.raises(KeyError, match="not in Antoine database"):
            antoine_pressure("unobtanium", 300.0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature_k must be positive"):
            antoine_pressure("methanol", -10.0)

    @pytest.mark.parametrize("comp", ["methanol", "water", "ethanol", "dme", "nitrogen", "oxygen"])
    def test_positive_pressure_all_components(self, comp: str) -> None:
        """All registered components must return positive P_sat."""
        p = antoine_pressure(comp, 300.0)
        assert p > 0.0


class TestKValuesRaoult:
    """Tests for k_values_raoult()."""

    def test_shape_matches_components(self) -> None:
        comps = ["methanol", "water", "ethanol"]
        k = k_values_raoult(comps, 340.0, P_ATM)
        assert k.shape == (3,)

    def test_lk_k_greater_than_one_at_bubble(self) -> None:
        """At bubble point of methanol-rich mixture, K_methanol > 1."""
        k = k_values_raoult(["methanol", "water"], 337.85, P_ATM)
        assert k[0] > 1.0  # methanol

    def test_hk_k_less_than_one_at_bubble(self) -> None:
        """At normal boiling point of methanol, K_water < 1."""
        k = k_values_raoult(["methanol", "water"], 337.85, P_ATM)
        assert k[1] < 1.0  # water

    def test_zero_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="pressure_pa must be positive"):
            k_values_raoult(["methanol", "water"], 337.0, 0.0)


class TestRachfordRice:
    """Tests for rachford_rice()."""

    def test_binary_known_solution(self) -> None:
        """z = [0.5, 0.5], K = [2, 0.5] → Ψ = 0.5 exactly.

        Derivation:
          0.5(2-1)/(1+Ψ) + 0.5(0.5-1)/(1-0.5Ψ) = 0
          0.5/(1+Ψ) = 0.25/(1-0.5Ψ)
          0.5(1-0.5Ψ) = 0.25(1+Ψ)  →  Ψ = 0.5
        """
        z = np.array([0.5, 0.5])
        k = np.array([2.0, 0.5])
        psi = rachford_rice(z, k)
        assert math.isclose(psi, 0.5, rel_tol=1e-6)

    def test_all_vapour_when_all_k_gt_one(self) -> None:
        z = np.array([0.5, 0.5])
        k = np.array([5.0, 3.0])  # both K > 1 → fully vaporised
        psi = rachford_rice(z, k)
        assert psi >= 0.99

    def test_all_liquid_when_all_k_lt_one(self) -> None:
        z = np.array([0.5, 0.5])
        k = np.array([0.1, 0.05])
        psi = rachford_rice(z, k)
        assert psi <= 0.01

    def test_z_not_summing_to_one_raises(self) -> None:
        z = np.array([0.3, 0.3])
        k = np.array([2.0, 0.5])
        with pytest.raises(ValueError, match="sum to 1.0"):
            rachford_rice(z, k)

    def test_shape_mismatch_raises(self) -> None:
        z = np.array([0.5, 0.5])
        k = np.array([2.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="same shape"):
            rachford_rice(z, k)

    @pytest.mark.parametrize("z0,k0,k1", [
        (0.3, 3.0, 0.3),
        (0.6, 2.0, 0.5),
        (0.8, 1.5, 0.7),
    ])
    def test_material_balance(self, z0: float, k0: float, k1: float) -> None:
        """Verify Rachford-Rice closure: z = Ψ·y + (1-Ψ)·x."""
        z = np.array([z0, 1.0 - z0])
        k = np.array([k0, k1])
        psi = rachford_rice(z, k)
        x = z / (1.0 + psi * (k - 1.0))
        y = k * x
        x /= x.sum(); y /= y.sum()
        reconstructed = psi * y + (1.0 - psi) * x
        np.testing.assert_allclose(reconstructed, z, atol=1e-6)


class TestRelativeVolatility:
    """Tests for relative_volatility()."""

    def test_methanol_more_volatile_than_water(self) -> None:
        alpha = relative_volatility("methanol", "water", 337.0, P_ATM)
        assert alpha > 1.0

    def test_water_less_volatile_than_methanol(self) -> None:
        alpha = relative_volatility("water", "methanol", 337.0, P_ATM)
        assert alpha < 1.0

    def test_dme_more_volatile_than_methanol(self) -> None:
        """DME (Tb=248 K) is more volatile than methanol (Tb=338 K)."""
        alpha = relative_volatility("dme", "methanol", 290.0, P_ATM)
        assert alpha > 1.0


class TestBubbleDewPoint:
    """Tests for bubble_point_temperature and dew_point_temperature."""

    def test_bubble_point_methanol_water(self) -> None:
        """Pure methanol bubble point ≈ 337.85 K at 1 atm."""
        comps = ["methanol", "water"]
        x = np.array([1.0, 1e-9])
        x /= x.sum()
        tbp = bubble_point_temperature(comps, x, P_ATM, t_init_k=338.0)
        assert 330.0 < tbp < 345.0

    def test_dew_point_methanol_water(self) -> None:
        """Dew point of pure methanol vapour ≈ 337.85 K at 1 atm."""
        comps = ["methanol", "water"]
        y = np.array([1.0, 1e-9])
        y /= y.sum()
        tdp = dew_point_temperature(comps, y, P_ATM, t_init_k=338.0)
        assert 330.0 < tdp < 345.0

    def test_bubble_below_dew_for_mixture(self) -> None:
        """Bubble point < Dew point for the same mixture composition."""
        comps = ["methanol", "water"]
        z = np.array([0.50, 0.50])
        tbp = bubble_point_temperature(comps, z, P_ATM, t_init_k=355.0)
        tdp = dew_point_temperature(comps, z, P_ATM, t_init_k=355.0)
        assert tbp < tdp
