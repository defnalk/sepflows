"""Unit tests for CO₂ capture (absorption) and cryogenic ASU modules."""

from __future__ import annotations

import pytest

from sepflows.absorption import AmineAbsorber, CO2CaptureResult
from sepflows.asu import CryogenicASU, ASUResult
from sepflows.constants import CO2_CAPTURE_DEFAULTS


# ══════════════════════════════════════════════════════════════════════════════
# AmineAbsorber tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAmineAbsorberInit:
    """Constructor validation."""

    def test_valid_construction(self) -> None:
        a = AmineAbsorber(removal_target=0.90, inlet_co2_mol_h=5000.0, inlet_total_mol_h=50000.0)
        assert a is not None

    @pytest.mark.parametrize("removal", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_removal_raises(self, removal: float) -> None:
        with pytest.raises(ValueError, match="removal_target must be in"):
            AmineAbsorber(removal_target=removal)

    def test_rich_less_than_lean_raises(self) -> None:
        with pytest.raises(ValueError, match="rich_loading.*must exceed"):
            AmineAbsorber(lean_loading=0.45, rich_loading=0.20)

    def test_invalid_lrhx_effectiveness_raises(self) -> None:
        with pytest.raises(ValueError, match="lrhx_effectiveness"):
            AmineAbsorber(lrhx_effectiveness=1.5)


class TestAmineAbsorberSolve:
    """Numerical correctness."""

    def test_removal_fraction_matches_target(self, amine_absorber_default: AmineAbsorber) -> None:
        res = amine_absorber_default.solve()
        assert abs(res.co2_removal_fraction - 0.90) < 1e-9

    def test_captured_co2_positive(self, amine_absorber_default: AmineAbsorber) -> None:
        res = amine_absorber_default.solve()
        assert res.co2_captured_mol_h > 0.0

    def test_reboiler_duty_positive(self, amine_absorber_default: AmineAbsorber) -> None:
        res = amine_absorber_default.solve()
        assert res.reboiler_duty_kw > 0.0

    def test_lrhx_duty_positive(self, amine_absorber_default: AmineAbsorber) -> None:
        res = amine_absorber_default.solve()
        assert res.lean_rich_hx_duty_kw > 0.0

    def test_absorber_height_positive(self, amine_absorber_default: AmineAbsorber) -> None:
        res = amine_absorber_default.solve()
        assert res.absorber_height_m > 0.0

    def test_converged_flag(self, amine_absorber_default: AmineAbsorber) -> None:
        res = amine_absorber_default.solve()
        assert res.converged is True

    def test_solvent_circulation_increases_with_co2_load(self) -> None:
        """Higher CO₂ load → more solvent required."""
        a_low = AmineAbsorber(removal_target=0.90, inlet_co2_mol_h=1_000.0, inlet_total_mol_h=20_000.0)
        a_high = AmineAbsorber(removal_target=0.90, inlet_co2_mol_h=10_000.0, inlet_total_mol_h=50_000.0)
        assert a_high.solve().solvent_circulation_mol_h > a_low.solve().solvent_circulation_mol_h

    def test_higher_removal_more_duty(self) -> None:
        """90% removal should need less duty than 98%."""
        a_90 = AmineAbsorber(removal_target=0.90, inlet_co2_mol_h=5_000.0, inlet_total_mol_h=50_000.0)
        a_98 = AmineAbsorber(removal_target=0.98, inlet_co2_mol_h=5_000.0, inlet_total_mol_h=50_000.0)
        assert a_98.solve().reboiler_duty_kw > a_90.solve().reboiler_duty_kw

    def test_repr_contains_removal(self, amine_absorber_default: AmineAbsorber) -> None:
        res = amine_absorber_default.solve()
        assert "90.0%" in repr(res)

    @pytest.mark.parametrize("removal", [0.70, 0.85, 0.95])
    def test_various_removal_targets(self, removal: float) -> None:
        a = AmineAbsorber(removal_target=removal, inlet_co2_mol_h=5_000.0, inlet_total_mol_h=50_000.0)
        res = a.solve()
        assert abs(res.co2_removal_fraction - removal) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# CryogenicASU tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCryogenicASUInit:
    """Constructor validation for CryogenicASU."""

    def test_valid_construction(self) -> None:
        asu = CryogenicASU(n2_demand_mol_h=10_000.0, o2_demand_mol_h=5_000.0)
        assert asu is not None

    def test_both_zero_demand_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one of"):
            CryogenicASU(n2_demand_mol_h=0.0, o2_demand_mol_h=0.0)

    def test_negative_demand_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CryogenicASU(n2_demand_mol_h=-100.0, o2_demand_mol_h=5_000.0)

    def test_purity_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="n2_purity"):
            CryogenicASU(n2_purity=0.5)

    def test_hp_below_lp_raises(self) -> None:
        with pytest.raises(ValueError, match="hp_pressure_bar.*must exceed"):
            CryogenicASU(hp_pressure_bar=1.0, lp_pressure_bar=5.0)


class TestCryogenicASUSolve:
    """Numerical correctness of the ASU sizing."""

    def test_solve_returns_result(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert isinstance(res, ASUResult)

    def test_n2_purity_spec_met(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert res.n2_purity >= 0.999

    def test_o2_purity_spec_met(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert res.o2_purity >= 0.990

    def test_air_feed_positive(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert res.air_feed_mol_h > 0.0

    def test_compression_power_positive(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert res.compression_power_kw > 0.0

    def test_column_stages_positive(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert res.hp_column_stages > 0
        assert res.lp_column_stages > 0

    def test_recoveries_in_bounds(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert 0.0 < res.n2_recovery <= 1.0
        assert 0.0 < res.o2_recovery <= 1.0

    def test_argon_zero_without_recovery(self) -> None:
        asu = CryogenicASU(n2_demand_mol_h=10_000.0, o2_demand_mol_h=5_000.0, recover_argon=False)
        res = asu.solve()
        assert res.ar_flow_mol_h == 0.0

    def test_argon_positive_with_recovery(self) -> None:
        asu = CryogenicASU(n2_demand_mol_h=10_000.0, o2_demand_mol_h=5_000.0, recover_argon=True)
        res = asu.solve()
        assert res.ar_flow_mol_h > 0.0

    def test_larger_demand_more_air(self) -> None:
        asu_small = CryogenicASU(n2_demand_mol_h=1_000.0, o2_demand_mol_h=500.0)
        asu_large = CryogenicASU(n2_demand_mol_h=100_000.0, o2_demand_mol_h=50_000.0)
        assert asu_large.solve().air_feed_mol_h > asu_small.solve().air_feed_mol_h

    def test_repr_contains_flows(self, cryogenic_asu_default: CryogenicASU) -> None:
        res = cryogenic_asu_default.solve()
        assert "N₂" in repr(res) and "O₂" in repr(res)

    def test_only_n2_demand(self) -> None:
        asu = CryogenicASU(n2_demand_mol_h=20_000.0, o2_demand_mol_h=0.0)
        res = asu.solve()
        assert res.n2_flow_mol_h > 0.0

    def test_only_o2_demand(self) -> None:
        asu = CryogenicASU(n2_demand_mol_h=0.0, o2_demand_mol_h=10_000.0)
        res = asu.solve()
        assert res.o2_flow_mol_h > 0.0
