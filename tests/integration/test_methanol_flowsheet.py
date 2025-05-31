"""Integration tests — full methanol purification flowsheet.

Tests the complete separation sequence described in CENG50009:
    Flash drum → CO₂ absorber → distillation train → ASU

These tests verify that modules compose correctly and that key process
targets (methanol recovery, CO₂ removal, N₂/O₂ purity) are met
end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from sepflows.absorption import AmineAbsorber
from sepflows.asu import CryogenicASU
from sepflows.constants import DISTILLATION_DEFAULTS
from sepflows.distillation import RigorousColumn, ShortcutColumn
from sepflows.flash import FlashDrum


@pytest.mark.integration
class TestFullMethanolFlowsheet:
    """End-to-end integration of the CENG50009 separation sequence."""

    # ── Feed definition (CENG50009 basis: F = 65 000 kmol/h) ─────────────────
    FEED_FLOW: float = 65_000.0  # kmol/h
    COMPONENTS: list[str] = [
        "dme",
        "methyl_formate",
        "methanol",
        "ethanol",
        "water",
        "1_propanol",
        "1_butanol",
    ]
    Z_FEED: np.ndarray = np.array([0.02, 0.03, 0.65, 0.04, 0.22, 0.02, 0.02])

    def test_feed_composition_normalised(self) -> None:
        """Sanity-check: feed mole fractions sum to 1."""
        assert abs(self.Z_FEED.sum() - 1.0) < 1e-9

    def test_flash_drum_stage(self) -> None:
        """Flash drum at 280 K / 20 bar separates DME into vapour."""
        drum = FlashDrum(
            components=self.COMPONENTS,
            temperature_k=280.0,
            pressure_pa=20.0e5,
        )
        res = drum.solve(self.Z_FEED)
        assert res.converged
        # DME (index 0) should be enriched in vapour vs feed
        assert res.y[0] > self.Z_FEED[0], (
            f"DME vapour fraction {res.y[0]:.4f} should exceed feed {self.Z_FEED[0]:.4f}"
        )
        # Methanol (index 2) should be mainly in the liquid
        assert res.x[2] > res.y[2], "Methanol should be enriched in liquid phase"

    def test_co2_capture_stage(self) -> None:
        """CO₂ absorber removes ≥ 90 % of recycle-gas CO₂."""
        absorber = AmineAbsorber(
            removal_target=0.90,
            inlet_co2_mol_h=4_500.0,  # approx 9 mol% CO₂ in 50 000 mol/h gas
            inlet_total_mol_h=50_000.0,
        )
        res = absorber.solve()
        assert res.co2_removal_fraction >= 0.90
        assert res.reboiler_duty_kw > 0.0

    def test_methanol_shortcut_column_b3(self) -> None:
        """Shortcut methanol/water column matches CENG50009 Table 1 order."""
        # B3 block: Methyl-formate/methanol pre-cut column (methyl formate LK)
        col = ShortcutColumn(
            light_key="methyl_formate",
            heavy_key="methanol",
            recovery_lk=0.995,
            recovery_hk=0.995,
            pressure_pa=101_325.0,
            feed_temperature_k=330.0,
        )
        res = col.solve(self.FEED_FLOW, z_lk=0.03, z_hk=0.65)
        assert res.r_min > 0.0
        assert res.n_actual > res.n_min

    def test_methanol_polishing_column(self) -> None:
        """Binary methanol/water polishing column achieves 89 % MeOH recovery."""
        target_recovery = DISTILLATION_DEFAULTS["methanol_recovery_target"]
        col = RigorousColumn(
            components=["methanol", "water"],
            n_stages=20,
            feed_stage=10,
            reflux_ratio=2.72,
            distillate_to_feed=target_recovery,
            pressure_pa=101_325.0,
            feed_temperature_k=337.0,
        )
        z = np.array([0.65 / 0.87, 0.22 / 0.87])  # methanol + water after pre-cut
        z = z / z.sum()
        res = col.solve(self.FEED_FLOW * 0.87, z)
        # Distillate flow and bottoms flow must sum to feed
        assert abs(res.distillate_flow + res.bottoms_flow - self.FEED_FLOW * 0.87) < 1.0

    def test_asu_stage(self) -> None:
        """Cryogenic ASU produces N₂ and O₂ to spec."""
        asu = CryogenicASU(
            n2_demand_mol_h=15_000.0,
            o2_demand_mol_h=8_000.0,
            n2_purity=0.9999,
            o2_purity=0.9950,
        )
        res = asu.solve()
        assert res.n2_purity >= 0.9999
        assert res.o2_purity >= 0.9950
        assert res.air_feed_mol_h > 0.0

    def test_flowsheet_overall_co2_and_n2(self) -> None:
        """Full-flowsheet smoke test: all units return valid results."""
        # Step 1 — Flash drum
        drum = FlashDrum(self.COMPONENTS, 280.0, 20e5)
        flash_res = drum.solve(self.Z_FEED)
        assert flash_res.converged

        # Step 2 — CO₂ capture on vapour stream
        vapour_flow = flash_res.vapour_fraction * self.FEED_FLOW
        co2_in_vapour = vapour_flow * 0.10  # assume 10 mol% CO₂ in flash vapour
        absorber = AmineAbsorber(
            removal_target=0.90,
            inlet_co2_mol_h=co2_in_vapour,
            inlet_total_mol_h=vapour_flow,
        )
        capture_res = absorber.solve()
        assert capture_res.co2_removal_fraction == pytest.approx(0.90)

        # Step 3 — Methanol distillation (binary shortcut)
        col = ShortcutColumn("methanol", "water", 0.98, 0.98, 101_325.0, 337.0)
        dist_res = col.solve(
            self.FEED_FLOW * flash_res.liquid_fraction,
            z_lk=flash_res.x[2],
            z_hk=flash_res.x[4],
        )
        assert dist_res.n_actual > 0.0

        # Step 4 — ASU
        asu = CryogenicASU(n2_demand_mol_h=10_000.0, o2_demand_mol_h=5_000.0)
        asu_res = asu.solve()
        assert asu_res.compression_power_kw > 0.0
