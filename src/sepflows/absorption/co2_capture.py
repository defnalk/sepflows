"""CO₂ capture by chemical absorption — amine solvent model.

Models a two-column CO₂ capture system (absorber + stripper/regenerator)
representative of the MEA-based unit in the methanol synthesis recycle loop
(CENG50009 Separation Processes 2).

Key modelling assumptions:
- Liquid film controls mass transfer (Hatta number < 1 for rich MEA).
- Henry's law K-value used for CO₂ partitioning at absorber conditions.
- Lean/rich heat exchanger modelled as a simple duty split.
- Stripper reboiler duty estimated from the industry benchmark for MEA.

References
----------
- Kohl, A. L. & Nielsen, R. *Gas Purification*, 5th ed. (Gulf, 1997)
- Rochelle, G. T., *Science* 325 (2009) — solvent screening review
- IPCC (2005) CCS Special Report, Annex I
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from sepflows.config import DEFAULT_CONFIG, SepConfig
from sepflows.constants import CO2_CAPTURE_DEFAULTS, P_ATM

__all__ = ["CO2CaptureResult", "AmineAbsorber"]

_log = logging.getLogger(__name__)

# CO₂ Henry's constant in aqueous MEA at 40 °C (absorber) — Pa·m³/mol
# Fitted from Jou et al. (1995) data
_HE_CO2_MEA_40C: float = 3_400.0  # Pa·m³/mol (rough average, loading-dependent)

# Solvent heat capacity (kJ/(kg·K))
_CP_MEA_SOLUTION: float = 3.6

# CO₂ heat of absorption in MEA (kJ/mol CO₂)
_DELTA_H_ABS: float = -85.0  # exothermic


@dataclass(frozen=True)
class CO2CaptureResult:
    """Results of a CO₂ capture system calculation.

    Attributes:
        co2_captured_mol_h: CO₂ captured in mol/h (or same unit as input).
        co2_removal_fraction: Fraction of inlet CO₂ removed.
        lean_loading: Lean solvent loading (mol CO₂/mol amine) after
            regeneration.
        rich_loading: Rich solvent loading (mol CO₂/mol amine) after
            absorption.
        solvent_circulation_mol_h: Amine solvent circulation rate
            (mol solvent/h).
        reboiler_duty_kw: Stripper reboiler duty in kW.
        lean_rich_hx_duty_kw: Lean/rich heat exchanger duty in kW.
        absorber_height_m: Estimated absorber packing height in metres.
        converged: Whether the calculation closed successfully.
    """

    co2_captured_mol_h: float
    co2_removal_fraction: float
    lean_loading: float
    rich_loading: float
    solvent_circulation_mol_h: float
    reboiler_duty_kw: float
    lean_rich_hx_duty_kw: float
    absorber_height_m: float
    converged: bool

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CO2CaptureResult("
            f"removal={self.co2_removal_fraction:.1%}, "
            f"Q_reb={self.reboiler_duty_kw:.1f} kW, "
            f"circ={self.solvent_circulation_mol_h:.0f} mol/h)"
        )


class AmineAbsorber:
    """MEA-based CO₂ absorber–stripper unit.

    Calculates solvent circulation, reboiler duty, and approximate column
    dimensions for a target CO₂ removal fraction.

    Args:
        removal_target: Fractional CO₂ removal (e.g. 0.90 for 90 %).
        inlet_co2_mol_h: CO₂ molar flow in the feed gas (mol/h).
        inlet_total_mol_h: Total molar flow of feed gas (mol/h).
        absorber_pressure_pa: Absorber operating pressure in Pa.
        stripper_pressure_pa: Stripper operating pressure in Pa.
        lean_loading: Target lean amine loading (mol CO₂/mol amine).
        rich_loading: Target rich amine loading (mol CO₂/mol amine).
        amine_concentration_wt: MEA weight fraction in aqueous solution.
        lrhx_effectiveness: Lean–rich heat exchanger effectiveness (0–1).
        config: Solver configuration.

    Example:
        >>> absorber = AmineAbsorber(
        ...     removal_target=0.90,
        ...     inlet_co2_mol_h=5_000.0,
        ...     inlet_total_mol_h=50_000.0,
        ... )
        >>> result = absorber.solve()
        >>> result.co2_removal_fraction
        0.9
    """

    def __init__(
        self,
        removal_target: float = CO2_CAPTURE_DEFAULTS["co2_removal_target"],
        inlet_co2_mol_h: float = 5_000.0,
        inlet_total_mol_h: float = 50_000.0,
        absorber_pressure_pa: float = CO2_CAPTURE_DEFAULTS["absorber_pressure_bar"] * 1e5,
        stripper_pressure_pa: float = CO2_CAPTURE_DEFAULTS["stripper_pressure_bar"] * 1e5,
        lean_loading: float = CO2_CAPTURE_DEFAULTS["lean_loading"],
        rich_loading: float = CO2_CAPTURE_DEFAULTS["rich_loading"],
        amine_concentration_wt: float = CO2_CAPTURE_DEFAULTS["amine_concentration"],
        lrhx_effectiveness: float = 0.85,
        config: SepConfig | None = None,
    ) -> None:
        self._validate_inputs(removal_target, lean_loading, rich_loading, lrhx_effectiveness)
        self._removal = removal_target
        self._f_co2 = inlet_co2_mol_h
        self._f_total = inlet_total_mol_h
        self._p_abs = absorber_pressure_pa
        self._p_str = stripper_pressure_pa
        self._lean = lean_loading
        self._rich = rich_loading
        self._c_amine = amine_concentration_wt / 100.0  # convert wt% → fraction
        self._eps_hx = lrhx_effectiveness
        self._cfg = config or DEFAULT_CONFIG

        # Molecular weight of MEA = 61.08 g/mol
        self._mw_mea: float = 61.08
        # Molecular weight of water = 18.015 g/mol
        self._mw_water: float = 18.015

        _log.info(
            "AmineAbsorber: target_removal=%.1f%%, F_CO2=%.0f mol/h, P_abs=%.2f bar",
            removal_target * 100,
            inlet_co2_mol_h,
            absorber_pressure_pa / 1e5,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(self) -> CO2CaptureResult:
        """Run the absorber–stripper design calculation.

        Returns:
            :class:`CO2CaptureResult` with duty, sizing, and loading data.
        """
        co2_captured = self._removal * self._f_co2
        _log.debug("CO₂ captured = %.1f mol/h", co2_captured)

        # ── Solvent circulation ───────────────────────────────────────────────
        delta_loading = self._rich - self._lean
        if delta_loading <= 0.0:
            raise ValueError(
                f"rich_loading ({self._rich}) must be greater than "
                f"lean_loading ({self._lean})."
            )
        # mol amine/h = mol CO₂ captured / Δloading
        amine_flow = co2_captured / delta_loading
        _log.debug("Amine circulation = %.1f mol MEA/h", amine_flow)

        # ── Reboiler duty ─────────────────────────────────────────────────────
        # = (sensible heat + heat of absorption + steam latent heat) × CO₂ captured
        # Simplified: use industry benchmark GJ/t CO₂
        mw_co2 = 44.01  # g/mol
        co2_tonne_h = co2_captured * mw_co2 / 1e6  # t/h
        reboiler_duty_gj_h = CO2_CAPTURE_DEFAULTS["reboiler_duty_gj_t"] * co2_tonne_h
        reboiler_duty_kw = reboiler_duty_gj_h * 1e6 / 3600.0  # GJ/h → kW

        # ── Lean–rich HX duty ─────────────────────────────────────────────────
        # Sensible heat = amine_flow × Cp × ΔT (rich → stripper temp)
        delta_t_hx = 80.0  # K (typical rich inlet ≈ 40°C → stripper ≈ 120°C)
        cp_kj_mol_k = _CP_MEA_SOLUTION * (
            self._c_amine * self._mw_mea + (1 - self._c_amine) * self._mw_water
        ) / 1000.0  # kJ/(mol·K)
        lrhx_duty_kw = (
            self._eps_hx * amine_flow * cp_kj_mol_k * delta_t_hx / 3600.0
        ) * 1000.0  # kJ/h → kW  (× 1000/3600)
        # Actually: kJ/mol·K × mol/h × K ÷ 3.6 → kW
        lrhx_duty_kw = self._eps_hx * amine_flow * cp_kj_mol_k * delta_t_hx / 3.6

        # ── Absorber height estimate ──────────────────────────────────────────
        absorber_height = self._estimate_packing_height(
            co2_captured, amine_flow
        )

        _log.info(
            "CO₂ capture: removal=%.1f%%, Q_reb=%.1f kW, Q_LRHX=%.1f kW, H_abs=%.1f m",
            self._removal * 100,
            reboiler_duty_kw,
            lrhx_duty_kw,
            absorber_height,
        )

        return CO2CaptureResult(
            co2_captured_mol_h=co2_captured,
            co2_removal_fraction=self._removal,
            lean_loading=self._lean,
            rich_loading=self._rich,
            solvent_circulation_mol_h=amine_flow,
            reboiler_duty_kw=reboiler_duty_kw,
            lean_rich_hx_duty_kw=lrhx_duty_kw,
            absorber_height_m=absorber_height,
            converged=True,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_packing_height(
        self,
        co2_captured: float,
        amine_flow: float,
    ) -> float:
        """Estimate absorber packing height using NTU–HTU approach.

        Uses a fixed overall height of a transfer unit (HTU) of 0.6 m
        (typical structured packing, Sulzer MellapakPlus) and estimates
        the number of transfer units from the log-mean driving force.

        Args:
            co2_captured: Captured CO₂ flow (mol/h).
            amine_flow: Circulating amine flow (mol/h).

        Returns:
            Estimated packing height in metres.
        """
        htu = 0.60  # m per transfer unit (structured packing heuristic)
        # NTU approximation: 3–8 for 90 % removal in MEA
        co2_inlet = self._f_co2
        co2_outlet = co2_inlet - co2_captured
        if co2_inlet <= 0 or co2_outlet <= 0:
            return 10.0  # fallback
        y_in = co2_inlet / self._f_total
        y_out = co2_outlet / (self._f_total - co2_captured)
        if y_out <= 0.0:
            y_out = 1e-6
        ntu = math.log(y_in / y_out) / (1.0 - y_out / y_in + 1e-9)
        ntu = max(ntu, 2.0)  # physical lower bound
        return htu * ntu

    @staticmethod
    def _validate_inputs(
        removal_target: float,
        lean_loading: float,
        rich_loading: float,
        lrhx_effectiveness: float,
    ) -> None:
        """Validate constructor arguments."""
        if not (0.0 < removal_target < 1.0):
            raise ValueError(
                f"removal_target must be in (0, 1), got {removal_target}."
            )
        if lean_loading <= 0.0:
            raise ValueError(f"lean_loading must be positive, got {lean_loading}.")
        if rich_loading <= lean_loading:
            raise ValueError(
                f"rich_loading ({rich_loading}) must exceed lean_loading ({lean_loading})."
            )
        if not (0.0 <= lrhx_effectiveness <= 1.0):
            raise ValueError(
                f"lrhx_effectiveness must be in [0, 1], got {lrhx_effectiveness}."
            )
