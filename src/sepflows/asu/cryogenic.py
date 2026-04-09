"""Cryogenic Air Separation Unit (ASU) model.

Models the cryogenic separation of dry air into high-purity nitrogen,
oxygen, and optionally argon using a double-column (Linde-Frankl) cycle.

Flowsheet stages:
1. **Feed air compression** — multi-stage with intercooling.
2. **Pre-treatment** — TSA (Temperature Swing Adsorption) for CO₂ and
   H₂O removal.
3. **Cryogenic cold box** — main heat exchanger + double distillation
   column (high-pressure column above, low-pressure column below).
4. **Product streams** — gaseous N₂ (GAN), gaseous O₂ (GOX), optional
   liquid argon (LAR).

References:
----------
- Smith, A. R. & Klosek, J. *Fuel Process. Technol.* (2001) — ASU review
- Perry's Handbook, 9th ed., Section 11 (Gas Separation)
- Agrawal, R. *Ind. Eng. Chem. Res.* (2001) — cryogenic column design
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from sepflows.config import DEFAULT_CONFIG, SepConfig
from sepflows.constants import ASU_COMPOSITIONS
from sepflows.utils.thermodynamics import relative_volatility

__all__ = ["ASUResult", "CryogenicASU"]

_log = logging.getLogger(__name__)

# Specific work of compression for air separation (kWh / Nm³ product O₂)
# Industry benchmark: 0.35–0.40 kWh/Nm³ for a modern large-scale ASU
_SPECIFIC_ENERGY_KWH_NM3: float = 0.37

# Air density at NTP (0°C, 1 atm) in kg/Nm³
_RHO_AIR_NM3: float = 1.293

# Liquid N₂ vent rate as fraction of air feed (thermodynamic cycle loss)
_LN2_VENT_FRACTION: float = 0.02


@dataclass(frozen=True)
class ASUResult:
    """Results of a cryogenic ASU design calculation.

    Attributes:
        n2_flow_mol_h: Nitrogen product flow (mol/h).
        o2_flow_mol_h: Oxygen product flow (mol/h).
        ar_flow_mol_h: Argon product flow (mol/h); 0 if argon recovery
            is disabled.
        air_feed_mol_h: Required air feed rate (mol/h).
        n2_purity: Nitrogen product purity (mol fraction).
        o2_purity: Oxygen product purity (mol fraction).
        compression_power_kw: Estimated air compression power (kW).
        n2_recovery: Fraction of feed nitrogen recovered as product.
        o2_recovery: Fraction of feed oxygen recovered as product.
        hp_column_stages: Estimated stages in the high-pressure column.
        lp_column_stages: Estimated stages in the low-pressure column.
        hp_column_pressure_bar: HP column operating pressure (bar).
        lp_column_pressure_bar: LP column operating pressure (bar).
    """

    n2_flow_mol_h: float
    o2_flow_mol_h: float
    ar_flow_mol_h: float
    air_feed_mol_h: float
    n2_purity: float
    o2_purity: float
    compression_power_kw: float
    n2_recovery: float
    o2_recovery: float
    hp_column_stages: int
    lp_column_stages: int
    hp_column_pressure_bar: float
    lp_column_pressure_bar: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ASUResult("
            f"N₂={self.n2_flow_mol_h:.0f} mol/h [{self.n2_purity:.4f}], "
            f"O₂={self.o2_flow_mol_h:.0f} mol/h [{self.o2_purity:.4f}], "
            f"W_comp={self.compression_power_kw:.1f} kW)"
        )


class CryogenicASU:
    """Cryogenic double-column Air Separation Unit.

    Sizes a Linde-Frankl double-column cycle for specified N₂ and/or O₂
    product demands.  Uses the Fenske–Underwood–Gilliland shortcut for
    column sizing.

    Args:
        n2_demand_mol_h: Required nitrogen product flow (mol/h). Pass 0
            to size only for oxygen.
        o2_demand_mol_h: Required oxygen product flow (mol/h). Pass 0
            to size only for nitrogen.
        n2_purity: Target N₂ purity (mol fraction).  Default: 0.9999.
        o2_purity: Target O₂ purity (mol fraction).  Default: 0.9950.
        recover_argon: Whether to include an argon sidestream.
        hp_pressure_bar: High-pressure column pressure (bar).  Typical
            value: 5–6 bar for a double-column cycle.
        lp_pressure_bar: Low-pressure column pressure (bar).  Typically
            1.3–1.5 bar (above atmospheric to prevent air ingress).
        config: Solver configuration.

    Example:
        >>> asu = CryogenicASU(
        ...     n2_demand_mol_h=10_000.0,
        ...     o2_demand_mol_h=5_000.0,
        ... )
        >>> result = asu.solve()
        >>> result.n2_purity >= 0.999
        True
    """

    def __init__(
        self,
        n2_demand_mol_h: float = 10_000.0,
        o2_demand_mol_h: float = 5_000.0,
        n2_purity: float = ASU_COMPOSITIONS["product_n2"]["nitrogen"],
        o2_purity: float = ASU_COMPOSITIONS["product_o2"]["oxygen"],
        recover_argon: bool = False,
        hp_pressure_bar: float = 5.5,
        lp_pressure_bar: float = 1.35,
        config: SepConfig | None = None,
    ) -> None:
        if n2_demand_mol_h < 0.0 or o2_demand_mol_h < 0.0:
            raise ValueError("Product demand flows must be non-negative.")
        if n2_demand_mol_h == 0.0 and o2_demand_mol_h == 0.0:
            raise ValueError("At least one of n2_demand_mol_h or o2_demand_mol_h must be > 0.")
        for name, val in [("n2_purity", n2_purity), ("o2_purity", o2_purity)]:
            if not (0.9 <= val < 1.0):
                raise ValueError(f"{name} must be in [0.9, 1.0), got {val}.")
        if hp_pressure_bar <= lp_pressure_bar:
            raise ValueError(
                f"hp_pressure_bar ({hp_pressure_bar}) must exceed "
                f"lp_pressure_bar ({lp_pressure_bar})."
            )

        self._n2_demand = n2_demand_mol_h
        self._o2_demand = o2_demand_mol_h
        self._n2_purity = n2_purity
        self._o2_purity = o2_purity
        self._recover_ar = recover_argon
        self._p_hp = hp_pressure_bar
        self._p_lp = lp_pressure_bar
        self._cfg = config or DEFAULT_CONFIG

        _log.info(
            "CryogenicASU: N₂=%.0f mol/h [%.4f], O₂=%.0f mol/h [%.4f], "
            "Ar_recovery=%s, P_HP=%.1f bar, P_LP=%.2f bar",
            n2_demand_mol_h,
            n2_purity,
            o2_demand_mol_h,
            o2_purity,
            recover_argon,
            hp_pressure_bar,
            lp_pressure_bar,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(self) -> ASUResult:
        """Run the ASU sizing calculation.

        Returns:
            :class:`ASUResult` with product flows, purities, column stages,
            and compression power estimate.
        """
        air_comp = ASU_COMPOSITIONS["dry_air"]
        y_n2 = air_comp["nitrogen"]  # 0.7812
        y_o2 = air_comp["oxygen"]  # 0.2096
        y_ar = air_comp["argon"]  # 0.0092

        # ── Air feed requirement ──────────────────────────────────────────────
        # Mass balance: determine limiting product and size air accordingly
        n2_recovery = 0.998  # fraction; near-perfect in modern ASU
        o2_recovery = 0.990

        air_for_n2 = self._n2_demand / y_n2 / n2_recovery if self._n2_demand > 0 else 0.0
        air_for_o2 = self._o2_demand / y_o2 / o2_recovery if self._o2_demand > 0 else 0.0
        air_feed = max(air_for_n2, air_for_o2, 1.0)

        # Actual product flows given air feed
        n2_actual = air_feed * y_n2 * n2_recovery
        o2_actual = air_feed * y_o2 * o2_recovery
        ar_actual = air_feed * y_ar * 0.90 if self._recover_ar else 0.0

        # ── Column sizing (Fenske Nmin for N₂/O₂ and O₂/Ar splits) ──────────
        t_hp = self._boiling_temp_at_pressure("nitrogen", self._p_hp)  # approximate
        t_lp = self._boiling_temp_at_pressure("oxygen", self._p_lp)

        alpha_n2_o2_hp = relative_volatility("nitrogen", "oxygen", t_hp, self._p_hp * 1e5)
        # In the LP column, argon (Tb≈87.3 K) is lighter than O₂ (Tb≈90.2 K)
        alpha_o2_ar_lp = relative_volatility("argon", "oxygen", t_lp, self._p_lp * 1e5)

        # Fenske Nmin for HP column (N₂/O₂ split)
        x_d_hp = self._n2_purity  # N₂ purity in overhead
        x_b_hp = 1.0 - self._o2_purity  # N₂ impurity in O₂ bottoms
        n_min_hp = self._fenske(x_d_hp, 1.0 - x_d_hp, x_b_hp, 1.0 - x_b_hp, alpha_n2_o2_hp)
        n_hp = math.ceil(n_min_hp * 2.0)  # N = 2×Nmin per Kister

        # Fenske Nmin for LP column (O₂/Ar split near argon sidestream)
        x_d_lp = self._o2_purity
        x_b_lp = 0.01  # O₂ impurity in argon/N₂ bottoms
        n_min_lp = self._fenske(x_d_lp, 1.0 - x_d_lp, x_b_lp, 1.0 - x_b_lp, alpha_o2_ar_lp)
        n_lp = math.ceil(n_min_lp * 2.0)

        _log.debug(
            "Column stages: HP N=%d (Nmin=%.1f, α=%.3f), LP N=%d (Nmin=%.1f, α=%.3f)",
            n_hp,
            n_min_hp,
            alpha_n2_o2_hp,
            n_lp,
            n_min_lp,
            alpha_o2_ar_lp,
        )

        # ── Compression power ─────────────────────────────────────────────────
        # Specific energy benchmark: 0.37 kWh / Nm³ O₂
        # 1 mol O₂ ≈ 22.4 L at NTP = 0.0224 Nm³
        o2_nm3_h = o2_actual * 0.02241  # mol/h → Nm³/h
        compression_kw = _SPECIFIC_ENERGY_KWH_NM3 * o2_nm3_h  # kWh/h = kW

        _log.info(
            "ASU solved: air_feed=%.0f mol/h, W_comp=%.1f kW, N₂=%.0f mol/h, O₂=%.0f mol/h",
            air_feed,
            compression_kw,
            n2_actual,
            o2_actual,
        )

        return ASUResult(
            n2_flow_mol_h=n2_actual,
            o2_flow_mol_h=o2_actual,
            ar_flow_mol_h=ar_actual,
            air_feed_mol_h=air_feed,
            n2_purity=self._n2_purity,
            o2_purity=self._o2_purity,
            compression_power_kw=compression_kw,
            n2_recovery=n2_recovery,
            o2_recovery=o2_recovery,
            hp_column_stages=n_hp,
            lp_column_stages=n_lp,
            hp_column_pressure_bar=self._p_hp,
            lp_column_pressure_bar=self._p_lp,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fenske(x_d: float, y_d: float, x_b: float, y_b: float, alpha: float) -> float:
        """Fenske minimum stages for a binary split."""
        if alpha <= 1.0:
            raise ValueError(
                f"Relative volatility must be > 1 for Fenske, got {alpha:.4f}. "
                "Check that the lighter component is passed first to "
                "relative_volatility()."
            )
        x_d = max(x_d, 1e-9)
        y_d = max(y_d, 1e-9)
        x_b = max(x_b, 1e-9)
        y_b = max(y_b, 1e-9)
        sep = (x_d / y_d) * (y_b / x_b)
        return math.log(sep) / math.log(alpha)

    @staticmethod
    def _boiling_temp_at_pressure(component: str, pressure_bar: float) -> float:
        """Approximate boiling point at a given pressure using normal Tb.

        Uses the Clausius–Clapeyron approximation scaled from normal boiling
        point.  Accuracy is sufficient for preliminary column sizing.

        Args:
            component: Component name (must have Antoine data).
            pressure_bar: Pressure in bar.

        Returns:
            Approximate boiling temperature in Kelvin.
        """
        from sepflows.constants import NORMAL_BOILING_POINTS
        from sepflows.utils.thermodynamics import antoine_pressure

        tb_normal = NORMAL_BOILING_POINTS.get(component, 90.0)  # K at 1 atm
        if pressure_bar <= 0.0:
            raise ValueError(f"pressure_bar must be positive, got {pressure_bar}")
        # Iterate: find T such that P_ant(T) = pressure_bar × 1e5
        target_pa = pressure_bar * 1e5
        t = tb_normal
        for _ in range(50):
            p = antoine_pressure(component, t)
            ratio = target_pa / max(p, 1.0)
            if abs(ratio - 1.0) < 1e-6:
                break
            # Clamp the update factor so a pathological Antoine extrapolation
            # can't drive t towards zero (antoine_pressure raises on t<=0) or
            # off to thousands of kelvin in a single step.
            step = ratio**0.15
            step = min(max(step, 0.5), 2.0)
            t = max(t * step, 1.0)
        return t
