"""Shortcut distillation design via the DSTWU / Fenske–Underwood–Gilliland method.

Implements the three-step shortcut procedure used in Aspen Plus DSTWU:

1. **Fenske equation** — minimum stages at total reflux (Nmin).
2. **Underwood equations** — minimum reflux at infinite stages (Rmin).
3. **Gilliland correlation** — actual stages N and feed stage Nf for
   a chosen operating reflux ratio R.

Results are validated against the DSTWU values from CENG50009 coursework
(Table 1: B3, B8, B11, B13 column blocks).

References:
----------
- Fenske, M. R., *Ind. Eng. Chem.* 24 (1932)
- Underwood, A. J. V., *Chem. Eng. Prog.* 44 (1948)
- Gilliland, E. R., *Ind. Eng. Chem.* 32 (1940)
- Kister, H. Z. *Distillation Design* (McGraw-Hill, 1992), Ch. 3
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from sepflows.config import DEFAULT_CONFIG, SepConfig
from sepflows.constants import DISTILLATION_DEFAULTS, P_ATM
from sepflows.utils.thermodynamics import (
    minimum_reflux_underwood,
    relative_volatility,
)

__all__ = ["DSWTUResult", "ShortcutColumn"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSWTUResult:
    """Results of a DSTWU shortcut distillation calculation.

    Attributes:
        r_min: Minimum reflux ratio (Underwood).
        r_actual: Actual (design) reflux ratio = r_min × multiplier.
        n_min: Minimum theoretical stages at total reflux (Fenske).
        n_actual: Actual theoretical stages (Gilliland).
        n_feed: Optimal feed stage (Kirkbride correlation, from top).
        alpha_lk_hk: Relative volatility of light key / heavy key at
            average column temperature.
        recovery_lk: Light-key recovery in distillate (mol fraction).
        recovery_hk: Heavy-key recovery in bottoms (mol fraction).
    """

    r_min: float
    r_actual: float
    n_min: float
    n_actual: float
    n_feed: float
    alpha_lk_hk: float
    recovery_lk: float
    recovery_hk: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DSWTUResult("
            f"Rmin={self.r_min:.4f}, R={self.r_actual:.4f}, "
            f"Nmin={self.n_min:.4f}, N={self.n_actual:.4f}, "
            f"Nf={self.n_feed:.1f})"
        )

    @property
    def separation_factor(self) -> float:
        """Product of LK distillate recovery and HK bottoms recovery."""
        return self.recovery_lk * self.recovery_hk


class ShortcutColumn:
    """DSTWU shortcut distillation column.

    Models a single binary or pseudo-binary distillation column using the
    Fenske–Underwood–Gilliland (FUG) procedure.

    Args:
        light_key: Component name for the light key.
        heavy_key: Component name for the heavy key.
        recovery_lk: Fractional recovery of light key in distillate
            (e.g. 0.995).
        recovery_hk: Fractional recovery of heavy key in bottoms
            (e.g. 0.995).
        pressure_pa: Column operating pressure in Pa.
        feed_temperature_k: Approximate feed temperature for K-value
            evaluation.
        q: Feed thermal condition.  ``q = 1`` for bubble-point (saturated
            liquid) feed, ``q = 0`` for dew-point feed.
        reflux_multiplier: Factor applied to Rmin to get operating R.
            Defaults to 1.30 (Kister heuristic).
        stages_multiplier: Factor applied to Nmin to get N.
            Defaults to 2.00.
        config: Solver configuration.

    Example:
        >>> col = ShortcutColumn(
        ...     light_key="methanol",
        ...     heavy_key="water",
        ...     recovery_lk=0.995,
        ...     recovery_hk=0.995,
        ...     pressure_pa=101_325.0,
        ...     feed_temperature_k=337.0,
        ... )
        >>> res = col.solve(feed_flow=65_000.0, z_lk=0.60, z_hk=0.35)
        >>> res.r_min  # doctest: +ELLIPSIS
        1...
    """

    def __init__(
        self,
        light_key: str,
        heavy_key: str,
        recovery_lk: float,
        recovery_hk: float,
        pressure_pa: float = P_ATM,
        feed_temperature_k: float = 337.0,
        q: float = 1.0,
        reflux_multiplier: float = DISTILLATION_DEFAULTS["reflux_ratio_multiplier"],
        stages_multiplier: float = DISTILLATION_DEFAULTS["stages_multiplier"],
        config: SepConfig | None = None,
    ) -> None:
        self._validate_recoveries(recovery_lk, recovery_hk)
        self._lk = light_key
        self._hk = heavy_key
        self._rec_lk = recovery_lk
        self._rec_hk = recovery_hk
        self._pressure_pa = pressure_pa
        self._t_feed = feed_temperature_k
        self._q = q
        self._r_mult = reflux_multiplier
        self._n_mult = stages_multiplier
        self._cfg = config or DEFAULT_CONFIG

        self._alpha = relative_volatility(self._lk, self._hk, self._t_feed, self._pressure_pa)
        _log.info(
            "ShortcutColumn: %s/%s, α=%.4f, P=%.2f bar, T_feed=%.1f K",
            self._lk,
            self._hk,
            self._alpha,
            self._pressure_pa / 1e5,
            self._t_feed,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def relative_volatility_lk_hk(self) -> float:
        """Relative volatility α(LK/HK) at feed conditions."""
        return self._alpha

    def solve(
        self,
        feed_flow: float,
        z_lk: float,
        z_hk: float,
    ) -> DSWTUResult:
        """Run the FUG shortcut calculation for a given feed.

        Args:
            feed_flow: Total molar feed flow rate (kmol/h or any consistent
                unit — only ratios are used internally).
            z_lk: Light-key mole fraction in feed.
            z_hk: Heavy-key mole fraction in feed.

        Returns:
            :class:`DSWTUResult` with design reflux, stages, and feed tray.

        Raises:
            ValueError: If feed compositions are invalid or feed flow ≤ 0.
        """
        if feed_flow <= 0.0:
            raise ValueError(f"feed_flow must be positive, got {feed_flow}")
        if not (0 < z_lk < 1) or not (0 < z_hk < 1):
            raise ValueError("Feed mole fractions z_lk and z_hk must each be in (0, 1).")
        if z_lk + z_hk > 1.0 + 1e-6:
            raise ValueError(
                f"z_lk + z_hk = {z_lk + z_hk:.4f} > 1.  "
                "Ensure non-key components account for the remainder."
            )

        # ── Step 1: Fenske (Nmin at total reflux) ────────────────────────────
        n_min = self._fenske(z_lk, z_hk)
        _log.debug("Fenske Nmin = %.4f", n_min)

        # ── Step 2: Underwood (Rmin) ──────────────────────────────────────────
        # Pseudo-binary basis: only LK and HK in Underwood
        alpha_arr = np.array([self._alpha, 1.0])
        z_f = np.array([z_lk, z_hk]) / (z_lk + z_hk)
        x_d = np.array([self._rec_lk * z_lk, (1 - self._rec_hk) * z_hk])
        x_d = x_d / x_d.sum()
        r_min = minimum_reflux_underwood(alpha_arr, z_f, x_d, q=self._q)
        _log.debug("Underwood Rmin = %.4f", r_min)

        # ── Step 3: Gilliland (N and Nf) ─────────────────────────────────────
        r_actual = max(r_min * self._r_mult, r_min + 0.01)
        n_actual = self._gilliland(n_min, r_min, r_actual)
        n_feed = self._kirkbride_feed_stage(n_actual, z_lk, z_hk, feed_flow)
        _log.info(
            "Column design: Rmin=%.4f R=%.4f Nmin=%.4f N=%.4f Nf=%.1f",
            r_min,
            r_actual,
            n_min,
            n_actual,
            n_feed,
        )

        return DSWTUResult(
            r_min=r_min,
            r_actual=r_actual,
            n_min=n_min,
            n_actual=n_actual,
            n_feed=n_feed,
            alpha_lk_hk=self._alpha,
            recovery_lk=self._rec_lk,
            recovery_hk=self._rec_hk,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fenske(self, z_lk: float, z_hk: float) -> float:
        """Apply Fenske equation for Nmin."""
        # Distillate and bottoms compositions in pseudo-binary basis
        x_d_lk = self._rec_lk * z_lk / (self._rec_lk * z_lk + (1.0 - self._rec_hk) * z_hk)
        x_b_lk = ((1.0 - self._rec_lk) * z_lk) / (
            (1.0 - self._rec_lk) * z_lk + self._rec_hk * z_hk
        )
        # Fenske: Nmin = log[(xd_lk/xd_hk)(xb_hk/xb_lk)] / log(α)
        x_d_hk = 1.0 - x_d_lk
        x_b_hk = 1.0 - x_b_lk
        if x_b_lk <= 0.0:
            x_b_lk = 1e-12  # guard against log(0)
        sep_factor = (x_d_lk / x_d_hk) * (x_b_hk / x_b_lk)
        return math.log(sep_factor) / math.log(self._alpha)

    def _gilliland(self, n_min: float, r_min: float, r: float) -> float:
        """Apply the Gilliland correlation to estimate N from Nmin and R.

        Uses the Molokanov (1972) fit to the Gilliland curve:
            Y = 1 − exp[(1+54.4X)/(11+117.2X) · (X−1)/X^0.5]
        where X = (R − Rmin)/(R + 1) and Y = (N − Nmin)/(N + 1).
        """
        x_g = (r - r_min) / (r + 1.0)
        # Molokanov equation
        y_g = 1.0 - math.exp((1.0 + 54.4 * x_g) / (11.0 + 117.2 * x_g) * (x_g - 1.0) / (x_g**0.5))
        n = (y_g + n_min) / (1.0 - y_g)
        return n

    def _kirkbride_feed_stage(
        self,
        n_actual: float,
        z_lk: float,
        z_hk: float,
        feed_flow: float,
    ) -> float:
        """Estimate feed-stage location via the Kirkbride equation."""
        # Simplified: feed stage ≈ 0.5 × N for near-symmetric columns,
        # refined by the Kirkbride ratio.
        # Kirkbride: log(Nr/Ns) = 0.206·log[(B/D)·(z_hk/z_lk)·(x_b_lk/x_d_hk)²]
        # Use equal-molar approximation for quick estimate
        nr_ns_ratio = (z_hk / z_lk) ** 0.5  # simplified
        n_rectifying = n_actual / (1.0 + 1.0 / nr_ns_ratio)
        return float(round(n_rectifying, 1))

    @staticmethod
    def _validate_recoveries(rec_lk: float, rec_hk: float) -> None:
        """Validate recovery specifications."""
        for name, val in [("recovery_lk", rec_lk), ("recovery_hk", rec_hk)]:
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} must be in the open interval (0, 1), got {val}.")
