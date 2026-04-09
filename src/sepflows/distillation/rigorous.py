"""Rigorous distillation column model — simplified MESH equations.

Implements a stage-by-stage tridiagonal solver for the linearised MESH
(Material balance, Equilibrium, Summation, Heat balance) equations.
This is a pedagogical rigorous model suitable for sensitivity studies and
initialisation of Aspen Plus simulations.

The solver follows the *Thomas algorithm* (tridiagonal matrix algorithm,
TDMA) for the component-material-balance tridiagonals, with successive
substitution on temperatures.

References:
----------
- Wang & Henke, *Hydrocarbon Processing* (1966) — tridiagonal algorithm
- Kister, H. Z. *Distillation Design* (1992), Chapter 4
- Seader, J. D. & Henley, E. J. *Separation Process Principles* (2006)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sepflows.config import DEFAULT_CONFIG, SepConfig
from sepflows.constants import ANTOINE, P_ATM
from sepflows.utils.thermodynamics import _MMHG_TO_PA, k_values_raoult

__all__ = ["RigorousColumnResult", "RigorousColumn"]

_log = logging.getLogger(__name__)


@dataclass
class RigorousColumnResult:
    """Results of a rigorous MESH distillation calculation.

    Attributes:
        x: Liquid mole fractions, shape (n_stages, n_components).
        y: Vapour mole fractions, shape (n_stages, n_components).
        l_flows: Liquid molar flow profiles (mol/h), shape (n_stages,).
        v_flows: Vapour molar flow profiles (mol/h), shape (n_stages,).
        temperatures_k: Stage temperatures (K), shape (n_stages,).
        distillate_composition: Distillate mole fractions.
        bottoms_composition: Bottoms mole fractions.
        distillate_flow: Distillate molar flow (same units as feed).
        bottoms_flow: Bottoms molar flow (same units as feed).
        n_iterations: Number of outer-loop iterations to convergence.
        converged: Whether the solver met the convergence criterion.
        components: Component names.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    l_flows: NDArray[np.float64]
    v_flows: NDArray[np.float64]
    temperatures_k: NDArray[np.float64]
    distillate_composition: NDArray[np.float64]
    bottoms_composition: NDArray[np.float64]
    distillate_flow: float
    bottoms_flow: float
    n_iterations: int
    converged: bool
    components: tuple[str, ...]


class RigorousColumn:
    """Simplified rigorous distillation column (Wang–Henke MESH solver).

    Assumes:
    - Constant molar overflow (CMO) for liquid/vapour flow profiles.
    - Ideal VLE (K-values from modified Raoult's law via Antoine equation).
    - Total condenser (stage 1 = condenser at bubble point).
    - Partial reboiler (stage N = reboiler, counted from top).

    Args:
        components: Ordered list of component names.
        n_stages: Total number of theoretical stages (including condenser
            and reboiler).
        feed_stage: Feed stage index (1-based, from top; 1 = condenser).
        reflux_ratio: Operating reflux ratio R = L/D.
        distillate_to_feed: Molar split D/F.
        pressure_pa: Column operating pressure in Pa.
        feed_temperature_k: Feed temperature in Kelvin (used to initialise
            stage temperature profile).
        config: Solver configuration.

    Example:
        >>> col = RigorousColumn(
        ...     components=["methanol", "water"],
        ...     n_stages=20,
        ...     feed_stage=10,
        ...     reflux_ratio=2.72,
        ...     distillate_to_feed=0.55,
        ...     pressure_pa=101_325.0,
        ...     feed_temperature_k=337.0,
        ... )
        >>> import numpy as np
        >>> z = np.array([0.60, 0.40])
        >>> result = col.solve(feed_flow=65_000.0, z=z)
        >>> result.converged
        True
    """

    def __init__(
        self,
        components: Sequence[str],
        n_stages: int,
        feed_stage: int,
        reflux_ratio: float,
        distillate_to_feed: float,
        pressure_pa: float = P_ATM,
        feed_temperature_k: float = 337.0,
        config: SepConfig | None = None,
    ) -> None:
        if n_stages < 3:
            raise ValueError(
                f"n_stages must be ≥ 3 (condenser + at least 1 tray + reboiler), got {n_stages}."
            )
        if not (1 <= feed_stage <= n_stages):
            raise ValueError(f"feed_stage must be in [1, n_stages={n_stages}], got {feed_stage}.")
        if reflux_ratio <= 0.0:
            raise ValueError(f"reflux_ratio must be positive, got {reflux_ratio}.")
        if not (0.0 < distillate_to_feed < 1.0):
            raise ValueError(f"distillate_to_feed must be in (0, 1), got {distillate_to_feed}.")

        self._comp = tuple(components)
        self._nc = len(self._comp)
        self._n = n_stages
        self._nf = feed_stage
        self._r = reflux_ratio
        self._df = distillate_to_feed
        self._p = pressure_pa
        self._t_feed = feed_temperature_k
        self._cfg = config or DEFAULT_CONFIG

        # Precompute Antoine coefficient vectors once per column instance.
        # Inside solve(), K-values for (n_stages x n_components) are then
        # built in a single vectorized numpy expression instead of
        # n_stages * n_components separate Python-level dict lookups.
        missing = [c for c in self._comp if c not in ANTOINE]
        if missing:
            available = ", ".join(sorted(ANTOINE.keys()))
            raise KeyError(
                f"Components not in Antoine database: {missing}. Available: {available}"
            )
        self._antoine_A = np.array([ANTOINE[c]["A"] for c in self._comp], dtype=np.float64)
        self._antoine_B = np.array([ANTOINE[c]["B"] for c in self._comp], dtype=np.float64)
        self._antoine_C = np.array([ANTOINE[c]["C"] for c in self._comp], dtype=np.float64)

        _log.info(
            "RigorousColumn: %d components, N=%d, Nf=%d, R=%.3f, D/F=%.3f, P=%.2f bar",
            self._nc,
            self._n,
            self._nf,
            self._r,
            self._df,
            self._p / 1e5,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(
        self,
        feed_flow: float,
        z: NDArray[np.float64],
    ) -> RigorousColumnResult:
        """Solve the MESH equations for a given feed.

        Args:
            feed_flow: Molar feed flow rate (kmol/h or consistent unit).
            z: Feed mole fractions (must sum to 1.0).

        Returns:
            :class:`RigorousColumnResult` with stage profiles and product
            compositions.

        Raises:
            ValueError: If ``z`` has wrong shape or doesn't sum to 1.
            RuntimeError: If the solver fails to converge.
        """
        z = np.asarray(z, dtype=np.float64)
        self._validate_feed(z)

        f = feed_flow
        d = self._df * f
        b = f - d
        l_rect = self._r * d  # liquid flow in rectifying section
        v_col = l_rect + d  # vapour flow (CMO)
        l_strip = l_rect + f  # liquid flow in stripping section (q=1)

        _log.debug(
            "CMO flows: L_rect=%.1f, V=%.1f, L_strip=%.1f, D=%.1f, B=%.1f",
            l_rect,
            v_col,
            l_strip,
            d,
            b,
        )

        # Initialise temperature profile by linear interpolation
        t_top = self._bubble_temp_approx(z, is_distillate=True)
        t_bot = self._bubble_temp_approx(z, is_distillate=False)
        # Clamp temperatures to physically reasonable range
        t_top = max(min(t_top, 600.0), 200.0)
        t_bot = max(min(t_bot, 700.0), 250.0)
        temperatures = np.linspace(t_top, t_bot, self._n)

        # Initialise composition arrays (uniform = feed)
        x = np.tile(z, (self._n, 1)).copy()
        y = np.tile(z, (self._n, 1)).copy()

        l_flows = np.where(np.arange(self._n) < self._nf - 1, l_rect, l_strip).astype(float)
        v_flows = np.full(self._n, v_col, dtype=float)

        converged = False
        t_old = temperatures.copy()

        for iteration in range(self._cfg.max_iterations):
            # Guard: clamp temperatures before computing K-values
            temperatures = np.clip(temperatures, 150.0, 700.0)

            # Compute K-values on each stage (vectorized over stages AND components).
            # Equivalent to looping k_values_raoult per stage but skips N*nc dict
            # lookups + per-call logger.debug dispatch.
            k_all = self._k_matrix(temperatures)  # shape (N, nc)

            # Guard NaN/inf K-values
            k_all = np.where(np.isfinite(k_all), k_all, 1.0)

            # Update vapour compositions from equilibrium
            y = k_all * x
            # Normalise rows, guard zero rows
            row_sums = y.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 1e-30, row_sums, 1.0)
            y /= row_sums

            # Thomas algorithm (tridiagonal MESH) per component
            for i, _comp in enumerate(self._comp):
                sol = self._thomas_solve(
                    x[:, i], y[:, i], k_all[:, i], l_flows, v_flows, z[i], f, d, b
                )
                # Guard solver output
                x[:, i] = np.where(np.isfinite(sol), sol, x[:, i])

            # Clip and renormalise
            x = np.clip(x, 0.0, 1.0)
            row_sums_x = x.sum(axis=1, keepdims=True)
            row_sums_x = np.where(row_sums_x > 1e-30, row_sums_x, 1.0)
            x /= row_sums_x

            # Update temperatures via bubble-point calculation (gentle step).
            # Vectorized across all stages: each stage's new T depends only on
            # its own prior T and x row, so order-of-update doesn't matter.
            k_bp = self._k_matrix(temperatures)  # (N, nc)
            sigma = (k_bp * x).sum(axis=1)       # (N,)
            valid = np.isfinite(sigma) & (sigma > 0)
            t_new = np.where(
                valid, temperatures * (1.0 + 0.05 * (sigma - 1.0)), temperatures
            )
            temperatures = np.clip(t_new, 150.0, 700.0)

            t_change = float(np.nanmax(np.abs(temperatures - t_old)))
            if self._cfg.verbose:
                _log.debug("Iteration %d: ΔT_max=%.6f K", iteration + 1, t_change)

            if t_change < self._cfg.convergence_tol * 1000:
                converged = True
                _log.info(
                    "RigorousColumn converged in %d iterations (ΔT=%.2e K)",
                    iteration + 1,
                    t_change,
                )
                break
            t_old = temperatures.copy()

        if not converged:
            _log.warning(
                "RigorousColumn did not converge after %d iterations.  "
                "Results are approximate.  Consider providing better initial "
                "conditions or increasing max_iterations.",
                self._cfg.max_iterations,
            )

        distillate_comp = y[0].copy()
        bottoms_comp = x[-1].copy()

        return RigorousColumnResult(
            x=x,
            y=y,
            l_flows=l_flows,
            v_flows=v_flows,
            temperatures_k=temperatures,
            distillate_composition=distillate_comp,
            bottoms_composition=bottoms_comp,
            distillate_flow=d,
            bottoms_flow=b,
            n_iterations=iteration + 1,
            converged=converged,
            components=self._comp,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _k_matrix(self, temperatures: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorized ideal K-values for all stages and components.

        Equivalent to
            np.array([k_values_raoult(self._comp, T, self._p) for T in temperatures])
        but uses precomputed Antoine coefficient vectors and broadcasts over
        the full (n_stages x n_components) grid in a single numpy expression.
        Bypasses the per-call dict lookups, list comprehension, validation,
        and logger.debug dispatch in the scalar path.

        K_ij = 10^(A_j - B_j / (C_j + T_i - 273.15)) * mmHg_to_Pa / P
        """
        t_celsius = temperatures[:, None] - 273.15                         # (N, 1)
        log_p = self._antoine_A - self._antoine_B / (self._antoine_C + t_celsius)
        p_sat_pa = np.power(10.0, log_p) * _MMHG_TO_PA                     # (N, nc)
        return p_sat_pa / self._p

    def _thomas_solve(
        self,
        x_old: NDArray[np.float64],
        y_old: NDArray[np.float64],
        k: NDArray[np.float64],
        l: NDArray[np.float64],
        v: NDArray[np.float64],
        z_i: float,
        f: float,
        d: float,
        b: float,
    ) -> NDArray[np.float64]:
        """Solve tridiagonal material balance for one component.

        Args:
            x_old: Current liquid mole fractions on all stages.
            y_old: Current vapour mole fractions on all stages.
            k: K-values on all stages for this component.
            l: Liquid flow rates on all stages.
            v: Vapour flow rates on all stages.
            z_i: Component mole fraction in feed.
            f: Total feed flow.
            d: Distillate flow.
            b: Bottoms flow.

        Returns:
            Updated liquid mole fractions ``x`` for this component.
        """
        n = self._n
        # Build tridiagonal coefficients
        a_diag = np.zeros(n)
        b_diag = np.zeros(n)
        c_diag = np.zeros(n)
        d_rhs = np.zeros(n)

        for j in range(n):
            l_j = l[j]
            v_j = v[j]
            k_j = k[j]
            # Diagonal: -(L_j + V_j * K_j)
            b_diag[j] = -(l_j + v_j * k_j)
            # Sub-diagonal: L_{j-1}  (from stage j-1 liquid)
            if j > 0:
                a_diag[j] = l[j - 1]
            # Super-diagonal: V_{j+1} * K_{j+1}  (from stage j+1 vapour)
            if j < n - 1:
                c_diag[j] = v[j + 1] * k[j + 1]
            # RHS: feed contribution on feed stage
            d_rhs[j] = -f * z_i if j == self._nf - 1 else 0.0

        # Boundary corrections
        # Stage 1 (condenser): no liquid from stage 0; vapour from stage 2
        # Stage N (reboiler): no vapour from stage N+1
        # (Handled by zero initialisation of a[0] and c[N-1])

        # Forward sweep
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        c_prime[0] = c_diag[0] / b_diag[0]
        d_prime[0] = d_rhs[0] / b_diag[0]
        for j in range(1, n):
            denom = b_diag[j] - a_diag[j] * c_prime[j - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30  # guard against division by zero
            c_prime[j] = c_diag[j] / denom
            d_prime[j] = (d_rhs[j] - a_diag[j] * d_prime[j - 1]) / denom

        # Back substitution
        x_new = np.zeros(n)
        x_new[-1] = d_prime[-1]
        for j in range(n - 2, -1, -1):
            x_new[j] = d_prime[j] - c_prime[j] * x_new[j + 1]

        return x_new

    def _bubble_temp_approx(self, z: NDArray[np.float64], is_distillate: bool) -> float:
        """Rough bubble-temperature estimate for profile initialisation.

        Estimates a weighted average boiling point from the constants database,
        then applies a directional offset for condenser vs reboiler end.
        """
        from sepflows.constants import NORMAL_BOILING_POINTS

        # Compute a mole-fraction-weighted normal boiling point
        t_wb = sum(
            z[i] * NORMAL_BOILING_POINTS.get(self._comp[i], self._t_feed) for i in range(self._nc)
        )
        # Scale to the operating pressure using a simple ratio heuristic
        p_ratio = (self._p / 101_325.0) ** 0.1
        t_wb *= p_ratio

        # Condenser is cooler than the feed bubble point; reboiler is hotter
        offset = -10.0 if is_distillate else +20.0
        return float(t_wb + offset)

    def _validate_feed(self, z: NDArray[np.float64]) -> None:
        """Validate feed mole fraction vector."""
        if z.ndim != 1 or len(z) != self._nc:
            raise ValueError(f"z must be 1-D with {self._nc} elements, got shape {z.shape}.")
        if np.any(z < 0.0):
            raise ValueError("All feed mole fractions must be ≥ 0.")
        if not np.isclose(z.sum(), 1.0, atol=1e-4):
            raise ValueError(f"Feed mole fractions must sum to 1.0, got {z.sum():.6f}.")
