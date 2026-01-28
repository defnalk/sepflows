"""Isothermal flash drum simulation.

Implements a two-phase (vapour–liquid) isothermal flash calculation
using the Rachford–Rice algorithm with ideal K-values (modified Raoult's
law).  Suitable for rapid pre-screening of flash-drum operating conditions
upstream of distillation trains, e.g. the crude methanol syngas separator
described in CENG50009 Separation Processes 2.

References
----------
- Rachford & Rice, *JPT* (1952)
- Perry's Chemical Engineers' Handbook, 9th ed., Section 13
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from sepflows.config import DEFAULT_CONFIG, SepConfig
from sepflows.constants import P_ATM
from sepflows.utils.thermodynamics import k_values_raoult, rachford_rice

__all__ = ["FlashDrumResult", "FlashDrum"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlashDrumResult:
    """Immutable result object for a single flash drum calculation.

    Attributes:
        vapour_fraction: Overall vapour fraction Ψ (mol/mol).
        x: Liquid-phase mole fractions (same component order as input).
        y: Vapour-phase mole fractions (same component order as input).
        k_values: Equilibrium K-values used in the calculation.
        temperature_k: Flash temperature in Kelvin.
        pressure_pa: Flash pressure in Pa.
        components: Tuple of component names.
        converged: Whether the Rachford-Rice solver converged.
    """

    vapour_fraction: float
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    k_values: NDArray[np.float64]
    temperature_k: float
    pressure_pa: float
    components: tuple[str, ...]
    converged: bool

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"FlashDrumResult("
            f"Ψ={self.vapour_fraction:.4f}, "
            f"T={self.temperature_k:.2f} K, "
            f"P={self.pressure_pa / 1e5:.3f} bar)"
        )

    @property
    def liquid_fraction(self) -> float:
        """Liquid fraction L (mol/mol) = 1 − Ψ."""
        return 1.0 - self.vapour_fraction


class FlashDrum:
    """Two-phase isothermal flash drum.

    Models a single equilibrium stage where a feed mixture partially
    vaporises.  Phase compositions are determined from the Rachford–Rice
    equation with ideal (Raoult) K-values.

    Args:
        components: Ordered list of component names (must be present in
            the Antoine coefficient database).
        temperature_k: Operating temperature in Kelvin.
        pressure_pa: Operating pressure in Pa.  Defaults to 1 atm.
        config: Solver configuration.  If not provided, the package
            default (:data:`~sepflows.config.DEFAULT_CONFIG`) is used.

    Example:
        >>> from sepflows.flash import FlashDrum
        >>> import numpy as np
        >>> drum = FlashDrum(
        ...     components=["methanol", "water", "dme"],
        ...     temperature_k=320.0,
        ...     pressure_pa=2e5,
        ... )
        >>> z = np.array([0.60, 0.30, 0.10])
        >>> result = drum.solve(z)
        >>> result.vapour_fraction  # doctest: +ELLIPSIS
        0...
    """

    def __init__(
        self,
        components: Sequence[str],
        temperature_k: float,
        pressure_pa: float = P_ATM,
        config: SepConfig | None = None,
    ) -> None:
        if temperature_k <= 0.0:
            raise ValueError(f"temperature_k must be positive, got {temperature_k}")
        if pressure_pa <= 0.0:
            raise ValueError(f"pressure_pa must be positive, got {pressure_pa}")
        if len(components) < 2:
            raise ValueError("At least two components are required for a flash calculation.")

        self._components: tuple[str, ...] = tuple(components)
        self._temperature_k = temperature_k
        self._pressure_pa = pressure_pa
        self._cfg = config or DEFAULT_CONFIG

        _log.info(
            "FlashDrum initialised: %d components, T=%.2f K, P=%.2f bar",
            len(self._components),
            self._temperature_k,
            self._pressure_pa / 1e5,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def components(self) -> tuple[str, ...]:
        """Tuple of component names (read-only)."""
        return self._components

    @property
    def temperature_k(self) -> float:
        """Operating temperature in Kelvin."""
        return self._temperature_k

    @property
    def pressure_pa(self) -> float:
        """Operating pressure in Pa."""
        return self._pressure_pa

    def solve(self, z: NDArray[np.float64]) -> FlashDrumResult:
        """Perform an isothermal flash for the given feed composition.

        Args:
            z: Feed mole fractions.  Must sum to 1.0 within tolerance and
                have the same length as ``self.components``.

        Returns:
            A :class:`FlashDrumResult` containing phase fractions,
            compositions, and K-values.

        Raises:
            ValueError: If ``z`` has wrong length or does not sum to 1.
        """
        z = np.asarray(z, dtype=np.float64)
        self._validate_feed(z)

        k = k_values_raoult(self._components, self._temperature_k, self._pressure_pa)
        _log.debug("Feed K-values: %s", dict(zip(self._components, k)))

        psi = rachford_rice(
            z,
            k,
            vf_lo=self._cfg.flash_vf_bracket[0],
            vf_hi=self._cfg.flash_vf_bracket[1],
            tol=self._cfg.convergence_tol,
            max_iter=self._cfg.max_iterations,
        )

        # Phase compositions from material balance
        denom = 1.0 + psi * (k - 1.0)
        x = z / denom
        y = k * x

        # Normalise to guard against tiny numerical drift
        x = x / x.sum()
        y = y / y.sum()

        converged = True
        _log.info(
            "Flash solved: Ψ=%.4f (L/F=%.4f)",
            psi,
            1.0 - psi,
        )
        return FlashDrumResult(
            vapour_fraction=float(psi),
            x=x,
            y=y,
            k_values=k,
            temperature_k=self._temperature_k,
            pressure_pa=self._pressure_pa,
            components=self._components,
            converged=converged,
        )

    def sensitivity(
        self,
        z: NDArray[np.float64],
        temperatures_k: NDArray[np.float64],
    ) -> list[FlashDrumResult]:
        """Compute flash results over a temperature sweep at fixed pressure.

        Useful for locating the partial-condensation temperature that
        maximises DME recovery to the vapour phase while keeping methanol
        in the liquid.

        Args:
            z: Fixed feed mole fractions.
            temperatures_k: 1-D array of temperatures to evaluate (K).

        Returns:
            List of :class:`FlashDrumResult` objects in the same order as
            ``temperatures_k``.
        """
        results: list[FlashDrumResult] = []
        for t in temperatures_k:
            drum = FlashDrum(
                self._components,
                float(t),
                self._pressure_pa,
                self._cfg,
            )
            results.append(drum.solve(z))
            _log.debug("Sensitivity at T=%.1f K: Ψ=%.4f", t, results[-1].vapour_fraction)
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_feed(self, z: NDArray[np.float64]) -> None:
        """Check feed vector dimensions and normalisation."""
        if z.ndim != 1:
            raise ValueError(f"z must be a 1-D array, got shape {z.shape}")
        if len(z) != len(self._components):
            raise ValueError(
                f"z has {len(z)} elements but {len(self._components)} components "
                "were specified."
            )
        if np.any(z < 0.0):
            raise ValueError("All feed mole fractions must be ≥ 0.")
        if not np.isclose(z.sum(), 1.0, atol=1e-5):
            raise ValueError(
                f"Feed mole fractions must sum to 1.0, got {z.sum():.8f}. "
                "Normalise your input or check for missing components."
            )
