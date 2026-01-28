"""Runtime configuration for sepflows.

Provides a single :class:`SepConfig` dataclass that aggregates solver
tolerances, logging verbosity, and unit-system preferences.  Instantiate
once and pass it through your simulation hierarchy, or rely on the
module-level :data:`DEFAULT_CONFIG` singleton for quick scripts.

Example
-------
>>> from sepflows.config import SepConfig, DEFAULT_CONFIG
>>> cfg = SepConfig(max_iterations=500, convergence_tol=1e-7)
>>> cfg.max_iterations
500
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

__all__ = ["SepConfig", "DEFAULT_CONFIG", "configure_logging"]


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the package-level logger.

    Args:
        level: Python logging level (e.g. ``logging.DEBUG``).

    Returns:
        Configured ``sepflows`` logger instance.
    """
    logger = logging.getLogger("sepflows")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@dataclass
class SepConfig:
    """Global solver and unit configuration.

    Attributes:
        max_iterations: Maximum iterations for iterative solvers (e.g.
            Rachford-Rice, MESH equations).
        convergence_tol: Absolute convergence tolerance for mole-fraction
            residuals.
        pressure_unit: Preferred pressure unit for display (``"bar"`` or
            ``"kPa"`` or ``"Pa"``).
        temperature_unit: Preferred temperature unit for display
            (``"K"`` or ``"C"``).
        energy_unit: Preferred energy unit for duty display
            (``"MW"`` or ``"GJ_h"``).
        verbose: Emit DEBUG-level solver iteration logs when ``True``.
        flash_vf_bracket: ``(lo, hi)`` vapour fraction bracket for the
            Rachford-Rice root search.  Must satisfy ``0 < lo < hi < 1``.
        eos_model: Equation of state for K-value estimation.  Currently
            ``"raoult"`` (ideal, Antoine-based) or ``"srk"`` (Soave-RK,
            future).
    """

    max_iterations: int = 200
    convergence_tol: float = 1.0e-8
    pressure_unit: str = "bar"
    temperature_unit: str = "K"
    energy_unit: str = "MW"
    verbose: bool = False
    flash_vf_bracket: tuple[float, float] = field(default_factory=lambda: (1e-6, 1 - 1e-6))
    eos_model: str = "raoult"

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be ≥ 1, got {self.max_iterations}"
            )
        if not (0.0 < self.convergence_tol < 1.0):
            raise ValueError(
                f"convergence_tol must be in (0, 1), got {self.convergence_tol}"
            )
        if self.pressure_unit not in {"Pa", "kPa", "bar"}:
            raise ValueError(
                f"pressure_unit must be 'Pa', 'kPa', or 'bar', got '{self.pressure_unit}'"
            )
        if self.temperature_unit not in {"K", "C"}:
            raise ValueError(
                f"temperature_unit must be 'K' or 'C', got '{self.temperature_unit}'"
            )
        lo, hi = self.flash_vf_bracket
        if not (0 < lo < hi < 1):
            raise ValueError(
                f"flash_vf_bracket must satisfy 0 < lo < hi < 1, got ({lo}, {hi})"
            )
        if self.eos_model not in {"raoult", "srk"}:
            raise ValueError(
                f"eos_model must be 'raoult' or 'srk', got '{self.eos_model}'"
            )


#: Module-level default configuration used when no explicit config is supplied.
DEFAULT_CONFIG: SepConfig = SepConfig()
