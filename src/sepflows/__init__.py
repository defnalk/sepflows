"""sepflows — Separation Process Simulation Toolkit.

A Python library for the design and analysis of industrial separation
sequences, with a focus on methanol purification, CO₂ capture, and
cryogenic air separation.  The API is modelled after scientific computing
packages such as *scipy* and *pvlib*, favouring composable dataclass
results, full type hints, and transparent access to intermediate values.

Quickstart
----------
>>> from sepflows.flash import FlashDrum
>>> from sepflows.distillation import ShortcutColumn
>>> from sepflows.absorption import AmineAbsorber
>>> from sepflows.asu import CryogenicASU

Flash drum
~~~~~~~~~~
>>> import numpy as np
>>> drum = FlashDrum(
...     components=["methanol", "water", "dme"],
...     temperature_k=320.0,
...     pressure_pa=2e5,
... )
>>> res = drum.solve(np.array([0.60, 0.30, 0.10]))

Shortcut distillation column
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> col = ShortcutColumn(
...     light_key="methanol",
...     heavy_key="water",
...     recovery_lk=0.995,
...     recovery_hk=0.995,
...     pressure_pa=101_325.0,
...     feed_temperature_k=337.0,
... )
>>> design = col.solve(feed_flow=65_000.0, z_lk=0.60, z_hk=0.35)
"""

from __future__ import annotations

from sepflows._version import __version__
from sepflows.absorption import AmineAbsorber, CO2CaptureResult
from sepflows.asu import ASUResult, CryogenicASU
from sepflows.config import DEFAULT_CONFIG, SepConfig, configure_logging
from sepflows.distillation import (
    DSWTUResult,
    RigorousColumn,
    RigorousColumnResult,
    ShortcutColumn,
)
from sepflows.flash import FlashDrum, FlashDrumResult

__all__ = [
    # Version
    "__version__",
    # Config
    "SepConfig",
    "DEFAULT_CONFIG",
    "configure_logging",
    # Flash
    "FlashDrum",
    "FlashDrumResult",
    # Distillation
    "ShortcutColumn",
    "DSWTUResult",
    "RigorousColumn",
    "RigorousColumnResult",
    # Absorption
    "AmineAbsorber",
    "CO2CaptureResult",
    # ASU
    "CryogenicASU",
    "ASUResult",
]
