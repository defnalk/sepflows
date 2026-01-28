"""Distillation sub-package.

Provides shortcut (FUG / DSTWU) and simplified rigorous (MESH) column models.
"""

from __future__ import annotations

from sepflows.distillation.rigorous import RigorousColumn, RigorousColumnResult
from sepflows.distillation.shortcut import DSWTUResult, ShortcutColumn

__all__ = [
    "DSWTUResult",
    "ShortcutColumn",
    "RigorousColumn",
    "RigorousColumnResult",
]
