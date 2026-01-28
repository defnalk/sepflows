"""Physical and process constants for the sepflows package.

All constants are SI units unless explicitly noted.  No magic numbers
should appear anywhere else in the codebase — import from here instead.
"""

from __future__ import annotations

__all__ = [
    "R_GAS",
    "T_REF",
    "P_ATM",
    "ANTOINE",
    "MOLECULAR_WEIGHTS",
    "NORMAL_BOILING_POINTS",
    "CRITICAL_PROPERTIES",
    "ASU_COMPOSITIONS",
    "CO2_CAPTURE_DEFAULTS",
    "DISTILLATION_DEFAULTS",
]

# ── Fundamental constants ─────────────────────────────────────────────────────

R_GAS: float = 8.314  # J mol⁻¹ K⁻¹  (universal gas constant)
T_REF: float = 298.15  # K            (standard reference temperature)
P_ATM: float = 101_325.0  # Pa        (standard atmosphere)

# ── Antoine equation coefficients (log₁₀(P/mmHg) = A − B/(C+T/°C)) ──────────
# Source: Perry's Chemical Engineers' Handbook, 9th ed., Table 2-8
ANTOINE: dict[str, dict[str, float]] = {
    "methanol": {"A": 8.08097, "B": 1582.271, "C": 239.726},
    "water": {"A": 8.07131, "B": 1730.63, "C": 233.426},
    "ethanol": {"A": 8.20417, "B": 1642.89, "C": 230.300},
    "dme": {"A": 7.18936, "B": 983.42, "C": 221.600},  # dimethyl ether
    "methyl_formate": {"A": 7.53050, "B": 1243.00, "C": 224.000},
    "1_propanol": {"A": 7.99733, "B": 1569.70, "C": 209.530},
    "1_butanol": {"A": 7.82483, "B": 1517.84, "C": 186.700},
    "nitrogen": {"A": 6.49457, "B": 255.68, "C": 266.550},
    "oxygen": {"A": 6.69147, "B": 319.01, "C": 266.700},
    "argon": {"A": 6.62223, "B": 304.01, "C": 269.870},
    "co2": {"A": 7.58828, "B": 863.00, "C": 230.000},
}

# ── Molecular weights (g mol⁻¹) ──────────────────────────────────────────────
MOLECULAR_WEIGHTS: dict[str, float] = {
    "methanol": 32.042,
    "water": 18.015,
    "ethanol": 46.069,
    "dme": 46.068,
    "methyl_formate": 60.052,
    "1_propanol": 60.096,
    "1_butanol": 74.122,
    "nitrogen": 28.014,
    "oxygen": 31.999,
    "argon": 39.948,
    "co2": 44.010,
    "hydrogen": 2.016,
    "co": 28.010,
}

# ── Normal boiling points (K, 1 atm) ─────────────────────────────────────────
NORMAL_BOILING_POINTS: dict[str, float] = {
    "methanol": 337.85,
    "water": 373.15,
    "ethanol": 351.44,
    "dme": 248.31,
    "methyl_formate": 304.85,
    "1_propanol": 370.35,
    "1_butanol": 390.88,
    "nitrogen": 77.36,
    "oxygen": 90.19,
    "argon": 87.30,
    "co2": 194.65,  # sublimation point (solid↔gas at 1 atm)
}

# ── Critical properties (Tc/K, Pc/bar, ω acentric factor) ────────────────────
CRITICAL_PROPERTIES: dict[str, dict[str, float]] = {
    "methanol": {"Tc": 512.64, "Pc": 80.97, "omega": 0.5625},
    "water": {"Tc": 647.10, "Pc": 220.64, "omega": 0.3449},
    "ethanol": {"Tc": 514.00, "Pc": 61.37, "omega": 0.6436},
    "dme": {"Tc": 400.10, "Pc": 53.70, "omega": 0.2000},
    "methyl_formate": {"Tc": 487.20, "Pc": 60.00, "omega": 0.2570},
    "1_propanol": {"Tc": 536.78, "Pc": 51.75, "omega": 0.6268},
    "1_butanol": {"Tc": 562.95, "Pc": 44.14, "omega": 0.5939},
    "nitrogen": {"Tc": 126.21, "Pc": 33.96, "omega": 0.0372},
    "oxygen": {"Tc": 154.58, "Pc": 50.43, "omega": 0.0222},
    "argon": {"Tc": 150.86, "Pc": 48.98, "omega": -0.0022},
    "co2": {"Tc": 304.21, "Pc": 73.83, "omega": 0.2236},
}

# ── Dry air composition (mole fractions) ─────────────────────────────────────
ASU_COMPOSITIONS: dict[str, dict[str, float]] = {
    "dry_air": {
        "nitrogen": 0.7812,
        "oxygen": 0.2096,
        "argon": 0.0092,
    },
    "product_n2": {  # high-purity nitrogen specification
        "nitrogen": 0.9999,
        "oxygen": 0.0001,
    },
    "product_o2": {  # high-purity oxygen specification
        "oxygen": 0.9950,
        "nitrogen": 0.0040,
        "argon": 0.0010,
    },
}

# ── CO₂ capture process defaults ─────────────────────────────────────────────
CO2_CAPTURE_DEFAULTS: dict[str, float] = {
    "lean_loading": 0.20,       # mol CO₂ / mol amine  (typical MEA lean)
    "rich_loading": 0.48,       # mol CO₂ / mol amine  (typical MEA rich)
    "amine_concentration": 30.0,  # wt %  (standard MEA aqueous solution)
    "stripper_pressure_bar": 1.8,  # bar  (regenerator operating pressure)
    "absorber_pressure_bar": 1.1,  # bar
    "reboiler_duty_gj_t": 3.7,  # GJ / t CO₂  (industry benchmark MEA)
    "co2_removal_target": 0.90,  # fraction of inlet CO₂ absorbed
}

# ── Distillation column design defaults ──────────────────────────────────────
DISTILLATION_DEFAULTS: dict[str, float] = {
    "reflux_ratio_multiplier": 1.30,  # R = 1.30 × Rmin  (Kister heuristic)
    "stages_multiplier": 2.00,        # N = 2.00 × Nmin
    "tray_efficiency": 0.70,          # overall Murphree efficiency
    "pressure_drop_per_tray_kpa": 0.7,  # kPa per tray (rule of thumb)
    "methanol_recovery_target": 0.89,  # CID-derived: B = 8 → 89 %
    "feed_basis_kmol_h": 65_000.0,    # A = 26 → F = 26 × 2500
}
