"""CENG50009 Integrated Methanol Purification Flowsheet — worked example.

Reproduces the complete separation sequence from the coursework:

    Flash drum (280 K, 20 bar)
        ↓ liquid phase
    CO₂ absorber (MEA, 90 % removal from recycle gas)
        ↓ lean recycle gas returned to synthesis loop
    Methyl-formate / methanol pre-cut column  (DSTWU)
        ↓ methanol-rich bottoms
    Methanol polishing column  (rigorous MESH, 20 stages)
        ↓ high-purity methanol distillate
    Ethanol / higher-alcohol separation  (DSTWU)
        ↓
    Air Separation Unit  (cryogenic, N₂ 99.99 %, O₂ 99.5 %)

Run with::

    python examples/methanol_purification_flowsheet.py
"""

from __future__ import annotations

import logging

import numpy as np

import sepflows
from sepflows.absorption import AmineAbsorber
from sepflows.asu import CryogenicASU
from sepflows.config import SepConfig, configure_logging
from sepflows.distillation import RigorousColumn, ShortcutColumn
from sepflows.flash import FlashDrum

# ── Logging setup ──────────────────────────────────────────────────────────────
configure_logging(logging.INFO)
log = logging.getLogger("example.flowsheet")

# ── Solver configuration ───────────────────────────────────────────────────────
cfg = SepConfig(max_iterations=300, convergence_tol=1e-8)

# ── Basis of design (CENG50009, CID 02390327) ─────────────────────────────────
# A = 26  →  F = 26 × 2500 = 65 000 kmol/h
# B = 8   →  methanol recovery target C = 89 %
FEED_FLOW = 65_000.0  # kmol/h

CRUDE_COMPONENTS = [
    "dme",          # dimethyl ether   — lightest, to flash vapour
    "methyl_formate",
    "methanol",     # primary product
    "ethanol",
    "water",
    "1_propanol",
    "1_butanol",    # heaviest oxygenate
]

# Approximate feed composition (mole fractions, sums to 1.0)
Z_CRUDE = np.array([0.02, 0.03, 0.65, 0.04, 0.22, 0.02, 0.02])

sep = "═" * 70


def header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Flash drum
# ══════════════════════════════════════════════════════════════════════════════
header("STEP 1 — Flash Drum  (280 K, 20 bar)")

drum = FlashDrum(CRUDE_COMPONENTS, temperature_k=280.0, pressure_pa=20.0e5, config=cfg)
flash = drum.solve(Z_CRUDE)

print(f"  Overall vapour fraction Ψ = {flash.vapour_fraction:.4f}")
print(f"  Liquid fraction  L/F       = {flash.liquid_fraction:.4f}")
print()
print(f"  {'Component':<16}  {'Feed z':<10}  {'Liquid x':<10}  {'Vapour y':<10}  {'K':<8}")
print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
for comp, zi, xi, yi, ki in zip(
    CRUDE_COMPONENTS, Z_CRUDE, flash.x, flash.y, flash.k_values
):
    print(f"  {comp:<16}  {zi:<10.4f}  {xi:<10.4f}  {yi:<10.4f}  {ki:<8.4f}")

# Liquid stream to distillation train
liquid_flow = flash.liquid_fraction * FEED_FLOW
z_liquid = flash.x.copy()
log.info("Flash liquid to distillation: %.0f kmol/h", liquid_flow)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CO₂ absorber on the vapour stream
# ══════════════════════════════════════════════════════════════════════════════
header("STEP 2 — MEA CO₂ Absorber on Flash Vapour (90 % removal)")

vapour_flow = flash.vapour_fraction * FEED_FLOW
co2_in_vapour = vapour_flow * 0.09   # assume 9 mol% CO₂ in flash vapour

absorber = AmineAbsorber(
    removal_target=0.90,
    inlet_co2_mol_h=co2_in_vapour,
    inlet_total_mol_h=vapour_flow,
    config=cfg,
)
capture = absorber.solve()

print(f"  CO₂ inlet:             {co2_in_vapour:>10,.0f}  mol/h")
print(f"  CO₂ captured (90 %):   {capture.co2_captured_mol_h:>10,.0f}  mol/h")
print(f"  Solvent circulation:   {capture.solvent_circulation_mol_h:>10,.0f}  mol/h")
print(f"  Reboiler duty:         {capture.reboiler_duty_kw:>10,.1f}  kW")
print(f"  Lean/rich HX duty:     {capture.lean_rich_hx_duty_kw:>10,.1f}  kW")
print(f"  Absorber height est.:  {capture.absorber_height_m:>10.1f}  m")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Methyl-formate / methanol pre-cut column (DSTWU shortcut)
# ══════════════════════════════════════════════════════════════════════════════
header("STEP 3 — Methyl Formate / Methanol Pre-cut Column (DSTWU)")

col_precut = ShortcutColumn(
    light_key="methyl_formate",
    heavy_key="methanol",
    recovery_lk=0.99,
    recovery_hk=0.99,
    pressure_pa=101_325.0,
    feed_temperature_k=330.0,
    config=cfg,
)
precut = col_precut.solve(
    feed_flow=liquid_flow,
    z_lk=float(z_liquid[1]),   # methyl formate index = 1
    z_hk=float(z_liquid[2]),   # methanol index = 2
)
print(f"  Rmin  = {precut.r_min:.4f}")
print(f"  R     = {precut.r_actual:.4f}")
print(f"  Nmin  = {precut.n_min:.4f}")
print(f"  N     = {precut.n_actual:.4f}")
print(f"  Nfeed = {precut.n_feed:.1f}")
print(f"  α(MeFm/MeOH) = {precut.alpha_lk_hk:.4f}")

# Approximate methanol-rich bottoms after pre-cut
methanol_bottoms_flow = liquid_flow * (z_liquid[2] + z_liquid[3] + z_liquid[4] + z_liquid[5] + z_liquid[6])
z_meoh_water = np.array([z_liquid[2], z_liquid[4]])
z_meoh_water /= z_meoh_water.sum()
log.info("Methanol-rich stream to polishing column: %.0f kmol/h", methanol_bottoms_flow)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Methanol polishing column (rigorous MESH, B3 block)
# ══════════════════════════════════════════════════════════════════════════════
header("STEP 4 — Methanol Polishing Column (Rigorous MESH, 20 stages)")

col_polish = RigorousColumn(
    components=["methanol", "water"],
    n_stages=20,
    feed_stage=10,
    reflux_ratio=2.72,           # R ≈ 1.30 × Rmin from shortcut
    distillate_to_feed=0.89,     # C = 89 % recovery target
    pressure_pa=101_325.0,
    feed_temperature_k=337.0,
    config=cfg,
)
rigorous = col_polish.solve(
    feed_flow=methanol_bottoms_flow,
    z=z_meoh_water,
)

print(f"  Converged:             {rigorous.converged} ({rigorous.n_iterations} iterations)")
print(f"  Distillate MeOH:       {rigorous.distillate_composition[0]:.4f} mol frac")
print(f"  Bottoms H₂O:           {rigorous.bottoms_composition[1]:.4f} mol frac")
print(f"  Distillate flow:       {rigorous.distillate_flow:,.0f} kmol/h")
print(f"  Bottoms flow:          {rigorous.bottoms_flow:,.0f} kmol/h")
print(f"  Condenser T (top):     {rigorous.temperatures_k[0]:.2f} K")
print(f"  Reboiler T (bottom):   {rigorous.temperatures_k[-1]:.2f} K")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Ethanol / higher-alcohol column (DSTWU)
# ══════════════════════════════════════════════════════════════════════════════
header("STEP 5 — Ethanol / 1-Propanol Column (DSTWU)")

col_etoh = ShortcutColumn(
    light_key="ethanol",
    heavy_key="1_propanol",
    recovery_lk=0.99,
    recovery_hk=0.99,
    pressure_pa=101_325.0,
    feed_temperature_k=351.0,
    config=cfg,
)
etoh_design = col_etoh.solve(
    feed_flow=liquid_flow * (z_liquid[3] + z_liquid[5]),
    z_lk=0.60,
    z_hk=0.35,
)
print(f"  Rmin  = {etoh_design.r_min:.4f}")
print(f"  R     = {etoh_design.r_actual:.4f}")
print(f"  Nmin  = {etoh_design.n_min:.4f}")
print(f"  N     = {etoh_design.n_actual:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Cryogenic Air Separation Unit
# ══════════════════════════════════════════════════════════════════════════════
header("STEP 6 — Cryogenic Air Separation Unit")

asu = CryogenicASU(
    n2_demand_mol_h=15_000.0,
    o2_demand_mol_h=8_000.0,
    n2_purity=0.9999,
    o2_purity=0.9950,
    recover_argon=True,
    hp_pressure_bar=5.5,
    lp_pressure_bar=1.35,
    config=cfg,
)
asu_result = asu.solve()

print(f"  Air feed required:     {asu_result.air_feed_mol_h:>10,.0f}  mol/h")
print(f"  N₂ product:            {asu_result.n2_flow_mol_h:>10,.0f}  mol/h  [{asu_result.n2_purity:.4f}]")
print(f"  O₂ product:            {asu_result.o2_flow_mol_h:>10,.0f}  mol/h  [{asu_result.o2_purity:.4f}]")
print(f"  Ar product:            {asu_result.ar_flow_mol_h:>10,.0f}  mol/h")
print(f"  Compression power:     {asu_result.compression_power_kw:>10,.1f}  kW")
print(f"  HP column stages:      {asu_result.hp_column_stages:>10d}")
print(f"  LP column stages:      {asu_result.lp_column_stages:>10d}")
print(f"  HP pressure:           {asu_result.hp_column_pressure_bar:>10.2f}  bar")
print(f"  LP pressure:           {asu_result.lp_column_pressure_bar:>10.2f}  bar")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
header("FLOWSHEET SUMMARY")
print(f"  Feed basis:             {FEED_FLOW:,.0f} kmol/h  (F = 26 × 2500)")
print(f"  MeOH recovery target:  89 %  (C = B = 8 → 89)")
print(f"  MeOH distillate purity: {rigorous.distillate_composition[0]:.2%}")
print(f"  CO₂ removal (absorber): {capture.co2_removal_fraction:.0%}")
print(f"  ASU N₂ purity:          {asu_result.n2_purity:.4%}")
print(f"  ASU O₂ purity:          {asu_result.o2_purity:.4%}")
print(f"  sepflows version:       {sepflows.__version__}")
print()
