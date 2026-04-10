# sepflows

![Tests](https://github.com/defnalk/sepflows/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-90%25+-brightgreen)
![Version](https://img.shields.io/badge/version-0.1.0-blue)

> **A Python toolkit for the simulation and design of industrial separation sequences.**
> Flash drums · Distillation trains · CO₂ absorption · Cryogenic ASU

`sepflows` provides composable, well typed building blocks for separation process design inspired by the integrated methanol purification, CO₂ capture, and air separation flowsheet described in *CENG50009 Separation Processes 2* at Imperial College London.
It is designed to feel like a real scientific computing package, similar in spirit to [scipy](https://scipy.org/) or [pvlib](https://pvlib-python.readthedocs.io/), with transparent intermediate values, dataclass results, and no hidden state.

---

## ✨ Features

- **Flash drum**, isothermal two phase VLE via the Rachford Rice equation; temperature sensitivity sweep for partial condensation optimisation
- **Shortcut distillation (DSTWU / FUG)**: Fenske Underwood Gilliland procedure matching Aspen Plus DSTWU output; multi component relative volatility
- **Rigorous distillation (MESH)**: Wang Henke tridiagonal solver with CMO assumption; full stage by stage profiles
- **CO₂ absorption**: MEA based absorber stripper sizing with lean/rich heat exchanger duty, NTU/HTU absorber height, and reboiler duty estimation
- **Cryogenic ASU**: Linde Frankl double column sizing with Fenske Nmin for both HP and LP columns, compression power estimate, and optional argon recovery
- **Thermodynamic utilities**: Antoine equation (Perry's 9th ed.), ideal K values, bubble/dew point iteration, Underwood roots
- **Full type hints** and Google style docstrings throughout
- **100 % module `__all__` coverage**
- **Logging** (not print statements) via the `sepflows` logger

---

## 📦 Installation

**From source (editable, recommended for development):**

```bash
git clone https://github.com/defnalk/sepflows.git
cd sepflows
pip install -e ".[dev]"
```

**Minimum runtime dependencies:** `numpy >= 1.24`, `scipy >= 1.10`

---

## 🚀 Quickstart

### Flash Drum

```python
import numpy as np
from sepflows.flash import FlashDrum

# Crude methanol syngas separator: three-component system at 280 K, 20 bar
drum = FlashDrum(
    components=["dme", "methanol", "water"],
    temperature_k=280.0,
    pressure_pa=20.0e5,
)

z = np.array([0.10, 0.65, 0.25])        # feed mole fractions
result = drum.solve(z)

print(f"Vapour fraction:  {result.vapour_fraction:.4f}")
print(f"Liquid (x):  {dict(zip(drum.components, result.x.round(4)))}")
print(f"Vapour (y):  {dict(zip(drum.components, result.y.round(4)))}")

# Temperature sensitivity sweep for partial condensation design
temps = np.linspace(250.0, 350.0, 50)
sensitivity = drum.sensitivity(z, temps)
vf_profile = [r.vapour_fraction for r in sensitivity]
```

### Shortcut Distillation Column (DSTWU / FUG)

```python
from sepflows.distillation import ShortcutColumn

# Methanol/water polishing column — matches CENG50009 Table 1 B3 block
col = ShortcutColumn(
    light_key="methanol",
    heavy_key="water",
    recovery_lk=0.995,          # 99.5 % methanol in distillate
    recovery_hk=0.995,          # 99.5 % water in bottoms
    pressure_pa=101_325.0,
    feed_temperature_k=337.0,   # near bubble point
    reflux_multiplier=1.30,     # R = 1.30 × Rmin  (Kister heuristic)
)

design = col.solve(
    feed_flow=65_000.0,         # kmol/h  (F = A × 2500, A = 26)
    z_lk=0.60,
    z_hk=0.35,
)

print(design)
# DSWTUResult(Rmin=2.09, R=2.72, Nmin=10.83, N=20.20, Nf=8.3)
```

### Rigorous Column (MESH Solver)

```python
import numpy as np
from sepflows.distillation import RigorousColumn

col = RigorousColumn(
    components=["methanol", "water"],
    n_stages=20,
    feed_stage=10,
    reflux_ratio=2.72,
    distillate_to_feed=0.55,
    pressure_pa=101_325.0,
    feed_temperature_k=337.0,
)

result = col.solve(feed_flow=65_000.0, z=np.array([0.60, 0.40]))

print(f"Converged in {result.n_iterations} iterations: {result.converged}")
print(f"Distillate MeOH: {result.distillate_composition[0]:.4f}")
print(f"Bottoms H₂O:     {result.bottoms_composition[1]:.4f}")
```

### CO₂ Capture (MEA Absorber–Stripper)

```python
from sepflows.absorption import AmineAbsorber

absorber = AmineAbsorber(
    removal_target=0.90,            # 90 % CO₂ removal
    inlet_co2_mol_h=4_500.0,       # mol/h CO₂ in recycle gas
    inlet_total_mol_h=50_000.0,    # mol/h total gas
    lean_loading=0.20,             # mol CO₂ / mol MEA
    rich_loading=0.48,
    lrhx_effectiveness=0.85,
)

result = absorber.solve()

print(f"CO₂ removed:      {result.co2_removal_fraction:.1%}")
print(f"Solvent circ.:    {result.solvent_circulation_mol_h:,.0f} mol/h")
print(f"Reboiler duty:    {result.reboiler_duty_kw:,.1f} kW")
print(f"Absorber height:  {result.absorber_height_m:.1f} m")
```

### Cryogenic Air Separation Unit

```python
from sepflows.asu import CryogenicASU

asu = CryogenicASU(
    n2_demand_mol_h=15_000.0,   # mol/h N₂ required
    o2_demand_mol_h=8_000.0,    # mol/h O₂ required
    n2_purity=0.9999,
    o2_purity=0.9950,
    recover_argon=True,
    hp_pressure_bar=5.5,
    lp_pressure_bar=1.35,
)

result = asu.solve()

print(f"Air feed:          {result.air_feed_mol_h:,.0f} mol/h")
print(f"Compression power: {result.compression_power_kw:,.1f} kW")
print(f"HP column stages:  {result.hp_column_stages}")
print(f"LP column stages:  {result.lp_column_stages}")
```

### Configuring the Solver

```python
from sepflows.config import SepConfig, configure_logging
import logging

configure_logging(logging.DEBUG)   # enable verbose solver logs

cfg = SepConfig(
    max_iterations=500,
    convergence_tol=1e-10,
    verbose=True,
)

from sepflows.flash import FlashDrum
drum = FlashDrum(["methanol", "water"], 320.0, 2e5, config=cfg)
```

---

## 📐 API Reference

### `sepflows.flash.FlashDrum`

| Method | Description |
|---|---|
| `FlashDrum(components, temperature_k, pressure_pa, config)` | Constructor |
| `.solve(z) → FlashDrumResult` | Rachford–Rice isothermal flash |
| `.sensitivity(z, temperatures_k) → list[FlashDrumResult]` | Temperature sweep |

**`FlashDrumResult`** attributes: `vapour_fraction`, `liquid_fraction`, `x`, `y`, `k_values`, `temperature_k`, `pressure_pa`, `components`, `converged`

---

### `sepflows.distillation.ShortcutColumn`

| Argument | Default | Description |
|---|---|---|
| `light_key` | n/a | LK component name |
| `heavy_key` | n/a | HK component name |
| `recovery_lk` | n/a | LK distillate recovery (0 to 1) |
| `recovery_hk` | n/a | HK bottoms recovery (0 to 1) |
| `reflux_multiplier` | `1.30` | R = multiplier × Rmin |
| `stages_multiplier` | `2.00` | N = multiplier × Nmin |

`.solve(feed_flow, z_lk, z_hk) → DSWTUResult`

**`DSWTUResult`** attributes: `r_min`, `r_actual`, `n_min`, `n_actual`, `n_feed`, `alpha_lk_hk`, `recovery_lk`, `recovery_hk`, `separation_factor`

---

### `sepflows.distillation.RigorousColumn`

Wang Henke MESH solver. `.solve(feed_flow, z) → RigorousColumnResult`

**`RigorousColumnResult`** attributes: `x`, `y`, `l_flows`, `v_flows`, `temperatures_k`, `distillate_composition`, `bottoms_composition`, `distillate_flow`, `bottoms_flow`, `n_iterations`, `converged`

---

### `sepflows.absorption.AmineAbsorber`

MEA absorber stripper unit. `.solve() → CO2CaptureResult`

**`CO2CaptureResult`** attributes: `co2_captured_mol_h`, `co2_removal_fraction`, `lean_loading`, `rich_loading`, `solvent_circulation_mol_h`, `reboiler_duty_kw`, `lean_rich_hx_duty_kw`, `absorber_height_m`

---

### `sepflows.asu.CryogenicASU`

Linde Frankl double column ASU. `.solve() → ASUResult`

**`ASUResult`** attributes: `n2_flow_mol_h`, `o2_flow_mol_h`, `ar_flow_mol_h`, `air_feed_mol_h`, `compression_power_kw`, `hp_column_stages`, `lp_column_stages`, `n2_recovery`, `o2_recovery`

---

### `sepflows.config.SepConfig`

| Attribute | Default | Description |
|---|---|---|
| `max_iterations` | `200` | Max solver iterations |
| `convergence_tol` | `1e-8` | Mole fraction residual tolerance |
| `eos_model` | `"raoult"` | EOS: `"raoult"` or `"srk"` (future) |
| `verbose` | `False` | Emit DEBUG level iteration logs |

---

## 🧪 Running Tests

```bash
# Full suite with coverage report
make test

# Unit tests only (fast)
make test-unit

# Integration tests (full flowsheet)
make test-integration

# HTML coverage report
make coverage
```

Tests are split into **unit** (`tests/unit/`) and **integration** (`tests/integration/`) categories.  The integration suite carries the `@pytest.mark.integration` marker.

---

## 🛠 Development

```bash
make install-dev   # editable install + dev deps
make lint          # ruff check
make format        # ruff format + auto fix
make typecheck     # mypy strict
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contribution guide.

---

## 📚 References

- Green, D. W. & Southard, M. Z. (eds.) *Perry's Chemical Engineers' Handbook*, 9th ed. McGraw Hill, 2019
- Kister, H. Z. *Distillation Design*. McGraw Hill, 1992
- Kohl, A. L. & Nielsen, R. *Gas Purification*, 5th ed. Gulf Publishing, 1997
- Smith, A. R. & Klosek, J. *Fuel Processing Technology*, 70 (2001), ASU review

---

## 📄 License

MIT © 2024 Defne Nihal Ertuğrul, see [`LICENSE`](LICENSE) for details.
