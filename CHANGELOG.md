# Changelog

All notable changes to **sepflows** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Soave–Redlich–Kwong (SRK) equation of state for non-ideal K-values
- Packed-bed absorber sizing (Onda correlation)
- Heat-integrated distillation (HIDiC) module
- Aspen Plus HYSYS-compatible JSON export for column initialisation

---

## [0.1.0] — 2024-09-01

### Added
- **`FlashDrum`** — isothermal two-phase flash via Rachford–Rice with Brent solver; `.sensitivity()` temperature sweep
- **`ShortcutColumn`** — Fenske–Underwood–Gilliland (DSTWU) shortcut distillation; Molokanov–Gilliland correlation; Kirkbride feed-stage estimate
- **`RigorousColumn`** — Wang–Henke MESH tridiagonal solver with CMO assumption and successive substitution temperature update
- **`AmineAbsorber`** — MEA absorber–stripper unit sizing: solvent circulation, reboiler duty, lean/rich HX duty, NTU/HTU absorber height
- **`CryogenicASU`** — Linde–Frankl double-column sizing: Fenske Nmin for HP (N₂/O₂) and LP (O₂/Ar) columns; compression power benchmark
- **`SepConfig`** dataclass for centralised solver control (tolerances, EOS, verbosity)
- **`configure_logging()`** for structured `sepflows` logger setup
- **Antoine equation** database (Perry's 9th ed.) for 11 components: methanol, water, ethanol, DME, methyl formate, 1-propanol, 1-butanol, N₂, O₂, Ar, CO₂
- **`constants`** module: molecular weights, normal boiling points, critical properties, ASU compositions, process defaults
- **`utils.thermodynamics`**: `rachford_rice`, `k_values_raoult`, `bubble_point_temperature`, `dew_point_temperature`, `relative_volatility`, `underwood_theta`, `minimum_reflux_underwood`
- 90+ unit and integration tests covering all public API surfaces
- Full type hints (`py.typed` marker) and Google-style docstrings throughout
- CI/CD via GitHub Actions: lint + type-check + test matrix (Python 3.10–3.12, Ubuntu/macOS/Windows)
- `Makefile` with `test`, `lint`, `format`, `typecheck`, `coverage`, `clean` targets

[Unreleased]: https://github.com/defnalk/sepflows/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/defnalk/sepflows/releases/tag/v0.1.0
