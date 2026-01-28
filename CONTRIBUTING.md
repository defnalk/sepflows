# Contributing to sepflows

Thank you for considering a contribution!  This document explains the process
for submitting bug reports, feature requests, and pull requests.

---

## Code of Conduct

All contributors are expected to be respectful and constructive.  Harassment
or dismissive behaviour will not be tolerated.

---

## Reporting Bugs

1. Search existing [Issues](https://github.com/defnalk/sepflows/issues) first.
2. If it's new, open an issue with the **Bug Report** template.
3. Include:
   - A minimal reproducible example (MRE)
   - Your Python version and OS
   - The full traceback

---

## Requesting Features

Open an issue with the **Feature Request** template.  Describe the
separation process context (equations, reference, physical interpretation)
so reviewers can evaluate physical correctness.

---

## Development Setup

```bash
git clone https://github.com/defnalk/sepflows.git
cd sepflows
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Run the full quality gate before opening a PR:

```bash
make lint       # ruff checks
make typecheck  # mypy strict
make test       # pytest + coverage ≥ 90 %
```

---

## Pull Request Guidelines

1. **Branch** from `main` with a descriptive name: `feat/srk-eos`, `fix/rachford-rice-edge`.
2. **Tests** — all new public functions must have unit tests.  Coverage must
   not drop below 90 %.
3. **Docstrings** — Google style, including `Args`, `Returns`, `Raises`, and
   at least one `Example` per public class method.
4. **Type hints** — every function parameter and return must be typed.
5. **No hardcoded values** — add constants to `constants.py` and reference them.
6. **Logging** — use `logging.getLogger(__name__)` inside each module; no
   `print()` statements.
7. **`__all__`** — update the module's `__all__` list for any new public symbol.
8. **CHANGELOG** — add a bullet under `[Unreleased]` describing your change.
9. Keep commits atomic and use
   [Conventional Commits](https://www.conventionalcommits.org/) style:
   `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.

---

## Running Tests

```bash
make test-unit         # fast unit tests only
make test-integration  # full flowsheet integration tests
make test              # both + coverage report
```

Integration tests are tagged `@pytest.mark.integration`.  They are slower
and require all four modules to work together.

---

## Coding Standards

- **Python 3.10+** syntax (use `X | Y` unions, `match`, etc. where appropriate)
- **ruff** for formatting and linting (configured in `pyproject.toml`)
- **mypy `--strict`** — no `Any`, no ignored errors without explanation
- Physical equations must cite a reference (paper, textbook section) in the
  module or function docstring

---

## Thermodynamic Extensions

sepflows uses ideal (Raoult) K-values by default.  If you are adding a
non-ideal EOS (e.g. SRK, PR, NRTL):
- Implement it behind the `eos_model` config key
- Ensure the new model is activated via `SepConfig(eos_model="srk")` and
  does not change existing default behaviour
- Provide at least one literature validation case in the test suite

---

## Questions?

Open a [Discussion](https://github.com/defnalk/sepflows/discussions) for
general questions that are not bug reports or feature requests.
