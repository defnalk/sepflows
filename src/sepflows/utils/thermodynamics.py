"""Thermodynamic property utilities.

Provides vapour-pressure correlations (Antoine equation), ideal K-values,
Rachford–Rice flash solution, and simple enthalpy estimates used
throughout sepflows.

References
----------
- Perry's Chemical Engineers' Handbook, 9th ed. (Green & Southard, 2019)
- Kister, H. Z. *Distillation Design* (1992)
- Rachford & Rice, *JPT* (1952)
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from sepflows.constants import ANTOINE, P_ATM

__all__ = [
    "antoine_pressure",
    "k_values_raoult",
    "rachford_rice",
    "bubble_point_temperature",
    "dew_point_temperature",
    "relative_volatility",
    "underwood_theta",
    "minimum_reflux_underwood",
]

_log = logging.getLogger(__name__)

# mmHg → Pa conversion (Antoine coefficients use mmHg)
_MMHG_TO_PA: float = 133.322


def antoine_pressure(component: str, temperature_k: float) -> float:
    """Compute saturation pressure via the Antoine equation.

    Uses the form:

    .. math::

        \\log_{10}(P^\\text{sat} / \\text{mmHg}) = A - \\frac{B}{C + T/°C}

    Args:
        component: Component key matching :data:`~sepflows.constants.ANTOINE`.
        temperature_k: Temperature in Kelvin.

    Returns:
        Saturation pressure in Pa.

    Raises:
        KeyError: If ``component`` is not found in the Antoine database.
        ValueError: If ``temperature_k`` is non-positive.

    Example:
        >>> from sepflows.utils.thermodynamics import antoine_pressure
        >>> round(antoine_pressure("methanol", 337.85), 0)
        101198.0
    """
    if temperature_k <= 0.0:
        raise ValueError(
            f"temperature_k must be positive, got {temperature_k}"
        )
    if component not in ANTOINE:
        available = ", ".join(sorted(ANTOINE.keys()))
        raise KeyError(
            f"Component '{component}' not in Antoine database.  "
            f"Available: {available}"
        )
    coeff = ANTOINE[component]
    t_celsius = temperature_k - 273.15
    log_p = coeff["A"] - coeff["B"] / (coeff["C"] + t_celsius)
    p_mmhg = 10.0**log_p
    return p_mmhg * _MMHG_TO_PA


def k_values_raoult(
    components: Sequence[str],
    temperature_k: float,
    pressure_pa: float,
) -> NDArray[np.float64]:
    """Compute ideal K-values using modified Raoult's law.

    .. math::

        K_i = P_i^\\text{sat}(T) / P

    Assumes ideal vapour phase and ideal liquid phase (γᵢ = 1).

    Args:
        components: Ordered list of component names.
        temperature_k: System temperature in Kelvin.
        pressure_pa: System pressure in Pa.

    Returns:
        1-D numpy array of K-values in the same order as ``components``.

    Raises:
        ValueError: If ``pressure_pa`` ≤ 0.
    """
    if pressure_pa <= 0.0:
        raise ValueError(f"pressure_pa must be positive, got {pressure_pa}")
    k = np.array(
        [antoine_pressure(c, temperature_k) / pressure_pa for c in components],
        dtype=np.float64,
    )
    _log.debug("K-values at T=%.2f K, P=%.0f Pa: %s", temperature_k, pressure_pa, k)
    return k


def rachford_rice(
    z: NDArray[np.float64],
    k: NDArray[np.float64],
    vf_lo: float = 1.0e-8,
    vf_hi: float = 1.0 - 1.0e-8,
    tol: float = 1.0e-10,
    max_iter: int = 200,
) -> float:
    """Solve the Rachford–Rice equation for vapour fraction Ψ.

    Solves:

    .. math::

        \\sum_i \\frac{z_i (K_i - 1)}{1 + \\Psi(K_i - 1)} = 0

    using Brent's method over the feasible bracket.

    Args:
        z: Feed mole fractions (must sum to 1.0).
        k: Equilibrium K-values (same component order as ``z``).
        vf_lo: Lower bracket bound for vapour fraction.
        vf_hi: Upper bracket bound for vapour fraction.
        tol: Convergence tolerance on Ψ.
        max_iter: Maximum Brent iterations.

    Returns:
        Vapour fraction Ψ ∈ (0, 1).

    Raises:
        ValueError: If feed or K-values are inconsistent, or if the
            mixture is entirely single-phase.
        RuntimeError: If the solver fails to converge.

    Example:
        >>> import numpy as np
        >>> from sepflows.utils.thermodynamics import rachford_rice
        >>> z = np.array([0.5, 0.5])
        >>> k = np.array([2.0, 0.5])
        >>> round(rachford_rice(z, k), 6)
        0.333333
    """
    z = np.asarray(z, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if z.shape != k.shape:
        raise ValueError(
            f"z and k must have the same shape, got {z.shape} vs {k.shape}"
        )
    if not math.isclose(z.sum(), 1.0, abs_tol=1e-6):
        raise ValueError(f"Feed fractions z must sum to 1.0, got {z.sum():.8f}")

    def _rr(psi: float) -> float:
        denom = 1.0 + psi * (k - 1.0)
        if np.any(denom <= 0.0):
            return float("nan")
        return float(np.sum(z * (k - 1.0) / denom))

    f_lo = _rr(vf_lo)
    f_hi = _rr(vf_hi)

    if f_lo < 0.0:
        _log.debug("Rachford-Rice: mixture is sub-cooled liquid (f_lo < 0).")
        return 0.0
    if f_hi > 0.0:
        _log.debug("Rachford-Rice: mixture is super-heated vapour (f_hi > 0).")
        return 1.0

    # Brent's method
    a, b = vf_lo, vf_hi
    fa, fb = f_lo, f_hi
    for iteration in range(max_iter):
        c = (a + b) / 2.0
        fc = _rr(c)
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            _log.debug("Rachford-Rice converged in %d iterations, Ψ=%.8f", iteration + 1, c)
            return c
        if fa * fc < 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raise RuntimeError(
        f"Rachford-Rice solver did not converge after {max_iter} iterations. "
        "Consider increasing max_iter or tightening the VF bracket."
    )


def bubble_point_temperature(
    components: Sequence[str],
    x: NDArray[np.float64],
    pressure_pa: float,
    t_init_k: float = 350.0,
    tol: float = 1.0e-4,
    max_iter: int = 300,
) -> float:
    """Compute bubble-point temperature for a liquid mixture.

    Finds the temperature T such that ΣᵢKᵢ(T)·xᵢ = 1 using bisection
    bracketed by a wide interval around the initial guess, then refined.

    Args:
        components: Component names matching the Antoine database.
        x: Liquid mole fractions (must sum to 1.0).
        pressure_pa: System pressure in Pa.
        t_init_k: Initial temperature guess in Kelvin.
        tol: Convergence tolerance on temperature (K).
        max_iter: Maximum bisection iterations.

    Returns:
        Bubble-point temperature in Kelvin.

    Raises:
        RuntimeError: If convergence is not achieved within ``max_iter``.
    """
    x = np.asarray(x, dtype=np.float64)

    def _residual(t: float) -> float:
        k = k_values_raoult(components, t, pressure_pa)
        return float(np.sum(k * x)) - 1.0

    # Find a bracket by scanning from t_init
    t_lo, t_hi = t_init_k * 0.6, t_init_k * 1.6
    t_lo = max(t_lo, 150.0)
    f_lo = _residual(t_lo)
    f_hi = _residual(t_hi)
    # Expand bracket if needed
    for _ in range(30):
        if f_lo * f_hi < 0:
            break
        if abs(f_lo) > abs(f_hi):
            t_lo = max(t_lo - 20.0, 150.0)
            f_lo = _residual(t_lo)
        else:
            t_hi += 20.0
            f_hi = _residual(t_hi)

    # Bisection
    for iteration in range(max_iter):
        t_mid = (t_lo + t_hi) / 2.0
        f_mid = _residual(t_mid)
        if abs(f_mid) < 1e-8 or (t_hi - t_lo) < tol:
            _log.debug("Bubble-point converged in %d iter: Tbp=%.4f K", iteration + 1, t_mid)
            return t_mid
        if f_lo * f_mid < 0:
            t_hi, f_hi = t_mid, f_mid
        else:
            t_lo, f_lo = t_mid, f_mid
    raise RuntimeError(
        f"Bubble-point temperature did not converge after {max_iter} iterations."
    )


def dew_point_temperature(
    components: Sequence[str],
    y: NDArray[np.float64],
    pressure_pa: float,
    t_init_k: float = 350.0,
    tol: float = 1.0e-4,
    max_iter: int = 300,
) -> float:
    """Compute dew-point temperature for a vapour mixture.

    Finds T such that Σᵢ(yᵢ/Kᵢ(T)) = 1 using bisection.

    Args:
        components: Component names matching the Antoine database.
        y: Vapour mole fractions (must sum to 1.0).
        pressure_pa: System pressure in Pa.
        t_init_k: Initial temperature guess in Kelvin.
        tol: Convergence tolerance on temperature (K).
        max_iter: Maximum bisection iterations.

    Returns:
        Dew-point temperature in Kelvin.

    Raises:
        RuntimeError: If convergence is not achieved within ``max_iter``.
    """
    y = np.asarray(y, dtype=np.float64)

    def _residual(t: float) -> float:
        k = k_values_raoult(components, t, pressure_pa)
        return float(np.sum(y / np.maximum(k, 1e-30))) - 1.0

    t_lo, t_hi = t_init_k * 0.6, t_init_k * 1.6
    t_lo = max(t_lo, 150.0)
    f_lo = _residual(t_lo)
    f_hi = _residual(t_hi)
    for _ in range(30):
        if f_lo * f_hi < 0:
            break
        if abs(f_lo) > abs(f_hi):
            t_lo = max(t_lo - 20.0, 150.0)
            f_lo = _residual(t_lo)
        else:
            t_hi += 20.0
            f_hi = _residual(t_hi)

    for iteration in range(max_iter):
        t_mid = (t_lo + t_hi) / 2.0
        f_mid = _residual(t_mid)
        if abs(f_mid) < 1e-8 or (t_hi - t_lo) < tol:
            _log.debug("Dew-point converged in %d iter: Tdp=%.4f K", iteration + 1, t_mid)
            return t_mid
        if f_lo * f_mid < 0:
            t_hi, f_hi = t_mid, f_mid
        else:
            t_lo, f_lo = t_mid, f_mid
    raise RuntimeError(
        f"Dew-point temperature did not converge after {max_iter} iterations."
    )


def relative_volatility(
    component: str,
    reference: str,
    temperature_k: float,
    pressure_pa: float = P_ATM,
) -> float:
    """Compute relative volatility of a component with respect to a reference.

    .. math::

        \\alpha_{i,\\text{ref}} = K_i / K_\\text{ref} = P_i^\\text{sat} / P_\\text{ref}^\\text{sat}

    Args:
        component: Light (more volatile) component name.
        reference: Heavy (less volatile) reference component name.
        temperature_k: Temperature in Kelvin.
        pressure_pa: System pressure in Pa (cancels for ideal systems;
            kept for API consistency).

    Returns:
        Relative volatility α (dimensionless).
    """
    p_i = antoine_pressure(component, temperature_k)
    p_ref = antoine_pressure(reference, temperature_k)
    alpha = p_i / p_ref
    _log.debug(
        "α(%s/%s) at %.1f K = %.4f", component, reference, temperature_k, alpha
    )
    return alpha


def underwood_theta(
    alpha: NDArray[np.float64],
    z_f: NDArray[np.float64],
    q: float,
    tol: float = 1.0e-10,
    max_iter: int = 500,
) -> NDArray[np.float64]:
    """Solve for Underwood roots θ in the range (αₗₖ, αₕₖ).

    Finds all roots of the Underwood equation:

    .. math::

        \\sum_i \\frac{\\alpha_i z_i}{\\alpha_i - \\theta} = 1 - q

    using Brent's method on each interval (αᵢ₊₁, αᵢ).

    Args:
        alpha: Array of component relative volatilities (sorted descending).
        z_f: Feed mole fractions (same order as ``alpha``).
        q: Feed thermal condition (q = 1 for saturated liquid, 0 for
            saturated vapour).
        tol: Root-finding tolerance.
        max_iter: Max iterations per Brent solve.

    Returns:
        Array of Underwood roots θ, one per pair of adjacent volatilities
        that straddles the RHS value (1 − q).
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    z_f = np.asarray(z_f, dtype=np.float64)
    rhs = 1.0 - q

    def _eq(theta: float) -> float:
        return float(np.sum(alpha * z_f / (alpha - theta))) - rhs

    roots: list[float] = []
    # Search between consecutive α values
    for i in range(len(alpha) - 1):
        a_lo = alpha[i + 1] + tol
        a_hi = alpha[i] - tol
        if _eq(a_lo) * _eq(a_hi) > 0:
            continue  # no sign change → no root in this interval
        # Brent solve
        fa, fb = _eq(a_lo), _eq(a_hi)
        a, b = a_lo, a_hi
        for _ in range(max_iter):
            c = (a + b) / 2.0
            fc = _eq(c)
            if abs(fc) < tol or (b - a) / 2.0 < tol:
                roots.append(c)
                break
            if fa * fc < 0.0:
                b, fb = c, fc
            else:
                a, fa = c, fc
    return np.array(roots, dtype=np.float64)


def minimum_reflux_underwood(
    alpha: NDArray[np.float64],
    z_f: NDArray[np.float64],
    x_d: NDArray[np.float64],
    q: float = 1.0,
) -> float:
    """Estimate minimum reflux ratio via the Underwood method.

    Args:
        alpha: Relative volatilities (descending, normalised to heavy key).
        z_f: Feed mole fractions.
        x_d: Distillate mole fractions.
        q: Feed thermal condition (1 = saturated liquid).

    Returns:
        Minimum reflux ratio Rmin (mol reflux / mol distillate).

    Raises:
        ValueError: If no Underwood roots are found (check α ordering).
    """
    thetas = underwood_theta(alpha, z_f, q)
    if thetas.size == 0:
        raise ValueError(
            "No Underwood roots found.  Verify that alpha is sorted "
            "descending and that z_f contains valid mole fractions."
        )
    # Vmin = sum_i [alpha_i * x_di / (alpha_i - theta)]  for each root
    v_min_candidates: list[float] = []
    for theta in thetas:
        v_min = float(np.sum(alpha * x_d / (alpha - theta)))
        v_min_candidates.append(v_min)
    v_min = max(v_min_candidates)
    r_min = v_min - 1.0  # Rmin = Vmin/D − 1  (basis: D = 1)
    _log.debug("Underwood Rmin = %.4f (Vmin candidates: %s)", r_min, v_min_candidates)
    return r_min
