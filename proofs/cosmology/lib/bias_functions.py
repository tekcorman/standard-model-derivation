"""
bias_functions.py — Parametric-class translation bias.

Background
----------
The framework predicts native-class quantities (typically a coasting H(z)
in observer or substrate frame). Humans extract LCDM parameters by
fitting Friedmann's two-component class to observed data. The two
parametric classes cannot agree at all redshifts simultaneously; the
local LCDM parameters that exactly reproduce the native H(z) at a single
z form a *bias function* of z.

This module provides:

  - `local_friedmann_two_component(native, z)` — given a native
    Cosmography, return (Omega_m_local(z), Omega_L_local(z)) such that
    flat LCDM with those parameters has the same H(z) as the native at
    that single z. Generalizes Lambda_CC_parametric_translation_bias.py.

  - `solve_z_eff(native, target_func, target_value)` — invert the local
    bias to find the effective z at which a given LCDM-extracted value
    lands. Used to reproduce z_eff ~ 1.92 for Planck Omega_m = 0.315.

  - `local_w_DE(native, z, Omega_m)` — local-fit constant-w dark energy
    parameter at fixed Omega_m. Future work (Tier 4 D2) will use this as
    a building block for the n_s / w_DE / sigma_8 bias functions.

DERIVATION (FOR FLAT TWO-COMPONENT)
-----------------------------------
Setting H_native(z)^2 = H_LCDM(z; Om, H_0_LCDM)^2 with H_0_LCDM = H_native(0)
and flat LCDM (Omega_L = 1 - Omega_m):

    [H_native(z) / H_native(0)]^2 = Omega_m (1+z)^3 + (1 - Omega_m)

Solving for Omega_m:

    Omega_m_local(z) = ( [H_native(z)/H_native(0)]^2 - 1 ) / ( (1+z)^3 - 1 )

For the special case of coasting (H_native(z) = H_0(1+z)):

    Omega_m_local(z) = ( (1+z)^2 - 1 ) / ( (1+z)^3 - 1 )
                     = (u + 1) / (u^2 + u + 1)   with u = 1+z

This matches the closed form derived in
Lambda_CC_parametric_translation_bias.py exactly; the present module
generalizes it to arbitrary native cosmographies.

PURITY
------
All functions take their cosmography input explicitly. There are no
hidden default H_0 or Omega_m values. The only literals in pure-function
bodies are 0.0 and 1.0 (bookkeeping) and the small integers in algebraic
identities — pi/e are not needed.
"""

from __future__ import annotations

import math
from typing import Tuple

from scipy import optimize

from .cosmography import Cosmography
from .ontology import Frame, Tagged


# ---------------------------------------------------------------------------
# Core: local Friedmann two-component bias.
# ---------------------------------------------------------------------------


def local_friedmann_two_component(
    native: Cosmography, z: float
) -> Tuple[float, float]:
    """Local-fit (Omega_m, Omega_L) for flat two-component LCDM at z.

    The returned (Omega_m, Omega_L) values, plugged into a flat LCDM with
    H_0 = H_native(0), exactly reproduce H_native(z) at the supplied z
    (and only at that z; the values vary across z).

    Parameters
    ----------
    native : Cosmography
        The framework-native cosmography. Frame is preserved at the
        Tagged-output layer; this function returns plain floats since the
        ratio H(z)/H(0) is frame-invariant.
    z : float
        Redshift at which to evaluate the local Friedmann decomposition.
        For z = 0 we use the closed-form limit (2/3 for coasting, etc.).

    Returns
    -------
    (Omega_m_local, Omega_L_local) : Tuple[float, float]
        Both in [0, 1] for sensible native cosmographies; sum to 1
        (flatness). Returned as plain floats.

    Notes
    -----
    For coasting native, this matches the closed form
        Omega_m_local(z) = (u+1) / (u^2+u+1),  u = 1+z
    exactly. Verified in the self-test below.
    """
    H0 = native.H_at(0.0).value
    Hz = native.H_at(z).value
    if z == 0.0:
        # Closed-form limit: differentiate H^2 / H_0^2 - 1 over (1+z)^3 - 1
        # near z = 0 with l'Hopital. For coasting, this gives 2/3.
        # We compute it numerically by a one-sided finite difference,
        # which is robust for any native cosmography.
        dz = 1.0e-6
        H_eps = native.H_at(dz).value
        ratio_sq = (H_eps / H0) ** 2
        omega_m = (ratio_sq - 1.0) / ((1.0 + dz) ** 3 - 1.0)
        return (omega_m, 1.0 - omega_m)

    one_plus_z = 1.0 + z
    ratio_sq = (Hz / H0) ** 2
    numerator = ratio_sq - 1.0
    denominator = one_plus_z ** 3 - 1.0
    omega_m = numerator / denominator
    return (omega_m, 1.0 - omega_m)


def Omega_m_local(native: Cosmography, z: float) -> float:
    """Convenience: just the Omega_m part of the local two-component fit."""
    return local_friedmann_two_component(native, z)[0]


def Omega_L_local(native: Cosmography, z: float) -> float:
    """Convenience: just the Omega_L part of the local two-component fit."""
    return local_friedmann_two_component(native, z)[1]


# ---------------------------------------------------------------------------
# Inverse: given a target Omega_m_LCDM, find the effective z at which the
# native cosmography's local fit produces that value.
# ---------------------------------------------------------------------------


def solve_z_eff_for_Omega_m(
    native: Cosmography,
    target_Omega_m: float,
    *,
    z_min: float = 0.0,
    z_max: float = 1.0e4,
) -> float:
    """Find z* such that Omega_m_local(native, z*) = target_Omega_m.

    Uses scipy.optimize.brentq. Returns NaN if no root in [z_min, z_max].
    Caller chooses the bracket; for typical native cosmographies the root
    lies in [0, 10]. For coasting + Planck Omega_m = 0.315, the answer is
    z_eff ~ 1.92.

    Sign of f = Omega_m_local - target_Omega_m:
      - For coasting, Omega_m_local(0) = 2/3 > target_Omega_m for typical
        target ~ 0.3, AND Omega_m_local -> 0 as z -> infinity, so f
        changes sign somewhere in (0, infty).
    """

    def f(z: float) -> float:
        return Omega_m_local(native, z) - target_Omega_m

    # Validate sign change before brentq
    fa = f(z_min if z_min > 0 else 1e-9)
    fb = f(z_max)
    if fa * fb > 0:
        return float("nan")
    return optimize.brentq(f, z_min if z_min > 0 else 1e-9, z_max, xtol=1e-10)


# ---------------------------------------------------------------------------
# Generic: local bias for any LCDM-extracted parameter.
#
# Given a native cosmography and a target LCDM-extracted parameter x, this
# returns the local-fit value of x at redshift z that exactly reproduces
# the native H(z) at that single z. The "fit class" is a callable
# parametrized by x (and possibly fixed values of other params).
#
# This is the building block for n_s / w_DE / sigma_8 bias derivations in
# future sessions: each of those parameters has a corresponding local-fit
# extraction expression; the bias function plays the same role as
# Omega_m_local does for two-component LCDM.
# ---------------------------------------------------------------------------


def local_fit_parameter(
    native: Cosmography,
    z: float,
    fit_H_squared_ratio: "callable",
    bracket: Tuple[float, float],
) -> float:
    """Solve fit_H_squared_ratio(z, x) = (H_native(z)/H_native(0))^2 for x.

    Parameters
    ----------
    native : Cosmography
        Native cosmography providing H(z) and H(0).
    z : float
        Redshift at which the local fit is performed.
    fit_H_squared_ratio : Callable[[float, float], float]
        A function (z, x) -> (H_fit(z) / H_fit(0))^2 in the LCDM-class
        that the LCDM extraction uses. Example for two-component:
            lambda z, Om: Om * (1+z)**3 + (1 - Om)
    bracket : (lo, hi)
        Search bracket for x. Caller's responsibility to choose. For
        Omega_m, (0.0, 1.0) is appropriate.

    Returns
    -------
    x : float
        Local-fit parameter value such that fit and native agree at this z.
        NaN if no root in the bracket.
    """
    if z == 0.0:
        # At z = 0 both sides equal 1 trivially; the local-fit is degenerate.
        # Use a small-z limit by finite difference.
        z = 1.0e-6

    H0 = native.H_at(0.0).value
    Hz = native.H_at(z).value
    target_ratio_sq = (Hz / H0) ** 2

    def f(x: float) -> float:
        return fit_H_squared_ratio(z, x) - target_ratio_sq

    fa = f(bracket[0])
    fb = f(bracket[1])
    if fa * fb > 0:
        return float("nan")
    return optimize.brentq(f, bracket[0], bracket[1], xtol=1e-10)


# ---------------------------------------------------------------------------
# Closed-form helpers for the coasting case (used by self-tests and as
# audit-traceable fast paths in derivation documents).
# ---------------------------------------------------------------------------


def Omega_m_local_coasting_closed_form(z: float) -> float:
    """Closed form for coasting native: Om_local = (u+1)/(u^2+u+1), u=1+z.

    Audit-traceable; matches the derivation in
    Lambda_CC_parametric_translation_bias.py exactly.
    """
    u = 1.0 + z
    return (u + 1.0) / (u * u + u + 1.0)


# ---------------------------------------------------------------------------
# wCDM single-parameter local fit at fixed Omega_m.
# ---------------------------------------------------------------------------


def w_local_at_fixed_Omega_m(
    native: Cosmography, z: float, Omega_m: float
) -> float:
    """Local-fit w in wCDM at fixed Omega_m, given native cosmography.

    Solves

        (H_native(z) / H_native(0))^2
            = Omega_m * (1+z)^3  +  (1 - Omega_m) * (1+z)^{3(1+w)}

    for w. Returns NaN when no real-w solution exists (the right-hand
    side without the dark-energy term must be positive AND the dark-
    energy coefficient (1 - Omega_m) must be positive; both conditions
    are checked).

    DERIVATION (CLOSED FORM)
    ------------------------
    With u = 1 + z, R = (H_native(z) / H_native(0))^2:

        (1 - Omega_m) u^{3(1+w)} = R - Omega_m u^3
        u^{3(1+w)} = (R - Omega_m u^3) / (1 - Omega_m)

    For real w with u > 1, we need (R - Omega_m u^3) > 0 (so the log
    is defined) and (1 - Omega_m) > 0:

        3 (1 + w) ln(u) = ln[(R - Omega_m u^3) / (1 - Omega_m)]
        w = -1 + (1/3) * ln[(R - Omega_m u^3) / (1 - Omega_m)] / ln(u)

    For the framework's coasting native (R = u^2):

        w = -1 + (1/3) * ln[(u^2 - Omega_m u^3) / (1 - Omega_m)] / ln(u)
          = -1 + (1/3) * ln[u^2 (1 - Omega_m u) / (1 - Omega_m)] / ln(u)

    Reduces to w = -1 at the self-consistency point where
    Omega_m = (u + 1) / (u^2 + u + 1) (i.e., when fixed Omega_m equals
    the bias-function value Omega_m_local(z)).

    Parameters
    ----------
    native : Cosmography
        Native cosmography providing H(z) and H(0).
    z : float
        Redshift at which to evaluate the local-fit w. Must be > 0
        (at z = 0 the equation is degenerate; both sides equal H_0^2
        identically and w is unconstrained).
    Omega_m : float
        Fixed matter-density parameter in the wCDM fit class. Must be
        in (0, 1).

    Returns
    -------
    w : float
        Local-fit w at this z and Omega_m. NaN when no real solution
        exists.
    """
    if z <= 0.0:
        return float("nan")
    if Omega_m <= 0.0 or Omega_m >= 1.0:
        return float("nan")

    H0 = native.H_at(0.0).value
    Hz = native.H_at(z).value
    R = (Hz / H0) ** 2
    u = 1.0 + z
    de_numerator = R - Omega_m * (u ** 3)
    if de_numerator <= 0.0:
        return float("nan")
    de_coefficient = 1.0 - Omega_m
    return -1.0 + (1.0 / 3.0) * math.log(
        de_numerator / de_coefficient
    ) / math.log(u)


def w_local_at_fixed_Omega_m_coasting_closed_form(
    z: float, Omega_m: float
) -> float:
    """Closed form for coasting native, audit-traceable.

    w = -1 + (1/3) * ln[u^2 (1 - Omega_m u) / (1 - Omega_m)] / ln(u)
    where u = 1 + z. Equivalent to the generic w_local_at_fixed_Omega_m
    when native is coasting; provided as an algebraic identity for use
    in derivation documents.
    """
    if z <= 0.0:
        return float("nan")
    if Omega_m <= 0.0 or Omega_m >= 1.0:
        return float("nan")
    u = 1.0 + z
    one_minus_Omega_m_u = 1.0 - Omega_m * u
    if one_minus_Omega_m_u <= 0.0:
        return float("nan")
    return -1.0 + (1.0 / 3.0) * math.log(
        u * u * one_minus_Omega_m_u / (1.0 - Omega_m)
    ) / math.log(u)


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .cosmography import coasting

    H_0 = 68.18  # arbitrary; the bias is H_0-invariant
    cosmo = coasting(H_0=H_0, frame=Frame.SUBSTRATE)

    print("Local two-component bias under coasting:")
    print(f"  {'z':>7} {'Om_local':>12} {'closed-form':>14} {'rel_err':>10}")
    for z in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 100.0):
        omega_m_num = Omega_m_local(cosmo, z)
        omega_m_cf = Omega_m_local_coasting_closed_form(z)
        rel_err = (
            abs(omega_m_num - omega_m_cf) / omega_m_cf if omega_m_cf else 0.0
        )
        print(
            f"  {z:>7.2f} {omega_m_num:>12.6f} {omega_m_cf:>14.6f} "
            f"{rel_err:>10.2e}"
        )

    print()
    print("Inverse: z_eff at which local fit gives target Omega_m:")
    for target in (1.0 / 3.0, 0.315, 0.3, 0.5):
        z_eff = solve_z_eff_for_Omega_m(cosmo, target)
        print(f"  target Om = {target:.4f}  ->  z_eff = {z_eff:.4f}")

    print()
    print("Generic local_fit_parameter reproduces Omega_m_local:")
    fit_two_component = lambda z, Om: Om * (1.0 + z) ** 3 + (1.0 - Om)
    for z in (0.5, 1.0, 2.0):
        x = local_fit_parameter(cosmo, z, fit_two_component, (0.0, 1.0))
        omega_m_direct = Omega_m_local(cosmo, z)
        print(
            f"  z={z:5.2f}  generic={x:.6f}  direct={omega_m_direct:.6f}  "
            f"agree={abs(x - omega_m_direct) < 1e-9}"
        )

    print()
    print("wCDM local-fit w at fixed Omega_m = 0.3153 (Planck):")
    print(
        f"  {'z':>7} {'w_local':>12} {'closed-form':>14} {'rel_err':>10}"
    )
    for z in (0.5, 1.0, 1.5, 1.9162, 2.0, 2.5, 3.0):
        w_num = w_local_at_fixed_Omega_m(cosmo, z, Omega_m=0.3153)
        w_cf = w_local_at_fixed_Omega_m_coasting_closed_form(z, 0.3153)
        rel_err = (
            abs(w_num - w_cf) / abs(w_cf) if abs(w_cf) > 0 else 0.0
        )
        print(f"  {z:>7.4f} {w_num:>12.6f} {w_cf:>14.6f} {rel_err:>10.2e}")
    print()
    print(
        "  At z = z_eff(Omega_m=0.3153) = 1.9162, w_local = -1 by "
        "construction"
    )
    print(
        "  (the wCDM and Friedmann two-component classes coincide there)."
    )
