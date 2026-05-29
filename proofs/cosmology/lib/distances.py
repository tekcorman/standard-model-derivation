"""
distances.py — Comoving / angular-diameter / luminosity distances.

All functions integrate against an arbitrary Cosmography. They do NOT
hardcode coasting or LCDM; the cosmography is an explicit argument. The
frame tag of the cosmography is propagated to the returned Tagged values.

EXTRACTION-LAYER STATUS (read first)
------------------------------------
The formulae in this module — D_C(z) = integral c/H dz, D_A = D_C/(1+z),
d_L = (1+z) D_C, mu = 5 log10(d_L) + 25 — are derived from the FRW
metric ds^2 = -dt^2 + a(t)^2 [dx^2 + ...] for a spatially flat universe
plus the standard candle / standard ruler interpretation of cosmological
observables.

These are NOT framework substrate physics. They describe what an
EXTERNAL FRW OBSERVER extracts from an H(z) function. Using these
formulae composed with framework's project-native H(z) gives "what an
FRW observer, doing standard cosmography on framework H(z) data, would
report" — a translation, not a substrate claim.

The framework's claim ends at H(z). Distance computations here are
deliberately framed as extraction-layer translations so that comparisons
with FRW-fit observation data (SN1a mu(z), CMB theta_*, BAO) are
conceptually clean. They MUST NOT be used to make framework substrate
claims about geometric distances. (See feedback_no_side_loaded_physics_no_adoptions.md.)

CONVENTIONS
-----------
We work in km/s for H_0 and km/s/Mpc-equivalent units throughout. The
speed-of-light constant `c_km_s` is an explicit parameter to every
distance function; callers supply it from a single source of truth (we
provide CODATA below as a named constant for convenience but the
distance functions never read it implicitly).

The integration uses scipy.integrate.quad with adaptive tolerance. Pure
functions: no global state, no hidden parameters. The Cosmography is the
only carrier of model state; c_km_s and z are explicit floats.

CLOSED FORMS
------------
For the *coasting* cosmography H(z) = H_0 (1+z), under FRW geometry the
comoving distance admits a closed form:

    D_C(z) = c/H_0 * ln(1+z)

(again, FRW-extraction interpretation; not a framework substrate claim).
This matches the generic numerical integration to ~1e-12 relative tolerance.
For flat LCDM, no closed form; numerical only.
"""

from __future__ import annotations

import math
from typing import Callable

from scipy import integrate

from .cosmography import Cosmography
from .ontology import Tagged


# ---------------------------------------------------------------------------
# CODATA speed of light, in km/s. Provided as a named module-level constant
# for convenience. Pure functions take c_km_s as an explicit parameter so
# callers may override (e.g., for unit-system experiments).
# ---------------------------------------------------------------------------

C_LIGHT_KM_S: float = 2.99792458e5
"""CODATA speed of light, km/s. Use as `c_km_s=C_LIGHT_KM_S` at call sites."""


# ---------------------------------------------------------------------------
# Generic distances against an arbitrary Cosmography.
# ---------------------------------------------------------------------------


def comoving_distance(
    z: float,
    cosmography: Cosmography,
    c_km_s: float,
    *,
    epsabs: float = 1e-12,
    epsrel: float = 1e-12,
) -> Tagged:
    """Comoving distance D_C(z) = integral_0^z c / H(z') dz'.

    Returned in the same length unit as c_km_s / H_0 (typically Mpc when
    c_km_s is in km/s and H_0 is in km/s/Mpc).

    Frame tag inherited from `cosmography`.
    """
    if z <= 0.0:
        return Tagged(value=0.0, frame=cosmography.frame)

    integrand: Callable[[float], float] = lambda zp: c_km_s / cosmography.H_of_z(zp)
    val, _err = integrate.quad(integrand, 0.0, z, epsabs=epsabs, epsrel=epsrel)
    return Tagged(value=val, frame=cosmography.frame)


def angular_diameter_distance(
    z: float,
    cosmography: Cosmography,
    c_km_s: float,
    *,
    epsabs: float = 1e-12,
    epsrel: float = 1e-12,
) -> Tagged:
    """D_A(z) = D_C(z) / (1 + z).  (Spatially flat.)

    Spatial flatness is ASSUMED (consistent with framework's substrate
    being effectively a flat 3-torus at cosmological scales). For curved
    cosmographies, a separate function would be needed.
    """
    dc = comoving_distance(
        z, cosmography, c_km_s, epsabs=epsabs, epsrel=epsrel
    )
    return Tagged(value=dc.value / (1.0 + z), frame=cosmography.frame)


def luminosity_distance(
    z: float,
    cosmography: Cosmography,
    c_km_s: float,
    *,
    epsabs: float = 1e-12,
    epsrel: float = 1e-12,
) -> Tagged:
    """d_L(z) = (1+z) * D_C(z).  (Spatially flat.)"""
    dc = comoving_distance(
        z, cosmography, c_km_s, epsabs=epsabs, epsrel=epsrel
    )
    return Tagged(value=(1.0 + z) * dc.value, frame=cosmography.frame)


def distance_modulus(
    z: float,
    cosmography: Cosmography,
    c_km_s: float,
    *,
    epsabs: float = 1e-12,
    epsrel: float = 1e-12,
) -> Tagged:
    """Distance modulus mu = 5 log10(d_L / Mpc) + 25.

    Assumes d_L is in Mpc (i.e., that c_km_s / H_0 has units of Mpc when
    H_0 is given in km/s/Mpc — the standard convention).
    """
    dl = luminosity_distance(
        z, cosmography, c_km_s, epsabs=epsabs, epsrel=epsrel
    )
    if dl.value <= 0.0:
        return Tagged(value=float("-inf"), frame=cosmography.frame)
    return Tagged(
        value=5.0 * math.log10(dl.value) + 25.0, frame=cosmography.frame
    )


# ---------------------------------------------------------------------------
# Closed-form fast paths.
# ---------------------------------------------------------------------------
# These reproduce the generic integrators at machine precision; provided
# for callers that want closed-form auditability (e.g., for analytic
# bias-function derivations).
# ---------------------------------------------------------------------------


def D_C_coasting_closed_form(z: float, H_0: float, c_km_s: float) -> float:
    """Closed-form D_C for coasting H(z) = H_0(1+z): D_C = (c/H_0) ln(1+z).

    Frame-agnostic; caller wraps with Tagged at the call site.
    """
    if z <= 0.0:
        return 0.0
    return (c_km_s / H_0) * math.log(1.0 + z)


def d_L_coasting_closed_form(z: float, H_0: float, c_km_s: float) -> float:
    """Closed-form d_L for coasting: d_L = (c/H_0)(1+z) ln(1+z)."""
    if z <= 0.0:
        return 0.0
    return (c_km_s / H_0) * (1.0 + z) * math.log(1.0 + z)


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .cosmography import coasting, flat_LCDM
    from .ontology import Frame

    H_0 = 68.18
    cosmo = coasting(H_0=H_0, frame=Frame.SUBSTRATE)
    cosmo_lcdm = flat_LCDM(
        H_0=67.36, Omega_m=0.3153, frame=Frame.LCDM_EXTRACTED
    )

    print("Comoving distance under coasting (Mpc):")
    for z in (0.1, 0.5, 1.0, 2.0, 5.0):
        d_num = comoving_distance(z, cosmo, c_km_s=C_LIGHT_KM_S)
        d_cf = D_C_coasting_closed_form(z, H_0, c_km_s=C_LIGHT_KM_S)
        rel_err = abs(d_num.value - d_cf) / d_cf if d_cf else 0.0
        print(
            f"  z={z:5.2f}  numerical={d_num.value:11.4f}  "
            f"closed-form={d_cf:11.4f}  rel_err={rel_err:.2e}"
        )

    print()
    print("Distance modulus under coasting vs LCDM (mag):")
    for z in (0.05, 0.5, 1.0, 2.0):
        mu_coast = distance_modulus(z, cosmo, c_km_s=C_LIGHT_KM_S)
        mu_lcdm = distance_modulus(z, cosmo_lcdm, c_km_s=C_LIGHT_KM_S)
        delta = mu_coast.value - mu_lcdm.value
        print(
            f"  z={z:5.2f}  mu_coast={mu_coast.value:7.4f}  "
            f"mu_lcdm={mu_lcdm.value:7.4f}  delta={delta:+.4f}"
        )
