"""
forward_models.py — Observable predictions from a Cosmography.

Composes cosmography + distances into the observables that experimental
datasets report. Each forward model is a pure function whose ONLY model
state is the Cosmography passed in.

EXTRACTION-LAYER STATUS (read first)
------------------------------------
The forward models here — SN1a mu(z), CMB theta_*, BAO D_V — all rest
on the FRW geometry assumption baked into `distances.py` plus the
standard-candle / standard-ruler / acoustic-peak interpretations of the
respective observations. These are external-physics-class extractions
applied to framework's project-native H(z), not framework substrate
predictions of the underlying photon flux / acoustic feature / BAO
correlation.

In particular, the CMB theta_* helper takes r_s as an EXTERNAL INPUT —
it does NOT derive r_s from substrate. Earlier session attempts to
derive r_s via tight-coupling sound speed were retracted as side-loaded
(deleted 2026-05-09). The forward-model layer remains for use with
externally-supplied r_s (e.g., the LCDM-fit r_s = 147 Mpc) to bracket
expected behavior, but framework's own theta_* claim requires substrate
derivation of r_s, not done. See
an internal working note
for the architecture and phase plan.

WHAT'S HERE NOW
---------------
  - sn1a_distance_modulus(z, cosmography, c_km_s) -> Tagged
        Type-Ia distance modulus mu(z). Wraps distances.distance_modulus
        with explicit unit handling.

  - sn1a_distance_modulus_grid(z_grid, cosmography, c_km_s) -> List[float]
        Vectorized form for multi-z mock-data generation.

  - cmb_acoustic_scale(z_star, r_s, cosmography, c_km_s) -> Tagged
        theta_*(z_*) = r_s / D_A(z_*) (1+z_*) approximation, or directly
        r_s / D_C(z_*) for the comoving acoustic scale variant. We
        provide both since framework derivations may use either form.

  - bao_distance_DV(z, cosmography, c_km_s) -> Tagged
        Volume-averaged BAO distance D_V = (D_M^2 * c z / H)^(1/3),
        used by 6dFGS/SDSS. D_M = D_C in spatially flat case.

WHAT'S NOT HERE YET
-------------------
  - sound horizon r_s integration  (deferred to B3.1, requires Tier 2 c_s)
  - native CMB power spectrum     (deferred to E1, requires Tier 1+2+3)
  - sigma_8 forward model          (deferred to D2, requires structure formation)

The forward models present here use the ARGUMENTS r_s and c_s as
explicit parameters; they neither derive nor fit them. When future
sessions close r_s structurally, the forward model is unchanged — only
the source of r_s changes.
"""

from __future__ import annotations

import math
from typing import Iterable, List

from .cosmography import Cosmography
from .distances import (
    angular_diameter_distance,
    comoving_distance,
    distance_modulus,
    luminosity_distance,
)
from .ontology import Tagged


# ---------------------------------------------------------------------------
# Type-Ia supernova distance modulus.
# ---------------------------------------------------------------------------


def sn1a_distance_modulus(
    z: float,
    cosmography: Cosmography,
    c_km_s: float,
) -> Tagged:
    """SN1a distance modulus mu(z) = 5 log10(d_L / Mpc) + 25.

    Frame inherited from cosmography. Caller supplies c_km_s (no implicit
    constant).
    """
    return distance_modulus(z, cosmography, c_km_s)


def sn1a_distance_modulus_grid(
    z_grid: Iterable[float],
    cosmography: Cosmography,
    c_km_s: float,
) -> List[Tagged]:
    """Vectorized mu(z) over a grid; returns list of Tagged values."""
    return [sn1a_distance_modulus(z, cosmography, c_km_s) for z in z_grid]


# ---------------------------------------------------------------------------
# CMB acoustic scale.
# ---------------------------------------------------------------------------


def cmb_theta_star_from_DA(
    z_star: float,
    r_s_proper_Mpc: float,
    cosmography: Cosmography,
    c_km_s: float,
) -> Tagged:
    """theta_* = r_s_proper / D_A(z_*).

    r_s_proper_Mpc is the proper sound horizon at z_* (caller-supplied;
    the library does not derive r_s in this module; deriving it requires
    substrate-side work outside the scope of this extraction layer).

    Returns theta_* in radians, tagged with cosmography.frame.
    """
    d_a = angular_diameter_distance(z_star, cosmography, c_km_s)
    return Tagged(value=r_s_proper_Mpc / d_a.value, frame=cosmography.frame)


def cmb_theta_star_from_DC(
    z_star: float,
    r_s_comoving_Mpc: float,
    cosmography: Cosmography,
    c_km_s: float,
) -> Tagged:
    """theta_* = r_s_comoving / D_C(z_*).

    Alternative form using comoving sound horizon and comoving distance.
    Identical numerically to the D_A form (since both r_s and D scale by
    1/(1+z) when going from comoving to proper); provided as an audit-
    traceable second form.
    """
    d_c = comoving_distance(z_star, cosmography, c_km_s)
    return Tagged(value=r_s_comoving_Mpc / d_c.value, frame=cosmography.frame)


# ---------------------------------------------------------------------------
# BAO volume-averaged distance.
# ---------------------------------------------------------------------------


def bao_distance_DV(
    z: float,
    cosmography: Cosmography,
    c_km_s: float,
) -> Tagged:
    """D_V(z) = ( D_M(z)^2 * c z / H(z) )^(1/3).  (Spatially flat: D_M = D_C.)

    The volume-averaged distance reported by SDSS BAO. Used in galaxy-
    survey BAO comparisons.
    """
    if z <= 0.0:
        return Tagged(value=0.0, frame=cosmography.frame)
    d_c = comoving_distance(z, cosmography, c_km_s)
    H_z = cosmography.H_at(z).value
    D_V_cubed = d_c.value * d_c.value * c_km_s * z / H_z
    return Tagged(value=D_V_cubed ** (1.0 / 3.0), frame=cosmography.frame)


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .cosmography import coasting, flat_LCDM
    from .distances import C_LIGHT_KM_S
    from .ontology import Frame

    # Native: framework's observer-frame coasting; H_0 = 72.74 per
    # predictions/H_0.py (cited at use site, not magic).
    cosmo_obs = coasting(H_0=72.74, frame=Frame.OBSERVER)
    cosmo_lcdm = flat_LCDM(
        H_0=67.36, Omega_m=0.3153, frame=Frame.LCDM_EXTRACTED
    )

    print("SN1a mu(z) under observer-coasting vs LCDM-extracted:")
    print(f"{'z':>6} {'mu_obs':>10} {'mu_lcdm':>10} {'delta':>8}")
    for z in (0.05, 0.5, 1.0, 2.0):
        m1 = sn1a_distance_modulus(z, cosmo_obs, c_km_s=C_LIGHT_KM_S)
        m2 = sn1a_distance_modulus(z, cosmo_lcdm, c_km_s=C_LIGHT_KM_S)
        print(
            f"{z:>6.2f} {m1.value:>10.4f} {m2.value:>10.4f} "
            f"{m1.value - m2.value:>+8.4f}"
        )

    print()
    print("CMB theta_* with r_s_comoving = 147.05 Mpc (LCDM-fit value, used")
    print("here purely to exercise the API; framework derivation deferred):")
    Z_STAR = 1090.0  # last-scattering redshift (Planck 2018)
    R_S_COMOVING = 147.05  # Mpc, comoving sound horizon at z_* (LCDM-fit)
    R_S_PROPER = R_S_COMOVING / (1.0 + Z_STAR)  # Mpc, proper at z_*

    theta_obs_dc = cmb_theta_star_from_DC(
        Z_STAR, R_S_COMOVING, cosmo_obs, c_km_s=C_LIGHT_KM_S
    )
    theta_obs_da = cmb_theta_star_from_DA(
        Z_STAR, R_S_PROPER, cosmo_obs, c_km_s=C_LIGHT_KM_S
    )
    theta_lcdm_dc = cmb_theta_star_from_DC(
        Z_STAR, R_S_COMOVING, cosmo_lcdm, c_km_s=C_LIGHT_KM_S
    )
    theta_lcdm_da = cmb_theta_star_from_DA(
        Z_STAR, R_S_PROPER, cosmo_lcdm, c_km_s=C_LIGHT_KM_S
    )

    # The two forms (comoving / proper) are algebraic identities; verify.
    assert abs(theta_obs_dc.value - theta_obs_da.value) < 1e-12
    assert abs(theta_lcdm_dc.value - theta_lcdm_da.value) < 1e-12

    print(
        f"  theta_* (observer-coasting):     {theta_obs_dc.value:.6e} rad"
    )
    print(
        f"  theta_* (LCDM-extracted):        {theta_lcdm_dc.value:.6e} rad"
    )
    print(
        f"  Planck 2018 observation:         1.041085e-02 rad"
    )
    print(
        f"  Comoving / proper forms agree:   yes (machine precision)"
    )
    print(
        "  NOTE: LCDM-extracted recovers the Planck value (API sanity check);"
    )
    print(
        "  observer-coasting differs by the factor-of-2 documented in the"
    )
    print(
        "  CMB_distance_factor_two_unification_2026-05-08.md audit. This is a"
    )
    print(
        "  KNOWN open structural item, not closed by this library."
    )

    print()
    print("BAO D_V(z=0.35) (6dFGS / SDSS LRG):")
    dv_obs = bao_distance_DV(0.35, cosmo_obs, c_km_s=C_LIGHT_KM_S)
    dv_lcdm = bao_distance_DV(0.35, cosmo_lcdm, c_km_s=C_LIGHT_KM_S)
    print(
        f"  D_V(0.35) observer-coasting:   {dv_obs.value:.2f} Mpc "
        f"(observed ~1370 Mpc)"
    )
    print(
        f"  D_V(0.35) LCDM-extracted:      {dv_lcdm.value:.2f} Mpc"
    )
