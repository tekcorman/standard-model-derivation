"""
cosmography.py — Generic H(z), a(t), t(a) for arbitrary cosmographies.

A Cosmography is a small object that bundles a Hubble function H(z) with
its frame tag and a label. The library exposes:

  - `coasting(H_0, frame)`   — a propto t, q_0 = 0, j_0 = 0.
  - `flat_LCDM(H_0, Omega_m, frame)` — flat two-component Friedmann.
  - `from_callable(H_of_z, H_0, frame, label)` — wrap any callable H(z).

Each returns a Cosmography with `.H_at(z)` (returns Tagged), `.a_of_t(t)`
when closed-form available, and `.t_at_z(z)` (cosmic time at redshift z).

PROJECT-NATIVE vs EXTERNAL COMPARISON CLASSES
---------------------------------------------
- `coasting`: PROJECT-NATIVE. The framework's substrate-frame and observer-
  frame H(z) = H_0 (1+z) is theorem-grade per cascade theorem D1+D2+D3
  (`docs/theorems/theorem_g1b_r2_closure.md`). The (16/15) substrate->observer
  rate gap is theorem-grade per cascade D2-extended.
- `flat_LCDM`, `flat_wCDM`: NOT FRAMEWORK PHYSICS. These are external
  parametric classes that humans use to fit cosmological data. They are
  provided ONLY as comparison objects — to compute "what would an LCDM
  fitter recover from coasting data?" via the bias function machinery in
  `bias_functions.py`. Frame these objects as Frame.LCDM_EXTRACTED to make
  this explicit. Using them as if they described framework substrate is
  side-loading and is forbidden per `feedback_no_side_loaded_physics_no_adoptions.md`.
- `from_callable`: arbitrary user input; status depends on the callable's
  origin.

DESIGN
------
Cosmographies are plain data. Integration of H(z) into distances happens
in distances.py.

Pure-function rule: H(z) callables take H_0 and any model parameters as
explicit arguments. The Cosmography wrapper closes over them at construction
time so .H_at(z) is just z -> H. This is compatible with parameter_linter's
pure-function discipline: the underlying H_of_z is a pure function; the
wrapper is a convenience.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from .ontology import Frame, Tagged


# ---------------------------------------------------------------------------
# Pure H(z) functions — no hidden state, all parameters explicit.
# ---------------------------------------------------------------------------


def H_coasting(z: float, H_0: float) -> float:
    """Coasting Hubble: H(z) = H_0 * (1+z).

    Derivation: a propto t implies H = 1/t and a propto 1/(1+z), so
    t propto 1/(1+z), so H propto (1+z). H_0 explicit, no hidden defaults.
    """
    return H_0 * (1.0 + z)


def H_flat_LCDM(z: float, H_0: float, Omega_m: float) -> float:
    """Flat two-component LCDM: H(z) = H_0 * sqrt(Om*(1+z)^3 + (1-Om)).

    Standard Friedmann equation, no curvature, no radiation.
    Caller is responsible for using a value of Omega_m that is consistent
    with the data's effective z (per the (gamma) parametric-class-translation
    framing — Omega_m has different local values at different z).
    """
    one_plus_z = 1.0 + z
    return H_0 * math.sqrt(
        Omega_m * one_plus_z * one_plus_z * one_plus_z + (1.0 - Omega_m)
    )


def H_flat_wCDM(z: float, H_0: float, Omega_m: float, w: float) -> float:
    """Flat one-parameter dark-energy LCDM extension with constant w.

    H(z)/H_0 = sqrt( Om*(1+z)^3 + (1-Om)*(1+z)^{3(1+w)} ).
    Reduces to flat LCDM at w = -1.
    """
    one_plus_z = 1.0 + z
    matter = Omega_m * one_plus_z * one_plus_z * one_plus_z
    de_exp = 3.0 * (1.0 + w)
    de = (1.0 - Omega_m) * (one_plus_z ** de_exp)
    return H_0 * math.sqrt(matter + de)


# ---------------------------------------------------------------------------
# Cosmography wrapper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cosmography:
    """A Hubble function + frame tag + label.

    Construct via the factory helpers below (coasting, flat_LCDM, etc.) or
    via `from_callable` for non-standard H(z).

    Fields
    ------
    H_of_z : Callable[[float], float]
        Pure function returning H(z) in the same units as H_0.
    H_0 : float
        Hubble constant at z = 0; consistent with frame tag.
    frame : Frame
        Frame tag; propagated through all derived quantities.
    label : str
        Short identifier (used in reports).

    Optional fields
    ---------------
    a_of_t : Optional[Callable[[float], float]]
        Closed-form a(t) when available; else None.
    t_of_a : Optional[Callable[[float], float]]
        Closed-form t(a) when available; else None.
    """

    H_of_z: Callable[[float], float]
    H_0: float
    frame: Frame
    label: str
    a_of_t: Optional[Callable[[float], float]] = None
    t_of_a: Optional[Callable[[float], float]] = None

    def H_at(self, z: float) -> Tagged:
        """Evaluate H(z) and return a Tagged value with this frame."""
        return Tagged(value=self.H_of_z(z), frame=self.frame)

    def t_at_z(self, z: float) -> Optional[Tagged]:
        """Cosmic time at redshift z, when t_of_a is closed-form."""
        if self.t_of_a is None:
            return None
        a = 1.0 / (1.0 + z)
        return Tagged(value=self.t_of_a(a), frame=self.frame)


# ---------------------------------------------------------------------------
# Factories.
# ---------------------------------------------------------------------------


def coasting(H_0: float, frame: Frame) -> Cosmography:
    """Coasting cosmography: a propto t, H = 1/t, q_0 = 0.

    a(t) = (t / t_0) with t_0 = 1/H_0.
    t(a) = a / H_0.
    H(z) = H_0 (1+z).

    Used by the framework as the *substrate-native* (and observer-MDL)
    cosmography per cascade theorem D1+D2+D3. H_0 must be in the same
    frame as `frame`.
    """
    inv_H_0 = 1.0 / H_0

    def _a_of_t(t: float) -> float:
        return t * H_0

    def _t_of_a(a: float) -> float:
        return a * inv_H_0

    def _H(z: float) -> float:
        return H_coasting(z, H_0)

    return Cosmography(
        H_of_z=_H,
        H_0=H_0,
        frame=frame,
        label=f"coasting(H_0={H_0:g}, {frame.value})",
        a_of_t=_a_of_t,
        t_of_a=_t_of_a,
    )


def flat_LCDM(H_0: float, Omega_m: float, frame: Frame) -> Cosmography:
    """Flat two-component Friedmann cosmography.

    Has no closed-form a(t) without elliptic functions; leaves a_of_t and
    t_of_a as None. Numerical t(a) can be obtained via integration in
    distances.py.

    Used as the LCDM_EXTRACTED comparison class. Caller must declare frame
    explicitly (typically Frame.LCDM_EXTRACTED for "what humans extract").
    """

    def _H(z: float) -> float:
        return H_flat_LCDM(z, H_0, Omega_m)

    return Cosmography(
        H_of_z=_H,
        H_0=H_0,
        frame=frame,
        label=f"flat_LCDM(H_0={H_0:g}, Om={Omega_m:g}, {frame.value})",
    )


def flat_wCDM(
    H_0: float, Omega_m: float, w: float, frame: Frame
) -> Cosmography:
    """Flat single-w dark-energy cosmography. Reduces to flat_LCDM at w=-1."""

    def _H(z: float) -> float:
        return H_flat_wCDM(z, H_0, Omega_m, w)

    return Cosmography(
        H_of_z=_H,
        H_0=H_0,
        frame=frame,
        label=f"flat_wCDM(H_0={H_0:g}, Om={Omega_m:g}, w={w:g}, {frame.value})",
    )


def from_callable(
    H_of_z: Callable[[float], float],
    H_0: float,
    frame: Frame,
    label: str,
) -> Cosmography:
    """Wrap any pure H(z) callable into a Cosmography.

    Use for ad-hoc cosmographies (e.g., a piecewise model, or framework
    H(z) with corrections beyond pure coasting). Caller is responsible
    for ensuring H_of_z(0) == H_0 and frame consistency.
    """
    return Cosmography(H_of_z=H_of_z, H_0=H_0, frame=frame, label=label)


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # H_0 numerical values are *not* fitted here. The test exercises the
    # API; values come from the predictions/H_0.py document at the call
    # site. (68.18 substrate / 72.74 observer per cascade D2-extended.)
    H_0_sub = 68.18
    cosmo_sub = coasting(H_0=H_0_sub, frame=Frame.SUBSTRATE)
    cosmo_lcdm = flat_LCDM(
        H_0=67.36, Omega_m=0.3153, frame=Frame.LCDM_EXTRACTED
    )

    print("Substrate coasting cosmography:", cosmo_sub.label)
    for z in (0.0, 0.5, 1.0, 1100.0):
        print(f"  z={z:8.2f}  H={cosmo_sub.H_at(z)!r}")

    print()
    print("LCDM-extracted comparison:", cosmo_lcdm.label)
    for z in (0.0, 0.5, 1.0, 1100.0):
        print(f"  z={z:8.2f}  H={cosmo_lcdm.H_at(z)!r}")

    print()
    print("Coasting closed-form t(a):")
    a_test = 0.5  # half today's scale factor
    t_at_a = cosmo_sub.t_of_a(a_test)
    print(f"  t(a={a_test}) = {t_at_a:g}  (units: 1/H_0_sub)")
