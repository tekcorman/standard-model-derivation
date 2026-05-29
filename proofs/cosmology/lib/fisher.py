"""
fisher.py — Per-dataset, per-parameter Fisher information.

LAYER 4 (LCDM-fit emulator) — pure numerical machinery applied at the
extraction-layer + LCDM-class-fit layer. Outputs are tagged with
Frame.LCDM_EXTRACTED. This module emulates what an LCDM-fit pipeline
recovers from data; it makes no substrate-physics claims.

LINE CLASSIFICATION (per cosmology architecture §2)
---------------------------------------------------
  (b) PURE MATHEMATICAL IDENTITY — Cramér–Rao information for Gaussian
      likelihood:

          F_{ij} = sum_n  (dO/dtheta_i)_n * (dO/dtheta_j)_n / sigma_n^2

      The integral form

          F_{ij} = int (dO/dtheta_i)(dO/dtheta_j) / sigma(z)^2 dz

      is the continuous limit of the same identity (folding a redshift
      density n(z) into 1/sigma^2). Central finite differences are
      numerical evaluation of dO/dtheta — no physics imported.

  (c) EXTRACTION-LAYER TRANSLATION — observables O in {mu, D_V, theta_*}
      are FRW-extraction layer (forward_models.py header). The parameter
      set {H_0, Omega_m, Omega_L, w} is LCDM-fit class. The default frame
      tag Frame.LCDM_EXTRACTED on outputs is the explicit acknowledgement
      that this module emulates an external fit pipeline.

  (a) PROJECT-NATIVE — NONE in this module. The Fisher machinery does
      not reach back into substrate dynamics.

NO SIDE-LOADED PHYSICS. The Cramér–Rao formula is a property of
likelihood construction, not a substrate claim. Per
feedback_no_side_loaded_physics_no_adoptions.md (memory) and the
architecture's §2 / §10 discipline, classification at introduction is
what prevents drift.

PHASE A CONTEXT
---------------
Phase A.1 of the cosmology simulator (see
an internal working note, §13).
Used by A.3's multi_dataset.py to derive z_eff (currently a data-side
empirical anchor at 1.916 from inverting Planck Omega_m = 0.3153) from
first-principles dataset specifications + Fisher analysis. Replaces the
heuristic Fisher weights in proofs/cosmology/O2_z_eff_multidataset_derivation.py
(self-flagged as approximations in that probe's own docstring).

API
---
  FisherMatrix(matrix, parameter_names, frame)
        Frame-tagged Fisher information matrix container.

  partial_observable_finite_difference(...)
        Central finite-difference partial dO/dtheta at a single z.

  fisher_information(...)
        Per-dataset Fisher matrix from a list of (z, sigma) measurement
        points. The continuous integral is recovered when the points are
        quadrature nodes with the appropriate density absorbed into
        1/sigma^2.

DISCIPLINE (parameter_linter R1 + cosmology architecture §10)
-------------------------------------------------------------
  - Every physical input (H_0, Omega_m, w, c_km_s, ...) is an explicit
    named argument. No hardcoded values inside function bodies.
  - Public functions are keyword-only to force named call sites.
  - Outputs are Tagged with Frame.LCDM_EXTRACTED by default.

REFERENCES
----------
  an internal working note (§3, §6, §13)
  an internal working note (§5)
  proofs/cosmology/O2_z_eff_multidataset_derivation.py (heuristic Fisher
    weights this module replaces)
  docs/parameters/parameter_linter.md (R1: physical inputs as named args)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .cosmography import Cosmography
from .ontology import Frame


# ---------------------------------------------------------------------------
# Fisher matrix container.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FisherMatrix:
    """A Fisher information matrix tagged with its frame and parameter names.

    Fields
    ------
    matrix : np.ndarray
        Square (P, P) matrix; F[i, j] is the Fisher information for the
        i-th and j-th parameters in `parameter_names`.
    parameter_names : Tuple[str, ...]
        Ordered names of the P parameters; index in this tuple matches
        index in `matrix`.
    frame : Frame
        Output frame. For LCDM-fit emulator usage this is
        Frame.LCDM_EXTRACTED.

    Invariants enforced in __post_init__:
      - matrix is 2-D and square
      - matrix.shape[0] == len(parameter_names)
    """

    matrix: np.ndarray
    parameter_names: Tuple[str, ...]
    frame: Frame

    def __post_init__(self) -> None:
        m = self.matrix
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(
                f"FisherMatrix.matrix must be square, got shape {m.shape}."
            )
        if m.shape[0] != len(self.parameter_names):
            raise ValueError(
                f"FisherMatrix.matrix shape {m.shape} inconsistent with "
                f"parameter_names {self.parameter_names}."
            )

    def index_of(self, name: str) -> int:
        """Index of a parameter name in this Fisher matrix."""
        return self.parameter_names.index(name)


# ---------------------------------------------------------------------------
# Finite-difference partial derivative of an observable w.r.t. a parameter.
# ---------------------------------------------------------------------------


def partial_observable_finite_difference(
    *,
    observable_fn: Callable[..., "object"],
    cosmography_factory: Callable[..., Cosmography],
    fiducial_params: Dict[str, float],
    parameter_name: str,
    z: float,
    c_km_s: float,
    step_relative: float,
    step_absolute: float,
    observable_kwargs: Optional[Dict[str, float]] = None,
) -> float:
    """Central finite-difference partial dO/dtheta at fixed z.

    Computes
        (O(theta_0 + h) - O(theta_0 - h)) / (2 h)
    where h = max(step_absolute, step_relative * |theta_0|).

    The cosmography is rebuilt from a perturbed copy of `fiducial_params`
    on each evaluation; the factory is the only contract between this
    module and the LCDM-class form being differentiated.

    Parameters
    ----------
    observable_fn : Callable
        Forward model with signature (z, cosmography, c_km_s, **kwargs)
        returning a Tagged value (.value extracted internally).
    cosmography_factory : Callable
        (**fiducial_params) -> Cosmography. The LCDM-class form is
        encoded by what this factory builds (e.g., flat_LCDM, flat_wCDM).
    fiducial_params : Dict[str, float]
        Fiducial point in parameter space.
    parameter_name : str
        Key in `fiducial_params` to perturb.
    z : float
        Redshift at which to evaluate the partial.
    c_km_s : float
        Speed of light, explicit per parameter_linter R1.
    step_relative : float
        Relative finite-difference step h_rel * |theta|.
    step_absolute : float
        Floor step (used when |theta| is near zero).
    observable_kwargs : Optional[Dict[str, float]]
        Extra keyword arguments forwarded to observable_fn (e.g., r_s for
        a theta_* adapter). None if the observable is fully specified by
        (z, cosmography, c_km_s).

    Returns
    -------
    float
        dO/dtheta at the fiducial point and supplied z.
    """
    if parameter_name not in fiducial_params:
        raise KeyError(
            f"parameter_name {parameter_name!r} not in fiducial_params "
            f"keys {list(fiducial_params.keys())}."
        )

    theta_0 = fiducial_params[parameter_name]
    h = max(step_absolute, step_relative * abs(theta_0))

    extra = dict(observable_kwargs or {})

    def evaluate_at(theta: float) -> float:
        params = dict(fiducial_params)
        params[parameter_name] = theta
        cosmo = cosmography_factory(**params)
        return observable_fn(z, cosmo, c_km_s, **extra).value

    O_plus = evaluate_at(theta_0 + h)
    O_minus = evaluate_at(theta_0 - h)
    return (O_plus - O_minus) / (2.0 * h)


# ---------------------------------------------------------------------------
# Fisher information matrix from a list of measurement points.
# ---------------------------------------------------------------------------


def fisher_information(
    *,
    observable_fn: Callable[..., "object"],
    cosmography_factory: Callable[..., Cosmography],
    fiducial_params: Dict[str, float],
    parameter_names: Sequence[str],
    measurement_points: Iterable[Tuple[float, float]],
    c_km_s: float,
    step_relative: float = 1.0e-4,
    step_absolute: float = 1.0e-8,
    observable_kwargs: Optional[Dict[str, float]] = None,
    frame: Frame = Frame.LCDM_EXTRACTED,
) -> FisherMatrix:
    """Per-dataset Fisher information matrix.

    Computes
        F_{ij} = sum_n (dO_n/dtheta_i) (dO_n/dtheta_j) / sigma_n^2
    with (z_n, sigma_n) drawn from `measurement_points`. The continuous
    integral form

        F_{ij} = int (dO/dtheta_i)(dO/dtheta_j) / sigma(z)^2 dz

    is recovered by passing quadrature nodes with the redshift density
    absorbed into 1/sigma^2.

    Parameters
    ----------
    observable_fn : Callable
        Forward model: (z, cosmography, c_km_s, **kwargs) -> Tagged.
        Examples: forward_models.sn1a_distance_modulus,
        forward_models.bao_distance_DV, an adapter wrapping
        forward_models.cmb_theta_star_from_DC with r_s_comoving_Mpc
        captured.
    cosmography_factory : Callable
        (**fiducial_params) -> Cosmography. Determines the LCDM class.
    fiducial_params : Dict[str, float]
        Fiducial point in parameter space.
    parameter_names : Sequence[str]
        Subset of fiducial_params keys to include in the Fisher matrix.
        Order determines the matrix index.
    measurement_points : Iterable[(z, sigma)]
        (redshift, observable uncertainty) pairs. sigma must be positive
        and in the same units as the observable (mag for mu, Mpc for D_V,
        rad for theta_*).
    c_km_s : float
        Speed of light, explicit per R1.
    step_relative, step_absolute : float
        Finite-difference step controls. Defaults are conservative for
        smooth analytic observables of the LCDM class.
    observable_kwargs : Optional[Dict[str, float]]
        Forwarded to observable_fn (e.g., r_s for theta_* adapter).
    frame : Frame
        Output frame. Default Frame.LCDM_EXTRACTED reflects this module's
        Layer-4 (LCDM-fit emulator) status.

    Returns
    -------
    FisherMatrix
        Symmetric P x P matrix tagged with `frame`.
    """
    names = tuple(parameter_names)
    if not names:
        raise ValueError(
            "fisher_information requires at least one parameter."
        )
    for n in names:
        if n not in fiducial_params:
            raise KeyError(
                f"parameter_name {n!r} not in fiducial_params "
                f"keys {list(fiducial_params.keys())}."
            )

    points = list(measurement_points)
    if not points:
        raise ValueError(
            "fisher_information requires at least one measurement point."
        )

    P = len(names)
    F = np.zeros((P, P), dtype=float)

    for z_n, sigma_n in points:
        if sigma_n <= 0.0:
            raise ValueError(
                f"sigma must be positive; got sigma={sigma_n} at z={z_n}."
            )
        partials = np.empty(P, dtype=float)
        for i, name_i in enumerate(names):
            partials[i] = partial_observable_finite_difference(
                observable_fn=observable_fn,
                cosmography_factory=cosmography_factory,
                fiducial_params=fiducial_params,
                parameter_name=name_i,
                z=z_n,
                c_km_s=c_km_s,
                step_relative=step_relative,
                step_absolute=step_absolute,
                observable_kwargs=observable_kwargs,
            )
        weight = 1.0 / (sigma_n * sigma_n)
        F += weight * np.outer(partials, partials)

    return FisherMatrix(matrix=F, parameter_names=names, frame=frame)


# ---------------------------------------------------------------------------
# Self-test — runs under `python -m proofs.cosmology.lib.fisher`.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import math

    from .cosmography import flat_LCDM
    from .distances import C_LIGHT_KM_S
    from .forward_models import sn1a_distance_modulus, bao_distance_DV

    # Fiducial point: a Planck-like flat LCDM (numbers are not fitted
    # here; they exercise the API at a representative point).
    H_0_fid = 67.36
    Om_fid = 0.3153

    def lcdm_factory(*, H_0: float, Omega_m: float) -> Cosmography:
        return flat_LCDM(
            H_0=H_0, Omega_m=Omega_m, frame=Frame.LCDM_EXTRACTED
        )

    fiducial = {"H_0": H_0_fid, "Omega_m": Om_fid}

    # ============================================================
    # ANALYTIC CHECK 1 — dmu/dH_0 = -5/(H_0 ln 10), z-independent.
    # ============================================================
    # Standard cosmology Fisher result: under any flat FRW class, mu(z)
    # depends on H_0 only through the d_L = (c/H_0) * f(z; Omega_m, w)
    # prefactor, so
    #     dmu/dH_0 = -5/(H_0 ln 10)
    # with NO z-dependence. This is a classical Fisher-info benchmark
    # for SN1a; finite-difference must reproduce it to numerical
    # precision under any LCDM-class factory.
    # ============================================================
    analytic_partial_H0 = -5.0 / (H_0_fid * math.log(10.0))

    print("=" * 72)
    print("Self-test 1 — dmu/dH_0 vs analytic -5/(H_0 ln 10)")
    print("=" * 72)
    print(f"  Analytic value (z-independent):  {analytic_partial_H0:+.10f}")
    print(f"  {'z':>6} {'numerical dmu/dH_0':>22} {'rel_err':>14}")
    for z in (0.05, 0.1, 0.5, 1.0, 1.5, 2.0):
        d = partial_observable_finite_difference(
            observable_fn=sn1a_distance_modulus,
            cosmography_factory=lcdm_factory,
            fiducial_params=fiducial,
            parameter_name="H_0",
            z=z,
            c_km_s=C_LIGHT_KM_S,
            step_relative=1.0e-4,
            step_absolute=1.0e-8,
        )
        rel_err = abs(d - analytic_partial_H0) / abs(analytic_partial_H0)
        print(f"  {z:>6.2f} {d:>+22.10f} {rel_err:>14.2e}")
        assert rel_err < 1.0e-6, (
            f"Numerical dmu/dH_0 at z={z} disagrees with analytic "
            f"by rel_err={rel_err:.2e} (threshold 1e-6)."
        )
    print("  PASS — z-independent agreement at <1e-6 relative.")
    print()

    # ============================================================
    # ANALYTIC CHECK 2 — dmu/dOmega_m sign + monotone-magnitude up to z~1.
    # ============================================================
    # At fixed H_0, increasing Omega_m raises H(z) at z > 0, decreasing
    # comoving distance, decreasing d_L, decreasing mu. So
    #     dmu/dOmega_m < 0 at z > 0.
    # The magnitude grows monotonically with z up to z ~ 1 (where the
    # matter term increasingly dominates the H(z) integral and Omega_m
    # bites harder); beyond z ~ 1 the growth slows. We test both the
    # sign and the monotone-magnitude property up to z = 1.
    # ============================================================
    print("=" * 72)
    print("Self-test 2 — dmu/dOmega_m sign + monotone magnitude up to z ~ 1")
    print("=" * 72)
    print(f"  {'z':>6} {'dmu/dOm':>22}")
    prev_abs = -1.0
    for z in (0.01, 0.1, 0.5, 1.0):
        d = partial_observable_finite_difference(
            observable_fn=sn1a_distance_modulus,
            cosmography_factory=lcdm_factory,
            fiducial_params=fiducial,
            parameter_name="Omega_m",
            z=z,
            c_km_s=C_LIGHT_KM_S,
            step_relative=1.0e-4,
            step_absolute=1.0e-8,
        )
        print(f"  {z:>6.2f} {d:>+22.10f}")
        assert d < 0.0, (
            f"dmu/dOmega_m at z={z} should be negative; got {d:+e}."
        )
        if prev_abs >= 0.0:
            assert abs(d) > prev_abs, (
                f"|dmu/dOmega_m| should grow monotonically up to z ~ 1; "
                f"at z={z} got {abs(d):.4e} not greater than "
                f"previous {prev_abs:.4e}."
            )
        prev_abs = abs(d)
    print("  PASS — sign negative, magnitude monotone up to z = 1.")
    print()

    # ============================================================
    # CHECK 3 — Fisher matrix on an SN1a-like 30-point sample.
    # ============================================================
    # Build z in [0.05, 1.5] with constant sigma_mu = 0.15 mag (a
    # canonical benchmark; e.g., Tegmark+ 2001). Verify:
    #   - matrix symmetric
    #   - matrix positive definite
    #   - F[H_0, H_0] = N / sigma^2 * (5/(H_0 ln 10))^2 exactly,
    #     since dmu/dH_0 is z-independent
    # ============================================================
    print("=" * 72)
    print("Self-test 3 — Fisher matrix on 30-point SN1a-like sample")
    print("=" * 72)
    z_sample = np.linspace(0.05, 1.5, 30)
    sigma_mu = 0.15
    points = [(float(z), sigma_mu) for z in z_sample]

    F = fisher_information(
        observable_fn=sn1a_distance_modulus,
        cosmography_factory=lcdm_factory,
        fiducial_params=fiducial,
        parameter_names=("H_0", "Omega_m"),
        measurement_points=points,
        c_km_s=C_LIGHT_KM_S,
    )

    M = F.matrix
    print(f"  Frame:            {F.frame.value}")
    print(f"  Parameter names:  {F.parameter_names}")
    print(f"  Matrix:")
    print(f"    F[H_0, H_0]    = {M[0, 0]:.6e}")
    print(f"    F[H_0, Om ]    = {M[0, 1]:.6e}")
    print(f"    F[Om,  Om ]    = {M[1, 1]:.6e}")

    assert abs(M[0, 1] - M[1, 0]) < 1.0e-10, (
        "Fisher matrix must be symmetric."
    )
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    assert M[0, 0] > 0 and M[1, 1] > 0 and det > 0, (
        "Fisher matrix must be positive definite."
    )

    F_HH_analytic = (
        len(points) / (sigma_mu * sigma_mu)
        * analytic_partial_H0 ** 2
    )
    rel_err_HH = abs(M[0, 0] - F_HH_analytic) / F_HH_analytic
    print(f"  F[H_0, H_0] analytic   = {F_HH_analytic:.6e}")
    print(f"  rel_err on F[H_0, H_0] = {rel_err_HH:.2e}")
    assert rel_err_HH < 1.0e-6, (
        "F[H_0, H_0] must match analytic exactly (z-independent partial)."
    )
    print("  PASS — symmetric, positive-definite, F[H_0, H_0] analytic match.")
    print()

    # ============================================================
    # CHECK 4 — D_V (BAO) Fisher info exists and is well-behaved.
    # ============================================================
    # No closed-form check required by the deliverable; just exercise
    # the API for a non-mu observable to confirm composition works.
    # ============================================================
    print("=" * 72)
    print("Self-test 4 — D_V (BAO) Fisher info smoke test")
    print("=" * 72)
    bao_points = [(0.35, 25.0), (0.57, 25.0), (0.80, 30.0)]  # z, sigma in Mpc
    F_bao = fisher_information(
        observable_fn=bao_distance_DV,
        cosmography_factory=lcdm_factory,
        fiducial_params=fiducial,
        parameter_names=("H_0", "Omega_m"),
        measurement_points=bao_points,
        c_km_s=C_LIGHT_KM_S,
    )
    print(f"  Frame:           {F_bao.frame.value}")
    print(f"  F[H_0, H_0]    = {F_bao.matrix[0, 0]:.6e}")
    print(f"  F[H_0, Om ]    = {F_bao.matrix[0, 1]:.6e}")
    print(f"  F[Om,  Om ]    = {F_bao.matrix[1, 1]:.6e}")
    assert F_bao.frame is Frame.LCDM_EXTRACTED
    assert abs(F_bao.matrix[0, 1] - F_bao.matrix[1, 0]) < 1.0e-10
    det_bao = (
        F_bao.matrix[0, 0] * F_bao.matrix[1, 1]
        - F_bao.matrix[0, 1] * F_bao.matrix[1, 0]
    )
    assert F_bao.matrix[0, 0] > 0 and F_bao.matrix[1, 1] > 0 and det_bao > 0
    print("  PASS — D_V Fisher matrix symmetric and positive-definite.")
    print()

    print("=" * 72)
    print("ALL FISHER SELF-TESTS PASS")
    print("=" * 72)
