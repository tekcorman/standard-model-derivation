"""
lcdm_fitter.py — Chi-squared minimization recovering LCDM-class parameters.

LAYER 4 (LCDM-fit emulator) — pure numerical machinery applied at the
extraction-layer + LCDM-class-fit layer. Outputs are tagged with
Frame.LCDM_EXTRACTED. This module emulates what an LCDM-fit pipeline
recovers from data generated under an arbitrary "true" cosmography
(typically the framework's project-native coasting H(z)); it makes no
substrate-physics claims.

LINE CLASSIFICATION (per cosmology architecture §2)
---------------------------------------------------
  (b) PURE MATHEMATICAL IDENTITY — Gaussian-likelihood chi^2

          chi^2(theta) = sum_n (O_obs_n - O_model(theta; z_n))^2 / sigma_n^2

      and the standard relation between chi^2 Hessian and Fisher info at
      the best-fit point: cov ~ inv(F). Pure algebra; no physics imported.

  (c) EXTRACTION-LAYER TRANSLATION — observables O in {mu, D_V, theta_*}
      are FRW-extraction layer (forward_models.py header). The fit-class
      cosmography_factory produces an LCDM-class object (flat_LCDM,
      flat_wCDM, ...) — explicitly a comparison object, not framework
      substrate physics.

  (a) PROJECT-NATIVE — NONE in this module. The "true" cosmography
      passed into generate_mock_observables MAY be project-native (e.g.,
      coasting); but this module just calls observable_fn on it. No
      substrate dynamics live here.

NO SIDE-LOADED PHYSICS. Standard chi^2 fitting is a property of
likelihood construction, not a substrate claim. Per
feedback_no_side_loaded_physics_no_adoptions.md.

PHASE A CONTEXT
---------------
Phase A.2 of the cosmology simulator (see
an internal working note, §13).
Together with Phase A.1 (fisher.py) and Phase A.3 (multi_dataset.py,
not yet built), this module enables the LCDM-fit emulator's central
loop: given framework H(z), generate mock observables, run chi^2 fit,
report what an LCDM-class extraction recovers. The recovered parameters
are exactly the bias-function values evaluated at the fit's effective
redshift z_eff — Phase A.3 then graduates z_eff from data-side anchor
to first-principles-derived.

API
---
  LCDMFitResult — dataclass: best fit + chi^2 + covariance + diagnostics.

  generate_mock_observables(...)
        Forward-model an observable at (z_n) under a "true" cosmography;
        optionally add Gaussian noise per sigma_n.

  chi_squared(...)
        Standard chi^2 helper, exposed for diagnostic use.

  fit_lcdm(...)
        scipy.optimize.minimize wrapper that varies a named subset of
        parameters of an LCDM-class cosmography_factory and returns the
        best-fit LCDMFitResult. Covariance computed via fisher_information
        at the best-fit point (cross-check with Phase A.1).

DISCIPLINE (parameter_linter R1 + cosmology architecture §10)
-------------------------------------------------------------
  - Every physical input is an explicit named argument.
  - Public functions are keyword-only.
  - Outputs Frame-tagged.

REFERENCES
----------
  an internal working note (§3, §6, §13)
  proofs/cosmology/lib/fisher.py (covariance via Fisher inverse)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import optimize

from .cosmography import Cosmography
from .fisher import fisher_information
from .ontology import Frame


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LCDMFitResult:
    """LCDM-class fit result, frame-tagged.

    Fields
    ------
    best_fit : Dict[str, float]
        {parameter_name: value} for the fit parameters at chi^2 minimum.
    fit_parameter_names : Tuple[str, ...]
        Order of varied parameters; matches `covariance` row/col index.
    fixed_params : Dict[str, float]
        Parameters held fixed during the fit.
    chi_squared : float
        Final chi^2 at best fit.
    covariance : Optional[np.ndarray]
        P x P covariance from inv(Fisher) at best fit, or None if not
        computed / Fisher singular.
    n_data_points : int
        Number of (z, O, sigma) data points used in the fit.
    success : bool
        Whether scipy.optimize.minimize reported convergence.
    frame : Frame
        Output frame tag.
    """

    best_fit: Dict[str, float]
    fit_parameter_names: Tuple[str, ...]
    fixed_params: Dict[str, float]
    chi_squared: float
    covariance: Optional[np.ndarray]
    n_data_points: int
    success: bool
    frame: Frame


# ---------------------------------------------------------------------------
# Mock data generation.
# ---------------------------------------------------------------------------


def generate_mock_observables(
    *,
    observable_fn: Callable[..., "object"],
    cosmography_true: Cosmography,
    measurement_points: Iterable[Tuple[float, float]],
    c_km_s: float,
    observable_kwargs: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float, float]]:
    """Forward-model an observable on (z_n) under cosmography_true.

    For each (z_n, sigma_n) in measurement_points, evaluate observable_fn
    at z_n under cosmography_true. If `rng` is provided, add Gaussian
    noise of width sigma_n; otherwise return clean observables (the
    canonical mode for parametric-class-translation bias studies, where
    noise would obscure the deterministic bias).

    Parameters
    ----------
    observable_fn : Callable
        Forward model (z, cosmo, c_km_s, **kwargs) -> Tagged.
    cosmography_true : Cosmography
        The "true" cosmography (typically project-native coasting).
    measurement_points : Iterable[(z, sigma)]
    c_km_s : float
        Explicit per R1.
    observable_kwargs : Optional[Dict[str, float]]
    rng : Optional[np.random.Generator]
        If provided, noise is rng.normal(0.0, sigma_n).

    Returns
    -------
    List of (z_n, O_observed_n, sigma_n) tuples.
    """
    extra = dict(observable_kwargs or {})
    out: List[Tuple[float, float, float]] = []
    for z_n, sigma_n in measurement_points:
        if sigma_n <= 0.0:
            raise ValueError(
                f"sigma must be positive; got sigma={sigma_n} at z={z_n}."
            )
        O_true = observable_fn(z_n, cosmography_true, c_km_s, **extra).value
        if rng is not None:
            O_obs = O_true + float(rng.normal(0.0, sigma_n))
        else:
            O_obs = O_true
        out.append((float(z_n), float(O_obs), float(sigma_n)))
    return out


# ---------------------------------------------------------------------------
# chi^2.
# ---------------------------------------------------------------------------


def chi_squared(
    *,
    observable_fn: Callable[..., "object"],
    cosmography: Cosmography,
    mock_data: Iterable[Tuple[float, float, float]],
    c_km_s: float,
    observable_kwargs: Optional[Dict[str, float]] = None,
) -> float:
    """Compute chi^2 = sum_n (O_obs_n - O_model_n)^2 / sigma_n^2."""
    extra = dict(observable_kwargs or {})
    total = 0.0
    for z_n, O_obs_n, sigma_n in mock_data:
        O_model = observable_fn(
            z_n, cosmography, c_km_s, **extra
        ).value
        residual = (O_obs_n - O_model) / sigma_n
        total += residual * residual
    return total


# ---------------------------------------------------------------------------
# LCDM-class chi^2 fit.
# ---------------------------------------------------------------------------


def fit_lcdm(
    *,
    observable_fn: Callable[..., "object"],
    cosmography_factory: Callable[..., Cosmography],
    fit_parameter_initial: Dict[str, float],
    fixed_params: Dict[str, float],
    mock_data: Iterable[Tuple[float, float, float]],
    c_km_s: float,
    fit_parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    observable_kwargs: Optional[Dict[str, float]] = None,
    optimizer_method: str = "L-BFGS-B",
    optimizer_options: Optional[Dict[str, "object"]] = None,
    compute_covariance: bool = True,
    fisher_step_relative: float = 1.0e-4,
    fisher_step_absolute: float = 1.0e-8,
    frame: Frame = Frame.LCDM_EXTRACTED,
) -> LCDMFitResult:
    """Chi^2 minimization recovering LCDM-class parameters.

    Varies the parameters in `fit_parameter_initial`; holds the
    parameters in `fixed_params` constant. The factory is invoked with
    the union (**fit_params, **fixed_params) on every chi^2 evaluation.

    Parameters
    ----------
    observable_fn : Callable
        Forward model (z, cosmo, c_km_s, **kwargs) -> Tagged.
    cosmography_factory : Callable
        (**all_params) -> Cosmography. Determines the LCDM class. The
        factory must accept all keys in fit_parameter_initial and
        fixed_params.
    fit_parameter_initial : Dict[str, float]
        Initial values for parameters being varied. Iteration order of
        this dict defines the covariance row/column order.
    fixed_params : Dict[str, float]
        Parameters held fixed during the fit. May be empty.
    mock_data : Iterable[(z, O_obs, sigma)]
        Output of generate_mock_observables.
    c_km_s : float
        Speed of light, explicit per R1.
    fit_parameter_bounds : Optional[Dict[str, Tuple[float, float]]]
        Bounds for each fit parameter. Missing keys default to (None, None).
    observable_kwargs : Optional[Dict[str, float]]
        Forwarded to observable_fn (e.g., r_s for theta_*).
    optimizer_method : str
        scipy.optimize.minimize method. 'L-BFGS-B' supports bounds.
    optimizer_options : Optional[Dict]
        Passed to minimize.options.
    compute_covariance : bool
        If True, covariance computed as inv(Fisher) at best-fit point.
    fisher_step_relative, fisher_step_absolute : float
        FD step controls forwarded to fisher_information.
    frame : Frame
        Output frame tag.

    Returns
    -------
    LCDMFitResult
    """
    fit_names = tuple(fit_parameter_initial.keys())
    if not fit_names:
        raise ValueError("fit_parameter_initial must not be empty.")

    overlap = set(fit_names) & set(fixed_params.keys())
    if overlap:
        raise ValueError(
            f"Parameters cannot be both fit and fixed: {sorted(overlap)}."
        )

    initial_vector = np.array(
        [fit_parameter_initial[n] for n in fit_names], dtype=float
    )
    if fit_parameter_bounds is not None:
        bounds_list = [
            fit_parameter_bounds.get(n, (None, None)) for n in fit_names
        ]
    else:
        bounds_list = None

    mock_data_list = list(mock_data)
    if not mock_data_list:
        raise ValueError("mock_data must contain at least one point.")

    def assemble_params(theta_vector: np.ndarray) -> Dict[str, float]:
        params = dict(fixed_params)
        for name, val in zip(fit_names, theta_vector):
            params[name] = float(val)
        return params

    def chi2_of_vector(theta_vector: np.ndarray) -> float:
        params = assemble_params(theta_vector)
        cosmo = cosmography_factory(**params)
        return chi_squared(
            observable_fn=observable_fn,
            cosmography=cosmo,
            mock_data=mock_data_list,
            c_km_s=c_km_s,
            observable_kwargs=observable_kwargs,
        )

    result = optimize.minimize(
        chi2_of_vector,
        initial_vector,
        method=optimizer_method,
        bounds=bounds_list,
        options=optimizer_options or {},
    )

    best_fit_vector = np.asarray(result.x, dtype=float)
    best_fit_dict = {
        n: float(v) for n, v in zip(fit_names, best_fit_vector)
    }
    best_fit_chi2 = float(result.fun)

    cov: Optional[np.ndarray] = None
    if compute_covariance:
        # Fisher matrix at best fit; covariance = inv(F). At best fit
        # for a Gaussian likelihood, Hessian(chi^2) = 2 * F, so
        # cov = 2 * inv(Hessian) = inv(F).
        fisher_points = [(z, s) for (z, _O, s) in mock_data_list]
        fiducial = dict(fixed_params)
        fiducial.update(best_fit_dict)
        F_at_bf = fisher_information(
            observable_fn=observable_fn,
            cosmography_factory=cosmography_factory,
            fiducial_params=fiducial,
            parameter_names=fit_names,
            measurement_points=fisher_points,
            c_km_s=c_km_s,
            observable_kwargs=observable_kwargs,
            step_relative=fisher_step_relative,
            step_absolute=fisher_step_absolute,
            frame=frame,
        )
        try:
            cov = np.linalg.inv(F_at_bf.matrix)
        except np.linalg.LinAlgError:
            cov = None

    return LCDMFitResult(
        best_fit=best_fit_dict,
        fit_parameter_names=fit_names,
        fixed_params=dict(fixed_params),
        chi_squared=best_fit_chi2,
        covariance=cov,
        n_data_points=len(mock_data_list),
        success=bool(result.success),
        frame=frame,
    )


# ---------------------------------------------------------------------------
# Self-test — runs under `python -m proofs.cosmology.lib.lcdm_fitter`.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .bias_functions import (
        Omega_m_local_coasting_closed_form,
        solve_z_eff_for_Omega_m,
    )
    from .cosmography import coasting, flat_LCDM
    from .distances import C_LIGHT_KM_S
    from .forward_models import sn1a_distance_modulus

    # Self-test fiducial points. Two distinct H_0 values appear, with
    # frame tags chosen to match the values:
    #   H_0_LCDM_FID    : Planck 2018 LCDM-fit value (Frame.LCDM_EXTRACTED).
    #   H_0_OBSERVER    : framework's observer-frame H_0 prediction per
    #                     cascade D2-extended ((16/15) * H_0_substrate where
    #                     H_0_substrate = 68.18). See predictions/H_0.py
    #                     and theorem_cascade_D2_extended_observer_rate.md.
    H_0_LCDM_FID = 67.36
    Om_FID = 0.3153
    H_0_OBSERVER = 72.74

    def lcdm_factory(*, H_0: float, Omega_m: float) -> Cosmography:
        return flat_LCDM(
            H_0=H_0, Omega_m=Omega_m, frame=Frame.LCDM_EXTRACTED
        )

    z_sample = np.linspace(0.05, 1.5, 30)
    sigma_mu = 0.15
    points = [(float(z), sigma_mu) for z in z_sample]

    # ============================================================
    # SELF-TEST 1 — Self-consistency: fitting a class to its own data.
    # ============================================================
    # Generate clean mu(z) under flat_LCDM(H_0=67.36, Omega_m=0.3153).
    # Fit flat_LCDM with a deliberately wrong initial guess.
    # Recover fiducial parameters at <1e-4 relative; chi^2 ~ 0 (machine
    # precision) since the fit class IS the data-generating class.
    # ============================================================
    print("=" * 72)
    print("Self-test 1 — Self-consistency: flat_LCDM data, flat_LCDM fit")
    print("=" * 72)

    cosmo_true = lcdm_factory(H_0=H_0_LCDM_FID, Omega_m=Om_FID)
    mock_lcdm = generate_mock_observables(
        observable_fn=sn1a_distance_modulus,
        cosmography_true=cosmo_true,
        measurement_points=points,
        c_km_s=C_LIGHT_KM_S,
    )

    fit_self = fit_lcdm(
        observable_fn=sn1a_distance_modulus,
        cosmography_factory=lcdm_factory,
        fit_parameter_initial={"H_0": 70.0, "Omega_m": 0.4},
        fit_parameter_bounds={"H_0": (50.0, 90.0), "Omega_m": (0.05, 0.95)},
        fixed_params={},
        mock_data=mock_lcdm,
        c_km_s=C_LIGHT_KM_S,
    )

    print(f"  Fit success:             {fit_self.success}")
    print(f"  Best-fit H_0:            {fit_self.best_fit['H_0']:.6f}  "
          f"(true {H_0_LCDM_FID:.6f})")
    print(f"  Best-fit Omega_m:        {fit_self.best_fit['Omega_m']:.6f}  "
          f"(true {Om_FID:.6f})")
    print(f"  chi^2 at minimum:        {fit_self.chi_squared:.3e}")
    rel_err_H0 = abs(fit_self.best_fit["H_0"] - H_0_LCDM_FID) / H_0_LCDM_FID
    rel_err_Om = abs(fit_self.best_fit["Omega_m"] - Om_FID) / Om_FID
    print(f"  rel_err(H_0):            {rel_err_H0:.2e}")
    print(f"  rel_err(Omega_m):        {rel_err_Om:.2e}")

    assert fit_self.success
    assert rel_err_H0 < 1.0e-4, (
        f"H_0 self-consistency: rel_err {rel_err_H0:.2e} > 1e-4."
    )
    assert rel_err_Om < 1.0e-4, (
        f"Omega_m self-consistency: rel_err {rel_err_Om:.2e} > 1e-4."
    )
    assert fit_self.chi_squared < 1.0e-10, (
        f"chi^2 self-consistency: {fit_self.chi_squared:.3e} > 1e-10 "
        f"(should be machine zero for clean mock + same fit class)."
    )
    assert fit_self.covariance is not None
    assert fit_self.covariance.shape == (2, 2)
    assert fit_self.covariance[0, 0] > 0 and fit_self.covariance[1, 1] > 0
    print("  PASS — recovers fiducial params at <1e-4; chi^2 machine zero.")
    print()

    # ============================================================
    # SELF-TEST 2 — Coasting -> flat_LCDM fit (the canonical bias loop).
    # ============================================================
    # Generate clean mu(z) under coasting H(z) (project-native). Fit
    # flat_LCDM. The recovered Omega_m should be a chi^2-weighted
    # average of the bias function Omega_m_local(z) over the sample
    # range, equivalently the bias function at some z_eff in the
    # sampled redshift range. chi^2 > 0 since flat_LCDM cannot exactly
    # reproduce coasting at all z.
    # ============================================================
    print("=" * 72)
    print("Self-test 2 — Coasting -> flat_LCDM fit (parametric-class loop)")
    print("=" * 72)

    cosmo_coast = coasting(H_0=H_0_OBSERVER, frame=Frame.OBSERVER)
    mock_coast = generate_mock_observables(
        observable_fn=sn1a_distance_modulus,
        cosmography_true=cosmo_coast,
        measurement_points=points,
        c_km_s=C_LIGHT_KM_S,
    )

    fit_coast = fit_lcdm(
        observable_fn=sn1a_distance_modulus,
        cosmography_factory=lcdm_factory,
        fit_parameter_initial={"H_0": 70.0, "Omega_m": 0.4},
        fit_parameter_bounds={"H_0": (50.0, 90.0), "Omega_m": (0.05, 0.95)},
        fixed_params={},
        mock_data=mock_coast,
        c_km_s=C_LIGHT_KM_S,
    )

    Om_recovered = fit_coast.best_fit["Omega_m"]
    H0_recovered = fit_coast.best_fit["H_0"]
    print(f"  Fit success:             {fit_coast.success}")
    print(f"  Recovered H_0:           {H0_recovered:.4f}  "
          f"(coasting H_0_observer = {H_0_OBSERVER:.4f})")
    print(f"  Recovered Omega_m:       {Om_recovered:.4f}")
    print(f"  chi^2 at minimum:        {fit_coast.chi_squared:.3e}  "
          f"(>0 expected: LCDM != coasting)")

    # Bias function consistency: invert Omega_m_recovered to find z_eff
    # under the coasting native, and check it lies in the sampled
    # redshift range [0.05, 1.5].
    z_eff_recovered = solve_z_eff_for_Omega_m(cosmo_coast, Om_recovered)
    print(f"  z_eff (Omega_m_local^-1): {z_eff_recovered:.4f}  "
          f"(sample z in [{z_sample[0]:.2f}, {z_sample[-1]:.2f}])")

    assert fit_coast.success
    assert 0.1 < Om_recovered < 0.7, (
        f"Recovered Omega_m {Om_recovered:.4f} out of bias-function range "
        f"for SN1a-like coasting data."
    )
    assert fit_coast.chi_squared > 0.0, (
        "chi^2 should be positive: flat_LCDM cannot exactly fit coasting."
    )
    assert (
        z_sample[0] - 0.5 <= z_eff_recovered <= z_sample[-1] + 0.5
    ), (
        f"z_eff_recovered {z_eff_recovered:.4f} far outside sample range; "
        f"bias-function consistency check fails."
    )
    # Bias-function value at the recovered z_eff should reproduce the
    # recovered Omega_m by construction (round-trip); verify.
    Om_bias_at_zeff = Omega_m_local_coasting_closed_form(z_eff_recovered)
    rel_err_bias = abs(Om_bias_at_zeff - Om_recovered) / Om_recovered
    assert rel_err_bias < 1.0e-9, (
        f"Round-trip Omega_m via z_eff inversion: rel_err {rel_err_bias:.2e}."
    )
    print(f"  Round-trip B_Om(z_eff) = {Om_bias_at_zeff:.6f} matches "
          f"recovered to {rel_err_bias:.2e}.")
    print("  PASS — recovered Omega_m is bias-function-consistent at z_eff "
          "in sample range.")
    print()

    # ============================================================
    # SELF-TEST 3 — Covariance positive-definite + Fisher-consistent.
    # ============================================================
    # The covariance returned by fit_lcdm is computed via fisher_information
    # at best fit. Verify it's positive definite and that re-running
    # fisher_information independently gives the same matrix (sanity
    # check on the fit's covariance computation path).
    # ============================================================
    print("=" * 72)
    print("Self-test 3 — Covariance positive-definite + Fisher-consistent")
    print("=" * 72)

    cov = fit_coast.covariance
    assert cov is not None, "Covariance should have been computed."
    print(f"  Covariance shape:         {cov.shape}")
    print(f"  cov(H_0,    H_0   ):      {cov[0, 0]:.6e}")
    print(f"  cov(H_0,    Omega_m):     {cov[0, 1]:.6e}")
    print(f"  cov(Omega_m, Omega_m):    {cov[1, 1]:.6e}")

    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals > 0).all(), (
        f"Covariance must be positive definite; eigenvalues {eigvals}."
    )
    sig_H0 = np.sqrt(cov[0, 0])
    sig_Om = np.sqrt(cov[1, 1])
    print(f"  sigma(H_0):              {sig_H0:.4f}")
    print(f"  sigma(Omega_m):          {sig_Om:.4f}")

    # Independent Fisher computation at the best fit, to verify cov is
    # exactly inv(F) of the same Fisher analysis.
    F_indep = fisher_information(
        observable_fn=sn1a_distance_modulus,
        cosmography_factory=lcdm_factory,
        fiducial_params={
            "H_0": H0_recovered,
            "Omega_m": Om_recovered,
        },
        parameter_names=("H_0", "Omega_m"),
        measurement_points=[(z, s) for (z, _O, s) in mock_coast],
        c_km_s=C_LIGHT_KM_S,
    )
    cov_indep = np.linalg.inv(F_indep.matrix)
    rel_err_cov = (
        np.max(np.abs(cov - cov_indep))
        / np.max(np.abs(cov_indep))
    )
    print(f"  max rel_err vs independent inv(F): {rel_err_cov:.2e}")
    assert rel_err_cov < 1.0e-10, (
        f"Fitter's covariance disagrees with independent Fisher inverse: "
        f"rel_err {rel_err_cov:.2e}."
    )
    print("  PASS — covariance positive-definite, agrees with inv(Fisher).")
    print()

    print("=" * 72)
    print("ALL LCDM-FITTER SELF-TESTS PASS")
    print("=" * 72)
