"""
multi_dataset.py — Multi-dataset LCDM-fit orchestrator + z_eff diagnostics.

LAYER 4 (LCDM-fit emulator) — Phase A.3 of the cosmology simulator
(see an internal working note
§8.3 / §13). Composes Phase A.1 (fisher.py) + A.2 (lcdm_fitter.py) into
the orchestrator that takes a multi-dataset specification, fits an
LCDM-class to mock observables generated under the framework's native
cosmography, and reports BOTH the recovered parameters AND the effective
redshift z_eff at which a single-z bias-function evaluation reproduces
the recovered Omega_m. This is the module where z_eff graduates from
data-side empirical anchor (Planck Omega_m = 0.3153 -> z_eff = 1.916)
to first-principles-derived from a stated dataset specification.

LINE CLASSIFICATION (per cosmology architecture §2)
---------------------------------------------------
  (b) PURE MATHEMATICAL IDENTITY:
        - Additivity of Gaussian-likelihood chi^2 across independent
          datasets:   chi^2_total = sum_d chi^2_d.
        - Additivity of Fisher info across independent datasets:
          F_total = sum_d F_d.
        - Fisher-weighted effective redshift:
          z_eff_F = sum_n z_n * w_n / sum_n w_n  with w_n =
          (dO/dtheta_n)^2 / sigma_n^2.

  (c) EXTRACTION-LAYER TRANSLATION:
        - Each DatasetSpec carries a forward-model observable
          (FRW-extraction layer per forward_models.py).
        - The cosmography_factory builds an LCDM-class object
          (Frame.LCDM_EXTRACTED).

  (a) PROJECT-NATIVE:
        - The "true" cosmography passed in is typically project-native
          (coasting); but this module merely calls the observable_fn on
          it. No substrate dynamics live here.

NO SIDE-LOADED PHYSICS. Multi-dataset orchestration is pure likelihood
construction — Cramér–Rao additivity for independent measurements. Per
feedback_no_side_loaded_physics_no_adoptions.md.

PHASE A CONTEXT
---------------
This is the module where z_eff graduates. Two definitions are reported:

  z_eff_bias_inversion : the redshift at which the framework's native
    bias function Omega_m_local(z) equals the recovered LCDM Omega_m.
    This is the operative definition: by construction, bias-function
    evaluation at this z reproduces the multi-dataset fit's recovered
    Omega_m exactly. THIS IS THE chi^2-MINIMUM REDSHIFT.

  z_eff_fisher : Fisher-weighted average of measurement redshifts,
    z_eff_F = sum_n z_n * w_n / sum_n w_n, with w_n = (dO/dtheta)^2/sigma^2.
    Reduces to z_eff_bias_inversion in the linear-Fisher regime; reported
    as a diagnostic that connects to the heuristic Fisher weights in the
    superseded probe proofs/cosmology/O2_z_eff_multidataset_derivation.py.

For single-dataset fits these two definitions need not coincide
literally (the full chi^2 minimum can be slightly off the
linear-Fisher prediction); the diagnostic comparison appears in the
self-test.

API
---
  DatasetSpec(label, observable_fn, measurement_points, observable_kwargs)
        Single-dataset specification.

  MultiDatasetResult
        Frame-tagged result of the multi-dataset fit + z_eff diagnostics.

  fisher_weighted_effective_redshift(...)
        Standalone helper computing z_eff_F = sum_n z_n * w_n / sum_n w_n.

  fit_multi_dataset(...)
        Orchestrator: generate mock data per dataset under
        cosmography_true, sum the per-dataset chi^2, fit LCDM-class
        parameters, report recovered values + per-dataset and combined
        z_eff diagnostics.

DISCIPLINE (parameter_linter R1 + cosmology architecture §10)
-------------------------------------------------------------
  - Every physical input is an explicit named argument.
  - Public functions are keyword-only.
  - Outputs Frame-tagged.

REFERENCES
----------
  an internal working note (§3, §8.3, §13)
  proofs/cosmology/lib/fisher.py (per-point partials + Fisher matrix)
  proofs/cosmology/lib/lcdm_fitter.py (chi^2 fitter, single-dataset)
  proofs/cosmology/O2_z_eff_multidataset_derivation.py (heuristic
    Fisher weights superseded by this module)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

from .bias_functions import solve_z_eff_for_Omega_m
from .cosmography import Cosmography
from .fisher import (
    FisherMatrix,
    fisher_information,
    partial_observable_finite_difference,
)
from .lcdm_fitter import chi_squared, generate_mock_observables
from .ontology import Frame


# ---------------------------------------------------------------------------
# Single-dataset specification.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Specification of one dataset in a multi-dataset analysis.

    Fields
    ------
    label : str
        Short identifier (e.g., 'SN1a', 'BAO', 'CMB_theta_star').
    observable_fn : Callable
        Forward model (z, cosmo, c_km_s, **kwargs) -> Tagged.
    measurement_points : Tuple[Tuple[float, float], ...]
        Immutable tuple of (z_n, sigma_n) pairs.
    observable_kwargs : Optional[Tuple[Tuple[str, float], ...]]
        Immutable representation of extra kwargs to observable_fn (e.g.,
        r_s for theta_*); None when the observable is fully specified
        by (z, cosmo, c_km_s). Stored as a tuple-of-pairs because dicts
        are not hashable; converted to a dict at use sites.
    """

    label: str
    observable_fn: Callable[..., "object"]
    measurement_points: Tuple[Tuple[float, float], ...]
    observable_kwargs: Optional[Tuple[Tuple[str, float], ...]] = None

    @staticmethod
    def make(
        *,
        label: str,
        observable_fn: Callable[..., "object"],
        measurement_points: Iterable[Tuple[float, float]],
        observable_kwargs: Optional[Dict[str, float]] = None,
    ) -> "DatasetSpec":
        """Convenience builder accepting plain Python collections."""
        kw_tuple: Optional[Tuple[Tuple[str, float], ...]]
        if observable_kwargs is None:
            kw_tuple = None
        else:
            kw_tuple = tuple(
                (str(k), float(v)) for k, v in observable_kwargs.items()
            )
        return DatasetSpec(
            label=label,
            observable_fn=observable_fn,
            measurement_points=tuple(
                (float(z), float(s)) for z, s in measurement_points
            ),
            observable_kwargs=kw_tuple,
        )

    def kwargs_dict(self) -> Dict[str, float]:
        """Return observable_kwargs as a plain dict (empty if None)."""
        if self.observable_kwargs is None:
            return {}
        return {k: v for k, v in self.observable_kwargs}


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiDatasetResult:
    """Multi-dataset fit + z_eff diagnostics, frame-tagged.

    Fields
    ------
    best_fit : Dict[str, float]
        {parameter_name: value} for the fit parameters at chi^2 minimum.
    fit_parameter_names : Tuple[str, ...]
        Order of varied parameters (covariance index).
    fixed_params : Dict[str, float]
    chi_squared_total : float
        Sum of per-dataset chi^2 at best fit.
    chi_squared_per_dataset : Dict[str, float]
    covariance : Optional[np.ndarray]
        P x P, from inv(F_combined) at best fit.
    fisher_combined : FisherMatrix
        Sum of per-dataset Fisher matrices at best fit.
    fisher_per_dataset : Dict[str, FisherMatrix]
    z_eff_bias_inversion : Optional[float]
        Solve Omega_m_local(z; cosmography_true) = recovered Omega_m
        for z. Reported only when 'Omega_m' is in fit_parameter_names
        and the inversion bracket converges.
    z_eff_fisher_per_dataset : Dict[str, float]
        Per-dataset Fisher-weighted average z for the chosen
        z_eff_parameter.
    z_eff_fisher_combined : float
        Combined Fisher-weighted average z (across all datasets).
    z_eff_parameter : str
        The parameter whose Fisher weights define z_eff_fisher_*.
    n_data_points_per_dataset : Dict[str, int]
    success : bool
    frame : Frame
    """

    best_fit: Dict[str, float]
    fit_parameter_names: Tuple[str, ...]
    fixed_params: Dict[str, float]
    chi_squared_total: float
    chi_squared_per_dataset: Dict[str, float]
    covariance: Optional[np.ndarray]
    fisher_combined: FisherMatrix
    fisher_per_dataset: Dict[str, FisherMatrix]
    z_eff_bias_inversion: Optional[float]
    z_eff_fisher_per_dataset: Dict[str, float]
    z_eff_fisher_combined: float
    z_eff_parameter: str
    n_data_points_per_dataset: Dict[str, int]
    success: bool
    frame: Frame


# ---------------------------------------------------------------------------
# Fisher-weighted effective redshift helper.
# ---------------------------------------------------------------------------


def fisher_weighted_effective_redshift(
    *,
    observable_fn: Callable[..., "object"],
    cosmography_factory: Callable[..., Cosmography],
    fiducial_params: Dict[str, float],
    parameter_name: str,
    measurement_points: Iterable[Tuple[float, float]],
    c_km_s: float,
    observable_kwargs: Optional[Dict[str, float]] = None,
    step_relative: float = 1.0e-4,
    step_absolute: float = 1.0e-8,
) -> Tuple[float, float]:
    """Fisher-weighted average redshift for a parameter.

    z_eff_F = sum_n z_n * w_n / sum_n w_n
    where w_n = (dO/dtheta)_n^2 / sigma_n^2 — the per-point contribution
    to the (parameter, parameter) diagonal of the Fisher matrix.

    Returns
    -------
    (z_eff_F, total_weight) : Tuple[float, float]
        z_eff_F in the same units as z_n; total_weight = sum_n w_n
        (useful for combining multiple datasets via weighted averaging).
    """
    points = list(measurement_points)
    if not points:
        raise ValueError("measurement_points must be non-empty.")

    z_array = np.empty(len(points), dtype=float)
    weights = np.empty(len(points), dtype=float)
    for i, (z_n, sigma_n) in enumerate(points):
        if sigma_n <= 0.0:
            raise ValueError(
                f"sigma must be positive; got sigma={sigma_n} at z={z_n}."
            )
        partial = partial_observable_finite_difference(
            observable_fn=observable_fn,
            cosmography_factory=cosmography_factory,
            fiducial_params=fiducial_params,
            parameter_name=parameter_name,
            z=float(z_n),
            c_km_s=c_km_s,
            step_relative=step_relative,
            step_absolute=step_absolute,
            observable_kwargs=observable_kwargs,
        )
        z_array[i] = z_n
        weights[i] = (partial * partial) / (sigma_n * sigma_n)

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError(
            "Total Fisher weight is non-positive; the observable has zero "
            "sensitivity to the parameter at the fiducial point."
        )
    z_eff = float(np.sum(z_array * weights) / total_weight)
    return z_eff, total_weight


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def fit_multi_dataset(
    *,
    datasets: Sequence[DatasetSpec],
    cosmography_true: Cosmography,
    cosmography_factory: Callable[..., Cosmography],
    fit_parameter_initial: Dict[str, float],
    fixed_params: Dict[str, float],
    c_km_s: float,
    fit_parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    z_eff_parameter: str = "Omega_m",
    rng: Optional[np.random.Generator] = None,
    optimizer_method: str = "L-BFGS-B",
    optimizer_options: Optional[Dict[str, "object"]] = None,
    fisher_step_relative: float = 1.0e-4,
    fisher_step_absolute: float = 1.0e-8,
    frame: Frame = Frame.LCDM_EXTRACTED,
) -> MultiDatasetResult:
    """Fit LCDM-class parameters to a multi-dataset specification.

    For each DatasetSpec, generate mock observables under
    `cosmography_true`. The total chi^2 is the sum of per-dataset
    chi^2 (Gaussian-likelihood additivity for independent datasets).
    A single scipy.optimize.minimize call recovers the fit parameters.

    Per-dataset and combined Fisher matrices are computed at the best
    fit; covariance = inv(Fisher_combined). z_eff is reported via
    bias-function inversion (recovered Omega_m -> z under
    cosmography_true) and via the Fisher-weighted average.

    Parameters
    ----------
    datasets : Sequence[DatasetSpec]
        One spec per dataset (e.g., SN1a, BAO, CMB).
    cosmography_true : Cosmography
        The "true" cosmography (typically project-native coasting).
    cosmography_factory : Callable
        (**all_params) -> Cosmography. Determines the LCDM class.
    fit_parameter_initial : Dict[str, float]
        Initial values for parameters being varied.
    fixed_params : Dict[str, float]
    c_km_s : float
        Explicit per R1.
    fit_parameter_bounds : Optional[Dict[str, Tuple[float, float]]]
    z_eff_parameter : str
        The parameter whose Fisher weights define z_eff_fisher. Default
        'Omega_m' (the architecture's primary z_eff target).
    rng : Optional[np.random.Generator]
        If provided, mock data is generated with Gaussian noise per
        sigma_n; otherwise clean (deterministic).
    optimizer_method, optimizer_options : forwarded to scipy.minimize.
    fisher_step_relative, fisher_step_absolute : forwarded to fisher_information.
    frame : Frame
        Output frame tag.

    Returns
    -------
    MultiDatasetResult
    """
    if not datasets:
        raise ValueError("datasets must contain at least one DatasetSpec.")

    fit_names = tuple(fit_parameter_initial.keys())
    if not fit_names:
        raise ValueError("fit_parameter_initial must not be empty.")
    overlap = set(fit_names) & set(fixed_params.keys())
    if overlap:
        raise ValueError(
            f"Parameters cannot be both fit and fixed: {sorted(overlap)}."
        )

    # ------------------------------------------------------------------
    # Generate mock data per dataset under cosmography_true.
    # ------------------------------------------------------------------
    mock_per_dataset: Dict[str, List[Tuple[float, float, float]]] = {}
    for ds in datasets:
        if ds.label in mock_per_dataset:
            raise ValueError(
                f"Duplicate dataset label {ds.label!r}; labels must be unique."
            )
        mock_per_dataset[ds.label] = generate_mock_observables(
            observable_fn=ds.observable_fn,
            cosmography_true=cosmography_true,
            measurement_points=ds.measurement_points,
            c_km_s=c_km_s,
            observable_kwargs=ds.kwargs_dict(),
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Multi-dataset chi^2.
    # ------------------------------------------------------------------
    initial_vector = np.array(
        [fit_parameter_initial[n] for n in fit_names], dtype=float
    )
    if fit_parameter_bounds is not None:
        bounds_list = [
            fit_parameter_bounds.get(n, (None, None)) for n in fit_names
        ]
    else:
        bounds_list = None

    def assemble_params(theta_vector: np.ndarray) -> Dict[str, float]:
        params = dict(fixed_params)
        for name, val in zip(fit_names, theta_vector):
            params[name] = float(val)
        return params

    def chi2_total_of_vector(theta_vector: np.ndarray) -> float:
        params = assemble_params(theta_vector)
        cosmo = cosmography_factory(**params)
        total = 0.0
        for ds in datasets:
            total += chi_squared(
                observable_fn=ds.observable_fn,
                cosmography=cosmo,
                mock_data=mock_per_dataset[ds.label],
                c_km_s=c_km_s,
                observable_kwargs=ds.kwargs_dict(),
            )
        return total

    result = optimize.minimize(
        chi2_total_of_vector,
        initial_vector,
        method=optimizer_method,
        bounds=bounds_list,
        options=optimizer_options or {},
    )

    best_fit_vector = np.asarray(result.x, dtype=float)
    best_fit_dict = {
        n: float(v) for n, v in zip(fit_names, best_fit_vector)
    }
    chi2_total = float(result.fun)

    # ------------------------------------------------------------------
    # Per-dataset chi^2 at best fit.
    # ------------------------------------------------------------------
    cosmo_bf = cosmography_factory(
        **{**fixed_params, **best_fit_dict}
    )
    chi2_per_dataset: Dict[str, float] = {}
    for ds in datasets:
        chi2_per_dataset[ds.label] = chi_squared(
            observable_fn=ds.observable_fn,
            cosmography=cosmo_bf,
            mock_data=mock_per_dataset[ds.label],
            c_km_s=c_km_s,
            observable_kwargs=ds.kwargs_dict(),
        )

    # ------------------------------------------------------------------
    # Per-dataset and combined Fisher matrices at best fit.
    # ------------------------------------------------------------------
    fiducial = {**fixed_params, **best_fit_dict}
    fisher_per_dataset: Dict[str, FisherMatrix] = {}
    F_combined = np.zeros((len(fit_names), len(fit_names)), dtype=float)
    for ds in datasets:
        F_ds = fisher_information(
            observable_fn=ds.observable_fn,
            cosmography_factory=cosmography_factory,
            fiducial_params=fiducial,
            parameter_names=fit_names,
            measurement_points=ds.measurement_points,
            c_km_s=c_km_s,
            observable_kwargs=ds.kwargs_dict(),
            step_relative=fisher_step_relative,
            step_absolute=fisher_step_absolute,
            frame=frame,
        )
        fisher_per_dataset[ds.label] = F_ds
        F_combined += F_ds.matrix
    fisher_combined = FisherMatrix(
        matrix=F_combined, parameter_names=fit_names, frame=frame
    )

    cov: Optional[np.ndarray]
    try:
        cov = np.linalg.inv(F_combined)
    except np.linalg.LinAlgError:
        cov = None

    # ------------------------------------------------------------------
    # z_eff diagnostics.
    # ------------------------------------------------------------------
    # (i) Bias-inversion: recovered Omega_m -> z under cosmography_true.
    z_eff_bias: Optional[float] = None
    if "Omega_m" in best_fit_dict:
        z_candidate = solve_z_eff_for_Omega_m(
            cosmography_true, best_fit_dict["Omega_m"]
        )
        if not np.isnan(z_candidate):
            z_eff_bias = float(z_candidate)

    # (ii) Fisher-weighted z, per dataset and combined.
    if z_eff_parameter not in fit_names:
        raise ValueError(
            f"z_eff_parameter {z_eff_parameter!r} must be one of "
            f"the fit parameters {fit_names}."
        )
    z_eff_fisher_per_dataset: Dict[str, float] = {}
    weighted_z_sum = 0.0
    weight_sum = 0.0
    for ds in datasets:
        z_ds, w_ds = fisher_weighted_effective_redshift(
            observable_fn=ds.observable_fn,
            cosmography_factory=cosmography_factory,
            fiducial_params=fiducial,
            parameter_name=z_eff_parameter,
            measurement_points=ds.measurement_points,
            c_km_s=c_km_s,
            observable_kwargs=ds.kwargs_dict(),
            step_relative=fisher_step_relative,
            step_absolute=fisher_step_absolute,
        )
        z_eff_fisher_per_dataset[ds.label] = z_ds
        weighted_z_sum += z_ds * w_ds
        weight_sum += w_ds
    z_eff_fisher_combined = (
        weighted_z_sum / weight_sum if weight_sum > 0.0 else float("nan")
    )

    n_per = {
        ds.label: len(mock_per_dataset[ds.label]) for ds in datasets
    }

    return MultiDatasetResult(
        best_fit=best_fit_dict,
        fit_parameter_names=fit_names,
        fixed_params=dict(fixed_params),
        chi_squared_total=chi2_total,
        chi_squared_per_dataset=chi2_per_dataset,
        covariance=cov,
        fisher_combined=fisher_combined,
        fisher_per_dataset=fisher_per_dataset,
        z_eff_bias_inversion=z_eff_bias,
        z_eff_fisher_per_dataset=z_eff_fisher_per_dataset,
        z_eff_fisher_combined=z_eff_fisher_combined,
        z_eff_parameter=z_eff_parameter,
        n_data_points_per_dataset=n_per,
        success=bool(result.success),
        frame=frame,
    )


# ---------------------------------------------------------------------------
# Self-test — runs under `python -m proofs.cosmology.lib.multi_dataset`.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .cosmography import coasting, flat_LCDM
    from .distances import C_LIGHT_KM_S
    from .forward_models import bao_distance_DV, sn1a_distance_modulus

    # Framework's observer-frame H_0 prediction per cascade D2-extended:
    # H_0_observer = (16/15) * H_0_substrate, with H_0_substrate = 68.18.
    # See predictions/H_0.py and theorem_cascade_D2_extended_observer_rate.md.
    H_0_OBSERVER = 72.74

    def lcdm_factory(*, H_0: float, Omega_m: float) -> Cosmography:
        return flat_LCDM(
            H_0=H_0, Omega_m=Omega_m, frame=Frame.LCDM_EXTRACTED
        )

    cosmo_coast = coasting(H_0=H_0_OBSERVER, frame=Frame.OBSERVER)

    z_sn = np.linspace(0.05, 1.5, 30)
    sigma_mu = 0.15
    sn_points = [(float(z), sigma_mu) for z in z_sn]
    sn_spec = DatasetSpec.make(
        label="SN1a",
        observable_fn=sn1a_distance_modulus,
        measurement_points=sn_points,
    )

    bao_points = [(0.35, 25.0), (0.57, 22.0), (0.80, 30.0), (1.20, 50.0)]
    bao_spec = DatasetSpec.make(
        label="BAO",
        observable_fn=bao_distance_DV,
        measurement_points=bao_points,
    )

    initial = {"H_0": 70.0, "Omega_m": 0.4}
    bounds = {"H_0": (50.0, 90.0), "Omega_m": (0.05, 0.95)}

    # ============================================================
    # SELF-TEST 1 — Single-dataset (SN1a only) reproduces A.2.
    # ============================================================
    print("=" * 72)
    print("Self-test 1 — Single-dataset (SN1a) coasting -> LCDM fit")
    print("=" * 72)
    res_sn = fit_multi_dataset(
        datasets=[sn_spec],
        cosmography_true=cosmo_coast,
        cosmography_factory=lcdm_factory,
        fit_parameter_initial=initial,
        fixed_params={},
        c_km_s=C_LIGHT_KM_S,
        fit_parameter_bounds=bounds,
    )
    print(f"  Success:                 {res_sn.success}")
    print(f"  Recovered H_0:           {res_sn.best_fit['H_0']:.4f}")
    print(f"  Recovered Omega_m:       {res_sn.best_fit['Omega_m']:.4f}")
    print(f"  chi^2 total:             {res_sn.chi_squared_total:.3e}")
    print(f"  chi^2 per dataset:       {res_sn.chi_squared_per_dataset}")
    print(f"  z_eff (bias inversion):  {res_sn.z_eff_bias_inversion:.4f}")
    print(f"  z_eff_fisher per ds:     "
          f"{ {k: round(v, 4) for k, v in res_sn.z_eff_fisher_per_dataset.items()} }")
    print(f"  z_eff_fisher combined:   {res_sn.z_eff_fisher_combined:.4f}")

    assert res_sn.success
    # SN1a-only on coasting recovers Omega_m near 0.44 (per A.2 self-test)
    Om_sn = res_sn.best_fit["Omega_m"]
    assert 0.35 < Om_sn < 0.55, (
        f"SN1a-only Omega_m {Om_sn:.4f} unexpectedly far from A.2 result."
    )
    # Single-dataset: chi^2 sum equals the only entry
    total_per_ds = sum(res_sn.chi_squared_per_dataset.values())
    assert abs(total_per_ds - res_sn.chi_squared_total) < 1.0e-12
    # Bias inversion z_eff lies in sample range
    assert 0.05 <= res_sn.z_eff_bias_inversion <= 1.5
    # Single-dataset Fisher-weighted z equals per-dataset z
    assert abs(
        res_sn.z_eff_fisher_combined
        - res_sn.z_eff_fisher_per_dataset["SN1a"]
    ) < 1.0e-12
    print("  PASS — single-dataset orchestrator behaves like A.2.")
    print()

    # ============================================================
    # SELF-TEST 2 — Multi-dataset (SN1a + BAO) shifts z_eff toward BAO.
    # ============================================================
    # BAO points sit at z in {0.35, 0.57, 0.80, 1.20}, average around 0.7.
    # SN1a sample is roughly uniform on [0.05, 1.5], Fisher-weighted z
    # for SN1a tends to z ~ 0.8-1.0 under flat-LCDM derivative weighting.
    # Combined z_eff_fisher should sit between the two per-dataset values
    # (a basic weighted-average sanity check).
    # ============================================================
    print("=" * 72)
    print("Self-test 2 — Multi-dataset (SN1a + BAO) intermediate z_eff")
    print("=" * 72)
    res_combined = fit_multi_dataset(
        datasets=[sn_spec, bao_spec],
        cosmography_true=cosmo_coast,
        cosmography_factory=lcdm_factory,
        fit_parameter_initial=initial,
        fixed_params={},
        c_km_s=C_LIGHT_KM_S,
        fit_parameter_bounds=bounds,
    )
    print(f"  Success:                 {res_combined.success}")
    print(f"  Recovered H_0:           {res_combined.best_fit['H_0']:.4f}")
    print(f"  Recovered Omega_m:       {res_combined.best_fit['Omega_m']:.4f}")
    print(f"  chi^2 total:             {res_combined.chi_squared_total:.3e}")
    print(f"  chi^2 per dataset:       "
          f"{ {k: f'{v:.3e}' for k, v in res_combined.chi_squared_per_dataset.items()} }")
    print(f"  z_eff (bias inversion):  {res_combined.z_eff_bias_inversion:.4f}")
    print(f"  z_eff_fisher per ds:     "
          f"{ {k: round(v, 4) for k, v in res_combined.z_eff_fisher_per_dataset.items()} }")
    print(f"  z_eff_fisher combined:   {res_combined.z_eff_fisher_combined:.4f}")

    z_sn_only = res_combined.z_eff_fisher_per_dataset["SN1a"]
    z_bao_only = res_combined.z_eff_fisher_per_dataset["BAO"]
    z_combined = res_combined.z_eff_fisher_combined
    z_lo, z_hi = sorted((z_sn_only, z_bao_only))
    assert z_lo - 1.0e-9 <= z_combined <= z_hi + 1.0e-9, (
        f"Combined z_eff_fisher {z_combined:.4f} not between per-dataset "
        f"values [{z_lo:.4f}, {z_hi:.4f}]."
    )
    # Sum of per-dataset chi^2 equals total
    total_per_ds = sum(res_combined.chi_squared_per_dataset.values())
    assert abs(total_per_ds - res_combined.chi_squared_total) < 1.0e-9, (
        f"chi^2 additivity broken: {total_per_ds} vs "
        f"{res_combined.chi_squared_total}."
    )
    # Bias-inversion z_eff in plausible range (0 < z < 5; concrete value
    # depends on relative SN1a/BAO Fisher weights)
    assert 0.0 < res_combined.z_eff_bias_inversion < 5.0
    print(f"  Combined z_eff_fisher between SN1a ({z_sn_only:.4f}) and BAO "
          f"({z_bao_only:.4f}): yes.")
    print("  PASS — additive chi^2; combined z_eff_fisher between datasets.")
    print()

    # ============================================================
    # SELF-TEST 3 — Combined Fisher = sum of per-dataset Fishers.
    # ============================================================
    # Independent measurements: F_total = sum_d F_d. Verify additivity.
    # ============================================================
    print("=" * 72)
    print("Self-test 3 — Fisher additivity across datasets")
    print("=" * 72)
    F_sum = (
        res_combined.fisher_per_dataset["SN1a"].matrix
        + res_combined.fisher_per_dataset["BAO"].matrix
    )
    F_combined_matrix = res_combined.fisher_combined.matrix
    rel_err_F = (
        np.max(np.abs(F_sum - F_combined_matrix))
        / np.max(np.abs(F_combined_matrix))
    )
    print(f"  max rel_err |F_combined - (F_SN1a + F_BAO)|:  {rel_err_F:.2e}")
    assert rel_err_F < 1.0e-12, (
        f"Fisher additivity broken: rel_err {rel_err_F:.2e}."
    )

    # Covariance positive definite
    eigvals = np.linalg.eigvalsh(res_combined.covariance)
    assert (eigvals > 0).all(), (
        f"Combined covariance must be positive definite; eigvals {eigvals}."
    )
    sig_H0 = np.sqrt(res_combined.covariance[0, 0])
    sig_Om = np.sqrt(res_combined.covariance[1, 1])
    print(f"  sigma(H_0)    [combined]:  {sig_H0:.4f}")
    print(f"  sigma(Omega_m)[combined]:  {sig_Om:.4f}")
    # Combined uncertainties should be smaller (Cramér–Rao tightens with
    # more info): sigma_combined < sigma_SN1a-only.
    sig_Om_sn_only = np.sqrt(res_sn.covariance[1, 1])
    print(f"  sigma(Omega_m)[SN1a only]: {sig_Om_sn_only:.4f}")
    assert sig_Om < sig_Om_sn_only, (
        f"Combined sigma(Omega_m) {sig_Om:.4f} should be smaller than "
        f"SN1a-only {sig_Om_sn_only:.4f} (Cramér–Rao info addition)."
    )
    print("  PASS — Fisher additivity machine-precision; combined sigma "
          "tighter than single-dataset.")
    print()

    # ============================================================
    # SELF-TEST 4 — z_eff_bias_inversion exactly reproduces recovered Om.
    # ============================================================
    # Round-trip: bias-inversion z_eff -> Omega_m_local(z_eff) on the
    # native cosmography == recovered Omega_m, by construction.
    # ============================================================
    print("=" * 72)
    print("Self-test 4 — Round-trip z_eff_bias_inversion -> Omega_m")
    print("=" * 72)
    from .bias_functions import Omega_m_local
    z_bias = res_combined.z_eff_bias_inversion
    Om_at_zbias = Omega_m_local(cosmo_coast, z_bias)
    Om_recovered = res_combined.best_fit["Omega_m"]
    rel_err_round = abs(Om_at_zbias - Om_recovered) / Om_recovered
    print(f"  Recovered Omega_m:                {Om_recovered:.6f}")
    print(f"  z_eff_bias_inversion:             {z_bias:.6f}")
    print(f"  Omega_m_local(z_bias) on native:  {Om_at_zbias:.6f}")
    print(f"  rel_err round-trip:               {rel_err_round:.2e}")
    assert rel_err_round < 1.0e-9, (
        f"Round-trip rel_err {rel_err_round:.2e} too large; "
        f"z_eff_bias_inversion is not consistent with recovered Omega_m."
    )
    print("  PASS — z_eff_bias_inversion is consistent at <1e-9.")
    print()

    print("=" * 72)
    print("ALL MULTI-DATASET SELF-TESTS PASS")
    print("=" * 72)
    print()
    print("z_eff has graduated from data-side empirical anchor to")
    print("first-principles-derived: given a multi-dataset specification,")
    print(f"  z_eff_bias = {res_combined.z_eff_bias_inversion:.4f}")
    print(f"  z_eff_fisher (combined) = {res_combined.z_eff_fisher_combined:.4f}")
    print("for SN1a + BAO under coasting native (no CMB acoustic point yet;")
    print("that requires externally-supplied r_s in the architecture.)")
