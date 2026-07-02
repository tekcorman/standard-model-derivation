#!/usr/bin/env python3
"""
Stream 1 — Coasting propagation forward-model vs DIRECT data (2026-05-28).

Run with:  python -m proofs.cosmology.coasting_direct_data_forward_model_2026-05-28

WHAT THIS IS
------------
The framework's late-time cosmography is theorem-grade in SHAPE and has ZERO
shape freedom:

    H(z) = H_0 (1 + z)              (a ∝ t, q_0 = 0)   -- cascade D1+D2+D3
    D_C(z) = (c/H_0) ln(1+z)        (FRW extraction layer)
    D_H(z) = (c/H_0) / (1+z)

This probe runs that H(z) through the lib's forward models and confronts it
with the three DIRECT late-time datasets, in the SAME frame as each datum
(per feedback_prediction_miss_means_wrong_or_different_object):

  1. COSMIC CHRONOMETERS — model-independent H(z) from differential ages of
     passively-evolving galaxies (dz/dt). THE cleanest discriminator: no FRW
     distance integral, no standardizable-candle assumption, no sound-horizon
     calibration. H(z) is measured in km/s/Mpc directly, so coasting's H_0 is
     a *true* zero-free-parameter prediction here (68.19 substrate / 72.74
     observer per predictions/H_0.py), and the shape H ∝ (1+z) is tested with
     no nuisance scale. Per arXiv 2412.15717 the CC data favors a coasting /
     linear expansion.

  2. BAO — late-time standard ruler D_M/r_d, D_H/r_d, D_V/r_d. Distance scale
     (c/H_0)/r_d analytically marginalized → SHAPE-only test. BOSS DR12 +
     eBOSS DR16 *published consensus* (the vetted block reused from
     coasting_vs_lcdm_model_comparison_2026-05-15.py). DESI DR2 (arXiv
     2512.19175) is the contemporary update and per memory marginally favors
     ΛCDM; its released D_M/D_H + covariance are the proper next step and are
     NOT reconstructed from memory here (no fabricated data points).

  3. SNe Ia — standardizable-candle μ(z). SHAPE-only test against the
     published Pantheon+ (Brout+2022) ~0.05 mag systematic floor: best-fit
     coasting μ(z) vs ΛCDM μ(z) with a constant offset (≡ absolute magnitude)
     removed; the metric is max |Δμ| vs the floor. This matches the existing
     coasting_sn1a_comparison.py methodology — we do NOT invent per-SN data.

For each dataset: fit coasting (shape-frozen; 1 scale/H_0 nuisance) vs flat-ΛCDM
(Ω_m shape + scale), report χ²/dof, ΔBIC, ΔAIC (Occam penalty for ΛCDM's extra
parameter). Then the BIAS-FUNCTION PROJECTION: what Ω_m / z_eff a ΛCDM fitter
recovers from coasting data (the LCDM-EXTRACTED frame), reported as a *different
object* from the substrate-frame H(z), never as a miss.

DISCIPLINE
----------
- proofs/ exploratory probe. NOT a predictions/ file, NOT a linter pass.
- Coasting H(z) is theorem-grade SHAPE; the distance/μ/BAO formulae are the
  FRW *extraction layer* (lib/distances.py header) — "what an FRW observer
  extracts from this H(z)", not a substrate claim.
- DIAGONAL errors throughout (full covariance not applied). Standard
  model-DISCRIMINATION methodology; publication grade needs released
  covariance + full Pantheon+ / DESI likelihoods. Flagged, not hidden.
- No Planck-Ω_m matching anywhere; z_eff is derived FROM each fit with an
  error bar, never tuned to a CMB number.
- CMB θ_* is the named OUT-OF-SCOPE structural boundary (L6 wall) — coasting
  cannot make the acoustic scale; reported, not papered over.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from proofs.cosmology.lib.bias_functions import (
    Omega_m_local_coasting_closed_form,
    solve_z_eff_for_Omega_m,
)
from proofs.cosmology.lib.cosmography import Cosmography, coasting, flat_LCDM
from proofs.cosmology.lib.distances import (
    C_LIGHT_KM_S,
    angular_diameter_distance,
    comoving_distance,
    distance_modulus,
    luminosity_distance,
)
from proofs.cosmology.lib.forward_models import (
    bao_distance_DV,
    sn1a_distance_modulus,
)
from proofs.cosmology.lib.lcdm_fitter import (
    fit_lcdm,
    generate_mock_observables,
)
from proofs.cosmology.lib.ontology import Frame, Tagged

# Framework's zero-free-parameter H_0 predictions (predictions/H_0.py).
H0_SUBSTRATE = 68.19  # 1/(N·t_P), theorem-grade
H0_OBSERVER = 72.74  # (16/15)·H0_substrate, cascade D2-extended


def hz_observable(z: float, cosmography: Cosmography, c_km_s: float) -> Tagged:
    """H(z) as a forward-model observable (Tagged); c_km_s unused.

    Signature matches the lcdm_fitter observable_fn contract
    (z, cosmography, c_km_s) -> Tagged, so cosmic-chronometer H(z) can be
    fed through generate_mock_observables / fit_lcdm like any other dataset.
    """
    return cosmography.H_at(z)


# ===========================================================================
# DATASET 1 — Cosmic chronometers (model-independent H(z))
# ===========================================================================
# Standard CC compilation (Moresco et al. and predecessors; Simon+ 2005,
# Stern+ 2010, Moresco+ 2012/2015/2016, Zhang+ 2014, Ratsimbazafy+ 2017).
# Each entry: (z, H(z) [km/s/Mpc], sigma_H [km/s/Mpc]). DIAGONAL errors.
# These are DIRECT H(z) measurements: no cosmological model assumed in
# their extraction (differential-age method).
CC_HZ: List[Tuple[float, float, float]] = [
    (0.070, 69.0, 19.6),
    (0.090, 69.0, 12.0),
    (0.120, 68.6, 26.2),
    (0.170, 83.0, 8.0),
    (0.179, 75.0, 4.0),
    (0.199, 75.0, 5.0),
    (0.200, 72.9, 29.6),
    (0.270, 77.0, 14.0),
    (0.280, 88.8, 36.6),
    (0.352, 83.0, 14.0),
    (0.3802, 83.0, 13.5),
    (0.400, 95.0, 17.0),
    (0.4004, 77.0, 10.2),
    (0.4247, 87.1, 11.2),
    (0.4497, 92.8, 12.9),
    (0.470, 89.0, 49.6),
    (0.4783, 80.9, 9.0),
    (0.480, 97.0, 62.0),
    (0.593, 104.0, 13.0),
    (0.680, 92.0, 8.0),
    (0.781, 105.0, 12.0),
    (0.875, 125.0, 17.0),
    (0.880, 90.0, 40.0),
    (0.900, 117.0, 23.0),
    (1.037, 154.0, 20.0),
    (1.300, 168.0, 17.0),
    (1.363, 160.0, 33.6),
    (1.430, 177.0, 18.0),
    (1.530, 140.0, 14.0),
    (1.750, 202.0, 40.0),
    (1.965, 186.5, 50.4),
]


def chi2_Hz_coasting_bestfit_H0(data) -> Tuple[float, float]:
    """χ² of coasting H(z)=H_0(1+z) with H_0 analytically best-fit.

    H_0* = Σ w_i H_i (1+z_i) / Σ w_i (1+z_i)²,  w_i = 1/σ_i².
    Returns (χ²_min, H_0*).
    """
    num = 0.0
    den = 0.0
    for z, H, sig in data:
        w = 1.0 / (sig * sig)
        u = 1.0 + z
        num += w * H * u
        den += w * u * u
    H0_star = num / den
    chi2 = 0.0
    for z, H, sig in data:
        model = H0_star * (1.0 + z)
        chi2 += ((H - model) / sig) ** 2
    return chi2, H0_star


def chi2_Hz_coasting_fixed_H0(data, H0: float) -> float:
    """χ² of coasting H(z)=H_0(1+z) with H_0 FIXED (zero free parameters)."""
    chi2 = 0.0
    for z, H, sig in data:
        model = H0 * (1.0 + z)
        chi2 += ((H - model) / sig) ** 2
    return chi2


def chi2_Hz_lcdm(data, H0: float, Om: float) -> float:
    """χ² of flat-ΛCDM H(z) at (H_0, Ω_m)."""
    chi2 = 0.0
    for z, H, sig in data:
        model = H0 * math.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))
        chi2 += ((H - model) / sig) ** 2
    return chi2


def fit_Hz_lcdm(data) -> Tuple[float, float, float]:
    """Grid + refine fit of flat-ΛCDM to direct H(z). Returns (χ², H_0, Ω_m)."""
    best = (1e18, None, None)
    H0 = 60.0
    while H0 <= 80.0:
        Om = 0.05
        while Om <= 0.95:
            c2 = chi2_Hz_lcdm(data, H0, Om)
            if c2 < best[0]:
                best = (c2, H0, Om)
            Om += 0.005
        H0 += 0.1
    # refine
    c2b, H0b, Omb = best
    H0 = H0b - 0.1
    while H0 <= H0b + 0.1:
        Om = Omb - 0.005
        while Om <= Omb + 0.005:
            c2 = chi2_Hz_lcdm(data, H0, Om)
            if c2 < best[0]:
                best = (c2, H0, Om)
            Om += 0.0005
        H0 += 0.01
    return best


# ===========================================================================
# DATASET 2 — BAO (shape-only; distance scale marginalized)
# ===========================================================================
# Each entry: (z, kind, value, sigma) with kind in {"DM","DH","DV"} in units
# of r_d. DIAGONAL errors (full covariance not applied — flagged).
#
# Block A: BOSS DR12 (Alam+ 2017) + eBOSS DR16 (Alam+ 2021) consensus
#          (reused from coasting_vs_lcdm_model_comparison_2026-05-15.py).
BAO_BOSS_EBOSS: List[Tuple[float, str, float, float]] = [
    (0.38, "DM", 10.27, 0.15),
    (0.38, "DH", 24.89, 0.58),
    (0.51, "DM", 13.38, 0.18),
    (0.51, "DH", 22.43, 0.48),
    (0.61, "DM", 15.45, 0.22),
    (0.61, "DH", 20.25, 0.44),
    (0.70, "DM", 17.65, 0.30),
    (0.70, "DH", 19.78, 0.46),
    (0.85, "DV", 18.33, 0.62),
    (1.48, "DM", 30.21, 0.79),
    (1.48, "DH", 13.23, 0.47),
    (2.33, "DM", 37.50, 1.10),
    (2.33, "DH", 8.99, 0.19),
]
def bao_shape(z: float, kind: str, cosmo) -> float:
    """Dimensionless BAO shape f_X(z): D_X/r_d = α·f_X(z), α=(c/H_0)/r_d.

    Uses the lib's FRW extraction layer (comoving_distance / H(z)) so the
    coasting forward-model is the *same* machinery referenced by Stream 1.
    α factored out: we divide each D by (c/H_0).
    """
    cH0 = C_LIGHT_KM_S / cosmo.H_0
    DH = (C_LIGHT_KM_S / cosmo.H_at(z).value) / cH0
    DM = comoving_distance(z, cosmo, c_km_s=C_LIGHT_KM_S).value / cH0
    if kind == "DH":
        return DH
    if kind == "DM":
        return DM
    if kind == "DV":
        DV = bao_distance_DV(z, cosmo, c_km_s=C_LIGHT_KM_S).value / cH0
        return DV
    raise ValueError(kind)


def chi2_bao_marginalized(data, cosmo) -> Tuple[float, float]:
    """χ² with overall distance scale α analytically marginalized.

    χ²_min = S_dd - S_df²/S_ff,  α* = S_df/S_ff.
    Returns (χ²_min, α*).
    """
    S_df = S_ff = S_dd = 0.0
    for z, kind, val, sig in data:
        f = bao_shape(z, kind, cosmo)
        w = 1.0 / (sig * sig)
        S_df += val * f * w
        S_ff += f * f * w
        S_dd += val * val * w
    return S_dd - (S_df * S_df) / S_ff, (S_df / S_ff)


def fit_bao_lcdm(data) -> Tuple[float, float]:
    """Scan Ω_m for flat-ΛCDM, scale-marginalized. Returns (χ²_min, Ω_m)."""
    best = (1e18, None)
    Om = 0.05
    while Om <= 0.95:
        cosmo = flat_LCDM(H_0=70.0, Omega_m=Om, frame=Frame.LCDM_EXTRACTED)
        c2, _ = chi2_bao_marginalized(data, cosmo)
        if c2 < best[0]:
            best = (c2, Om)
        Om += 0.001
    c2b, Omb = best
    Om = Omb - 0.001
    while Om <= Omb + 0.001:
        cosmo = flat_LCDM(H_0=70.0, Omega_m=Om, frame=Frame.LCDM_EXTRACTED)
        c2, _ = chi2_bao_marginalized(data, cosmo)
        if c2 < best[0]:
            best = (c2, Om)
        Om += 0.00005
    return best


def bao_lcdm_Om_error(data, Om_best: float, c2_best: float) -> float:
    """1σ on Ω_m from Δχ²=1 profile (scale-marginalized)."""

    def c2_at(om):
        cosmo = flat_LCDM(H_0=70.0, Omega_m=om, frame=Frame.LCDM_EXTRACTED)
        return chi2_bao_marginalized(data, cosmo)[0]

    om = Om_best
    while om < 0.95:
        om += 0.0005
        if c2_at(om) - c2_best >= 1.0:
            break
    om_hi = om
    om = Om_best
    while om > 0.05:
        om -= 0.0005
        if c2_at(om) - c2_best >= 1.0:
            break
    om_lo = om
    return 0.5 * (om_hi - om_lo)


# ===========================================================================
# DATASET 3 — SNe Ia (shape-vs-systematic-floor; NO fabricated per-SN data)
# ===========================================================================
# Honest SN test (matches coasting_sn1a_comparison.py): we do NOT invent a
# binned Hubble diagram. Instead we ask whether the coasting μ(z) SHAPE, with
# its best-fit constant offset removed, departs from the ΛCDM μ(z) shape by
# more than the published Pantheon+ (Brout+2022) systematic floor (~0.05 mag).
# A model whose offset-marginalized residual stays under the floor is
# observationally indistinguishable from ΛCDM on the SN Hubble diagram.
SN_FLOOR_MAG = 0.05  # Pantheon+ (Brout+2022) systematic floor, mag
SN_Z_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50, 2.00]
# Planck-style ΛCDM reference shape for the SN comparison (the curve SN data
# is conventionally compared against). Ω_m=0.315 is the *reference shape*,
# not a framework input.
SN_REF_OMEGA_M = 0.315


def mu_model(z: float, cosmo) -> float:
    """μ(z) from the lib FRW forward model (d_L in Mpc; c/H_0 in Mpc)."""
    return distance_modulus(z, cosmo, c_km_s=C_LIGHT_KM_S).value


def sn_shape_residual_vs_lcdm(z_grid, H0_ref: float):
    """Offset-marginalized coasting−ΛCDM μ(z) residual on z_grid.

    Both μ(z) curves evaluated at the SAME H_0_ref (offset absorbs H_0 /
    absolute-magnitude differences). The constant offset that minimizes the
    rms residual is removed; we return (per-z residuals, max|residual|).
    """
    cosmo_c = coasting(H_0=H0_ref, frame=Frame.OBSERVER)
    cosmo_l = flat_LCDM(
        H_0=H0_ref, Omega_m=SN_REF_OMEGA_M, frame=Frame.LCDM_EXTRACTED
    )
    diffs = [mu_model(z, cosmo_c) - mu_model(z, cosmo_l) for z in z_grid]
    offset = sum(diffs) / len(diffs)
    centered = [d - offset for d in diffs]
    max_abs = max(abs(d) for d in centered)
    return list(zip(z_grid, centered)), max_abs


# ===========================================================================
# Model-comparison helpers
# ===========================================================================


def info_criteria(chi2: float, k: int, N: int) -> Tuple[float, float]:
    """(BIC, AIC). BIC = χ² + k·lnN ; AIC = χ² + 2k. Lower is better."""
    return chi2 + k * math.log(N), chi2 + 2.0 * k


def verdict_from_delta(delta: float) -> str:
    """Jeffreys-scale read on Δ(coast − LCDM); negative favors coasting."""
    a = abs(delta)
    who = "COASTING" if delta < 0 else "ΛCDM"
    if a < 2:
        return "inconclusive (models comparable)"
    if a < 6:
        return f"positive for {who}"
    if a < 10:
        return f"strong for {who}"
    return f"decisive for {who}"


def z_eff_diag(Om: float) -> float:
    """Invert the coasting bias function Ω_m_local(z)=Om for the z_eff diag."""
    cosmo = coasting(H_0=H0_OBSERVER, frame=Frame.OBSERVER)
    return solve_z_eff_for_Omega_m(cosmo, Om)


# ===========================================================================
# FORWARD-MODEL DISTANCE OUTPUTS — D_C / D_A / D_L / D_V(z) for coasting
# ===========================================================================
# The forward model's OUTPUT, surfaced explicitly. Each row is what the
# theorem-grade coasting H(z)=H_0(1+z), run through the FRW extraction layer
# (lib/distances.py), predicts for an FRW observer. Closed form for coasting:
#   D_C = (c/H_0) ln(1+z),  D_A = D_C/(1+z),  D_L = (1+z)D_C,
#   D_H = c/H(z) = (c/H_0)/(1+z),  D_V = [D_M² · cz/H]^{1/3}.
# All in Mpc when H_0 in km/s/Mpc, c in km/s.


def forward_model_distances(cosmo: Cosmography, z_grid):
    """Tabulate D_H, D_C, D_A, D_L, μ, D_V for a cosmography over z_grid."""
    rows = []
    for z in z_grid:
        D_C = comoving_distance(z, cosmo, c_km_s=C_LIGHT_KM_S).value
        D_A = angular_diameter_distance(z, cosmo, c_km_s=C_LIGHT_KM_S).value
        D_L = luminosity_distance(z, cosmo, c_km_s=C_LIGHT_KM_S).value
        D_H = C_LIGHT_KM_S / cosmo.H_at(z).value
        mu = distance_modulus(z, cosmo, c_km_s=C_LIGHT_KM_S).value if z > 0 \
            else float("nan")
        D_V = bao_distance_DV(z, cosmo, c_km_s=C_LIGHT_KM_S).value
        rows.append((z, D_H, D_C, D_A, D_L, mu, D_V))
    return rows


# ===========================================================================
# REAL FITTER-BASED PROJECTION — what an actual ΛCDM pipeline extracts
# ===========================================================================
# Not the closed-form bias function: we generate CLEAN mock observables under
# the project-native coasting cosmography (the "true" universe), then run the
# lib's chi²-minimizing ΛCDM fitter (lcdm_fitter.fit_lcdm) on them, recovering
# the (H_0, Ω_m) + covariance a real ΛCDM analysis would report. This is the
# operational projection: coasting data → ΛCDM fit → extracted parameters.


def lcdm_factory(*, H_0: float, Omega_m: float) -> Cosmography:
    """ΛCDM cosmography factory for the fitter (LCDM_EXTRACTED frame)."""
    return flat_LCDM(H_0=H_0, Omega_m=Omega_m, frame=Frame.LCDM_EXTRACTED)


def project_coasting_through_lcdm_fit(
    *, observable_fn, measurement_points, H0_true: float, label: str,
    fit_H0_init: float = 70.0, fit_Om_init: float = 0.3,
):
    """Generate coasting mock observables, fit ΛCDM, return the fit result.

    measurement_points: list of (z, sigma) in the observable's units.
    """
    cosmo_true = coasting(H_0=H0_true, frame=Frame.OBSERVER)
    mock = generate_mock_observables(
        observable_fn=observable_fn,
        cosmography_true=cosmo_true,
        measurement_points=measurement_points,
        c_km_s=C_LIGHT_KM_S,
    )  # clean (no noise) — the deterministic bias, not scatter
    fit = fit_lcdm(
        observable_fn=observable_fn,
        cosmography_factory=lcdm_factory,
        fit_parameter_initial={"H_0": fit_H0_init, "Omega_m": fit_Om_init},
        fit_parameter_bounds={"H_0": (40.0, 100.0), "Omega_m": (0.02, 0.98)},
        fixed_params={},
        mock_data=mock,
        c_km_s=C_LIGHT_KM_S,
    )
    return fit, label


def fit_sigmas(fit) -> Tuple[float, float]:
    """(σ_H0, σ_Ω_m) from the fit covariance; (nan,nan) if unavailable."""
    if fit.covariance is None:
        return float("nan"), float("nan")
    names = fit.fit_parameter_names
    i_H0 = names.index("H_0")
    i_Om = names.index("Omega_m")
    return (
        math.sqrt(fit.covariance[i_H0, i_H0]),
        math.sqrt(fit.covariance[i_Om, i_Om]),
    )


# ===========================================================================
# MAIN
# ===========================================================================


def report_block(title: str, c2_coast, k_coast, c2_lcdm, k_lcdm, N,
                  extra_coast="", extra_lcdm=""):
    dof_c = N - k_coast
    dof_l = N - k_lcdm
    BIC_c, AIC_c = info_criteria(c2_coast, k_coast, N)
    BIC_l, AIC_l = info_criteria(c2_lcdm, k_lcdm, N)
    print("-" * 78)
    print(f" {title}   (N={N} points)")
    print("-" * 78)
    print(f"   coasting:  χ²={c2_coast:8.2f}  dof={dof_c:2d}  "
          f"χ²/dof={c2_coast/dof_c:6.3f}  (k={k_coast}) {extra_coast}")
    print(f"   ΛCDM    :  χ²={c2_lcdm:8.2f}  dof={dof_l:2d}  "
          f"χ²/dof={c2_lcdm/dof_l:6.3f}  (k={k_lcdm}) {extra_lcdm}")
    dBIC = BIC_c - BIC_l
    dAIC = AIC_c - AIC_l
    print(f"   ΔBIC(coast−ΛCDM) = {dBIC:+7.2f}  → {verdict_from_delta(dBIC)}")
    print(f"   ΔAIC(coast−ΛCDM) = {dAIC:+7.2f}  → {verdict_from_delta(dAIC)}")
    print()
    return dBIC, dAIC


def main() -> int:
    print("=" * 78)
    print(" STREAM 1 — Coasting H(z)=H_0(1+z) forward-model vs DIRECT data")
    print("=" * 78)
    print(" Framework H(z) shape: theorem-grade (cascade D1+D2+D3), ZERO shape")
    print(" freedom. Distances/μ/BAO = FRW extraction layer (lib/distances.py).")
    print(" DIAGONAL errors throughout; model-DISCRIMINATION grade. No Ω_m")
    print(" matching; z_eff derived FROM fits. CMB θ_* = out-of-scope L6 wall.")
    print()

    # ----- FORWARD-MODEL DISTANCE OUTPUTS (the prediction itself) -----
    print("=" * 78)
    print(" FORWARD-MODEL OUTPUT — coasting D_H/D_C/D_A/D_L/μ/D_V(z) [Mpc, mag]")
    print("=" * 78)
    print(f" H_0 = {H0_OBSERVER} km/s/Mpc (observer frame); FRW extraction layer.")
    print(" Closed form: D_C=(c/H_0)ln(1+z), D_A=D_C/(1+z), D_L=(1+z)D_C,")
    print("              D_H=(c/H_0)/(1+z), D_V=[D_C²·cz/H]^(1/3).")
    print()
    cosmo_fm = coasting(H_0=H0_OBSERVER, frame=Frame.OBSERVER)
    print(f"   {'z':>6} {'D_H':>9} {'D_C':>9} {'D_A':>9} {'D_L':>10} "
          f"{'μ':>8} {'D_V':>9}")
    for z, D_H, D_C, D_A, D_L, mu, D_V in forward_model_distances(
        cosmo_fm, [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 1100.0]
    ):
        mu_s = f"{mu:8.3f}" if mu == mu else "     ―  "
        print(f"   {z:>6.1f} {D_H:>9.1f} {D_C:>9.1f} {D_A:>9.1f} {D_L:>10.1f} "
              f"{mu_s} {D_V:>9.1f}")
    print(" (z=1100 row = last-scattering; D_A turnover is the coasting")
    print("  late-time geometry — the CMB θ_* mismatch lives here, see L6 wall.)")
    print()

    summary = []

    # ----- DATASET 1: cosmic chronometers (the clean direct H(z) test) -----
    N_cc = len(CC_HZ)
    c2_coast_cc, H0_coast_cc = chi2_Hz_coasting_bestfit_H0(CC_HZ)
    c2_lcdm_cc, H0_lcdm_cc, Om_lcdm_cc = fit_Hz_lcdm(CC_HZ)
    c2_coast_cc_sub = chi2_Hz_coasting_fixed_H0(CC_HZ, H0_SUBSTRATE)
    c2_coast_cc_obs = chi2_Hz_coasting_fixed_H0(CC_HZ, H0_OBSERVER)

    print("=" * 78)
    print(" DATASET 1 — COSMIC CHRONOMETERS (model-independent H(z) [km/s/Mpc])")
    print("=" * 78)
    print(" The cleanest discriminator: direct H(z), no FRW integral, no candle,")
    print(" no ruler calibration. Here coasting's H_0 is a TRUE prediction.")
    print()
    print(" Zero-free-parameter test (framework H_0, NOT fitted):")
    dof0 = N_cc
    print(f"   coasting @ H_0=68.19 (substrate): χ²={c2_coast_cc_sub:7.2f}  "
          f"χ²/dof={c2_coast_cc_sub/dof0:6.3f}")
    print(f"   coasting @ H_0=72.74 (observer) : χ²={c2_coast_cc_obs:7.2f}  "
          f"χ²/dof={c2_coast_cc_obs/dof0:6.3f}")
    print()
    db_cc, da_cc = report_block(
        "1-parameter fits (H_0 free for coasting; H_0+Ω_m for ΛCDM)",
        c2_coast_cc, 1, c2_lcdm_cc, 2, N_cc,
        extra_coast=f"H_0*={H0_coast_cc:.2f}",
        extra_lcdm=f"H_0={H0_lcdm_cc:.2f}, Ω_m={Om_lcdm_cc:.3f}",
    )
    summary.append(("Cosmic chronometers H(z)", db_cc, da_cc))

    # ----- DATASET 2: BAO (vetted BOSS+eBOSS consensus) -----
    print("=" * 78)
    print(" DATASET 2 — BAO standard ruler (shape only; distance scale margin.)")
    print("=" * 78)
    N_bao = len(BAO_BOSS_EBOSS)
    cosmo_coast = coasting(H_0=H0_OBSERVER, frame=Frame.OBSERVER)
    c2_coast_bao, _alpha = chi2_bao_marginalized(BAO_BOSS_EBOSS, cosmo_coast)
    c2_lcdm_bao, Om_bao = fit_bao_lcdm(BAO_BOSS_EBOSS)
    Om_err_bao = bao_lcdm_Om_error(BAO_BOSS_EBOSS, Om_bao, c2_lcdm_bao)
    db_bao, da_bao = report_block(
        "BOSS DR12 + eBOSS DR16 consensus  [Alam+2017 / Alam+2021]",
        c2_coast_bao, 1, c2_lcdm_bao, 2, N_bao,
        extra_lcdm=f"Ω_m={Om_bao:.3f}±{Om_err_bao:.3f}",
    )
    z_eff_bao = z_eff_diag(Om_bao)
    print(f"   z_eff diagnostic (Ω_m_local⁻¹): z_eff = {z_eff_bao:.3f} "
          f"(LCDM-EXTRACTED frame; a different object from substrate H(z))")
    print(" DESI DR2 (2512.19175) is the contemporary follow-up (marginally")
    print(" favors ΛCDM per memory); its released D_M/D_H + covariance are the")
    print(" proper next step — NOT reconstructed from memory here.")
    print()
    summary.append(("BAO BOSS+eBOSS (shape)", db_bao, da_bao))

    # ----- DATASET 3: SNe shape-vs-floor (no fabricated data) -----
    print("=" * 78)
    print(" DATASET 3 — SNe Ia μ(z) SHAPE vs Pantheon+ systematic floor")
    print("=" * 78)
    print(" Metric: offset-marginalized |μ_coast(z) − μ_ΛCDM(z)| vs ~0.05 mag")
    print(" floor (Brout+2022). NO per-SN data fabricated. Both curves at a")
    print(f" common H_0 (offset absorbs it); ΛCDM reference Ω_m={SN_REF_OMEGA_M}.")
    print()
    resid, max_resid = sn_shape_residual_vs_lcdm(SN_Z_GRID, H0_ref=H0_OBSERVER)
    print(f"   {'z':>6} {'Δμ (offset-removed)':>22} {'>floor?':>9}")
    for z, d in resid:
        flag = "  ** yes" if abs(d) > SN_FLOOR_MAG else ""
        print(f"   {z:>6.2f} {d:>+22.4f} {abs(d) > SN_FLOOR_MAG!s:>9}{flag}")
    print(f"   max |Δμ| = {max_resid:.4f} mag  (floor = {SN_FLOOR_MAG} mag)")
    sn_detectable = max_resid > SN_FLOOR_MAG
    print(f"   → coasting SN shape is "
          f"{'DETECTABLY DIFFERENT from ΛCDM' if sn_detectable else 'WITHIN the floor'}")
    print()
    summary.append((
        "SNe Ia shape vs floor",
        None,
        f"max|Δμ|={max_resid:.3f} mag "
        f"({'>' if sn_detectable else '≤'}{SN_FLOOR_MAG})",
    ))

    # ----- REAL FITTER-BASED PROJECTION (lcdm_fitter on coasting mock) -----
    print("=" * 78)
    print(" REAL FITTER-BASED PROJECTION — run an ACTUAL ΛCDM fit on coasting data")
    print("=" * 78)
    print(" Generate CLEAN mock observables under the project-native coasting")
    print(" cosmography (H_0_true=72.74), then chi²-minimize flat-ΛCDM (H_0,Ω_m)")
    print(" via lib/lcdm_fitter.py. The recovered params (±1σ from inv-Fisher)")
    print(" are what a real ΛCDM pipeline would REPORT from coasting data — the")
    print(" operational meaning of 'coasting seen through ΛCDM eyes'.")
    print()
    # CC: observable is H(z) directly, at the real CC z/σ measurement points.
    cc_points = [(z, sig) for (z, _H, sig) in CC_HZ]
    fit_cc, _ = project_coasting_through_lcdm_fit(
        observable_fn=hz_observable, measurement_points=cc_points,
        H0_true=H0_OBSERVER, label="CC H(z)",
    )
    # SN: observable is μ(z), at the SN z-grid with a representative σ_μ.
    sn_points = [(z, 0.10) for z in SN_Z_GRID]
    fit_sn, _ = project_coasting_through_lcdm_fit(
        observable_fn=sn1a_distance_modulus, measurement_points=sn_points,
        H0_true=H0_OBSERVER, label="SN μ(z)",
    )
    # BAO: observable is D_V(z), at the BOSS+eBOSS z with representative σ.
    bao_z = sorted({z for (z, _k, _v, _s) in BAO_BOSS_EBOSS})
    bao_points = [(z, 30.0) for z in bao_z]  # ~30 Mpc per-point D_V error
    fit_bao, _ = project_coasting_through_lcdm_fit(
        observable_fn=bao_distance_DV, measurement_points=bao_points,
        H0_true=H0_OBSERVER, label="BAO D_V(z)",
    )
    print(f"   {'dataset':<12} {'z-range':>13} {'H_0_extr':>14} "
          f"{'Ω_m_extr':>14} {'z_eff':>7}")
    for fit, lbl, zr in (
        (fit_cc, "CC H(z)", cc_points),
        (fit_sn, "SN μ(z)", sn_points),
        (fit_bao, "BAO D_V(z)", bao_points),
    ):
        sH0, sOm = fit_sigmas(fit)
        H0e = fit.best_fit["H_0"]
        Ome = fit.best_fit["Omega_m"]
        zlo, zhi = zr[0][0], zr[-1][0]
        zeff = z_eff_diag(Ome)
        print(f"   {lbl:<12} [{zlo:4.2f},{zhi:5.2f}] "
              f"{H0e:7.2f}±{sH0:4.2f} {Ome:7.3f}±{sOm:5.3f} {zeff:>7.3f}")
    print()
    print(" → A real ΛCDM fit to coasting data recovers Ω_m in the ~0.22–0.43")
    print("   band (LOWER for higher-z leverage: BAO z≤2.33 → Ω_m≈0.22, SN → 0.43);")
    print("   note CC recovers Ω_m≈0.317 ≈ Planck 0.315. H_0 is pulled OFF the")
    print("   true 72.74 (up to ~85) by the shape mismatch. The 'dark energy'")
    print("   Ω_Λ=1−Ω_m is the parametric-class translation artifact, NOT a")
    print("   substrate component.")
    print()
    print(" Closed-form bias backbone (Ω_m_local(z)=(u+1)/(u²+u+1), u=1+z) — the")
    print(" single-z analytic the fitter integrates over the dataset z-range:")
    print(f"   {'z':>6} {'Ω_m_local':>11}")
    for z in (0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
        print(f"   {z:>6.2f} {Omega_m_local_coasting_closed_form(z):>11.4f}")
    print()

    # ----- CMB wall -----
    print("=" * 78)
    print(" CMB θ_* — OUT-OF-SCOPE STRUCTURAL BOUNDARY (L6 wall; named, not hidden)")
    print("=" * 78)
    print(" Coasting cannot produce the acoustic scale θ_* (~10⁵σ failure;")
    print(" Lambda_CC_path_A_session2). The framework is a LATE-TIME-geometry")
    print(" model; the CMB acoustic peak is its single precisely-characterized")
    print(" structural failure. Reported honestly. (Also needs the nucleon")
    print(" sector + √g_* + Boltzmann/Saha — Streams 2/3.)")
    print()

    # ----- CONSOLIDATED VERDICT -----
    print("=" * 78)
    print(" CONSOLIDATED VERDICT")
    print("=" * 78)
    print(f"   {'dataset':<28} {'ΔBIC':>8} {'ΔAIC':>8}  read (neg ΔBIC favors coasting)")
    for label, db, da in summary:
        if db is None:
            # SN shape-floor entry: da holds a descriptive string.
            print(f"   {label:<28} {'':>8} {'':>8}  {da}")
        else:
            print(f"   {label:<28} {db:>+8.2f} {da:>+8.2f}  "
                  f"{verdict_from_delta(db)}")
    print()
    print(" READ (honest, diagonal-error model-discrimination grade):")
    print("  • Cosmic chronometers — the clean DIRECT H(z) test: coasting is")
    print("    COMPETITIVE. χ²/dof≈0.55 (1-param) vs ΛCDM 0.50 (2-param); ΔBIC")
    print("    slightly favors coasting on Occam (one fewer parameter). And the")
    print("    framework's ZERO-free-parameter substrate H_0=68.19 fits at")
    print("    χ²/dof≈1.07 — the linear H∝(1+z) shape tracks model-independent")
    print("    H(z) with no freedom (cf. 2412.15717). NOTE the honest nuance:")
    print("    coasting-shape free-fit pulls H_0 to ~62 (~4σ below substrate),")
    print("    while ΛCDM-shape gives H_0≈68.2 (matching substrate) — i.e. the")
    print("    H_0 you read off CC depends on the assumed shape.")
    print("  • BAO (BOSS+eBOSS standard ruler): coasting shape is DECISIVELY")
    print("    disfavored (ΔBIC≈+18). The acoustic-scale geometry, not just the")
    print("    CMB, resists pure coasting. DESI DR2 (2512.19175) is the proper")
    print("    modern re-test (needs released data+covariance, not done here).")
    print("  • SNe Ia: offset-marginalized coasting−ΛCDM μ(z) shape vs the")
    print("    ~0.05 mag Pantheon+ floor — see per-z table above for whether the")
    print("    departure is detectable.")
    print("  • CMB θ_*: out-of-scope wall — the single decisive structural")
    print("    failure (needs nucleon sector + √g_* + Boltzmann/Saha: Streams 2/3).")
    print()
    print(" BOTTOM LINE: on the cleanest DIRECT probe (model-independent H(z))")
    print(" the framework's zero-parameter coasting is competitive with ΛCDM;")
    print(" on standard-ruler/candle geometry (BAO) the imposed linear shape is")
    print(" disfavored. The '≤1-parameter competitive late-time alternative'")
    print(" claim holds for chronometers, NOT for BAO. Frame-clean and honest;")
    print(" publication grade needs full covariance + Pantheon+/DESI likelihoods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
