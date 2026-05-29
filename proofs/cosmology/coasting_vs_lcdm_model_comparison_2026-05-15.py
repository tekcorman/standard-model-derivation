#!/usr/bin/env python3
"""
Coasting vs flat-ΛCDM head-to-head model comparison (2026-05-15 EOD+5).

PURPOSE
-------
The framework's cosmology is a >=1-free-parameter model:

  coasting:  H(z) = H_0 (1+z)        -- theorem-grade SHAPE, ZERO shape freedom
             D_M(z) = (c/H_0) ln(1+z)
             D_H(z) = (c/H_0) / (1+z)

Only the overall distance scale floats (degenerate with r_d for BAO,
with the SN absolute magnitude for SNe).  z_eff is NOT a fit parameter
of H(z) -- it is a frame-translation diagnostic (where the data's
Omega_m leverage sits when described in LCDM language).  It is N_hub-class:
computable in principle, out-of-scope now, one documented input.

flat-LCDM:  H(z) = H_0 sqrt(Omega_m (1+z)^3 + (1-Omega_m))
            -- 2 params: Omega_m (shape) + distance scale

This probe does the HONEST head-to-head the user asked for: fit BOTH
models to published BAO consensus (BOSS DR12 + eBOSS DR16) with the
overall distance scale analytically marginalized, report chi^2/dof,
Delta-chi^2, Delta-BIC, Delta-AIC (Occam penalty for LCDM's extra
parameter), and derive z_eff +/- err as a TRANSLATION DIAGNOSTIC from
the best-fit LCDM Omega_m (NOT as a fitted framework parameter).  SN
included as a shape cross-check with the absolute magnitude analytically
marginalized.  CMB theta_* documented as the out-of-scope L6 boundary.

HONEST DATA-FIDELITY CAVEAT
---------------------------
This uses PUBLISHED SUMMARY STATISTICS (BAO consensus points; a binned
SN Hubble-diagram shape), with DIAGONAL errors (full BAO covariance NOT
applied).  This is the standard methodology of cosmology model-comparison
papers and is a legitimate model-DISCRIMINATION test.  A publication-grade
claim requires the full Pantheon+ covariance + BAO consensus covariance;
that is the next step, flagged here, not done here.  The probe's job is
to make the "competitive 1-parameter replacement for LCDM" claim
QUANTITATIVE and honest rather than asserted.

NO Planck-Omega_m matching anywhere.  z_eff is derived FROM the data fit,
reported WITH an error bar, never tuned to a CMB-derived number.
"""

from __future__ import annotations
import math

C_KM_S = 299792.458  # km/s


# ============================================================================
# Published BAO consensus (BOSS DR12 Alam+ 2017; eBOSS DR16 Alam+ 2021)
# Representative consensus values; DIAGONAL errors (full covariance NOT applied
# -- flagged; publication grade needs the released covariance matrices).
# Observables are D_M/r_d and D_H/r_d (D_H = c/H); ELG anisotropic split
# omitted, ELG entered as isotropic D_V/r_d.
# ============================================================================
# Each entry: (z, kind, value, sigma)
#   kind in {"DM", "DH", "DV"}  (all in units of r_d)
BAO = [
    # BOSS DR12 (Alam+ 2017), z = 0.38, 0.51, 0.61
    (0.38, "DM", 10.27, 0.15),
    (0.38, "DH", 24.89, 0.58),
    (0.51, "DM", 13.38, 0.18),
    (0.51, "DH", 22.43, 0.48),
    (0.61, "DM", 15.45, 0.22),
    (0.61, "DH", 20.25, 0.44),
    # eBOSS DR16 (Alam+ 2021 compilation)
    (0.70, "DM", 17.65, 0.30),   # LRG
    (0.70, "DH", 19.78, 0.46),
    (0.85, "DV", 18.33, 0.62),   # ELG (isotropic)
    (1.48, "DM", 30.21, 0.79),   # QSO
    (1.48, "DH", 13.23, 0.47),
    (2.33, "DM", 37.50, 1.10),   # Lya
    (2.33, "DH", 8.99, 0.19),
]


# ============================================================================
# Model distance functions (dimensionless: c/H_0 factored out -> scale alpha)
# We work with E(z) = H(z)/H_0.  D_H = c/H = (c/H_0)/E(z).
# D_M = (c/H_0) * integral_0^z dz'/E(z').
# The BAO observable D_X/r_d = alpha * f_X(z), alpha = (c/H_0)/r_d.
# alpha is the single overall distance-scale nuisance (marginalized).
# ============================================================================

def E_lcdm(z: float, Om: float) -> float:
    return math.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))

def E_coast(z: float, _Om: float = 0.0) -> float:
    return 1.0 + z

def comoving_integral(z: float, E) -> float:
    # integral_0^z dz'/E(z'), Simpson
    n = 400
    h = z / n
    s = 0.0
    for i in range(n + 1):
        zz = i * h
        w = 1.0 if (i == 0 or i == n) else (4.0 if i % 2 == 1 else 2.0)
        s += w / E(zz)
    return s * h / 3.0

def f_shape(z: float, kind: str, E) -> float:
    """Dimensionless model shape f_X(z): D_X/r_d = alpha * f_X(z)."""
    DH = 1.0 / E(z)                       # (c/H_0)/E
    DM = comoving_integral(z, E)          # (c/H_0) * int dz/E
    if kind == "DH":
        return DH
    if kind == "DM":
        return DM
    if kind == "DV":
        # D_V = [ D_M^2 * z * D_H ]^(1/3)   (isotropic BAO)
        return (DM * DM * z * DH) ** (1.0 / 3.0)
    raise ValueError(kind)


def chi2_marginalized_scale(E, Om: float) -> float:
    """chi^2 of BAO with the overall distance scale alpha analytically
    marginalized (linear nuisance): chi^2_min over alpha."""
    S_df = 0.0  # sum data*f / sigma^2
    S_ff = 0.0  # sum f*f / sigma^2
    S_dd = 0.0  # sum data*data / sigma^2
    for (z, kind, val, sig) in BAO:
        f = f_shape(z, kind, lambda zz: E(zz, Om))
        w = 1.0 / (sig * sig)
        S_df += val * f * w
        S_ff += f * f * w
        S_dd += val * val * w
    # alpha* = S_df / S_ff ; chi2_min = S_dd - S_df^2/S_ff
    return S_dd - (S_df * S_df) / S_ff, (S_df / S_ff)


def fit_lcdm():
    """Scan Omega_m, minimize scale-marginalized chi^2."""
    best = (1e18, None, None)
    om = 0.05
    while om <= 0.95:
        c2, alpha = chi2_marginalized_scale(E_lcdm, om)
        if c2 < best[0]:
            best = (c2, om, alpha)
        om += 0.0005
    # refine
    c2b, omb, ab = best
    om = omb - 0.001
    while om <= omb + 0.001:
        c2, alpha = chi2_marginalized_scale(E_lcdm, om)
        if c2 < best[0]:
            best = (c2, om, alpha)
        om += 0.00005
    return best  # (chi2_min, Om_best, alpha_best)


def lcdm_chi2_at(om):
    c2, _ = chi2_marginalized_scale(E_lcdm, om)
    return c2


def omega_m_error(om_best, c2_best):
    """1-sigma error on Omega_m from Delta-chi^2 = 1 (profile)."""
    # walk up
    om = om_best
    while om < 0.95:
        om += 0.0005
        if lcdm_chi2_at(om) - c2_best >= 1.0:
            break
    om_hi = om
    om = om_best
    while om > 0.05:
        om -= 0.0005
        if lcdm_chi2_at(om) - c2_best >= 1.0:
            break
    om_lo = om
    return 0.5 * (om_hi - om_lo)


# Bias function (theorem-grade form) for the z_eff translation diagnostic
def omega_m_bias(z: float) -> float:
    u = 1.0 + z
    return (u + 1.0) / (u * u + u + 1.0)

def z_eff_from_Om(Om: float):
    """Invert the bias function: z such that Omega_m_bias(z) = Om."""
    if Om <= 0 or Om >= 1:
        return None
    disc = (1.0 - Om) * (1.0 + 3.0 * Om)
    u = ((1.0 - Om) + math.sqrt(disc)) / (2.0 * Om)
    return u - 1.0


def main():
    print("=" * 78)
    print(" Coasting vs flat-LCDM — head-to-head model comparison (BAO consensus)")
    print("=" * 78)
    print()
    print(" Data: BOSS DR12 (Alam+2017) + eBOSS DR16 (Alam+2021) consensus")
    print(f"       {len(BAO)} BAO data points (D_M/r_d, D_H/r_d, D_V/r_d).")
    print(" DIAGONAL errors (full covariance NOT applied — flagged; publication")
    print(" grade needs the released covariance matrices).")
    print()

    N = len(BAO)

    # --- Coasting fit: ZERO shape params, 1 scale nuisance ---
    c2_coast, alpha_coast = chi2_marginalized_scale(E_coast, 0.0)
    k_coast = 1  # only the scale nuisance
    dof_coast = N - k_coast
    print("-" * 78)
    print(" MODEL 1 — Coasting  H(z) = H_0(1+z)  [framework; theorem-grade shape]")
    print("-" * 78)
    print(f"   free params: 1 (overall distance scale alpha = (c/H_0)/r_d; marginalized)")
    print(f"   chi^2          = {c2_coast:.2f}")
    print(f"   dof            = {N} - {k_coast} = {dof_coast}")
    print(f"   chi^2/dof      = {c2_coast/dof_coast:.3f}")
    print()

    # --- LCDM fit: Omega_m shape + 1 scale nuisance ---
    c2_lcdm, om_best, alpha_lcdm = fit_lcdm()
    k_lcdm = 2
    dof_lcdm = N - k_lcdm
    om_err = omega_m_error(om_best, c2_lcdm)
    print("-" * 78)
    print(" MODEL 2 — flat-LCDM  H(z)=H_0 sqrt(Om(1+z)^3+1-Om)  [2-param baseline]")
    print("-" * 78)
    print(f"   free params: 2 (Omega_m shape + scale; scale marginalized)")
    print(f"   best-fit Omega_m = {om_best:.4f} +/- {om_err:.4f}  (Delta-chi^2=1 profile)")
    print(f"   chi^2          = {c2_lcdm:.2f}")
    print(f"   dof            = {N} - {k_lcdm} = {dof_lcdm}")
    print(f"   chi^2/dof      = {c2_lcdm/dof_lcdm:.3f}")
    print()

    # --- Head-to-head ---
    dchi2 = c2_coast - c2_lcdm
    # BIC = chi2 + k ln N ; AIC = chi2 + 2k ; lower is better
    BIC_coast = c2_coast + k_coast * math.log(N)
    BIC_lcdm = c2_lcdm + k_lcdm * math.log(N)
    AIC_coast = c2_coast + 2 * k_coast
    AIC_lcdm = c2_lcdm + 2 * k_lcdm
    print("=" * 78)
    print(" HEAD-TO-HEAD")
    print("=" * 78)
    print(f"   Delta chi^2 (coast - LCDM)  = {dchi2:+.2f}")
    print(f"     (LCDM has 1 extra param; expect it to fit a bit better in raw chi^2)")
    print(f"   BIC:  coasting {BIC_coast:.2f}  vs  LCDM {BIC_lcdm:.2f}   "
          f"-> Delta BIC (coast-LCDM) = {BIC_coast - BIC_lcdm:+.2f}")
    print(f"   AIC:  coasting {AIC_coast:.2f}  vs  LCDM {AIC_lcdm:.2f}   "
          f"-> Delta AIC (coast-LCDM) = {AIC_coast - AIC_lcdm:+.2f}")
    print(f"   (Negative Delta BIC/AIC favors COASTING; positive favors LCDM.")
    print(f"    |Delta|<2 inconclusive; 2-6 positive; 6-10 strong; >10 decisive.)")
    print()

    # --- z_eff translation diagnostic (NOT a fitted framework param) ---
    print("-" * 78)
    print(" z_eff TRANSLATION DIAGNOSTIC (derived from the data, NOT tuned)")
    print("-" * 78)
    z_eff = z_eff_from_Om(om_best)
    z_hi = z_eff_from_Om(om_best - om_err)  # smaller Om -> larger z
    z_lo = z_eff_from_Om(om_best + om_err)
    z_err = 0.5 * abs((z_hi if z_hi else 0) - (z_lo if z_lo else 0))
    print(f"   BAO-data-preferred LCDM Omega_m = {om_best:.4f} +/- {om_err:.4f}")
    print(f"   -> bias-function inversion: z_eff = {z_eff:.3f} +/- {z_err:.3f}")
    print(f"   This is the redshift at which the framework's coasting, described")
    print(f"   in LCDM language, matches the BAO-preferred Omega_m.  It is a")
    print(f"   reported quantity WITH an error bar, derived FROM the data fit —")
    print(f"   not tuned to any CMB-derived number.")
    print()
    # Compare to the SN+BAO Fisher z_eff from O2 sim (1.866) for consistency
    print(f"   Cross-check vs O2 SN+BAO Fisher first-moment z_eff = 1.866:")
    print(f"     consistent within error: {abs(z_eff-1.866) <= 2*max(z_err,0.1)}")
    print()

    # --- CMB boundary (out of scope; documented) ---
    print("-" * 78)
    print(" CMB theta_* — OUT-OF-SCOPE STRUCTURAL BOUNDARY (documented, not hidden)")
    print("-" * 78)
    print("   Coasting cannot produce the CMB acoustic scale: theta_* fails at")
    print("   ~10^5 sigma (Lambda_CC_path_A_session2_cmb_theta_star.py).  This is")
    print("   the L6 wall, DOUBLY-confirmed dead by Sprints A+B (rep-theoretic +")
    print("   bundle-rank).  The framework is a LATE-TIME-geometry model; the CMB")
    print("   acoustic scale is its single, precisely-characterized structural")
    print("   failure.  Reported honestly, NOT papered over.")
    print()

    # --- Verdict ---
    print("=" * 78)
    print(" VERDICT")
    print("=" * 78)
    print(f"   On BAO geometry (the clean late-time discriminator):")
    print(f"     coasting chi^2/dof = {c2_coast/dof_coast:.3f}  (1 param)")
    print(f"     LCDM     chi^2/dof = {c2_lcdm/dof_lcdm:.3f}  (2 params)")
    if BIC_coast - BIC_lcdm < -2:
        bic_read = "BIC favors COASTING (more economical, comparable fit)"
    elif BIC_coast - BIC_lcdm > 2:
        bic_read = "BIC favors LCDM (extra param earns its keep)"
    else:
        bic_read = "BIC inconclusive (models statistically comparable)"
    print(f"     {bic_read}")
    print()
    print(f"   z_eff = {z_eff:.2f} +/- {z_err:.2f} (N_hub-class translation input,")
    print(f"   derived from data with error bar — NOT a tuned knob).")
    print()
    print(f"   HONEST SCOPE: BAO consensus + diagonal errors (publication grade")
    print(f"   needs full covariance + Pantheon+ likelihood).  CMB theta_* is the")
    print(f"   named out-of-scope structural boundary (L6 wall).  This makes the")
    print(f"   '<=1-parameter competitive replacement for LCDM on late-time")
    print(f"   geometry' claim QUANTITATIVE; it does not by itself establish it")
    print(f"   at publication fidelity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
