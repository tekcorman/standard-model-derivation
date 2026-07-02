#!/usr/bin/env python3
"""
ν decoupling temperature from framework G_F + H(N) — Session 1 probe (P1).

Scoping: an internal working note (P1)

GOAL: compute T_ν_dec from rate balance Γ_weak(T) = H(T) using framework-
internal G_F (theorem-grade) and the framework's coasting H(N) (theorem-grade),
and compare to:
  (a) standard cosmology T_ν_dec ≈ 1 MeV (which uses radiation-era H ∝ T²/M_Pl);
  (b) framework BBN scale T_BBN ≈ 1 MeV (Q_np-derived, currently OPEN).

If T_ν_dec under framework's H(N) is close to T_BBN, then ν decoupling and BBN
are plausibly the same F-fiber (Phase IIb interpretation of an existing
framework-flagged epoch).

If T_ν_dec ≠ T_BBN by > 0.3 decades (AB2 of scoping), they are distinct
F-fibers, and the scoping proposes two named beats.

Method:
  - Weak interaction rate: Γ_weak(T) ≈ G_F² · T^5  (relativistic ν, leading order)
  - Framework Hubble rate: H_framework(T) under coasting + temperature scaling.
  - Solve Γ_weak(T) = H_framework(T) for T.
"""

import math

# ============================================================
# FRAMEWORK PRIMITIVES (all theorem-grade)
# ============================================================
G_F_GeV_inv_sq = 1.1663787e-5   # GeV^-2, framework-theorem-grade via predictions/G_F.py
M_PL_GeV = 1.22089e19            # Planck mass, framework-theorem-grade
ALPHA_TEMP = 25.0/48.0           # T(N) = T_P · N^(-α), cumulative-Perron value


# ============================================================
# RATE BALANCE
# ============================================================
# Standard cosmology (for comparison):
#   H_std(T) = sqrt(8π³ g* / 90) · T² / M_Pl  ≈ 1.66 √g* · T²/M_Pl
#   At T~1 MeV, g* ~ 10.75 (γ, e±, 3ν).
G_STAR_MEV = 10.75

def H_standard_at_T(T_GeV):
    """Standard radiation-era Hubble (using full M_Pl, not reduced)."""
    return 1.66 * math.sqrt(G_STAR_MEV) * T_GeV**2 / M_PL_GeV


# Framework coasting:
#   T(N) = T_P · N^(-α), so N = (T_P/T)^(1/α).
#   H(N) = 1/(N·t_P) = M_PL · N^(-1) = M_PL · (T/T_P)^(1/α)
#         = T^(1/α) · M_PL^(1 - 1/α)
def H_framework_at_T(T_GeV):
    """Framework coasting H(T) using T(N) and H(N) = 1/(N·t_P)."""
    inv_alpha = 1.0/ALPHA_TEMP
    return T_GeV**inv_alpha * M_PL_GeV**(1 - inv_alpha)


# Weak interaction rate (relativistic ν):
def Gamma_weak_at_T(T_GeV):
    return G_F_GeV_inv_sq**2 * T_GeV**5


# ============================================================
# SOLVE Γ = H NUMERICALLY
# ============================================================
def find_T_dec(H_func, T_low=1e-9, T_high=1.0, tol=1e-12):
    """Bisection for T satisfying Γ_weak(T) = H_func(T).
    Γ_weak ∝ T^5 grows faster than H ∝ T^2 (standard) or T^(48/25) (framework),
    so diff = Γ - H is monotonic: negative at low T, positive at high T.
    """
    def diff(T): return Gamma_weak_at_T(T) - H_func(T)

    if diff(T_low) * diff(T_high) > 0:
        return None

    for _ in range(200):
        T_mid = math.sqrt(T_low * T_high)
        if abs(T_high - T_low)/T_mid < tol:
            return T_mid
        if diff(T_mid) > 0:
            # T_mid is above the root → lower the upper bound
            T_high = T_mid
        else:
            # T_mid is below the root → raise the lower bound
            T_low = T_mid
    return math.sqrt(T_low * T_high)


# ============================================================
# REPORT
# ============================================================
def report():
    T_dec_std = find_T_dec(H_standard_at_T)
    T_dec_fw = find_T_dec(H_framework_at_T)

    print("=" * 78)
    print("  ν decoupling temperature from framework G_F + H(N)")
    print("  Session 1 probe of BBN→recomb species-decoupling scoping")
    print("=" * 78)

    print(f"\n  Framework primitives (all theorem-grade):")
    print(f"    G_F        = {G_F_GeV_inv_sq:.6e} GeV^-2   (predictions/G_F.py)")
    print(f"    M_Pl       = {M_PL_GeV:.4e} GeV          (predictions/M_Pl_natural.py)")
    print(f"    α_temp     = 25/48 = {ALPHA_TEMP:.6f}    (cumulative-Perron, A1 reframe)")

    print(f"\n  Rate balance Γ_weak(T) = G_F² · T^5  vs  H(T)")

    # Standard
    print(f"\n  STANDARD cosmology  (H_std = 1.66·√g* · T²/M_Pl, g* = {G_STAR_MEV}):")
    print(f"    T_ν_dec_std       = {T_dec_std*1e3:.3f} MeV")
    print(f"    Γ_weak(T_dec)     = {Gamma_weak_at_T(T_dec_std):.4e} GeV")
    print(f"    H_std(T_dec)      = {H_standard_at_T(T_dec_std):.4e} GeV")

    # Framework
    print(f"\n  FRAMEWORK  (H_fw = T^(1/α) · M_Pl^(1-1/α), α = 25/48):")
    print(f"    T_ν_dec_fw        = {T_dec_fw*1e3:.3f} MeV")
    print(f"    Γ_weak(T_dec)     = {Gamma_weak_at_T(T_dec_fw):.4e} GeV")
    print(f"    H_fw(T_dec)       = {H_framework_at_T(T_dec_fw):.4e} GeV")

    ratio = T_dec_fw / T_dec_std
    decades = math.log10(ratio)
    print(f"\n  Ratio T_ν_dec_fw / T_ν_dec_std = {ratio:.3f}  ({decades:+.3f} decades)")

    print(f"\n  H(T) comparison at T = 1 MeV:")
    T_1MeV = 1e-3
    H_std_1 = H_standard_at_T(T_1MeV)
    H_fw_1 = H_framework_at_T(T_1MeV)
    print(f"    H_std(1 MeV)      = {H_std_1:.4e} GeV")
    print(f"    H_fw(1 MeV)       = {H_fw_1:.4e} GeV")
    print(f"    Ratio H_fw/H_std  = {H_fw_1/H_std_1:.3f}")

    # ============================================================
    # VERDICT
    # ============================================================
    T_BBN_MeV = 1.0  # standard BBN scale (currently OPEN in framework cascade)
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)

    print(f"\n  Framework T_ν_dec  = {T_dec_fw*1e3:.2f} MeV")
    print(f"  Standard T_ν_dec   ≈ 1 MeV  (canonical literature value)")
    print(f"  Framework T_BBN    ≈ 1 MeV  (Q_np-derived; cascade OPEN)")
    print()

    # AB2 check: are ν decoupling and BBN separated by > 0.3 decades?
    AB2_threshold_decades = 0.3
    T_diff_decades = abs(math.log10(T_dec_fw*1e3 / T_BBN_MeV))

    if T_diff_decades < AB2_threshold_decades:
        print(f"  ν decoupling and BBN differ by {T_diff_decades:.3f} decades.")
        print(f"  AB2 NOT triggered ({AB2_threshold_decades} decades threshold).")
        print(f"  -> Reading: ν decoupling and BBN are plausibly the SAME F-fiber.")
        same_fiber = True
    else:
        print(f"  ν decoupling and BBN differ by {T_diff_decades:.3f} decades.")
        print(f"  AB2 TRIGGERED ({AB2_threshold_decades} decades threshold).")
        print(f"  -> Reading: ν decoupling and BBN are DISTINCT F-fibers.")
        same_fiber = False

    print()
    print("  Honest structural reading:")
    print(f"    - Framework's H(T) under coasting + α = 25/48 differs from")
    print(f"      standard radiation-era H by factor {H_fw_1/H_std_1:.2f} at T=1 MeV.")
    print(f"      This is a STRUCTURAL difference: framework doesn't have a")
    print(f"      radiation-era Hubble rate, it has coasting H = 1/(N·t_P)")
    print(f"      with T = T_P · N^(-25/48).")
    print(f"    - The factor difference shifts T_ν_dec by {decades:+.2f} decades")
    print(f"      relative to standard.")
    if same_fiber:
        print(f"    - ν decoupling ≈ BBN scale → Phase IIb F-fiber at ~few MeV")
        print(f"      coincides with the currently-OPEN BBN beat. Likely same")
        print(f"      F-fiber under different labels.")
    else:
        print(f"    - ν decoupling and BBN are at different scales → two distinct")
        print(f"      Phase IIb F-fibers needed in the cascade.")
    print()
    print("  Session 1 grade: CANDIDATE — derivation chain is framework-")
    print("  internal and theorem-grade in inputs (G_F, M_Pl, α), but the rate")
    print("  balance formula uses g* = 10.75 as standard-cosmology input. A")
    print("  framework-internal g* derivation would graduate this to")
    print("  theorem-grade-conditional.")
    print()
    print("  Next: Session 2 P2 — matter-radiation equality structural-absence")
    print("  proof under coasting (deepens the proposal independent of T_ν_dec).")
    print("=" * 78)


if __name__ == "__main__":
    report()
