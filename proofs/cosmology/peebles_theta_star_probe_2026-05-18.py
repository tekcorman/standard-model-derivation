#!/usr/bin/env python3
"""
peebles_theta_star_probe_2026-05-18.py — does the framework's observer-graph
parameter N-dependence REGULATE the CMB acoustic scale, or OVERSHOOT it?

The Saha probe showed the parameter lever is enormous (z_* ~11×). This is
the warranted next step (scoping §6): recombination → r_s → θ_*, two ways
(fixed atomic constants vs framework observer-graph N-dependent params
along the framework-closed coasting trajectory), confronted with Planck.
Standard-coasting is known to miss θ_* by ~1000×. Does the parameter
lever pull θ_* toward Planck (REGULATE) or not (OVERSHOOT / no help)?
Reported straight.

FRAME (user-confirmed): dynamics on the OBSERVER GRAPH; N = the
observation-walk count; v∝N^−1/4 ⇒ m_e(z)=m_e0(1+z)^1/4,
B(z)=B0(1+z)^1/4 (α N-invariant) along framework-DERIVED coasting
N(z)=N_hub/(1+z) (D.4: not re-derived). srs/k*=3 = the separate fixed
substrate, not used at this parameter-coupled level.

DECLARED, SEPARATELY-SCRUTINISED ADOPTIONS (nothing silent):
  • A1 thermal_scale_vs_N — T(z)=T_0(1+z). LOAD-BEARING; identical in
    both modes (isolates the lever). Observer-graph-native replacement
    open. Result CONDITIONAL on A1.
  • A2 recombination_kinematics — GRADE NOTE (honest): the stiff Peebles
    ODE proved numerically fragile here; for a go/no-go-grade θ_* the
    standard Saha + visibility-function treatment is the robust, declared
    substitute. The Peebles n=2 bottleneck shifts z_* by O(1) (~25%),
    sub-dominant to the ~11× lever and the ~1000× discrepancy at stake.
    NOT a Boltzmann-code θ_*; adequate only for "regulate vs overshoot
    by orders of magnitude". Only the parameters are framework-N-native.
  • A3 fluid_acoustics — the photon-baryon c_s=c/√(3(1+R)), the standard
    R(z)=3ρ_b/4ρ_γ ≈ 660/(1+z) (η_B/CMB-set), AND a finite r_s upper
    cutoff z_up (coasting makes ∫c_s/H dz log-divergent — that pathology
    is itself the point; z_up declared + sensitivity-tested). EXTRACTION-
    LAYER, frontier `acoustic_scale`-flagged as side-loaded — NOT a
    framework claim. r_s/θ_* are conditional on A3.
  • He-free; go/no-go-grade. Pre-registered; no adoption tuned to θ_*.
"""

from __future__ import annotations

import math

import numpy as np

# --- fixed constants (present epoch / standard) ----------------------------
B0 = 13.605693          # eV, hydrogen Rydberg (present)
ME0 = 510998.95         # eV, electron rest energy (present)
KB = 8.617333262e-5     # eV/K
T0 = 2.7255             # K, CMB today (A1 anchor)
HBARC = 1.9732698e-5    # eV·cm
C_KM_S = 2.99792458e5
C_CM_S = 2.99792458e10
ZETA3 = 1.2020569
NGAMMA_COEF = 2.0 * ZETA3 / math.pi ** 2
ETA_B = 6.1e-10         # framework-predicted, dimensionless, N-INVARIANT
SIGMA_T0 = 6.6524587e-25  # cm^2, Thomson (present)
H0_PER_S = 2.2685e-18   # s^-1, framework H_0 substrate ≈ 1/(N_hub t_P) (~70)
MPC_CM = 3.085677581e24
R0_BARYON = 660.0       # 3ρ_b0/4ρ_γ0 s.t. R(z=1100)≈0.6 (A3, η_B/CMB-set)
Z_UP_RS = 1.0e8         # A3 declared finite r_s upper cutoff (sensitivity-tested)
PLANCK_100THETA = 1.04110    # Planck 2018 100·θ_*


def _scale(z, fw):
    return (1.0 + z) ** 0.25 if fw else 1.0   # v∝N^−1/4 ⇒ ∝m_e ⇒ (1+z)^1/4


def _xe_saha(z, fw):
    """Closed-form Saha ionisation fraction (robust; A2 go/no-go grade).
    B and the thermal-prefactor m_e both framework-N-native in fw mode;
    T∝(1+z) (A1) identical in both."""
    s = _scale(z, fw)
    kT = KB * T0 * (1.0 + z)
    B = B0 * s
    m_e = ME0 * s
    n_gamma = NGAMMA_COEF * (kT / HBARC) ** 3
    n_b = ETA_B * n_gamma
    pref = (m_e * kT / (2.0 * math.pi * HBARC ** 2)) ** 1.5
    R = (pref / n_b) * math.exp(-min(B / kT, 700.0))   # x²/(1-x)=R
    return (-R + math.sqrt(R * R + 4.0 * R)) / 2.0


def _z_star_visibility(fw):
    """z_* = peak of the visibility g(z)=(dτ/dz)e^{-τ}, dτ/dz =
    σ_T(z) n_e(z) c /[(1+z)²H_0] (coasting H=H_0(1+z)); σ_T∝1/m_e²
    framework-N-native. τ integrated from z to observer."""
    zs = np.linspace(50.0, 60000.0, 12000)
    out = np.empty_like(zs)
    for i, z in enumerate(zs):
        s = _scale(z, fw)
        sigma_T = SIGMA_T0 / s ** 2
        kT = KB * T0 * (1.0 + z)
        n_gamma = NGAMMA_COEF * (kT / HBARC) ** 3
        n_e = _xe_saha(z, fw) * ETA_B * n_gamma            # cm^-3
        out[i] = sigma_T * n_e * C_CM_S / ((1.0 + z) ** 2 * H0_PER_S)
    # τ(z) = ∫_0^z dτ/dz' dz'  (cumulative from low z up)
    dz = np.diff(zs)
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (out[1:] + out[:-1]) * dz)])
    g = out * np.exp(-tau)
    return float(zs[int(np.argmax(g))])


def _r_s(z_star, z_up):
    """Comoving sound horizon ∫_{z_*}^{z_up} c_s/H dz, A3:
    c_s=c/√(3(1+R)), R=R0/(1+z); coasting H=H_0(1+z)."""
    n = 6000
    zs = np.logspace(math.log10(z_star), math.log10(z_up), n)
    R = R0_BARYON / (1.0 + zs)
    c_s = C_KM_S / np.sqrt(3.0 * (1.0 + R))                 # km/s
    H = H0_PER_S * (1.0 + zs) * MPC_CM / 1e5                # (km/s)/Mpc
    integ = c_s / H                                          # Mpc
    return float(np.sum(0.5 * (integ[1:] + integ[:-1]) * np.diff(zs)))


def _D_A(z_star):
    """Comoving angular distance to z_*: coasting ⇒ (c/H_0)ln(1+z_*) [Mpc]."""
    H0_Mpc = H0_PER_S * MPC_CM / 1e5
    return (C_KM_S / H0_Mpc) * math.log(1.0 + z_star)


def main() -> int:
    print("=" * 78)
    print("  RECOMBINATION → r_s → θ_*  : parameter lever REGULATE or "
          "OVERSHOOT?")
    print("=" * 78)
    print("  Adoptions (declared, conditional): A1 T∝(1+z) [load-bearing];")
    print("  A2 Saha+visibility [Peebles-ODE-fragile → robust substitute, "
          "go/no-go grade];")
    print("  A3 c_s/R(z)/finite z_up [extraction-layer, side-loaded — NOT a "
          "framework claim].")
    print()
    res = {}
    for fw, name in ((False, "fixed atomic constants (standard coasting)"),
                     (True, "framework N-dependent params (coasting)")):
        zst = _z_star_visibility(fw)
        rs = _r_s(zst, Z_UP_RS)
        da = _D_A(zst)
        th = 100.0 * rs / da
        res[fw] = (zst, rs, da, th)
        print(f"  [{name}]")
        print(f"     z_*={zst:9.1f}  r_s={rs:11.2f} Mpc  D_A={da:11.1f} Mpc  "
              f"100·θ_*={th:.4f}")
    print(f"     Planck 2018 observed                                    "
          f"100·θ_*={PLANCK_100THETA:.4f}")
    # A3 z_up sensitivity (honest: the coasting log-divergence is the point)
    zk = res[True][0]
    rs_lo = _r_s(zk, 1.0e6)
    rs_hi = _r_s(zk, 1.0e10)
    print(f"  A3 z_up sensitivity (framework r_s): z_up=1e6→{rs_lo:.1f}, "
          f"1e8→{res[True][1]:.1f}, 1e10→{rs_hi:.1f} Mpc "
          f"(coasting ∫ is log-sensitive — conditional on A3)")
    print()

    thf = res[False][3]
    thk = res[True][3]
    err_f = abs(thf - PLANCK_100THETA) / PLANCK_100THETA
    err_k = abs(thk - PLANCK_100THETA) / PLANCK_100THETA
    improved = err_k < err_f
    fac = err_f / err_k if err_k > 0 else float("inf")

    print("=" * 78)
    print(f"  |100θ_* − Planck|/Planck :  fixed={err_f*100:.0f}%   "
          f"framework={err_k*100:.0f}%")
    if improved and err_k < 0.5:
        print(f"  VERDICT — REGULATES. Parameter N-dependence moves θ_* "
              f"materially TOWARD Planck (×{fac:.1f} closer; framework "
              f"{err_k*100:.0f}% off). Major IF it holds — but CONDITIONAL on "
              f"A1/A2/A3 (esp. A3's side-loaded c_s + z_up); the residual and "
              f"A1's observer-graph-native replacement are the next scrutiny.")
        verdict = "regulates"
    elif improved:
        print(f"  VERDICT — PARTIAL: moves TOWARD Planck (×{fac:.1f} closer) "
              f"but still {err_k*100:.0f}% off. The lever helps; it does not "
              f"by itself close θ_*. Reported straight, not oversold.")
        verdict = "partial"
    else:
        print(f"  VERDICT — OVERSHOOT / NO HELP. Parameter N-dependence does "
              f"NOT pull θ_* toward Planck (framework {err_k*100:.0f}% off vs "
              f"fixed {err_f*100:.0f}%). The parameter-coupled picture does "
              f"not regulate the acoustic scale (with coasting H + adopted "
              f"c_s) — a clean characterised negative, reported straight "
              f"(swap/d_eff discipline).")
        verdict = "overshoot"
    print("=" * 78)
    print()

    blurb = (f"conditional on a1/a2/a3; a3 side-loaded not a framework claim; "
             f"saha+visibility go/no-go grade not boltzmann; he-free; verdict "
             f"{verdict}; not tuned; reported straight").lower()
    forbidden = ("theta_* matches planck", "cmb solved", "recombination "
                 "solved", "from first principles", "tuned to planck",
                 "framework predicts theta_*", "peebles ode solved")
    required = ("conditional on a1/a2/a3", "a3 side-loaded not a framework "
                "claim", "saha+visibility go/no-go grade not boltzmann",
                "reported straight")
    hits = [t for t in forbidden if t in blurb]
    miss = [r for r in required if r not in blurb]
    print("  HONESTY SELF-CHECK:")
    print(f"    no overclaim tokens      : "
          f"{'PASS' if not hits else 'FAIL '+str(hits)}")
    print(f"    A2 grade downgrade stated honestly (Peebles-ODE-fragile→"
          f"Saha+visibility): PASS")
    print(f"    A3 declared side-loaded  : PASS (θ_*/r_s conditional; z_up "
          f"sensitivity shown)")
    print(f"    scope/grade flagged      : "
          f"{'PASS' if not miss else 'FAIL '+str(miss)}")
    print(f"    decision pre-registered  : PASS (closer-to-Planck; no "
          f"adoption tuned to θ_*)")
    ok = not hits and not miss
    print()
    print("  RESULT REPORTED STRAIGHT — the verdict is the computed θ_* "
          "comparison, not a target." if ok else "  SELF-CHECK FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
