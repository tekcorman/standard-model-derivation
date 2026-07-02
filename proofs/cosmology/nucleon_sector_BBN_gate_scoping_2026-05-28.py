#!/usr/bin/env python3
"""
Stream 3 — nucleon-sector BBN gate: scoping + Y_p sensitivity (2026-05-28).

Run:  python -m proofs.cosmology.nucleon_sector_BBN_gate_scoping_2026-05-28

WHY THIS IS NOT A BBN RUN
-------------------------
Stream 3 (BBN forward-model → D/H, ³He, ⁷Li, Y_p) is GATED on the framework's
ONE missing sector: nucleon bound-state physics. Per the cleanup finding
, all
7 cosmology files moved out of predictions/ bottleneck on the same three
nucleon quantities:

  • Q_np = m_n − m_p = 1.293 MeV   — sets the n/p freeze-out ratio exp(-Q_np/T)
  • g_A = 1.2723 (→ τ_n = 879.4 s) — sets the weak-rate normalization
  • Δα_had / nucleon EM self-energy — the EM part of Q_np; Δα_had-class BLOCKED

The framework cleanly has lepton + EW + quark-MASS sectors but NO baryon
(3-quark bound-state) sector. So this probe does NOT run BBN; it (A) states the
nucleon inputs BBN needs, (B) decomposes Q_np and inventories what the framework
already has vs what is missing, (C) measures — via the validated weak-sector
harness (lib/bbn_network.py) — HOW PRECISELY Q_np and g_A must be derived to
make Y_p a real prediction. That precision target is the spec for the
(multi-session, Need-B/BR4-walled) nucleon-sector derivation.

DISCIPLINE: proofs/ probe. The nucleon inputs are external measured physics
here (out-of-scope-by-construction, like B_D already is in bbn_network.py); we
do NOT smuggle them into a prediction. The sensitivity uses ΛCDM expansion as
the validated baseline so it isolates NUCLEON-input dependence, separate from
the √g_* leading-factor question.
"""

from __future__ import annotations

import math

from scipy.integrate import quad

import proofs.cosmology.lib.bbn_network as bbn

# Framework quark-mass predictions (predictions/m_u.py, m_d.py; M_persistence,
# theorem-grade-structural-conditional). MS-bar at 2 GeV.
M_U_FRAMEWORK_MeV = 2.16   # predictions/m_u.py
M_D_FRAMEWORK_MeV = 4.67   # predictions/m_d.py

# External nucleon-sector targets (PDG / lattice — what BBN needs).
Q_NP_OBS_MeV = 1.293332
G_A_OBS = 1.2723
TAU_N_OBS_s = 879.4
Y_P_OBS = 0.245
Y_P_OBS_SIGMA = 0.003   # Aver+2015 / PDG primordial He-4 (~1.2%)

# Lattice decomposition of Q_np (BMW 2015, Science 347:1452): the strong /
# m_d−m_u (QCD) contribution and the QED (EM self-energy) contribution.
Q_NP_QCD_LATTICE_MeV = 2.49     # +2.49(17)(10) MeV, strong-isospin (m_d−m_u)
Q_NP_QCD_LATTICE_SIG = 0.20
Q_NP_QED_LATTICE_MeV = -1.00    # −1.00(07)(14) MeV, electromagnetic
Q_NP_QED_LATTICE_SIG = 0.16


# ===========================================================================
# Harness control: vary the nucleon inputs that bbn_network bakes in as globals
# ===========================================================================
def _set_Qnp(val_MeV: float) -> None:
    """Override Q_np and recompute the derived globals (Q, _I0)."""
    bbn.Q_NP_MeV = val_MeV
    bbn.Q = val_MeV / bbn.M_E_MeV
    bbn._I0 = quad(
        lambda E: math.sqrt(E * E - 1.0) * E * (bbn.Q - E) ** 2, 1.0, bbn.Q
    )[0]


def _set_taun(val_s: float) -> None:
    bbn.TAU_N_s = val_s


def _restore_defaults() -> None:
    _set_Qnp(Q_NP_OBS_MeV)
    _set_taun(TAU_N_OBS_s)


def _Yp(eta: float) -> float:
    """Y_p from the validated weak-sector harness under ΛCDM expansion."""
    return bbn.run_weak_sector(bbn.lcdm_expansion(), eta).Y_p


# ===========================================================================
# PART A — the nucleon inputs BBN needs
# ===========================================================================
def part_A() -> None:
    print("=" * 78)
    print(" PART A — nucleon inputs the BBN weak sector requires")
    print("=" * 78)
    print(f"   Q_np = m_n−m_p = {Q_NP_OBS_MeV} MeV   → n/p freeze-out ∝ exp(-Q_np/T)")
    print(f"   g_A  = {G_A_OBS}  (→ τ_n = {TAU_N_OBS_s} s) → weak-rate normalization")
    print(f"          [1/τ_n ∝ G_F²(1+3g_A²); the (1+3g_A²) is the axial piece]")
    print( "   Δα_had / nucleon EM self-energy → the EM part of Q_np (see Part B)")
    print( "   B_D, ⟨σv⟩ nuclear rates → external, out-of-scope-by-construction.")
    print()
    print(" Of these, Q_np and g_A are the two the framework would have to DERIVE")
    print(" to turn Y_p (and the D/³He/⁷Li network) into a real prediction.")
    print()


# ===========================================================================
# PART B — Q_np decomposition: what the framework HAS vs MISSING
# ===========================================================================
def part_B() -> None:
    print("=" * 78)
    print(" PART B — Q_np = QCD(m_d−m_u) + QED(EM self-energy): have vs missing")
    print("=" * 78)
    md_minus_mu = M_D_FRAMEWORK_MeV - M_U_FRAMEWORK_MeV
    print(f"   Q_np(obs) = {Q_NP_OBS_MeV:.3f} MeV decomposes (BMW 2015 lattice) as")
    print(f"     QCD (strong-isospin, m_d−m_u-driven) = {Q_NP_QCD_LATTICE_MeV:+.2f}"
          f" ± {Q_NP_QCD_LATTICE_SIG} MeV")
    print(f"     QED (electromagnetic self-energy)     = {Q_NP_QED_LATTICE_MeV:+.2f}"
          f" ± {Q_NP_QED_LATTICE_SIG} MeV")
    print(f"     sum = {Q_NP_QCD_LATTICE_MeV + Q_NP_QED_LATTICE_MeV:+.2f} MeV"
          f"  (≈ obs {Q_NP_OBS_MeV:.2f})")
    print()
    print(" FRAMEWORK HAS (predictions/m_u.py, m_d.py — M_persistence):")
    print(f"     m_u = {M_U_FRAMEWORK_MeV} MeV,  m_d = {M_D_FRAMEWORK_MeV} MeV"
          f"  (MS-bar 2 GeV)")
    print(f"     m_d − m_u = {md_minus_mu:.2f} MeV")
    print(f"     vs lattice QCD contribution to Q_np = {Q_NP_QCD_LATTICE_MeV:.2f}"
          f" ± {Q_NP_QCD_LATTICE_SIG} MeV")
    delta = md_minus_mu - Q_NP_QCD_LATTICE_MeV
    print(f"     → framework m_d−m_u is in the RIGHT BALLPARK (Δ={delta:+.2f} MeV,"
          f" {delta/Q_NP_QCD_LATTICE_SIG:+.1f}σ_lat)")
    print()
    print(" BUT — the framework has the INPUT (m_d−m_u) not the MAP:")
    print("   • MISSING: the 3-quark BOUND-STATE matrix element that maps a")
    print("     quark mass difference (MS-bar 2 GeV) to a NUCLEON mass difference.")
    print("     This is the absent baryon sector. (m_d−m_u at 2 GeV ≠ the QCD")
    print("     contribution to M_n−M_p; they agree numerically here only because")
    print("     the ⟨N|q̄q|N⟩-type matrix element happens to be O(1) — that O(1)")
    print("     is exactly the unbuilt piece.)")
    print("   • MISSING/BLOCKED: the QED part (−1.00 MeV). Nucleon EM self-energy")
    print("     is the same continuum-QCD/HVP class as Δα_had — Clause-9 BLOCKED")
    print(".")
    print("   • Q_np is the Need-B/BR4 wall.")
    print()


# ===========================================================================
# PART C — g_A: no framework handle (fully open)
# ===========================================================================
def part_C() -> None:
    print("=" * 78)
    print(" PART C — g_A nucleon axial charge: no framework handle yet")
    print("=" * 78)
    print(f"   g_A(obs) = {G_A_OBS}.  Non-rel quark model gives g_A = 5/3 = 1.667;")
    print(f"   the reduction to {G_A_OBS} is a relativistic/QCD bound-state effect.")
    print( "   The framework has spinor-return / walker structure but NO derived")
    print( "   nucleon spin content → g_A is fully open, same baryon-sector gap as")
    print( "   Q_np's QCD matrix element. (5/3 is a free-quark count; the framework")
    print( "   could plausibly reach 5/3 but not the QCD-renormalized 1.27 without")
    print( "   the bound-state sector.)")
    print()


# ===========================================================================
# PART D — Y_p sensitivity → precision targets for the nucleon derivation
# ===========================================================================
def part_D() -> None:
    print("=" * 78)
    print(" PART D — Y_p sensitivity to Q_np and g_A (sets the precision target)")
    print("=" * 78)
    print(" Validated weak sector (lib/bbn_network.py) under ΛCDM expansion, so")
    print(" this isolates NUCLEON-input dependence (NOT the √g_* question).")
    print()
    eta = 6.1e-10
    _restore_defaults()
    Yp0 = _Yp(eta)
    print(f"   baseline (Q_np={Q_NP_OBS_MeV}, τ_n={TAU_N_OBS_s}, η={eta:.1e}):"
          f"  Y_p = {Yp0:.4f}")
    print(f"   (observed Y_p = {Y_P_OBS} ± {Y_P_OBS_SIGMA}, i.e. ±"
          f"{100*Y_P_OBS_SIGMA/Y_P_OBS:.1f}%)")
    print()

    # --- logarithmic sensitivity to Q_np ---
    frac = 0.02
    _set_Qnp(Q_NP_OBS_MeV * (1 + frac))
    Yp_qp = _Yp(eta)
    _set_Qnp(Q_NP_OBS_MeV * (1 - frac))
    Yp_qm = _Yp(eta)
    _set_Qnp(Q_NP_OBS_MeV)
    dlnYp_dlnQ = (math.log(Yp_qp) - math.log(Yp_qm)) / (2 * frac)

    # --- logarithmic sensitivity to τ_n (then convert to g_A) ---
    _set_taun(TAU_N_OBS_s * (1 + frac))
    Yp_tp = _Yp(eta)
    _set_taun(TAU_N_OBS_s * (1 - frac))
    Yp_tm = _Yp(eta)
    _restore_defaults()
    dlnYp_dlnTau = (math.log(Yp_tp) - math.log(Yp_tm)) / (2 * frac)
    # τ_n ∝ 1/(1+3 g_A²)  ⇒  d ln τ_n / d ln g_A = −2·3g_A²/(1+3g_A²)
    dlnTau_dlnGA = -2.0 * 3.0 * G_A_OBS ** 2 / (1.0 + 3.0 * G_A_OBS ** 2)
    dlnYp_dlnGA = dlnYp_dlnTau * dlnTau_dlnGA

    print(f"   d ln Y_p / d ln Q_np = {dlnYp_dlnQ:+.3f}")
    print(f"   d ln Y_p / d ln τ_n  = {dlnYp_dlnTau:+.3f}")
    print(f"   d ln τ_n / d ln g_A  = {dlnTau_dlnGA:+.3f}  (from 1/τ_n∝1+3g_A²)")
    print(f"   d ln Y_p / d ln g_A  = {dlnYp_dlnGA:+.3f}")
    print()
    obs_prec = Y_P_OBS_SIGMA / Y_P_OBS  # ~1.2%
    print(f"   To predict Y_p to the observed ±{100*obs_prec:.1f}%, the nucleon")
    print(f"   inputs must be derived to:")
    print(f"     Q_np :  ±{100*obs_prec/abs(dlnYp_dlnQ):.2f}%   "
          f"(= ±{Q_NP_OBS_MeV*obs_prec/abs(dlnYp_dlnQ):.3f} MeV on Q_np)")
    print(f"     g_A  :  ±{100*obs_prec/abs(dlnYp_dlnGA):.2f}%   "
          f"(= ±{G_A_OBS*obs_prec/abs(dlnYp_dlnGA):.3f} on g_A)")
    print()
    print(" → These are the SPECS the nucleon-sector derivation must hit. Q_np is")
    print("   the dominant lever (n/p freeze-out is exponential in Q_np/T).")
    print()


def verdict() -> None:
    print("=" * 78)
    print(" VERDICT — Stream 3 is GATED on the nucleon (baryon) sector")
    print("=" * 78)
    print(" • The BBN harness (weak sector → Y_p) is BUILT and validated; only the")
    print("   nucleon INPUTS (Q_np, g_A) are missing — plus the √g_* H-normalization")
    print("   (Stream-2-adjacent leading-factor question, separate).")
    print(" • Minimal unlock = the baryon (3-quark bound-state) sector, which the")
    print("   framework entirely lacks. It would supply: (i) the matrix element")
    print("   mapping m_d−m_u → the QCD part of Q_np (framework already has")
    print("   m_d−m_u≈2.51 MeV ≈ lattice +2.49 MeV — only the O(1) map is missing);")
    print("   (ii) the nucleon spin content → g_A.")
    print(" • STILL BLOCKED even with the baryon sector: the QED part of Q_np")
    print("   (−1.00 MeV nucleon EM self-energy) is Δα_had-class (Clause 9).")
    print(" • Part D gives the precision SPEC the derivation must hit.")
    print()
    print(" So Stream 3 cannot produce a Y_p prediction this session — and SHOULD")
    print(" NOT (the 7 moved files stay in proofs/). The honest path is: build the")
    print(" baryon sector (multi-session, Need-B/BR4 wall) → then the harness runs")
    print(" with framework-derived Q_np, g_A and Stream 2's MDL-grounded Boltzmann.")


def main() -> int:
    print("=" * 78)
    print(" STREAM 3 — nucleon-sector BBN gate: scoping + Y_p sensitivity")
    print("=" * 78)
    print()
    part_A()
    part_B()
    part_C()
    part_D()
    verdict()
    _restore_defaults()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
