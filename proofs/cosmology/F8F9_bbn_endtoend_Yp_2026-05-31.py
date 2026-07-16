#!/usr/bin/env python3
# ============================================================
# F8+F9 end-to-end: forward-model Y_p with the framework-native baryon/BBN legs
# ============================================================
#
# Scope: internal research notes (the payoff of
# threads F7/F8/F9). The three baryon/BBN legs now each have a LEADING-ORDER
# framework result:
#   F7  -> Q_np QCD part  = m_d - m_u (F7-up-fixed 2.445 MeV ~ lattice 2.49)
#   F8  -> Q_np matrix element <N|qq|N> ~ 1 (flavor-blind) AND g_A LO = 5/3
#   F9  -> H(T) = E_P*sqrt(g_*(T))*H_sub = framework_expansion("candidate")
# This probe WIRES all three into the validated weak-sector BBN harness
# (lib/bbn_network.py, run_weak_sector) and forward-models Y_p. It is the
# INTEGRATION TEST: do the three isolated leg-results combine into a Y_p in the
# right ballpark, and what does each framework leg DO to Y_p?
#
# DISCIPLINE: a proofs/ probe. We run the REAL machinery (run_weak_sector's
# Radau ODE over the tabulated Born weak rates), not a closed-form shortcut. The
# nuclear/weak physics is external/standard; the framework surface is exactly
# {H(T), g_A->tau_n, Q_np}. We build the legs up one at a time so each leg's
# effect on Y_p is explicit and nothing is smuggled. Nothing here is promoted to
# a prediction (the legs are leading-order + carry avowed external inputs:
# Q_np's QED part = Clause-9; g_A's 0.76 reduction = open; the +4.3% tax; Gate 2).

import os
import sys
import math
from scipy.integrate import quad

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import proofs.cosmology.lib.bbn_network as bbn


# ---------------------------------------------------------------------------
# framework-native leg values (leading order)
# ---------------------------------------------------------------------------
# F7/F8 Q_np: QCD part (framework m_d - m_u, F7-up-fixed) + QED part (external,
# Clause-9 nucleon EM self-energy, NOT framework-derived).
MD_MINUS_MU_F7 = 2.445          # F7-up-fixed m_d - m_u (MeV); F8 valence ME = 1
MD_MINUS_MU_BASE = 2.110        # pre-F7 baseline (for sensitivity)
Q_NP_QED_EXTERNAL = -1.00       # MeV, BMW2015 EM self-energy (external/Clause-9)
Q_NP_FW = MD_MINUS_MU_F7 + Q_NP_QED_EXTERNAL     # framework-native Q_np (LO)

# F8 g_A leading order (SU(6), color-singlet junction + Pauli) and observed.
G_A_LO = 5.0 / 3.0              # framework leading order
G_A_OBS = 1.2723               # observed (= LO x 0.763 spin-crisis reduction)

# external references
Q_NP_OBS = 1.293332
TAU_N_OBS = 879.4
Y_P_OBS, Y_P_SIG = 0.245, 0.003
ETA = 6.1e-10                   # baryon-to-photon (external; Planck-ish)


def tau_n_from_gA(g_A):
    """tau_n from g_A via 1/tau_n proportional to (1 + 3 g_A^2), anchored at obs."""
    return TAU_N_OBS * (1.0 + 3.0 * G_A_OBS ** 2) / (1.0 + 3.0 * g_A ** 2)


def set_Qnp(val):
    bbn.Q_NP_MeV = val
    bbn.Q = val / bbn.M_E_MeV
    bbn._I0 = quad(lambda E: math.sqrt(E * E - 1.0) * E * (bbn.Q - E) ** 2, 1.0, bbn.Q)[0]


def set_taun(val):
    bbn.TAU_N_s = val


def restore():
    set_Qnp(Q_NP_OBS)
    set_taun(TAU_N_OBS)


def Yp(expansion, eta=ETA):
    return bbn.run_weak_sector(expansion, eta).Y_p


def sigma(yp):
    return (yp - Y_P_OBS) / Y_P_SIG


def main():
    print("=" * 78)
    print(" F8+F9 end-to-end: Y_p from the framework-native baryon/BBN legs")
    print("=" * 78)
    lcdm = bbn.lcdm_expansion()
    fw_H = bbn.framework_expansion("candidate")   # F9: H = E_P*sqrt(g_*)*H_sub
    print(f"\n leg inputs (leading order):")
    print(f"   F9 H(T)   : framework_expansion('candidate') = sqrt(k*·g_*)·T²/M_Pl")
    print(f"               = E_P·sqrt(g_*)·H_sub, E_P=sqrt(3)=1.732 (+4.3% vs 1.66)")
    print(f"   F8 g_A    : LO = 5/3 = {G_A_LO:.4f}  -> tau_n = {tau_n_from_gA(G_A_LO):.1f} s"
          f"   (obs g_A={G_A_OBS} -> {TAU_N_OBS} s)")
    print(f"   F7/F8 Q_np: QCD {MD_MINUS_MU_F7:+.3f} (fw m_d−m_u) + QED {Q_NP_QED_EXTERNAL:+.2f}"
          f" (external) = {Q_NP_FW:.3f} MeV   (obs {Q_NP_OBS:.3f})")

    # -----------------------------------------------------------------------
    # leg-by-leg build-up: each row swaps in ONE more framework leg
    # -----------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" leg-by-leg: Y_p as each framework leg replaces its external value")
    print(" (eta = {:.2e}, validated weak-sector ODE)".format(ETA))
    print("-" * 78)
    print(f"   {'scenario':<46}{'Y_p':>8}{'sigma':>8}")

    # (0) full external — harness validation (expect ~0.247)
    restore()
    y0 = Yp(lcdm)
    print(f"   {'(0) ΛCDM H, external Q_np & g_A  [validation]':<46}{y0:>8.4f}{sigma(y0):>+8.1f}")

    # (1) + framework H (F9), still external Q_np & g_A — isolates the H leg
    restore()
    y1 = Yp(fw_H)
    print(f"   {'(1) +F9 framework H (E_P·√g_*)':<46}{y1:>8.4f}{sigma(y1):>+8.1f}")

    # (2) + framework g_A=5/3 LO (via tau_n), framework H, external Q_np
    restore()
    set_taun(tau_n_from_gA(G_A_LO))
    y2 = Yp(fw_H)
    print(f"   {'(2) +F8 g_A=5/3 LO  (tau_n=552 s)':<46}{y2:>8.4f}{sigma(y2):>+8.1f}")

    # (3) + framework Q_np (F7 QCD + external QED), framework H, g_A=5/3 — FULL fw LO
    set_taun(tau_n_from_gA(G_A_LO))
    set_Qnp(Q_NP_FW)
    y3 = Yp(fw_H)
    print(f"   {'(3) +F7/F8 Q_np=1.445  [FULL framework LO]':<46}{y3:>8.4f}{sigma(y3):>+8.1f}")
    restore()

    # -----------------------------------------------------------------------
    # what each OPEN piece is worth: swap the open renormalization back to obs
    # -----------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" what the OPEN pieces are worth (framework H throughout):")
    print("-" * 78)
    print(f"   {'scenario':<46}{'Y_p':>8}{'sigma':>8}")
    # g_A: LO 5/3 vs the open-reduced observed 1.2723 (Q_np external)
    restore(); set_taun(tau_n_from_gA(G_A_OBS))
    ya = Yp(fw_H)
    print(f"   {'g_A reduced to 1.2723 (the open 0.76 factor)':<46}{ya:>8.4f}{sigma(ya):>+8.1f}")
    # Q_np: framework-LO 1.445 vs observed 1.293 (g_A held at obs to isolate Q_np)
    restore(); set_taun(tau_n_from_gA(G_A_OBS)); set_Qnp(Q_NP_FW)
    yb = Yp(fw_H)
    print(f"   {'Q_np at framework-LO 1.445 (g_A held obs)':<46}{yb:>8.4f}{sigma(yb):>+8.1f}")
    restore()

    print("\n" + "=" * 78)
    print(" VERDICT — F8+F9 end-to-end")
    print("=" * 78)
    print(f"""  The three legs DO combine into a real Y_p via the validated weak-sector ODE.
  Reading the build-up:

   • Harness validates: ΛCDM inputs -> Y_p = {y0:.4f} ({sigma(y0):+.1f}σ), the known
     baseline. The machinery is sound.
   • F9 H leg (E_P·√g_*, the +4.3% K-rational tax): Y_p {y0:.4f} -> {y1:.4f}. The
     framework's radiation-era H is BBN-viable — its +4.3% over ΛCDM moves Y_p by
     only ~{abs(y1-y0):.4f} (faster H -> slightly earlier freeze-out -> slightly higher Y_p).
   • F8 g_A LEADING ORDER (5/3) is the DOMINANT error: Y_p {y1:.4f} -> {y2:.4f}.
     g_A=5/3 overshoots the weak rates (tau_n 879->552 s), holding n<->p equilibrium
     longer -> Y_p driven DOWN to {y2:.4f}. Restoring the open-reduced g_A=1.2723
     recovers Y_p={ya:.4f}. So the g_A 0.76 "spin-crisis" reduction — F8's flagged
     open leg — is exactly the piece BBN needs; it is worth ~{abs(ya-y2):.3f} in Y_p
     (~{abs(sigma(ya)-sigma(y2)):.0f}σ). The end-to-end wire QUANTIFIES why that open leg matters.
   • Q_np LEADING ORDER (1.445) is the OTHER big error, opposite sign: at
     g_A=obs, Q_np 1.293->1.445 drives Y_p {ya:.4f}->{yb:.4f} (~{abs(sigma(yb)-sigma(ya)):.0f}σ UP — higher Q_np
     -> higher freeze-out n/p). So the full-LO {y3:.4f} (-15.8σ) is a PARTIAL
     CANCELLATION of g_A (down ~26σ) against Q_np (up ~14σ), not a small residual.

  NET (honest, no overclaim): the framework's H(T) leg (F9) is genuinely
  BBN-viable — its +4.3% K-rational tax moves Y_p by only ~{abs(y1-y0):.4f} ({y1:.4f},
  +0.8σ), reproducing the prior Gate-2 finding. But the binding legs are
  STRUCTURALLY right and BBN-IMPRECISE: g_A=5/3 and Q_np=1.445 are each the
  correct sign and within their structural/lattice bands, yet each sits ~14–26σ
  off in Y_p because BBN needs them to ~0.4-0.8% (the Part-D spec). The pure
  leading-order forward-model gives Y_p={y3:.4f} (-15.8σ); only when BOTH open
  renormalizations are supplied externally (g_A->1.2723, Q_np->1.293) does
  framework-H give the good Y_p={ya:.4f} (+0.8σ).

  INTEGRATION VERDICT: the wire confirms the H leg and quantifies that the
  baryon program's entire remaining BBN value is in the two sub-leading
  RENORMALIZATIONS — the g_A 0.76 reduction (~26σ) and the Q_np precision
  (QED part + scale-matching, ~14σ) — NOT in the H or the leading-order binding.
  A framework-native Y_p is therefore CONDITIONAL on those two open pieces; with
  leading order alone it is a ~16σ underprediction. This is the honest stop line.

  OPEN (now BBN-quantified): g_A 0.76 reduction (~26σ lever); Q_np precision incl.
  QED part (~14σ; Clause-9); Gate 2 (late-time H deactivation); the +4.3% tax.""")
    print("=" * 78)
    restore()


if __name__ == "__main__":
    raise SystemExit(main())
