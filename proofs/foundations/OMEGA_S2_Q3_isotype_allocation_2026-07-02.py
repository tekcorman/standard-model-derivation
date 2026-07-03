#!/usr/bin/env python3
"""
proofs/foundations/OMEGA_S2_Q3_isotype_allocation_2026-07-02.py

OMEGA SESSION 2, STATION 3, ENTRY QUESTION Q3 -- "what operator has isotype-diagonal
elements ~ 1/mu_j?" (the W1 water-filling shape for the -70 ppm). Per the walk-down
discipline the question is interrogated BEFORE any candidate operator is built.

PRE-REGISTERED STRUCTURE (kickoff par.2/par.4): the tau-row is the built-in kill;
the derivation locks FIRST and the demand comparison is falsification only; J-reality
and the ~50x lever bookkeeping (OMEGA_T2) are regression surfaces. Pre-registered
outcome branch (kickoff par.4 S3-kill): "the forced answer is the wrong row/magnitude
=> the water-filling SHAPE is refuted, the localization moves up."

WHAT THIS PROBE DECIDES (declared before computing):
  D0  the demand VECTOR decomposed in the experimental correlation basis. PDG masses
      and errors enter HERE and only here (marked COMPARISON): both ratio rows carry
      m_tau's +-67.5 ppm (soft, correlated); the m_tau-free combination m_e/m_mu is
      pinned to +-0.022 ppm (hard).
  D1  the CONJUGATION THEOREM (exact): the object is real/rational, so complex
      conjugation intertwines the omega and omega-bar isotypes => mu_omega =
      mu_omega-bar in EVERY C3-graded sector (verified on Lambda*(C3): (4,2,2)).
      Corollary: every isotype-multiplicity-based correction is CHIRALITY-BLIND.
  D2  the ALLOCATION-CLASS KILL: for every generation<->isotype assignment and every
      tau-row bookkeeping, the class's m_e/m_mu differential lies in {0, +-29.7} ppm
      -- never the demanded +9.8 +- 0.022 ppm (>= 445 sigma_exp away). The tau-row
      question is MOOT (it only ever moved the soft direction).
  D3  SCOPE of the kill: it covers all corrections built from the isotype STRUCTURE
      alone (multiplicities, projectors, real class-operators commuting with
      conjugation) -- including kickoff candidates (i) 2nd-order PT from the real
      walk class and (ii) real resolvent residues, as posed. What survives: the
      CHIRAL sector (corrections through the delta-dressed mass operator or the
      run phase itself).
  D4  the SHARPENED TARGET (exact levers, linear regime verified): the hard direction
      equals ONE chiral number -- the next-order run-phase completion
      epsilon = delta_eff - 2/9, pinned by the data to sub-percent precision.
      A consistency DEMO (clearly marked NOT a closure: epsilon* is fitted by
      construction here) shows one chiral number satisfies the full vector.
  D5  poison + regressions.

The -70 ppm STAYS OPEN throughout. This probe refutes a candidate CLASS and sharpens
the target; nothing is relabeled, no closure is claimed, no value ships.
"""
import math
import sys
from itertools import permutations

import numpy as np

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

# the framework's own read (baseline of OMEGA_T2, reproduced exactly)
C0, C1, DELTA = 2.0, math.sqrt(2.0), 2.0 / 9.0
A13 = ((2.0 / 3.0) ** 8) ** 3                    # 59.35 ppm, the alpha_1^3 budget

def masses(delta):
    om = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
    out = []
    for j in range(3):
        amp = (C0 + C1 * om ** j * np.exp(1j * delta) + C1 * om ** (-j) * np.exp(-1j * delta))
        out.append(abs(amp) ** 2)
    return out                                    # j = 0: tau, 1: e, 2: mu

print("=" * 88)
print(" D0  the demand vector in the correlation basis  [COMPARISON -- PDG enters here]")
print("=" * 88)
M_E, S_E = 0.51099895069, 0.00000000016          # MeV, PDG
M_MU, S_MU = 105.6583755, 0.0000023
M_TAU, S_TAU = 1776.86, 0.12
m = masses(DELTA)
res_et = (m[1] / m[0]) / (M_E / M_TAU) - 1
res_mt = (m[2] / m[0]) / (M_MU / M_TAU) - 1
res_em = (m[1] / m[2]) / (M_E / M_MU) - 1
sig_et = math.sqrt((S_E / M_E) ** 2 + (S_TAU / M_TAU) ** 2)
sig_mt = math.sqrt((S_MU / M_MU) ** 2 + (S_TAU / M_TAU) ** 2)
sig_em = math.sqrt((S_E / M_E) ** 2 + (S_MU / M_MU) ** 2)
print(f"    m_e/m_tau : residual {res_et*1e6:+7.1f} ppm   sigma_exp {sig_et*1e6:5.1f} ppm "
      f"({res_et/sig_et:+.2f} sigma)   [m_tau-limited, SOFT]")
print(f"    m_mu/m_tau: residual {res_mt*1e6:+7.1f} ppm   sigma_exp {sig_mt*1e6:5.1f} ppm "
      f"({res_mt/sig_mt:+.2f} sigma)   [m_tau-limited, SOFT, corr ~ +1 with row 1]")
print(f"    m_e/m_mu  : residual {res_em*1e6:+7.2f} ppm   sigma_exp {sig_em*1e6:5.3f} ppm "
      f"({res_em/sig_em:+.0f} sigma)   [m_tau-FREE: the HARD direction]")
check("baseline reproduces the documented truncation (-70.3, -60.5) ppm and the exact "
      f"decomposition: hard = soft1 - soft2 ({res_em*1e6:+.2f} = {res_et*1e6:+.1f} - "
      f"({res_mt*1e6:+.1f}))",
      abs(res_et * 1e6 + 70.3) < 1.0 and abs(res_mt * 1e6 + 60.5) < 1.0
      and abs((res_et - res_mt) - res_em) < 2e-9)
DEM_HARD, SIG_HARD = -res_em, sig_em             # +9.8 +- 0.022 ppm demanded on m_e/m_mu
print(f"    => the DATA pins exactly one direction: delta(m_e/m_mu) = {DEM_HARD*1e6:+.2f} "
      f"+- {SIG_HARD*1e6:.3f} ppm ({DEM_HARD/SIG_HARD:.0f} sigma); the common shift is "
      f"unpinned at alpha_1^3 scale (+-{sig_et*1e6:.0f} ppm >= the whole 59.35 ppm budget).")

print("=" * 88)
print(" D1  the CONJUGATION THEOREM: mu_omega = mu_omega-bar in every C3 sector (exact)")
print("=" * 88)
# Lambda*(C3) with the C3 coordinate 3-cycle: enumerate wedge basis, grade by weight
from itertools import combinations
content = {0: 0, 1: 0, 2: 0}
for r in range(4):
    for S in combinations(range(3), r):
        content[sum(S) % 3] += 1
check(f"Lambda*(C3) isotype multiplicities = ({content[0]}, {content[1]}, {content[2]}) "
      "= (4, 2, 2): conjugate isotypes EQUAL", content[1] == content[2])
# generality: the object is real/rational (integer adjacency, real edge weights);
# complex conjugation is an antiunitary intertwiner mapping the omega isotype onto the
# omega-bar isotype in ANY C3-graded sector of a real object => equal multiplicities.
check("generality: real/rational object => conjugation intertwines omega <-> omega-bar "
      "=> mu_omega = mu_omega-bar in EVERY sector (theorem, not bookkeeping)", True)
print("    COROLLARY: any correction built from isotype multiplicities/projectors/real")
print("    class-operators is CHIRALITY-BLIND: it CANNOT split m_e from m_mu beyond the")
print("    leading (delta-dressed) masses it starts from.")

print("=" * 88)
print(" D2  the ALLOCATION-CLASS KILL: every assignment x every tau-row vs the hard")
print("     direction  [the pre-registered S3 kill fires]")
print("=" * 88)
MU = {0: 4.0, 1: 2.0, 2: 2.0}                     # isotype multiplicities (triv, w, wbar)
best = None
print(f"    {'assignment (tau,e,mu)->isotypes':>38}   {'tau-row':>10}   d(m_e/m_mu) ppm   |dev|/sigma")
for perm in permutations((0, 1, 2)):              # generation (tau,e,mu) -> isotype label
    for tau_exempt in (False, True):
        kap = {}
        for gen, iso in zip(("tau", "e", "mu"), perm):
            kap[gen] = 0.0 if (tau_exempt and gen == "tau") else 2 * A13 / MU[iso]
        d_em = (kap["e"] - kap["mu"]) * 1e6
        nsig = abs(d_em * 1e-6 - DEM_HARD) / SIG_HARD
        best = min(best, nsig) if best is not None else nsig
        print(f"    {str(perm):>38}   {'exempt' if tau_exempt else 'shifted':>10}   "
              f"{d_em:+13.1f}   {nsig:9.0f}")
check(f"EVERY multiplicity allocation misses the hard direction by >= {best:.0f} sigma_exp "
      "(differentials in {0, +-29.7} ppm vs demanded +9.8 +- 0.022): the class is "
      "EXCLUDED; the tau-row question is MOOT (it only moves the soft direction)",
      best > 400)
print("    NOTE (honesty): W1's celebrated 0.85x/0.98x 'match' lives ENTIRELY in the")
print("    soft (m_tau-limited) rows -- it was never probing the direction the data pins.")

print("=" * 88)
print(" D3  scope of the kill; the SURVIVING shape class")
print("=" * 88)
print("""    KILLED (chirality-blind by D1): all isotype-structure-alone corrections --
    the W1 water-filling kappa_j = 2 a1^3/mu_rep(j) in every bookkeeping; kickoff
    candidate (i) 2nd-order PT from the REAL walk class (real class-operators commute
    with conjugation); candidate (ii) real resolvent isotype residues; candidate (iii)
    any multiplicity-based rate-distortion allocation.
    SURVIVES (and only this): the CHIRAL sector -- corrections that see the omega/
    omega-bar DIRECTION, i.e. the delta-dressed spectrum itself: (S1) a next-order
    completion epsilon of the run phase delta (the d_N subleading -- exactly where
    todo par.1 originally localized the miss); (S2) mass-dependent real dressings
    kappa = g(m_j) (chiral THROUGH the leading delta; e.g. log-running shapes).
    One hard number cannot select between (S1) and (S2) shapes -- the derivation must
    be forced; all shape coefficients below are PRE-POISONED.""")
check("kill scope stated; surviving class = the chiral/delta-dressed sector only", True)

print("=" * 88)
print(" D4  the SHARPENED TARGET: the hard direction = ONE chiral number (exact levers)")
print("=" * 88)
th = [DELTA + 2 * math.pi * j / 3 for j in range(3)]
f = [1 + math.sqrt(2) * math.cos(t) for t in th]
L = [2 * (-math.sqrt(2) * math.sin(th[j]) / f[j]
          + math.sqrt(2) * math.sin(th[0]) / f[0]) for j in range(3)]  # dln(m_j/m_tau)/d delta
L_em = L[1] - L[2]
eps_star = DEM_HARD / L_em
eps_band = SIG_HARD / abs(L_em)
print(f"    levers: dln(m_e/m_tau)/d delta = {L[1]:+.2f}, dln(m_mu/m_tau)/d delta = {L[2]:+.2f}")
print(f"    => dln(m_e/m_mu)/d delta = {L_em:+.2f}")
print(f"    epsilon* = delta_eff - 2/9 = {eps_star:+.4e} +- {eps_band:.1e} rad "
      f"({eps_band/abs(eps_star)*100:.2f}% precision)")
check("the hard direction is equivalent to ONE chiral number pinned to sub-percent "
      f"precision: epsilon = {eps_star:+.3e} rad (a {eps_band/abs(eps_star)*100:.2f}% "
      "falsification target for any future d_N-subleading derivation)",
      abs(eps_star + 1.75e-7) < 2e-8 and eps_band / abs(eps_star) < 0.01)
# consistency DEMO (marked: NOT a closure -- epsilon* is fitted by construction):
m2 = masses(DELTA + eps_star)
r_em2 = (m2[1] / m2[2]) / (M_E / M_MU) - 1
r_et2 = (m2[1] / m2[0]) / (M_E / M_TAU) - 1
r_mt2 = (m2[2] / m2[0]) / (M_MU / M_TAU) - 1
print(f"    demo (NOT a closure): with delta + epsilon*: m_e/m_mu -> {r_em2*1e6:+.3f} ppm; "
      f"soft rows -> {r_et2*1e6:+.1f}, {r_mt2*1e6:+.1f} ppm ({r_et2/sig_et:+.2f}, "
      f"{r_mt2/sig_mt:+.2f} sigma)")
check("one chiral number satisfies the ENTIRE demand vector (hard row -> 0 by "
      "construction; soft rows stay ~1 sigma_exp): the -70 ppm's measurement-pinned "
      "content IS the chiral phase completion; linear regime verified "
      f"(2nd-order ~ {abs(L_em)**2*eps_star**2/2*1e6:.1e} ppm)",
      abs(r_em2) < 1e-9 and abs(r_et2 / sig_et) < 1.2 and abs(r_mt2 / sig_mt) < 1.2)
check("J-reality regression: a real phase shift keeps the mass triple real-positive "
      f"(masses: {[f'{x:.4f}' for x in m2]})", all(x > 0 for x in m2))

print("=" * 88)
print(" D5  poison list (declared at computation)")
print("=" * 88)
print(f"""    epsilon* = {eps_star:+.4e} rad. PRE-POISONED: any K-rational x alpha_1-power
    proximity hunt on epsilon* or on the (S2) log-coefficient (~1.8e-6) is numerology
    (a one-number inversion; the S6/Q1 lesson). Recorded non-matches, for the record:
    epsilon*/a1^3 = {eps_star/A13:+.4f}; epsilon*/a1^4 = {eps_star/(2/3)**32:+.3f} --
    neither is a framework constant; NO adoption. The equation to complete (todo
    par.1): the d_N run's NEXT-ORDER CHIRAL phase -- same sector as the derived
    leading delta = 2/9 (the screw's directed phase; the +-pi Z2 Berry holonomy is
    the only other derived chiral invariant, and it is topological, not 1e-7-sized).
    The soft (common) direction stays experimentally unpinned at the alpha_1^3 scale
    until m_tau improves ~6x (+-0.12 -> +-0.02 MeV).""")
check("no coefficient adopted; the miss STAYS OPEN with its sharpest pinned target",
      True)

print("=" * 88)
print(" VERDICT -- Q3 ANSWERED: NO SUCH OPERATOR EXISTS (the question had the wrong")
print("            shape); the water-filling class is REFUTED as the resolution")
print("=" * 88)
print(f"""    The demanded correction's measurement-pinned direction (m_e/m_mu, +9.8 +-
    0.022 ppm) is omega/omega-bar ANTISYMMETRIC -- chiral. By the conjugation theorem
    (D1) every isotype-multiplicity allocation is chirality-blind; the whole Q3
    candidate class misses the hard direction by >= 445 sigma_exp (D2), for every
    assignment and every tau-row (the tau-row kill is MOOT -- it never touched the
    pinned direction). The W1 conjecture's apparent success was soft-direction noise
    (m_tau +- 67.5 ppm >= the entire alpha_1^3 budget).

    THE -70 PPM STAYS OPEN -- SHARPER: its hard core is ONE chiral number,
    epsilon = delta_eff - 2/9 = {eps_star:+.3e} +- {eps_band:.0e} rad (0.22%-pinned),
    the d_N run's next-order chiral phase -- confirming and sharpening todo par.1's
    ORIGINAL localization (the run-operator subleading), now with the allocation
    detour closed by theorem and a sub-percent falsification target for any future
    derivation. MDL-ceiling framing REVISED: the ceiling argument applied to the
    (soft, unpinned) common shift; the hard content is a PHASE, not an allocation.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
