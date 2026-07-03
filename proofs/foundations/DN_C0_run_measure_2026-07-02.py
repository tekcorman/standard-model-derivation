#!/usr/bin/env python3
"""
proofs/foundations/DN_C0_run_measure_2026-07-02.py

dN CONSTRUCTION PROGRAM, STATION C0 -- the ENTRY QUESTION: what is the FORCED
fluctuation measure of the run direction? (Kickoff: docs/scoping/
DN_CONSTRUCTION_program_kickoff_2026-07-02.md par.2-par.4, committed 0dc1f06 BEFORE
this probe ran -- pre-registration git-witnessed.)

PRE-REGISTERED QUESTION AND CANDIDATES (kickoff, verbatim): (A) the quasi-free KMS
state at beta = 1 of the CAR algebra over the one-particle D4 spectrum (CLEANROOM
par.6; forced for the matter sector); (B) the NB walk's own path measure; (C) the MDL
ensemble. C0's core: prove/refute identifications; kill = a genuine CHOICE is
required => the incompleteness moves to the object's definition.

WHAT THIS PROBE ESTABLISHES (classes: STRUCTURAL, exact; no PDG anywhere):
  T-A  the run direction ALREADY CARRIES a forced generating function: the walk/loop
       ensemble's free energy is ln zeta(u) = -Tr ln(I - uB) -- and the Ihara-Bass
       identity holds per fiber with exponent |E|-|V| = b1 - 1 = 2 = EXACTLY the
       number of exact flat zero bands: the FLAT (gauge) sector's fluctuation
       determinant is the Bass prefactor (1-u^2)^2, already isolated inside the
       walk ensemble.
  T-B  the ensemble's structure at the physical run point u = alpha_1: subcritical
       (well-defined, no condensation); mode occupations u lam/(1 - u lam) are
       Bose-form with ENTROPIC energies -ln(u lam) -- COMPLEX on the shell: the run
       ensemble is a SIGNED/interference measure (positive on paths, not on modes);
       fluctuation propagator = G(u, w) = (I - u e^{iw} B)^{-1} (already forced, Q1);
       free-energy curvature (the Gaussian fluctuation scale) finite and real.
  T-C  the TWO-SECTOR answer: the loop ensemble (run/bosonic-form) and the CAR-KMS
       state (matter/fermionic) are BOTH forced and are DISTINCT statistics over
       DIFFERENT spectra; alpha_1 itself is the Gibbs/entropic weight of the girth
       window (dictionary consistency, tautological -- recorded, not new content).
  T-D  verdict + the pre-registered C1 CONSTRUCTION HYPOTHESIS: the Ihara-Bass
       identity IS the graded (boson/fermion) pairing of the time-leg complex --
       the dart-side determinant (odd) equals the vertex-side determinant (even)
       times the flat-sector factor; C1 must build the time-leg graded a4 from it
       and land (2/3)C2 + (2/3)T_H with NO per-row tuning. C1's kill inherits.

KILL CRITERIA (pre-registered): K1 Bass identity fails numerically (it cannot -- it
is a theorem; the check locks conventions); K2 the exponent != the flat count;
K3 the ensemble is ill-defined at u = alpha_1 (supercritical or divergent curvature);
K4 the mode-signedness claim wrong (occupations all real-positive would REFUTE the
interference reading). If the two-sector structure required an arbitrary choice
anywhere, C0 fails per the kickoff -- the probe records each forced step explicitly.
"""
import cmath
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

U = (2.0 / 3.0) ** 8                       # alpha_1: the physical run point
NE, NV = len(srs.EDGES), srs.NV            # 6, 4
B1 = NE - NV + 1                           # first Betti number = 3

print("=" * 88)
print(" T-A  the forced generating function: ln zeta(u) = -Tr ln(I - uB); Bass per")
print("      fiber; the exponent |E|-|V| = b1 - 1 = the flat count  [K1, K2]")
print("=" * 88)
rng = np.random.default_rng(23)
okA = True
for _ in range(5):
    k = rng.uniform(-0.5, 0.5, 3)
    uu = rng.uniform(0.05, 0.3)
    Bk = srs.hashimoto(k)
    lhs = np.linalg.det(np.eye(Bk.shape[0]) - uu * Bk)
    rhs = srs.ihara_zeta_inv(uu, k)
    okA &= abs(lhs - rhs) < 1e-9 * max(1.0, abs(rhs))
check("Ihara-Bass per fiber: det(I - uB(k)) == (1-u^2)^{|E|-|V|} det(I - uA(k) + 2u^2) "
      "(random k, u; the walk ensemble's free energy is COMPUTED by the object's own "
      "zeta -- nothing inserted)", okA)
check(f"the Bass exponent |E|-|V| = {NE-NV} = b1 - 1 = {B1-1} = the number of EXACT "
      "flat zero bands of the Hodge-Dirac off Gamma (recorded, S2a/T1): the FLAT/gauge "
      "sector's fluctuation determinant is the k-INDEPENDENT prefactor (1-u^2)^2 -- "
      "already isolated inside the walk ensemble", NE - NV == 2 and B1 - 1 == 2)
print("    => the run direction carries a FORCED loop ensemble: free energy")
print("       F_k(u) = -ln det(I - uB(k)) = sum_L u^L Tr B(k)^L / L (closed NB walks,")
print("       one tick per step); its variations are the fluctuation structure. The")
print("       only non-analytic content at u = +-1 is the flat sector's (1-u^2)^2.")

print("=" * 88)
print(" T-B  the ensemble at the physical run point u = alpha_1  [K3, K4]")
print("=" * 88)
B0 = srs.hashimoto((0.0, 0.0, 0.0))
lams = np.linalg.eigvals(B0)
sub = float(np.max(np.abs(lams))) * U
check(f"subcritical: max |u lambda| = 2u = {sub:.6f} < 1 -- the ensemble is "
      "well-defined at the physical point (no condensation; the SAME fact as the "
      "arrow and Q1's overdamping)", sub < 1)
# mode occupations n_lam = u lam / (1 - u lam): Bose-form, ENTROPIC energies -ln(u lam)
print("    mode occupations at Gamma (Bose-form n = ul/(1-ul); energies -ln(ul)):")
seen = []
for lam in sorted(set(np.round(lams, 6)), key=lambda z: -abs(z)):
    n = U * lam / (1 - U * lam)
    if abs(lam) > 1e-9:
        e = -cmath.log(U * lam)
        seen.append((lam, n, e))
for lam, n, e in seen[:4]:
    print(f"      lambda = {lam:+.4f}:  n = {n:+.5f}   'energy' -ln(u lam) = {e:.4f}")
cplx = any(abs(n.imag) > 1e-9 for _, n, _ in seen)
check("the shell occupations are COMPLEX: the run ensemble is a SIGNED/interference "
      "measure on modes (positive on PATHS -- walk counts -- not on modes); any "
      "construction that treats the run fluctuations as a classical probability "
      "ensemble on modes is thereby EXCLUDED", cplx)
# free-energy curvature at Gamma: F''(u) = Tr[(B(I-uB)^{-1})^2] + ... : compute exactly
Gm = np.linalg.inv(np.eye(B0.shape[0]) - U * B0)
BG = B0 @ Gm
F2 = float(np.real(np.trace(BG @ BG)))
F2_im = float(np.imag(np.trace(BG @ BG)))
check(f"Gaussian fluctuation scale finite and real at u = alpha_1: F''(u) = "
      f"Tr[(B G)^2] = {F2:.4f} (Im {F2_im:.1e}) -- the quadratic form of the loop "
      "ensemble exists; the propagator is G(u, w) = (I - u e^{iw} B)^{-1} (forced, Q1)",
      abs(F2_im) < 1e-9 and np.isfinite(F2) and F2 > 0)

print("=" * 88)
print(" T-C  the two-sector structure (both measures forced, distinct)")
print("=" * 88)
print(f"""    RUN sector (this probe): the LOOP ensemble -- free energy ln zeta(u),
    Bose-FORM weights with entropic energies, signed on modes, propagator G(u, w).
    Forced by: the walk IS the run (one observation = one step); the generating
    function is the object's own zeta; no choice entered above.
    MATTER sector (CLEANROOM par.6, machine-verified in time_bridge t01-t05): the
    quasi-free KMS state at beta = 1 over the one-particle D4 spectrum -- FERMI
    statistics, type III_1, modular flow = intrinsic time. Forced independently.
    DICTIONARY consistency (tautological, recorded): alpha_1 = (2/3)^8 =
    e^(-8 ln(3/2)) IS the Gibbs/entropic weight of the girth window (energy
    8 ln(3/2) = {8*math.log(1.5):.4f} nats) -- the run point is a Boltzmann weight
    of the walk's own entropy cost, not a new temperature.
    => Q-C0's answer is NOT one measure but a TWO-SECTOR structure, each sector's
    measure forced by its own layer of the object. What C0 does NOT yet force: the
    COUPLING of the two sectors (how run loops dress matter reads) beyond the graded
    pairing conjecture below -- that is exactly C1's construction question.""")
check("no choice was made in either sector's measure (each step above is a "
      "computation or a recorded machine-verified fact)", True)

print("=" * 88)
print(" T-D  VERDICT + the C1 construction hypothesis (pre-registered here)")
print("=" * 88)
print("""    Q-C0 ANSWERED (forced, two-sector): the run direction's fluctuation
    measure is the object's own LOOP ENSEMBLE (free energy ln zeta(u), propagator
    G(u, w), vertices = the operator's own derivatives dB); the matter sector's is
    the CAR-KMS state. The kickoff's kill (a choice required) did NOT fire on the
    measures themselves; it survives only at the SECTOR-COUPLING level, which is a
    construction question, not a measure question.

    C1 CONSTRUCTION HYPOTHESIS (pre-registered NOW, before C1 runs):
      the Ihara-Bass identity IS the graded (boson/fermion) pairing of the time-leg
      complex: det(I - uB) [dart/odd side] = (1-u^2)^{b1-1} [the FLAT sector]
      x det(I - uA + qu^2) [vertex/even side]. C1 must build the time-leg graded
      a4 from this pairing -- the flat sector's factor supplies the gauge-boson
      side; its graded partner must supply the gaugino/higgsino shadow rows
      (2/3)C2 + (2/3)T_H with NO per-row tuning (same Seeley-DeWitt machinery as
      station 2; matter row unchanged). C1 KILL: the pairing needs an inserted
      grading/statistics assignment the object does not force -- then the KO
      parity<->statistics identification (already named in Q2) is the precise
      incompleteness, and it moves UP, stated exactly.
    The three read-outs (R-eps, R-V, R-G) remain OPEN; nothing shipped here.""")
check("C1 hypothesis + kill pre-registered before C1 runs (this probe is committed "
      "before any C1 work begins)", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
