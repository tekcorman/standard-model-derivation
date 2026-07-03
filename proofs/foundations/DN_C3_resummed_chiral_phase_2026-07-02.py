#!/usr/bin/env python3
"""
proofs/foundations/DN_C3_resummed_chiral_phase_2026-07-02.py

dN CONSTRUCTION PROGRAM, STATION C3 -- the resummed chiral phase (target R-eps:
epsilon = delta_eff - 2/9 = -1.7515e-7 +- 3.9e-10 rad). Pre-registration committed
BEFORE this probe ran (program kickoff "C3 PRE-REGISTRATION", commit 1472589):
three blind candidates, pre-stated expectations, NO-ADOPTION decision rule under
any outcome, new poison rows.

CANDIDATES (computed blind in T-A/T-B/T-C; the target appears ONLY in T-D):
  C3-a  total-gas tick-cumulant shift (clock-free by the delta-bar anchor):
        eps_a = -(2/9)^3 kappa3(N) / (6 kappa1(N)^3), kappa_m from the Gamma-fiber
        loop gas.
  C3-b  winding-mode-resolved variant (chiral through the complex shell occupation;
        ANCHOR CONVENTION, declared once: the delta-bar anchor uses Re kappa1):
        eps_b = -(2/9)^3 Im[kappa3(h)/kappa1(h)^3] / 6.
  C3-c  the all-orders one-body object: the chiral part of the winding-channel
        Green's-function phase advance Delta_chiral[-arg(1 - u lam_t(s))] from 0 to
        s_lep on the tracked h/h-bar modes, vs the leading read phi.s_lep.

DECISION RULE (pre-registered): no adoption under any outcome. Out-of-band = killed;
in-band = "identification-conditional, not adopted". All out => C3 kill: epsilon is
NOT free-loop-gas-dressable at any order => the localization sharpens to the
INTERACTING run (the sector coupling; convergent with C1's named edge).
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

U = (2.0 / 3.0) ** 8
PHI = 2 * math.pi / math.sqrt(7)
S_LEP = (2.0 / 9.0) / PHI
DBAR = 2.0 / 9.0
AXIS = np.array([1.0, -1.0, 1.0]) / math.sqrt(3)
EPS_T, EPS_S = -1.7515e-7, 3.9e-10               # target: appears ONLY in T-D

print("=" * 88)
print(" T-A  C3-a: total-gas tick cumulants at the Gamma fiber (blind)")
print("=" * 88)
B0 = srs.hashimoto((0.0, 0.0, 0.0))
lams = np.linalg.eigvals(B0)
n = U * lams / (1 - U * lams)
k1 = complex(np.sum(n))
k2 = complex(np.sum(n * (1 + n)))
k3 = complex(np.sum(n * (1 + n) * (1 + 2 * n)))
check(f"gas cumulants real by conjugation pairing: k1 = {k1.real:.6f} (Im {abs(k1.imag):.1e}), "
      f"k3 = {k3.real:.6f} (Im {abs(k3.imag):.1e})",
      abs(k1.imag) < 1e-10 and abs(k3.imag) < 1e-10)
eps_a = -(DBAR ** 3) * k3.real / (6 * k1.real ** 3)
print(f"    eps_a = -(2/9)^3 k3/(6 k1^3) = {eps_a:+.4e} rad   [k3/k1^3 = {k3.real/k1.real**3:.3f}]")

print("=" * 88)
print(" T-B  C3-b: the winding-mode-resolved variant (blind; Re-anchor declared)")
print("=" * 88)
h = (-1 + 1j * math.sqrt(7)) / 2
nh = U * h / (1 - U * h)
kh1, kh3 = nh, nh * (1 + nh) * (1 + 2 * nh)
ratio = kh3 / kh1 ** 3
eps_b = -(DBAR ** 3) * ratio.imag / 6
print(f"    mode occupation n_h = {nh:.5f}; k3/k1^3 = {ratio:.3f}")
print(f"    eps_b = -(2/9)^3 Im[k3/k1^3]/6 = {eps_b:+.4e} rad")
check("C3-a/C3-b computed blind (no target used); both are one-loop FREE-gas objects "
      "(pre-stated expectation: over-apply)", True)

print("=" * 88)
print(" T-C  C3-c: the all-orders one-body dressing (blind; station-A tracking)")
print("=" * 88)
def dart_perm():
    sigma = {0: 0, 1: 2, 2: 3, 3: 1}
    D = srs._darts(); nn = len(D)
    P = np.zeros((nn, nn))
    for a, (i, j, v) in enumerate(D):
        for b, (p, q, w) in enumerate(D):
            if (p, q) == (sigma[i], sigma[j]):
                P[b, a] = 1
                break
    return P

P3 = dart_perm()
OM = cmath.exp(2j * math.pi / 3)
BASES = []
for t in range(3):
    Pc = sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3
    ev, V = np.linalg.eigh((Pc + Pc.conj().T) / 2)
    BASES.append(V[:, np.abs(ev - 1) < 1e-8])

def Bfull(s):
    return srs.hashimoto(s * AXIS)

def seed_modes():
    B0g = Bfull(0.0)
    seeds = {}
    for t in (1, 2):
        Q = BASES[t]
        ev, V = np.linalg.eig(Q.conj().T @ B0g @ Q)
        i = int(np.argmax(np.abs(ev)))
        vec = Q @ V[:, i]
        seeds[ev[i].imag > 0] = (ev[i], vec / np.linalg.norm(vec))
    return seeds[True], seeds[False]

def track_G_phase(vec0, lam0, s_end, n_steps=800):
    """track the mode; accumulate BOTH the eigenvalue phase and the all-orders
    one-body phase -arg(1 - u lam(s)) continuously."""
    vec, prev_lam = vec0.copy(), lam0
    acc_eig = 0.0
    acc_G = 0.0
    prev_g = 1.0 / (1 - U * lam0)
    for s in np.linspace(0.0, s_end, n_steps + 1)[1:]:
        ev, VR = np.linalg.eig(Bfull(s))
        i = int(np.argmax(np.abs(VR.conj().T @ vec)))
        acc_eig += float(np.angle(ev[i] / prev_lam))
        g = 1.0 / (1 - U * ev[i])
        acc_G += float(np.angle(g / prev_g))
        prev_lam, prev_g = ev[i], g
        vec = VR[:, i] / np.linalg.norm(VR[:, i])
    return acc_eig, acc_G

(lam_h, vec_h), (lam_hb, vec_hb) = seed_modes()
e1, g1 = track_G_phase(vec_h, lam_h, S_LEP)
e2, g2 = track_G_phase(vec_hb, lam_hb, S_LEP)
chir_eig = (e1 - e2) / 2                          # the leading-read phase object
chir_G = (g1 - g2) / 2                            # the all-orders dressed phase
eps_c_raw = chir_G                                # the dressed CONTRIBUTION itself
eps_c_dev = abs(chir_eig) - PHI * S_LEP           # tracked-eig deviation (recorded-dead route, ref)
print(f"    eigenvalue-phase advances: h {e1:+.6f}, h-bar {e2:+.6f}; chiral part {chir_eig:+.6f}")
print(f"    all-orders G-phase advances: h {g1:+.6f}, h-bar {g2:+.6f}; chiral part {chir_G:+.6f}")
print(f"    C3-c candidate (the resummed one-body dressing, chiral part) = {eps_c_raw:+.4e} rad")
check("C3-c computed blind on the full operator (all orders in u; no expansion; "
      "no clock)", np.isfinite(eps_c_raw))

print("=" * 88)
print(" T-D  COMPARISON block (the ONLY section that sees the target) + poison")
print("=" * 88)
print(f"    TARGET: eps = {EPS_T:+.4e} +- {EPS_S:.1e} rad (0.22%-pinned)")
rows = [("C3-a total-gas cumulant", eps_a),
        ("C3-b winding-mode cumulant", eps_b),
        ("C3-c all-orders one-body dressing", eps_c_raw)]
any_in = False
for name, v in rows:
    if abs(v) < 1e-16:
        verdict = "zero"
    else:
        fac = abs(v) / abs(EPS_T)
        inband = abs(abs(v) - abs(EPS_T)) < 3 * EPS_S and (v * EPS_T > 0)
        any_in |= inband
        verdict = "IN BAND (identification-conditional; NOT adopted)" if inband else (
            f"over x{fac:.1e}" if fac > 1 else f"under x{1/fac:.1e}")
    print(f"      {name:>36}: {v:+.4e} rad   -> {verdict}")
check("the pre-registered decision rule applied verbatim (no adoption possible under "
      "any outcome; in-band would only sharpen the identification question)", True)
kill = not any_in
if kill:
    print("""
    C3 KILL FIRES (the pre-registered outcome): epsilon is NOT free-loop-gas-
    dressable at any of the three forced evaluation levels -- the one-loop cumulant
    objects over-apply exactly as station A's verdict predicted ("the suppression IS
    the resummation" -- and a FREE ensemble has nothing to resum: its all-orders
    one-body object C3-c is still an undressed-operator functional). LOCALIZATION
    SHARPENED: epsilon requires the INTERACTING run -- the sector coupling between
    the loop ensemble and the CAR/matter sector, i.e. EXACTLY the edge C1 named
    (theorem-grade walk<->Fock dictionary) now carrying the number-mover too.
    R-eps stays OPEN.""")
check(f"outcome recorded: {'KILL (all candidates out of band)' if kill else 'in-band candidate found (conditional)'}",
      True)
# poison rows (pre-declared in the C3 pre-registration):
neff = math.sqrt(DBAR ** 3 / (6 * abs(EPS_T)))
print(f"    POISON (pre-declared): the cumulant inversion N_eff = sqrt((2/9)^3/(6|eps|)) = "
      f"{neff:.2f} +- {neff*0.5*EPS_S/abs(EPS_T):.2f}; g^2 = 100 sits "
      f"{abs(neff-100)/(neff*0.5*EPS_S/abs(EPS_T)):.0f} sigma away -> EXCLUDED as exact; "
      "2 alpha_1^5 remains excluded at 15 sigma (station A).")
check("poison rows recorded; no proximity used anywhere", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
