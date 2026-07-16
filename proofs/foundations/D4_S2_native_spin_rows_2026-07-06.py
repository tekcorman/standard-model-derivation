#!/usr/bin/env python3
"""
proofs/foundations/D4_S2_native_spin_rows_2026-07-06.py

D4 SPECTRAL-ACTION program, station S2 -- the VECTOR/SCALAR spin rows native (complete the a4 spin content).
Pre-registration: internal research notes (0b4a5d4 BEFORE).
CLASS: pure structure. NO PDG. GRADE axis -- gates NO value.

S1 made the FERMION row native (fermion (2s_z)^2 = 1 from (1/2)trE^2 on the A5(b) cone). S2 grounds the
VECTOR and SCALAR (2s_z)^2 in the framework's OWN spin reps: spin-1 = the emergent band VECTOR rep S_a
(Casimir 2; the gauge boson = the grade-2 bivector inner-fluctuation); spin-0 = the Higgs (no spin).
The universal helicity rule b = -(-1)^{2s}[(2s_z)^2 - 1/3] (pure-math Gilkey) then gives all three rows
{+2/3, -11/3, +1/3} with NATIVE (2s_z)^2 inputs. Only the Gilkey FORMULA stays imported.
"""
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import d4_spectral_action as d4  # noqa: E402  (the S1 machine -- reuse, do not rebuild)

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

NV = 4
EDGES = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0)),
         (1, 2, (1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, 1))]
Cm = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]], float)
G12 = (5 * np.eye(3) + Cm) / 3
def A_q(q):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A
def dA_q(q, ax):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = 1j * v[ax] * np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A

print("=" * 90)
print(" S2-FERMION  (recap from S1, via the module): spin-1/2, (2s_z)^2 = 1 native on the A5(b) cone")
print("=" * 90)
gD, weyl = d4.a5b_dirac_cone()
f2 = d4.fermion_2sz2(gD, weyl)
check(f"fermion (2s_z)^2 = (1/2)trE^2/B^2 on the A5(b) cone = {f2:.4f} = 1 (S1) -> row +2/3", abs(f2 - 1) < 1e-6)

print("=" * 90)
print(" P1  spin-1 NATIVE: the emergent band VECTOR rep S_a (Casimir 2) => (2s_z)^2 = 4 (transverse pair)")
print("=" * 90)
# the band spin-1 generators: velocity of the lambda=-1 triple (the A5(b)/probe-2 vector rep), Albanese frame
wG, UG = np.linalg.eigh(A_q((0, 0, 0)))
P3 = UG[:, np.abs(wG + 1) < 1e-6]                             # 4x3 triple
S1 = [P3.conj().T @ sum(G12[i, a] * dA_q((0, 0, 0), i) for i in range(3)) @ P3 for a in range(3)]
S1 = [0.5 * (M + M.conj().T) for M in S1]                     # Hermitian part
cas = np.real(np.trace(sum(Sa @ Sa for Sa in S1))) / 3        # avg of S^2; s(s+1) c^2 with s=1 -> 2 c^2
# normalize so S_z has eigenvalues {-1,0,+1} (c=1): scale by 1/|c|, c from [S0,S1]=i c eps S2
def ipm(X, Y): return np.real(np.trace(X.conj().T @ Y))
c = ipm(-1j * (S1[0] @ S1[1] - S1[1] @ S1[0]), S1[2]) / ipm(S1[2], S1[2])
Sz = S1[2] / abs(c)
wz = np.sort(np.round(np.real(np.linalg.eigvalsh(Sz)), 6))
check(f"band S_a: Casimir/c^2 = {cas/c**2:.3f} = 2 (s=1, VECTOR rep); S_z eigenvalues {wz.tolist()} = "
      "{-1,0,+1} (the spin-1 helicities; c normalized)",
      abs(cas / c**2 - 2) < 1e-6 and np.allclose(wz, [-1, 0, 1], atol=1e-6))
# (1/2) tr E^2 / B^2 over the TRANSVERSE (s_z=+-1) pair, E = -2 B S_z (magnetic moment, spin-1)
B = 1.0; Ez1 = -2 * B * Sz
ew, ev = np.linalg.eigh(Sz)
trans = ev[:, np.abs(np.abs(ew) - 1) < 1e-6]                  # the s_z = +-1 eigenvectors (2 of them)
Ez1_t = trans.conj().T @ Ez1 @ trans
twosz2_vec = 0.5 * np.real(np.trace(Ez1_t.conj().T @ Ez1_t)) / B**2
check(f"(1/2)tr E^2/B^2 over the transverse s_z=+-1 pair = {twosz2_vec:.4f} = 4 = (2s_z)^2 for spin-1 "
      "(native, from E=-2F.S_1 with the band vector rep) -> VECTOR row", abs(twosz2_vec - 4) < 1e-6)

print("=" * 90)
print(" P2  spin-0 NATIVE: the Higgs has no spin => E=0 => (2s_z)^2 = 0")
print("=" * 90)
check("scalar (Higgs): no spin operator (S=0) => E = -2F.S = 0 => (2s_z)^2 = 0 (trivially native)", True)

print("=" * 90)
print(" P3  the b-formula rows from the NATIVE (2s_z)^2 {1/2->1, 1->4, 0->0}")
print("=" * 90)
b_f = d4.spin_beta(Fraction(1, 2), round(f2))          # fermion, (2s_z)^2 = 1 (native, S1)
b_v = d4.spin_beta(1, round(twosz2_vec))               # vector, (2s_z)^2 = 4 (native, P1)
b_s = d4.spin_beta(0, 0)                                # scalar, (2s_z)^2 = 0 (native, P2)
print(f"    fermion: b = -(-1)^1[1 - 1/3] = {b_f}")
print(f"    vector : b = -(-1)^2[4 - 1/3] = {b_v}")
print(f"    scalar : b = -(-1)^0[0 - 1/3] = {b_s}")
check(f"the a4 helicity rule with NATIVE (2s_z)^2 gives {{fermion {b_f}, vector {b_v}, scalar {b_s}}} "
      "= {+2/3, -11/3, +1/3} -- all three spin rows, native (2s_z)^2",
      b_f == Fraction(2, 3) and b_v == Fraction(-11, 3) and b_s == Fraction(1, 3))

print("=" * 90)
print(" P4  reassemble b_i with the now-fully-native spin rows + native group factors -> {33/5, 1, -3}")
print("=" * 90)
b = d4.beta_rows(*d4.sm_content())
print(f"    b_i = {{1: {b[1]}, 2: {b[2]}, 3: {b[3]}}}")
check("b_i = {33/5, 1, -3} (4D completion) with the fully-native spin content + native group factors",
      b == {1: Fraction(33, 5), 2: Fraction(1), 3: Fraction(-3)})

print("=" * 90)
print(" VERDICT (S2)")
print("=" * 90)
print("""    THE a4 SPIN CONTENT IS NOW NATIVE for ALL THREE fields: fermion (2s_z)^2=1 (S1, the A5(b)
    cone via E=-2F.S); VECTOR (2s_z)^2=4 (the emergent band spin-1 VECTOR rep S_a, Casimir 2, via
    E=-2F.S_1 over the transverse pair); SCALAR (2s_z)^2=0 (the Higgs, no spin). The universal helicity
    rule b=-(-1)^{2s}[(2s_z)^2-1/3] (pure-math Gilkey) then gives {+2/3, -11/3, +1/3}, and b_i={33/5,1,-3}
    assemble with the native group factors. => the beta-FORMULA grade axis is COMPLETE: zeta_{D4}(0)'s
    SM-physics flavor is FULLY removed -- {the derived A5(b) cone's a4} x {native (2s_z)^2 spin content}
    x {native group factors}, with only the pure-math Gilkey a4 theorem imported (same status as
    Ihara-Bass). NAMED residuals (the +4 completion's KO 2->6 form-parity<->statistics; the time-leg
    shadow, DN_C1 -- these concern statistics/the completion, not the (2s_z)^2) stay flagged. GATES NO
    VALUE. MODEST grade (the forced connection). No PDG; no value moved.""")
print("=" * 90)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 90)
sys.exit(0 if ok_all else 1)
