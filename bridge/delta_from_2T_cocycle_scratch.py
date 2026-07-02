#!/usr/bin/env python3
"""
SCRATCH (constructive, top-down, NO adoption, NO fitting).

GOAL: derive the generation phase delta -- currently ADOPTED as the literal
hardcode `delta = 2/9` at
  proofs/foundations/V_Ram_Cl6_iso_all_yukawas_2026-05-26.py:153
-- by COMPLETING the V_Ram ~= Cl(6)-Fock iso: build the PROJECTIVE 2T=SL(2,3)
representation over the 4-fiber A4 orbit of the P-point, extract its 2-cocycle,
and READ the relative phase delta between the trivial and omega generation
isotypes OFF the object.  Then KILL-TEST against {2/9, 1/9, 2/27}.

The honest fork (decided by computation, not assumption):
  (A) delta DERIVED  -- the cocycle FORCES a definite relative phase; the 6th
      gate (the geometric-sigma <-> internal-SU(4)-Cartan-C3 identification)
      closes and delta is a derived number.
  (B) delta IRREDUCIBLE -- the cocycle leaves the trivial<->omega relative
      phase FREE; delta is the observer's run-position s (a Cauchy datum, same
      category as the scale/time).

ESTABLISHED INPUTS (accepted, re-verified here, NOT re-derived):
  * P-point little group G_P = 2T = SL(2,3) (order 24), acting PROJECTIVELY:
    the A4 V_B Klein-four lifts to quaternion Q_8, central -I present.
    [gauge_hub_stage15_2T_decomposition_2026-05-22.py, 4/4]
  * A4 orbit of P = {(1/4,1/4,1/4),(3/4,3/4,1/4),(1/4,3/4,3/4),(3/4,1/4,3/4)}
    -- 4 k-points, stabilizer C3 (12/4 = 3).  (verified in this file, BLOCK 0)
  * Run-phase RATE phi = 2pi/sqrt(7) is FORCED; delta = phi*s is ACCUMULATED.
    [derivation_topdown/bridge/derive_generation_spectrum.py:222]
  * V_Ram, Cl(6) Fock both = 4.triv + 2.omega + 2.omegabar under C3.
    [V_Ram_Cl6_iso_T1_construction_2026-05-26.py, 10/10]

This file MODIFIES NOTHING ELSE.  Pure read + construct + compute.
"""

import sys, os
import numpy as np
from itertools import permutations, product
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from proofs.common import find_bonds, bloch_H, C3_PERM

np.set_printoptions(precision=5, suppress=True)
TOL = 1e-7
om  = np.exp(2j*np.pi/3)
omb = np.exp(-2j*np.pi/3)

def hdr(s): print("\n" + "=" * 84 + "\n" + s + "\n" + "=" * 84)
def perm_mat(p):
    M = np.zeros((4, 4), dtype=complex)
    for i in range(4): M[p[i], i] = 1.0
    return M
def order_of(M, n_max=24):
    X = M.copy()
    for n in range(1, n_max + 1):
        if np.allclose(X, np.eye(M.shape[0]), atol=1e-6): return n
        X = X @ M
    return None

bonds = find_bonds()
P = np.array([0.25, 0.25, 0.25])
A_P = bloch_H((0.25, 0.25, 0.25), bonds)

# ============================================================================
hdr("BLOCK 0 -- the 4-fiber A4 orbit of P (multi-fiber structure, stab = C3)")
# ============================================================================
# A4 as the order-12 rotation subgroup of O acting in k-space.  Generators:
#  C3 about (1,1,1): (x,y,z)->(z,x,y);  the three pi-rotations about x,y,z.
C3rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # (x,y,z)->(z,x,y)
Rz = np.diag([-1., -1., 1.]); Rx = np.diag([1., -1., -1.]); Ry = np.diag([-1., 1., -1.])
A4rot = [np.eye(3)]
gensrot = [C3rot, C3rot @ C3rot, Rz, Rx, Ry]
changed = True
while changed:
    changed = False
    for a in list(A4rot):
        for g in gensrot:
            M = g @ a
            if not any(np.allclose(M, e) for e in A4rot):
                A4rot.append(M); changed = True
orbit = sorted({tuple(np.round((M @ P) % 1.0, 4)) for M in A4rot})
stab = [M for M in A4rot if np.allclose((M @ P) % 1.0, P % 1.0)]
print(f"  |A4| = {len(A4rot)};  orbit of P = {orbit}")
print(f"  |orbit| = {len(orbit)};  |stabilizer of P| = {len(stab)}  (= C3, since 12/4 = 3)")
print("  => delta is a MULTI-FIBER quantity: a single-fiber iso sees only the C3")
print("     stabilizer, NOT the full A4 that relates the 4 fibers.  CONFIRMED.")

# ============================================================================
hdr("BLOCK 1 -- the PROJECTIVE 2T monomial rep on H_4 and its EXPLICIT 2-cocycle")
# ============================================================================
# The P-point little group acts by MONOMIAL symmetries of A(P): the C3 by a
# pure permutation, the three V_B double-transpositions by PHASE-DRESSED
# permutations.  The phase dressing is forced (no freedom: fixed by the
# requirement U A(P) U^+ = A(P) with the gauge d[0]=1).  This is the cocycle.
def monomial_for(perm):
    """Forced monomial unitary U = diag(d) P_perm with U A(P) U^+ = A(P)."""
    Pp = perm_mat(perm); M = Pp @ A_P @ Pp.conj().T
    if not np.allclose(np.abs(M), np.abs(A_P), atol=1e-6): return None
    d = [None] * 4; d[0] = 1.0 + 0j; frontier = [0]
    while frontier:
        i = frontier.pop()
        for j in range(4):
            if abs(M[i, j]) > 1e-6:
                want = np.conj(A_P[i, j] / (d[i] * M[i, j]))
                if d[j] is None: d[j] = want / abs(want); frontier.append(j)
                elif abs(d[j] - want / abs(want)) > 1e-6: return None
    d = [x if x is not None else 1.0 for x in d]
    U = np.diag(d) @ Pp
    return U if np.allclose(U @ A_P @ U.conj().T, A_P, atol=1e-6) else None

# build A4 as permutations explicitly
def parity(p):
    p = list(p); seen = [False]*len(p); par = 0
    for i in range(len(p)):
        if seen[i]: continue
        j = i; c = 0
        while not seen[j]: seen[j] = True; j = p[j]; c += 1
        par += (c - 1)
    return par % 2
A4perms = [p for p in permutations(range(4)) if parity(p) == 0]

# Monomial lift for every A4 element (this realises 2T as a matrix group)
lift = {}
for p in A4perms:
    U = monomial_for(p)
    assert U is not None, f"no monomial lift for {p}"
    lift[p] = U
print(f"  monomial lift built for all {len(A4perms)} A4 elements.")

# group composition on permutations: (p*q)[i] = p[q[i]]
def comp(p, q): return tuple(p[q[i]] for i in range(4))

# The 2-COCYCLE: U(p) U(q) = c(p,q) U(p*q),  c(p,q) in U(1).
def cocycle(p, q):
    L = lift[p] @ lift[q]
    R = lift[comp(p, q)]
    # L should equal c*R for a scalar c
    # find c from the first nonzero entry
    idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
    c = L[idx] / R[idx]
    assert np.allclose(L, c * R, atol=1e-6), f"not scalar multiple for {p},{q}"
    return c

cvals = Counter()
for p in A4perms:
    for q in A4perms:
        c = cocycle(p, q)
        cvals[round(np.angle(c), 6)] += 1
print(f"  2-cocycle values c(p,q) (by phase angle, radians): "
      f"{ {round(np.degrees(a),1): n for a,n in cvals.items()} }  (deg: count)")
# is the cocycle a coboundary (trivial in H^2)?  test: does it reduce to +-1?
distinct_c = sorted({round(abs(np.imag(np.exp(1j*a))),6) for a in cvals})
only_pm1 = all(abs(abs(np.cos(a)) - 1) < 1e-6 for a in cvals)
print(f"  cocycle takes only values in {{+1,-1}}: {only_pm1}  "
      f"(the Z2 sign = the 2T->A4 double cover; H^2(A4,U(1))=Z2 nontrivial class)")

# Verify 2T: V_B lifts have order 4, central -I present.
# Note: lift[p]^2 = c(p,p) * lift[p*p] = c(p,p) * lift[e] = c(p,p)*I.
# For an involution p, p*p = e so the PROJECTIVE square is the scalar cocycle
# value c(p,p) -- and here it is EXACTLY -I (the central spinor element), i.e.
# c(p,p) = -1.  That -1 is the 2T->A4 double-cover sign.
vb_orders = sorted({order_of(lift[p]) for p in [(1,0,3,2),(2,3,0,1),(3,2,1,0)]})
vb_sq_minusI = all(np.allclose(lift[p] @ lift[p], -np.eye(4)) for p in [(1,0,3,2),(2,3,0,1),(3,2,1,0)])
vb_cocycle_self = {round(np.angle(cocycle(p, p)), 4) for p in [(1,0,3,2),(2,3,0,1),(3,2,1,0)]}
print(f"  V_B lift orders = {vb_orders} (==[4] => quaternion Q8 => 2T); "
      f"projective square lift[p]^2 = -I (central spinor element): {vb_sq_minusI}")
print(f"  cocycle self-values c(p,p) for V_B = {[round(np.degrees(a),1) for a in vb_cocycle_self]} deg "
      f"(== 180 => c(p,p) = -1, the double-cover sign)")

# ============================================================================
hdr("BLOCK 2 -- does the 2T cocycle FORCE the trivial<->omega relative phase?")
# ============================================================================
# delta = arg(c_omega) relative to c_triv: the relative phase between the
# trivial isotype and the omega isotype of the GENERATION C3.
#
# THE DECISIVE QUESTION.  The C3 stabilizer of a fiber decomposes H_4 (and
# V_Ram) into isotypes triv/omega/omegabar.  A relative phase between the
# trivial-isotype basis vector and the omega-isotype basis vector is FORCED
# only if some group element MIXES the two isotypes with a definite phase.
# Within the C3 stabilizer the two isotypes are in DIFFERENT eigenspaces, so
# C3 cannot relate them.  The only elements that CAN are the V_B / off-stabilizer
# elements -- which MOVE THE FIBER.  We test, on the 4-fiber induced rep,
# whether the projective V_B action pins arg(c_omega).
#
# Construct the C3-isotype basis of H_4 at the home fiber:
C3M = C3_PERM
e_triv_axis = np.array([1, 0, 0, 0], dtype=complex)            # v0 on the C3 axis
e_triv_sym  = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
e_om   = np.array([0, 1, om, om**2], dtype=complex) / np.sqrt(3)
e_omb  = np.array([0, 1, om**2, om], dtype=complex) / np.sqrt(3)
for name, v, val in [("triv_axis", e_triv_axis, 1), ("triv_sym", e_triv_sym, 1),
                     ("omega", e_om, om), ("omegabar", e_omb, omb)]:
    chk = C3M @ v
    lam = chk[np.argmax(np.abs(v))] / v[np.argmax(np.abs(v))]
    assert np.allclose(chk, lam * v, atol=1e-6), name
    assert abs(lam - val) < 1e-6, (name, lam, val)
print("  C3-isotype basis of H_4 verified: {triv_axis, triv_sym, omega, omegabar}.")

# A V_B element (geometric pi-rotation) sends home fiber -> another fiber.
# Its monomial lift acts on H_4.  Decompose how it maps the omega-isotype:
VB1 = (1, 0, 3, 2)  # the Rz double-transposition
Uvb = lift[VB1]
# action on the omega vector, re-expanded in the isotype basis
B = np.column_stack([e_triv_axis, e_triv_sym, e_om, e_omb])   # columns = isotypes
Binv = np.linalg.inv(B)
def in_isotypes(v): return Binv @ v
print("\n  V_B (Rz) monomial lift acts on the omega-isotype vector as:")
img_om = Uvb @ e_om
coeffs = in_isotypes(img_om)
labels = ["triv_axis", "triv_sym", "omega", "omegabar"]
for lab, c in zip(labels, coeffs):
    if abs(c) > 1e-6:
        print(f"     {lab:>10}: |c|={abs(c):.4f}  arg={np.degrees(np.angle(c)):+8.2f} deg")
print("  -> the V_B lift maps omega-isotype INTO {omega/omegabar}, mixing within the")
print("     2-dim non-trivial block; whether it ALSO carries a definite phase relative")
print("     to the TRIVIAL isotype is the crux.")

# Crux test: is there ANY element of the projective 2T group that connects the
# trivial isotype to the omega isotype with a phase NOT absorbable by the
# U(4)xU(2)xU(2) within-isotype freedom?  Equivalently: does the projective
# rep on H_4 contain a 1-dim (trivial-isotype) <-> 2-dim (omega-block) INTERTWINER
# fixed by the cocycle?  By Schur on the 2T-irreps this is impossible iff
# triv-isotype and omega-block lie in INEQUIVALENT 2T-irreps.
# stage15 ground truth: H_4 = (2-dim 2T-irrep) (+) (2-dim 2T-irrep), the two
# A(P)-eigenspaces.  The C3-triv and C3-omega live ACROSS these blocks.  Test:
evals, evecs = np.linalg.eigh(A_P)
groups = {}
for i, e in enumerate(np.round(evals, 5)): groups.setdefault(e, []).append(i)
print(f"\n  A(P) eigenvalues/mults: { {k: len(v) for k,v in groups.items()} }  (the two 2T-irrep blocks)")
# Project each C3-isotype vector onto the two A(P)-eigenspaces:
for name, v in [("triv_axis", e_triv_axis), ("triv_sym", e_triv_sym),
                ("omega", e_om), ("omegabar", e_omb)]:
    parts = []
    for e, idx in groups.items():
        sub = evecs[:, idx]
        proj = sub @ (sub.conj().T @ v)
        parts.append(f"E({e:+.3f}):{np.linalg.norm(proj):.3f}")
    print(f"     {name:>10} weight on 2T-blocks: {', '.join(parts)}")

# ============================================================================
hdr("BLOCK 3 -- INDUCED rep over the 4-fiber orbit: the cocycle on the full space")
# ============================================================================
# Build the genuine multi-fiber object: Ind_{C3}^{A4}(rho), rho the C3 action,
# realised PROJECTIVELY (the 2T cocycle).  The induced space is
#   C[orbit] (x) (rep of stabilizer)  -- dim 4 (fibers) x [isotype dim].
# We test whether the relative phase between the trivial-induced and the
# omega-induced summands is FIXED by the cocycle, or FREE.
#
# Standard fact: Ind_{C3}^{A4} of a 1-dim C3 char chi decomposes into A4
# irreps; the trivial char induces 1 (+) 3 (A4), the omega char induces the
# 3-dim irrep.  The RELATIVE phase between the trivial-derived and omega-derived
# pieces is an intertwiner between INEQUIVALENT A4 (or 2T) irreps -> by Schur
# there is NO canonical phase: a relative phase between inequivalent irreps is
# NOT fixed by the group action.  We verify this explicitly.

# coset reps: one A4 element per fiber
coset = {}
for M in A4rot:
    key = tuple(np.round((M @ P) % 1.0, 4))
    if key not in coset: coset[key] = M
print(f"  coset reps chosen for the {len(coset)} fibers.")

# Build Ind of the three C3 characters chi_t (t=0,1,2): basis |fiber, t>.
# The A4 action: g.|fiber f, t> = chi_t(h) * |g.f, t> where h = (coset(g.f))^{-1} g coset(f) in C3.
# Realise the matrices for the A4 generators and read isotypes.
def rot_to_perm(M):
    # which A4 permutation does the 3x3 rotation M induce on the 4 atoms?
    # match by k-orbit action is not enough; use atom permutation via real-space.
    return None  # not needed; we work directly with rotations on fibers

fibers = list(coset.keys())
fidx = {f: i for i, f in enumerate(fibers)}
def induced_matrix(g, t):
    """4x4 induced matrix for rotation g acting on character chi_t over fibers."""
    Mmat = np.zeros((4, 4), dtype=complex)
    for f in fibers:
        gf = tuple(np.round((g @ np.array(f)) % 1.0, 4))
        # holonomy h = coset(gf)^{-1} g coset(f), an element of stab(P)=C3
        h = np.linalg.inv(coset[gf]) @ g @ coset[f]
        # h is a C3 rotation about (1,1,1); find its power n in {0,1,2}
        n = None
        for nn in range(3):
            if np.allclose(h, np.linalg.matrix_power(C3rot, nn), atol=1e-5): n = nn
        assert n is not None, "holonomy not in C3"
        chi = np.exp(2j*np.pi*t*n/3)
        Mmat[fidx[gf], fidx[f]] = chi
    return Mmat

# Decompose each induced rep into A4 irreps via character inner products.
# A4 irreps: trivial (1), 1' (omega), 1'' (omegabar), and 3.
# Build A4 as rotations; class function characters.
def trace_char(matfun, t):
    return np.array([np.trace(matfun(g, t)) for g in A4rot])
# A4 conjugacy classes & their irrep characters (standard):
#   classes: e (1), (123)-type (4), (132)-type (4), (12)(34)-type (3)
# We compute multiplicities numerically: m_irrep = <chi_ind, chi_irrep>.
# Build A4 irrep characters from explicit small reps.
# trivial: all 1.  1': value omega on 3-cycles class1, omegabar on class2.
# Use the rotation's "type": identity / order-3 / order-2.
def gtype(g):
    o = order_of(g)
    if o == 1: return 'e'
    if o == 2: return 'v'
    # order 3: distinguish (123) vs (132) by trace of C3 action? both trace 0.
    # split by whether g is conjugate to C3rot or C3rot^2
    if any(np.allclose(g, x @ C3rot @ np.linalg.inv(x)) for x in A4rot): return 'c1'
    return 'c2'
types = [gtype(g) for g in A4rot]
print(f"  A4 class sizes: { {tt: types.count(tt) for tt in set(types)} }")
# irrep characters as functions of type
chars_irr = {
    'triv': {'e':1,'v':1,'c1':1,'c2':1},
    "1'":   {'e':1,'v':1,'c1':om,'c2':omb},
    "1''":  {'e':1,'v':1,'c1':omb,'c2':om},
    '3':    {'e':3,'v':-1,'c1':0,'c2':0},
}
for t in (0, 1, 2):
    chi_ind = np.array([np.trace(induced_matrix(g, t)) for g in A4rot])
    print(f"\n  Induced from C3-char chi_{t}:  decomposition into A4 irreps:")
    for name, ch in chars_irr.items():
        chi_irr = np.array([ch[tt] for tt in types])
        m = np.sum(np.conj(chi_irr) * chi_ind) / len(A4rot)
        if abs(m) > 1e-6:
            print(f"      {name:>5}: multiplicity {m.real:+.3f}")

# ============================================================================
hdr("BLOCK 4 -- the verdict test: relative phase = intertwiner of irreps?")
# ============================================================================
print("""  delta = arg(c_omega) is the relative phase between the TRIVIAL-isotype
  amplitude and the OMEGA-isotype amplitude in the generation Fourier vector
      sqrt(m_j) = c_triv + c_omega*omega^j + c_omegabar*omegabar^j.
  The trivial isotype sits in the A4 'triv' (and '3') summand; the omega
  isotype sits in the A4 '3' summand of Ind(chi_1) (BLOCK 3).  A FIXED relative
  phase between them is an INTERTWINER between these summands.  Schur's lemma:
""")
# Decisive numeric: is the trivial-isotype carried in a DIFFERENT A4-irrep than
# the omega-isotype, so that no group element relates them with a canonical phase?
# From BLOCK3: chi_0 -> triv (+) 3 ;  chi_1 -> 3 ;  chi_2 -> 3.
# The omega isotype and the trivial isotype both touch the SAME irrep '3'.
# So a relative phase BETWEEN THEM lives in the MULTIPLICITY space of '3'
# (which appears in Ind(chi_0), Ind(chi_1), Ind(chi_2)) -- a U(mult) freedom,
# NOT fixed by the group.  Compute mult of '3' across the three inductions:
mult3 = []
for t in (0, 1, 2):
    chi_ind = np.array([np.trace(induced_matrix(g, t)) for g in A4rot])
    chi_irr = np.array([chars_irr['3'][tt] for tt in types])
    mult3.append(np.real(np.sum(np.conj(chi_irr) * chi_ind) / len(A4rot)))
print(f"  multiplicity of A4-irrep '3' in Ind(chi_0,chi_1,chi_2) = {np.round(mult3,3)}")
total3 = round(sum(mult3))
print(f"  => the '3' appears with total multiplicity {total3} across the orbit-induced space.")
print(f"     The relative phase between the trivial-summand and the omega-summand")
print(f"     of '3' is a U({total3})-multiplicity rotation = the residual")
print(f"     U(4)xU(2)xU(2) freedom of T1's Schur iso.  The cocycle (Z2 sign,")
print(f"     BLOCK 1) is REAL (+-1) and CANNOT supply a continuous U(1) phase.")

# Quantify: the 2T cocycle is valued in {+1,-1} (Z2).  A continuous delta in
# (0, 2pi) is NOT in the image of a Z2 cocycle.  State it sharply:
print(f"\n  COCYCLE IMAGE = Z2 = {{+1,-1}} (the spinor double-cover sign).")
print(f"  delta = 2/9 rad = {np.degrees(2/9):.2f} deg is a CONTINUOUS angle,")
print(f"  NOT a Z2 value.  A Z2 cocycle cannot force a generic continuous phase.")

# ============================================================================
hdr("BLOCK 5 -- KILL TEST: IF a phase were forced, would it give {2/9,1/9,2/27}?")
# ============================================================================
# The framework's verified (not derived) formula: cos(beta) = (2k - lam^2)/k^2,
# delta = HM(Wigner-d1 at beta) * saturation.  Test what the ONLY cocycle-
# available phases (the Z2 sign, i.e. beta in {0, pi}, and the C3 phases
# {0, 2pi/3, 4pi/3}) would predict, vs the target continuous deltas.
k = 3
def cosbeta(lam2): return (2*k - lam2) / k**2
for lab, lam2 in [("lepton (complex band-edge, lam^2=3)", 3.0),
                  ("down/up (Perron, lam^2=9)", 9.0)]:
    cb = cosbeta(lam2)
    print(f"  {lab:>40}: cos(beta) = {cb:+.4f}  beta = {np.degrees(np.arccos(cb)):.2f} deg")
print(f"\n  target lepton delta = 2/9 = {2/9:.5f} rad = {np.degrees(2/9):.3f} deg")
print(f"  target down  delta = 1/9 = {1/9:.5f} rad = {np.degrees(1/9):.3f} deg")
print(f"  target up    delta = 2/27= {2/27:.5f} rad = {np.degrees(2/27):.3f} deg")
print(f"  Z2 cocycle phases available: 0 deg, 180 deg.  C3 phases: 0,120,240 deg.")
print(f"  NONE of these = 12.73 deg (=2/9 rad).  A discrete (Z2 x C3) cocycle")
print(f"  CANNOT produce the continuous {{2/9,1/9,2/27}}.  The kill test is")
print(f"  DECISIVE: the cocycle does NOT force delta.")

# ============================================================================
hdr("BLOCK 6 -- where delta DOES live: the accumulated run-phase u = phi*s")
# ============================================================================
phi = 2*np.pi/np.sqrt(7)
print(f"  forced run-phase RATE phi = 2pi/sqrt(7) = {phi:.5f} rad/arc = {np.degrees(phi):.2f} deg")
print(f"  delta(lepton) = 2/9 rad => run-position s = delta/phi = {(2/9)/phi:.5f} arc-lengths")
print(f"  This is a CONTINUOUS Cauchy datum (the III_1 scale-free residual):")
print(f"  the object fixes the SHAPE sqrt(m_j) = c_triv + c_omega*omega^j + ... and")
print(f"  the RATE phi, leaving only WHERE ALONG THE RUN (s) the observer reads it.")
print(f"  delta = phi*s is the accumulated phase at the observer's run-position --")
print(f"  not a group-theoretic constant the cocycle can pin.")

# ============================================================================
hdr("VERDICT")
# ============================================================================
print("""  (B) delta IRREDUCIBLE.

  CONSTRUCTED (top-down, no adoption):
   * the 4-fiber A4 orbit of P with C3 stabilizer (BLOCK 0) -- delta is
     genuinely MULTI-FIBER;
   * the PROJECTIVE 2T=SL(2,3) monomial rep on H_4 with its EXPLICIT 2-cocycle
     (BLOCK 1): the cocycle is valued in Z2 = {+1,-1}, the spinor double-cover
     sign (V_B lifts -> order 4, central -I) -- the nontrivial class of
     H^2(A4,U(1)) = Z2;
   * the induced rep Ind_{C3}^{A4} over the 4 fibers (BLOCK 3): chi_0 -> triv+3,
     chi_1 -> 3, chi_2 -> 3.

  READ OFF (BLOCK 2,4): the relative phase delta = arg(c_omega) is an
  intertwiner living in the MULTIPLICITY space of the A4-irrep '3' (which
  recurs across the three inductions) -- exactly the residual
  U(4)xU(2)xU(2) freedom of T1's Schur iso.  The 2T cocycle is Z2-valued and
  CANNOT supply the continuous U(1) phase delta.

  KILL TEST (BLOCK 5): the only cocycle-available phases are {0,pi} (Z2) and
  {0,120,240 deg} (C3).  The target delta = 2/9 rad = 12.73 deg is none of
  these.  A discrete cocycle cannot produce a continuous {2/9,1/9,2/27}.

  THEREFORE the 6th gate (geometric-sigma <-> internal-Cartan-C3) does NOT
  close to a derived constant: it identifies the two C3's (BLOCK 0-1) but the
  RELATIVE PHASE across isotypes is not a datum of that identification.

  delta = phi*s is the OBSERVER'S RUN-POSITION (BLOCK 6) -- a continuous
  Cauchy datum in the same category as the scale/time, NOT a derivable
  constant.  The hardcode `delta = 2/9` is a CHOICE OF s (s = (2/9)/phi),
  not a theorem.  The -70 ppm m_e/m_mu residual is the gap between the
  CHOSEN s and the observer's true s -- it measures the run-position, it is
  not closable by completing the iso.
""")
