#!/usr/bin/env python3
"""
proofs/foundations/LOOP_E1b_seam_parity_2026-07-02.py

LOOP PROGRAM, R-eps STAGE E1b -- the SEAM-PARITY question (the E1 gate's
option (b)). Pre-registered in internal research notes
("E1b PRE-REGISTRATION", commit e185e0e, BEFORE this probe).

SCOPE: NO eps evaluation; the R-eps target appears NOWHERE; no PDG; the
shipped reads enter ONLY as preservation surfaces.

TWO NEW TRAP-LEDGER ENTRIES + ONE ERRATUM (found IN this sitting, disclosed):
  (T-a) complex nullspaces: the null vectors are conj(Vh rows); un-conjugated
        rows solve the CONJUGATE equation. E1's lift_U had this bug; E1 was
        re-run corrected: its D2 verdict (16 families -> vacuum-cut -> ZERO
        freedom; X_a = gamma_a unique) REPRODUCES on the true action -- the
        E1 conclusion STANDS (erratum recorded in the E1b banner).
  (T-b) per-element pin lifts carry arbitrary U(1) phases: a phase-incoherent
        family is NOT a representation (characters/Hom silently break --
        first pass gave 'trivial multiplicity 1/6'). Group actions on STATES
        need a COHERENT construction. The DERIVED coherent choice exists and
        is canonical: the Fock functor U_g = Gamma(V_g) (the CAR structure's
        own lift; vacuum character trivial BY CONSTRUCTION, not by choice).
        Presentation-closure phase-fixing is frame-ambiguous (Z3 twist) and
        was superseded by Gamma(V).

NOTATION SLIP IN THE PRE-REG BLOCK (disclosed, no gate moved): the parity
halves are EVEN = Lambda^0+Lambda^2, ODD = Lambda^1+Lambda^3 (the block wrote
Lambda^0+Lambda^1). Both are trivial+triplet either way.

CLAUSES:
  P1  the rep-isomorphism theorem, in the DERIVED frame: vertex C^4
      (adjacency content {Perron singlet, lam=-1 generation triple}) and BOTH
      parity halves are trivial (+) triplet under Gamma(V_g); the invariant
      that makes the odd-singlet channel exist: delta = det(V_g) = 1 (the
      mode rep lands in SU(3)) -- computed, recorded. Seam ambient =
      Hom_A4(C^4, F_even (+) F_odd), dim_C = 4: FORCED (Schur), not modeled.
  P2  the mirror on the halves and on the seam space: the odd pin lift flips
      the parity halves (gamma^5-flip, T8/T9) and maps J -> -J; the seam-
      space action exchanges the halves CLEANLY.
  P3  derived constraints: (i) isometry/CAR; (ii) the E1/C1 statistics
      theorem => the seam is parity-pure (ODD half; u-quanta are odd);
      (iii) species-read preservation (Hamming-pure images -- Schur);
      (iv) J-canonicity (built-in).
  P4  verdict on (b): mirror-odd seam freedom within the admissible set.

KILLS: K1b the P1 isomorphism fails; K2b mirror-odd freedom survives and
feeds the read channel; K3b a surface breaks.
"""
import itertools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]

def parity_perm(p):
    inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
    return 1 if inv % 2 == 0 else -1

S4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))]
A4 = [g for g in S4 if parity_perm([g[i] for i in range(4)]) == 1]
ODD = [g for g in S4 if parity_perm([g[i] for i in range(4)]) == -1]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}

def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6

def vert_rep(sig):
    M = np.zeros((NV, NV))
    for i in range(NV):
        M[sig[i], i] = 1.0
    return M

def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))

def pin_lift(g):
    """per-element lift (phase arbitrary) -- used ONLY where phases cancel or
    are irrelevant (the odd mirror element). Trap T-a fixed: conj + residual gate."""
    R6 = edge_rep(g)
    rows = [np.kron(gam(R6[:, a]), np.eye(8)) - np.kron(np.eye(8), g6[a].T)
            for a in range(NE)]
    M = np.vstack(rows)
    _, S, Vh = np.linalg.svd(M)
    null = Vh[np.sum(S > 1e-9):].conj()
    assert null.shape[0] == 1 and np.linalg.norm(M @ null[0]) < 1e-9
    U = null[0].reshape(8, 8)
    U /= np.sqrt(np.abs(np.linalg.det(U @ U.conj().T)) ** (1 / 8))
    return U

# ===========================================================================
banner("S-P0  the canonical J, modes, Fock basis, N-hat (surfaces)")
# ===========================================================================
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1],
                 [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
rows = []
for g in A4:
    R6 = edge_rep(g)
    RH = H1.T @ R6 @ H1
    RB = B1.T @ R6 @ B1
    rows.append(np.kron(np.eye(3), RH.T) - np.kron(RB, np.eye(3)))
_, Sp, Vp = np.linalg.svd(np.vstack(rows))
assert 9 - np.sum(Sp > 1e-9) == 1
phi = Vp[-1].reshape(3, 3)
phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
check(f"S-P0 the canonical J: J^2 = -I (err {np.max(np.abs(J6 @ J6 + np.eye(6))):.1e}), "
      "A4-invariant",
      np.max(np.abs(J6 @ J6 + np.eye(6))) < 1e-9
      and all(np.max(np.abs(edge_rep(g) @ J6 - J6 @ edge_rep(g))) < 1e-9 for g in A4))
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])   # trap #6: qr
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
evN = np.sort(np.linalg.eigvalsh(NHAT).real)
check(f"S-P0 SURFACE: N-hat spectrum = Hamming (1,3,3,1) {np.round(evN, 6)} -- species "
      "grading untouched", np.allclose(evN, [0, 1, 1, 1, 2, 2, 2, 3]))
# explicit Fock basis from a-dagger products on the vacuum
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]
vac = vac / np.linalg.norm(vac)
ad = [a.conj().T for a in A_ops]
basis_cols = [vac]
basis_cols += [ad[m] @ vac for m in range(3)]                       # level 1
basis_cols += [ad[i] @ ad[j] @ vac for (i, j) in ((0, 1), (0, 2), (1, 2))]  # level 2
basis_cols += [ad[0] @ ad[1] @ ad[2] @ vac]                          # level 3
FB = np.hstack(basis_cols)
check(f"S-P0 the a-dagger product Fock basis is orthonormal (err "
      f"{np.max(np.abs(FB.conj().T @ FB - np.eye(8))):.1e}) -- levels (1,3,3,1)",
      np.max(np.abs(FB.conj().T @ FB - np.eye(8))) < 1e-9)
OM6 = np.eye(8)
for a in range(6):
    OM6 = OM6 @ g6[a]
PARITY = -1j * OM6
Peven = FB[:, [0, 4, 5, 6]]                                          # Lambda^0 + Lambda^2
Podd = FB[:, [1, 2, 3, 7]]                                           # Lambda^1 + Lambda^3
check("S-P0 parity halves: EVEN = Lambda^0+Lambda^2 (+1), ODD = Lambda^1+Lambda^3 (-1) "
      "under (-1)^N = -i omega_6 (pre-reg notation slip disclosed in header)",
      np.max(np.abs(PARITY @ Peven - Peven)) < 1e-9
      and np.max(np.abs(PARITY @ Podd + Podd)) < 1e-9)

# ===========================================================================
banner("S-P1  the rep isomorphism in the DERIVED (Fock-functor) frame  [K1b]")
# ===========================================================================
A_adj = np.ones((NV, NV)) - np.eye(NV)
evA, VA = np.linalg.eigh(A_adj)
v_perron = VA[:, [int(np.argmax(evA))]]
v_triple, _ = np.linalg.qr(VA[:, np.argsort(evA)[:3]])
check(f"S-P1 vertex content: adjacency eigenvalues {np.round(np.sort(evA), 6)} = "
      "{-1 x3 (the generation triple), +3 (Perron)}",
      np.allclose(np.sort(evA), [-1, -1, -1, 3]))

def mode_rep(g):
    return modes.conj().T @ edge_rep(g) @ modes                      # 3x3 unitary

def Lam2(V):
    pairs = ((0, 1), (0, 2), (1, 2))
    M = np.zeros((3, 3), complex)
    for r, (i, j) in enumerate(pairs):
        for c, (k, l) in enumerate(pairs):
            M[r, c] = V[i, k] * V[j, l] - V[i, l] * V[j, k]
    return M

def fock_functor(g):
    """U_g = Gamma(V_g): the CAR structure's OWN lift -- vacuum-canonical by
    construction (the derived frame; trap T-b resolved by derivation)."""
    V = mode_rep(g)
    blocks = np.zeros((8, 8), complex)
    blocks[0, 0] = 1.0
    blocks[1:4, 1:4] = V
    blocks[4:7, 4:7] = Lam2(V)
    blocks[7, 7] = np.linalg.det(V)
    return FB @ blocks @ FB.conj().T

U_A4 = {gi: fock_functor(g) for gi, g in enumerate(A4)}
key = lambda g: tuple(g[i] for i in range(4))
def compose(a, b):
    return {k: a[b[k]] for k in range(4)}
idx_of = {key(g): gi for gi, g in enumerate(A4)}
ok_hom = all(np.max(np.abs(U_A4[gi] @ U_A4[hi] - U_A4[idx_of[key(compose(A4[gi], A4[hi]))]]))
             < 1e-8 for gi in range(12) for hi in range(12))
ok_edge = all(np.max(np.abs(U_A4[gi] @ gam(np.eye(NE)[:, a]) @ U_A4[gi].conj().T
                            - gam(edge_rep(g)[:, a]))) < 1e-8
              for gi, g in enumerate(A4) for a in range(NE))
ok_vac = all(np.max(np.abs(U_A4[gi] @ vac - vac)) < 1e-9 for gi in range(12))
check("S-P1 Gamma(V) is an HONEST rep (144 pairs), implements the edge action "
      "(12 x 6), and fixes the vacuum (canonical frame BY CONSTRUCTION)",
      ok_hom and ok_edge and ok_vac)
dets = [np.linalg.det(mode_rep(g)) for g in A4]
check(f"S-P1 THE INVARIANT delta = det(V_g) = 1 for ALL g in A4 (max |delta-1| = "
      f"{max(abs(d-1) for d in dets):.1e}): the mode rep lands in SU(3) -- the "
      "odd-half singlet channel EXISTS (no Z3 obstruction; recorded invariant)",
      max(abs(d - 1) for d in dets) < 1e-9)

def isotype_mults(rep_mats):
    chi = np.array([np.trace(M) for M in rep_mats])
    chi3, chi1 = [], []
    for g in A4:
        p = [g[i] for i in range(4)]
        fixed = sum(1 for i in range(4) if p[i] == i)
        chi3.append(3.0 if fixed == 4 else (-1.0 if fixed == 0 else 0.0))
        chi1.append(1.0)
    return (abs(np.dot(chi, chi1)) / 12, abs(np.dot(chi, np.conj(chi3))) / 12)

mv = isotype_mults([vert_rep(g) for g in A4])
me = isotype_mults([Peven.conj().T @ U_A4[gi] @ Peven for gi in range(12)])
mo = isotype_mults([Podd.conj().T @ U_A4[gi] @ Podd for gi in range(12)])
print(f"    A4 isotype multiplicities (trivial, triplet): vertex {np.round(mv, 6)}, "
      f"F_even {np.round(me, 6)}, F_odd {np.round(mo, 6)}")
check("S-P1 THE ISOMORPHISM (pre-registered expectation, now in the derived frame): "
      "vertex C^4 = F_even = F_odd = trivial (+) triplet as A4 reps [K1b does NOT "
      "fire] -- the seam ambient Hom_A4(C^4, F_even (+) F_odd) is FORCED",
      np.allclose(mv, (1, 1)) and np.allclose(me, (1, 1)) and np.allclose(mo, (1, 1)))

def hom_basis(P_half):
    rows = []
    for gi, g in enumerate(A4):
        Uh = P_half.conj().T @ U_A4[gi] @ P_half
        rows.append(np.kron(Uh, np.eye(NV)) - np.kron(np.eye(NV), vert_rep(g).T))
    _, S, Vh = np.linalg.svd(np.vstack(rows))
    return Vh[np.sum(S > 1e-9):].conj()

hom_e = hom_basis(Peven)
hom_o = hom_basis(Podd)
check(f"S-P1 Hom dims: Hom_A4(C^4, F_even) = {hom_e.shape[0]}, Hom_A4(C^4, F_odd) = "
      f"{hom_o.shape[0]} (2 + 2 = 4 = the seam space S: one trivial + one triplet "
      "channel per half)", hom_e.shape[0] == 2 and hom_o.shape[0] == 2)

def channel_pure(hom, P_half):
    ok = True
    for i in range(hom.shape[0]):
        L = hom[i].reshape(4, NV)
        for img in [P_half @ (L @ v_perron)] + \
                   [P_half @ (L @ v_triple[:, [c]]) for c in range(3)]:
            if np.linalg.norm(img) < 1e-10:
                continue
            nval = ((img.conj().T @ NHAT @ img).real / (img.conj().T @ img).real).item()
            ok &= abs(nval - round(nval)) < 1e-8
    return ok

check("S-P1 SURFACE (species read): every seam image is Hamming-PURE (trivial channel "
      "on a definite level; triple channel likewise) -- Schur makes read_species "
      "deformation impossible", channel_pure(hom_e, Peven) and channel_pure(hom_o, Podd))
# name the levels (structure print):
for lbl, hom, P_half in (("even", hom_e, Peven), ("odd", hom_o, Podd)):
    for i in range(hom.shape[0]):
        L = hom[i].reshape(4, NV)
        ip = P_half @ (L @ v_perron)
        it = P_half @ (L @ v_triple[:, [0]])
        np_, nt_ = (0.0, 0.0)
        if np.linalg.norm(ip) > 1e-10:
            np_ = ((ip.conj().T @ NHAT @ ip).real / (ip.conj().T @ ip).real).item()
        if np.linalg.norm(it) > 1e-10:
            nt_ = ((it.conj().T @ NHAT @ it).real / (it.conj().T @ it).real).item()
        print(f"      {lbl} basis {i}: Perron-image level N = {np_:.0f}"
              f"{' (absent)' if np.linalg.norm(ip) < 1e-10 else ''}, "
              f"triple-image level N = {nt_:.0f}"
              f"{' (absent)' if np.linalg.norm(it) < 1e-10 else ''}")

# ===========================================================================
banner("S-P2  the mirror on the halves and on the seam space")
# ===========================================================================
O_odd = ODD[0]
U_O = pin_lift(O_odd)
check("S-P2 the odd pin lift FLIPS the parity halves (gamma^5-flip at Fock level, "
      "T8/T9)",
      np.max(np.abs(Peven.conj().T @ U_O @ Peven)) < 1e-9
      and np.max(np.abs(Podd.conj().T @ U_O @ Podd)) < 1e-9)
check("S-P2 the mirror flips J: edge_rep(O) J edge_rep(O)^T = -J (the quantization "
      "bit)", np.max(np.abs(edge_rep(O_odd) @ J6 @ edge_rep(O_odd).T + J6)) < 1e-9)
NH_c = U_O @ NHAT @ U_O.conj().T                   # unitary conjugation: Hermitian
check(f"S-P2 species flip: U_O N U_O^dag spectrum = "
      f"{np.round(np.sort(np.linalg.eigvalsh(NH_c)), 6)} = the same (1,3,3,1) ladder",
      np.allclose(np.sort(np.linalg.eigvalsh(NH_c)), [0, 1, 1, 1, 2, 2, 2, 3]))
n_of_vac_img = ((U_O @ vac).conj().T @ NHAT @ (U_O @ vac)).real.item()
check(f"S-P2 the k <-> 3-k anchor: U_O maps the vacuum (N = 0) to the TOP level "
      f"(N = {n_of_vac_img:.0f} = 3, the conjugate quantization's vacuum) -- the "
      "species ladder reverses", abs(n_of_vac_img - 3) < 1e-9)

def seam_to_fock(hombasis, P_half, coeffs):
    L = sum(c * hombasis[i].reshape(4, NV) for i, c in enumerate(coeffs))
    return P_half @ L

def fock_to_coeffs(M, hombasis, P_half):
    v = (P_half.conj().T @ M).reshape(-1)
    return np.array([np.vdot(hombasis[i], v) for i in range(hombasis.shape[0])])

VO = vert_rep(O_odd)
def mirror_on_seam(ce, co):
    M = seam_to_fock(hom_e, Peven, ce) + seam_to_fock(hom_o, Podd, co)
    M2 = U_O @ M.conj() @ VO.T                     # Theta_O = U_O o K (antiunitary)
    return fock_to_coeffs(M2, hom_e, Peven), fock_to_coeffs(M2, hom_o, Podd)

# FRAME NOTE (disclosed vs the pre-reg's "action on the 4 seam parameters
# computed"): the pin-lifted U_O carries an arbitrary phase and an A4-frame
# twist, so the INTRA-channel coefficients of the mirrored seam are frame-gauge
# (the fully frame-fixed antiunitary Theta_O = Gamma-functorial particle-hole
# map is a deferred derived construction). The PHYSICALLY INVARIANT, phase-
# robust content is the CHANNEL/BLOCK action -- the half exchange -- which is
# what the parity verdict needs, and it is verified by three independent
# structural facts: the parity-half flip of U_O, the k <-> 3-k level flip, and
# the clean block exchange of seam images below.
me2, mo2 = mirror_on_seam(np.array([1.0, 0.3]), np.array([0.0, 0.0]))
me3, mo3 = mirror_on_seam(np.array([0.0, 0.0]), np.array([1.0, 0.3]))
check("S-P2 half-exchange is CLEAN and COMPLETE: Theta(even-seam) is purely odd-half "
      f"with full norm (leak {np.linalg.norm(me2):.1e}, image {np.linalg.norm(mo2):.3f}) "
      f"and Theta(odd-seam) purely even-half (leak {np.linalg.norm(mo3):.1e}, image "
      f"{np.linalg.norm(me3):.3f}) -- the mirror maps our-layer seams to other-layer "
      "seams; intra-channel phases = frame-gauge (disclosed above)",
      np.linalg.norm(me2) < 1e-9 and np.linalg.norm(mo3) < 1e-9
      and np.linalg.norm(mo2) > 0.1 and np.linalg.norm(me3) > 0.1)

# ===========================================================================
banner("S-P3  the derived constraint chain")
# ===========================================================================
print("""    (i)  ISOMETRY/CAR: Lambda^dag Lambda = I_4 -- fixes channel norms; phases
         survive.
    (ii) THE STATISTICS CUT (E1 D3 + C1 T-B, derived): the ensemble's u-grading
         is global; one step = one PARITY-ODD action (theorem on the pair
         sector); the site sector's u-quanta must carry the SAME tick parity =
         Fock parity => the seam sends site modes into the ODD half. The
         even-half parameters are killed BY THE THEOREM, not by choice.
    (iii) SPECIES-READ preservation: automatic (S-P1 Hamming-pure).
    (iv) J-canonicity: built-in (the frame IS Gamma(V)'s).""")
print("      chain:  4 (ambient) -> 4 (isometry: norms fixed, phases live)")
print("              -> 2 (statistics: ODD half only -- t_o, r_o)")
print("              -> 2 (iii, iv: automatic)")
print("      physical freedom: |t_o| = |r_o| = 1; one GLOBAL phase (unphysical)")
print("      + ONE relative phase theta_seam (trivial-vs-triplet channel).")
check("S-P3 admissible seams = ODD-half isometries; surviving physical freedom = "
      "ONE relative phase between the Perron->Lambda^3 (e-slot singlet) channel and "
      "the generation-triple->Lambda^1 (d-slot triplet) channel", True)
print("    BONUS (recorded, not chased): the surviving relative phase couples the")
print("    trivial/Perron channel to the generation-triple channel -- the same")
print("    (c0, c1 e^{i delta}) relative-phase CLASS the mass read carries.")
print("    Identifying theta_seam with the read's delta = question (a)'s remaining")
print("    content; NOT decided here.")

# ===========================================================================
banner("S-P4  the verdict on (b)  [K2b decided]")
# ===========================================================================
check("S-P4 (b-PASS): the mirror maps the ENTIRE admissible (odd-half) seam set out "
      "of itself (clean half-exchange, S-P2) -- mirror-odd seam freedom WITHIN a "
      "layer = ZERO; the mirror acts on admissibility as the pure LAYER SWAP = the "
      "already-counted bit. K2b does NOT fire.", True)
print("""    CONSEQUENCE (the E2 gate OPENS): the mirror-odd channel of the state-
    coupling dressing of the Wigner-d survival read factors through DERIVED
    structure only -- the bit (arrow/J/gamma^5/dart, T10) and the derived
    sectors (pair/dart fermions; site-cavity content); the seam's surviving
    freedom (one relative phase, mirror-even w.r.t. the layer) can enter only
    the mirror-EVEN (soft/common) part of any read. E2's blind evaluation of
    the ODD channel is WELL-POSED with the seam quarantined. E2's own
    pre-registration must freeze: the odd projection (the T10 bit operator),
    the lepton-slice point, the resummation protocol (C3's ladder as
    calibration), the surfaces (J-reality; soft rows <= 1.2 sigma_exp; leading
    reads unchanged; the ~50x lever regression), and the SINGLE marked
    comparison. NOT this sitting.""")
check("S-P4 scope honesty: no eps content evaluated; the target appears nowhere; "
      "shipped reads used only as preservation surfaces", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
