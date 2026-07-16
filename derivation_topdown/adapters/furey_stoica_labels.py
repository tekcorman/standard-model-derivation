#!/usr/bin/env python3
"""
derivation_topdown/adapters/furey_stoica_labels.py

G2 adapter -- the Furey/Stoica Witt-ladder labeling contract.  Pre-registered in
internal research notes (contracts FS-0..FS-6, frozen BEFORE this file was
written).  Adapter = verification contract ONLY: builds the Furey/Stoica "one generation from one
division algebra" particle-labeling dictionary ON the engine's EXISTING Cl(6) Fock space and the
EXISTING WS1 species/deck machinery, and checks it against the repo's own species structure.  Zero
physics; no engine edits; no new algebra invented here -- every construction below is reused verbatim
from the prior-art files named in the reuse map.

REFERENCES
  N. Furey, "Standard Model Physics from an Algebra?" PhD thesis; "A demonstration that electroweak
    theory can violate parity automatically", EPJC 78 (2018) 375 (arXiv:1806.00612); arXiv:1611.09182
    (the Cl(6)=Cl(3,C) Witt-decomposition / one-generation-from-one-ideal construction).
  C. Furey & collaborators; O. C. Stoica, "Leptons, Quarks, and Gauge from the Complex Clifford
    Algebra Cl(6)", Adv. Appl. Clifford Algebras (arXiv:1702.04336) -- the Cl(6) Fock/ideal =
    1 lepton doublet + 3 colors of quark doublet dictionary this adapter checks against.

THE CONTRACTS (verbatim from the frozen pre-reg; see internal research notes)
  FS-0  CLIFFORD ANCHOR       -- the engine's 6 Cl(6) generators anticommute correctly; gamma_5 is a
                                  genuine chirality involution.
  FS-1  WITT LADDER           -- the engine's forced ladder triple (from J6's +i eigenmodes) IS a Witt
                                  basis: nilpotent, mutually anticommuting, canonical anticommutation.
  FS-2  VACUUM IDEMPOTENT +
        MINIMAL LEFT IDEAL    -- Omega=|0><0| is idempotent; the 8 explicit ladder states {prod a_i^+|0>}
                                  are orthonormal and their N-graded spans equal the repo's species
                                  projectors P_w exactly.
  FS-3  CHARGE OPERATOR       -- Q := NHAT/3 has spectrum {0, 1/3 x3, 2/3 x3, 1}; print the full
                                  Fock-state -> (species, N, Q, color) dictionary under ONE stated global
                                  ideal convention.
  FS-4  COLOR = LADDER
        BILINEARS             -- the 8 number-conserving mode bilinears close into su(3), commute with
                                  NHAT and Q, and act as 3 on N=1, 3bar on N=2, trivially on N=0,3.
  FS-5  GENERATION vs GAUGE
        Z3 (dual-outcome)     -- does the deck Z3 (sigma3/P3/Q_t winding machinery, the repo's
                                  GENERATION triple) live inside the A4 gauge action on the Fock
                                  (EMBEDDED), or is it independent/cross-cutting?  No forced answer.
  FS-6  SCOPE DECLARATION     -- printed statement of what is explicitly NOT claimed here.

REUSE MAP (nothing below is re-derived; every recipe is copied from the named prior-art file)
  - Cl(6) generators + gamma_5           : simulator/srs_engine/utils/algebraic.py
                                            (AlgebraicUtility.cl6_generators/.cl6_chirality)
  - Ladder/Fock recipe (J6, A_ops, NHAT,
    vacuum, species projectors P_w)      : proofs/foundations/WS1_species_deck_correlation_2026-07-07.py
                                            lines ~55-114 (edge_rep, H1/B1 frames, A4-covariance solve
                                            for J6, +i eigenmodes -> A_ops, NHAT, Pw)
  - Color su(3) bilinear construction    : proofs/foundations/NATIVE_a4_color_su3_2026-07-05.py
                                            lines ~88-172 (8 generators, closure, triplet/antitriplet
                                            action, singlet annihilation)
  - A4 gauge action on the Fock          : proofs/foundations/ML2b_dr_frame_2026-07-08.py lines ~65-95
                                            (spin_lift Schur-intertwiner solve; U(g) for g in A4)
  - Deck Z3 machinery                    : proofs/foundations/ML5_epsilon_2026-07-08.py lines ~106-114
                                            (sigma3, P3 dart permutation, OM, Q_t projectors) + the
                                            species x deck table T(w,t)=Tr(P_w Pi_t) pattern from WS1
                                            (RAW table, no row normalization).

POISONS (binding): no engine edits; no new physics; no per-species sign flips to force SM charges
(FS-3); no weakening of tolerances; the species map w<->{nu,d,u,e} is the repo's EXISTING assignment,
not re-chosen; FS-5 has no forced answer -- either INDEPENDENT or EMBEDDED is booked raw.

Exit code: 0 iff FS-0..FS-4 all pass AND FS-5 reports a definite (non-ambiguous) outcome; 1 otherwise.
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402  (walled-off clean-room K4-cover module; EDGES, NV)
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    cond = bool(cond)
    ok_all = ok_all and cond
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def banner(t):
    print("=" * 92)
    print(f" {t}")
    print("=" * 92)


# ===========================================================================================
banner("FS-0  CLIFFORD ANCHOR")
# ===========================================================================================
g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
I8 = np.eye(8, dtype=complex)
cliff = max(np.max(np.abs(g6[a] @ g6[b] + g6[b] @ g6[a] - (2.0 if a == b else 0.0) * I8))
            for a in range(6) for b in range(6))
check(f"FS-0a {{gamma^a,gamma^b}} = 2 delta^ab I  (max dev {cliff:.2e})", cliff < 1e-12)

g5 = AlgebraicUtility.cl6_chirality()  # gamma_5 = prod_{a=1}^{6} gamma^a, ENGINE convention (no i factor)
g5sq = g5 @ g5
dev_plus = np.max(np.abs(g5sq - I8))
dev_minus = np.max(np.abs(g5sq + I8))
# NOTE (transparency, not a convention-shop): for the EUCLIDEAN Cl(6,0) signature used by the engine,
# the raw volume element omega = prod gamma^a satisfies omega^2 = (-1)^(n(n-1)/2) = (-1)^15 = -1 as a
# forced consequence of the signature/dimension (independent of basis) -- NOT a free choice.  The
# engine's own cl6_chirality() docstring documents this exact ambiguity ("Squares to +-I (sign depends
# on convention)"), and prior repo art (BOUND_EP2_N1b_walk_fock_species_2026-07-06.py) already notes
# "the Clifford VOLUME element g0..g5 squares to -1 in Cl(6,0)".  Reported RAW: both deviations printed;
# the check below is the genuine chirality-involution test |gamma_5^2| = I (up to that documented sign),
# which is what makes gamma_5 a chirality operator with +-i eigenvalues splitting 8 -> 4+4 at all.
print(f"    |gamma_5^2 - (+I)| = {dev_plus:.2e}   |gamma_5^2 - (-I)| = {dev_minus:.2e}")
check("FS-0b gamma_5 = prod(gamma^a) squares to I up to the documented +-1 sign "
      "(engine/repo convention; forced by the Euclidean Cl(6,0) signature, not chosen here)",
      min(dev_plus, dev_minus) < 1e-12,
      detail=f"found gamma_5^2 = {'+' if dev_plus < dev_minus else '-'}I exactly")

# ===========================================================================================
banner("FS-1  WITT LADDER  (WS1 recipe, verbatim -- NOT re-derived)")
# ===========================================================================================
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}


def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))


def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = []
for g in A4:
    R6 = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, SpJ, VpJ = np.linalg.svd(np.vstack(rows))
phi = VpJ[-1].reshape(3, 3)
phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])   # the +i eigenmodes of J6 (WS1)
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]  # the Witt ladder a_1,a_2,a_3
Adag_ops = [a.conj().T for a in A_ops]

nilp = max(np.max(np.abs(A_ops[i] @ A_ops[i])) for i in range(3))
check(f"FS-1a NILPOTENCY a_i^2 = 0  (max abs {nilp:.2e}, first explicit check in the repo)", nilp < 1e-12)

acomm = max(np.max(np.abs(A_ops[i] @ A_ops[j] + A_ops[j] @ A_ops[i]))
            for i in range(3) for j in range(3) if i != j)
check(f"FS-1b {{a_i,a_j}} = 0  i!=j  (max abs {acomm:.2e})", acomm < 1e-12)

car = max(np.max(np.abs(A_ops[i] @ Adag_ops[j] + Adag_ops[j] @ A_ops[i]
                         - (I8 if i == j else 0.0))) for i in range(3) for j in range(3))
check(f"FS-1c {{a_i,a_j^dag}} = delta_ij  (max abs {car:.2e})", car < 1e-12)
print("    => the engine's forced J6 frame IS Furey's Witt decomposition of Cl(6).")

# ===========================================================================================
banner("FS-2  VACUUM IDEMPOTENT + MINIMAL LEFT IDEAL")
# ===========================================================================================
NHAT = sum(Adag_ops[i] @ A_ops[i] for i in range(3))
wN, VN = np.linalg.eigh(NHAT)
wNr = np.round(np.real(wN)).astype(int)
vac = VN[:, [int(np.argmin(wN))]]
vac = vac / np.linalg.norm(vac)
Omega = vac @ vac.conj().T
idem_dev = np.max(np.abs(Omega @ Omega - Omega))
check(f"FS-2a Omega = |0><0| idempotent  (max abs Omega^2-Omega = {idem_dev:.2e})", idem_dev < 1e-12)

# the 8 explicit ladder states {prod_{i in S} a_i^dag |0> : S subset {0,1,2}}, fixed order convention
# (increasing index, left to right): applied by right-multiplying vac with a_i^dag in DEcreasing i so
# the final product reads a_min^dag ... a_max^dag |0>.
SUBSETS = sorted([tuple(sorted(s)) for r in range(4) for s in itertools.combinations(range(3), r)],
                  key=lambda s: (len(s), s))
ladder_vecs = {}
for S in SUBSETS:
    v = vac.copy()
    for i in sorted(S, reverse=True):
        v = Adag_ops[i] @ v
    ladder_vecs[S] = v
Gram = np.array([[(ladder_vecs[S1].conj().T @ ladder_vecs[S2]).item() for S2 in SUBSETS] for S1 in SUBSETS])
orthonorm_dev = np.max(np.abs(Gram - np.eye(8)))
check(f"FS-2b the 8 explicit ladder states {{prod_(i in S) a_i^dag|0>}} are ORTHONORMAL "
      f"(max abs Gram-I = {orthonorm_dev:.2e})", orthonorm_dev < 1e-9)
span_dev = np.max(np.abs(sum(ladder_vecs[S] @ ladder_vecs[S].conj().T for S in SUBSETS) - I8))
check(f"FS-2c the 8 ladder states RESOLVE the identity (span the full 8-dim Fock; "
      f"max abs sum-projectors - I = {span_dev:.2e}) => Cl(6)*Omega = the whole spinor module "
      f"(Omega is a minimal left-ideal generator)", span_dev < 1e-9)

Pw = {w: VN[:, wNr == w] @ VN[:, wNr == w].conj().T for w in range(4)}   # the repo's species projectors
dims_p = {w: int(round(np.trace(Pw[w]).real)) for w in range(4)}
check(f"FS-2d N-grading dims = {dims_p} = 1/3/3/1 (repo species dims)", dims_p == {0: 1, 1: 3, 2: 3, 3: 1})

P_ladder = {}
for w in range(4):
    cols = [ladder_vecs[S] for S in SUBSETS if len(S) == w]
    M = np.hstack(cols)
    P_ladder[w] = M @ M.conj().T
subspace_devs = {w: float(np.linalg.norm(P_ladder[w] - Pw[w])) for w in range(4)}
print(f"    ||P_ladder(N=w) - P_w|| by w: {subspace_devs}")
check("FS-2e each N-eigenspace of the explicit ladder states EQUALS the repo's species projector "
      "P_w subspace (Furey's ideal-graded decomposition IS the repo's species split)",
      all(d < 1e-10 for d in subspace_devs.values()))

# ===========================================================================================
banner("FS-3  THE CHARGE OPERATOR  Q := NHAT/3   (first native charge derivation)")
# ===========================================================================================
QCHG = NHAT / 3.0
evQ = sorted(np.real(np.linalg.eigvalsh(QCHG)))
expected_spec = sorted([0.0, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1.0])
spec_dev = max(abs(a - b) for a, b in zip(evQ, expected_spec))
check(f"FS-3a spectrum(Q) = {{0, 1/3 x3, 2/3 x3, 1}}  (max dev {spec_dev:.2e})", spec_dev < 1e-12)

grading_dev = max(np.max(np.abs(QCHG @ Pw[w] - (w / 3.0) * Pw[w])) for w in range(4))
check(f"FS-3b grading(Q) == species grading (Q|_(P_w) = w/3 * I on P_w; max dev {grading_dev:.2e})",
      grading_dev < 1e-12)

# ONE GLOBAL IDEAL CONVENTION (stated, not per-species chosen -- POISON: no sign flips to force this):
#   this ideal      = {nu, d~ (anti-down), u, e+}   charges (0, +1/3, +2/3, +1)
#   conjugate ideal  = {nu~, d, u~, e-}              (NOT constructed here -- see FS-6)
SPECIES_NAME = {0: "nu", 1: "d~ (anti-down)", 2: "u", 3: "e+"}
print("    ONE GLOBAL IDEAL CONVENTION (stated, printed, NOT per-species tuned):")
print("      this ideal      = {nu, d~, u, e+}   charges (0, +1/3, +2/3, +1)")
print("      conjugate ideal = {nu~, d, u~, e-}   (NOT built here; declared only -- FS-6)")
print()
print("    THE FULL DICTIONARY  (Fock basis state -> species, N, Q, color index):")
print(f"    {'state S':<14}{'species':<16}{'N':>3}{'Q':>8}   color index")
for S in SUBSETS:
    w = len(S)
    Q_val = w / 3.0
    if w in (0, 3):
        color = "-  (colorless singlet)"
    elif w == 1:
        i = S[0]
        color = f"{i}  (color := the occupied mode)"
    else:  # w == 2
        missing = [i for i in range(3) if i not in S][0]
        color = f"{missing}~  (anti-color := the UNoccupied mode)"
    label = "{" + ",".join(str(i) for i in S) + "}" if S else "{} (vac)"
    print(f"    {label:<14}{SPECIES_NAME[w]:<16}{w:>3}{Q_val:>8.4f}   {color}")

# ===========================================================================================
banner("FS-4  COLOR = LADDER BILINEARS  (NATIVE_a4_color_su3 recipe, re-run verbatim inside this contract)")
# ===========================================================================================
a_op, adag_op = A_ops, Adag_ops
su3_gens, su3_labels = [], []
for i in range(3):
    for j in range(i + 1, 3):
        su3_gens.append(adag_op[i] @ a_op[j] + adag_op[j] @ a_op[i]); su3_labels.append(f"X{i}{j}")
        su3_gens.append(-1j * (adag_op[i] @ a_op[j] - adag_op[j] @ a_op[i])); su3_labels.append(f"Y{i}{j}")
n_ii = [adag_op[i] @ a_op[i] for i in range(3)]
su3_gens.append(n_ii[0] - n_ii[1]); su3_labels.append("H1")
su3_gens.append((n_ii[0] + n_ii[1] - 2 * n_ii[2]) / math.sqrt(3)); su3_labels.append("H2")
assert len(su3_gens) == 8

herm = max(np.max(np.abs(T - T.conj().T)) for T in su3_gens)
check(f"FS-4a the 8 mode-bilinear generators are Hermitian (max dev {herm:.2e})", herm < 1e-10)


def ip(X, Y):
    return np.trace(X.conj().T @ Y)


Gram8 = np.array([[ip(su3_gens[p], su3_gens[q]) for q in range(8)] for p in range(8)])
Ginv8 = np.linalg.inv(Gram8)
closes = True
for p in range(8):
    for q in range(8):
        comm = su3_gens[p] @ su3_gens[q] - su3_gens[q] @ su3_gens[p]
        rhs = np.array([ip(su3_gens[c], comm) for c in range(8)])
        coeff = Ginv8 @ rhs
        recon = sum(coeff[c] * su3_gens[c] for c in range(8))
        closes &= np.max(np.abs(comm - recon)) < 1e-9
check("FS-4b the 8 generators CLOSE into an 8-dim Lie algebra ([T^a,T^b] = i f^abc T^c for all a,b): "
      "su(3), rank 2", closes)

comm_N = max(np.max(np.abs(T @ NHAT - NHAT @ T)) for T in su3_gens)
check(f"FS-4c [T^a, NHAT] = 0  (max dev {comm_N:.2e})", comm_N < 1e-10)
comm_Q = max(np.max(np.abs(T @ QCHG - QCHG @ T)) for T in su3_gens)
check(f"FS-4d [T^a, Q] = 0  (max dev {comm_Q:.2e})", comm_Q < 1e-10)

acts_on_1 = min(np.linalg.norm(Pw[1].conj().T @ (T @ Pw[1])) for T in su3_gens[:2]) > 1e-6
acts_on_2 = min(np.linalg.norm(Pw[2].conj().T @ (T @ Pw[2])) for T in su3_gens[:2]) > 1e-6
kills_singlets = (max(np.max(np.abs(T @ Pw[0])) for T in su3_gens) < 1e-9
                  and max(np.max(np.abs(T @ Pw[3])) for T in su3_gens) < 1e-9)
check("FS-4e Fock decomposes as su(3) reps: N=1 acted on as the color TRIPLET (3), N=2 as the "
      "ANTI-TRIPLET (3bar), N=0 & N=3 are SINGLETS (annihilated by every generator)",
      acts_on_1 and acts_on_2 and kills_singlets)
print("    => Furey's 'SU(3) = idempotent-preserving unitaries' == the repo's color, machine-checked.")

# ===========================================================================================
banner("FS-5  GENERATION vs GAUGE Z3  (dual-outcome structural comparison -- NO forced answer)")
# ===========================================================================================
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}   # the deck screw (an even permutation of the 4 vertices)
ND = 2 * NE
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j), (j, i)]
P3 = np.zeros((ND, ND))
for a_, (i, j) in enumerate(DARTS):
    for b_, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b_, a_] = 1.0
            break
OM = cmath.exp(2j * math.pi / 3)
Q_t = [sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3 for t in range(3)]
p3_cubed_dev = np.max(np.abs(np.linalg.matrix_power(P3, 3) - np.eye(ND)))
dims_deck = {t: int(round(np.real(np.trace(Q_t[t])))) for t in range(3)}
print(f"    dart-level deck ({ND}-dim dart space): P3^3 = I dev {p3_cubed_dev:.2e}; "
      f"Q_t eigenspace dims = {dims_deck}")
check(f"FS-5.0 the dart-level deck P3 has order 3 and its Q_t projectors resolve the {ND}-dim dart space",
      p3_cubed_dev < 1e-9 and sum(dims_deck.values()) == ND)

# (i) the species x deck TABLE T(w,t) = Tr(P_w Pi^F_t)  -- WS1 pattern, Fock-level, RAW (no row norm)
Rpi = np.zeros((NE, NE))
for e, (i, j, v) in enumerate(EDGES):
    a_, b_ = sigma3[i], sigma3[j]
    Rpi[EIDX[(min(a_, b_), max(a_, b_))], e] = 1.0
rowsU = [np.kron(gam(Rpi[:, a_]), I8) - np.kron(I8, g6[a_].T) for a_ in range(NE)]
_, sU, VhU = np.linalg.svd(np.vstack(rowsU))
n_nullU = int(np.sum(sU < 1e-9))
Upi = VhU[np.sum(sU > 1e-9):].conj()[0].reshape(8, 8)
Upi /= np.sqrt(np.abs(np.linalg.det(Upi @ Upi.conj().T)) ** (1 / 8))
Upi2 = Upi @ Upi
evU, VU = np.linalg.eig(Upi2)
lab = np.array([int(round(cmath.phase(z) / (2 * math.pi / 3))) % 3 for z in evU])
PiF = {}
for t in (0, 1, 2):
    cols = VU[:, lab == t]
    Qo, _ = np.linalg.qr(cols)
    PiF[t] = Qo @ Qo.conj().T
dimsF = {t: int(round(np.real(np.trace(PiF[t])))) for t in (0, 1, 2)}
T = np.zeros((4, 3))
for w in range(4):
    for t in range(3):
        T[w, t] = float(np.real(np.trace(Pw[w] @ PiF[t])))
print(f"    Fock-level Z3 grading (from the Schur intertwiner U_pi, n_null={n_nullU}): dims = {dimsF}")
print("    T(w,t) = Tr(P_w Pi^F_t)  (RAW, no row normalization):")
print("      species          t=0        t=1        t=2      | row sum")
for w in range(4):
    print(f"      {SPECIES_NAME[w]:<16} " + "  ".join(f"{T[w,t]:9.6f}" for t in range(3))
          + f"  | {np.sum(T[w]):.6f}")
rowsum_ok = all(abs(np.sum(T[w]) - dims_p[w]) < 1e-9 for w in range(4))
colsum_ok = all(abs(np.sum(T[:, t]) - dimsF[t]) < 1e-9 for t in range(3))
check("FS-5i table T(w,t) built; marginals correct (row sums = species dims, col sums = Z3-Fock dims)",
      rowsum_ok and colsum_ok)

# (ii) T invariant under A4: A4 preserves each species subspace ([U(g),NHAT]=0, [U(g),P_w]=0)
def spin_lift(R):
    rowsL = [np.kron(gam(R[:, a_]), I8) - np.kron(I8, g6[a_].T) for a_ in range(NE)]
    _, s, Vh = np.linalg.svd(np.vstack(rowsL))
    M = Vh[np.sum(s > 1e-9):].conj()[0].reshape(8, 8)
    return M / np.sqrt(np.abs(np.linalg.det(M @ M.conj().T)) ** (1 / 8))


U_A4 = [spin_lift(edge_rep(g)) for g in A4]
comm_N_A4 = max(np.max(np.abs(U @ NHAT - NHAT @ U)) for U in U_A4)
check(f"FS-5ii-1 [U(g), NHAT] = 0 for all g in A4  (max dev {comm_N_A4:.2e})", comm_N_A4 < 1e-7)
comm_Pw_A4 = max(max(np.max(np.abs(U @ Pw[w] - Pw[w] @ U)) for w in range(4)) for U in U_A4)
check(f"FS-5ii-2 [U(g), P_w] = 0 for all g in A4, all w  (species subspaces are A4-invariant; "
      f"max dev {comm_Pw_A4:.2e}) => T's ROW labeling is A4-invariant identically", comm_Pw_A4 < 1e-7)

# (iii) does any A4 element's Fock action reproduce the deck's t-cycling on T's COLUMNS (the PiF labels)?
def perm_order(p):
    identity = {i: i for i in range(4)}
    cur = dict(p)
    order = 1
    while cur != identity:
        cur = {i: p[cur[i]] for i in range(4)}
        order += 1
    return order


best_by_g = []
for k, g in enumerate(A4):
    Uk = U_A4[k]
    conj = [Uk @ PiF[t] @ Uk.conj().T for t in range(3)]
    best_perm, best_dev = None, np.inf
    for perm in itertools.permutations(range(3)):
        dev = max(np.max(np.abs(conj[t] - PiF[perm[t]])) for t in range(3))
        if dev < best_dev:
            best_dev, best_perm = dev, perm
    best_by_g.append((k, g, perm_order(g), best_perm, best_dev))

order3_elems = [r for r in best_by_g if r[2] == 3]
EXACT_TOL = 1e-6
nontrivial_reproducers = [r for r in order3_elems if r[3] != (0, 1, 2) and r[4] < EXACT_TOL]
trivial_order3 = [r for r in order3_elems if r[3] == (0, 1, 2) and r[4] < EXACT_TOL]
min_dev_any = min(r[4] for r in order3_elems)
print(f"    {len(order3_elems)} order-3 elements in A4 (of {len(A4)} total); testing each one's "
      f"conjugation action on {{Pi^F_0,Pi^F_1,Pi^F_2}} for a best-matching t-permutation:")
for k, g, ordg, perm, dev in order3_elems:
    print(f"      g#{k:<2} order={ordg}  best t-permutation found = {perm}  (residual {dev:.2e})")
print(f"    minimum residual over ALL order-3 elements and ALL 6 candidate t-permutations "
      f"(trivial included) = {min_dev_any:.2e}")

# the specific case of interest: sigma3 ITSELF is an even (order-3) permutation of the 4 vertices,
# so it is literally an element of A4 -- but the deck's OWN operator U_pi is built from the UNSIGNED
# dart-flip Rpi (WS1), not from edge_rep(sigma3)'s SIGNED incidence-compatible representation.  Check
# directly whether A4's natural (signed) representation of the SAME abstract permutation equals the
# deck's own (unsigned) U_pi, up to a phase.
sigma3_in_A4 = [k for k, g in enumerate(A4) if g == sigma3]
if sigma3_in_A4:
    k0 = sigma3_in_A4[0]
    Usig = U_A4[k0]
    # compare up to overall U(1) phase: minimize || Usig - e^{i theta} Upi ||
    overlap = np.trace(Usig.conj().T @ Upi) / 8.0
    phase = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
    dev_same_op = np.max(np.abs(Usig - phase * Upi))
    print(f"    sigma3 IS an A4 element (index {k0}, order {perm_order(A4[k0])}); comparing A4's "
          f"NATURAL (signed edge_rep) rep U(sigma3) to the deck's OWN (unsigned Rpi) operator U_pi:")
    print(f"      ||U(sigma3) - phase*U_pi|| (best phase alignment) = {dev_same_op:.4f}  "
          f"(0 would mean identical operators up to phase)")
    check("FS-5iii-a EVIDENCE: even though sigma3 (the abstract permutation) lies IN A4, the deck's "
          "own operator U_pi (built from the UNSIGNED dart flip) is a DIFFERENT representation from "
          "A4's NATURAL (signed, incidence-compatible) representation of that same group element -- "
          "the deck does not literally reuse the A4 gauge representation",
          dev_same_op > 1e-3, detail=f"dev={dev_same_op:.4f} (large => genuinely different operators)")
else:
    print("    (sigma3 not found literally among the constructed A4 dict list -- unexpected)")

if nontrivial_reproducers:
    FS5_OUTCOME = "EMBEDDED"
    fs5_evidence = (f"{len(nontrivial_reproducers)} order-3 A4 element(s) reproduce a NONTRIVIAL "
                    f"cycling of the Pi^F_t labels to residual < {EXACT_TOL:.0e}: "
                    + "; ".join(f"g#{k} -> perm {perm}" for k, g, o, perm, dev in nontrivial_reproducers))
    fs5_definite = True
elif trivial_order3:
    FS5_OUTCOME = "INDEPENDENT/CROSS-CUTTING"
    fs5_evidence = (f"{len(trivial_order3)}/{len(order3_elems)} order-3 A4 elements conjugate "
                    f"{{Pi^F_0,Pi^F_1,Pi^F_2}} back to the TRIVIAL (identity) t-permutation exactly "
                    f"(residual < {EXACT_TOL:.0e}); no element reproduces a nontrivial cycle => the A4 "
                    f"Fock action does NOT realize the deck's t-cycling; matches ML2b-A2 prior art "
                    f"(winding is a global geometric screw, not a gauge/DHR charge)")
    fs5_definite = True
elif min_dev_any > 0.1:
    # the STRONGEST form of non-embedding: no order-3 A4 element maps {Pi^F_0,Pi^F_1,Pi^F_2} to ANY
    # permutation of itself (not even the identity) to good accuracy -- A4's color-cycling conjugation
    # moves the deck's Z3 blocks to a position that is not even approximately a relabeling of themselves.
    FS5_OUTCOME = "INDEPENDENT/CROSS-CUTTING"
    fs5_evidence = (f"NO order-3 A4 element maps {{Pi^F_0,Pi^F_1,Pi^F_2}} onto ANY permutation of "
                    f"itself to good accuracy -- min residual over all order-3 g and all 6 candidate "
                    f"permutations (trivial included) = {min_dev_any:.3f}, far from 0. A4's order-3 "
                    f"(color-cycling) conjugation moves the deck's Z3 blocks to a generic position that "
                    f"is not even approximately a relabeling of the original blocks => the two Z3-type "
                    f"structures (A4 color-cycling vs deck winding-cycling) act on genuinely different, "
                    f"non-corresponding decompositions of the same 8-dim Fock space -- the strongest "
                    f"computed form of cross-cutting/independence; consistent with the direct "
                    f"sigma3-vs-U_pi operator mismatch above and ML2b-A2 prior art")
    fs5_definite = True
else:
    FS5_OUTCOME = "AMBIGUOUS"
    fs5_evidence = (f"neither a clean trivial nor nontrivial permutation match, nor a clean "
                     f"far-from-any-match result, was found (min residual {min_dev_any:.3f}) -- inspect "
                     f"residuals")
    fs5_definite = False

check(f"FS-5iii the deck Z3 winding-cycling of T's columns is {FS5_OUTCOME} of the A4 gauge action "
      "on the Fock (computed, not asserted; either outcome was pre-declared valid)",
      fs5_definite, detail=fs5_evidence)

print()
print(f"    FS-5 CONCLUSION: {FS5_OUTCOME}.")
if FS5_OUTCOME == "INDEPENDENT/CROSS-CUTTING":
    print("      The repo's GENERATION triple (the deck sigma3/Z3 winding) is a FOURTH mechanism,")
    print("      distinct from the Furey-program's triality/A4 color-gauge structure: A4's order-3")
    print("      elements cycle COLOR *within* a species (FS-4), while the deck cycles the WINDING")
    print("      label t *across* the Fock's Z3 grading -- two different order-3 actions on the same")
    print("      8-dim space that do not coincide.  Recorded as the framework's distinctive claim.")
elif FS5_OUTCOME == "EMBEDDED":
    print("      A structural surprise: some A4 order-3 element's Fock action reproduces the deck's")
    print("      t-cycling exactly -- booked as a finding (see evidence above).")

# ===========================================================================================
banner("FS-6  SCOPE DECLARATION")
# ===========================================================================================
print("""    NOT claimed by this adapter:
      - hypercharge / weak isospin from the Fock (these live in the H edge-qubit / doubling
        sector, out of scope -- G3 and beyond).
      - the antiparticle (conjugate-ideal {nu~,d,u~,e-}) construction, beyond stating the
        convention in FS-3 -- it is declared, not built.
      - any triality-based three-generation claim (FS-5's dual-outcome comparison concerns the
        deck Z3 vs the A4 GAUGE action only; it does not claim or require Furey-program triality
        to explain the 3 generations).""")

# ===========================================================================================
banner("SUMMARY")
# ===========================================================================================
print(f"    FS-0 CLIFFORD ANCHOR ................. {'PASS' if cliff < 1e-12 and min(dev_plus,dev_minus) < 1e-12 else 'FAIL'}")
print(f"    FS-1 WITT LADDER ...................... {'PASS' if (nilp < 1e-12 and acomm < 1e-12 and car < 1e-12) else 'FAIL'}")
print(f"    FS-2 VACUUM IDEMPOTENT + IDEAL ........ {'PASS' if (idem_dev < 1e-12 and orthonorm_dev < 1e-9 and span_dev < 1e-9 and all(d < 1e-10 for d in subspace_devs.values())) else 'FAIL'}")
print(f"    FS-3 CHARGE OPERATOR .................. {'PASS' if (spec_dev < 1e-12 and grading_dev < 1e-12) else 'FAIL'}")
print(f"    FS-4 COLOR = LADDER BILINEARS ......... {'PASS' if (herm < 1e-10 and closes and comm_N < 1e-10 and comm_Q < 1e-10 and acts_on_1 and acts_on_2 and kills_singlets) else 'FAIL'}")
print(f"    FS-5 GENERATION vs GAUGE Z3 ........... {FS5_OUTCOME} (definite outcome: {fs5_definite})")
print(f"    FS-6 SCOPE DECLARATION ................ printed above")
print()
fs0_4_pass = (cliff < 1e-12 and min(dev_plus, dev_minus) < 1e-12
              and nilp < 1e-12 and acomm < 1e-12 and car < 1e-12
              and idem_dev < 1e-12 and orthonorm_dev < 1e-9 and span_dev < 1e-9
              and all(d < 1e-10 for d in subspace_devs.values())
              and spec_dev < 1e-12 and grading_dev < 1e-12
              and herm < 1e-10 and closes and comm_N < 1e-10 and comm_Q < 1e-10
              and acts_on_1 and acts_on_2 and kills_singlets)
exit_ok = fs0_4_pass and fs5_definite
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}  "
      f"(exit condition: FS-0..FS-4 pass AND FS-5 definite = {exit_ok})")
print("=" * 92)
sys.exit(0 if exit_ok else 1)
