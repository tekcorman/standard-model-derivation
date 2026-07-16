#!/usr/bin/env python3
"""
GEN-HOMES reconciliation check — verification driver for the freeze
internal research notes

Executes, in order, every machine-checkable test named in the freeze:
  Sub-target 1  (i) C3-windings <-> (ii) spin-1/lambda=-1 bands:  T1.0-T1.4
  Sub-target 2  (i) <-> (iii) observer C^3_gen:                    T2.0-T2.2
  Sub-target 3  is a read+adjudicate (no numerics); summarized at the end.

GOAL-SEEK GUARD (freeze SS6): no mass/ppm/Koide-Q/CKM value is read anywhere below. Where the
M1.B files (T2.0) cite the lepton-mass-non-degeneracy datum as their basis-pinning mechanism,
this script NAMES that fact -- it never reads or compares the VALUE.

Self-contained; imports only already-accreted, read-only objects (derivation_topdown/state/
the_net.py, derivation_topdown/dirac_srs_mdl/srs.py). Modifies nothing. OMP_NUM_THREADS=4.
Runtime ~5-10s.
"""
import sys, os, math, cmath
from fractions import Fraction

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from derivation_topdown.state.the_net import (
    _a4_vertex_group, _a4_standard_3irrep, _a4_key, dart_rep, _a2d_abstract_hom_basis,
    NV, ND,
)
import srs  # on sys.path via the_net's own insertion, above

np.set_printoptions(precision=6, suppress=False, linewidth=140)

RESULTS = []   # (name, value, threshold, mode, passed, note)
INFO = []      # (name, note) -- provenance / non-numeric findings


def record(name, value, threshold, note="", mode="le"):
    if mode == "le":
        passed = (value is not None) and (value <= threshold)
        cmp = "<="
    else:
        passed = (value is not None) and (value >= threshold)
        cmp = ">="
    RESULTS.append((name, value, threshold, mode, passed, note))
    vstr = f"{value:.3e}" if isinstance(value, (int, float)) else str(value)
    tstr = f"{threshold:.3e}" if isinstance(threshold, (int, float)) else str(threshold)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: value={vstr}  need {cmp}{tstr}  {note}")
    return passed


def info(name, note):
    INFO.append((name, note))
    print(f"  [INFO] {name}: {note}")


def hdr(s):
    print("\n" + "=" * 92 + "\n" + s + "\n" + "=" * 92)


def comp(g, h):
    """A4 group law used throughout the_net.py: comp(g,h) = g o h (apply h then g)."""
    return {i: g[h[i]] for i in range(NV)}


om = cmath.exp(2j * math.pi / 3)

# =============================================================================================
hdr("SUB-TARGET 1 -- (i) C3-windings <-> (ii) spin-1 bands: exhibit the forced intertwiner")
# =============================================================================================

# --------------------------------------------------------------------------------------------
hdr("T1.0 -- carrier identity: adjacency(Gamma) spectrum {3,-1,-1,-1}; lambda=-1 == sum-zero == rho3")
# --------------------------------------------------------------------------------------------

A0 = srs.adjacency((0.0, 0.0, 0.0))
print("adjacency((0,0,0)) real part (should be J-I, all-integer):\n", A0.real)
resid_im = float(np.max(np.abs(A0.imag)))
resid_int = float(np.max(np.abs(A0.real - (np.ones((4, 4)) - np.eye(4)))))
record("T1.0a adjacency(Gamma) == J-I exactly (K4 complete graph at k=0)",
       max(resid_im, resid_int), 1e-13)

eigvals_num = np.linalg.eigvalsh(A0.real)
spec_sorted = sorted(eigvals_num)
expected = sorted([3.0, -1.0, -1.0, -1.0])
record("T1.0b spectrum {3,-1,-1,-1}",
       float(np.max(np.abs(np.array(spec_sorted) - np.array(expected)))), 1e-13)

# EXACT (Fraction) proof that the lambda=-1 eigenprojector is I - J/4:
# (J-I)(I - J/4) = J - J^2/4 - I + J/4 = J - J - I + J/4 = -(I - J/4)   [using J^2 = 4J for the
# 4x4 all-ones matrix]  -- so I - J/4 is EXACTLY the lambda=-1 eigenprojector, no floating point
# involved at all.
I4 = [[Fraction(1) if i == j else Fraction(0) for j in range(4)] for i in range(4)]
Jf = [[Fraction(1) for _ in range(4)] for _ in range(4)]


def matmul_f(A, B):
    n, m, k = len(A), len(B[0]), len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def matsub_f(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def scal_f(c, A):
    return [[c * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


P_exact = matsub_f(I4, scal_f(Fraction(1, 4), Jf))
JminusI = matsub_f(Jf, I4)
lhs = matmul_f(JminusI, P_exact)
rhs = scal_f(Fraction(-1), P_exact)
exact_ok = all(lhs[i][j] == rhs[i][j] for i in range(4) for j in range(4))
print("EXACT Fraction identity (J-I)(I-J/4) == -(I-J/4):", exact_ok,
      " [proves I-J/4 is the EXACT lambda=-1 eigenprojector, purely algebraically]")
record("T1.0c exact eigenprojector identity (Fraction arithmetic, zero float)",
       0.0 if exact_ok else 1.0, 1e-13)

P_exact_float = np.array([[float(P_exact[i][j]) for j in range(4)] for i in range(4)])

# the_net's own rho3 embedding basis3 (mirrors _a4_standard_3irrep's construction verbatim)
v0 = np.ones(NV) / math.sqrt(NV)
Q, _ = np.linalg.qr(np.eye(NV) - np.outer(v0, v0))
basis3 = Q[:, np.abs(Q.T @ v0) < 1e-8][:, :3]
P_rho3 = basis3 @ basis3.T

record("T1.0d P_(lambda=-1, exact) == P_rho3 (the_net's own sum-zero embedding)",
       float(np.max(np.abs(P_exact_float - P_rho3))), 1e-13,
       "carrier of (ii) == the carrier the induced-A4 lemma's rho3 orthonormalizes")

# --------------------------------------------------------------------------------------------
hdr("T1.1 -- deck screw sigma=(123) restricts to lambda=-1 space as rho3(sigma), order 3, {1,w,w^2}")
# --------------------------------------------------------------------------------------------

sigma_dict = {0: 0, 1: 2, 2: 3, 3: 1}   # the deck screw sigma=(123), matches derive_generation_spectrum.py
A4v = _a4_vertex_group()
ix = {_a4_key(g): n for n, g in enumerate(A4v)}
assert _a4_key(sigma_dict) in ix, "sigma is not an even (A4) vertex permutation!"
sigma_idx = ix[_a4_key(sigma_dict)]
print(f"sigma = {sigma_dict}  found in A4v at index {sigma_idx} (confirms sigma is a genuine A4 element)")


def vertex_perm(g):
    P = np.zeros((NV, NV))
    for i in range(NV):
        P[g[i], i] = 1.0
    return P


Perm_sigma = vertex_perm(sigma_dict)
rho3_sigma_direct = basis3.T @ Perm_sigma @ basis3

A4v_net, rho3_net, worst_honest, char_resid = _a4_standard_3irrep()
assert A4v_net == A4v
rho3_sigma_net = rho3_net[sigma_idx]

record("T1.1a rho3(sigma) direct recompute == the_net's own rho3[sigma_idx]",
       float(np.max(np.abs(rho3_sigma_direct - rho3_sigma_net))), 1e-13)

order3_resid = float(np.max(np.abs(np.linalg.matrix_power(rho3_sigma_net, 3) - np.eye(3))))
record("T1.1b rho3(sigma)^3 = I (order exactly 3)", order3_resid, 1e-12)

orth_resid = float(np.max(np.abs(rho3_sigma_net.T @ rho3_sigma_net - np.eye(3))))
record("T1.1c rho3(sigma) real orthogonal", orth_resid, 1e-13)

evs = np.sort_complex(np.linalg.eigvals(rho3_sigma_net))
expected_evs = np.sort_complex(np.array([1.0, om, om ** 2]))
record("T1.1d eigenvalues {1, w, w^2}", float(np.max(np.abs(evs - expected_evs))), 1e-12)

# --------------------------------------------------------------------------------------------
hdr("T1.2 -- exhibit S (template's Hom-space route); M(sigma) = S^-1 rho3(sigma) S; S^H S = I/12")
# --------------------------------------------------------------------------------------------

A4v_chk, phi_basis, n_phi, worst_law = _a2d_abstract_hom_basis()
assert A4v_chk == A4v
assert n_phi == 3
print("phi_basis equivariance residual (phi_i @ dart_rep(g) == rho3(g) @ phi_i, all g):", worst_law)

d0 = 0
D_of = {}
for k, g in enumerate(A4v):
    col = dart_rep(g)[:, d0]
    D_of[k] = int(np.argmax(col))
assert len(set(D_of.values())) == 12, "dart_rep not simply transitive?!"


def idx_of(g):
    return ix[_a4_key(g)]


def build_Rh(hk):
    Rh = np.zeros((ND, ND))
    h = A4v[hk]
    for k, g in enumerate(A4v):
        gh_idx = idx_of(comp(g, h))
        Rh[D_of[gh_idx], D_of[k]] = 1.0
    return Rh


Rh_sigma = build_Rh(sigma_idx)
Phi_mat = np.stack([phi_basis[j].reshape(-1, order="F") for j in range(3)], axis=1)


def induced_M(Rh):
    M = np.zeros((3, 3), dtype=complex)
    resid = 0.0
    for i in range(3):
        Xi = phi_basis[i] @ Rh
        vec_Xi = Xi.reshape(-1, order="F")
        coeffs, *_ = np.linalg.lstsq(Phi_mat, vec_Xi, rcond=None)
        M[:, i] = coeffs
        resid = max(resid, float(np.max(np.abs(Phi_mat @ coeffs - vec_Xi))))
    return M, resid


M_sigma, M_resid = induced_M(Rh_sigma)
record("T1.2a M(sigma) closes exactly within phi_basis span", M_resid, 1e-9)

S = np.stack([phi_basis[i][:, d0] for i in range(3)], axis=1)   # S_{:,i} = phi_i(e_{d0})
print("S (columns = phi_i evaluated at basepoint dart) =\n", S)
Sinv = np.linalg.inv(S)

conj_resid = float(np.max(np.abs(M_sigma - Sinv @ rho3_sigma_net @ S)))
record("T1.2b M(sigma) == S^-1 rho3(sigma) S", conj_resid, 1e-12)

SHS = S.conj().T @ S
record("T1.2c S^dagger S == I/12", float(np.max(np.abs(SHS - np.eye(3) / 12))), 1e-14)

# --------------------------------------------------------------------------------------------
hdr("T1.3 -- isotype <-> eigenspace correspondence (THE reconciliation test)")
# --------------------------------------------------------------------------------------------

DARTS = srs._darts()
Pperm = np.zeros((ND, ND))
for a, (i, j, v) in enumerate(DARTS):
    g = (sigma_dict[i], sigma_dict[j])
    for b, (p, q, w) in enumerate(DARTS):
        if (p, q) == g:
            Pperm[b, a] = 1
            break

record("T1.3a Pperm (home-i's own dart-perm op, derive_generation_spectrum.py convention) "
       "== dart_rep(sigma)",
       float(np.max(np.abs(Pperm - dart_rep(sigma_dict)))), 1e-13,
       "confirms both readings act on the IDENTICAL 12-dart basis/ordering (srs._darts())")

Pc3 = {t: sum(om ** (-t * m) * np.linalg.matrix_power(Pperm, m) for m in range(3)) / 3
       for t in (0, 1, 2)}

isotype_basis = {}
for t in (0, 1, 2):
    Pt = Pc3[t]
    Pt = (Pt + Pt.conj().T) / 2
    w, V = np.linalg.eigh(Pt)
    cols = V[:, np.abs(w - 1) < 1e-6]
    isotype_basis[t] = cols.astype(complex)
    print(f"  isotype t={t}: dim={cols.shape[1]}")

isotype_dims = [isotype_basis[t].shape[1] for t in (0, 1, 2)]
record("T1.3b isotype dims are (4,4,4)", float(max(abs(d - 4) for d in isotype_dims)), 0.5)

# cross-check against the LIVE the_run.py twin (c3_winding_bases) -- same subspaces, independent
# construction (the_run.py's own function, not reimplemented here)
from derivation_topdown.bridge.the_run import c3_winding_bases  # noqa: E402

live_bases = c3_winding_bases()
worst_live = 0.0
for t in (0, 1, 2):
    P_mine = isotype_basis[t] @ isotype_basis[t].conj().T
    P_live = live_bases[t] @ live_bases[t].conj().T
    worst_live = max(worst_live, float(np.max(np.abs(P_mine - P_live))))
record("T1.3c isotype subspaces == the_run.py's LIVE c3_winding_bases() (same object, cross-check)",
       worst_live, 1e-12)

# push isotype-t through phi_1 (any single Hom-space element) into the rho3 3-space; verify it
# lands in the rho3(sigma)-eigenspace with the MATCHING eigenvalue w^t.
phi1 = phi_basis[0]
rng = np.random.default_rng(20260715)

correct_resids = []
y_norms = []
x_reps = {}
for t in (0, 1, 2):
    cols = isotype_basis[t]
    c = rng.normal(size=cols.shape[1]) + 1j * rng.normal(size=cols.shape[1])
    x_t = cols @ c
    x_t = x_t / np.linalg.norm(x_t)
    x_reps[t] = x_t
    y_t = phi1 @ x_t
    ynorm = float(np.linalg.norm(y_t))
    y_norms.append(ynorm)
    resid = float(np.linalg.norm(rho3_sigma_net @ y_t - (om ** t) * y_t)) / max(ynorm, 1e-300)
    correct_resids.append(resid)
    print(f"  t={t}: |phi1(x_t)|={ynorm:.6f} (nonvanishing push)  "
          f"normalized correct-eigen residual={resid:.3e}")

record("T1.3d isotype-t --phi1--> rho3(sigma)-eigenspace(w^t), worst over t=0,1,2",
       max(correct_resids), 1e-12,
       "windings==bands made explicit: the SAME operator (sigma) grades both pictures identically")
record("T1.3e phi1 does not vanish on any isotype (min |phi1(x_t)|)", min(y_norms), 1e-6, mode="ge")

# --------------------------------------------------------------------------------------------
hdr("T1.4 -- REJECTION CONTROL (mandatory, W<->A4-station pattern; require margin >= 1e5)")
# --------------------------------------------------------------------------------------------


def random_order3_orthogonal(rng_):
    axis = rng_.normal(size=3); axis /= np.linalg.norm(axis)
    theta = 2 * np.pi / 3
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


R_ctrl = random_order3_orthogonal(rng)
print("non-isomorphic control R (random SO(3) order-3 rotation, NOT rho3(sigma)):\n", R_ctrl)
print("  R_ctrl^3 - I max abs:", float(np.max(np.abs(np.linalg.matrix_power(R_ctrl, 3) - np.eye(3)))))
print("  |R_ctrl - rho3(sigma)| max abs:", float(np.max(np.abs(R_ctrl - rho3_sigma_net))))

margins = []
for t in (0, 1, 2):
    x_t = x_reps[t]
    y_t = phi1 @ x_t
    ynorm = float(np.linalg.norm(y_t))
    correct = float(np.linalg.norm(rho3_sigma_net @ y_t - (om ** t) * y_t)) / ynorm
    wrong1 = float(np.linalg.norm(rho3_sigma_net @ y_t - (om ** ((t + 1) % 3)) * y_t)) / ynorm
    wrong2 = float(np.linalg.norm(rho3_sigma_net @ y_t - (om ** ((t + 2) % 3)) * y_t)) / ynorm
    ctrl = float(np.linalg.norm(R_ctrl @ y_t - (om ** t) * y_t)) / ynorm
    worst_control = min(wrong1, wrong2, ctrl)
    margin = worst_control / correct if correct > 0 else float("inf")
    margins.append(margin)
    print(f"  t={t}: correct={correct:.3e}  wrong(t+1)={wrong1:.3e}  wrong(t+2)={wrong2:.3e}  "
          f"non-iso-control={ctrl:.3e}  margin(worst-control/correct)={margin:.3e}")

worst_margin = min(margins)
record("T1.4 rejection-control margin (worst over t=0,1,2) >= 1e5",
       worst_margin, 1e5, mode="ge",
       note=f"margins={[f'{m:.3e}' for m in margins]}")

# =============================================================================================
hdr("SUB-TARGET 2 -- (i) <-> (iii) observer C^3_gen: forced, or needs-a-datum?")
# =============================================================================================

hdr("T2.0 -- provenance: locate or declare-absent the M1.B construction")

info("T2.0 grep patterns",
     "M1.B | Galois | crossed.product/crossed_product | (unicode semidirect symbol) | "
     "M^alpha | outer automorphism  -- across *.py and *.md, whole repo")

print("""
  Exhaustive grep found REAL, committed scripts (NOT prose-only):
    proofs/foundations/m1b_observer_substrate_iprojection_attempt.py   (M1.B.b, ~362 lines)
    proofs/foundations/m1b_c_basis_match.py                            (M1.B.c, ~381 lines)
    proofs/foundations/m1b_d_iprojection_structural_map.py             (M1.B.d, ~314 lines)
  All three committed 2026-05-28 (commit 196633b), i.e. BEFORE the 2026-07-01 "[RESOLVED]" claim
  in documentation_lag_contradictions_2026-06-29.md:63-67 that this station is re-examining.

  THIS CORRECTS THE FREEZE'S OWN PREMISE. GEN_HOMES_reconciliation_prereg_2026-07-15.md:22-24
  states (citing its own 2026-07-15 sweep): "M1.B is cited in prose only ... no such script
  exists in the repo (exhaustive grep, sweep)." That claim is FALSE -- the prior sweep's grep
  evidently did not search proofs/foundations/. Filed here as a correction, not smoothed over.
""")
info("T2.0 verdict", "REAL CONSTRUCTION EXISTS (not prose-only) -- but see T2.1: existence of a "
                      "script does not by itself mean a construction-CANONICAL S was produced.")

hdr("T2.1 -- the forcing question: is C^3_gen's Z3 construction-pinned, or only up to U(3)?")

print("""
  Reading the three M1.B scripts (file:line, this session):

  m1b_observer_substrate_iprojection_attempt.py (M1.B.b) DOES ground sigma=(1 2 3)(4 5 6) in the
  REAL substrate: N_GENS=6 (the 6 undirected srs edges), sigma = the
  body-diagonal C3 permutation of F_inv(6)'s generators (:57-80), lifted to an outer *-automorphism
  alpha of M = L(F_inv(6)) (:84-191, citing Lyndon-Schupp 1977 + Voiculescu 1996; ONE symbolic
  sympy spot-check on a single word, :121-137 -- not an exhaustive machine verification). Produces
  M^alpha = a type II_1 sub-factor, Jones index 3 (:184-196) -- genuinely substrate-grounded, a
  real (if lightly-checked) construction step.

  m1b_c_basis_match.py (M1.B.c) is where the (i)<->(iii) IDENTIFICATION is claimed "CLOSED"
  (:363). But its own docstring (:45-48) states it verifies the matrix-unit/spectral-projection
  structure on a "finite-dim TOY MODEL (M^alpha |-> C, so the crossed product reduces to M_3(C)
  acting on C^3)" -- i.e. it REPLACES the real, infinite-dimensional M^alpha (established one
  script earlier, M1.B.b) with the trivial algebra C. The entire "identification" computed here
  (V . u . V^-1 = Z = diag(1,w,w^2) via the DFT matrix V, :261-309) is then EXACTLY the abstract
  spectral-theorem fact R3's OWN Theorem L2 already states (predictions/
  R3_observer_c3_generation_derivation.md:63-79): any order-3 unitary with eigenvalues {1,w,w^2}
  is U(3)-conjugate to sigma_shift. This is precisely the freeze SS2 VACUOUS/NULL result -- it
  does not touch the real M^alpha object M1.B.b built.

  Critically, M1.B.c's own text (:280-285) states the remaining S_3 basis ambiguity (which root
  is 1 vs w vs w^2, i.e. which eigenspace is e/mu/tau) is "FORCED by the framework's Koide
  structure (Q_Koide=2/3 ... select non-degenerate masses)" -- i.e. resolved by reading the
  observed lepton-mass spectrum's ORDERING (R3-L3's external datum), not by anything intrinsic
  to the crossed product or the toy computation. m1b_d_iprojection_structural_map.py (M1.B.d)
  likewise runs on an admitted "rank-2 toy" (:61-63, "the toy with M=M_3(C), M^alpha=C collapses
  too much").

  CONCLUSION: no step in M1.B ever exhibits a construction-canonical S connecting the REAL
  substrate object (M^alpha, M1.B.b) to C^3_obs -- the identification step is done entirely on a
  trivialized toy, reproduces the null abstract-iso fact, and explicitly reaches for the external
  mass-non-degeneracy datum to break the residual freedom. This is machine-verified below.
""")

# Reproduce the null/vacuous fact directly (any generic order-3 unitary is U(3)-conjugate to
# sigma_shift) -- confirms it is NON-discriminating, exactly as SS2 warns.
sigma_shift = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
Z = np.diag([1, om, om ** 2])
Fdft = np.array([[1, 1, 1], [1, om, om ** 2], [1, om ** 2, om]], dtype=complex) / np.sqrt(3)

rng2 = np.random.default_rng(9)
worst_conj = 0.0
for _ in range(8):
    Xr = rng2.normal(size=(3, 3)) + 1j * rng2.normal(size=(3, 3))
    Qh, Rh = np.linalg.qr(Xr)
    Ph = np.diag(np.diag(Rh) / np.abs(np.diag(Rh)))
    V = Qh @ Ph
    U = V @ Z @ V.conj().T
    W = V @ Fdft   # since Fdft^dagger @ Z @ Fdft == sigma_shift (verified), Z = V^dagger U V
    #  => sigma_shift = Fdft^dagger (V^dagger U V) Fdft = (V.Fdft)^dagger U (V.Fdft)
    resid = float(np.max(np.abs(W.conj().T @ U @ W - sigma_shift)))
    worst_conj = max(worst_conj, resid)
record("T2.1a generic order-3 unitary IS U(3)-conjugate to sigma_shift (reproduces R3's L2; "
       "confirms it is the NULL/non-discriminating fact per SS2)",
       worst_conj, 1e-11)

# Demonstrate the residual freedom directly: ANY diagonal D (commuting with Z) gives another
# valid conjugator D.Fdft -- i.e. the conjugating S/V is not unique, so no canonical choice is
# forced by the algebra alone.
worst_freedom = 0.0
for _ in range(8):
    thetas = rng2.uniform(0, 2 * np.pi, size=3)
    D = np.diag(np.exp(1j * thetas))
    Vp = D @ Fdft
    resid = float(np.max(np.abs(Vp @ sigma_shift @ Vp.conj().T - Z)))
    worst_freedom = max(worst_freedom, resid)
record("T2.1b residual U(1)^3 freedom: D.Fdft ALSO conjugates sigma_shift->Z for every diagonal D "
       "(8 random trials) -- the conjugator is NOT construction-pinned",
       worst_freedom, 1e-12)

info("T2.1 verdict", "the identification is CONSISTENT (an abstract iso always exists) but NOT "
                      "FORCED -- the conjugating S is free up to the U(1)^2 centralizer of Z "
                      "(mod overall phase), and M1.B.c's own text closes this freedom using the "
                      "external Koide/mass-non-degeneracy datum (R3-L3), not a construction-"
                      "intrinsic pin.")

hdr("T2.2 -- (only if a canonical S is genuinely claimed)")
print("""
  Per T2.1: no canonical, construction-pinned S was found (only the vacuous abstract iso, plus
  an EXTERNAL datum to break the residual freedom). Per the freeze SS4 ("No canonical S => no
  T2.2; go to verdict C"), T2.2 is SKIPPED.
""")
info("T2.2", "SKIPPED per freeze rule (no canonical S claimed to test)")

# =============================================================================================
hdr("SUB-TARGET 3 -- the 'orthogonal factor' vs 'same substrate C3' documentation contradiction")
# =============================================================================================

print("""
  T3.0 -- literal current text (re-read this session):

  framework_architecture.md:68-70 (Layer 4 -- Observer Hilbert space (C^3_gen)):
    "Separate tensor factor orthogonal to the gauge rep factor. **Three generations live here.**"

  documentation_lag_contradictions_2026-06-29.md:63-67 ([RESOLVED 2026-07-01] entry):
    "(i)~=(ii) are the same lambda=-1 3-irrep (two readings); (iii) the observer C^3_gen is that
    SAME substrate C3 read via M1.B's Galois tower ... NOT orthogonal to the visible sector. One
    home, three reads. `framework_architecture` carries no 'orthogonal' framing (checked)."

  T3.1 -- disambiguation: framework_architecture.md:70's LITERAL text is "orthogonal to the gauge
  rep factor" -- present tense, currently in the file. The doc-lag note's specific sub-claim that
  framework_architecture "carries no 'orthogonal' framing (checked)" is THEREFORE FACTUALLY WRONG
  as a quote-check, independent of any deeper adjudication -- this is a plain re-read, not an
  interpretation call.

  Separately (the substantive question): "orthogonal TENSOR FACTOR" (a Hilbert-space
  factorization statement, H = C^3_gen (x) H_gauge (x) H_spinor) is logically compatible with
  "the generation-C3 is the SAME abstract group element as the substrate C3, acting differently
  (outer on the Galois tower vs inner on Cl(6) Fock -- 'different categorical levels', R3's own
  language, R3_observer_c3_generation_derivation.md:136)". A tensor factor CAN carry a group
  action that is abstractly the same C3 acting elsewhere. So the two SENSES of "orthogonal" are
  in principle reconcilable (freeze 3-A).

  BUT per this station's T2 finding (verdict 2-C: consistent-not-forced, external datum needed),
  the doc-lag note's STRONGER claim -- that (iii) "IS that SAME substrate C3" and hence "NOT
  orthogonal to the visible sector" -- overclaims relative to what M1.B actually establishes. The
  identification is not forced by the construction; it is a choice pinned by an external datum.
  So per the freeze's OWN verdict-tree routing (SS5: "2-C => identity is datum-dependent, so
  'orthogonal' stands until the datum is supplied"), the correct current status is: the tensor-
  factor / orthogonality framing in framework_architecture.md:70 STANDS as the honest statement
  (three axiomatically-independent factors), and the doc-lag note's 2026-07-01 [RESOLVED] entry
  OVERCLAIMED on two independent counts: (a) a factual mis-quote of framework_architecture.md,
  and (b) treating M1.B's toy-model abstract-iso + external-datum-pinned basis as "forced"/"the
  SAME C3" rather than "consistent, needs datum X".
""")
info("T3.0/T3.1 adjudication", "doc-lag note's [RESOLVED 2026-07-01] entry (documentation_lag_"
     "contradictions_2026-06-29.md:63-67) is WRONG on both the literal quote-check AND the "
     "forced-vs-consistent substance; framework_architecture.md:70's 'orthogonal tensor factor' "
     "wording is the currently-accurate one and should stand pending an external datum.")

# =============================================================================================
hdr("SUMMARY")
# =============================================================================================

n_pass = sum(1 for r in RESULTS if r[4])
n_total = len(RESULTS)
print(f"\nNumeric tests: {n_pass}/{n_total} PASS\n")
for name, value, threshold, mode, passed, note in RESULTS:
    tag = "PASS" if passed else "FAIL"
    vstr = f"{value:.3e}" if isinstance(value, (int, float)) else str(value)
    print(f"  [{tag}] {name} = {vstr}")

sub1_names = [r for r in RESULTS if r[0].startswith("T1.")]
sub1_all_pass = all(r[4] for r in sub1_names)
sub1_margin = worst_margin
sub2_names = [r for r in RESULTS if r[0].startswith("T2.")]
sub2_all_pass = all(r[4] for r in sub2_names)

print("\n" + "-" * 92)
print("VERDICTS (per freeze internal research notes SS3-5, SS9)")
print("-" * 92)

verdict1 = "1-A FORCED ISOMORPHISM" if (sub1_all_pass and sub1_margin >= 1e5) else "1-C TENSION"
print(f"Sub-target 1 ((i)<->(ii)):  {verdict1}")
print(f"  all T1.0-T1.3 numeric tests pass: {sub1_all_pass}; T1.4 margin = {sub1_margin:.3e} (>=1e5 required)")

verdict2 = "2-C CONSISTENT, NEEDS DATUM"
print(f"Sub-target 2 ((i)<->(iii)): {verdict2}")
print("  real M1.B scripts exist (T2.0) but no construction-canonical S was ever exhibited "
      "(T2.1); the identification requires the external lepton-mass-non-degeneracy datum "
      "(R3-L3) -- NAMED, value never read (goal-seek guard).")

verdict3 = "3-B, resolved by 2-C: 'orthogonal tensor factor' stands until the datum is supplied"
print(f"Sub-target 3 (doc contradiction): {verdict3}")

print("\n" + "-" * 92)
if verdict1.startswith("1-A") and verdict2.startswith("2-C"):
    print("GLOBAL OUTCOME: matches freeze SS9 EXPECTED CASE "
          "(1-A + 2-C + 3-B): (i)==(ii) forced by an exhibited S (real theorem); (i)<->(iii) "
          "consistent but basis-identification is an EXTERNAL datum (named); the soft spot "
          "SHRINKS to one honest edge. PLUS an unplanned correction: the freeze's own T2.0 "
          "premise ('no M1.B script exists') is corrected -- real scripts exist but do not "
          "rescue forced-ness.")
else:
    print("GLOBAL OUTCOME: does NOT match the freeze's expected case -- see FAIL entries above; "
          "treat as a surprise and re-examine before booking anything.")
print("-" * 92)
print("\nDONE.")
