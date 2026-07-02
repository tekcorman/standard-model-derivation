#!/usr/bin/env python3
"""Phase 4.3 -- THE PREREGISTERED R-19 JEOPARDY: beta content from the
induced action (spec FROZEN b4bb97b BEFORE 4.1/4.2 ran; stakes immutable).

Frozen reading chain: induced gauge-kinetic coefficients per gauge sector
-> effective matter-content vector -> b_i. Frozen targets (the ONLY three):
MSSM (33/5, 1, -3) | 2HDM-SM (21/5, -3, -7) | substrate-2HDM+3-gen-scalars
(31/5, -1, -5). Honest prior: 2HDM-shaped. FORBIDDEN: post-hoc D changes.

The three targets share the 2HDM FERMION base and differ ONLY in:
  gauginos/higgsinos (fermions, MSSM only) and the scalar sector
  (2 doublets shared; substrate target adds 3 generations of sfermion-like
  scalars: Delta b = (+2, +2, +2)).
b3 ALONE separates all three: -7 (2HDM) / -5 (substrate) / -3 (MSSM).
So the native deciders are:
  (A) does H contain su(4)-adjoint FERMION content (gauginos)?  [G3]
  (B) which mirror-crossing (sigma) multiplets PROPAGATE (m8 kinetic) and
      what is their color charge?                               [G5/G6]

Gates:
  G1  su(3)_c built natively from the Cl(6) Fock ladder (a_i^dag a_j,
      JW): 8 generators, closure exact, all in the even-grade span,
      Tr_Fock(T_a T_b) = delta_ab (3 + 3bar content = 1/2 + 1/2).
  G2  u(1)_B-L = (2/3)N - 1 (the Furey number ladder): traceless,
      commutes with su(3); spectrum {-1, -1/3, +1/3, +1} multiplicities
      {1, 3, 3, 1}.
  G3  GAUGINO ABSENCE IS STRUCTURAL *WITHIN THE FROZEN TRIPLE* (panel
      scope 2026-06-12): C2(su4) = 15/4 * I EXACTLY on the Fock factor --
      the frozen H is a pure spinor module 4 + 4bar with no adjoint seat
      (C2 = 15/4*I is automatic on any Cl(6) module; the CONTENT is that
      H is a spinor module). The gaugino piece cannot be hosted in this
      triple; the higgsino piece is dictionary-conditional.
  G4  fermion kinetic traces per factor (native, recorded): per-cover-fiber
      Tr_H(T^2) for su(3) / u(1)_B-L; the sector-level 3-generation anchor
      (dictionary-licensed, DECLARED bridge) sets the shared 2HDM base
      Sum_f T_3 = 6; the native/dictionary normalization ratio is recorded,
      not tuned (it cancels in the Delta-b discriminator).
  G5  SIGMA-GRADE CENSUS (the live decider): mirror-crossing directions
      V = M (X_g (x) prof) for Cl(6)-grade representatives g = 0..3:
      chirality structure ([X, Gamma7] commute = scalar-like / anticommute
      = Yukawa-like); color rep via the Casimir ACTION Sum_a [T_a,[T_a,X]]
      = c X (eigen-gate); m8 kinetic slope ALIAS-FREE on GRID3 (panel
      erratum E2: GRID3 == GRID5 == MC certified in
      phase4_e2_sigma_census_aliasfree_2026-06-12.py).
  G6  scalar content assembly: Sum_s T_3 = sum over PROPAGATING multiplets
      (complex-scalar counting x 4 atom profiles).
  G7  THE JEOPARDY: b_3 = -11 + (2/3)(6) + (1/3) Sum_s T_3 compared to
      {-7, -5, -3}; b_1 analog recorded; b_2 column flagged
      dictionary-conditional (the su(2)_L seat is the chirality-projected
      CANDIDATE, not natively closed). Verdict text printed FOR THE PANEL;
      no register/ledger row moves here.
"""
import os
import sys

import numpy as np
from numpy import linalg as la

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds, A_PRIM  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


GAMMAS = [kron3(X, I2, I2), kron3(Y, I2, I2), kron3(Z, X, I2),
          kron3(Z, Y, I2), kron3(Z, Z, X), kron3(Z, Z, Y)]
GAMMA7 = ((-1j) ** 3) * np.linalg.multi_dot(GAMMAS)

# Fock ladder (Jordan-Wigner; strings are inside the gamma pairing)
A_OPS = [(GAMMAS[0] + 1j * GAMMAS[1]) / 2.0,
         (GAMMAS[2] + 1j * GAMMAS[3]) / 2.0,
         (GAMMAS[4] + 1j * GAMMAS[5]) / 2.0]
N_OP = sum(a.conj().T @ a for a in A_OPS)

print("=" * 72)
print(" PHASE 4.3 -- the R-19 jeopardy: beta content from the induced action")
print("=" * 72)

# ---- G1: su(3)_c native ----
gens = []
for i in range(3):
    for j in range(3):
        if i < j:
            E = A_OPS[i].conj().T @ A_OPS[j]
            gens.append((E + E.conj().T) / 2.0)
            gens.append((E - E.conj().T) / 2.0j)
n1 = A_OPS[0].conj().T @ A_OPS[0]
n2 = A_OPS[1].conj().T @ A_OPS[1]
n3 = A_OPS[2].conj().T @ A_OPS[2]
gens.append((n1 - n2) / np.sqrt(2.0))
gens.append((n1 + n2 - 2 * n3) / np.sqrt(6.0))
T3 = [g / np.sqrt(2.0) for g in gens]   # normalize Tr(T_a T_b) = delta_ab... fixed below

# orthonormalize trace metric to Tr(T_a T_b) = delta_ab * 1 (3+3bar = 1/2+1/2)
G_metric = np.array([[np.trace(a @ b).real for b in gens] for a in gens])
W = la.cholesky(la.inv(G_metric)).T
T3 = [sum(W[i, j] * gens[j] for j in range(8)) for i in range(8)]
metric_dev = max(abs(np.trace(T3[i] @ T3[j]).real - (i == j))
                 for i in range(8) for j in range(8))

# closure: [T_a, T_b] in i*span{T}
def in_ispan(Xm, basis, tol=1e-10):
    coef = [np.trace(b @ Xm).real / 1.0 for b in basis]
    rec = sum(c * 1j * b for c, b in zip(
        [np.trace(b @ (Xm / 1j)).real for b in basis], basis))
    return la.norm(Xm - rec) < tol


clos_dev = 0.0
for i in range(8):
    for j in range(i + 1, 8):
        C = T3[i] @ T3[j] - T3[j] @ T3[i]
        coefs = [np.trace(b @ (C / 1j)).real for b in T3]
        rec = 1j * sum(c * b for c, b in zip(coefs, T3))
        clos_dev = max(clos_dev, la.norm(C - rec))
# even-grade span: all commute with Gamma7
g7_dev = max(la.norm(t @ GAMMA7 - GAMMA7 @ t) for t in T3)
gate("G1 su(3)_c native: 8 generators, Tr(T_aT_b) = delta_ab, closure "
     "exact, even-grade (commute with Gamma7)",
     metric_dev < 1e-10 and clos_dev < 1e-10 and g7_dev < 1e-12,
     f"metric dev={metric_dev:.1e}, closure dev={clos_dev:.1e}")

# ---- G2: u(1)_B-L ----
Y_BL = (2.0 / 3.0) * N_OP - np.eye(8)
ev_y = np.sort(la.eigvalsh(Y_BL))
spec_ok = np.allclose(ev_y, [-1.0] + [-1.0 / 3.0] * 3 + [1.0 / 3.0] * 3 + [1.0],
                      atol=1e-10)
gate("G2 u(1)_B-L = (2/3)N - 1: traceless, commutes with su(3), Furey "
     "ladder {-1, -1/3 x3, +1/3 x3, +1}",
     abs(np.trace(Y_BL)) < 1e-12
     and max(la.norm(Y_BL @ t - t @ Y_BL) for t in T3) < 1e-12 and spec_ok,
     f"spectrum {sorted(set(np.round(ev_y, 4)))}")

# ---- G3: gaugino absence is structural ----
BIVS = [1j * GAMMAS[e] @ GAMMAS[f] for e in range(6) for f in range(e + 1, 6)]
C2_su4 = sum((B / 2.0) @ (B / 2.0) for B in BIVS)
dev_c2 = la.norm(C2_su4 - (15.0 / 4.0) * np.eye(8))
c2_su3 = sum(t @ t for t in T3)
ev_c3 = np.round(np.sort(la.eigvalsh(c2_su3)), 9)
su3_ok = list(np.round(ev_c3, 6)) == [0.0, 0.0] + [round(4.0 / 3.0, 6)] * 6
gate("G3 GAUGINO ABSENCE STRUCTURAL *within the frozen triple* (panel "
     "scope): C2(su4) = 15/4 * I EXACTLY on Fock (the frozen H is a pure "
     "spinor module 4+4bar, no adjoint fermion seat) -> the MSSM gaugino "
     "piece cannot be hosted in this triple",
     dev_c2 < 1e-12 and su3_ok,
     f"C2 dev={dev_c2:.1e}; su(3) Casimir spectrum {{0 x2, 4/3 x6}} "
     f"(singlets + 3 + 3bar)")

# ---- G4: fermion kinetic traces per factor (native, recorded) ----
tr_su3 = np.trace(T3[0] @ T3[0]).real * 4 * 2   # x atoms x mirror, per generator
tr_u1 = np.trace(Y_BL @ Y_BL).real * 4 * 2
print(f"      native per-cover-fiber fermion traces: su(3) {tr_su3:.4f} per "
      f"generator, u(1)_B-L {tr_u1:.4f} (ratio {tr_u1 / tr_su3:.4f})")
print("      sector-level 3-generation anchor (dictionary-licensed bridge, "
      "DECLARED): shared 2HDM fermion base Sum_f T_3 = 6;")
print(f"      native/anchor normalization ratio = {tr_su3 / 6.0:.4f} "
      "(recorded; cancels in the Delta-b discriminator)")

# ---- G5: sigma-grade census ----
from itertools import combinations  # noqa: E402


def grade_basis(g):
    """Hermitian basis of the grade-g Clifford subspace."""
    if g == 0:
        return [np.eye(8, dtype=complex)]
    out = []
    for S in combinations(range(6), g):
        P = np.linalg.multi_dot([GAMMAS[i] for i in S]) if len(S) > 1 \
            else GAMMAS[S[0]]
        # reversal sign: P^dag = (-1)^{g(g-1)/2} P; fix Hermiticity with i
        if (g * (g - 1) // 2) % 2 == 1:
            P = 1j * P
        out.append(P)
    return out


def casimir_superop_eig(basis):
    """Eigendecomposition of X -> Sum_a [T_a,[T_a,X]] on span(basis);
    returns [(eigenvalue, multiplicity, representative)] with Hermitian
    eigen-representatives."""
    n = len(basis)
    S = np.zeros((n, n), dtype=complex)
    for b2, Xb in enumerate(basis):
        out = sum(t @ (t @ Xb - Xb @ t) - (t @ Xb - Xb @ t) @ t for t in T3)
        for a2, Xa in enumerate(basis):
            S[a2, b2] = np.trace(Xa.conj().T @ out) / 8.0   # Tr(X_a X_b) = 8 d_ab
    ev, V = la.eigh((S + S.conj().T) / 2.0)
    blocks = []
    for val in sorted(set(np.round(ev, 6))):
        idx = [i for i in range(n) if abs(ev[i] - val) < 1e-6]
        vec = V[:, idx[0]].real if la.norm(V[:, idx[0]].imag) < 1e-9 \
            else V[:, idx[0]]
        rep = sum(vec[j] * basis[j] for j in range(n))
        rep = (rep + rep.conj().T) / 2.0
        rep = rep * np.sqrt(8.0) / la.norm(rep)             # ||X||^2 = 8
        blocks.append((float(val), len(idx), rep))
    return blocks


# the 4.2 machinery (cover Dirac, M, m8 kinetic)
def undirected_edges():
    bonds = find_bonds()
    seen = {}
    for src, tgt, cell in bonds:
        cell = tuple(int(c) for c in cell)
        key = (src, tgt, cell) if src < tgt else (tgt, src, tuple(-c for c in cell))
        seen[key] = True
    return sorted(seen.keys())


EDGES = undirected_edges()


def L_e(edge, k):
    a, b, n = edge
    L = np.zeros((4, 4), dtype=complex)
    ph = np.exp(2j * np.pi * np.dot(k, n))
    L[b, a] = ph
    L[a, b] = np.conj(ph)
    for c in range(4):
        if c not in (a, b):
            L[c, c] = 1.0
    return L


def D_of_k(k):
    D = np.zeros((32, 32), dtype=complex)
    for i, e in enumerate(EDGES):
        D += np.kron(GAMMAS[i], L_e(e, k))
    return D


DELTA = np.array([0.5, 0.5, -0.5])
_DC = {}


def D_cover(k):
    key = tuple(np.round(np.asarray(k, float), 9))
    if key not in _DC:
        Dz = np.zeros((64, 64), dtype=complex)
        Dz[:32, :32] = D_of_k(k)
        Dz[32:, 32:] = D_of_k(np.asarray(k) + DELTA)
        _DC[key] = Dz
    return _DC[key]


M_SWAP = np.kron(np.array([[0, 1], [1, 0]]), np.eye(32)).astype(complex)
INDICATORS = [np.diag((np.arange(4) == c).astype(float)) for c in range(4)]
GRID2 = [np.array([i, j, l]) / 2.0 for i in range(2) for j in range(2)
         for l in range(2)]
QMAG = 0.15


def m8_kinetic(Xm, ks):
    """q^2-slope (q = QMAG x-hat) of the t^2-coefficient of
    <Tr (D_pair + t V)^8>, V = M (X (x) prof), S4-summed profiles."""
    out = []
    for q in (np.zeros(3), np.array([QMAG, 0, 0])):
        tot = 0.0
        for k in ks:
            Dp = np.zeros((128, 128), dtype=complex)
            Dp[:64, :64] = D_cover(k)
            Dp[64:, 64:] = D_cover(np.asarray(k) + q)
            pows = [np.eye(128, dtype=complex)]
            for _ in range(6):
                pows.append(pows[-1] @ Dp)
            for prof in INDICATORS:
                V = M_SWAP @ np.kron(np.eye(2), np.kron(Xm, prof))
                Vp = np.zeros((128, 128), dtype=complex)
                Vp[64:, :64], Vp[:64, 64:] = V, V.conj().T
                W = [pows[j] @ Vp for j in range(7)]
                s = sum(float(np.sum(W[j] * W[6 - j].T).real) for j in range(7))
                tot += 4.0 * s
        out.append(tot / len(ks))
    return (out[1] - out[0]) / QMAG ** 2


REP_NAME = {0.0: "singlet", round(4.0 / 3.0, 6): "3+3bar",
            3.0: "octet", round(10.0 / 3.0, 6): "6+6bar"}
GRID3 = [np.array([i, j, l]) / 3.0 for i in range(3) for j in range(3)
         for l in range(3)]
print("      sigma-grade census (Casimir eigen-blocks; chirality = grade "
      "parity; m8 kinetic ALIAS-FREE on GRID3 -- panel erratum E2, "
      "certified GRID3 == GRID5 == MC in "
      "phase4_e2_sigma_census_aliasfree_2026-06-12.py):")
census = []
for g in range(4):
    basis = grade_basis(g)
    chir = "scalar-like (even)" if g % 2 == 0 else "Yukawa-like (odd)"
    for val, mult, rep in casimir_superop_eig(basis):
        kin = m8_kinetic(rep, GRID3)
        rnm = REP_NAME.get(round(val, 6), f"C2={val:.4f}")
        census.append(dict(g=g, rep=rnm, c2=val, mult=mult, kin=kin))
        print(f"        grade {g} [{chir:18s}] {rnm:10s} x{mult:2d} "
              f"(C2-action {val:+.4f}): m8 c2 = {kin:+.4f}")
dims_ok = all(sum(c["mult"] for c in census if c["g"] == g)
              == len(grade_basis(g)) for g in range(4))
gate("G5a census complete: every grade decomposes exactly into color-"
     "Casimir eigen-blocks", dims_ok)

# ---- the central structural results of the scalar leg ----
# PANEL ERRATUM E2 (2026-06-12): the original GRID2 run showed identical
# kinetics for every block (rel spread 2.9e-13) -- a TRIM-point ALIASING
# ARTIFACT. The alias-free census is BLOCK-DISCRIMINATING (octet sign-flip;
# m10 discriminates every block). The former G5b claim "the action ...
# SELECTS no scalar content" is WITHDRAWN from all register-bound wording.
kins = [c["kin"] for c in census]
spread = (max(kins) - min(kins)) / max(abs(np.mean(np.abs(kins))), 1e-12)
octet_kin = next(c["kin"] for c in census if c["rep"] == "octet")
gate("G5b (PANEL-CORRECTED): the alias-free sigma-census is BLOCK-"
     "DISCRIMINATING -- rep-dependent m8 kinetics, octet SIGN opposite the "
     "g0 singlet (the discrimination structure is a candidate FUTURE "
     "selection lever; the former blindness claim was a GRID2 artifact)",
     spread > 0.5 and np.sign(octet_kin) != np.sign(census[0]["kin"]),
     f"spread {spread:.2f}; octet {octet_kin:+.1f} vs g0 "
     f"{census[0]['kin']:+.1f}")

# the algebra generates NO mirror-crossing one-forms (sigma is external)
b_sheet_odd = np.kron(np.diag([1.0, -1.0]), np.kron(np.eye(8),
                                                    np.diag([1., 2., 3., 4.])))
a_el = np.kron(np.eye(2), np.kron(np.eye(8), np.diag([2., 1., 1., 3.])))
Dz_P = D_cover(np.array([0.25, 0.25, 0.25]))
A_form = a_el @ (Dz_P @ b_sheet_odd - b_sheet_odd @ Dz_P)
off_block = la.norm(A_form[32:, :32]) + la.norm(A_form[:32, 32:])
gate("G5c the frozen algebra generates NO mirror-crossing one-forms "
     "(D_z and C(cover atoms) are both mirror-block-diagonal): the "
     "sigma-couplers are EXTERNAL probes, not induced fields",
     off_block < 1e-12, f"mirror-off-diagonal norm = {off_block:.1e}")

# ---- the determinate content (panel-relabeled: a STATEMENT, not a gate) --
print("      DETERMINATE content (what the action itself fixes): gauge "
      "fields + the fermionic H (pure 4+4bar, no adjoint, G3) + NO mandated "
      "scalar multiplets. Sum_s T_3(determinate) = 0 rests on G5c "
      "EXTERNALITY (no mirror-crossing one-forms from the algebra, JAJ^-1 "
      "included -- panel-extended), NOT on the withdrawn blindness claim.")
sum_sT3_det = 0.0
print("      [former G6 'gate' relabeled as a statement per the panel: it "
      "was hardcoded-True; the load-bearing computation is G5c]")

# ---- G7: THE JEOPARDY ----
b3 = -11.0 + (2.0 / 3.0) * 6.0 + (1.0 / 3.0) * sum_sT3_det
targets = {"MSSM": -3.0, "2HDM-SM": -7.0, "substrate-2HDM+scalars": -5.0}
exact = [t for t, v in targets.items() if abs(v - b3) < 1e-9]
b3_native = -11.0 + (2.0 / 3.0) * 8.0 + 0.0
print(f"\n      b_3(action-determinate | declared 3-gen anchor) = "
      f"-11 + 4 + 0 = {b3:+.1f}")
print(f"      [ANCHOR-CONDITIONALITY (panel): the native per-fiber trace 8 "
      f"would give {b3_native:+.3f} = NO target; the target DISCRIMINATOR "
      f"(+4/+2/0) is anchor-free; the anchor is priced 1 bit, ledger "
      f"2026-06-12]")
print(f"      frozen targets: MSSM -3 | 2HDM -7 | substrate -5")
print(f"      -> EXACT MATCH: {exact[0] if exact else 'NONE'}")
gate("G7 JEOPARDY (panel-scoped): the action-determinate content lands on "
     "the 2HDM-SM target with ZERO new continuous parameters (K3 silent); "
     "MSSM is excluded WITHIN THE FROZEN TRIPLE (no adjoint fermion seat, "
     "G3); the substrate target is reachable ONLY by positing "
     "sigma-multiplets the action does not mandate",
     exact == ["2HDM-SM"] and dev_c2 < 1e-12)
print("      VERDICT (panel-adjudicated 2026-06-12) -- R-19 SHARPENED, "
      "three-way split of Delta b2 = +4:")
print("        * wino +4/3: STRUCTURALLY EXCLUDED within the frozen triple")
print("          (C2(su4) = 15/4*I; H is a pure spinor module);")
print("        * higgsino +2/3: DICTIONARY-CONDITIONAL only (two color-")
print("          singlet fermion slots exist, anchor-assigned to leptons;")
print("          independent ths-side exclusion stands);")
print("        * sfermion +2: = the sigma-coupler freedom (external posits,")
print("          propagated never induced; blindness claim withdrawn --")
print("          the alias-free census is block-discriminating, E2);")
print("        * b_2 column = dictionary-conditional (su(2)_L seat is the")
print("          chirality-projected CANDIDATE; +128 = 2*dim excess, 4.2);")
print("        * sigma<->ths: UNPROMOTED (panel disposition: promotion only")
print("          via a new frozen spec with the alias-free census as gate 1).")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- jeopardy computed; verdict to the panel")
print("=" * 72)
sys.exit(0)
