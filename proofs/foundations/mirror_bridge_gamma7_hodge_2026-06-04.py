"""
The mirror bridge: does the Clifford-form correspondence make the fermion mirror
(chirality gamma7) and the boson mirror ONE operator?

Companion to the bridge round (2026-06-04). Result is a SPLIT verdict:

  CONFIRMED  gamma7 (volume element) is literally ONE operator with two faces:
             left-multiplication = the Hodge star (grade k -> 6-k) on forms,
             and chirality (S = S+ (+) S-) on spinors. So 'fermion chirality' and
             'form Hodge-duality' ARE one volume-element duality.

  REFUTED    that duality is NOT the boson-MASS mirror. gamma7 is adjoint-central
             (commutes with all 15 gauge bivectors) => it gaps NO gauge boson.
             The gauge mass comes from INNER symmetry breaking (a bivector
             involution, centralizer < 15) -- a different operator. So the
             fermion-mass mirror and the boson-mass mirror genuinely DIFFER; the
             'single mirror' of the grade-blind classification is a shared MOTIF,
             not one operator.

This downgrades the strong 'one mirror, both grades' reading of
`docs/theorems/theorem_grade_blind_mass_classification_2026-06-03.md` (the
classification/enumeration stands; the single-operator unification does not).
"""
import numpy as np
import itertools

PASS = []
def check(name, cond):
    PASS.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# Cl(6) via Jordan-Wigner on 3 qubits (8-dim spinor module S)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]], complex)
def kron(*ops):
    r = np.array([[1]], complex)
    for o in ops: r = np.kron(r, o)
    return r
c = [kron(*([Z] * k + [p] + [I2] * (2 - k))) for k in range(3) for p in (X, Y)]
# volume element gamma7, phase fixed so gamma7^2 = I (eigenvalues +-1)
g7 = c[0] @ c[1] @ c[2] @ c[3] @ c[4] @ c[5]
for ph in (1, 1j, -1, -1j):
    if np.allclose((ph * g7) @ (ph * g7), np.eye(8)):
        g7 = ph * g7
        break

def monomial(S):
    M = np.eye(8, dtype=complex)
    for i in S: M = M @ c[i]
    return M
def grades_present(M):
    gs = set()
    for r in range(7):
        for S in itertools.combinations(range(6), r):
            if abs(np.trace(monomial(S).conj().T @ M) / 8) > 1e-9:
                gs.add(len(S))
    return gs

# G1 — FACE 1 (boson/form): left-mult by gamma7 = Hodge star (grade reversal)
print("G1  FACE 1: L_gamma7 reverses Clifford grade k -> 6-k (= Hodge star)")
hodge = all(grades_present(g7 @ monomial(S)) == {6 - r}
            for r in range(7) for S in itertools.combinations(range(6), r))
check("L_gamma7 sends every grade-k monomial to grade (6-k)", hodge)

# G2 — FACE 2 (fermion/spinor): gamma7 = chirality, S = S+ (+) S-
print("G2  FACE 2: gamma7 on the spinor module = chirality (4 + 4)")
ev = np.round(np.linalg.eigvalsh(g7), 6)
check("gamma7 eigenvalues are +-1 with multiplicity 4,4",
      int(np.sum(ev > 0)) == 4 and int(np.sum(ev < 0)) == 4)

# G3 — BRIDGE: it is the SAME matrix gamma7 carrying both faces
print("G3  BRIDGE: one element gamma7 = Hodge star (forms) AND chirality (spinors)")
check("the two faces are the same operator (gamma7)", hodge and len(ev) == 8)

# G4 — gamma7 is adjoint-central: gaps NO gauge boson
print("G4  gamma7 adjoint action on the 15 gauge bivectors")
bivs = [c[i] @ c[j] for i in range(6) for j in range(i + 1, 6)]
g7_cent = sum(np.allclose(g7 @ B - B @ g7, 0) for B in bivs)
check("gamma7 commutes with ALL 15 bivectors (gaps nothing)", g7_cent == 15)

# G5 — inner bivector involution DOES gap (centralizer < 15): boson mass != gamma7
print("G5  an inner gauge bivector gaps (centralizer < 15) -> boson mass is a DIFFERENT op")
b0_cent = sum(np.allclose(bivs[0] @ B - B @ bivs[0], 0) for B in bivs)
check("a gauge bivector centralizer is < 15 (inner breaking gaps)", b0_cent < 15)

print()
print("VERDICT: gamma7 == Hodge-* CONFIRMED (one volume-element duality, spinors<->forms);")
print("         boson MASS mirror is INNER breaking, NOT gamma7 -> 'single mirror' REFUTED")
print("         as one operator. Classification stands; the unification is a shared motif.")
print()
print(f"{sum(PASS)}/{len(PASS)} gates PASS")
if not all(PASS):
    raise SystemExit("mirror bridge probe: FAIL")
