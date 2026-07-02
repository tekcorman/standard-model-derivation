"""
The mirror: internal vs spatial cohomology, and the Hodge-SUSY hint.

Companion to the mirror-identification round (2026-06-03), banked in
`docs/scoping/mirror_intrinsic_duality_hodge_susy_2026-06-03.md`, and the
correction to lemma L4 of
`docs/theorems/theorem_grade_blind_mass_classification_2026-06-03.md`.

Findings (honest, including two corrections to the first-pass slogans):
  G1  the substrate's spatial de Rham complex carries Hodge-SUSY QM (Q=d):
      nonzero spec(delta d) == nonzero spec(d delta) (modes pair); ground states
      = harmonic = cohomology. (On K4, the srs quotient graph.)
  G2  CORRECTION: spatial b1(srs quotient = K4) = 3, NOT 9. So 'massless gauge
      = spatial harmonic' is FALSE; the spatial Hodge-SUSY ground states (3 loop
      modes) are a different (flavor/winding) sector.
  G3  the massless-gauge count is INTERNAL: the centralizer of the natural
      Pati-Salam involution J = diag(1,1,1,-1) ('lepton = 4th color') in su(4)
      is su(3)+u(1) = 9 = the unbroken SU(3)xU(1). This is Lie-algebra harmonic,
      not spatial.
  G4  CORRECTION/clarification: the fermion mirror (chirality gamma7) commutes
      with every gauge bivector, so it CANNOT gap gauge bosons. The fermion-mass
      mirror and the gauge-mass mirror are DIFFERENT operators (different faces
      of the duality), not literally one Hodge star -> unifying them is open.
"""
import numpy as np

PASS = []
def check(name, cond):
    PASS.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ----------------------------------------------------------------------------
# G1 — Hodge-SUSY on the srs quotient graph (K4): Q=d, ground states = harmonic
# ----------------------------------------------------------------------------
print("G1  Hodge-SUSY on K4 (Q=d): nonzero spectra pair; ground states = harmonic")
edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]; V, E = 4, 6
d0 = np.zeros((E, V))
for e,(a,b) in enumerate(edges): d0[e,a] = -1; d0[e,b] = 1
L0 = d0.T @ d0      # delta d on 0-forms
L1 = d0 @ d0.T      # d delta on 1-forms
nz0 = np.sort(np.round(np.linalg.eigvalsh(L0),6)); nz0 = nz0[np.abs(nz0)>1e-9]
nz1 = np.sort(np.round(np.linalg.eigvalsh(L1),6)); nz1 = nz1[np.abs(nz1)>1e-9]
check("nonzero spec(delta d) == nonzero spec(d delta)  (SUSY pairing)",
      np.allclose(nz0, nz1))


# ----------------------------------------------------------------------------
# G2 — CORRECTION: spatial b1 = 3, not 9
# ----------------------------------------------------------------------------
print("G2  spatial b1(srs quotient = K4) = 3  (NOT 9)")
b1 = int(np.sum(np.abs(np.linalg.eigvalsh(L1)) < 1e-9))
check("spatial b1 = 3", b1 == 3)
check("spatial b1 != massless-gauge count (9)", b1 != 9)


# ----------------------------------------------------------------------------
# G3 — INTERNAL: massless gauge = centralizer of the PS mirror involution
# ----------------------------------------------------------------------------
print("G3  internal centralizer of J=diag(1,1,1,-1) in su(4) = 9 = su(3)+u(1)")
def su_n_basis(n):
    B = []
    for i in range(n):
        for j in range(i+1, n):
            S = np.zeros((n,n),complex); S[i,j]=S[j,i]=1; B.append(S)
            A = np.zeros((n,n),complex); A[i,j]=-1j; A[j,i]=1j; B.append(A)
    for k in range(1, n):
        D = np.zeros((n,n),complex)
        for m in range(k): D[m,m]=1
        D[k,k] = -k; B.append(D/np.sqrt(k*(k+1)))
    return B
su4 = su_n_basis(4)
J = np.diag([1,1,1,-1]).astype(complex)
cent = sum(1 for X in su4 if np.allclose(X@J - J@X, 0))
check("dim su(4) = 15", len(su4) == 15)
check("centralizer(J) = 9  (unbroken SU(3)xU(1))", cent == 9)


# ----------------------------------------------------------------------------
# G4 — CORRECTION: gamma7 (fermion mirror) is central on gauge bivectors
# ----------------------------------------------------------------------------
print("G4  fermion mirror gamma7 commutes with all gauge bivectors (=> different op)")
# Cl(6) via Jordan-Wigner on 3 qubits (8-dim). 6 Majoranas c_1..c_6.
I2 = np.eye(2,dtype=complex)
X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.array([[1,0],[0,-1]],complex)
def kron(*ops):
    out = np.array([[1]],complex)
    for o in ops: out = np.kron(out, o)
    return out
c = []
paulis = [X, Y]
for k in range(3):
    for p in paulis:
        ops = [Z]*k + [p] + [I2]*(2-k)
        c.append(kron(*ops))
gamma7 = c[0]@c[1]@c[2]@c[3]@c[4]@c[5]      # volume element (chirality)
bivs = [c[i]@c[j] for i in range(6) for j in range(i+1,6)]   # 15 bivectors = so(6)
all_commute = all(np.allclose(gamma7@B - B@gamma7, 0) for B in bivs)
check("len(bivectors) = 15", len(bivs) == 15)
check("gamma7 commutes with every bivector (cannot gap gauge bosons)", all_commute)


# ----------------------------------------------------------------------------
print()
print("SYNTHESIS: two cohomologies, two faces of the mirror.")
print("  spatial  (graph de Rham, b1=3): carries Hodge-SUSY; ground states = loop/flavor sector.")
print("  internal (Cl(6)/PS, centralizer): carries the gauge-mass split; massless = 9.")
print("  fermion mirror = chirality gamma7 (central on bivectors); gauge mirror = inner PS")
print("  involution + EW vev. Same DUALITY motif, NOT shown to be one operator -> open.")
print()
print(f"{sum(PASS)}/{len(PASS)} gates PASS")
if not all(PASS):
    raise SystemExit("mirror internal-vs-spatial probe: FAIL")
