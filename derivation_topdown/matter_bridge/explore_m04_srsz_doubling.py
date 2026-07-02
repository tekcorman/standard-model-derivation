"""
explore_m04 — introduce srs-z (the mirror copy) and derive the coupling. PURE MATH, walled (no physics).

Result (theorem): a single 3D chiral spinor (one srs copy, C^2 = Cl(3) = the 3 Pauli) admits NO
gap-opening term — no 2x2 matrix anticommutes with all three Pauli. A spectral gap REQUIRES doubling
to C^4 (Cl(4)) = srs (+) srs-z (the two chirality halves); the gap-opening term is then the 4th gamma —
the off-diagonal inter-copy coupling. Structure forced; strength free. Geometrically, srs-z = the
complex-conjugate (mirror) net, with opposite Berry/Weyl charge.
"""
import numpy as np, math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]

# (1) one copy (C^2): is there ANY gap-opening term (anticommuting with all 3 Pauli)?
print("(1) Single 3D chiral spinor C^2 (Cl(3) = the 3 Pauli):")
basis = [np.eye(2), s[0], s[1], s[2]]
rows = []
for B in basis:
    rows.append(np.concatenate([(B@s[a] + s[a]@B).flatten() for a in range(3)]))
Amat = np.array(rows).T
nullity = Amat.shape[1] - np.linalg.matrix_rank(Amat)
print(f"    dim{{ M (2x2) : {{M, σ^a}}=0 ∀a }} = {nullity}  =>  NO nonzero gap term: one copy is forced gapless.")

# (2) doubling to C^4 = Cl(4): the 4th gamma is the gap-opening term
I2 = np.eye(2); kr = lambda a, b: np.kron(a, b)
g = [kr(s[0], s[0]), kr(s[0], s[1]), kr(s[0], s[2]), kr(s[1], I2)]   # g1,g2,g3 spatial; g4 the new one
g5 = g[0] @ g[1] @ g[2] @ g[3]
print("\n(2) Doubling C^2 -> C^4 (Cl(4)) = srs (+) srs-z (the two chirality halves):")
print(f"    γ^4 exists, anticommutes with γ^1,γ^2,γ^3 ? {all(np.allclose(g[3]@g[a]+g[a]@g[3], 0) for a in range(3))}")
print(f"    => D = Σ_{{a=1,2,3}} γ^a p_a + m·γ^4  has  D² = (Σ p_a²) + m²  : a spectral gap of magnitude m.")
w, V = np.linalg.eigh(g5)                                  # chirality eigenbasis (srs/srs-z split)
g4c = np.abs(np.round(V.conj().T @ g[3] @ V, 6))
offdiag_only = np.allclose(g4c[:2, :2], 0) and np.allclose(g4c[2:, 2:], 0)
print(f"    γ^4 in the chirality basis is purely OFF-DIAGONAL (connects the two halves) ? {offdiag_only}")
print(f"    => the gap term γ^4 IS the srs<->srs-z coupling.")

# (3) geometric instantiation: srs-z = conj(srs net), opposite Berry/Weyl charge
def chern_sphere(Afun, k0, band=0, eps=0.04, N=20):
    k0 = np.array(k0, float); th = np.linspace(.02, math.pi-.02, N); ph = np.linspace(0, 2*math.pi, N, endpoint=False)
    U = np.empty((N, N), object)
    for a in range(N):
        for b in range(N):
            kk = k0 + eps*np.array([math.sin(th[a])*math.cos(ph[b]), math.sin(th[a])*math.sin(ph[b]), math.cos(th[a])])
            U[a, b] = np.linalg.eigh(Afun(kk))[1][:, band]
    F = 0.0
    for a in range(N-1):
        for b in range(N):
            bn = (b+1) % N
            F += np.angle(np.vdot(U[a, b], U[a, bn])*np.vdot(U[a, bn], U[a+1, bn])*np.vdot(U[a+1, bn], U[a+1, b])*np.vdot(U[a+1, b], U[a, b]))
    return F/(2*math.pi)
cz_srs = chern_sphere(lambda k: srs.adjacency(k), (0, 0, 0))
cz_srsz = chern_sphere(lambda k: np.conj(srs.adjacency(k)), (0, 0, 0))
print(f"\n(3) Geometric srs-z = conj(srs net): Weyl charge at Γ  srs = {cz_srs:+.2f},  srs-z = {cz_srsz:+.2f}  (opposite).")

print("\n--- THEOREM (matter bridge, walled) ---")
print("  One srs copy (C^2) admits NO gap-opening term: it is forced gapless. A spectral gap REQUIRES the")
print("  doubling to C^4 (Cl(4)) = srs (+) srs-z; the gap term is the 4th gamma — the off-diagonal inter-")
print("  copy coupling, and geometrically srs-z is the mirror (opposite Weyl charge) copy.")
print("  STRUCTURE (gap term = 4th gamma = srs<->srs-z coupling) is FORCED by the doubling + chirality;")
print("  the STRENGTH m is a FREE parameter, not fixed by the geometry.")
