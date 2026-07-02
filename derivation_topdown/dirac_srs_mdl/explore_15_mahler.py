"""
explore_15 — the cover's Ihara zeta as a Mahler measure; the srs spanning-tree entropy. Pure math.

Per-cell cover Ihara zeta:   log ζ_cell(u)^{-1} = 2 log(1-u^2) + M(u),  where
   M(u) = ∫_{T^3} log|det(I - uA(k) + 2u^2 I)| dk   (Mahler measure of the Bloch polynomial).
Special value:  at u=1,  I - A + 2I = 3I - A(k) = the Bloch Laplacian, so
   M(1) = ∫_{T^3} log det(3I - A(k)) dk = the spanning-tree (tree) entropy per cell  [Lyons 2005].
Midpoint BZ grid avoids k=0 and the high-symmetry band-touchings (integrable log singularities).
"""
import numpy as np, srs, math

def A_batch(K):
    A = np.zeros((len(K), 4, 4), complex)
    for i, j, v in srs.EDGES:
        ph = np.exp(2j*np.pi*(K @ np.array(v, float)))
        A[:, i, j] += ph; A[:, j, i] += np.conj(ph)
    return A

def grid(N):
    idx = (np.arange(N) + 0.5)/N
    return np.array(np.meshgrid(idx, idx, idx)).reshape(3, -1).T

def M(u, N):
    A = A_batch(grid(N))
    D = (1 + 2*u*u)*np.eye(4)[None] - u*A
    return float(np.mean(np.log(np.abs(np.linalg.det(D)))))

print("Cover Ihara zeta as a Mahler measure")
print("  M(u) = ∫_T3 log|det(I - uA + 2u^2 I)| dk   (the nontrivial part of log ζ_cell(u)^-1)\n")
for u in [0.3, 0.5, 1/math.sqrt(2), 0.8]:
    vs = [M(u, N) for N in (24, 40, 56)]
    print(f"  u = {u:.5f} :  M(u) = {vs[-1]:+.6f}    (N=24,40,56: {[round(v,5) for v in vs]})")

print("\nSpanning-tree entropy of the srs net   h = M(1) = ∫ log det(3I - A(k)) dk:")
te = [M(1.0, N) for N in (32, 48, 64, 80)]
h = te[-1]
print(f"  per cell    h_cell  = {h:.6f}    (N=32,48,64,80: {[round(v,5) for v in te]})")
print(f"  per vertex  h_vert  = {h/4:.6f}   (4 vertices/cell)")
print(f"  per edge    h_edge  = {h/6:.6f}   (6 edges/cell)")

print("\n  recognizable-constant check vs h_vert = %.6f :" % (h/4))
G = 0.915965594177; z3 = 1.202056903160; pi = math.pi
for name, val in [("log 2", math.log(2)), ("log 3", math.log(3)),
                  ("(2/pi) Catalan G", (2/pi)*G), ("(3/pi) Catalan G", (3/pi)*G),
                  ("zeta(3)/pi", z3/pi), ("(1/3) log(2^? )", math.log(2)/3 + math.log(3)/3),
                  ("log(3) - log(2)", math.log(3)-math.log(2))]:
    flag = "  <== close" if abs(val - h/4) < 5e-3 else ""
    print(f"    {name:20} = {val:.6f}{flag}")
print("\n  (honest: identification requires ~7-digit accuracy; report the number + convergence, and")
print("   only claim 'special' if a clean closed form matches within the convergence error.)")
