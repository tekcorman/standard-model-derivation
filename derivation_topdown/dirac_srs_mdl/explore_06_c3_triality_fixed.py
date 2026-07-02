"""
explore_06 — C3 triality, FIXED.  The explore_03 bug was the WRONG LINE.
sigma=(123) acts on H_1=Z^3 by M (e1->e3, e2->-e1, e3->-e2), so the sigma-FIXED k-line is
t*(1,-1,1), NOT the (1,1,1) diagonal.  The naive dart-permutation is already correct (the voltage
assignment is sigma-equivariant: v_{sigma a} = sigma_* v_a); only the line was wrong.
"""
import numpy as np, srs, cmath
om = cmath.exp(2j*np.pi/3)
sigma = {0: 0, 1: 2, 2: 3, 3: 1}                       # the 3-cycle (1 2 3)
M = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])      # sigma_* on H_1
print("sigma_* = M has order 3 ?", np.allclose(np.linalg.matrix_power(M, 3), np.eye(3)))
w, V = np.linalg.eig(M.T)
fix = V[:, np.argmin(np.abs(w - 1))].real; fix = np.round(fix/np.max(np.abs(fix)), 3)
print(f"sigma-fixed k-line (M^T eigenvector, eigenvalue 1) = t*{fix}   (= t*(1,-1,1))")

DARTS = srs._darts()
P = np.zeros((12, 12))
for a, (i, j, v) in enumerate(DARTS):
    g = (sigma[i], sigma[j])
    for b, (p, q, ww) in enumerate(DARTS):
        if (p, q) == g: P[b, a] = 1; break

Pa = {s: sum(om**(-s*m)*np.linalg.matrix_power(P, m) for m in range(3))/3 for s in (0, 1, 2)}
def basis(s):
    e, U = np.linalg.eigh(Pa[s]); return U[:, np.abs(e - 1) < 1e-6]

print("\nC3 triality along the TRUE fixed line  k = t*(1,-1,1):   |h|^2 by C3 sector")
for t in [0.0, 0.1, 0.2, 0.3, 0.45]:
    k = t*np.array([1.0, -1.0, 1.0]); B = srs.hashimoto(k)
    if not np.allclose(B@P, P@B, atol=1e-9):
        print(f"  t={t}: [B,P] != 0  (unexpected!)"); continue
    row = []
    for s in (0, 1, 2):
        Q = basis(s); Bs = Q.conj().T @ B @ Q
        row.append(f"{'1 w w2'.split()[s]}:{sorted(round(abs(e)**2, 2) for e in np.linalg.eigvals(Bs))}")
    print(f"  t={t:.2f}:   " + "   ".join(row))
print("\n  => triality resolved along the whole fixed line: the Ramanujan shell |h|^2=2 distributes")
print("     across the three C3 sectors {1,w,w2} — the C3-Fourier ('generation') structure, now clean.")
