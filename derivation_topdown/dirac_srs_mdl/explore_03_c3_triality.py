"""
explore_03 — C3 triality along the diagonal: where the three A4 3-irreps go. Pure math.
Along k=(t,t,t) the residual symmetry is C3 = <(123)>; each A4 3-irrep restricts to 1+w+w^2.
"""
import numpy as np, srs, cmath
om = cmath.exp(2j*np.pi/3)
sigma = {0: 0, 1: 2, 2: 3, 3: 1}          # the 3-cycle (1 2 3), fixes vertex 0 and the diagonal
DARTS = srs._darts(); n = 12

C3 = np.zeros((n, n))
for d, (i, j, v) in enumerate(DARTS):
    g = (sigma[i], sigma[j])
    for f, (a, b, w) in enumerate(DARTS):
        if (a, b) == g: C3[f, d] = 1; break
print("C3 dart-permutation has order 3 ?", np.allclose(np.linalg.matrix_power(C3, 3), np.eye(n)))

Pa = {a: sum(om**(-a*m)*np.linalg.matrix_power(C3, m) for m in range(3))/3 for a in (0, 1, 2)}
def sector_basis(a):
    w, V = np.linalg.eigh(Pa[a]); return V[:, np.abs(w-1) < 1e-6]

print("\nC3 triality along the diagonal  k=(t,t,t):  Hashimoto |h|^2 resolved by C3 sector")
for t in [0.0, 0.15, 0.25, 0.40]:
    B = srs.hashimoto((t, t, t))
    if not np.allclose(B@C3, C3@B):
        print(f"  t={t}: [B,C3] != 0 (off the symmetric diagonal) — skip"); continue
    print(f"  t={t:.2f}:")
    for a in (0, 1, 2):
        Q = sector_basis(a); Ba = Q.conj().T @ B @ Q
        mods = sorted(round(abs(e)**2, 3) for e in np.linalg.eigvals(Ba))
        print(f"     sector {'1 w w2'.split()[a]:2} (dim {Q.shape[1]}):  |h|^2 = {mods}")
print("\n  => each C3 sector carries dim 4 = (one singlet) + (one slot from each of the three 3-irreps).")
print("     The Ramanujan shell |h|^2=2 is present in every sector: that distribution across {1,w,w2}")
print("     IS the C3-triality (the C3-Fourier 'generation' structure), here as pure rep theory.")
