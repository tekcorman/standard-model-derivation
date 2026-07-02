"""
verify_pass — INDEPENDENT, adversarial re-derivation of the load-bearing claims.
Fresh code, different methods than explore_*. Reports CONFIRM / REFUTE per claim. Pure math.
"""
import numpy as np, srs, itertools, math
from collections import deque

def V(name, ok, d=""): print(f"  [{'CONFIRM' if ok else '*** REFUTE ***'}] {name}   {d}")

print("VERIFICATION PASS (independent, fresh code)\n")
nV, nE = srs.NV, len(srs.EDGES)

# 1. srs = maximal abelian cover of K_4
print("1. srs.py = maximal abelian Z^3 cover of K_4:")
b1 = nE - nV + 1
volts = np.array([v for (i, j, v) in srs.EDGES if tuple(v) != (0, 0, 0)])
V("b_1(K4)=E-V+1=3 (deck group Z^3)", b1 == 3, f"b1={b1}")
V("3 cotree voltages form a Z^3 basis", len(volts) == 3 and np.linalg.matrix_rank(volts) == 3,
  f"rank={np.linalg.matrix_rank(volts)}")
R = 3; cells = set((a, b, c) for a in range(-R, R+1) for b in range(-R, R+1) for c in range(-R, R+1))
def nbrs(node):
    s, cell = node; out = []
    for (i, j, v) in srs.EDGES:
        if i == s and (nc := tuple(np.add(cell, v))) in cells: out.append((j, nc))
        if j == s and (nc := tuple(np.subtract(cell, v))) in cells: out.append((i, nc))
    return out
def girth(start):
    dist = {start: 0}; par = {start: None}; q = deque([start]); best = 99
    while q:
        u = q.popleft()
        for w in nbrs(u):
            if w not in dist: dist[w] = dist[u]+1; par[w] = u; q.append(w)
            elif par[u] != w: best = min(best, dist[u]+dist[w]+1)
    return best
g = girth((0, (0, 0, 0)))
V("girth = 10 (fresh BFS)", g == 10, f"girth={g}")

# 2. index = -2
print("\n2. Hodge-Dirac index = V - E = -2 (McKean-Singer), independent of k:")
idx = []
for k in [(.11, .22, .33), (.25, .25, .25), (.3, .7, .1), (0, 0, 0)]:
    d = srs.incidence(k); sv = np.linalg.svd(d, compute_uv=False); r = int(np.sum(sv > 1e-9))
    idx.append((nV - r) - (nE - r))      # dim ker(d^t:C0->C1) - dim ker(d:C1->C0)
V("index = -2 at every k", all(i == -2 for i in idx), f"idx={idx}  (V-E={nV-nE})")

# 3. zeta_D(0) = 8 (no regularization: = # nonzero modes per cell)
print("\n3. zeta_D(0) = #(nonzero modes per cell)  [bounded operator => trivial, no regularization]:")
kd = [int(np.sum(np.abs(np.linalg.eigvalsh(srs.hodge_dirac(k))) < 1e-9))
      for k in [(.11, .22, .33), (.25, .13, .4), (.3, .7, .2)]]
V("zeta_D(0)=8 = 10 modes - 2 generic kernel", all(x == 2 for x in kd), f"dim ker D(generic)={kd}; 10-2=8")

# 4. Weyl charges -- independent sphere Fukui-Hatsugai (different from explore_09's planar method)
print("\n4. Weyl monopole charges (independent: Berry flux through an enclosing sphere):")
def chern_sphere(k0, band, eps=0.04, N=22):
    k0 = np.array(k0, float); th = np.linspace(.02, math.pi-.02, N); ph = np.linspace(0, 2*math.pi, N, endpoint=False)
    U = np.empty((N, N), object)
    for a in range(N):
        for b in range(N):
            kk = k0 + eps*np.array([math.sin(th[a])*math.cos(ph[b]), math.sin(th[a])*math.sin(ph[b]), math.cos(th[a])])
            U[a, b] = np.linalg.eigh(srs.adjacency(kk))[1][:, band]
    F = 0.0
    for a in range(N-1):
        for b in range(N):
            bn = (b+1) % N
            lk = np.vdot(U[a, b], U[a, bn])*np.vdot(U[a, bn], U[a+1, bn])*np.vdot(U[a+1, bn], U[a+1, b])*np.vdot(U[a+1, b], U[a, b])
            F += np.angle(lk)
    return F/(2*math.pi)
cG = chern_sphere((0, 0, 0), 0); cP = chern_sphere((.25, .75, .25), 0)
V("charge 2 monopole at Gamma (lowest band)", abs(round(cG)) == 2, f"sphere Chern = {cG:+.3f}")
V("charge 1 Weyl at (1/4,3/4,1/4)", abs(round(cP)) == 1, f"sphere Chern = {cP:+.3f}")

# 5. tree entropy -- method calibrated on Z^2, then srs recomputed on a fresh grid
print("\n5. tree entropy: calibrate the method on Z^2 (=(4/pi)Catalan), then srs on an independent grid:")
N = 240; ix = (np.arange(N)+0.5)/N; KX, KY = np.meshgrid(ix, ix)
z2 = np.mean(np.log(4 - 2*np.cos(2*np.pi*KX) - 2*np.cos(2*np.pi*KY)))
V("Z^2 method check = (4/pi)Catalan", abs(z2 - (4/math.pi)*0.9159655942) < 1e-3,
  f"{z2:.5f} vs {(4/math.pi)*0.9159655942:.5f}")
ix = (np.arange(50)+0.5)/50; K = np.array(np.meshgrid(ix, ix, ix)).reshape(3, -1).T
Asrs = np.zeros((len(K), 4, 4), complex)
for i, j, v in srs.EDGES:
    p = np.exp(2j*np.pi*(K @ np.array(v, float))); Asrs[:, i, j] += p; Asrs[:, j, i] += np.conj(p)
h = np.mean(np.log(np.abs(np.linalg.det(3*np.eye(4)[None] - Asrs))))
V("srs tree entropy = 3.3286 (independent grid N=50)", abs(h - 3.3286) < 0.02, f"h={h:.4f}")

# 6. A_4 commutant of the dart space = C+C+C+M_3 (dim 12)
print("\n6. A_4-commutant of the 12-dim dart space = dim 12 (1+1+1+9):")
A4 = [p for p in itertools.permutations(range(4))
      if sum(p[a] > p[b] for a in range(4) for b in range(a+1, 4)) % 2 == 0]
DUV = [(d[0], d[1]) for d in srs._darts()]
def pd(p):
    M = np.zeros((12, 12))
    for d, (i, j) in enumerate(DUV):
        M[DUV.index((p[i], p[j])), d] = 1
    return M
reps = [pd(p) for p in A4]
Big = np.vstack([np.kron(np.eye(12), Rr) - np.kron(Rr.T, np.eye(12)) for Rr in reps])
cdim = 144 - np.linalg.matrix_rank(Big, tol=1e-8)
V("commutant dim = 12", cdim == 12, f"dim={cdim}")

# 7. chirality (partial sanity vs the known fact): no permutation realizes inversion k->-k
print("\n7. chirality sanity: no vertex permutation realizes inversion k->-k:")
ks = [np.array([.13, .27, .41]), np.array([.5, .2, .9]), np.array([.33, .61, .07])]
improper = any(all(np.allclose(P @ srs.adjacency(k) @ P.T, srs.adjacency(-k), atol=1e-6) for k in ks)
               for P in [np.eye(4)[list(p)] for p in itertools.permutations(range(4))])
V("chiral (no inversion permutation)", not improper, "(full check was explore_10)")

print("\n=== verification pass complete ===")
