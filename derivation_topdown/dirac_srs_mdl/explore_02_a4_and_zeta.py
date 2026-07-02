"""
explore_02 — (2) A4 representation structure of the spectrum; (3) spectral zeta zeta_D(s).
Pure math, walled off.  A4 = rotation group of K4 (order 12); irreps {1, 1', 1'', 3}.
"""
import numpy as np, itertools, srs

# ---------- A4 = even permutations of {0,1,2,3} ----------
def parity(p):
    seen = [False]*4; par = 0
    for i in range(4):
        if not seen[i]:
            j = i; c = 0
            while not seen[j]: seen[j] = True; j = p[j]; c += 1
            par += c-1
    return par % 2
A4 = [p for p in itertools.permutations(range(4)) if parity(p) == 0]

def chi3(p):                       # character of the standard 3-irrep
    fx = sum(1 for i in range(4) if p[i] == i)
    return 3 if fx == 4 else (-1 if fx == 0 else 0)

def mults(perm_fn, dim):           # multiplicities of {1, 1'(=1''), 3}; real reps => m(1')=m(1'')
    chs = [np.trace(perm_fn(p)).real for p in A4]
    m1 = sum(chs)/12
    m3 = sum(c*chi3(p) for c, p in zip(chs, A4))/12
    m1p = (dim - m1 - 3*m3)/2
    return f"1:{round(m1)}  (1'+1''):{round(2*m1p)}  3:{round(m3)}"

def pv(p):
    M = np.zeros((4, 4))
    for i in range(4): M[p[i], i] = 1
    return M
EUV = [(e[0], e[1]) for e in srs.EDGES]
def pe(p):
    M = np.zeros((6, 6))
    for e, (i, j) in enumerate(EUV):
        s = {p[i], p[j]}
        for f, (a, b) in enumerate(EUV):
            if {a, b} == s: M[f, e] = 1; break
    return M
DUV = [(d[0], d[1]) for d in srs._darts()]
def pd(p):
    M = np.zeros((12, 12))
    for d, (i, j) in enumerate(DUV):
        t = (p[i], p[j])
        for f, (a, b) in enumerate(DUV):
            if (a, b) == t: M[f, d] = 1; break
    return M

print("=== (2) A4 representation structure of the K4-cover (at Gamma, full A4) ===")
print(f"  vertices C0 (dim 4):   {mults(pv, 4)}")
print(f"  edges    C1 (dim 6):   {mults(pe, 6)}")
print(f"  darts       (dim 12):  {mults(pd, 12)}")
print("  => the directed-edge space (where the non-backtracking operator lives) carries")
print("     THREE copies of the 3-irrep.  (1 + 1' + 1'' + 3+3+3 = 12.)\n")

print("  Non-backtracking B(Gamma): eigenvalue (multiplicity) -> |h|^2")
B = srs.hashimoto((0, 0, 0)); ev = np.linalg.eigvals(B)
clusters = []
for l in ev:
    for cl in clusters:
        if abs(l-cl[0]) < 1e-6: cl[1] += 1; break
    else: clusters.append([l, 1])
for l, m in sorted(clusters, key=lambda c: -abs(c[0])):
    print(f"     h = {l.real:+.4f}{l.imag:+.4f}i   (mult {m})   |h|^2 = {abs(l)**2:.3f}")
print("  => 3-fold eigenvalues = the A4 3-irrep.  The Ramanujan pair |h|^2=2 IS two copies of 3.\n")

print("=== (3) spectral zeta of the Hodge-Dirac:  zeta_D(s) = Tr|D|^{-s}  over the BZ ===")
Ng = 16
ks = [(a/Ng, b/Ng, c/Ng) for a in range(Ng) for b in range(Ng) for c in range(Ng)]
E = []
for k in ks:
    E += [e for e in np.linalg.eigvalsh(srs.hodge_dirac(k)) if e > 1e-9]
E = np.sort(np.array(E)); Nk = len(ks)
print(f"  {Nk} k-points, {len(E)} positive Dirac eigenvalues")
xs = np.array([0.15, 0.25, 0.45])
N_lt = np.array([np.sum(E < x) for x in xs])/Nk
slope = np.polyfit(np.log(xs), np.log(N_lt), 1)[0]
print(f"  integrated DOS  N(<E) ~ E^{slope:.2f}   =>  spectral dimension d_s ~ {slope:.2f}  (expect 3)")
print(f"  zeta_D(s) = (1/Nk) sum_lambda lambda^-s  (converges for s < d_s; pole at s=d_s):")
for s in [1.0, 1.5, 2.0, 2.5, 2.9]:
    print(f"     zeta_D({s}) = {np.sum(E**(-s))/Nk:8.4f}")
print("  => the sum grows as s -> 3^-, the spectral-dimension pole.  No UV divergence (bounded spectrum).")
