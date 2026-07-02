"""
explore_21 — close Bucket A: (A1) the full point group via SPECTRUM symmetry (captures the
non-symmorphic screws the constant-U search missed); (A2) level statistics with proper unfolding
+ van Hove energies. Pure math.
"""
import numpy as np, srs, itertools

# ===== A1: full point group order =====
print("A1. Full point group of the band structure (spectrum symmetry spec(A(Rk))=spec(A(k))):")
def order(R, m=8):
    P = np.eye(3, dtype=int)
    for n in range(1, m+1):
        P = P @ R
        if np.array_equal(P, np.eye(3, dtype=int)): return n
    return 0
cands = []
for e in itertools.product([-1, 0, 1], repeat=9):
    R = np.array(e).reshape(3, 3)
    if abs(round(np.linalg.det(R))) == 1 and order(R): cands.append(R)
ks = [np.array([.137, .271, .413]), np.array([.553, .211, .897]), np.array([.331, .617, .073])]
def sym(R):
    return all(np.allclose(np.sort(np.linalg.eigvalsh(srs.adjacency(R @ k))),
                           np.sort(np.linalg.eigvalsh(srs.adjacency(k))), atol=1e-7) for k in ks)
S = [R for R in cands if sym(R)]
proper = [R for R in S if round(np.linalg.det(R)) == 1]
improper = [R for R in S if round(np.linalg.det(R)) == -1]
from collections import Counter
ords = Counter(order(R) for R in proper)
print(f"  candidates tested (entries -1,0,1, det +-1, finite order): {len(cands)}")
print(f"  symmetries found: {len(S)}  ->  proper {len(proper)}, improper {len(improper)}")
print(f"  proper element orders: {dict(sorted(ords.items()))}")
has4 = ords.get(4, 0) > 0
print(f"  => point group order {len(proper)}: {'O (order 24, has 4-fold axes)' if len(proper)==24 and has4 else 'A_4 (order 12, no 4-fold)' if len(proper)==12 else 'other'}")
print(f"     (A_4 has NO order-4 elements; O has 6 four-fold rotations. order-4 present: {has4})")
print(f"     improper (det -1) elements: {len(improper)} -> {'chiral (none)' if len(improper)==0 else 'achiral'}")

# ===== A2: level statistics (proper unfolding) + van Hove =====
print("\nA2. Level statistics (proper smooth unfolding) and van Hove energies:")
N = 16; ix = (np.arange(N)+0.5)/N
E = np.sort(np.concatenate([np.linalg.eigvalsh(srs.adjacency((a, b, c))) for a in ix for b in ix for c in ix]))
stair = np.arange(1, len(E)+1)
coef = np.polyfit(E, stair, 12)                 # smooth integrated-DOS fit
unf = np.polyval(coef, E)
s = np.diff(unf); s = s[s > 1e-9]; s = s/s.mean()
print(f"  unfolded NN spacing: mean {s.mean():.3f}, variance {s.var():.3f}   (Poisson var=1; Wigner~0.27)")
print(f"  P(s<0.1) = {np.mean(s < 0.1):.3f}   (Poisson ~0.10 [no repulsion]; Wigner ~0 [repulsion])")
verdict = "POISSON / integrable (no level repulsion)" if s.var() > 0.7 and np.mean(s < 0.1) > 0.05 else "shows repulsion?"
print(f"  => {verdict}")
# van Hove = band critical-point energies (the high-symmetry band values)
vh = set()
for k in [(0, 0, 0), (.25, .25, .25), (.5, .5, .5), (.5, .5, 0), (.25, .75, .25)]:
    for e in np.round(np.linalg.eigvalsh(srs.adjacency(k)), 4): vh.add(e)
print(f"  van Hove energies (band critical points at high-symmetry k): {sorted(vh)}")
