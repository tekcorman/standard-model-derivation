"""
explore_m02 — the forced even Clifford extension of Cl(3). PURE MATH (see README wall; no physics).

Cl(3) is odd => no chirality grading. The EVEN Clifford algebras are Cl(0), Cl(2), Cl(4), ...;
Cl(2) has only 2 generators (too few to contain the 3 spatial directions), so the MINIMAL even
Clifford algebra containing Cl(3) is Cl(4) — exactly ONE added generator. We build it and verify the
chirality grading and a real structure J, then state honestly what is forced vs free. No target
Clifford dimension assumed.
"""
import numpy as np

I2 = np.eye(2); s1 = np.array([[0, 1], [1, 0]]); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]])
kron = lambda a, b: np.kron(a, b)
g = [kron(s1, s1), kron(s1, s2), kron(s1, s3), kron(s2, I2)]   # Cl(4) generators (4x4)

ok = all(np.allclose(g[a]@g[b] + g[b]@g[a], 2*(a == b)*np.eye(4)) for a in range(4) for b in range(4))
print(f"minimal even extension Cl(4):  {{γ^a,γ^b}} = 2δ^ab ?  {ok}")
print(f"  γ^1,γ^2,γ^3 = the spatial Cl(3) (3 edge-directions);  γ^4 = the single forced added generator.")

g5 = g[0] @ g[1] @ g[2] @ g[3]
print(f"  chirality  γ_c = γ^1γ^2γ^3γ^4:  γ_c²=1 ? {np.allclose(g5@g5, np.eye(4))};  "
      f"{{γ_c,γ^a}}=0 ∀a ? {all(np.allclose(g5@g[a] + g[a]@g5, 0) for a in range(4))}")
print(f"  γ_c eigenvalues {np.round(np.linalg.eigvalsh(g5), 3).tolist()}  =>  spinor C^4 = C²₊ ⊕ C²₋ (chiral halves).")

# real structure J = C ∘ (complex conjugation)
C = g[1] @ g[3]
J2 = C @ np.conj(C)
j2 = np.real(J2[0, 0]) if np.allclose(J2, J2[0, 0]*np.eye(4)) else None
commutes_g5 = np.allclose(C @ np.conj(g5), g5 @ C)         # J γ_c = γ_c J  ?
print(f"  real structure J = (γ²γ⁴)·conj:  J² = {j2:+.0f}·I ;  J γ_c = γ_c J ? {commutes_g5}")
print(f"  => (J²=-1, [J,γ_c]=0) is the KO-dimension-4 signature: Cl(4) carries a real, even, chiral triple.")

print("\n--- FORCED (pure math) ---")
print("  Chirality forces an EVEN Clifford; the minimal one containing the 3 spatial directions is Cl(4)")
print("  — exactly ONE added generator. Result: chiral spinor C^4 = C²₊⊕C²₋, chirality γ_c, real J (KO-dim 4).")
print("  The matter bridge FORCES precisely one extra Clifford generator (a 4th direction).")
print("  Its interpretation (geometric/temporal/internal) is DEFERRED — not decided here.")
print("\n--- NOT forced (honest) ---")
print("  Larger even Clifford algebras are PERMITTED by the axioms but NOT forced by the 3-regular geometry;")
print("  going beyond Cl(4) requires an extra principle ABSENT from the bare object. The internal *algebra*")
print("  (any 'matter content' beyond the spinor) is a SEPARATE choice, not fixed by the chirality requirement.")
