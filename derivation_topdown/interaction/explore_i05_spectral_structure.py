"""
explore_i05 (walled — replaces the removed physics version) — the spectral STRUCTURE of the C_3-Fourier
operator family on the A_4 3-irrep. PURE MATH: isotypic decompositions, spectral moments/invariants, and
dependence on the forced fiber eigenvalue. No physics vocabulary; structure is DISCOVERED, not targeted.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

# (1) C_3-isotypic decomposition of the ambient dart space (a 3-cycle in A_4)
darts = []
for (i, j, *_) in srs.EDGES:
    darts += [(i, j), (j, i)]
e, g, g2 = (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)             # identity, (1 2 3), (1 3 2)
fixed = lambda p: sum((p[a], p[b]) == (a, b) for (a, b) in darts)
chi = [fixed(e), fixed(g), fixed(g2)]
mult = [sum(chi[t]*np.exp(-2j*np.pi*t*c/3) for t, c in enumerate([0, 1, 2])).real/3 for c in range(3)]
print(f"(1) C_3 on the 12 ambient darts: fixed-point character (e, g, g^2) = {chi}")
print(f"    isotypic multiplicities (triv, ω, ω²) = {[round(m) for m in mult]}  →  UNIFORM (4,4,4).")
print(f"    ⇒ the ambient space has NO C_3 preference; any weighting must come from a forced OPERATOR.")

# (2) the C_3-Fourier operator family: eigenvalues a_j = c0 + 2r cos(2πj/3 + φ). Moment structure.
print("\n(2) C_3-Fourier operator  a_j = c0 + 2r cos(2πj/3 + φ):  which spectral moments are φ-independent?")
c0, r = 1/np.sqrt(2), 0.5
for phi in [0.0, 0.5, 1.3, 2.7]:
    a = np.array([c0 + 2*r*np.cos(2*np.pi*j/3 + phi) for j in range(3)])
    print(f"    φ={phi:.1f}:  Σa = {a.sum():.4f}   Σa² = {np.sum(a**2):.4f}   Σa³ = {np.sum(a**3):+.4f}")
print("    ⇒ Σa = 3c0 and Σa² = 3c0²+6r² are φ-INDEPENDENT (fixed by the weights); the phase φ first")
print("      appears at Σa³ (the asymmetry). Shape invariant Σa²/(Σa)² = 1/(3P), P = trivial-character weight.")

# (3) the phase is set by the FORCED fiber eigenvalue (pure srs spectral data)
print("\n(3) forced fiber eigenvalues (srs spectrum) and the induced angle cos β = (2k − λ²)/k², k=3:")
for kk in [(0, 0, 0), (0.25, 0.75, 0.25)]:
    ev = np.sort(np.linalg.eigvalsh(srs.adjacency(kk)))
    print(f"    k={kk}: eigenvalues {np.round(ev,3)};  λ² = {sorted(set(np.round(ev**2,3)))}")
for lam2 in [9, 3]:
    print(f"    λ²={lam2}:  cos β = (6−{lam2})/9 = {(6-lam2)/9:+.3f}")

print("\n--- determined structure, and the genuine open question (walled) ---")
print("  DETERMINED (math): the C_3-Fourier spectrum's first two power-sums are phase-independent (forced by")
print("  the weights); the phase enters only at the third; the phase itself is fixed by the forced fiber")
print("  eigenvalue via cos β = (2k−λ²)/k², which takes the values ∓1/3 at the two high-symmetry fibers.")
print("  OPEN (a real math problem, not a target): the ambient C_3 structure is UNIFORM (4,4,4), so the")
print("  trivial-character weight P — the one number that sets the shape invariant 1/(3P) — must come from a")
print("  SPECIFIC forced operator that breaks the uniformity. Identifying that operator, and computing P, is")
print("  the remaining structural work. (We do NOT assume a value for P.)")
