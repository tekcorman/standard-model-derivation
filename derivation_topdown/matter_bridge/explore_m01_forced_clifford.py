"""
explore_m01 — the minimal internal structure FORCED by a genuine Dirac operator on srs. PURE MATH.
NO physics (see README wall). Builds on the verified bare geometry ../dirac_srs_mdl/srs.py.

A genuine (first-order, Clifford-type) Dirac operator is D = sum_a gamma^a (nabla_a), with Clifford
generators {gamma^a, gamma^b} = 2 delta^{ab}, one per local direction. The srs net is k-regular with
k=3 directions per vertex => need 3 generators => the Clifford algebra Cl(3). We derive the minimal
spinor space and the chirality obstruction. No target structure is assumed.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs  # verified bare geometry (Sunada K_4 crystal); we use only its coordination number

k = srs.DEG
print(f"Bare object: {k}-regular srs net (verified).")
print(f"A genuine Dirac D = Σ_a γ^a ∇_a needs {k} Clifford generators (one per edge-direction)")
print(f"  => the Clifford algebra Cl({k}).\n")

# Cl(3) on the minimal module C^2 : the Pauli matrices
s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
anti_ok = all(np.allclose(s[a]@s[b] + s[b]@s[a], 2*(a == b)*np.eye(2)) for a in range(3) for b in range(3))
print(f"Cl(3) generators (Pauli on C^2): {{γ^a,γ^b}} = 2δ^ab ?  {anti_ok}")
print(f"  minimal Cl(3) module (spinor space) = C^2  (the Pauli representation is irreducible).")

omega = s[0] @ s[1] @ s[2]
print(f"  volume element ω = γ¹γ²γ³ = {np.round(omega, 6).tolist()}   (= i·I, a central scalar).")
print(f"  => Cl(3) is ODD-dimensional: ω is a scalar, so there is NO intrinsic chirality grading")
print(f"     from the 3-direction geometry alone (an odd Clifford algebra has no Z_2 grading operator).")

# count the Clifford dimension
print(f"\n  dim Cl(3) = 2^{k} = {2**k} (real);  Cl(3,0) ≅ M_2(C);  spinor dim = 2.")

print("\n--- FORCED so far (pure math) ---")
print("  3-regular geometry  =>  Cl(3)  =>  minimal internal spinor space C^2 (2-component).")
print("\n--- OPEN (next sub-step; NOT assumed) ---")
print("  A CHIRAL (even, Z_2-graded) Dirac triple — needed for an index beyond the de Rham one, and")
print("  for a real structure J — requires an EVEN Clifford extension of the odd Cl(3). WHICH extension")
print("  the spectral-triple axioms (grading γ, real structure J, KO-dimension) FORCE is to be DERIVED.")
print("  No target Clifford dimension is assumed here.")
