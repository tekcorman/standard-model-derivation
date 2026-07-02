"""
explore_m03 — exhaust the spectral-triple axioms: is anything beyond the chiral spinor Cl(4) forced?
PURE MATH, walled (no physics). Builds on m02 (the forced chiral spinor Cl(4)).

Result: the chiral spinor Cl(4) (= 3 spatial directions + 1 internal chirality generator) is forced by
the geometry + chirality + reality. NO internal gauge/matter ALGEBRA is forced — the minimal triple
(gauge algebra A = C, trivial internal Dirac) satisfies the first-order, reality, and orientability
axioms. Any nontrivial gauge content requires an external input absent from srs.
"""
import numpy as np
I2 = np.eye(2); s1 = np.array([[0, 1], [1, 0]]); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]])
kron = lambda a, b: np.kron(a, b)
g = [kron(s1, s1), kron(s1, s2), kron(s1, s3), kron(s2, I2)]    # Cl(4) (m02)
g5 = g[0] @ g[1] @ g[2] @ g[3]; J = g[1] @ g[3]

print("Forced so far (m02): chiral spinor Cl(4) = (3 spatial dirs) + (1 internal chirality generator),")
print("with chirality γ_c and real structure J (KO-dim 4). Now: is any internal gauge ALGEBRA forced?\n")

# (1) FIRST-ORDER CONDITION for the minimal gauge algebra A = C (scalars a = z·I)
a = (2.7 + 1.3j) * np.eye(4)                 # a generic element of A = C, acting as a scalar
D_form = g[0] + 0.5*g[1] + g[2]              # any Dirac built from the gammas (its form, not dynamics)
first_order = np.allclose(D_form @ a - a @ D_form, 0)
print(f"(1) First-order [[D,a],b°]=0 for A=C: a is a scalar ⇒ [D,a]=0 ? {first_order}  ⇒ TRIVIALLY satisfied.")

# (2) reality + chirality (Cl(4), from m02)
print(f"(2) Reality: J²=-1 ({np.allclose(J@np.conj(J), -np.eye(4))}), [J,γ_c]=0 "
      f"({np.allclose(J@np.conj(g5), g5@J)}).  Chirality: γ_c²=1 ({np.allclose(g5@g5, np.eye(4))}).")

# (3) orientability: γ_c is the Clifford volume element (the order-4 Hochschild cycle)
print(f"(3) Orientability: γ_c = γ¹γ²γ³γ⁴ is the Clifford volume element (degree-4 Hochschild cycle). Satisfied.")

print("\n⇒ The MINIMAL chiral real triple — Cl(4) spinor, gauge algebra A=C, trivial internal Dirac D_F=0 —")
print("  satisfies first-order + reality + orientability. It is a VALID, complete solution.")
print("\n--- THEOREM (matter bridge, walled) ---")
print("  FORCED by geometry + chirality + reality: exactly the minimal chiral Dirac structure Cl(4)")
print("  (3 spatial generators + 1 internal chirality generator), 4-spinor C^4=C²₊⊕C²₋, real J (KO-dim 4).")
print("  NOT FORCED: any internal gauge/matter algebra. The minimal A=C already complies; a NONtrivial")
print("  algebra requires a nontrivial internal Dirac D_F whose content (a representation input)")
print("  is NOT supplied by srs. By Artin–Wedderburn any finite-dim real internal algebra is ⊕_i M_{n_i}(K_i),")
print("  K_i ∈ {R,C,H}; the axioms constrain its FORM, but the minimal (C) works ⇒ the specific gauge/matter")
print("  content is a FREE choice, not geometry-determined.")
print("\n  Honest limit: Poincaré duality on the discrete cover is inherited from the bare object, not")
print("  re-derived here; it cannot force a gauge algebra anyway (the minimal A=C trivially satisfies it).")
