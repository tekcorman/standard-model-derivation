"""
explore_m05 — internal representation content of the object (PURE representation theory).

Salvaged and restated with NO physics from the removed physics-comparison sandbox. These are three
structural facts about the forced object — no Standard Model, no masses, no couplings, no Koide, nothing
physical. Just group/representation theory of the srs symmetry and its spinor structure.

  (1) the 12-dart space carries the A_4 REGULAR representation:  1 + 1' + 1'' + 3·(3).
  (2) the double cover 2T = SL(2,3) (binary tetrahedral) has VECTOR irreps {1,1',1'',3} (centre +1) and
      SPINOR irreps {2,2',2''} (centre −1). The dart space contains only vector irreps ⇒ the central
      element of 2T acts as +1 there; the spinor sectors are absent from the dart space.
  (3) the exterior algebra of the 3-dim irrep, Λ^•(C³) (dim 2³ = 8), has C_3-isotypic content
      Λ⁰ ⊕ Λ¹ ⊕ Λ² ⊕ Λ³  →  (4, 2, 2).
"""
import numpy as np
from itertools import combinations

# (1) A_4 content of the 12-dart space, from the permutation character
w = np.exp(2j*np.pi/3)
A4 = {"1": [1, 1, 1, 1], "1'": [1, 1, w, w**2], "1''": [1, 1, w**2, w], "3": [3, -1, 0, 0]}
sizes = [1, 3, 4, 4]                                   # classes: e, (12)(34), (123), (132)
dart = [12, 0, 0, 0]                                   # darts fixed by a representative of each class
print("(1) 12-dart space — A_4 irrep multiplicities:")
for nm, ch in A4.items():
    m = sum(sizes[i]*dart[i]*np.conj(ch[i]) for i in range(4)).real/12
    print(f"      {nm:4s}: {round(m)}")
print("    = 1 + 1' + 1'' + 3·(3)  (the regular representation).")

# (2) only vector irreps present ⇒ the 2T centre acts trivially on the dart space
print("\n(2) 2T = SL(2,3): vector {1,1',1'',3} centre +1 ; spinor {2,2',2''} centre −1.")
print("    the dart space has only vector irreps ⇒ the 2T central element acts as +1 there;")
print("    the spinor sectors are absent from the dart space.")

# (3) C_3-isotypic content of the exterior algebra of the 3-irrep
weights = [0, 1, 2]
content = {0: 0, 1: 0, 2: 0}
for kdeg in range(4):
    for S in combinations(range(3), kdeg):
        content[sum(weights[i] for i in S) % 3] += 1
print(f"\n(3) Λ^•(C³): dim {sum(content.values())} = 2³;  "
      f"C_3-isotypic content (triv, ω, ω̄) = {tuple(content.values())} = (4, 2, 2).")
