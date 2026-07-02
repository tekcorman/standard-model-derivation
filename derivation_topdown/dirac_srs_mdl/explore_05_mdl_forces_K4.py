"""
explore_05 — does MDL force K_4 ?  Pure math.
Claim: K_4 is the UNIQUE simple regular graph whose maximal abelian cover is 3-dimensional.
'3-dimensional cover' = first Betti number b_1 = E - V + 1 = 3  =>  E = V + 2.
'regular' (uniform substrate, shortest description) = k-regular => E = k V / 2.
"""
import numpy as np

print("MDL -> K_4  (pure combinatorics):")
print("  maximal abelian cover dimension = b_1 = E - V + 1.")
print("  require a 3D cover:  b_1 = 3  =>  E = V + 2.")
print("  require a uniform (regular) substrate:  k-regular  =>  E = kV/2.")
print("  =>  kV/2 = V + 2  =>  V (k - 2) = 4.\n")

print("  integer solutions with k >= 3, and whether a SIMPLE k-regular graph on V vertices can exist")
print("  (needs V >= k+1):")
sols = []
for k in range(3, 9):
    if 4 % (k-2) == 0:
        V = 4 // (k-2)
        feasible = V >= k+1
        sols.append((k, V, feasible))
        print(f"     k={k}:  V={V:2}   simple k-regular feasible (V>=k+1)? {feasible}"
              f"{'   <== K_4' if (k, V) == (3, 4) else ''}")

print("\n  Only (k=3, V=4) admits a simple regular graph.")
print("  The unique 3-regular graph on 4 vertices is the complete graph K_4.")
print("  => 'uniform substrate + 3-dimensional abelian cover' forces K_4 uniquely.")
print("  Among all candidate substrates it also has the maximal automorphism group (S_4, order 24)")
print("  per vertex, i.e. the shortest description — the MDL optimum is K_4.")
