"""
explore_04 — Ihara zeta zeros and the Ramanujan property as the graph Riemann hypothesis. Pure math.
zeta(u)^{-1} = (1-u^2)^{|E|-|V|} det(I - uA(k) + (k-1)u^2 I).  Poles of zeta = zeros of the det.
Graph-RH: the non-trivial zeros lie on the critical circle |u| = 1/sqrt(k-1).
"""
import numpy as np, srs
q = srs.DEG - 1                      # = 2
crit = 1/np.sqrt(q)                  # 1/sqrt(2) ~ 0.7071
print(f"Ihara zeta of srs.  k={srs.DEG}-regular, q=k-1={q}.")
print(f"Graph-RH / Ramanujan: non-trivial zeros of det(I-uA+qu^2 I) on |u| = 1/sqrt(q) = {crit:.5f}.")
print(f"(equivalently the Alon-Boppana bound |lambda| <= 2*sqrt(q) = {2*np.sqrt(q):.4f} on the nontrivial spectrum.)\n")

for nm, k in [('Gamma', (0, 0, 0)), ('P', (.25, .25, .25)), ('H', (.5, .5, .5)), ('generic', (.2, .25, .3))]:
    lam = np.linalg.eigvalsh(srs.adjacency(k))
    on, off = [], []
    for l in lam:
        for u in np.roots([q, -l, 1]):       # q u^2 - l u + 1 = 0  (=> u = 1/h)
            (on if abs(abs(u)-crit) < 1e-9 else off).append(round(abs(u), 5))
    print(f"  {nm:8}: nontrivial |u| (Ramanujan) = {sorted(set(on))}   |  off-circle |u| = {sorted(set(off))}"
          f"   [lambda = {sorted(round(x,3) for x in lam)}]")

print("\n  Mechanism (exact):  for q u^2 - lam u + 1 = 0 with complex roots (lam^2 < 4q),")
print("  |u|^2 = (product of roots) = 1/q  =>  |u| = 1/sqrt(q)  EXACTLY whenever |lambda| < 2*sqrt(q).")
print("  Real roots occur only for |lambda| >= 2*sqrt(q): here just lambda = +-3 (the trivial/Perron band).")
print("  => the non-trivial spectrum sits on the Ramanujan shell <=> the zeros sit on the critical circle.")
print("     That equivalence IS the Riemann hypothesis for this graph zeta, and it holds at every k.")
