"""
explore_01 — what mathematical structure flows from {srs, Dirac, MDL}?  Pure math.
"""
import numpy as np, srs
np.set_printoptions(precision=4, suppress=True)

print("SUBSTRATE  srs = maximal abelian Z^3-cover of K_4 (Sunada).")
print("  4 vertices/cell, 6 edges, 3-regular.  Aut(K_4)=S_4 (24); rotations A_4 (12).")
print("  H_1(K_4)=Z^3 IS the deck/translation group.  k in [0,1)^3.\n")

pts = {'Gamma': (0, 0, 0), 'P': (.25, .25, .25), 'H': (.5, .5, .5), 'generic': (.2, .25, .3)}

print("[1] ADJACENCY band structure  lambda(A(k)):")
for nm, k in pts.items():
    print(f"      {nm:8} {sorted(np.round(np.linalg.eigvalsh(srs.adjacency(k)), 4))}")

print("\n[2] HODGE-DIRAC  D=[[0,d],[d*,0]] :  check D^2|_C0 == graph Laplacian 3I-A")
k = pts['P']; D = srs.hodge_dirac(k)
print(f"      D^2 vertex-block == 3I - A(P) ? {np.allclose((D@D)[:srs.NV, :srs.NV], 3*np.eye(srs.NV)-srs.adjacency(k))}")
print(f"      spec D(P) (10) = {sorted(np.round(np.linalg.eigvalsh(D), 3))}")
print(f"      spec D(Gamma)  = {sorted(np.round(np.linalg.eigvalsh(srs.hodge_dirac(pts['Gamma'])), 3))}")

print("\n[3] NON-BACKTRACKING B(k) (12x12) :  validate Ihara-Bass  h^2 - lambda*h + (k-1) = 0")
for nm in ['Gamma', 'P', 'H']:
    k = pts[nm]; A = srs.adjacency(k); B = srs.hashimoto(k)
    hB = np.linalg.eigvals(B); lam = np.linalg.eigvalsh(A)
    ok = True
    for l in lam:
        roots = [(l + np.sqrt(l*l - 4*(srs.DEG-1) + 0j))/2, (l - np.sqrt(l*l - 4*(srs.DEG-1) + 0j))/2]
        for h in roots:
            if not np.any(np.abs(hB - h) < 1e-6): ok = False
    mods = sorted(set(np.round(np.abs(hB)**2, 3)))
    print(f"      {nm:6} Ihara-Bass roots subset of spec(B)? {ok}   |h|^2 multiset = {mods}")

print("\n[4] RAMANUJAN / number-theoretic structure")
print(f"      Ramanujan bound for 3-regular: nontrivial |h| = sqrt(k-1) = sqrt(2).")
print(f"      The arg of the Ramanujan roots (a pure invariant):")
for nm in ['Gamma', 'P', 'H']:
    B = srs.hashimoto(pts[nm]); hB = np.linalg.eigvals(B)
    ram = [h for h in hB if abs(abs(h)**2 - (srs.DEG-1)) < 1e-6]
    args = sorted(set(round(abs(np.degrees(np.angle(h))), 2) for h in ram))
    print(f"      {nm:6} |arg h| (deg) on the Ramanujan shell = {args}")

print("\n[5] IHARA ZETA  zeta(u)^{-1} = (1-u^2)^2 det(I - uA + 2u^2 I)  (per cell)")
for nm in ['Gamma', 'P']:
    k = pts[nm]
    print(f"      {nm:6} zeta(0.3)^-1 = {srs.ihara_zeta_inv(0.3, k):.5f}")
print("\n      (the secular equation det(I - uA + 2u^2 I)=0 IS the Ihara-Bass relation -> the spectrum.)")
