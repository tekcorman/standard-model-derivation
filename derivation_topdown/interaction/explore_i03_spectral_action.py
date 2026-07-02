"""
explore_i03 — the spectral action Tr f(D/Λ) for the srs Dirac. PURE MATH, walled.

D² = the graph Laplacian L = 3I − A (spectrum μ = 3 − λ_A ∈ [0,6]). The spectral action's asymptotic
content is the Seeley–deWitt / heat-kernel coefficients = the spectral moments of L (closed-walk counts) —
all GEOMETRIC. We extract: the spectral dimension (Weyl law), the volume coefficient, the moments, and the
natural lattice cutoff. Then we ask honestly whether it forces the coupling g.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

N = 24; idx = (np.arange(N)+0.5)/N
lamA = np.concatenate([np.linalg.eigvalsh(srs.adjacency((a, b, c))) for a in idx for b in idx for c in idx])
mu = 3.0 - lamA                                                # Laplacian eigenvalues D² ∈ [0,6]
percell = lambda v: float(np.mean(v))*4                        # per unit cell (4 bands)

print("(1) spectral dimension — heat trace Z(t)=Tr e^(−tL)/cell ~ t^(−d/2) at large t:")
ts = np.array([4., 8., 16., 32., 64., 128.])
Z = np.array([percell(np.exp(-t*mu)) for t in ts])
slope = np.polyfit(np.log(ts), np.log(Z), 1)[0]
print(f"    d ln Z / d ln t = {slope:.3f}   (expect −d/2 = −1.5 ⇒ spectral dimension d = {-2*slope:.2f})")

print("\n(2) heat-kernel coefficients = spectral moments Tr(Lⁿ)/cell  (closed-walk counts, geometric):")
for n in range(0, 5):
    print(f"    a_{2*n} ~ Tr(L^{n})/cell = {percell(mu**n):.3f}")
print(f"    (Tr L/cell=12=3·deg·? ; Tr L²/cell=48 ; … all fixed by the srs adjacency — purely geometric.)")

print("\n(3) spectral action S(Λ)=Tr e^(−L/Λ²)/cell  vs Λ  (lattice ⇒ finite, natural cutoff Λ~√6):")
for Lam in [0.5, 1.0, 2.0, 4.0]:
    print(f"    Λ={Lam}:  S = {percell(np.exp(-mu/Lam**2)):.3f}   (→ 4·f(0)=4 as Λ→∞; ~Λ³·DOS as Λ→0)")
print(f"    max μ = {mu.max():.3f} ⇒ the lattice supplies its OWN cutoff Λ_max=√6≈2.449 (geometric, not free).")

print("\n--- finding (spectral action, walled) ---")
print("  The spectral action of the BARE srs Dirac FORCES the GEOMETRIC/GRAVITATIONAL sector — the spectral")
print("  dimension d=3 (Weyl law), the volume coefficient a_0, the curvature-type a_2, all the moments Tr(Lⁿ)")
print("  (closed-walk counts), and even the cutoff (the lattice's own bandwidth √6). All geometric, all forced.")
print("  But it does NOT force the coupling g: g lives in the GAUGE/MATTER sector (the a_4 Yang–Mills term),")
print("  which exists only once an INTERNAL GAUGE STRUCTURE is fixed — and m03 proved that structure is FREE")
print("  (the bare srs Dirac has no gauge field, so no gauge kinetic term, so no g to anchor).")
print("  ⇒ CONFIRMS i02 from the spectral-action side: the gravity sector is forced & geometric; the matter")
print("    coupling g is free, because forcing it requires choosing the internal gauge structure — which is")
print("    exactly the broader project / the deferred cross-pollination, not anything the bare geometry fixes.")
