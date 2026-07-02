"""
explore_14 — WHERE ARE THE DYNAMICS?
The non-backtracking walk IS the geodesic flow; the Ihara zeta IS its dynamical (Ruelle) zeta.
Plus the continuous Dirac/heat flow, and an honest statement about the modular ('observer') time.
Pure math.
"""
import numpy as np, srs
B0 = srs.hashimoto((0, 0, 0)).real     # NB operator of the K_4 quotient (Bloch phases = 1 at k=0)

print("THE DYNAMICS = the non-backtracking (geodesic) flow on the substrate.")
sr = max(abs(np.linalg.eigvals(B0)))
print(f"  spectral radius rho(B) = {sr:.4f} = k-1 = {srs.DEG-1}")
print(f"  topological entropy  h = log(k-1) = {np.log(srs.DEG-1):.5f}   (exponential orbit growth)")

print("\n  closed NB orbits of length m   N_m = Tr(B^m)   (the periodic orbits of the flow):")
Bm = np.eye(12)
for m in range(1, 13):
    Bm = Bm @ B0
    print(f"    m={m:2}:  N_m = {round(np.trace(Bm).real):8}      (k-1)^m = {(srs.DEG-1)**m:8}")

u = 0.25
det = np.linalg.det(np.eye(12) - u*B0)
dyn = np.exp(-sum(np.trace(np.linalg.matrix_power(B0, m)).real * u**m / m for m in range(1, 80)))
print("\n  THE STATIC ZETA IS A DYNAMICAL ZETA:")
print(f"    det(I - uB)               [Ihara zeta^-1, the spectral object] = {det:.6f}")
print(f"    exp(-sum_m N_m u^m / m)   [Ruelle dynamical zeta^-1, periodic orbits] = {dyn:.6f}")
print(f"    match: {np.isclose(det, dyn, atol=1e-5)}")
print("  => the Ihara zeta we computed = the Ruelle zeta of the NB walk; its Euler product")
print("     prod_prime (1-u^len)^-1 runs over PRIME CYCLES = primitive periodic orbits.")
print("     The Ramanujan shell |h|^2 = k-1 is exactly the resonance spectrum of this flow.")

print("\n  CONTINUOUS dynamics: the spectral triple gives the Dirac flow e^{itD} and the heat flow")
print("  e^{-tD^2} (diffusion); D^2 = the graph Laplacian = generator of the walk's continuum limit.")

print("\n  MODULAR ('observer') time -- the honest statement:")
print("  The natural trace on the translation (Z^3) von Neumann algebra is TRACIAL, so its")
print("  Tomita-Takesaki modular flow is TRIVIAL (sigma_t = id). A nontrivial thermodynamic time")
print("  (a 'd_N' / Connes-Rovelli thermal time) requires a NON-tracial state = an observer's")
print("  coarse-graining. That state is structure ADDED to the bare {D, srs, MDL}; it is NOT intrinsic.")
print("  => Intrinsic dynamics of the bare object: the geodesic (NB-walk) flow + the Dirac/heat flow.")
print("     The observer/thermal time is the one dynamical ingredient the bare crystal does NOT contain.")
