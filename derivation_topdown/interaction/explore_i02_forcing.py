"""
explore_i02 — are g and β forced? what are they as objects? PURE MATH, walled.

β  = the KMS inverse-temperature of the field state; modular Hamiltonian K = β·dΓ(D), so β is the RATE of
     the modular (intrinsic-time) clock relative to the geometric Dirac D — equivalently the field state's
     temperature. We show β must be FINITE for intrinsic time (III_1) and is not forced to any value.
g  = the dimensionless inter-enantiomer quartic COUPLING (effective object λ = g·N₀); it sets the
     generated scale by transmutation. We show g is a genuine free 1-parameter family, not forced.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

N = 22; idx = (np.arange(N)+0.5)/N
eps = np.concatenate([np.linalg.eigvalsh(srs.adjacency((a, b, c))) for a in idx for b in idx for c in idx])
eps = eps - np.mean(eps); W = eps.max()-eps.min()
hist, edges = np.histogram(eps, bins=400, density=True); ec = 0.5*(edges[:-1]+edges[1:]); de = edges[1]-edges[0]
N0 = hist[np.argmin(np.abs(ec))]

print("=== β — what it is, and is it forced? ===")
print("β = KMS inverse-temperature of the field state; modular Hamiltonian K = β·dΓ(D)")
print("  ⇒ β = the rate of the modular (intrinsic-time) clock relative to the geometric Dirac D.")
for beta in [np.inf, 4.0, 1.0]:
    if np.isinf(beta):
        print("  β=∞ (ground state): occupations are a STEP (0/1) ⇒ PURE state ⇒ type I ⇒ NO intrinsic time.")
    else:
        occ = 1.0/(1.0+np.exp(beta*eps))
        frac = np.mean((occ > 0.02) & (occ < 0.98))
        print(f"  β={beta}: occupations SMOOTH (partially-filled fraction {frac:.2f}) ⇒ MIXED ⇒ Araki–Woods III_1 ⇒ intrinsic time.")
print("  ⇒ β must be FINITE for intrinsic time; and (t04) the type is III_1 for ALL finite β —")
print("     III_1 has NO preferred temperature ⇒ β is NOT forced (a free STATE label).")

print("\n=== g — what it is, and is it forced? ===")
print("g = the dimensionless inter-enantiomer quartic COUPLING; effective object λ = g·N₀.")
I = lambda m: float(np.sum(hist*de*0.5/np.sqrt(ec**2+m**2)))
def gap(g):
    if g*I(1e-11) < 1: return 0.0
    lo, hi = 1e-12, 8.0
    for _ in range(300):
        mid = np.sqrt(lo*hi); lo, hi = (mid, hi) if g*I(mid) > 1 else (lo, mid)
    return np.sqrt(lo*hi)
print("  the gap m(g) is a genuine 1-parameter family (distinct g ⇒ physically distinct scale):")
for g in [1.5, 2.0, 2.5, 3.0]:
    print(f"    g={g}  (λ=g·N₀={g*N0:.2f}):  m/W = {gap(g)/W:.4f}")
print("  criticality g_c ⇒ m=0 (no scale); a scale REQUIRES g in the gapped phase ⇒ g ≠ g_c ⇒ g is free.")
print("  nothing within the wall (geometry / MDL waterline / fixed point) selects a specific g.")

print("\n--- finding (walled) ---")
print("  WHAT THEY ARE:  g = the dimensionless inter-enantiomer quartic COUPLING (sets the generated")
print("    scale via transmutation m ~ W·e^(−1/(g·N₀)));  β = the dimensionless KMS TEMPERATURE of the field")
print("    state (must be finite for intrinsic time; III_1 for all finite β).")
print("  FORCED?  NO — within the wall, g (a THEORY coupling) and β (a STATE temperature) are the")
print("    IRREDUCIBLE dimensionless free content: one coupling + one temperature. The dimensionful scale")
print("    is generated; the structures and intrinsic flows are forced; the data bottoms out at (g, β).")
print("  ROUTE TO FORCING (outside this wall): g via the spectral action (geometric origin of couplings);")
print("    β via the cosmological state — both belong to the broader project / the deferred cross-pollination.")
