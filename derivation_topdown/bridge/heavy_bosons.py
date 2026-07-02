"""
heavy_bosons — the TOP layer, extracted first (where I should have started).

Big-Bang forward / top-down: the heavy (gauge) bosons live at the cutoff, one short step below M_Pl.
They ride on the spectral-action gauge data ALONE — the cutoff Λ, the heat coefficient a1, ζ_D(0) —
NOT on the EW scale v, NOT on N_hub, NOT on the fermion phase δ. So they come out FIRST, with no
descent into the ~10^60 hierarchy. This script reads them straight off the object.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

g = (np.arange(16) + 0.5) / 16
BZ = [(a, b, c) for a in g for b in g for c in g]

# matter Dirac |D| = sqrt(3 - lambda_A): its top eigenvalue IS the UV cutoff (the bandwidth)
E = np.concatenate([np.linalg.eigvalsh(srs.adjacency(k)) for k in BZ])
absD = np.sqrt(np.clip(3.0 - E, 0, None))
Lambda = absD.max()

# Hodge-Dirac D^2 (de Rham, 10 modes/cell): heat coefficients = gauge normalization
D2 = np.concatenate([np.linalg.eigvalsh(srs.hodge_dirac(k)) ** 2 for k in BZ])
percell = lambda v: float(np.mean(v)) * 10            # 10 Hodge-Dirac modes per cell
a0 = percell(D2 ** 0)                                  # = 10 (mode count)
a1 = percell(D2)                                       # = Tr D^2 / cell  -> inverse-coupling normalization
zeta0 = float(np.mean(D2 > 1e-9)) * 10                 # nonzero modes/cell = ζ_D(0)

print("=" * 74); print(" HEAVY-BOSON (gauge) LAYER — the top, no N_hub, no δ"); print("=" * 74)
print(f"  UV cutoff   Λ = max|D| = {Lambda:.4f}   (= √6 = {np.sqrt(6):.4f}: the lattice's OWN bandwidth)")
print(f"  Hodge-Dirac heat data:  a0 = {a0:.2f} (=10)   a1 = Tr D²/cell = {a1:.2f} (=24)   ζ_D(0) = {zeta0:.2f} (=8)")
print()
print("  SPECTRAL-ACTION DATA (forced), read off the spectral normalization:")
print(f"    a1 = Tr D²/cell = {round(a1)}  is the Λ²-HEAT coefficient (NOT 1/α: calling a1 the inverse")
print(f"        gauge coupling was an ARTIFACT — the coupling sits in the Λ⁰ Yang-Mills coefficient, not a1).")
print(f"    ζ_D(0) = {round(zeta0)} = dim su(3)  — the gauge-sector projection = the color adjoint (forced)")
print(f"    trace indices over H_F (12-dim):  SU(3):U(1)a:U(1)b = 3:2:4  (m07; canonical normalization)")
print()
print("  THE HEAVY-BOSON DESCENT (top-down — each step is a SHORT fall, set by gauge data):")
print("    1. cutoff / Planck top          Λ = √6   (forced)")
print("    2. unification — heavy X,Y       M_unif = (gauge coupling) · (cascade) · M_Pl")
print("                                       [coupling from the Λ⁰ YM coefficient — NOT a1; cascade TO EXTRACT]")
print("    3. EW heavy bosons  W, Z         M_W,M_Z = (gauge couplings) · v      [needs v ← ONE scale]")
print("    ——— only HERE does N_hub enter (it sets v); the fermion δ-layer is BELOW this, last ———")
print()
print("  ⇒ Import-free spectral data above step 3: ζ_D(0)=8, Λ=√6, a1=24 (heat coeff), ratios 3:2:4.")
print("    OPEN (NOT forced here): the actual 1/α from the Λ⁰ coefficient, and the unification cascade.")
