# matter_bridge — findings (Interval 1)

Pure math; **no physics** (see README). Builds on the verified srs geometry (`../dirac_srs_mdl/srs.py`).

## m01 — minimal Clifford structure
3-regular srs ⇒ a genuine Dirac `D = Σ_a γ^a ∇_a` needs 3 Clifford generators ⇒ **Cl(3)** ⇒ minimal
spinor **ℂ²**. Cl(3) is ODD (ω = γ¹γ²γ³ = i·I, central) ⇒ no intrinsic chirality grading.

## m02 — the forced even extension  (FORCED RESULT)
Chirality forces an even Clifford; the minimal even algebra containing the 3 spatial directions is
**Cl(4)** — exactly ONE added generator. Gives the chiral spinor **ℂ⁴ = ℂ²₊ ⊕ ℂ²₋**, chirality γ_c,
and a real structure **J** (J² = −1, [J,γ_c]=0, **KO-dimension 4**). Verified computationally.

## m03 — axiom exhaustion (THEOREM)
The minimal chiral real triple — Cl(4) spinor, gauge algebra A=ℂ, trivial internal Dirac — satisfies
first-order + reality + orientability (verified). Therefore **nothing beyond Cl(4) is forced**: the
chiral spinor is forced; any internal gauge/matter *algebra* is a free finite-dim real algebra
(Artin–Wedderburn ⊕_i M_{n_i}(𝕂_i)) requiring an external input absent from srs. The gauge/matter
content is **not geometry-determined.** (Honest limit: Poincaré duality on the discrete cover inherited
from the bare object, not re-derived; it cannot force a gauge algebra regardless.)

## m04 — srs-z and the spectral gap (THEOREM)
A single srs copy (ℂ² = Cl(3) = the 3 Pauli) admits **NO gap-opening term** — no 2×2 matrix anticommutes
with all three Pauli (nullspace dim 0) ⇒ forced gapless. A spectral gap **requires** doubling to ℂ⁴
(Cl(4)) = **srs ⊕ srs-z**; the gap term is then the **4th gamma** — purely off-diagonal in the chirality
basis, i.e. the **inter-copy (srs↔srs-z) coupling** — and `D = Σ_{a=1,2,3} γ^a p_a + m·γ^4 ⇒ D² = Σp_a² + m²`.
Geometrically srs-z = the complex-conjugate (mirror) net, with **opposite Weyl charge** (+2 vs −2 at Γ).
**STRUCTURE** (gap = 4th gamma = srs↔srs-z coupling) is **FORCED** by the doubling + chirality; the
**STRENGTH m is FREE.** This ties m02 (the 4th gamma forced by chirality) to the gap: the forced 4th
generator *is* the inter-enantiomer coupling.

## INTERVAL 1 — LANDED  (matter bridge; resume marker)

**Result (all pure math, walled), building on the verified bare object (`../dirac_srs_mdl/`):**
- srs (3-regular) ⇒ Cl(3) ⇒ minimal spinor ℂ² — a *gapless* 3D chiral/Weyl spinor. [m01]
- chirality forces the minimal even extension **Cl(4)** (one added 4th generator): chiral 4-spinor
  ℂ⁴=ℂ²₊⊕ℂ²₋, chirality γ_c, real structure J (KO-dim 4). [m02]
- **nothing beyond Cl(4) is forced**: the minimal triple (Cl(4), gauge algebra A=ℂ, trivial D_F)
  satisfies first-order + reality + orientability ⇒ the internal gauge/matter *algebra* is a free
  Artin–Wedderburn choice, not geometry-determined. [m03]
  **[CORRECTED by ../interaction/i04: the *axioms alone* don't force a nontrivial algebra, but the srs
  SYMMETRY does — the A₄ regular action on the 12 darts forces the internal algebra to be ℂ[A₄] ≅ ℂ³⊕M₃.
  So the matter algebra is FORCED by the geometry's symmetry, not free.]**
- a **spectral gap REQUIRES the doubling srs ⊕ srs-z**: one copy (ℂ²) admits no gap term; the gap term is
  the 4th gamma = the off-diagonal inter-enantiomer coupling. srs-z = conj(srs net), opposite Weyl charge.
  The 4th gamma forced by chirality (m02) *is* this gap-coupling (m04) — the same generator. [m04]

**FORCED:** the chiral Dirac structure **Cl(4)** + the gap-mechanism (the srs⊕srs-z 4th-gamma coupling).
**FREE:** the gap *strength*, and the internal gauge algebra.

**Seam to Interval 2 (the time bridge):** a *scale* enters at exactly the one free strength; the time
bridge (the non-tracial state / the N-flow) is where to fix it. Not opened.

**Wall status:** PURE MATH only — no physics / SM / Cl(6) / "mass" written here. Comparison to the broader
project is DEFERRED and permission-gated. User is NOT ready to cross-pollinate.
