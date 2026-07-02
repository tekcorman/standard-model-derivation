# interaction — findings (Interval 3)

Pure math; **no physics** (see README). Builds on `../dirac_srs_mdl/` (the band DOS) + the two bridges.

## i01 — the gap equation bootstraps a scale (dimensional transmutation)
An attractive four-fermion interaction (strength g) + mean-field (NJL/BCS) ⇒ the self-consistent gap
equation `1 = g·I(m)`, `I(m) = ∫ DOS(ε)/(2√(ε²+m²)) dε` over the srs band DOS (W = 5.99, N(0) = 0.113).
Solved at T=0: the gap m is **dynamically generated** and follows **dimensional transmutation**
`m ~ W·exp(−1/(g·N₀))` — verified: ln(m/W) is **linear in −1/g, R² = 0.998**, slope 9.82 ≈ 1/N₀ = 8.86
(the ~10% is the effective vs point DOS). The gap sits **exponentially below** W (m/W ≈ 0.002–0.13 for
g = 1.3–3.0).

- **⇒ the dimensionful scale m is BOOTSTRAPPED** from the dimensionless coupling g and the geometric
  bandwidth W. The wall **generates its own scale**; the free input drops from a *dimensionful* m to a
  *dimensionless* g. (The m=0 at g=1.0 is a finite-bin threshold; in the continuum g_c→0, any g>0 gaps.)

## What this changes
The push (time_bridge/t04) left two free **scales** (matter gap m, KMS β). The interaction **replaces the
dimensionful m by a dimensionless g** — m is now *derived*, exponentially below W. So the walled program's
free content is reduced to **dimensionless data** (g, β); the dimensionful scale is *generated*, not input.

**Updated wall verdict:** {D, srs, MDL} (+ one interaction) forces all structures and intrinsic flows,
**generates the dimensionful scale by transmutation**, and leaves only **dimensionless** free content (g, β).

## i02 — are g and β forced? what are they as objects?
**What they ARE:**
- **g** = the dimensionless inter-enantiomer four-fermion **COUPLING** (effective object λ = g·N₀); sets
  the generated scale via transmutation m ~ W·e^(−1/(g·N₀)). A **theory** parameter.
- **β** = the dimensionless KMS **TEMPERATURE** of the field state; modular Hamiltonian K = β·dΓ(D), so β
  is the **rate of the modular (intrinsic-time) clock relative to the geometric Dirac D**. A **state**
  parameter. Must be **finite** for intrinsic time (β=∞ ground state ⇒ pure ⇒ type I ⇒ no time; finite β
  ⇒ mixed ⇒ III_1).

**Forced? NO:**
- **β**: III_1 for *all* finite β (no preferred temperature, t04) ⇒ β is a free state-label (finite, but
  not a specific value).
- **g**: a genuine free 1-parameter family (m(g) distinct: m/W ≈ 0.005–0.13 for g = 1.5–3.0); criticality
  g_c ⇒ m=0 (no scale), so a scale needs g ≠ g_c ⇒ g free. Nothing in the wall (geometry / MDL waterline /
  fixed point) selects a specific g.

**THE WALLED ENDPOINT:** {D,srs,MDL} + matter + time + interaction **forces** all structures + intrinsic
flows, **generates** the dimensionful scale (transmutation), and bottoms out at **exactly two dimensionless
inputs (g, β) = (one coupling, one temperature)** — g a *theory* parameter, β a *state* parameter.
**Route to forcing them (OUTSIDE this wall):** g via the **spectral action** (the geometric origin of
couplings); β via the **cosmological state** — both belong to the broader project / the deferred cross-pollination.

## i03 — the spectral action Tr f(D/Λ): forces gravity/geometry, NOT g
Computed for the srs Dirac (D² = L = 3I−A). **Forced (all geometric):** spectral dimension **d = 3**
(Weyl law; heat-trace slope −1.549 ≈ −1.5); volume a_0 = 4; the moments Tr(Lⁿ)/cell = 12, 48, 216, 1032
(closed-walk counts — and **Tr(A³)=0 encodes girth 10**, no triangles); the natural cutoff **Λ = √6** (the
lattice's *own* bandwidth — geometric, not free). **Not forced — g:** it lives in the a_4 **gauge /
Yang–Mills** sector, which exists only once an internal gauge structure is fixed; m03 proved that structure
is **free** (the bare srs Dirac has no gauge field ⇒ no gauge kinetic term ⇒ no g to anchor).
- **⇒ CONFIRMS i02 from the spectral-action side:** the gravity/geometry sector is forced and geometric;
  the matter coupling g is free, and forcing it needs the internal gauge structure — i.e. the broader
  project / the deferred cross-pollination, not anything the bare geometry fixes.

## i04 — completing the object: there is NO free parameter (no "slot to plug data into")
The "free g, β" were **artifacts**. The disciplined matter sector is the internal Dirac D_F, and D_F is
*not* a free input — it is **built from forced srs operators**:
- **Internal ALGEBRA — forced.** The A₄ symmetry acts **regularly** on the 12 darts (orbit = 12), so its
  commutant is the **group algebra ℂ[A₄] ≅ ℂ³ ⊕ M₃(ℂ)** (dim 12, verified) — the M₃ carrying the three
  generations. *This corrects m03's "free Wedderburn algebra":* the axioms alone don't force a nontrivial
  algebra, but the **srs symmetry does** — it is ℂ[A₄], not a choice. (My first commutant pass took the
  wrong object — the generic {B(k)} commutant = scalars, dim 1; the symmetry lives in the A₄ action on the
  *family*. 5th real-time verification catch this arc.)
- **Internal DIRAC spectral data — forced.** The Hashimoto / Ihara–Bass spectrum, **|h|² = k−1 = 2**
  (Ramanujan shell, verified to 1e-15).

So every coupling/mass is a **function of forced eigenvalues + the A₄/C₃ representation data = spectral
data, determined.** The four-fermion g I introduced was the artifact of treating a coupling as an *input*
instead of reading it off D_F. And **β is gauge** (the III_1 modular flow is canonical, state-independent — t04).

**THE OBJECT IS COMPLETE — no free parameters, no slot.** D on srs determines the structure, the intrinsic
flows, the gravitational sector (i03), *and* the matter sector (D_F = forced algebra ℂ[A₄] + forced
spectrum |h|²=2). The values are **read off the spectrum, not plugged in.** The only residual is a **proof
frontier** — deriving the determined values explicitly (the broader project's mass/phase computation) —
**not** a free parameter. There was never a port for "data."
