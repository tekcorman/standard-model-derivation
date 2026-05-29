# δ_CP_PMNS — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; see docs/framework_axioms.md), G.1 and G.5 are now DERIVED via CDP 2011 (see predictions/observer_hilbert_space.py), but the B6 color-vs-generation retraction remains the load-bearing block here. Re-derivation on C^3_gen awaits Need-A2 closure.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation's Step 4 invokes the Type-C sub-postulate

> "The specific walk class for the PMNS Dirac CP phase is the *Jarlskog loop*: a closed walk of length g on srs with exactly one C_3 generation transition … Hence the number of free NB edges is n = g − 1."

The exponent `g − 1` comes from one edge being fixed by the "C_3 generation transition" at a trivalent vertex, with the three edges labelled by the three C_3 irreps `{1, ω, ω²}` treated as three generations. Under B6 (`docs/theorem_B6_bridge.md`), those irreps are color components within one Pati-Salam family, not three generations. The "one edge fixed by a generation transition" sub-postulate loses its physical content.

**Re-derivation target**: Sprint 11 workstream B7.5 (PMNS under C³_gen; see `docs/master_plan.md` §Sprint 11). Under Sprint 11, the Dirac CP phase emerges from the mismatch between neutrino and charged-lepton mass operators on C³_gen.

**What survives as math**: the holonomy arithmetic

$$\arg((h^{*})^{g-1}) \;=\; (g-1) \cdot \arg(h^{*}) \;=\; 9 \cdot (-\arctan\sqrt{5/3}) \;\pmod{360°} \;=\; 249.85°$$

is a rigorous statement (de Moivre + complex conjugation) about the srs P-point eigenvalue `h = (√3 + i√5)/2` with srs girth `g = 10`. This **standalone mathematical lemma** is label-agnostic; only the identification with the Dirac CP phase is retracted.

## Specific failing step

Step 4 fixes `n = g − 1` via the Jarlskog loop sub-postulate: a closed walk of length `g` with "exactly one C_3 generation transition," forcing one of the edges at a trivalent vertex. The underlying physical reading uses the three C_3 irreps `{1, ω, ω²}` at a trivalent vertex as three generation labels `{α, β, γ}`, with a transition `α → β` using the unique edge labelled `β·α^{-1}`. Under B6, those irreps are color labels, so "generation transition" is not a physical operation on the P-point spectral data.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (NuFIT 6.0, NO) | Status |
|---|---|---|---|
| δ_CP | 249.85° | 177°⁺¹⁹₋₂₀ (NuFIT 6.0) | not explanatory under current framework (~3.7σ tension) |
| δ_CP | 249.85° | 230° ± 36° (NuFIT 5.3, older combined fit) | not explanatory under current framework (~0.55σ) |

The experimental value of δ_CP remains in flux; DUNE/Hyper-Kamiokande will provide a decisive measurement. Under B6, the framework does not claim 249.85° as a derived Dirac CP phase.

## Preserved original derivation (for reference; superseded)

---

# Derivation of the PMNS Dirac CP phase $\delta_{CP}$ (SUPERSEDED, retained for reference)

## Abstract

We derive
$$\delta_{CP} \;=\; (g - 1) \cdot \arg(h^{*}) \;\bmod\; 360° \;=\; 9\cdot\arg(h^{*}) \;\bmod\; 360° \;\approx\; 249.85°$$
as the Type-C (phase-from-holonomy) observable associated to the Jarlskog loop on the srs lattice.

## Framework axioms invoked

Same upstream content as `alpha_21_PMNS_derivation.md`; plus three P-phase sub-postulates:
(sub-1) phases are walk holonomies; (sub-2) Jarlskog loop has n = g − 1 free edges due to one C_3 transition edge; (sub-3) CP covariance: h → h*.

## Derivation

### Step 1. Upstream: $d = 3$, $k^{*} = 3$, srs, $g = 10$

### Step 2. Walker eigenvalue at $P$: $h = (\sqrt{3} + i\sqrt{5})/2$

$\arg(h) = \arctan\sqrt{5/3} \approx 52.2388°$, so $\arg(h^{*}) = -\arg(h) \approx -52.2388°$.

### Step 3. Invoke P-phase-from-holonomy (sub-1)

$\delta_{CP}$ is a walk holonomy.

### Step 4. Invoke sub-2 and sub-3: $n = g - 1$ and conjugation [FAILING STEP under B6]

> The walk class for $\delta_{CP}$ is the Jarlskog loop — a closed walk of length $g$ on srs with exactly one $C_{3}$ generation transition, located at one specific edge. At a trivalent ($k^{*} = 3$) vertex of srs the three incident directed edges carry the three distinct one-dimensional $C_{3}$ irreducible representations $\{1, \omega, \omega^{2}\}$ … A transition from sector $\alpha$ to sector $\beta$ must exit along the unique edge whose $C_{3}$ label is $\beta\cdot\alpha^{-1}$. That edge is therefore fixed by the transition; it is not a free NB choice.
>
> Hence the number of free NB edges in the Jarlskog loop is n = g − 1 = 9.

**This is the failing step under B6**: the "C_3 generation transition" reading requires C_3 irreps to label generations. B6 proves they label colors.

### Step 5. Apply de Moivre's theorem

$\delta_{CP} \equiv (g - 1) \cdot \arg(h^{*}) \bmod 360°$.

### Step 6. Numerical evaluation

$9 \times (-52.23876°) = -470.149°$; adding 720° gives $249.851°$.

## Result (holonomy arithmetic lemma only; Dirac-CP-phase identification retracted under B6)

$$\delta_{CP,\text{lemma}} \;=\; (g - 1) \cdot \arg(h^{*}) \bmod 360° \;\approx\; 249.85°.$$

## References

- Ahlfors, L. V. (1979). *Complex Analysis.* 3rd ed., McGraw–Hill.
- Needham, T. (1997). *Visual Complex Analysis.* Oxford University Press.
- Jarlskog, C. (1985). Commutator of the quark mass matrices. *Phys. Rev. Lett.* **55**, 1039–1042.
- Peskin, M. E. & Schroeder, D. V. (1995). *An Introduction to Quantum Field Theory.* Westview.
- Serre, J.-P. (1977). *Linear Representations of Finite Groups.* Springer GTM 42.
- Esteban, I. et al. (2024). NuFIT 6.0.
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
