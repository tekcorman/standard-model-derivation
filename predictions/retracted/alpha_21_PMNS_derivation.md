# α_21_PMNS — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; see docs/framework_axioms.md), G.1 and G.5 are now DERIVED via CDP 2011 (see predictions/observer_hilbert_space.py), but the B6 color-vs-generation retraction remains the load-bearing block here. Re-derivation on C^3_gen awaits Need-A2 closure.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation's Step 2 invokes the Type-C sub-postulate

> "α_21 assignment: α_21 is the holonomy of the h-eigenspace of B(P) around a **full girth cycle**, i.e. n = g."

where the walk-class assignment is tied to the inter-generation transition `ν_1 → ν_2`. Under B6 (`docs/theorem_B6_bridge.md`), the h-eigenspace of B(P) does not label generations; B6 proves the srs body-diagonal C_3 is the color-Z_3 of SU(3)_c via Spin(6)≅SU(4)→PS embedding, so the h-eigenspace at P is a color-isotypic sub-bundle within one Pati-Salam family, not a set of three neutrino generations. The "walk class for α_21 = full girth cycle, gen 1 → gen 2" sub-postulate therefore loses its underlying physical content.

**Re-derivation target**: Sprint 11 workstream B7.5 (PMNS under C³_gen; see `docs/master_plan.md` §Sprint 11). Under Sprint 11, Majorana phases emerge from the mass operator on C³_gen, which is orthogonal to the srs C_3 color structure.

**What survives as math**: the holonomy arithmetic

$$\arg(h^{g}) \;=\; g \cdot \arg(h) \;=\; 10 \cdot \arctan\sqrt{5/3} \;\pmod{360°} \;=\; 162.388°$$

is a rigorous statement (de Moivre's theorem applied to the srs P-point eigenvalue). It is a **standalone mathematical lemma** about srs spectral data: the accumulated argument of `h^g` where `h = (√3 + i√5)/2` and `g = 10` is the srs girth. This lemma is label-agnostic and remains valid under B6; only its identification with the PMNS Majorana phase `α_21` (an inter-generation object) is retracted.

## Specific failing step

Step 2 invokes the Type-C sub-postulate that identifies a closed walk of length `g` on the srs P-point h-eigenspace with the inter-generation Majorana phase `α_21`. Quoting:

> "α_21 walk class = full girth cycle, gen 1 → gen 2 (no sector-crossing edge) ⇒ n = g."

This requires the h-eigenspace to support a gen-1-to-gen-2 transition, which in turn requires the C_3 irreps `{1, ω, ω²}` to label three generations. B6 proves those irreps are color labels within one family, not generation labels. The `n = g` walk-topology input has no physical content as a generation transition under B6.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed | Status |
|---|---|---|---|
| α_21 | 162.388° | unconstrained (0νββ bounds only) | not explanatory under current framework; empirically unconstrained |

The Majorana phase α_21 is not directly measurable; it enters `⟨m_ββ⟩` in 0νββ decay. The derived value 162.388° is therefore a falsifiable prediction in principle but has no experimental discriminator today. Under B6, the framework does not claim 162.388° as a derived PMNS Majorana phase.

## Preserved original derivation (for reference; superseded)

---

# Derivation of $\alpha_{21}$ (PMNS first Majorana phase) (SUPERSEDED, retained for reference)

## Abstract

We derive the first PMNS Majorana phase

$$\alpha_{21} \;=\; g\cdot\arg(h) \bmod 360° \;=\; 10\,\arctan\!\sqrt{5/3} \bmod 360° \;\approx\; 162.388°$$

as the accumulated argument of the Hashimoto walker eigenvalue $h$ over one full girth-cycle closed walk on the srs lattice. The derivation combines two upstream closed results — $h = (\sqrt{3} + i\sqrt{5})/2$ from `docs/theorem_BP_doubly_degenerate_h.md` and $g = 10$ from `predictions/g_girth.py` — with the framework's adopted Type-C structural postulate **P-phase-from-holonomy** (`docs/W4_identification_catalog.md` §2C), together with de Moivre's theorem.

## Framework axioms invoked

No new axioms; same upstream content as `Q_Koide_derivation.md` plus g = 10 and P-phase-from-holonomy.

## Derivation

### Step 1. Upstream structural data

- $h = (\sqrt{3} + i\sqrt{5})/2$ as a $C_3$-protected doubly-degenerate eigenvalue at P.
- $g = 10$, the srs girth.

### Step 2. Adopted identification: P-phase-from-holonomy (Type C) [FAILING STEP under B6]

By W4-catalog §2C:
> α_21 is the holonomy of the h-eigenspace of B(P) around a full girth cycle, i.e. n = g.

Under this postulate:
$$\alpha_{21} \;\equiv\; \arg\!\left(h^{g}\right) \pmod{360°}.$$

**This is the failing step under B6**: the walk class "full girth cycle, gen 1 → gen 2" requires the h-eigenspace to host three generations. B6 proves the C_3 structure at P is color-Z_3, not generation.

### Step 3. de Moivre's theorem

$\arg(h^{n}) = n\,\arg(h) \pmod{2\pi}$ for every integer $n \ge 1$.

Applied with $n = g = 10$:
$$\alpha_{21} \;=\; g\cdot\arg(h) \pmod{360°}.$$

### Step 4. Closed-form argument of $h$

$\arg(h) = \arctan(\sqrt{5}/\sqrt{3}) = \arctan\sqrt{5/3} \approx 52.23876°$.

### Step 5. Reduction mod $360°$

$10 \cdot 52.23876° = 522.388° \equiv 162.388° \pmod{360°}$.

## Result (holonomy arithmetic lemma only; Majorana-phase identification retracted under B6)

$$\alpha_{21,\text{lemma}} \;=\; g\,\arctan\sqrt{5/3} - 360° \;=\; 162.388°.$$

## References

- Ahlfors, L. V. (1978). *Complex Analysis.* 3rd ed., McGraw-Hill. §1.2 (de Moivre's theorem).
- Sunada, T. (2012). Lecture on topological crystallography. *Notices AMS* **59**(2), 208–215.
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
