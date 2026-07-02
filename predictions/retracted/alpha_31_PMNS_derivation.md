# α_31_PMNS — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; see docs/framework_axioms.md), G.1 and G.5 are now DERIVED via CDP 2011 (see predictions/observer_hilbert_space.py), but the B6 color-vs-generation retraction remains the load-bearing block here. Re-derivation on C^3_gen awaits Need-A2 closure.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation's Step 3 invokes the Type-C sub-postulate

> "α_31 walk class = inter-generation 1 → 3, two girth cycles, n = 2g."

The "1 → 3" tag identifies the walk as an inter-generation Majorana-phase transition that skips generation 2. Under B6 (`docs/theorem_B6_bridge.md`), the C_3 irreps `{1, ω, ω²}` at the srs P-point are color labels of one Pati-Salam family, not three generations. The walk-class assignment "generation-1 to generation-3 transition" therefore loses its physical content; the integer `n = 2g` is no longer tied to any generation structure.

**Re-derivation target**: Sprint 11 workstream B7.5 (PMNS under C³_gen; see `docs/master_plan.md` §Sprint 11).

**What survives as math**: the holonomy arithmetic

$$\arg(h^{2g}) \;=\; 2g \cdot \arg(h) \;=\; 20 \cdot \arctan\sqrt{5/3} \;\pmod{360°} \;=\; 324.775°$$

is a rigorous statement (de Moivre) about the srs P-point eigenvalue `h = (√3 + i√5)/2` with srs girth `g = 10`. This **standalone mathematical lemma** is label-agnostic; only the identification with the PMNS Majorana phase `α_31` is retracted.

## Specific failing step

Step 3 assigns the walk class `n = 2g` to `α_31` on the grounds that `α_31` is the inter-generation Majorana phase coupling `ν_1` to `ν_3`, traversing "two girth cycles." Quoting:

> "α_31 is the generation-1 → generation-3 inter-generation Majorana phase, traversing two girth cycles on srs. We therefore set n = 2g = 20."

This requires the walk to be interpretable as a generation transition. Under B6, the C_3 irreps at P label colors within one family, so there is no natural "generation 1 → generation 3" walk on the P-point spectral data.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed | Status |
|---|---|---|---|
| α_31 | 324.775° | unconstrained (0νββ bounds only) | not explanatory under current framework; empirically unconstrained |

Majorana phases are not directly measurable; the derived value is not experimentally confrontable today.

## Preserved original derivation (for reference; superseded)

---

# Derivation of $\alpha_{31}$ (PMNS second Majorana phase) (SUPERSEDED, retained for reference)

## Abstract

We derive the PMNS second Majorana phase
$$\alpha_{31} \;=\; 2g \cdot \arg(h) \bmod 360° \;\approx\; 324.78°$$
as the accumulated argument of $h^{2g}$, where $h = (\sqrt{3} + i\sqrt{5})/2$ is the Bloch Hashimoto walker eigenvalue at the $P$-point of srs and $g = 10$ is the srs girth. The derivation rests on upstream theorems, de Moivre's theorem, and the adopted Type-C structural postulate P-phase-from-holonomy with an identification-layer sub-choice $n = 2g$ (two girth cycles) for the specific $\alpha_{31}$ observable.

## Framework axioms invoked

Same upstream content as `alpha_21_PMNS_derivation.md`.

## Derivation

### Step 1. Upstream: $h$, $g$

$h = (\sqrt{3} + i\sqrt{5})/2$, $g = 10$, $\arg(h) = \arctan\sqrt{5/3} \approx 52.2388°$.

### Step 2. Adopt P-phase-from-holonomy (W4 Type C)

Each PMNS / CKM phase observable corresponds to the argument of $h^n$ along a specific closed walk class on srs, with $n$ fixed by a walk-topology invariant.

### Step 3. Identify the walk class for $\alpha_{31}$: $n = 2g$ [FAILING STEP under B6]

> α_31 is the generation-1 → generation-3 inter-generation Majorana phase, traversing two girth cycles on srs. We therefore set n = 2g = 20.

**This is the failing step under B6**: the "generation-1 → generation-3" reading requires C_3 irreps to label generations. B6 proves they label colors.

### Step 4. De Moivre

$\alpha_{31} = 2g \cdot \arg(h) \pmod{2\pi}$.

### Step 5. Numerical evaluation

$20 \times 52.2388° = 1044.775°$; reducing mod 360° gives $324.775°$.

## Result (holonomy arithmetic lemma only; Majorana-phase identification retracted under B6)

$$\alpha_{31,\text{lemma}} \;=\; 2g \cdot \arg(h) \bmod 360° \;=\; 324.78°.$$

## References

- Ahlfors, L. V. (1979). *Complex Analysis*, 3rd ed. McGraw-Hill.
- Sunada, T. (2012). Lecture on topological crystallography. *Notices AMS* **59**(2), 208–215.
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
