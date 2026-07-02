# R_nu — Neutrino Mass Splitting Ratio

## Abstract

We derive the structural constant R = 228/7 = 32.5714... from the spectral
theory of K4, the quotient graph of the srs crystal net. The argument uses only
k* = 3 (selected by axioms A1 + A2-T via MDL) and proceeds entirely through the
Ihara zeta function of K4, the Chebyshev propagator at the algebraically
selected distance n = 5, and explicit Gaussian integer arithmetic. No free
parameters appear; every step is exact rational-radical arithmetic. The
result matches the NuFIT 6.0 ratio Δm²₃₁/Δm²₂₁ to 1.4σ. The physical
identification of this K4 structural constant with the neutrino mass ratio
is discussed as an open question.

## Framework Axioms Invoked

- **A1** (self-inverse binary toggle): each edge of the srs walk is either on or
  off; no complex phase. This forces integer adjacency matrices.
- **A2-T** (MDL canonicalization; derived theorem, see `docs/theorems/theorem_A2_mdl_from_finite_register.md`): among all k-regular lattice graphs consistent
  with A1, the minimal description length criterion selects k* = 3 and the srs
  crystal net (predictions/k_star.py, predictions/d_spatial.py).

Both axioms enter through the upstream chain: A1 + A2-T → k* = 3 → srs lattice
→ K4 quotient → the spectral computation below.

## Derivation

### Step 1 — Coordination number and the K4 quotient

By A1 and A2-T, the coordination number is k* = 3 (predictions/k_star.py,
theorem-grade). The srs crystal net (Laves graph, space group I4₁32) is the
unique chiral k*=3 lattice with four vertices per primitive cell. Its translation
quotient is the complete graph on four vertices:

$$
\text{srs} / T \cong K_4
$$

K4 is 3-regular on 4 vertices (6 edges). Its adjacency spectrum is:

$$
\operatorname{spec}(A_{K_4}) = \{3,\, -1,\, -1,\, -1\}
$$

**Justification:** $A_{K_4} = J_4 - I_4$ where $J_4$ is the all-ones matrix.
Eigenvalues: $J_4$ has eigenvalue 4 (all-ones eigenvector) and 0 (multiplicity 3),
so $A_{K_4}$ has eigenvalues $4-1 = 3$ and $0-1 = -1$ (multiplicity 3). Standard
linear algebra.

### Step 2 — Ihara phase of the triplet eigenvalue

The Ihara-Bass identity (Bass 1992, Theorem 1) for a k-regular graph G with |V|
vertices and |E| edges reads:

$$
\det(I - uB) = (1 - u^2)^{|E|-|V|} \det\bigl(I - uA + u^2(k-1)I\bigr)
$$

For K4: |V| = 4, |E| = 6, k = 3. The second determinant factor is:

$$
\det\bigl((1 + 2u^2)I - u A_{K_4}\bigr)
$$

The triplet eigenvalue $\lambda = -1$ of $A_{K_4}$ contributes poles where
$(1 + 2u^2) = u \cdot (-1)$, i.e.:

$$
2u^2 + u + 1 = 0 \implies u = \frac{-1 \pm i\sqrt{7}}{4}
$$

(Discriminant: $1 - 8 = -7$; roots are complex with $|u|^2 = 1/(k-1) = 1/2$ — the
Ramanujan bound.) The Ihara phase is:

$$
\varphi := \arctan\!\left(\sqrt{7}\right), \qquad \cos\varphi = \frac{1}{\sqrt{8}}, \qquad \sin\varphi = \sqrt{\frac{7}{8}}
$$

where we choose the supplementary convention with $\cos\varphi > 0$ so that
$\varphi \in (0, \pi/2)$. The Ihara discriminant is $|D| = 4(k^*-1)-1 = 7$
(substituting k* = 3 into the standard formula for a $k$-regular graph).

### Step 3 — Chebyshev propagator and distance selection

For a k-regular graph with Ihara phase $\varphi$, the hop-$n$ propagator in the
triplet sector is (Terras 2011, §2.2):

$$
G_n = \frac{\sin(n\varphi)}{\sin\varphi} = U_{n-1}(\cos\varphi)
$$

where $U_{n-1}$ is the Chebyshev polynomial of the second kind. With
$\cos\varphi = 1/\sqrt{8}$ and $n = 5$:

$$
U_4(x) = 16x^4 - 12x^2 + 1
$$

$$
G_5 = U_4\!\left(\tfrac{1}{\sqrt{8}}\right)
    = 16 \cdot \frac{1}{64} - 12 \cdot \frac{1}{8} + 1
    = \frac{1}{4} - \frac{3}{2} + 1
    = -\frac{1}{4}
    = -\frac{1}{k^*+1}
$$

**Why n = 5 is uniquely selected.** The condition $G_n = -1/(k^*+1)$ with
$q = k^*-1$ is equivalent, via the Chebyshev recursion, to the polynomial
equation $q^3 = 5q - 2$. At $n = 5$ and $q = k^*-1 = 2$:

$$
2^3 = 8 = 5(2) - 2 = 8 \checkmark
$$

Uniqueness as a positive integer root: factor $x^3 - 5x + 2 = (x - 2)(x^2 + 2x - 1)$.
The quadratic factor has roots $x = -1 \pm \sqrt{2}$, both irrational. The only
positive integer root is $x = 2$, i.e., $k^* = 3$ is the unique coordination
number for which an integer hop distance $n = 5$ gives a clean propagator.

### Step 4 — Gaussian integer computation of sin²(5φ)

Set $z = 1 + i\sqrt{7}$. Then $\arg(z) = \varphi$ and $|z|^2 = 8$. Compute $z^5$
by sequential multiplication:

$$
z^2 = (1 + i\sqrt{7})^2 = 1 - 7 + 2i\sqrt{7} = -6 + 2i\sqrt{7}
$$

$$
z^3 = z^2 \cdot z = (-6 + 2i\sqrt{7})(1 + i\sqrt{7})
    = -6 - 6i\sqrt{7} + 2i\sqrt{7} + 2i^2 \cdot 7
    = -6 - 14 + (-6 + 2)i\sqrt{7}
    = -20 - 4i\sqrt{7}
$$

$$
z^4 = z^3 \cdot z = (-20 - 4i\sqrt{7})(1 + i\sqrt{7})
    = -20 - 20i\sqrt{7} - 4i\sqrt{7} - 4i^2 \cdot 7
    = -20 + 28 + (-20 - 4)i\sqrt{7}
    = 8 - 24i\sqrt{7}
$$

$$
z^5 = z^4 \cdot z = (8 - 24i\sqrt{7})(1 + i\sqrt{7})
    = 8 + 8i\sqrt{7} - 24i\sqrt{7} - 24i^2 \cdot 7
    = 8 + 168 + (8 - 24)i\sqrt{7}
    = 176 - 16i\sqrt{7}
$$

Since $z^5 = |z|^5 e^{5i\varphi}$ and $|z|^5 = (\sqrt{8})^5 = 128\sqrt{2}$:

$$
\sin(5\varphi) = \operatorname{Im}(e^{5i\varphi})
  = \frac{\operatorname{Im}(z^5)}{|z|^5}
  = \frac{-16\sqrt{7}}{128\sqrt{2}}
  = -\sqrt{\frac{7}{128}}
$$

$$
\sin^2(5\varphi) = \frac{7}{128}
$$

**Cross-check via G₅:** $\sin(5\varphi) = G_5 \cdot \sin\varphi
= (-\tfrac{1}{4})\sqrt{\tfrac{7}{8}} = -\sqrt{\tfrac{7}{128}}$. ✓

**Cross-check via double-angle:** $\cos(10\varphi) = 1 - 2\sin^2(5\varphi)
= 1 - \tfrac{14}{128} = \tfrac{57}{64} = T_{10}(1/\sqrt{8})$. This is exact
Chebyshev arithmetic (Bass 1992; Terras 2011 §2.2). ✓

### Step 5 — The structural constant R

$$
R := \frac{2}{\sin^2(5\varphi)} - (k^* + 1)
   = \frac{2}{\,7/128\,} - 4
   = \frac{256}{7} - \frac{28}{7}
   = \frac{228}{7}
$$

This is an exact rational number determined by k* = 3 alone. Every step from
Step 1 to Step 5 is explicit rational-radical arithmetic.

## Result

$$
R = \frac{228}{7} = 32.5714\overline{285714}\ldots
$$

**Numerical chain (k* = 3):**

| Quantity | Exact value | Float |
|---------|-------------|-------|
| φ | arctan(√7) | 1.2094 rad |
| cos φ | 1/√8 | 0.3536 |
| sin φ | √(7/8) | 0.9354 |
| sin²(5φ) | 7/128 | 0.0547 |
| R | 228/7 | 32.5714... |

## Comparison with Experiment

| Quantity | Value |
|---------|-------|
| Predicted R | 228/7 = 32.5714... |
| Observed R (NuFIT 6.0, Sept 2024) | 33.83 ± 0.92 |
| Absolute deviation | 1.26 |
| Deviation in σ | 1.4σ |

Source: NuFIT 6.0 (September 2024), normal ordering:
- Δm²₂₁ = (7.49 ± 0.19) × 10⁻⁵ eV²
- Δm²₃₁ = (2.534 ± 0.024) × 10⁻³ eV²
- R_obs = 2.534/0.0749 = 33.83, σ_R ≈ 0.92

The 1.4σ agreement is within normal statistical variation and obtained with
zero free parameters.

## Open Questions

1. **Physical identification of R.** The derivation establishes 228/7 as a
   structural constant of K4 via the Chebyshev propagator at the unique
   algebraically selected distance n = 5. The identification of this constant
   with Δm²₃₁/Δm²₂₁ relies on the physical interpretation in docs/parameters/R_theorem.md
   (§"Physical interpretation"): R is the "anisotropy of the propagator across
   the three Z3 channels minus the isotropic background k*+1." This
   interpretation is a physical argument, not a formal proof. Closing it would
   require proving that the neutrino mass-squared ratio is equal to the
   Chebyshev anisotropy in the Z3 triplet sector of the K4 quotient — a step
   that involves the identification of neutrino propagation with the K4 Green's
   function (the W1–W4 walker-identification scoping doc).

2. **The 1.4σ discrepancy.** docs/parameters/R_theorem.md §"Why no dark correction"
   argues that R is a topological invariant of K4, immune to dark-sector
   corrections. If future neutrino data (e.g., JUNO, Hyper-K) tightens Δm²₃₁,
   this discrepancy may grow or resolve. No mechanism is identified that would
   shift 228/7 toward the current central value.

3. **Sensitivity to k*.** The uniqueness of q = 2 as a positive integer root
   of q³ = 5q − 2 depends on k* = 3. The analogous construction for k* = 4
   (q = 3) gives no integer solution at n = 5; different lattices produce
   different structural constants, none of which match the neutrino data.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
