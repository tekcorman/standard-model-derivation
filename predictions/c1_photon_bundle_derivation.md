# c₁ — First Chern Class of the Photon Hodge Bundle

## Abstract

We derive c₁ = 0 for the first Chern class of the photon Hodge bundle on the
srs Brillouin zone. The core argument is a three-step topological proof: (1)
the Bloch incidence matrix satisfies d(−k) = d(k)* (time-reversal symmetry),
a direct consequence of real-valued hopping amplitudes forced by axiom A1;
(2) complex conjugation reverses the Chern number (standard Chern-Weil theory);
(3) on self-conjugate BZ slices, c₁ = −c₁, forcing c₁ = 0. The result is
extended to all BZ slices via constancy of the Chern number under continuous
deformation, with the Γ-point defect verified numerically to carry zero
topological charge. One analytical gap remains: the complete absence of Weyl
points away from Γ has been verified numerically but not proven symbolically.

## Framework Axioms Invoked

- **A1** (self-inverse binary toggle): each edge carries a real sign (±1) with
  no complex phase. This is the sole source of the T-symmetry in Step 1; the
  entire proof depends on it.
- **A2** (MDL): selects k* = 3 and the srs lattice geometry (predictions/k_star.py).

## Derivation

### Step 1 — Time-reversal symmetry of the Bloch incidence matrix

The srs Bloch incidence matrix $d(\mathbf{k}): \mathbb{C}^4 \to \mathbb{C}^6$
is constructed from the primitive-cell edge list. Each entry has the form:

$$
d(\mathbf{k})_{e,v} = \epsilon_{ev} \, e^{-2\pi i \mathbf{k} \cdot \mathbf{R}_{ev}}
$$

where $\epsilon_{ev} \in \{+1, -1\}$ is the signed incidence and
$\mathbf{R}_{ev} \in \mathbb{Z}^3$ is the cell displacement. Under
$\mathbf{k} \to -\mathbf{k}$, each phase factor becomes its complex conjugate:

$$
e^{-2\pi i (-\mathbf{k}) \cdot \mathbf{R}} = e^{+2\pi i \mathbf{k} \cdot \mathbf{R}} = \overline{e^{-2\pi i \mathbf{k} \cdot \mathbf{R}}}
$$

Since $\epsilon_{ev} \in \mathbb{R}$, it is unaffected by conjugation. Therefore:

$$
d(-\mathbf{k}) = \overline{d(\mathbf{k})} \quad \text{(element-wise)}
$$

**Justification:** This is an immediate consequence of the definition of
$d(\mathbf{k})$ applied element-wise. No graph-specific information is needed;
the argument holds for any tight-binding model with real hopping. Symbolically
verified in sympy via $A(-k) - A(k)^* = 0$ for the scalar Bloch adjacency
(an internal working note §Step 1; Ashcroft & Mermin, *Solid State
Physics*, Ch. 8).

### Step 2 — The photon bundle at −k is the complex conjugate of the bundle at k

The photon Hodge subspace at $\mathbf{k}$ is $\ker d^\dagger(\mathbf{k}) \subset \mathbb{C}^6$,
2-dimensional at generic $\mathbf{k}$. Using Step 1:

$$
d^\dagger(-\mathbf{k}) = \overline{d(-\mathbf{k})}^\top = \overline{\overline{d(\mathbf{k})}}^\top = d(\mathbf{k})^\top
$$

For any $\psi \in \ker d^\dagger(\mathbf{k})$ (i.e., $d(\mathbf{k})^\dagger \psi = 0$):

$$
d^\dagger(-\mathbf{k})\,\overline{\psi} = d(\mathbf{k})^\top \overline{\psi} = \overline{d(\mathbf{k})^\dagger \psi} = \overline{0} = 0
$$

Therefore $\overline{\psi} \in \ker d^\dagger(-\mathbf{k})$. Taking an orthonormal
basis $\{\psi_1, \psi_2\}$ of $\ker d^\dagger(\mathbf{k})$ gives an orthonormal
basis $\{\overline{\psi_1}, \overline{\psi_2}\}$ of $\ker d^\dagger(-\mathbf{k})$.

**Conclusion:** The photon bundle $E \to T^3$ satisfies
$E_{-\mathbf{k}} = \overline{E_{\mathbf{k}}}$ as a bundle over $T^3$.

### Step 3 — Complex conjugation reverses the Chern number

For a complex Hermitian vector bundle $E \to X$, the Berry connection transforms
under complex conjugation as $A \to \bar{A} = -A^*$ (the conjugate bundle $\bar{E}$
has the conjugate transition functions $\bar{g}_{\alpha\beta}$, giving conjugate
connection 1-forms). The curvature satisfies $\bar{F} = -F^*$.

For the first Chern class:

$$
c_1(\bar{E}) = \frac{i}{2\pi} \int_{\Sigma} \operatorname{tr}\bar{F}
             = \frac{i}{2\pi} \int_{\Sigma} \operatorname{tr}(-F^*)
             = -\overline{\left(\frac{i}{2\pi}\int_\Sigma \operatorname{tr} F\right)}
             = -c_1(E)
$$

where the last step uses $c_1(E) \in \mathbb{Z}$ (so $c_1(E)^* = c_1(E)$).

**Justification:** Nakahara, *Geometry, Topology and Physics* (2nd ed., 2003),
§11.1 (Chern classes of conjugate bundles); also Milnor & Stasheff, *Characteristic
Classes* (1974), Problem 14-B. The sign flip is an algebraic consequence of the
definition of $c_1$ via the curvature form.

### Step 4 — Self-conjugate slices force c₁ = 0

For any axis $l \in \{1, 2, 3\}$, consider the 2D BZ slice at fixed $k_l$:

$$
\Sigma_{k_l} = \{(k_1, k_2, k_l) : k_1, k_2 \in T^1\} \cong T^2
$$

The T-map $\mathbf{k} \to -\mathbf{k}$ sends $\Sigma_{k_l} \to \Sigma_{-k_l}$.
The slice is **self-conjugate** (invariant as a set under T) iff $k_l \equiv -k_l$
mod 1, i.e., $k_l \in \{0, \tfrac{1}{2}\}$.

On a self-conjugate slice, T is an involution of $T^2$. By Step 2, the
photon bundle on $\Sigma_{k_l}$ satisfies $E|_{\Sigma_{k_l}} \cong \overline{E|_{\Sigma_{k_l}}}$
as bundles over $\Sigma_{k_l}$ (the T-involution interchanges each fibre
$E_\mathbf{k}$ with $E_{-\mathbf{k}} = \overline{E_\mathbf{k}}$, which lies
on the same slice). Applying Step 3:

$$
c_1\bigl(E|_{\Sigma_{k_l}}\bigr) = c_1\bigl(\overline{E|_{\Sigma_{k_l}}}\bigr) = -c_1\bigl(E|_{\Sigma_{k_l}}\bigr)
$$

The first equality uses $E|_{\Sigma} \cong \overline{E|_\Sigma}$; the second
uses Step 3. Therefore:

$$
c_1(\Sigma_0) = 0 \quad \text{and} \quad c_1(\Sigma_{1/2}) = 0
$$

for every axis direction.

### Step 5 — c₁ is piecewise constant with no charged defects

The integer $c_1(k_l)$ is constant under continuous deformation of the 2D slice,
changing only when a topological defect (Weyl point) crosses the slice. This is
the standard Chern number cobordism argument: $c_1$ is a homotopy invariant of
the bundle restricted to the slice, and it changes by the topological charge of
any singularity that the slice passes through as $k_l$ varies.

For the srs photon bundle:

- At the Γ-point $\mathbf{k} = 0$, $\dim \ker d^\dagger(\Gamma) = 3$ (rank
  drops by 1), creating a potential defect. The topological charge was computed
  by sphere integration of the Berry curvature around Γ in
  `proofs/cosmology/srs_gamma_defect_charge.py` on a $32 \times 48$ grid:

  | sphere radius | $|c_1^{\text{sphere}}|$ |
  |--------------|------------------------|
  | 0.01 | $1.6 \times 10^{-4}$ |
  | 0.02 | $3.2 \times 10^{-4}$ |
  | 0.05 | $7.9 \times 10^{-4}$ |

  All consistent with zero to numerical precision. **The Γ defect carries zero
  topological charge.**

- No Weyl points were detected at any other $\mathbf{k}$ in the BZ in numerical
  sampling (`proofs/cosmology/srs_photon_berry.py`).

**Open gap (noted):** The complete absence of Weyl points away from Γ is
confirmed numerically but not yet proven by an analytic argument (e.g., showing
$\det(d(\mathbf{k}) d(\mathbf{k})^\dagger) \neq 0$ for all $\mathbf{k} \neq \Gamma$
via a resultant computation). This is the one remaining gap before the proof is
fully symbolic.

### Step 6 — Conclusion

- $c_1(k_l = 0) = 0$ (Step 4, for every axis)
- $c_1(k_l = 1/2) = 0$ (Step 4, for every axis)
- $c_1(k_l)$ integer-valued, piecewise constant, no charged defects in between (Step 5)

Therefore $c_1(k_l) = 0$ for all $k_l \in [0, 1)$ and all slice axes. **QED.**

## Result

$$
c_1 = 0
$$

on every 2D slice of the srs Brillouin zone. The photon Hodge bundle is
topologically trivial in the first Chern class sense.

**Physical consequence:** No bulk axion angle of the form $\theta F \tilde{F}$
can arise from this bundle's topology. Any observed cosmic birefringence β must
arise from a dynamical mechanism — specifically the dark sector correction
β = sin(arg h) · α_EM (predictions/B_P_doubly_degenerate_h.py).

## Comparison with Experiment

| Quantity | Value |
|---------|-------|
| Predicted c₁ | 0 (exact) |
| Observed | 0 (no anomalous photon Berry phase measured) |
| Deviation | 0 |

This is a topological invariant, not a parameter fit. Numerical verification:
- $16 \times 16$ grid at 10 values of $k_z$: $|c_1| < 6 \times 10^{-4}$
- $24 \times 24$ grid at 5 values of $k_z$: $|c_1| < 3 \times 10^{-6}$
- Sphere integration around Γ: $|c_1^{\text{sphere}}| < 10^{-3}$

All consistent with $c_1 = 0$ to numerical precision.

## Open Questions

1. **Analytic proof of "no Weyl points except Γ."** Step 5 uses numerical
   sampling to establish the absence of topological defects away from Γ.
   Closing this analytically — e.g., via a symbolic determinant computation
   showing $\operatorname{rank}(d(\mathbf{k})) = 4$ for all $\mathbf{k} \neq \Gamma$
   — would make the proof entirely symbolic. Estimated effort: half a session
   of sympy computation on the explicit $4 \times 6$ Bloch incidence matrix.
   Until closed, the derivation is theorem-grade modulo this one numerically
   verified step.

2. **Higher Chern classes.** The argument is for $c_1$ (Abelian / U(1) content).
   The photon bundle is rank 2 (U(2)); its second Chern class $c_2$ is not
   addressed here. Non-zero $c_2$ would not affect $c_1 = 0$ but would imply
   non-trivial SU(2) structure.

3. **Non-Hodge photon bundle.** The analysis uses the Hodge (coexact) photon
   subspace $\ker d^\dagger$. If a different gauge choice for the photon Hilbert
   space is adopted (e.g., exact 1-forms $\operatorname{im} d$), the argument
   would need to be re-done for that bundle.
