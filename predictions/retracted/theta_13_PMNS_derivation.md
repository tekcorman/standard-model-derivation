# θ_13_PMNS — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; see docs/framework_axioms.md), G.1 and G.5 are now DERIVED via CDP 2011 (predictions/observer_hilbert_space.py), but B6 retraction and the V_us block remain load-bearing here.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation's Step 1 identifies the C_3-charged eigenvectors `|ω⟩, |ω²⟩` of A(P) with a generation-mixing structure of `U_TBM` — specifically, the third column of U_TBM `(0, 1/√(k*-1), 1/√(k*-1))` is constructed as the "generation-mixed" component orthogonal to the fixed-vertex `|v_0⟩` direction, with the two nonzero entries implicitly labelling generations 2 and 3. B6 (`docs/theorem_B6_bridge.md`) proves the srs body-diagonal C_3 is the color-Z_3 of SU(3)_c via Spin(6)≅SU(4)→PS embedding, not a generation label. Under B6, `|ω⟩` and `|ω²⟩` label color components within ONE Pati-Salam family, not three neutrino generations, so the TBM third-column construction as a generation-mixing structure fails.

Separately, Step 3 imports `V_us` from `predictions/V_us_derivation.md`, which is itself BLOCKED (see `docs/master_plan.md` Sprint 2 for the post-B3+B6 V_us retirement).

**Re-derivation target**: Sprint 11 workstream B7.5 (PMNS/CKM under C³_gen; see `docs/master_plan.md` §Sprint 11). Under Sprint 11, PMNS mixing emerges from the mismatch between the neutrino- and charged-lepton mass operators on C³_gen, which is orthogonal to the srs C_3 color structure.

**What survives as math**: the A(P) spectral decomposition into C_3-charged eigenvectors `|ω⟩ = (0, 1, ω, ω²)/√3` and `|ω²⟩ = (0, 1, ω², ω)/√3` is a rigorous color-sector spectral fact about the srs P-point (`docs/theorem_BP_doubly_degenerate_h.md` Step 3). The construction of the rank-1 vector `(0, 1, 1)/√(k*-1)` from an even superposition of `|ω⟩ + |ω²⟩` in the restricted 3-vertex subspace is label-agnostic algebra. Only the identification of this vector with a generation-mixing column of the PMNS matrix is retracted.

## Specific failing step

Step 1 constructs the tribimaximal third column from the C_3-charged eigenvectors at the P-point:

> "the third column of the leptonic mixing matrix is orthogonal to the fixed-vertex $|v_{0}\rangle$ direction and equally distributes the remaining weight across $\{v_{1}, v_{2}, v_{3}\}$ with the trivial-rep phase pattern $(0, 1, 1)/\sqrt{k^{*}-1}$."

This identifies a rank-1 combination of `|ω⟩` and `|ω²⟩` (both of which are C_3-charged srs eigenvectors at P) with the `e`-`ν_3` neutrino mixing component. Under B6, `|ω⟩` and `|ω²⟩` are color labels, so the "mixing" being constructed is between color components of one family, not between neutrino mass eigenstates. In addition, Step 3 invokes the V_us identification `|(U_l)_{2,1}| = V_us` from `V_us_derivation.md`, which is independently BLOCKED.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (NuFIT 6.0, NO) | Status |
|---|---|---|---|
| θ_13 (route i, dark-corrected V_us) | 9.24° | 8.57° ± 0.13° | not explanatory under current framework (+5σ tension) |
| θ_13 (route ii, bare V_us × (1-α_1)) | 8.61° | 8.57° ± 0.13° | not explanatory under current framework (within 1σ) |

Neither route derives θ_13 under post-B6 rigor; both depend on the color-label-as-generation identification retired by B6 and on the independently BLOCKED V_us.

## Preserved original derivation (for reference; superseded)

---

# Derivation of theta_13 (PMNS reactor mixing angle) (SUPERSEDED, retained for reference)

## Abstract

We derive
$$\sin\theta_{13} \;=\; \frac{|V_{us}|}{\sqrt{k^{*}-1}}$$
with $k^{*} = 3$ and $V_{us}$ as produced by `predictions/V_us.py`. The derivation is the Type-D Class-3 ("edge-local") entry of the W4 identification catalog (`docs/W4_identification_catalog.md` §2D): the tribimaximal baseline $\theta_{13}^{\text{TBM}} = 0$ is a theorem inherited from the $C_{3}$-protected double degeneracy of $A(P)$ (`docs/theorem_BP_doubly_degenerate_h.md` Step 3), and the dark-correction coefficient is forced to $c=1$ by the character-orthogonality identity $\mathrm{Tr}(\sigma_{x}) = 0$ at a $C_{3}$-symmetric vertex (Serre 1977 §2.4 Theorem 3). The Class-3 absorption is already carried by $V_{us}$ itself (same mechanism, `predictions/V_us_derivation.md` §Step 4), so no additional $(1-\alpha_{1})$ factor is applied on top of $V_{us}$. The predicted value is $\theta_{13} = 9.24^\circ$, against the observed $8.57\pm 0.13^\circ$ (NuFIT 6.0); this is a $+5\sigma$ tension that tracks a pre-existing $+1\sigma$ tension in $V_{us}$, not a defect introduced at this step — see Open Questions.

## Framework axioms invoked

No new axioms are introduced here.

- **(A1)** Binary self-inverse toggle, **(A2)** MDL compression.
- **Theorem** `docs/theorem_walker_dynamics.md`, **Theorem** `docs/theorem_BP_doubly_degenerate_h.md`.
- **Upstream prediction** `predictions/V_us.py` / `V_us_derivation.md` — $V_{us}$ from Pati–Salam SU(4) perpendicularity. [BLOCKED.]
- **Classification** `docs/W4_identification_catalog.md` §2D Class 3 — edge-local, coefficient $c=1$.

## Derivation

### Step 1. TBM baseline $\theta_{13}^{\text{TBM}} = 0$ [FAILING STEP under B6]

By `docs/theorem_BP_doubly_degenerate_h.md` Step 1, the scalar Bloch adjacency at the $P$-point has characteristic polynomial $(\lambda^{2}-3)^{2}$. By Step 3, each 2-dim eigenspace decomposes under C_3 as (trivial + charged), with charged piece $\omega$ for +√3 and $\omega^2$ for −√3. The two C_3-charged eigenvectors are
$$|\omega\rangle \;=\; \frac{1}{\sqrt{3}}(0, 1, \omega, \omega^{2})^{\!T}, \qquad |\omega^{2}\rangle \;=\; \frac{1}{\sqrt{3}}(0, 1, \omega^{2}, \omega)^{\!T}.$$

**The failing identification under B6**: the derivation proceeds to treat these C_3-charged eigenvectors as generation-mixing components, constructing the third column of U_TBM as `(0, 1/√(k*-1), 1/√(k*-1))`. Under B6, `|ω⟩` and `|ω²⟩` are color labels, not generation labels.

### Step 2. PMNS factorisation and the 1–2 charged-lepton rotation

$U_{\text{PMNS}} = U_{l}^{\dagger}\, U_{\text{TBM}}$, giving $\sin\theta_{13} = |(U_{l})_{2,1}| / \sqrt{k^{*}-1}$ in the small-mixing limit.

### Step 3. Identification of $|(U_{l})_{2,1}|$ with $V_{us}$ [BLOCKED upstream]

`predictions/V_us_derivation.md` closes $|(U_{l})_{2,1}| = V_{us}$; this upstream is BLOCKED.

### Step 4. Class-3 edge-local dark correction has coefficient $c = 1$

At a $C_{3}$-symmetric vertex, character orthogonality forces $c = 1$.

### Step 5. Absorption of the Class-3 coefficient into $V_{us}$

To avoid double-counting, no additional $(1-\alpha_1)$ factor is applied on top of V_us:
$$\sin\theta_{13} \;=\; \frac{V_{us}}{\sqrt{k^{*}-1}}.$$

### Step 6. Numerical evaluation

With $k^{*} = 3$ and $V_{us} = 0.22707$:
$$\sin\theta_{13} \;=\; 0.16056, \qquad \theta_{13} \;=\; 9.24^\circ.$$

## Result (SUPERSEDED under B6)

$$\theta_{13} \;=\; 9.24^\circ.$$

## References

- Serre, J.-P. (1977). *Linear Representations of Finite Groups*. Graduate Texts in Mathematics 42. Springer.
- Esteban, I. *et al.* (NuFIT collaboration). *NuFIT 6.0* (September 2024). http://www.nu-fit.org.
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
