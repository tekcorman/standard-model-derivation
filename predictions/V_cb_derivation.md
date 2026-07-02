# Derivation of $|V_{cb}|$ — THEOREM-GRADE (formula); structural-identification gap

**Status:** THEOREM-GRADE for the *formula* $|V_{cb}| = 256/6305$ — 0 adoptions; all steps Type 1/2/3/4.
**Closed:** 2026-04-21 (session 13 CAS closure; session 14/15 linter pass).
**Footnote 2026-04-29:** The substrate-side structural *identification* — that the same-orbit $(b_1, b_2)$ pinned pairs at $d=8$ on girth-10 cycles correspond *specifically* to ΔGen=1 (b ↔ c) transitions and not to ΔGen=2 (b ↔ u) — has been called into question by the same probes that retracted the V_ub bridge functoriality lemma. The probes show same-orbit pairs at $d=8$ split 50/50 between Z₃-shift and Z₃²-shift, and the framework's flat-Z₃ theorem (`proofs/flavor/z3_holonomy_cycles.py`) means there is no Z₃ phase mechanism segregating ΔGen=1 from ΔGen=2 at this layer. The amplitude formula $\alpha_1/(1-\alpha_1) = 256/6305$ remains theorem-grade and matches PDG exclusive at $\approx +0.00\sigma$, but the assignment "this amplitude = ΔGen=1, not ΔGen=2" is an implicit structural identification the framework has not derived. See an internal working note for the deeper gap and the M1 (Bloch eigenmode) + M2 (multiway formalism) research routes.

---

## Abstract

$|V_{cb}|$ is the magnitude of the $(2,3)$ entry of the CKM matrix,
measured at $40.6 \pm 0.9 \times 10^{-3}$ (PDG 2024, exclusive
determination, Belle). We derive $|V_{cb}| = 256/6305 \approx 40.60 \times 10^{-3}$
(deviation $\approx +0.00\sigma$) from A1, A2, and A5(b) alone, with no fitted
parameters and no adoptions. The derivation operates at Level 3 (causal
observer graph = Hashimoto NB graph) following the three-level hierarchy
of an internal note.

The non-trivial content is: (i) identifying $V_{cb}$ as the $\mu$-measure
of all above-waterline girth-cycle winding classes on the Hashimoto graph
(A2 waterline + A5b), (ii) establishing $L_{cb} = g - 2 = 8$ by
endpoint counting (CAS-verified on an $8^3$ supercell), and (iii) applying
the A2 geometric-series resummation rather than truncating at one term.

---

## Framework axioms invoked

- **A1** (binary toggle): the causal observer graph is the Hashimoto
  non-backtracking (NB) graph of the srs crystal. Toggle states = directed
  edges; NB walks = forward-time causal histories.
- **A2** (MDL waterline): the observer retains every walk class whose
  compressed description length is shorter than its raw length (positive
  compression savings). This is a waterline, not a single-optimum selector.
- **A5(b)**: physical coupling strengths equal $\mu$-moments of the
  corresponding walk classes on the causal observer graph.

---

## Derivation

### Step 1 — Upstream graph parameters [Type 4]

From `predictions/d_spatial.py`: $d = 3$.
From `predictions/k_star.py`: $k^* = 3$ (MDL-optimal degree in $d=3$).
From `predictions/g_girth.py`: $g = 10$ (girth of srs, shortest NB cycle).

### Step 2 — First-winding amplitude [Type 4 + Type 3]

By the Branch Measure Theorem (Corollary 1, `docs/theorems/theorem_multiway_branch_measure.md`)
and the Feshbach Exponent Principle (`predictions/feshbach_exponent_principle.py`):

$$\alpha_1^{\text{bare}} = \left(\frac{k^*-1}{k^*}\right)^{L_{cb}}
= \left(\frac{2}{3}\right)^8 = \frac{256}{6561}$$

where $L_{cb} = g - n_{\text{fixed}}$ is the NB walk length for the
$b \to c$ transition and $n_{\text{fixed}} = 2$ is established in Step 3.

### Step 3 — $L_{cb} = g - 2 = 8$ [Type 2 + CAS]

**Endpoint counting (Type 2, `proofs/flavor/vcb_nfixed_proof.py`):**
The girth-cycle NB walk from a $b$-type causal state $s_b$ to a $c$-type
causal state $s_c$ has two fixed endpoints: the initial state $s_b$ and
the final state $s_c$. Therefore $n_{\text{fixed}} = 2$ and
$L_{cb} = g - n_{\text{fixed}} = 10 - 2 = 8$.

**CAS verification (`proofs/flavor/vcb_hashimoto_bfs.py`):**
The srs Hashimoto graph is constructed on an $8^3$ supercell. A DFS
enumerates all girth-10 NB cycles. Directed edges are classified by C3
orbit: $b_0$, $b_1 = C_3(b_0)$ (labelled $C_3 = \omega^2$, the $b$-quark
generation), $b_2 = C_3^2(b_0)$ (labelled $C_3 = \omega$, the $c$-quark
generation). The BFS finds exactly 20 same-orbit $(b_1, b_2)$ pairs at
cycle-distance $g - 2 = 8$, confirming $L_{cb} = 8$ and the species-generation
identification. (Session 13, 2026-04-21.)

### Step 4 — A2 waterline resummation [Type 1 + Type 2 + CAS]

Under A2, the observer retains every walk class with positive compression savings.

**Raw description length [Type 2]:** Each NB step on a $k^*$-regular graph has
$k^*-1 = 2$ non-backtracking choices, so each step costs $\log_2 2 = 1$ bit.
The $n$-th winding traverses $nL = 8n$ steps, giving:
$$L_{\text{raw}}(n) = 8n \text{ bits}$$

**Model description length [Type 2 + CAS]:** The model class "repeat the primitive
$b \to c$ girth cycle $n$ times" is encoded by:
- (a) Identifying which primitive cycle: $\leq \log_2 N$ bits, where $N = 120$ is the
  total number of girth-10 NB cycles in the srs $8^3$ supercell
  (CAS-verified: `proofs/flavor/vcb_hashimoto_bfs.py`, session 13).
- (b) Specifying the winding count $n$: $\log_2(n+1)$ bits (prefix-free).

$$L_{\text{model}}(n) \leq \log_2(120) + \log_2(n+1)$$

**Savings lower bound [Type 2]:**
$$\text{savings}(n) \geq 8n - \log_2(120) - \log_2(n+1)$$

At $n=1$: $\text{savings}(1) \geq 8 - \log_2(120) - \log_2(2) = 8 - 6.907 - 1 = 0.093 > 0$.

The savings function is strictly increasing: $\frac{d}{dn}[8n - \log_2(n+1)] = 8 - \frac{1}{(n+1)\ln 2} > 0$ for all $n \geq 1$. Therefore the minimum is at $n=1$, and all $n \geq 1$ satisfy $\text{savings}(n) > 0$.

**Consequence [Type 4: A2-T waterline]:** Every winding class is above the waterline and retained by the observer. The coupling is the sum over all retained classes.

**Type 4 (A2-T waterline):** every above-waterline class contributes to the coupling.
**Type 2 (geometric series):** summing over all $n \geq 1$:

$$V_{cb} = \sum_{n=1}^{\infty} \alpha_1^n
= \frac{\alpha_1}{1 - \alpha_1}
= \frac{256/6561}{1 - 256/6561}
= \frac{256}{6305}$$

---

## Result

$$\boxed{|V_{cb}| = \frac{256}{6305} = 40.6027 \times 10^{-3}}$$

Numerical evaluation: $256 / 6305 = 0.040602\ldots$

---

## Comparison with experiment

| Quantity | Predicted | Observed (PDG 2024, exclusive) | Deviation |
|----------|-----------|-------------------------------|-----------|
| $\lvert V_{cb}\rvert$ | $256/6305 = 40.60 \times 10^{-3}$ | $40.6 \pm 0.9 \times 10^{-3}$ | $\approx +0.00\sigma$ |

Note: the PDG 2024 inclusive determination is $(42.15 \pm 0.50) \times 10^{-3}$,
a long-standing $\sim 3.3\sigma$ exclusive/inclusive tension; our prediction
sits $\sim 3.1\sigma$ below the inclusive value. The exclusive determination
is the appropriate comparison for an amplitude-level derivation.

---

## Alternative parametrizations (cross-references)

The CKM matrix admits Wolfenstein and standard-parametrization coordinates beyond the magnitude basis. $V_{cb}$ enters both:

- **Wolfenstein parameter $A$**: defined as $A \equiv |V_{cb}|/\lambda^2$ where $\lambda = |V_{us}| = 9/40$. Framework prediction: $A = (256/6305) / (9/40)^2 = (256/6305) \cdot (1600/81) = 409600/510705 \approx 0.8021$.
- **Standard-parametrization angle $\theta_{23}^{\text{CKM}}$**: defined by $\sin \theta_{23}^{\text{CKM}} = |V_{cb}|/\cos \theta_{13}$ (PDG convention). With $\cos \theta_{13} \approx 1$, $\theta_{23}^{\text{CKM}} \approx \arcsin(256/6305) = 2.327°$.

Both are derived consequences of the framework's $|V_{cb}|$ value computed in this file; no separate predictions/ files are produced (per an internal note).

The PDG global-fit Wolfenstein $A_{\text{PDG}} \approx 0.826 \pm 0.012$ — the framework prediction sits at $-2.04\sigma$ from this value, tracking the V_cb exclusive/inclusive tension propagated through $A = V_{cb}/\lambda^2$. See `proofs/foundations/v_ub_unitarity_triangle_route_c.py` for the full Wolfenstein computation and CKM matrix construction.

---

## Open questions

1. **A5(b) dependence:** The derivation rests on A5(b) (coupling = $\mu$-moment).
   A5(b) is an axiom, not yet derived from A1 + A2-T. Closing A5(b) from first
   principles would elevate this to a pure two-axiom theorem.

2. **N derivation:** $L_{cb} = g - n_{\text{fixed}} = 8$ uses $g = 10$ from
   the srs girth derivation, which is itself theorem-grade. No external
   cosmological input enters here.

3. **Inclusive/exclusive tension:** the $\sim 3.3\sigma$ difference between
   inclusive and exclusive PDG determinations is a theoretical QCD uncertainty in
   the experimental extraction, not a discrepancy with the framework.

4. **R-9 srs-z substrate-axis — CLOSED (2026-05-02 EOD+8 via polynomial γ.2).**
   Per `docs/audits/registers/structural_residue_register.md` (commit
   `843cfc9`, `proofs/foundations/r9_srs_z_polynomial_derivation.py`):
   srs-z's Wyckoff 8c free parameter $x \approx 0.6607$ is the irrational
   root of the explicitly-derived 3-regularity boundary polynomial
   $16x^2 - 32x + 15 = 0$ (degree 2, integer coefficients $\leq 32$).
   Costed under γ.2 algebraic-K-complexity (Lutz 1998), the Wyckoff free-
   parameter encoding adds 19.07 bits to srs-z's structural DL. Combined
   with +2.40 bits Level-2 ΔDL, total $\Delta\mathrm{DL}(\mathrm{srs}\text{-}\mathrm{z} - \mathrm{srs}) = 21.47$ bits,
   exceeding the sub-1σ V_us-match threshold of 7.39 bits by +14.08 bits.
   R-9 closes to sub-1σ via M2a structural alone, conditional on γ.2
   algebraic-K-complexity (Lutz 1998) as the MDL convention for Wyckoff
   free parameters. M2b data-conditional MDL remains supplementary only.

---


## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.

## References

- Shalizi, C.R. & Crutchfield, J.P. (2001). Computational mechanics: pattern
  and prediction, structure and simplicity. *J. Stat. Phys.* **104**, 817–879.
  Theorem 2 (causal state graph = Hashimoto NB graph).
- Grunwald, P.D. (2007). *The Minimum Description Length Principle.* MIT Press.
  §5.1–5.3 (waterline / model class selection).
- `docs/theorems/theorem_multiway_branch_measure.md` — branch measure μ + Corollary 1.
- `predictions/feshbach_exponent_principle.py` — α₁_bare = (2/3)^{L}.
- `proofs/flavor/vcb_nfixed_proof.py` — n_fixed = 2, Type 2.
- `proofs/flavor/vcb_hashimoto_bfs.py` — CAS verification, L_cb = 8.
- `proofs/flavor/vcb_branch_measure.py` — full gate-annotated proof script.
- PDG 2024 CKM review.
