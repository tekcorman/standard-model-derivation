# Derivation of $|V_{td}|$ — THEOREM-GRADE-NUMERICAL via CKM unitarity (Type-4 inheritance)

**Status (created 2026-05-02 via parameter_linter combined cleanup walk):** THEOREM-GRADE-NUMERICAL via Type-4 inheritance from Rows P3, P4, P14, P15. Labeling layer is OTHER-SMUGGLE residue inherited from Row P14, NON-BLOCKING for predictive content per the (Z/2)³ Angle D verdict (commit `e5ef667`). Numerical match: +0.42σ on PDG 2024.

**Audit anchor:** Row P14 V_ub family (M1c) of `docs/parameters/parameter_uniqueness_ledger.md`. Helper module: `predictions/_ckm_unitarity.py`. Cross-verification: `proofs/foundations/v_ub_unitarity_triangle_route_c.py` (||V·V† − I|| ~ 1e-18 to machine precision).

---

## Abstract

The CKM matrix element $|V_{td}|$ is observed at $0.00854 \pm 0.00023$ (PDG 2024 (B_d mixing)). The framework predicts $|V_{td}| = 0.008636$ at $+0.42\sigma$ via Type-4 inheritance from the four independent framework-derived inputs

$$V_{us} = \frac{9}{40}, \quad V_{cb} = \frac{256}{6305}, \quad V_{ub} = \sum_{m \geq 2} \frac{(2/3)^{6m+2}}{1 - (2/3)^{6m+2}}, \quad \cos \delta_{CP} = \frac{1}{3},$$

combined via the standard Chau-Keung 1984 / PDG parameterization of a $3 \times 3$ unitary CKM matrix.

---

## Framework axioms invoked

This is a Type-4 closure: every step is proven in another `predictions/` file or a published mathematical theorem.

- **A1** (binary toggle), **A2** (MDL waterline), **A5(b)** Case B — inherited via the four upstream prediction files.
- **M1 amplitude-form theorem** (`proofs/foundations/m1_twisted_walker_v_cb_v_ub.py`, commit `753f4cf`, 2026-04-30) — fixes V_cb / V_ub amplitude assignments at theorem grade.
- **(Z/2)³ Angle D verdict** (an internal working note, commit `e5ef667`) — labeling layer non-blocking.
- **Chau-Keung 1984** ("Comments on the Parameterization of the Kobayashi-Maskawa Matrix", Phys. Rev. D 30, 1837) — standard parameterization of unitary CKM.
- **PDG CKM Review 2024** — current observed value.

---

## Derivation

### Step 1 — Upstream framework-derived inputs [Type 4]

From Rows P3, P4, P14, P15 of `docs/parameters/parameter_uniqueness_ledger.md`:

$$V_{us} = \frac{9}{40}, \qquad V_{cb} = \frac{256}{6305}, \qquad V_{ub} = \sum_{m \geq 2} \frac{(2/3)^{6m+2}}{1 - (2/3)^{6m+2}} \approx 3.767 \times 10^{-3},$$

$$\cos \delta_{CP} = \frac{1}{3}, \quad \sin \delta_{CP} = \frac{2\sqrt{2}}{3}.$$

### Step 2 — Standard-parameterization angles [Type 2 — algebra]

The PDG / Chau-Keung 1984 standard parameterization (positive-root branch):

$$s_{13} = V_{ub}, \qquad c_{13} = \sqrt{1 - V_{ub}^2},$$

$$s_{12} = \frac{V_{us}}{c_{13}}, \qquad c_{12} = \sqrt{1 - s_{12}^2},$$

$$s_{23} = \frac{V_{cb}}{c_{13}}, \qquad c_{23} = \sqrt{1 - s_{23}^2}.$$

These are pure algebraic identities — no fitting, no choice of branch beyond the conventional positive-root selection.

### Step 3 — CKM matrix element [Type 3 — Chau-Keung 1984]

The standard parameterization of the $3 \times 3$ unitary CKM matrix gives:

$$|V_{td}| = \big| s_{12} s_{23} - c_{12} c_{23} s_{13} e^{i\delta} \big|.$$

### Step 4 — Numerical evaluation [Type 2 — CAS]

Substituting Step 1's inputs into Steps 2-3:

$$|V_{td}| = 0.008636.$$

CAS-verified by `predictions/V_td.py` (assertion: implementation matches pure function to $10^{-12}$).

### Step 5 — Cross-verification via unitarity triangle [Type 2 — CAS]

The probe `proofs/foundations/v_ub_unitarity_triangle_route_c.py` constructs the full $3 \times 3$ CKM matrix from the four inputs and verifies:

$$\|V \cdot V^\dagger - I\|_F = 1.7 \times 10^{-18} \quad \text{(machine precision)},$$

confirming the matrix is unitary by construction.

---

## Result

$$\boxed{|V_{td}| = 0.008636.}$$

---

## Comparison with experiment

- PDG 2024 (B_d mixing): $|V_{td}| = 0.00854 \pm 0.00023$.
- Framework prediction: $|V_{td}| = 0.008636$.
- Deviation: $+0.42\sigma$.

---

## Open questions

### 1. Labeling layer (data-anchored, inherited from Row P14, non-blocking)

The four amplitudes that combine into $|V_{td}|$ each carry an OTHER-SMUGGLE residue: which (i,j) pair gets which framework amplitude is fixed by empirical anchoring of names rather than by structural derivation. The (Z/2)³ Angle D verdict (2026-04-30) verifies all 77 prediction values are invariant under PS-spinor-weight relabeling; only (PDG name → value) pairings shift. Therefore the labeling residue does not affect $|V_{td}|$'s numerical value — it is a global naming convention pinned empirically.

### 2. R-9 srs-z substrate-axis (closed 2026-05-02 EOD+8 via polynomial γ.2)

Per `docs/audits/registers/structural_residue_register.md` R-9 closure 2026-05-02 (commit `843cfc9`, `proofs/foundations/r9_srs_z_polynomial_derivation.py`): srs-z's Wyckoff 8c free parameter $x \approx 0.6607$ is the irrational root of the explicitly-derived 3-regularity boundary polynomial $16x^2 - 32x + 15 = 0$ (degree 2, integer coefficients $\leq 32$). Costed under γ.2 algebraic-K-complexity encoding (Lutz 1998), the Wyckoff free-parameter encoding adds 19.07 bits to srs-z's structural DL. Combined with +2.40 bits Level-2 ΔDL (primitive-cell atom count + directed-edge orbit count), total $\Delta\mathrm{DL}(\mathrm{srs\text{-}z} - \mathrm{srs}) = 21.47$ bits, exceeding the sub-1σ V_us-match threshold of 7.39 bits by +14.08 bits. **R-9 closes to sub-1σ via M2a structural alone**, conditional on adopting γ.2 algebraic-K-complexity (Lutz 1998 computable-real Kolmogorov complexity) as the MDL convention for Wyckoff free parameters. M2b data-conditional MDL remains supplementary only — non-load-bearing per 2026-05-01 PM rule.

### 3. PDG global-fit correlation

Some CKM PDG values come from global fits that use unitarity as a constraint (notably $V_{tb}$, $V_{ts}$, $V_{td}$). The framework's prediction is independent of unitarity at the input level (each of $V_{us}, V_{cb}, V_{ub}, \delta_{CP}$ is computed from its own structural mechanism), so any deviation versus PDG is genuine — not an artifact of double-counting unitarity. Where the PDG uncertainty is itself derived via unitarity, this can produce tighter quoted error bars and correspondingly larger nominal $\sigma$ values; this is noted in the Clause 8 disclosure.

---

## Audit v2 (Clause 7 + Clause 8) status

This prediction inherits Row 4 audit v2 closure + Rows P3/P4/P14/P15 graduation per an internal working note.

- **Clause 7 (uniqueness):** PASS-CITED via Type-4 algebraic inheritance from Rows P3, P4, P14, P15 + Chau-Keung 1984 (standard CKM parameterization is the unique unitary $3 \times 3$ form up to rephasing freedom). Six-mechanism gating against alternative axes inherits from each upstream row's closure.
- **Clause 8 (numerical match):** PASS. Systematic floor: zero — $|V_{td}|$ is a "pure" structural prediction per Clause 8(b).
- **Label vocabulary:** **THEOREM-GRADE-NUMERICAL** for the amplitude form and predictive content; OTHER-SMUGGLE residue on labeling is inherited from Row P14, disclosed and non-blocking.
