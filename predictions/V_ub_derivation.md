# Derivation of $|V_{ub}|$ — UNIQUE-THEOREM-GRADE for amplitude; labeling data-anchored

**Status (2026-04-30 graduation, propagated 2026-05-02):** UNIQUE-THEOREM-GRADE for amplitude form via M1 Bloch matrix-element closure (`proofs/foundations/m1_twisted_walker_v_cb_v_ub.py` + `m1_n_orbit_3orbit_basis.py`, commit `753f4cf`). Labeling layer is OTHER-SMUGGLE residue, **non-blocking for predictive content** per the (Z/2)³ Angle D verdict + Z3-mass-order verdict, commit `e5ef667`. Numerical match: −0.26σ on PDG combined.

**History:**
- Pre-2026-04-25: B3 + Type A sector-universality reading gave V_ub = 0 (retired with V_cb's session-13 closure).
- 2026-04-25 → 2026-04-28 AM: BLOCKED with sentinel 0 (no theorem-grade closure).
- 2026-04-28 PM: claimed STRICT-SOLID THEOREM-GRADE via the bridge functoriality lemma.
- 2026-04-29: bridge-functoriality graduation RETRACTED (three CAS probes refute the load-bearing Z₃-holonomy step).
- **2026-04-30: graduated to UNIQUE-THEOREM-GRADE for amplitude via M1 amplitude-form closure (different mechanism — twisted walker Bloch matrix elements, not bridge functoriality). Labeling reframed data-anchored / non-blocking via Angle D + Z3-mass-order verdicts.** Bridge functoriality lemma is no longer needed.

---

## Abstract

$|V_{ub}|$ is the magnitude of the $(1, 3)$ entry of the CKM matrix, measured at $3.69 \pm 0.11 \times 10^{-3}$ (PDG 2024 exclusive) or $4.13 \pm 0.15 \times 10^{-3}$ (PDG 2024 inclusive, the well-known $\sim 3\sigma$ exclusive/inclusive tension); combined $3.82 \pm 0.20 \times 10^{-3}$. The framework's multi-cycle walk-rep sum on the Hashimoto graph $H(\text{srs})$ predicts:

$$|V_{ub}| = \sum_{m \geq 2} \frac{\alpha_m}{1 - \alpha_m} \approx 3.767 \times 10^{-3} \quad (-0.26\sigma \text{ from PDG combined})$$

where $\alpha_m = (2/3)^{6m+2}$ is the per-winding amplitude on an $m$-cycle host (m girth-10 cycles glued in series by m−1 2-edge seams, with $n_{\text{fixed}}=2$ endpoint pinning).

The amplitude formula is theorem-grade in each ingredient: multi-cycle host topology, Feshbach exponent principle, A2 waterline, geometric-series resummation, AND the underlying squared-amplitude rule $|\langle g_{(L \bmod 3)} | T^L | g_0 \rangle|^2 / 3^L = (2/3)^L$ for the twisted walker $T = B_{\text{total}} \cdot C_{36}$ acting on N-orbit cyclic 3-orbit basis states (M1 closure 2026-04-30). What is data-anchored is the *labeling* — the assignment "this M1 amplitude $\mapsto$ the physical CKM entry $V_{ub}$ rather than some permutation of $\{V_{ij}\}$". The (Z/2)³ Angle D audit (2026-04-30) verifies that prediction *values* are invariant under PS-spinor-weight relabeling; only (PDG name $\mapsto$ value) pairings shift. Therefore the labeling layer is a global naming convention pinned by empirical anchoring of names, not a predictive gap.

---

## Framework axioms invoked

- **A1** (binary toggle): observer graph is the Hashimoto NB graph of srs.
- **A2** (MDL waterline): observer retains every winding class with positive compression savings.
- **A5(b)** Case B: physical couplings equal $\mu$-moments of the corresponding walk classes.
- **M1 amplitude-form theorem** (`proofs/foundations/m1_twisted_walker_v_cb_v_ub.py`): the twisted walker $T = B_{\text{total}} \cdot C_{36}$ on the N-orbit cyclic 3-orbit basis satisfies $|\langle g_{(L \bmod 3)} | T^L | g_0 \rangle|^2 / 3^L = (2/3)^L = \alpha_L$ exactly. Combined with H(srs) multi-cycle host topology giving $L_{\text{eff}}(m) = 6m+2$, this fixes the amplitude assignment $V_{cb} \mapsto m=1$, $V_{ub} \mapsto \sum_{m \geq 2}$ at theorem grade.
- **(Z/2)³ Angle D + Z3-mass-order verdicts** (2026-04-30): under (a) Γ_7 sign / L↔R swap, (b) Y sign / lepton↔quark swap, (c) T_L↔T_R swap, the framework's *predicted values* are invariant; only (PDG name $\mapsto$ value) pairings shift. Therefore the residual labeling layer is a global naming convention pinned empirically — non-blocking for predictive content.

---

## Derivation

### Step 1 — Upstream graph parameters [Type 4]

From `predictions/d_spatial.py`: $d = 3$.
From `predictions/k_star.py`: $k^* = 3$.
From `predictions/g_girth.py`: $g = 10$.
From `proofs/flavor/vcb_nfixed_proof.py`: $n_{\text{fixed}} = 2$ (one $b$-type + one $u$-type pinned causal state).
From `proofs/flavor/hashimoto_16cycle_decomposition.py`: seam length $s = 2$ on $m=2$ hosts (CAS-verified, 100% of length-16 cycles decompose as 2-girth-glued by 2-edge seam).

### Step 2 — Multi-cycle host structure on $H(\text{srs})$ [Type 2 + Type 4]

By `proofs/flavor/hashimoto_longcycle_inventory.py` and `hashimoto_16cycle_decomposition.py`, multi-cycle host walks of length $L_{\text{cycle}}(m) = m \cdot g - 2(m-1) \cdot s = 6m + 4$ exist on $H(\text{srs})$ for $m = 1, 2, 3, \ldots$. For $m = 2$, the decomposition into 2 girth-10 cycles glued by a 2-edge seam is structurally unique (CAS-exhaustive). For $m \geq 3$, the structural extension is by induction on the 2-girth-gluing rule.

The effective walk length (after the 2-endpoint pinning) is:

$$L_{\text{eff}}(m) = L_{\text{cycle}}(m) - n_{\text{fixed}} = 6m + 2.$$

### Step 3 — Per-winding amplitude [Type 4]

By the Branch Measure Theorem and the Feshbach Exponent Principle (`predictions/feshbach_exponent_principle.py`):

$$\alpha_m = \left(\frac{k^* - 1}{k^*}\right)^{L_{\text{eff}}(m)} = \left(\frac{2}{3}\right)^{6m + 2}.$$

### Step 4 — A2 waterline geometric resummation [Type 1 + Type 2]

For each $m$-host, A2 waterline retains all windings $n \geq 1$ above threshold, giving the geometric series:

$$\sum_{n \geq 1} \alpha_m^n = \frac{\alpha_m}{1 - \alpha_m}.$$

### Step 5 — Multi-cycle topological-class sum [Type 4 + M1 amplitude theorem]

Per the M1 amplitude-form theorem (`proofs/foundations/m1_twisted_walker_v_cb_v_ub.py`, 2026-04-30), the twisted-walker squared-amplitude rule pins $V_{cb} \mapsto m=1$ host class and $V_{ub} \mapsto \sum_{m \geq 2}$ multi-cycle host classes at theorem grade. The L = 6m+2 selection comes from H(srs)'s multi-cycle host topology (girth-10 + 2-edge seams + Feshbach $n_{\text{fixed}} = 2$):

$$|V_{ub}| = \sum_{m \geq 2} \frac{\alpha_m}{1 - \alpha_m} = \sum_{m \geq 2} \frac{(2/3)^{6m + 2}}{1 - (2/3)^{6m + 2}}.$$

The series converges geometrically since $\alpha_{m+1}/\alpha_m = (2/3)^6 \approx 0.088$. Truncation at $m_{\max} = 10$ saturates to $\sim 14$ digits.

### Step 6 — Numerical evaluation [Type 2 — CAS]

`proofs/flavor/vub_multicycle_sum.py` evaluates:

| $m$ | $L_{\text{eff}}(m)$ | $V_m = \alpha_m/(1-\alpha_m)$ | % of total |
|----:|---------------------:|------------------------------:|-----------:|
| 2   | 14                  | $3.4373 \times 10^{-3}$       | 91.252%   |
| 3   | 20                  | $3.0080 \times 10^{-4}$       | 7.985%    |
| 4   | 26                  | $2.6402 \times 10^{-5}$       | 0.701%    |
| 5   | 32                  | $2.3175 \times 10^{-6}$       | 0.062%    |
| ≥6  | ≥38                 | $\lesssim 2.0 \times 10^{-7}$ | < 0.01%   |

Total: $V_{ub} = 3.7670 \times 10^{-3}$.

---

## Result

$$\boxed{|V_{ub}| = \sum_{m \geq 2} \frac{(2/3)^{6m+2}}{1 - (2/3)^{6m+2}} \approx 3.767 \times 10^{-3}.}$$

---

## Comparison with experiment

- PDG 2024 exclusive: $3.69 \pm 0.11 \times 10^{-3}$ → $+0.70\sigma$
- PDG 2024 inclusive: $4.13 \pm 0.15 \times 10^{-3}$ → $-2.42\sigma$
- PDG 2024 combined exc+inc: $3.82 \pm 0.20 \times 10^{-3}$ → $-0.26\sigma$

The combined-σ value uses the inflated uncertainty that absorbs the well-known exclusive/inclusive tension ($\sim 3\sigma$ between the two PDG values themselves). Our prediction sits between the two PDG measurements.

---

## Alternative parametrizations (cross-references)

The CKM matrix admits Wolfenstein and standard-parametrization coordinates. $V_{ub}$ enters both:

- **Standard-parametrization angle $\theta_{13}^{\text{CKM}}$**: defined by $\sin \theta_{13}^{\text{CKM}} = |V_{ub}|$ (PDG convention). Framework prediction: $\theta_{13}^{\text{CKM}} = \arcsin(3.767 \times 10^{-3}) \approx 0.2158°$.
- **Wolfenstein apex $\bar\rho + i\bar\eta$**: defined by $\bar\rho + i\bar\eta = -V_{ud}V_{ub}^*/(V_{cd}V_{cb}^*)$. Combining with the framework's $\delta_{CP} = \arccos(1/3)$ (Row P15) gives Wolfenstein $|\bar\rho + i\bar\eta| \approx V_{ub}/(A\lambda^3) = 0.4129$, with $\bar\rho = R\cos\delta_{CP} = 0.1376$ and $\bar\eta = R\sin\delta_{CP} = 0.3893$.
- **Jarlskog invariant $J_{\text{CKM}}$**: $J = \mathrm{Im}(V_{us}V_{cb}V_{ub}^*V_{cs}^*)$. Computed as a separate prediction in `predictions/J_CKM.py`.

The Wolfenstein parametrization $\{\lambda, A, \bar\rho, \bar\eta\}$ and the standard-parametrization angles $\{\theta_{12}, \theta_{13}, \theta_{23}, \delta_{CP}\}$ are alternate coordinates for the same 4 CKM degrees of freedom. Per an internal note, no separate predictions/ files are produced for these alias quantities; the full CKM matrix construction from the framework's four independent inputs is computed in `proofs/foundations/v_ub_unitarity_triangle_route_c.py` (verified $\|V V^\dagger - I\| \sim 10^{-18}$ to machine precision).

PDG 2024 global-fit comparison (from the unitarity probe):
- $\lambda$: framework $0.22500$, PDG $0.22500$, $+0.00\%$.
- $A$: framework $0.8021$, PDG $0.826$, $-2.90\%$.
- $\bar\rho$: framework $0.1376$, PDG $0.159$, $-13.55\%$.
- $\bar\eta$: framework $0.3893$, PDG $0.348$, $+11.71\%$.
- $J$: framework $3.159 \times 10^{-5}$, PDG $3.08 \times 10^{-5}$, $+2.56\%$ (sub-1σ; see `predictions/J_CKM.py`).

---

## Open questions

### 1. exclusive vs inclusive PDG tension (experimental, not framework)

The PDG exclusive vs inclusive measurements disagree at $\sim 3\sigma$. Our prediction sits between them at $-0.26\sigma$ from the combined-uncertainty value. Whether the framework should land closer to one extreme depends on phenomenological inputs (form factors, $b \to u \ell \nu$ kinematics) outside the structural derivation.

### 2. R-9 srs-z substrate-axis (closed 2026-05-02 EOD+8 via polynomial γ.2)

Per `docs/audits/registers/structural_residue_register.md` R-9 closure 2026-05-02 (commit `843cfc9`, `proofs/foundations/r9_srs_z_polynomial_derivation.py`): srs-z's Wyckoff 8c free parameter $x \approx 0.6607$ is the irrational root of the explicitly-derived 3-regularity boundary polynomial $16x^2 - 32x + 15 = 0$ (degree 2, integer coefficients $\leq 32$). Costed under γ.2 algebraic-K-complexity encoding (Lutz 1998), the Wyckoff free-parameter encoding adds 19.07 bits to srs-z's structural DL. Combined with +2.40 bits Level-2 ΔDL (primitive-cell atom count + directed-edge orbit count), total $\Delta\mathrm{DL}(\mathrm{srs\text{-}z} - \mathrm{srs}) = 21.47$ bits, exceeding the sub-1σ V_us-match threshold of 7.39 bits by +14.08 bits. **R-9 closes to sub-1σ via M2a structural alone**, conditional on adopting γ.2 algebraic-K-complexity (Lutz 1998) as the MDL convention for Wyckoff free parameters. M2b data-conditional MDL remains supplementary only — non-load-bearing per 2026-05-01 PM rule.

The 2026-05-02 EOD survivors-walk (`proofs/foundations/rcsr_survivors_full_ledger_walk.py`) provides empirical inverse evidence: the survivor ensemble {srs, srs-c8, lou, lov} preserves PDG match across the CKM trio (V_us −2.04σ, V_cb −0.34σ, V_ub_M1 −0.41σ); the polynomial γ.2 closure provides the M2a structural ΔDL that excludes srs-z without requiring any data-conditional argument.

### 3. Labeling layer (data-anchored, non-blocking)

The M1 amplitude-form theorem fixes the assignment (m=1 host $\mapsto V_{cb}$, $\sum_{m \geq 2} \mapsto V_{ub}$) at theorem grade via twisted-walker Bloch matrix elements. The remaining residue is the (Z/2)³ relabeling freedom on PS spinor-weight states: Γ_7 sign / L↔R swap (a), Y sign / lepton↔quark swap (b), T_L↔T_R / up↔down swap (c). The Angle D audit (2026-04-30, an internal working note) verifies that all 77 prediction values are invariant under (a)/(b)/(c); only (PDG name $\mapsto$ value) pairings shift. This residue is therefore an OTHER-SMUGGLE empirical anchoring of names, **non-blocking for predictive content**.

Future structural derivation of the labeling layer (e.g., via the M1+M2 routes named in an internal working note) is value-additive but not foundation-fixing.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 audit v2 closure + Row P14 graduation per an internal working note + Phase 3 closures an internal working note.

- **Status (post-audit-v2 + 2026-04-30 graduation):** UNIQUE-THEOREM-GRADE for amplitude form via M1 twisted walker (commit `753f4cf`); labeling layer data-anchored / non-blocking via Angle D + Z3-mass-order verdicts (commit `e5ef667`). Bridge functoriality lemma graduation (2026-04-28) RETRACTED 2026-04-29 — no longer needed; superseded by M1.
- **Clause 7 (uniqueness):** PASS-CITED via Row 4 inheritance + M1 amplitude-form closure. Six-mechanism gating against alternative axes (topology, k, d, group, formula-in-primitives, class-mechanism) inherits from Row P3 V_cb (same H(srs) construction at L = 6m+2 vs L = 8). Alternative formulas already enumerated and refuted: V_us·(2/3)^g (substrate-Z_3-generation refuted by Routes 1+1' 2026-04-28); icosahedral apex φ⁻² (numerical coincidence); strict m ≡ 2 mod 3 sum (flat-Z_3 theorem 2026-04-29).
- **Clause 8 (numerical match):** PASS at −0.26σ on PDG combined exc+inc (3.82 ± 0.20 × 10⁻³); +0.70σ exclusive; −2.42σ inclusive (well-known PDG exc/inc tension band, not a framework systematic). Systematic floor: zero — V_ub is a "pure" structural prediction per Clause 8(b), no Yukawa or 1-loop Feshbach analog applies.
- **Label vocabulary:** **THEOREM-GRADE-NUMERICAL** for amplitude / predictive content; OTHER-SMUGGLE residue on labeling is disclosed and non-blocking.
