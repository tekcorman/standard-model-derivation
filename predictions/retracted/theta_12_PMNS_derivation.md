# Derivation of $\theta_{12}$ (PMNS solar mixing angle) — STATUS: BLOCKED (updated 2026-04-17 under B6)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; see docs/framework_axioms.md), G.1 and G.5 are now DERIVED via CDP 2011 (predictions/observer_hilbert_space.py), but the B6 color-vs-generation retraction and the V_us block remain load-bearing here.

## B6 retraction update (2026-04-17)

This file was already BLOCKED pre-session on V_us transitive grounds. Theorem B6 (`docs/theorem_B6_bridge.md`) adds a **second, independent** retraction reason that applies regardless of whether V_us is ever unblocked:

**Additional failing step under B6.** The TBM baseline in Step 1 / eq. (1) (`θ_TBM = arctan(1/√(k*−1)) ≈ 35.26°`) is derived from the $(4, 2, 2)$ C_3-isotypic decomposition of the 8-dim Ramanujan subspace at the P-point, treating the three C_3 irreps $\{\text{triv}, \omega, \omega^2\}$ as three physical neutrino generations. B6 proves the srs body-diagonal C_3 is the **color-Z_3** of SU(3)_c via Spin(6)≅SU(4)→PS, not a generation index. Under B6, the three C_3 irreps label color components within one Pati-Salam family, not three neutrino generations. The TBM baseline's 1-2 sector identification therefore loses its physical basis.

**Re-derivation target.** Sprint 11 workstream B7.5 (see `docs/master_plan.md` §Sprint 11). Under the C³_gen framework (`docs/theorem_observer_dim_three.md` + `docs/theorem_generation_C3_bridge.md`), PMNS angles emerge as mismatch between mass-operator eigenbases for charged-lepton vs neutrino sectors on the observer's C³_gen Hilbert space. $\theta_{12}$ specifically would be the solar-sector Euler angle of that mismatch.

**What survives as math.** The spherical Pythagorean arithmetic $\cos\theta_{12} = \cos\theta_\text{TBM}/\cos\theta_C$ (Route B, eq. 4) is a standard SU(3)-manifold identity (conditional on Killing-form perpendicularity — **B-Gap 1, 2** below). The Killing-form calculation $B(T_C, T_\text{TBM}) = 0$ in SU(4) (`proofs/flavor/srs_theta12_perp.py`) remains a valid Lie-algebra identity. Neither survives as a charged-lepton / neutrino PMNS derivation under B6.

The original derivation body below (Abstract through Section 5) is retained for historical reference; it documents the pre-B6 blocking structure (A-Gap 1 on $L_{us} = 2 + \sqrt{3}$, B-Gaps 1–4 on Pati-Salam perpendicularity identifications) which remains operative in addition to the B6 retraction.

---

## Abstract

$\theta_{12}$ is the PMNS solar mixing angle; experimentally $\theta_{12} = 33.68° \pm 0.73°$ (NuFIT 6.0, normal ordering, September 2024; equivalently $\sin^{2}\theta_{12} = 0.308^{+0.012}_{-0.011}$). Within the present framework it is catalogued as **Type D** in `docs/W4_identification_catalog.md` §2D (mixing angle = tribimaximal baseline + dark-extraction correction). The only route currently implemented — the Pati–Salam SU(4) perpendicularity chain of `predictions/V_us.py` (Route B) — derives $\theta_{12}$ *downstream* of $V_{us}$ and therefore inherits every open gap of `predictions/V_us_derivation.md`.

Under the rigor bar (`docs/parameter_linter.md` "Hard quality gate", an internal note) this parameter is **BLOCKED transitively on $V_{us}$**. The accompanying `predictions/theta_12_PMNS.py` script evaluates the Route-B arithmetic at the bare $V_{us}^{\text{bare}} = (2/3)^{2+\sqrt{3}}$ (so that it is upstream-identical to `proofs/foundations/theta_12_PMNS_derivation.py`), gets $\theta_{12} \approx 33.17°$ ($-0.70\sigma$ vs NuFIT 6.0), and prints `OK: outputs agree` together with an explicit BLOCKED classification.

The two new closures since the last audit — `docs/theorem_uniform_Q_density.md` Part A (theorem, $\rho_{Q}$ uniform) and `docs/theorem_Feshbach_coupling_strength.md` (Lemma 1 + the Exponent Principle gives $\alpha_{1} = (2/3)^{g-2}$) — tighten upstream dependencies for Class-1 and Class-2 dark corrections *in general* but do not provide a $V_{us}$-free path to $\theta_{12}$ (see §5 below for the reason).

## Framework axioms invoked

No new foundational axioms are introduced. The derivation consumes upstream results (each listed with its status under `docs/parameter_linter.md`):

- **(A1)** Binary self-inverse toggle (`predictions/p_toggle.py`). *Closed.*
- **(A2)** MDL compression (`predictions/d_spatial.py`, `predictions/k_star.py`). *Closed* — gives $k^{*} = 3$, $d = 3$.
- **Theorem** `docs/theorem_walker_dynamics.md`. *Closed* — walker dynamics on srs are non-backtracking walks with Hashimoto $B$ as the one-step operator.
- **Theorem** `docs/theorem_BP_doubly_degenerate_h.md`. *Closed* — $A(P)$ has characteristic polynomial $(\lambda^{2}-3)^{2}$; eigenvalues $\pm\sqrt{3}$ with multiplicity 2 each; $B(P)$ has eigenvalue $h = (\sqrt{3}+i\sqrt{5})/2$ with multiplicity 2, and each $\pm\sqrt{3}$ A-eigenspace decomposes under $C_{3}$ as $\text{trivial} \oplus \omega$ (resp. $\text{trivial} \oplus \omega^{2}$).
- **Prediction** `predictions/srs_E_at_P.py`. *Closed* — $E_{P} = \sqrt{k^{*}} = \sqrt{3}$.
- **Prediction** `predictions/g_girth.py`, `predictions/alpha_1.py`. *Closed* — $g = 10$, $\alpha_{1} = (2/3)^{8}$ (the latter via Lemma 1 of `docs/theorem_Feshbach_coupling_strength.md`; the Exponent-Principle-level identification of this with the Feshbach coupling remains adopted structure — §3 of that document).
- **Theorem A** `docs/theorem_uniform_Q_density.md` Part A. *Closed* — $\rho_{Q}$ uniform on the Ramanujan circle to $O(\sqrt{\log N/N})$.

Route B additionally requires:

- **P1** Ramanujan selection (`docs/W4_identification_catalog.md` §3). *Adopted structural postulate, Option-2.*
- **P2** $\sqrt{\textrm{multiplicity}}$ aggregation (catalog §3). *Adopted structural postulate, Option-2.*
- **P-mixing-from-Green's-function** (catalog §2B, three sub-items). *Adopted.*
- **P-dark-density** (catalog §2D / Claim B of `docs/theorem_Feshbach_coupling_strength.md`). *Partially closed* (Theorem A handles the shape; the Feshbach magnitude identification still rests on the Exponent Principle as adopted theorem, §3 of that document).
- **Route-B-specific identifications (B-Gap 1, 2, 3, 4)** from `predictions/V_us_derivation.md` §2.3. *BLOCKED* — no proof.

## Derivation

### Section 1. Upstream results

By (A1)–(A2), `k_star.py`, `d_spatial.py`: $k^{*} = 3$, $d = 3$, srs is the MDL-optimal 3-regular 3D crystal net (`predictions/g_girth_derivation.md` §2; Sunada 2012, *Notices AMS* **59**(2)). By `docs/theorem_walker_dynamics.md`, dynamics are non-backtracking walks with Hashimoto $B$ as the 1-step directed-edge transition operator. By `docs/theorem_BP_doubly_degenerate_h.md`, the Bloch scalar adjacency $A(P)$ has spectrum $\{\pm\sqrt{3}\}^{2}$, and $B(P)$'s 8-dimensional Ramanujan subspace carries eigenvalues $\{h, h^{*}, -h, -h^{*}\}$ with multiplicity 2 each. By `predictions/srs_E_at_P.py`, $E_{P} = \sqrt{k^{*}} = \sqrt{3}$. By `predictions/alpha_1.py` and Lemma 1 of `docs/theorem_Feshbach_coupling_strength.md`, the tree NB-walk survival at distance $g-2$ is $\alpha_{1} = (2/3)^{8}$.

### Section 2. Type-D class assignment

Per `docs/W4_identification_catalog.md` §2D, every PMNS/CKM mixing angle is written as a TBM baseline plus a dark-extraction correction. The TBM baseline is closed: it follows from `docs/theorem_BP_doubly_degenerate_h.md` (the $C_{3}$-protected degeneracy on the 8-dim Ramanujan subspace forces a specific first column / third column structure). The dark-extraction coefficient is class-dependent:

- **Class 3** — edge-local ($\text{Tr}(\sigma_{x}) = 0$ by character orthogonality, Serre 1977 §2.4 Theorem 3). Coefficient $c = 1$. Closed. Applies to $\theta_{13}$ (see `predictions/theta_13_PMNS_derivation.md`).
- **Class 2** — mass²-class (coefficient $\tan^{2}(\arg h) = 5/3$). Part A closed (uniform $\rho_{Q}$ is now Theorem A); part B (Feshbach magnitude) closed via the Exponent Principle (`docs/theorem_Feshbach_coupling_strength.md` §3) as an adopted structural theorem at tier P1/P2. Applies to $\theta_{23}$ (see `predictions/theta_23_PMNS_derivation.md`).
- **Class 1** — amplitude-class (coefficient $\sqrt{5}/4$). Same status as Class 2 (Theorem A + Exponent Principle cover the structural content; the overall class-coefficient computation still rests on the same density-plus-coupling construction). Applies to $V_{us}$ Route A.

**$\theta_{12}$ is *not* directly assigned to any of these three classes.** The 2–3 sector of TBM is exactly degenerate at the $P$-point (both bands have $|h|^{2} = k^{*} - 1$ with $\omega$ vs $\omega^{2}$ labels), which is what lets $\theta_{23}$'s Class-2 Feshbach splitting apply in first-order *degenerate* perturbation theory (Sakurai §5.2). The 1–2 sector of TBM is not degenerate in the same way: under the decomposition of `docs/theorem_BP_doubly_degenerate_h.md` Step 3, the 2–3 pair live in the $\omega \oplus \omega^{2}$ part of the Ramanujan subspace (same $|h|^{2}$) while the 1-generation state sits in the trivial-irrep part, which carries multiplicity 4 — the $\mu_{\text{triv}}$ of `predictions/Q_Koide_derivation.md` Step 4. Non-degenerate perturbation theory (Sakurai §5.1) gives a *quadratic* shift $\propto \alpha_{1}/(E_{2} - E_{1})$ at leading order, not a linear one, and the gap $E_{2} - E_{1}$ is not supplied by any upstream file.

The route actually implemented in `predictions/theta_12_PMNS.py` is therefore the **Pati–Salam SU(4) perpendicularity chain** of `predictions/V_us.py` (Route B). This chain gives $\theta_{12}$ only after $V_{us}$ is taken as input, and so $\theta_{12}$ inherits every open gap of `predictions/V_us_derivation.md`.

### Section 3. Route B — the chain in five explicit steps

1. **TBM solar baseline.**
   $$\theta_{\text{TBM}} \;=\; \arctan\!\left(\frac{1}{\sqrt{k^{*}-1}}\right) \;=\; \arctan\!\left(\frac{1}{\sqrt{2}}\right) \;\approx\; 35.2644°. \tag{1}$$
   This is the asserted *tribimaximal* solar angle: $\sin^{2}\theta_{\text{TBM}} = 1/k^{*} = 1/3$. Under the W4 catalog it rests on postulates P1, P2 (Type-A mass-amplitude mapping of Ramanujan multiplicities $(4,2,2)$) together with the Type-D sub-identification that places $\theta_{12}^{\text{TBM}}$ in the 1-2 sector specifically. This is **B-Gap 4** of `predictions/V_us_derivation.md` — an adopted postulate, admissible under Option-2 when flagged.

2. **Bare Cabibbo walk amplitude.** (Route-A numerator of $V_{us}$.)
   $$V_{us}^{\text{bare}} \;=\; \left(\frac{k^{*}-1}{k^{*}}\right)^{2+E_{P}} \;=\; (2/3)^{2+\sqrt{3}} \;\approx\; 0.22020. \tag{2}$$
   This is the NB walk amplitude at distance $L_{us} = 2 + E_{P}$. The Lemma 1 tree-survival statement $(k^{*}-1)/k^{*}$ per step is theorem-grade (`docs/theorem_Feshbach_coupling_strength.md`). The distance identification $L_{us} = 2 + E_{P}$, however, has **no upstream derivation** — this is **A-Gap 1** of `predictions/V_us_derivation.md` (a non-integer NB length is not a standard combinatorial quantity; the Exponent Principle gives integer exponents $g$, $g-1$, $g-2$ but not $2+\sqrt{3}$).

3. **Cabibbo angle.**
   $$\theta_{C} \;=\; \arcsin V_{us}^{\text{bare}} \;\approx\; 12.7208°. \tag{3}$$
   The choice $V_{us} = \sin\theta_{C}$ (as opposed to $V_{us} = \tan\theta_{C}$ or $\theta_{C}$ itself in radians) is the standard Cabibbo convention (PDG 2024 CKM review) and is CAS arithmetic given (2).

4. **Spherical Pythagorean identity (perpendicularity postulate).**
   $$\cos\theta_{12} \;=\; \frac{\cos\theta_{\text{TBM}}}{\cos\theta_{C}}. \tag{4}$$
   This is the step that `proofs/flavor/srs_theta12_perp.py` motivates by the Killing-form perpendicularity of the Cabibbo generator (in the SU(3) adjoint **8** of the Pati–Salam decomposition $\mathbf{15} = \mathbf{8}\oplus\mathbf{1}\oplus\mathbf{3}\oplus\bar{\mathbf{3}}$) and the TBM generator (in the leptoquark $\mathbf{3}+\bar{\mathbf{3}}$). That script verifies $B(T_{C}, T_{\text{TBM}}) = 0$ numerically, but — per **B-Gap 1** of `predictions/V_us_derivation.md` — the assignment of those specific Lie-algebra basis elements to "the Cabibbo rotation" and "the TBM rotation" is itself a smuggled identification. Furthermore — **B-Gap 2** — on a non-abelian group manifold (SU(4) has rank 3), Killing-form perpendicularity does *not* generally imply the scalar spherical Pythagorean identity (4): the BCH series has non-trivial higher-order terms unless $[T_{C}, T_{\text{TBM}}] = 0$, and the script itself records $\|[T_{C}, T_{\text{TBM}}]\| \neq 0$. The identity (4) is therefore asserted, not derived. Under the rigor bar, **this alone blocks theorem-grade closure.**

5. **Solar mixing angle.**
   $$\theta_{12} \;=\; \arccos\!\left(\frac{\cos\theta_{\text{TBM}}}{\cos\theta_{C}}\right). \tag{5}$$
   Substituting (1), (3) into (4) and taking arccos:
   $$\theta_{12} \;=\; \arccos\!\left(\frac{0.81650}{0.97546}\right) \;=\; \arccos(0.83704) \;\approx\; 33.171°. \tag{6}$$

### Section 4. Rigor verdict

Evaluating under `docs/parameter_linter.md` "Hard quality gate":

- **Step 1** (TBM baseline): admissible under Option-2 via postulates P1, P2 (catalog §3), provided they are flagged — as this file does.
- **Step 2** (bare Cabibbo): blocked by **A-Gap 1** (the distance $L_{us} = 2 + E_{P}$ has no upstream derivation; $2+\sqrt{3}$ is not an NB-walk length in the Ihara sense).
- **Step 3**: admissible CAS arithmetic.
- **Step 4**: blocked by **B-Gap 1** (Lie-algebra assignment of Cabibbo / TBM generators) and **B-Gap 2** (non-abelian spherical Pythagorean identity unproven, $[T_{C}, T_{\text{TBM}}] \neq 0$).
- **Step 5**: admissible CAS arithmetic given (4).

Any one of A-Gap 1, B-Gap 1, B-Gap 2 is individually sufficient to block the parameter.

**Classification:** $\theta_{12}$ is **BLOCKED — transitively on $V_{us}$.** When `predictions/V_us.py` closes (i.e. when either Route A's (A-Gap 1, A-Gap 2) or Route B's (B-Gap 1, 2, 3) are derived from MDL + toggle + upstream closed files), this file will upgrade automatically.

### Section 5. Why there is no $V_{us}$-free direct derivation

The neighbouring PMNS angles have $V_{us}$-independent routes:

- **$\theta_{23}$** — Class 2 (mass²-class, coefficient $5/3$). Works because the 2–3 sector of TBM is *exactly degenerate* at the $P$-point (both bands sit on the Ramanujan circle with equal $|h|^{2} = k^{*}-1$; their $C_{3}$ labels are $\omega$ vs $\omega^{2}$ — see `docs/theorem_BP_doubly_degenerate_h.md` Step 3). Degenerate perturbation theory (Sakurai §5.2) then gives a *linear* splitting $\propto \alpha_{1}$.
- **$\theta_{13}$** — Class 3 (edge-local). Works because $\theta_{13}^{\text{TBM}} = 0$ by the third-column structure of the TBM mixing matrix (Ramanujan degeneracy forces $U_{\text{TBM}}(e,3) = 0$), and the $C_{3}$ character orthogonality $\sum_{j}\chi(g^{j}) = 0$ kills the parity-odd channel ($\text{Tr}(\sigma_{x}) = 0$) at a $C_{3}$-symmetric vertex.

The 1–2 sector of TBM has *neither* of these features:

- It is not $C_{3}$-degenerate at $P$. The generation-1 state sits in the trivial-irrep part of the Ramanujan subspace ($\mu_{\text{triv}} = 4$; `predictions/Q_Koide_derivation.md` Step 4), the generation-2 state in the $\omega$-irrep part ($\mu_{\omega} = 2$). These are *different* spectral sub-blocks with *different* multiplicities, so the first-order splitting is governed by non-degenerate perturbation theory (Sakurai §5.1), which at leading order gives a *quadratic* shift proportional to $\alpha_{1} / (E_{2} - E_{1})$. The gap $E_{2} - E_{1}$ is not supplied by any upstream closed file.
- The TBM entry $U_{\text{TBM}}(e,1)$ is not zero, so the $\theta_{13}$-style "$\theta^{\text{TBM}} = 0 \Rightarrow$ everything is first-order small" simplification does not apply.

A direct $V_{us}$-free derivation of $\theta_{12}$ would therefore need, at minimum, (i) the $E_{2} - E_{1}$ spectral gap computed explicitly on srs at the relevant Brillouin-zone point (not the $P$-point, where the 1–2 sector sits in two different multiplicity blocks), and (ii) a Class-4 dark-correction calculation for non-degenerate first-order perturbation. Neither is written up. The only realised route is therefore Route B, with its $V_{us}$ dependency.

### Section 6. What Theorem A and the Exponent Principle do (and do not) change

Before the promotions in `docs/theorem_uniform_Q_density.md` and `docs/theorem_Feshbach_coupling_strength.md`:

- $\rho_{Q}$ uniform on the Ramanujan circle was an *adopted ansatz*, partially fit against Kesten–McKay (catalog §2D lines 100–108).
- $\alpha_{1} = (2/3)^{g-2}$ was an *asserted* Feshbach coupling magnitude.

After:

- $\rho_{Q}$ uniform is **Theorem A**, from Rissanen/MDL code-length + the Pinsker/$\chi^{2}$ expansion (Cover & Thomas 2006 Lemma 17.3.2), to $O(\sqrt{\log N / N})$ — far below any observable precision for $N \sim 10^{60}$.
- $\alpha_{1} = (2/3)^{g-2}$ follows by Lemma 1 (tree NB-walk survival is theorem-grade) plus the **Exponent Principle** applied to scattering with $n_{\text{fixed}} = 2$ external edges (→ internal length $g-2$). The Exponent Principle itself is at adopted-structural-theorem status, same tier as P1, P2.

Both upgrades affect the Class-1 ($V_{us}$ Feshbach, Route A) and Class-2 ($\theta_{23}$) dark-correction coefficients by *tightening* their upstream dependencies from "density ansatz + coupling claim" to "density theorem + Exponent Principle (adopted theorem)". Neither touches:

- the walk-length identification $L_{us} = 2 + E_{P}$ inside $V_{us}^{\text{bare}}$ (A-Gap 1 stands);
- the Lie-algebra generator identifications on SU(4)_PS (B-Gap 1 stands);
- the non-abelian spherical Pythagorean identity (B-Gap 2 stands).

Therefore the promotions do not upgrade $\theta_{12}$'s status: it remains transitively BLOCKED on $V_{us}$.

## Result

$$\boxed{\text{Under the framework rigor bar, } \theta_{12} \text{ is BLOCKED — transitively on } V_{us}.}$$

For downstream continuity, the Route-B numerical value is
$$\theta_{12}^{\text{Route B}} \;=\; \arccos\!\left(\frac{\cos(\arctan(1/\sqrt{2}))}{\cos(\arcsin((2/3)^{2+\sqrt{3}}))}\right) \;\approx\; 33.171°$$
(computed in `predictions/theta_12_PMNS.py`, which prints `OK: outputs agree`).

## Comparison with experiment

| Quantity | Predicted (Route B) | Observed (NuFIT 6.0, NO) | Deviation | Rigor status |
|----------|---------------------|--------------------------|-----------|---------------|
| $\theta_{12}$ | $33.17°$ | $33.68° \pm 0.73°$ | $-0.70\sigma$ | BLOCKED (transitively on $V_{us}$) |

The $0.70\sigma$ agreement is suggestive but is not, under an internal note, evidence of correctness: the old $V_{us} = (2/3)^{2+\sqrt{3}}$ matched observation and was numerological (see `predictions/V_us_derivation.md` §2.1). The same caveat applies here: matching observation is NOT evidence of correctness.

## Open questions (inherited from upstream)

1. **Close $V_{us}$ (Route A or Route B).** $\theta_{12}$ closes automatically when $V_{us}$ closes. The concrete gaps are the five listed in `predictions/V_us_derivation.md` §6.
2. **Derive a $V_{us}$-free direct route.** As §5 explains, this would require writing out the non-degenerate first-order dark correction for the 1–2 sector on srs at a Brillouin-zone point where the 1–2 spectral gap $E_{2}-E_{1}$ is accessible. No file in the codebase attempts this; it would be a multi-session research project comparable in depth to the $\theta_{23}$ Class-2 derivation.
3. **Reduction of P1, P2.** Both postulates underpin the TBM step (1). The catalog §5 outlines Reductions A (derive $\sqrt{\mu}$ aggregation from a "symmetric observer" principle) and B (derive Ramanujan selection from the Ihara-zeta–MDL bridge). Either would upgrade step 1 from "adopted" to "theorem-mod-upstream".
4. **Generation labelling.** Assigning specific $C_{3}$ indices $j \in \{0,1,2\}$ to the physical states $(\nu_{1}, \nu_{2}, \nu_{3})$ is a shared open question with every generation-structured prediction (`predictions/Q_Koide_derivation.md` Open Question 1).

None of these can be closed by the present file. They are flagged for upstream work.

## References

- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley-Interscience. Lemma 17.3.2.
- Esteban, I., Gonzalez-Garcia, M. C., Maltoni, M., Schwetz, T., Zhou, A. (2024). NuFIT 6.0 global fit of neutrino oscillation parameters.
- Grünwald, P. (2007). *The Minimum Description Length Principle.* MIT Press. §5.3, §7.1, §14.3.
- Koide, Y. (1983). A fermion–boson composite model of quarks and leptons. *Phys. Lett. B* **120**, 161–165.
- Particle Data Group (2024). *Review of Particle Physics*, Workman *et al.*, Prog. Theor. Exp. Phys. **2024**, 083C01. CKM / PMNS reviews.
- Rissanen, J. (1978, 1983). MDL foundational papers.
- Sakurai, J. J., Napolitano, J. (2011). *Modern Quantum Mechanics*, 2nd ed. §5.1 (non-degenerate perturbation theory), §5.2 (degenerate perturbation theory).
- Serre, J.-P. (1977). *Linear Representations of Finite Groups.* GTM 42, Springer. §2.4 Theorem 3.
- Sunada, T. (2012). Lecture on topological crystallography. *Notices AMS* **59**(2), 208–215.
- Terras, A. (2011). *Zeta Functions of Graphs: A Stroll through the Garden.* Cambridge University Press.
- Framework internal: `docs/W4_identification_catalog.md` (§2D Type D, §3 postulates, §5 reductions); `docs/theorem_walker_dynamics.md`; `docs/theorem_BP_doubly_degenerate_h.md`; `docs/theorem_uniform_Q_density.md` (Theorem A); `docs/theorem_Feshbach_coupling_strength.md` (Lemma 1 + Exponent Principle); `predictions/Q_Koide_derivation.md` (template + multiplicity structure); `predictions/V_us_derivation.md` (A-Gap 1, A-Gap 2, B-Gap 1, 2, 3, 4); `predictions/theta_23_PMNS_derivation.md` and `predictions/theta_13_PMNS_derivation.md` (Type-D sibling angles); `proofs/flavor/srs_theta12_perp.py` (perpendicularity script).
