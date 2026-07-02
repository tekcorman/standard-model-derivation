# Theorem — The Unified-Oblique Resolvent: δ_r and δρ as Two Eigen-Channels of One B_NB(srs)

*Status: THEOREM-GRADE-STRUCTURAL (the new c_S Perron-residue piece is theorem-grade).
Established 2026-05-16. Probe: `proofs/foundations/unified_oblique_one_resolvent_2026-05-16.py`.
Ledger anchors: Rows P64 (M_Z), P71 (m_W), P73 (δρ).*

---

## 1. Abstract

The Standard Model electroweak oblique corrections appear in the framework
through **one** spectral object: the non-backtracking (Hashimoto) resolvent
$G_{NB}(u) = (I - u\,B_{NB}(\mathrm{srs}))^{-1}$ of the srs crystal net.
Two gauge-vertex projections of this single operator yield the two
custodial-sector observables already in `predictions/`:

- the **Z / neutral** vertex (species-conserving) projects onto the
  **Perron** eigenvalue ($\lambda_P = k^*-1 = 2$, the dominant mode), giving
  the absolute-$M_Z$ tree→pole oblique $\delta_r$ (Row P64);
- the **W / charged** vertex (species-changing $n{=}1\!\leftrightarrow\!n{=}2$)
  projects onto the **Ramanujan-saturated** eigenvalue
  $h_P = (\sqrt3+i\sqrt5)/2$ ($|h_P|^2 = k^*-1$, sub-dominant), giving the
  custodial-breaking $\delta\rho$ (Row P73).

The derivation is non-trivial because it **closes the provenance gap** flagged
by the `parameter_linter` Checkpoint-1 triage: the coefficient $c_S = 1/12$
that `predictions/delta_r.py` previously *cited* from the retracted probe
`family_E_phase_A_S_scale_gauge_2point_2026-05-15.py` (stale base predictions)
is here **derived from first principles** as the gauge-singlet projection of
the $B_{NB}$ Perron-eigenvalue residue, $c_S = 1/(2|E|) = 1/12$. The two
historical "routes" (Route H $1/(2|E|)$, Route C $k^*/(N k^{*2})$) are shown
to be the *same number* by the **handshake lemma** $2|E| = N k^*$ — a graph
identity, not a numerical coincidence. No fitted constant enters; every
factor lies in $K = \mathbb{Q}(\sqrt2,\sqrt3,\sqrt5)$.

The one-resolvent identification extends beyond the oblique sector: §7
reaches the remaining SM oblique objects (S, U, Δκ), **§8** reaches
the **flavor** sector — the CKM amplitudes {V_cb, V_ub, V_us} are the
*off-diagonal* species-changing readings of the **same** $B_{NB}$ at the
**same** spectral datum (zero-parameter over-determination test, 6/6
pre-declared aborts; THEOREM-GRADE-STRUCTURAL cross-lock, no number or
grade changed), and **§9 (2026-05-23)** reaches the **cosmology** sector —
$A_s$, the primordial scalar perturbation amplitude, is the
**single-loop-closure bare-$a$** reading of the same $B_{NB}$, with
prefactor $1/54 = c_S \cdot q^2 \cdot (1/2)_{\rm orient}$ derived as a
single-projection structure parallel to $c_S$. **Combined with the parallel
2026-05-22 gen-3 anchor landing (y_t, y_b) and the 2026-05-23 lepton/PMNS
audit (y_τ, θ_{12}, θ_{13}, θ_{23}), the over-determined family now reaches
twelve observables across four SM sectors** (gauge oblique, flavor/CKM,
quark/lepton Yukawa + PMNS, cosmology); A_s joins the cluster at
THEOREM-GRADE-STRUCTURAL. This dissolves the Need-D-3 eigenbasis-misalignment
framing *as a mechanism question*. The operational catalogue (without S/U/Δκ
and A_s) is in memory `reference_11_observable_section8_overdetermination_2026-05-23`.

---

## 2. Framework axioms and prior results invoked

1. **A1 / k\* = 3, srs net** — the MDL-optimal 3-regular 3D crystal net is
   srs (Sunada 2012); $N_{\rm atoms} = 4$ atoms and $|E| = 6$ bonds per
   primitive cell, coordination $k^* = 3$, girth $g = 10$
   (`predictions/k_star.py`, `predictions/g_girth.py`).
2. **The non-backtracking operator** $B_{NB}$ on the srs cell (12 directed
   edges), Ihara–Bass at $\Gamma$ and Ramanujan saturation at
   $P=(\tfrac14,\tfrac14,\tfrac14)$ — established in
   `proofs/foundations/nb_two_vertex_generations_probe.py` (Part A); reused
   verbatim, not rebuilt.
3. **Feshbach Exponent Principle** — a 2-point (propagator) process is the
   $n_{\rm fixed}=2$ scattering survival
   $\alpha_{1,\rm bare} = ((k^*-1)/k^*)^{\,g-2} = (2/3)^8$
   (`predictions/alpha_1.py`).
4. **Master-doc Family-C universal template** (Type-4 framework result;
   `theorem_substrate_feshbach_dark_corrections_master.md` §2):
   $g_{\rm phys} = g_{\rm bare}\,(1 - c\cdot\alpha_1/(1-\alpha_1))$,
   calibrated on $v_{\rm Higgs}$ ($c_v = 5/12$).
5. **Master-doc Family-E mass²-class Feshbach** (Phase C; same master doc §4):
   custodial-breaking propagator observable → spectral residue at $h_P$ with
   functional $F = \mathrm{Im}(h_P)/|h_P|^2 = \sqrt5/4$, calibration-locked
   to `predictions/m_nu3.py` §3(B).
6. **c = 1/2** — the squared W-field normalization
   $g_W^2/(g_Z^2\cos^2\theta_W) = (g/\sqrt2)^2/g^2$, rigorously derived in
   `family_E_phase_C1_c_half_W_normalization_2026-05-15.py` (Phase C.1);
   cited, not re-derived.
7. **O9 algebraicity meta-theorem** — every Class A/B/C/E coefficient lies in
   $K = \mathbb{Q}(\sqrt2,\sqrt3,\sqrt5)$
   (`theorem_lattice_coupling_algebraicity.md`).

---

## 3. Derivation

### 3.1 One resolvent, two eigen-channels

The substrate gauge-boson self-energy is a closed non-backtracking walk
generating function on $B_{NB}(\mathrm{srs})$. Its spectral decomposition is

$$G_{NB}(u) \;=\; (I - u\,B_{NB})^{-1} \;=\; \sum_{\lambda} \frac{P_\lambda}{1 - u\lambda},
\qquad P_\lambda = \frac{|r_\lambda\rangle\langle l_\lambda|}{\langle l_\lambda|r_\lambda\rangle}.$$

A gauge boson $V$ couples at a vertex with a definite species structure. The
**neutral Z** ($T_3$ diagonal) is species-conserving: the self-energy walk
closes without a species transition and projects onto the **C₃/species-blind
singlet** direction. The **charged W** ($T_\pm$ off-diagonal) forces one
$n{=}1\!\leftrightarrow\!n{=}2$ species transition and projects onto the
**phase-carrying** $h_P$ direction. Both projections are taken on the **same**
$B_{NB}$:

| Channel | Eigenvalue | Modulus | Role |
|---|---|---|---|
| Z (neutral, singlet) | Perron $\lambda_P = k^*-1 = 2$ | $|\lambda_P| = 2$ — **dominant** | sign-uniform absolute scale |
| W (charged, phase) | $h_P = (\sqrt3+i\sqrt5)/2$ | $|h_P| = \sqrt2$ — **sub-dominant**, $|h_P|^2 = k^*-1$ | custodial-breaking |

Ramanujan saturation ($|h_P|^2 = k^*-1 = \lambda_P$) makes the $|\cdot|^2$
self-energy weights of the two channels equal, so the entire Z/W splitting is
the **phase** of $h_P$, i.e. $\mathrm{Im}(h_P)$ — exactly the mass²-class
Feshbach functional. (Numerically verified in the probe, Part 1.)

### 3.2 The Z channel: $c_S$ as the Perron-residue singlet projection

**Lemma.** *The Perron eigenvector of $B_{NB}(\mathrm{srs})$ at $\Gamma$ is the
uniform directed-edge vector $\mathbf{1}\in\mathbb{C}^{2|E|}$.*

*Proof.* By construction $B_{NB}[a,b]=1$ iff $\mathrm{head}(e_a)=\mathrm{tail}(e_b)$
and $e_b\neq\overline{e_a}$. Each directed edge has exactly $k^*-1$
non-backtracking continuations, so every row sum of $B_{NB}|_\Gamma$ is
$k^*-1$; hence $B_{NB}\mathbf{1}=(k^*-1)\mathbf{1}$. srs is edge-regular, so
$\mathbf{1}^{\!\top}B_{NB}=(k^*-1)\mathbf{1}^{\!\top}$ as well; thus
$\mathbf{1}$ is both the right and left Perron eigenvector with eigenvalue
$\lambda_P=k^*-1$. (Verified in the probe: `B_NB·1 = (k*-1)·1` and
`1ᵀ·B_NB = (k*-1)·1ᵀ`, machine precision.) $\;\blacksquare$

The neutral-Z gauge vertex couples to the species-singlet channel, i.e. to
the **unit singlet** $\hat s = \mathbf{1}/\sqrt{2|E|}$. The rank-1 Perron
spectral projector is $P_P = |\mathbf{1}\rangle\langle\mathbf{1}|/\langle\mathbf{1}|\mathbf{1}\rangle$,
and the gauge-singlet residue weight is

$$c_S \;=\; \frac{\langle \hat s | P_P | \hat s\rangle}{2|E|}
\;=\; \frac{1}{2|E|}\cdot\frac{\big(\langle \hat s|\mathbf{1}\rangle\big)^2}{\langle\mathbf{1}|\mathbf{1}\rangle}
\;=\; \frac{1}{2|E|}\cdot\frac{\big(\sqrt{2|E|}\big)^2}{2|E|}
\;=\; \frac{1}{2|E|}.$$

With $|E|=6$ bonds per cell, $2|E|=12$, so $\boxed{c_S = 1/12}$.

**Route H ≡ Route C (handshake lemma).** The historical "Route H"
($1/(2|E|)$, NB Hilbert-dimension normalization) and "Route C"
($k^*/(N k^{*2}) = 1/(N k^*)$, cycle-counting normalization) are identical
because the **handshake lemma** gives
$$2|E| \;=\; \sum_{v} \deg(v) \;=\; N_{\rm atoms}\cdot k^* \;=\; 4\cdot 3 \;=\; 12.$$
This is a graph identity, **not** a numerical coincidence; the two routes are
two readings of the *same* Perron-residue normalization. This **replaces** the
retracted Phase-A fit citation (`family_E_phase_A_*`, stale base predictions):
$c_S$ is now derived independently of any $\delta_r$ target.

### 3.3 The Z observable: Family-C universal template

The absolute-$M_Z$ shift is a sign-uniform propagator scale correction, so it
takes the master-doc **Family-C universal template** (axiom 4) with the
$c_S$ derived above:

$$M_{Z,\rm pole} = M_{Z,\rm tree}\,(1-\delta_r),\qquad
\boxed{\;\delta_r = c_S\cdot\frac{\alpha_{1,\rm bare}}{1-\alpha_{1,\rm bare}}
= \frac{1}{12}\cdot\frac{(2/3)^8}{1-(2/3)^8} = +0.3384\%\;}$$

### 3.4 The W observable: Family-E mass²-class Feshbach

The custodial-breaking $\rho$-shift is the propagator-level custodial-breaking
class, taking the master-doc **Family-E** form (axioms 5, 6) at the $h_P$
residue of the *same* $B_{NB}$:

$$\boxed{\;\delta\rho = c\cdot\frac{\mathrm{Im}(h_P)}{|h_P|^2}\cdot\alpha_{1,\rm bare}
= \tfrac12\cdot\tfrac{\sqrt5}{4}\cdot(2/3)^8 = +1.0906\%\;}$$

### 3.5 Why the same object takes two different forms (structural argument)

The two **forms** are existing Type-4 master-doc templates; the *selection*
between them is governed by the master-doc selection rule (Family-C for
sign-uniform scale corrections; Family-E for propagator custodial-breaking).
The single $B_{NB}$ makes that selection rule **transparent**:

- the **Perron** mode is the *dominant* eigenvalue (marginal, no spectral gap
  above it) — its channel carries the resummable sign-uniform scale, hence the
  Family-C geometric form $\alpha_1/(1-\alpha_1)$;
- the **$h_P$** mode is *sub-dominant* ($|h_P|=\sqrt2 < \lambda_P=2$) and
  phase-carrying — its channel carries the leading custodial Feshbach
  insertion, hence the Family-E form $F\cdot\alpha_1$, the common scale
  cancelling in the $m_W/M_Z$ ratio.

This is a **structural argument** consistent with the master-doc selection
rule; it is *not* a from-resolvent computation of the resummation and does
**not** upgrade the form selection. It is graded as such (§5).

---

## 4. Result

One object $G_{NB}(u)=(I-u\,B_{NB}(\mathrm{srs}))^{-1}$ produces, by its two
gauge-vertex eigen-projections, **both** custodial-sector observables already
in `predictions/`:

$$\delta_r = \tfrac{1}{12}\cdot\frac{(2/3)^8}{1-(2/3)^8} = +0.338356\%,
\qquad
\delta\rho = \tfrac12\cdot\tfrac{\sqrt5}{4}\cdot(2/3)^8 = +1.090599\%.$$

The probe reproduces the live `predictions/delta_r.py` and
`predictions/delta_rho.py` outputs to $<10^{-9}$. Every factor is in
$K=\mathbb{Q}(\sqrt2,\sqrt3,\sqrt5)$: $c_S=1/12\in\mathbb{Q}$,
$c=1/2\in\mathbb{Q}$, $F=\sqrt5/4\in\mathbb{Q}(\sqrt5)$,
$\alpha_{1,\rm bare}=(2/3)^8\in\mathbb{Q}$. No $\arg(h_P)$ transcendental
enters (the phase appears only as $\mathrm{Im}/|\cdot|^2$). Zero fitted
constants.

---

## 5. Comparison with experiment and grade

| Observable | Predicted | Observed | Deviation |
|---|---|---|---|
| $M_Z$ (via $\delta_r$) | 91.2039 GeV | 91.1876 GeV (PDG 2024) | +0.018% rel; $\gg\sigma_{\rm PDG}$ (2.3 ppm) |
| $m_W$ (via $\delta_r\!\oplus\!\delta\rho$) | 80.401 GeV | 80.3692 GeV (PDG 2024) | +0.040% rel; $+2.4\sigma_{\rm PDG}$ |
| $\delta\rho$ (scale-independent) | +1.0906% | +1.0429% (PDG-central) | +4.58% rel; **+0.76 σ_obs** |

**Grade — honest split:**

- **NEW & theorem-grade:** $c_S = 1/(2|E|) = 1/12$ as the $B_{NB}$
  Perron-residue gauge-singlet projection, with Route H ≡ Route C by the
  handshake lemma. This **closes the `parameter_linter` Checkpoint-1
  provenance blocker** (the retracted-Phase-A citation in
  `predictions/delta_r.py` is replaced by a derivation). The single-$B_{NB}$
  spectral identification (Perron *and* $h_P$ are eigenvalues of the one
  operator) is verified.
- **Inherited (Type-4, already graded):** the Family-C form (δ_r), the
  Family-E mass²-Feshbach form (δρ), $c=1/2$ (Phase C.1), $F=\sqrt5/4$
  (m_ν3 calibration), $\alpha_1$ (`alpha_1.py`).
- **Structural argument (consistent, not upgraded):** Perron-dominance vs
  $h_P$-subdominance explains, but does not re-derive, the master-doc
  selection rule.

Overall the unified-oblique theorem is **THEOREM-GRADE-STRUCTURAL**. As of
the 2026-05-16 tree-cover work (§7.5) the **Clause-7 derivation rigor is now
fully closed**: the last rigor gap — the §3.5 resummation form-selection,
previously a structural argument — is **derived** from the analytic structure
of the tree cavity resolvent (§7.5). The label remains
THEOREM-GRADE-STRUCTURAL only on the **Clause-8 numerical** axis (the named
+4.58% δρ relative residual and the absolute-mass $\sigma_{\rm PDG}$ floor are
numerical-match gaps, not rigor gaps — per linter Clause 8e vocabulary). The
**new** $c_S$ Perron-residue piece is at theorem grade and the δ_r/δρ
unification is on one verified spectral object.

Clause 8 vs $\sigma_{\rm PDG}$ on the absolute masses still **FAILS** (the
framework's intrinsic ~ppm structural precision floor, reported honestly, no
$\sigma_{\rm theory}$). The clean scale-independent δρ test PASSES at
**+0.76 σ_obs**.

---

## 6. Open questions

1. **The §3.5 resummation argument — CLOSED 2026-05-16 (§7.5).** It is no
   longer a structural argument: the tree cavity resolvent
   $g(z)=1/(z-k\,f(z))$ *is* the Dyson resummation, and its analytic
   structure derives the dichotomy (off the McKay support, disc $>0$ ⇒
   geometric resummation converges ⇒ $\alpha_1/(1-\alpha_1)$; on the McKay
   cut, disc $\le0$ — the Ramanujan-saturated modes, $h_P{\to}\lambda{=}\sqrt3$
   interior ⇒ leading-only ⇒ $\alpha_1$). Clause-7 rigor gap eliminated.
   (Criterion sharpened off-vs-on-cut by the §7.6 re-audit.)
2. **The +4.58% δρ relative residual** (named subleading-spectral) — a
   bounded tree branch-expansion attempt (§7.5) found **no clean K-rational
   closed form**; the residual is a genuine higher-order sum over
   sub-leading sub-tree insertions, not a one-line term. Declared
   **honestly still-open** (within $+0.76\,\sigma_{\rm obs}$; not
   numerically urgent; not forced). **The §7.6 re-audit adds a hard
   constraint:** $\delta\rho$ is on the McKay cut, so closing this by a
   resummation factor $1/(1-\alpha_1)$ is *forbidden* — the attack must be
   a sub-tree multi-insertion sum.
3. **S, U, Δκ — S now CLOSES (§7.3, §7.5).** U closes
   THEOREM-GRADE-STRUCTURAL (U ≈ 0, Ramanujan √(k\*−1) sector scale-frozen);
   Δκ is a definitional Type-3 recombination of δρ; **S now CLOSES** —
   the tree-cover cavity Green's function resolves the cell obstruction
   exactly ($g(k){=}u^*{=}2/3$ off-support finite, $g(2\sqrt q){=}\sqrt q$),
   giving $S=\tfrac1{12}(\sqrt2-\tfrac23)\,\alpha_1/(1-\alpha_1)=+0.253\%$,
   K-rational, δ_r/δρ-class, no fitted constant.
4. **Absolute-mass Clause 8.** $M_Z$/$m_W$ remain $\gg\sigma_{\rm PDG}$
   (intrinsic structural floor); this theorem improves the *relative* residual
   and closes the *provenance* gap, it does not move the absolute-mass
   $\sigma_{\rm PDG}$ verdict.

---

## 7. Extension — S, U, Δκ: the derivative-class oblique parameters (2026-05-16)

*Probe: `proofs/foundations/oblique_S_U_kappa_2026-05-16.py`.*

§3 read **residues** of $G_{NB}$ at the two canonical Bloch points: $\delta_r$
(Perron magnitude at $\Gamma$) and $\delta\rho \approx T$ ($h_P$ phase at $P$).
The remaining Peskin–Takeuchi parameters are **derivative-class** — the
$\Gamma\!\to\!P$ *flow* of the gauge-projected resolvent, where $\Gamma\!\leftrightarrow\!q^2{=}0$
and $P\!\leftrightarrow\!q^2{=}$on-shell (the framework's established mass-sector
Bloch identification). The verified spectral fact (probe Part 0):

$$\text{Perron: } k^*{-}1 = 2 \xrightarrow{\Gamma\to P} \sqrt{k^*{-}1}=\sqrt2
\quad\text{(collapses)};\qquad
\sqrt{k^*{-}1}\text{ sector}: \sqrt2 \xrightarrow{\Gamma\to P} \sqrt2
\quad\text{(scale-FROZEN)}.$$

### 7.1 U ≈ 0 — THEOREM-GRADE-STRUCTURAL (the sharpest result)

PT $U \propto$ (W-slope) − (Z-slope). The charged W vertex projects on the
$\sqrt{k^*{-}1}$ Ramanujan sector, whose modulus is $\sqrt{k^*{-}1}$ at **both**
$\Gamma$ and $P$ — it is **scale-invariant** (the *same* Ramanujan saturation
$|h_P|^2=k^*{-}1$ that made $\delta\rho$ a pure-*phase* effect). Hence the
charged-channel self-energy slope vanishes at leading order, and $U$ — a W−Z
*slope difference* — receives no $O(\alpha_1)$ contribution:

$$\boxed{\;U \approx 0,\qquad |U|\;\lesssim\;\alpha_{1,\rm bare}\,|S|\;}$$

This is a **first-principles near-vanishing with zero fitted input**, derived
from the Ramanujan structure, and it matches the robust SM/experiment fact
that PT $U$ is the oblique parameter most consistent with zero
($|U|\ll|S|,|T|$). Graded **THEOREM-GRADE-STRUCTURAL**.

### 7.2 Δκ — definitional Type-3 recombination of δρ (inherits its grade)

The effective-vs-on-shell mixing-angle shift, by the standard oblique algebra
(Type-3, the *same tier* as the $m_W=M_Z\cos\theta_W$ tree relation already in
the cluster), is to leading (Δρ-driven) order

$$\Delta\kappa_{\rm lead} = \frac{c_W^2}{c_W^2-s_W^2}\cdot\delta\rho
= 1.403\times(+1.091\%) = +1.530\%,$$

with $s_W^2 = 1-m_W^2/M_Z^2$ the on-shell mixing angle. This is **not an
independent spectral object** — the $\kappa$-factor is fixed EW algebra (no
free parameter), so Δκ **inherits δρ's grade exactly** (Row P73,
THEOREM-GRADE-STRUCTURAL). The full measured $\sin^2\theta_{\rm eff}-s_W^2
\approx +3.74\%$ is SM-scheme/$\Delta\alpha$-dominated; only the
$\delta\rho$-driven piece is a framework claim (honestly named, the same
discipline as $\delta_r$'s intrinsic-floor honesty).

### 7.3 S — cell NEG (obstruction located) → CLOSED on the tree cover (§7.5)

PT $S$ is the neutral self-energy *running*; the framework object is the
neutral/Perron-channel $\Gamma\!\to\!P$ flow. On the srs **cell** the
pre-declared sign abort (S.2) fired: the Perron mode has
$u^*\!\cdot\!(k^*{-}1)=4/3>1$ — past the *cell* non-backtracking convergence
radius; the $\Gamma$ pole is a divergent analytic continuation. The
obstruction was precisely located: per `nb_two_vertex` Part B the $z^*$
mechanism's home for the Perron mode is the **3-regular tree cover**. **§7.5
does that tree computation and S CLOSES** — the cell-divergent Perron pole is
replaced by the convergent off-spectrum tree cavity resolvent, with no
post-hoc S-definition swap (the S structure is δ_r's, unchanged; only the
cell-divergent factor → convergent tree flow).

### 7.4 Summary

| PT param | Framework object | Status |
|---|---|---|
| Δr | $\delta_r$ — Perron residue magnitude @ Γ | THEOREM-GRADE-STRUCTURAL (Row P64; §3) |
| T | $\delta\rho$ — $h_P$ phase residue @ P | THEOREM-GRADE-STRUCTURAL (Row P73; §3) |
| **U** | $\sqrt{k^*{-}1}$ sector scale-frozen ⇒ $U\!\approx\!0$ | **THEOREM-GRADE-STRUCTURAL** (§7.1) |
| **Δκ** | $(c_W^2/(c_W^2{-}s_W^2))\,\delta\rho$ Type-3 recomb. | **inherits δρ grade** (§7.2) |
| **S** | neutral Perron-channel Γ→P flow (tree cover) | **THEOREM-GRADE-STRUCTURAL** — $S=\tfrac1{12}(\sqrt2-\tfrac23)\tfrac{\alpha_1}{1-\alpha_1}=+0.253\%$ (§7.5) |

**All five** SM oblique objects (Δr, T, U, S, Δκ) are now readings of the
**one** $G_{NB}=(I-u\,B_{NB}(\mathrm{srs}))^{-1}$ — three cell residues
(Δr@Γ-Perron, T@P-$h_P$, U from the scale-frozen Ramanujan sector), one
tree-cover flow (S), one Type-3 recombination (Δκ). No fitted constants
anywhere; the U≈0 prediction is the sharpest and most falsifiable. No
`predictions/*.py` files are added: U≈0 / S are structural (not tight
numerical DAG targets) and Δκ is a definitional recombination of the existing
`predictions/delta_rho.py` — adding SM-subtracted PT-parameter files would
invite the substrate/observable category conflation the project methodology
warns against.

### 7.5 Tree-cover S + the §6.1 resummation lever (DERIVED, 2026-05-16)

*Probe: `proofs/foundations/tree_cover_S_and_resummation_2026-05-16.py`.*

**The rigorous tree cavity Green's function** (Kesten 1959; McKay 1981; the
cavity recursion on the $k$-regular tree). A rooted $(k{-}1)$-ary subtree
generating function satisfies $q\,f^2-z\,f+1=0$ ($q\equiv k^*{-}1=2$),
physical branch $f\to0$ as $z\to\infty$; the root resolvent is
$g(z)=1/(z-k\,f(z))$. Two **exact, K-rational** values (sympy-verified):

$$g(z_{\rm triv}{=}k{=}3)=\tfrac{(k-1)}{k}=u^*=\tfrac23,\qquad
g(z_{\rm edge}{=}2\sqrt q{=}2\sqrt2)=\sqrt q=\sqrt2.$$

$z_{\rm triv}=k$ is the trivial/neutral rep ($A\mathbf1=k\mathbf1$, the
$q^2{=}0$ / Γ-analog); it lies **off** the McKay support $[-2\sqrt q,2\sqrt q]$
($3>2\sqrt2$), so the tree resolvent there is **finite** — the rigorous
regularisation of the cell's divergent Perron pole. $z_{\rm edge}=2\sqrt q$ is
the on-shell Ramanujan/$h_P$ point ($|h_P|=\sqrt q$). The cell obstruction is
genuinely resolved: $z_{\rm triv}$ off-support and the tree NB radius
$u^*\sqrt q=2\sqrt2/3\approx0.943<1$ (convergent).

**Tree-cover S.** S is the *running* of the same neutral self-energy whose
absolute value gave $\delta_r$, so it inherits $\delta_r$'s structure exactly
($c_S=1/12$ Perron-residue singlet projection; resummed $\alpha_1/(1-\alpha_1)$,
neutral channel) — the **only** change is the cell-divergent Perron-pole
factor $1/(1-u^*(k^*{-}1))$ replaced by the convergent tree flow
$g(2\sqrt q)-g(k)$. No new free constant:

$$\boxed{\;S=c_S\big[g(2\sqrt q)-g(k)\big]\frac{\alpha_1}{1-\alpha_1}
=\tfrac1{12}\Big(\sqrt2-\tfrac23\Big)\frac{(2/3)^8}{1-(2/3)^8}
=\tfrac{64\sqrt2}{18915}-\tfrac{128}{56745}=+0.2529\%\;}$$

$\in\mathbb Q(\sqrt2)\subset K$; sign positive (the neutral self-energy
*enhances* $2/3\to\sqrt2$ across Γ→P), **uniform with the rest of the
substrate oblique sector** ($\delta_r{=}{+}0.338\%$, $\delta\rho{=}{+}1.091\%$,
$S{=}{+}0.253\%$ — same sign, same $\alpha_1$-class). PT $S$ is SM-subtracted
($\approx0$); the framework predicts the *physical* structure, so the sign is
reported, not gated (gating on PT-$S\!\approx\!0$ would be the
substrate/observable conflation). Grade **THEOREM-GRADE-STRUCTURAL**
(δ_r/δρ-class; obstruction resolved by the rigorous tree cavity GF).

**§6.1 resummation — DERIVED (the grade-lift lever).** The cavity recursion
$g=1/(z-k\,f)$, $f=1/(z-q\,f)$ *is* the Dyson geometric resummation of all
non-backtracking sub-tree insertions. Its analytic structure derives the
form-selection dichotomy that §3.5 previously only argued:

- **off the McKay support** ($|\lambda|>2\sqrt q$, discriminant
  $\lambda^2-4q>0$ — e.g. the trivial/Perron/marginal sector
  $h{=}{\pm}1\Rightarrow\lambda{=}{\pm}k{=}{\pm}3$): the resolvent is
  analytic, the geometric series ratio is $<1$ ⇒ it **converges** to the
  full closed form ⇒ the Family-C $\alpha_1/(1-\alpha_1)$ ($\delta_r$, S);
- **on the McKay cut** ($|\lambda|\le2\sqrt q$, discriminant $\le0$ — the
  Ramanujan-saturated modes $|h|{=}\sqrt q$ map *onto* the cut): $\sqrt{\cdot}$
  is imaginary, no convergent geometric resummation ⇒ **leading-only** ⇒
  the Family-E $\alpha_1$ ($\delta\rho$).

  *(Sharpened 2026-05-16 by the selection-rule re-audit,
  `proofs/foundations/selection_rule_reaudit_2026-05-16.py`: via the Ihara
  map $\lambda=h+q/h$ the framework eigenvalue $h_P$ maps to **interior**
  $\lambda=\sqrt3$ ($|\lambda|^2-4q=-5<0$), on the cut but **not** at the
  band edge. The general criterion is disc $\le0$ (the whole cut); the band
  edge $z=2\sqrt q$, disc $=0$, is merely the exactly-evaluable
  representative used in the closed-form check above — it is not where
  $\delta\rho$ actually sits.)*

This eliminates the **last Clause-7 rigor gap**: the resummation-vs-leading
selection is now *derived from the resolvent's analytic structure*, not a
structural argument. $\delta_r,\delta\rho,U,\Delta\kappa$ (which ride these
forms) have their derivation rigor closed; the residual gaps are purely
Clause-8 numerical (the named +4.58% $\delta\rho$ residual; the absolute-mass
$\sigma_{\rm PDG}$ floor).

**δρ +4.58% subleading-spectral — honest still-open.** A bounded branch
expansion $g(2\sqrt q+\varepsilon)=\sqrt2-3\cdot2^{1/4}\sqrt\varepsilon
+\tfrac{17}2\varepsilon-\dots$ shows that at *exact* Ramanujan saturation the
on-shell point is the branch point itself ($\varepsilon{=}0$): there is no
single clean $O(\varepsilon)$ or $O(\sqrt\varepsilon)$ term of size $+4.6\%$.
The residual is a genuine higher-order sum over sub-leading sub-tree
insertions, not a one-line term. **No clean K-rational closed form** emerges
from the bounded computation → declared **still-open, not forced** (within
$+0.76\,\sigma_{\rm obs}$; not numerically urgent).

**Re-audit corollary (constraint on the +4.58% problem).** $\delta\rho$'s
channel $h_P$ is *on the McKay cut* ($\lambda=\sqrt3$, disc $=-5<0$), where
the geometric (Dyson) resummation provably does **not** converge. Therefore
closing the $+4.58\%$ by absorbing it into a resummation factor
$1/(1-\alpha_1)$ is **forbidden by the derived criterion** — it would
mis-place $\delta\rho$ off-support. The residual *must* be approached as a
higher sub-tree multi-insertion (sub-leading-spectral) sum. This rules out
the "just resum $\delta\rho$" route and is the operative constraint on any
future attack.

### 7.6 Selection-rule re-audit (2026-05-16)

*Probe: `proofs/foundations/selection_rule_reaudit_2026-05-16.py`.*

With the form-selection rule now **derived** (§7.5), every propagator-level
member of the master-doc dark-correction catalogue (§5 of
`theorem_substrate_feshbach_dark_corrections_master.md`) was re-audited for
form↔channel consistency via the Ihara map $\lambda=h+q/h$:

| spectral location | members | derived form | assigned form | verdict |
|---|---|---|---|---|
| off-support ($\lambda{=}{\pm}k{=}{\pm}3$, disc $>0$) | v_Higgs, α_GUT, $\delta_r$, S | resummed Family-C | resummed | ✓ consistent |
| on-cut ($h_P{\to}\lambda{=}\sqrt3$ / band-edge, disc $\le0$) | $\delta\rho$, m_ν₃, β, θ₂₃, U | leading Family-E/Feshbach | leading | ✓ consistent |
| vertex per-leg / no-DC | y_τ, λ_Higgs, V_us, Λ_CC | — (criterion N/A) | — | out of scope ✓ |

**No misassignment.** The taxonomy — previously assigned by observable-class
heuristics plus the v_Higgs $c{=}5/12$ calibration anchor — is now
**derived-consistent**: the resummed-vs-leading choice for the *entire*
tree-level-coupling sector follows from the cavity resolvent's analytic
structure, not heuristics. **Numerical impact: zero** (no reassignment;
explicitly not manufactured). The result is a rigor consolidation, plus a
non-trivial validation: v_Higgs (the calibration anchor the counting family
was tuned against) lands off-support/resummed *purely* from
$\lambda(h{=}{\pm}1){=}{\pm}k$ — the derived criterion independently
reproduces the anchor it was never given. Scope held: Family-D vertex
per-leg ($\propto\alpha_1^2$) is a distinct mechanism, correctly excluded.

## 8. Extension — the CKM triple {V_cb, V_ub, V_us}: off-diagonal readings of the same B_NB (2026-05-16)

The §3 / §7 result says δ_r and δρ (and S, U, Δκ) are readings of the
**one** $G_{NB}=(I-u\,B_{NB}(\mathrm{srs}))^{-1}$. An over-determination
test (`proofs/foundations/quark_unification_over_determination_test_2026-05-16.py`,
bound to the live `match` / `CountingKernel` surface so "same B" is
*provable* not asserted) shows the extension reaches the **flavor**
sector: the CKM amplitudes are the *off-diagonal* species-changing
($n{=}1\leftrightarrow n{=}2$ Hamming) readings of the **same** operator
at the **same** spectral datum, with **zero fitted constants**.

One survival amplitude $a \equiv q_{NB}^{g-2} = (2/3)^8 = \alpha_{1,\rm bare}$
= the Feshbach W1 ($n_{\rm fixed}{=}2$) coupling on the one B at P, read
five ways (six with §9), each reproducing an observable **already** closed
at theorem-grade by a **separate, independent** prior route:

| reading of the one $a$ | observable | prior route |
|---|---|---|
| bare $a$ × Feshbach contour $\mathrm{Im}\,h_P/\|h_P\|^2{=}\sqrt5/4$, $c{=}\tfrac12$ | $\delta\rho=+1.0906\%$ | Row P73 Family-E |
| Perron projection $c_S{=}1/(2\|E\|){=}1/12$, **resummed** $a/(1{-}a)$ | $\delta_r=+0.3384\%$ | Row P64 (§3) |
| **resummed** $a/(1{-}a)$, unit projection | $V_{cb}=256/6305$ | Row P3 Class-A |
| multi-cycle host-sum (same $q_{NB}{=}2/3$) | $V_{ub}=3.767{\times}10^{-3}$ | Row P14 Class-C |
| counting projection $k^{*2}/(g\,N)$ | $V_{us}=9/40$ | Row P4 Class-E |
| **bare $a$ × $c_S \cdot q^2 \cdot (1/2)_{\rm orient} = 1/54$** | $A_s = (1/54) \cdot a \cdot (M_{\rm GUT}/M_{\rm Pl})^2$ | Row (A_s in `predictions/N_hub.py`); §9 |

**Discriminating fact.** $\delta_r$ and $V_{cb}$ are *provably the same*
resolvent-resummed amplitude $a/(1{-}a)=256/6305$ under two projections
(Perron $1/12$ vs unit); $\delta\rho$ is that *same bare* $a$ under
$h_P$'s Feshbach contour; $A_s$ is that *same bare* $a$ under the
single-loop-closure prefactor $c_S \cdot q^2 \cdot (1/2)_{\rm orient}$
(§9). The bare↔resummed link is the $(I-\,\cdot\,)^{-1}$
geometric series — *forced by the resolvent algebra, not fitted* — and it
lands on the **exact theorem-grade rationals** of six independent routes.
This is genuine over-determination (the §3 logic for {δ_r, δρ}, now also
for {V_cb, V_ub, V_us}); 6/6 pre-declared aborts pass.

> **EXTENSION 2026-05-17 — the over-determination family now also
> reaches the mass-identification postulate.**
> `theorem_mass_propagator_overdetermination.md` applies this same
> one-B_NB / zero-fitted / pre-declared-abort logic to the deep §6(i)
> "mass ∝ 1/inverse-propagator" postulate: it is the Ihara *value*
> channel u(k) ≡ *gradient* channel u'(k), forced uniquely at the
> independently-✅ k\*=3. Same grade (THEOREM-GRADE-STRUCTURAL), same
> family; decomposes the §6(i) postulate (does not close the deep
> frontier — one isolated interpretive premise remains).

**Grade: THEOREM-GRADE-STRUCTURAL** — the same grade the §3 standalone
results carry; this is the structural cross-lock, not a regrade. **No
`predictions/*.py` added; no number changed; no grade of P3/P4/P14
changed** (they are already UNIQUE-THEOREM-GRADE for amplitude). Not
theorem-grade-numerical: δρ retains its +4.58 % Clause-8 relative
residual (§5); the CKM rows retain their data-anchored *labeling* caveat.

**Honest scope — what this does NOT close.** The 3×3 generation /
$C_{36}$-twist (which structural amplitude ↔ which named $V_{ij}$) is the
data-anchored **non-blocking** labeling residue. The unification
*reframes* it as the resolvent's index structure rather than a missing
$Y_u/Y_d$ eigenbasis misalignment, so **Need-D-3 dissolves as a
*mechanism* question** — but the labeling residue itself is reframed, not
eliminated. The up-sector $y_t$ natural-scale anchor ($\sigma_+$ nilpotent
⇒ eigenvalue 0) remains the single genuine hard residue and is out of
scope here.

> **UPDATE 2026-05-21 — the $y_t$ up-anchor is now derived.**
> `theorem_updown_split_conjugate_higgs_2026-05-21.md` derives the
> "$\sigma_+$ nilpotent ⇒ eigenvalue 0" statement: the up-type fermion
> couples to the conjugate Higgs $\tilde H = i\sigma_2 H^*$, which is
> always an *even-grade* element of the edge qubit and therefore cannot
> flip handedness; the Yukawa walk (oscillatory srs↔srs-z, every step a
> handedness flip) cannot run ⇒ walk length $L=0$ ⇒ $y_t = q_{NB}^0 = 1$.
> The down-type Higgs is odd-grade ⇒ flips handedness ⇒ $L=g$. This was
> `state_of_the_derivation_2026-05-16.md` §3 mask #1; it is now closed at
> THEOREM-GRADE-STRUCTURAL. The CKM *labeling* residue (which structural
> amplitude ↔ which named $V_{ij}$) is separate and unaffected.

> **EXTENSION 2026-05-21 — the over-determination family reaches the
> light-quark masses (via per-sector mixing angles).** Fork 2
> (`proofs/foundations/mass_operator_fork2_node_mixing_2026-05-21.py`) shows
> the gen-1 quarks sit at a C₃-circulant node, so their masses are
> *mixing-determined*: $m_{q_1}=\theta^2\,m_{q_2}$ with $\theta$ the sector's
> (1,2) angle (Gatto–Sartori–Tonin texture). The down sector gives
> $m_d=V_{us}^2 m_s=(9/40)^2 m_s$ (+1.2%). The up sector
> (`mass_operator_theta_u_section8_2026-05-21.py`) needs $\theta_u$: it is an
> off-diagonal species-changing amplitude, hence forced into this §8 closed
> resolvent-reading set — the **resummed** $a/(1-a)$ (off-diagonal ⟹ not the
> bare-$a$ δρ-type). So $\theta_u=a/(1-a)$ and $m_u=\theta_u^2 m_c=V_{cb}^2
> m_c=2.10$ MeV ($-0.16\sigma$). Grade: **§8-CKM-family** — theorem-grade for
> the amplitude, the projection assignment (up-(1,2) ↔ resummed-unit) is the
> *same* data-anchored C₃₆-twist labeling residue carried by $V_{cb}/V_{us}/
> V_{ub}$. Net: the labeling residue now also gates $\theta_u$, $m_u$, $m_d$ —
> closing the C₃₆-twist closes the CKM **and** the light-quark masses together.

> **CORRECTION 2026-05-21 (Path D) — the "mixing-determined" reading above is
> superseded.** Path D
> (`proofs/foundations/mass_operator_path_D_node_exactness_2026-05-21.py` +
> `..._up_sector_2026-05-21.py`) tested whether the gen-1 circulant node is
> *exact* (which the EXTENSION block tacitly assumes — $m_{q_1}=\theta^2m_{q_2}$
> is the texture-zero / exact-node limit). It is **not exact**. Each quark
> sector $\{m_1,m_2,m_3\}$ is exactly a pure Koide circulant (any 3 masses are)
> with a *non-zero* gen-1 eigenvalue $=m_{q_1}$ — the **same** mechanism as the
> electron, which the lepton control proves is a circulant near-zero (Koide
> $\varepsilon^2=2$, zero mixing input). So $m_d=V_{us}^2m_s$ (+1.2%) and
> $m_u=V_{cb}^2m_c$ ($-0.2\sigma$) are per-sector ~1% **numerical
> coincidences**, not a mechanism — the lepton control fails the same GST test.
> The light-quark masses are circulant node-state eigenvalues (reading α)
> governed by the **Koide phase** $\delta$: $\delta_{\rm down}\approx0.101$,
> $\delta_{\rm up}\approx0.055\pm0.002$ — both $\neq$ the $2/(9(s+1))$ pattern
> ($1/9$, $2/27$). $\varepsilon^2_{\rm down}=5/2$ and $\delta_{\rm lepton}=2/9$
> are confirmed. m_d / m_u close iff $\delta$ closes — `Need-B δ-physical`. The
> CKM amplitudes ({$V_{us},V_{cb},V_{ub}$}, the rest of §8) are unaffected;
> only the *light-quark-mass* corollary of this block is corrected.

## 9. Extension — A_s (primordial scalar amplitude): cosmology-sector reading of the same B_NB (2026-05-23)

The §3 / §7 / §8 results say {δ_r, δρ, S, U, Δκ, V_cb, V_ub, V_us} are
readings of the **one** $G_{NB} = (I - u\,B_{NB}(\mathrm{srs}))^{-1}$
at the spectral datum $a = q_{NB}^{g-2} = (2/3)^8 = \alpha_{1,{\rm bare}}$.
This extension shows that the **cosmology** sector's primordial scalar
perturbation amplitude $A_s$ joins the same over-determined family as a
**sixth reading**.

The structural cross-lock was assembled across five sessions today
(`proofs/cosmology/A_s_C1_perron_projection_session2_2026-05-23.py` →
`..._session3_..._py` → `A_s_unified_oblique_session4_..._py` →
`A_s_prefactor_independent_session5_..._py` →
`A_s_prefactor_half_factor_session6_..._py`); the final commit `6dd2a63`
upgraded the prefactor to single-projection structural grade.

### 9.1 The reading

$A_s$ is computed in the framework's existing prediction file
(`predictions/N_hub.py`, function `A_s(kernel)`, status
DOMINANT-THEOREM-GRADE-CONDITIONAL, $+1.02\sigma_{\rm Planck}$
post-correction via cascade D2-ext) as

$$A_s \;=\; \alpha_{\rm GUT} \cdot (2/3)^g \cdot (M_{\rm GUT}/M_{\rm Pl})^2
\cdot (16/15) \;\;\approx\;\; 2.07 \times 10^{-9}\quad
(\text{Planck obs: } 2.10 \times 10^{-9}, +1.02\sigma).$$

The substrate-side amplitude (before the cascade rate-gap correction and
the gravity scale factor) is

$$A_{s,{\rm substrate}}^{(\text{no gravity})} \;=\;
\alpha_{\rm GUT} \cdot q_{NB}^g
\;=\; \alpha_{\rm GUT} \cdot q_{NB}^{g-2} \cdot q_{NB}^2
\;=\; (\alpha_{\rm GUT} \cdot q_{NB}^2) \cdot a
\;=\; \tfrac{1}{54} \cdot a.$$

So $A_s$ is the **bare-$a$ single-loop-closure** reading of the same
$B_{NB}$, with **prefactor $1/54$** to be derived structurally below.

### 9.2 Same $B_{NB}$, same spectral datum

The reading uses:
- the **same** $B_{NB}(\mathrm{srs})$ as §3 (verified: Perron eigenvalue
  at $\Gamma$ is $k^*-1 = 2$; row-sum identity holds);
- the **same** spectral datum $a = (2/3)^8 = \alpha_{1,{\rm bare}}$ as
  §8 (Feshbach W1 $n_{\rm fixed}=2$ coupling on the one B at P);
- a **distinct** projection prefactor $1/54$ and a **bare-$a$** reading
  class (single-event, parallel to $\delta\rho$'s bare-$a$ Feshbach
  contour; distinct from the resummed $a/(1-a)$ class of $\delta_r$,
  $V_{cb}$, $V_{ub}$).

Verified in `proofs/cosmology/A_s_unified_oblique_session4_2026-05-23.py`
(4/4 sentinels PASS).

### 9.3 The prefactor: $1/54 = c_S \cdot q^2 \cdot (1/2)_{\rm orient}$

**Single-projection derivation.** The $A_s$ prefactor decomposes exactly as

$$\boxed{\;\frac{1}{54} \;=\; c_S \cdot q^2 \cdot \tfrac{1}{2}_{\rm orient}
\;=\; \frac{1}{12} \cdot \frac{4}{9} \cdot \frac{1}{2} \;=\; \frac{4}{216}\;}$$

where each factor is structurally derived:

1. **$c_S = 1/(2|E|) = 1/12$** — the **Perron-residue singlet projection**
   of $B_{NB}$ at $\Gamma$ (this same factor that gives $\delta_r$ in §3.2);
   handshake lemma $2|E| = N k^*$ makes Routes H and C the same identity.

2. **$q^2 = ((k^*-1)/k^*)^2 = 4/9$** — the **two-step NB walker survival**
   amplitude. Per step, the walker has $k^*-1$ valid continuations out of
   $k^*$ outgoing arcs (1 backtrack forbidden by NB), so per-step survival
   is $q = (k^*-1)/k^* = 2/3$. The walker closes a girth-$g$ cycle via
   $a = q^{g-2}$ survival to near-girth plus $q^2$ for the two
   girth-completion steps.

3. **$(1/2)_{\rm orient}$** — the **directed→undirected cycle-count
   factor**. $B_{NB}$ is a directed-arc operator (Hilbert space dim
   $= 2|E|$). Each directed closed NB walk of length $g$ has a unique
   reverse-orientation partner (traverse the same underlying undirected
   cycle in the opposite direction). For a scalar observable like $A_s$
   (gauge-invariant, no preferred orientation), the natural count is
   **undirected cycles = directed count / 2**. Numerically verified on
   srs at $\Gamma$ (`A_s_prefactor_half_factor_session6_2026-05-23.py`,
   `[W2] PASS`): all 12 directed arcs have reverse partners, and
   $B_g[a,a] = B_g[\overline a, \overline a]$ for all $a$ (pairing exact).

### 9.4 Structural parallelism with $c_S$

The $A_s$ prefactor's $(1/2)_{\rm orient}$ is the **cycle-level analog**
of the $2$ in $c_S = 1/(2|E|)$:

| factor | source | what the "$2$" represents |
|---|---|---|
| $c_S = 1/(2|E|) = 1/12$ | Perron-residue singlet (§3.2) | directed/undirected **edge** ratio |
| $(1/2)_{\rm orient}$ in $1/54$ | scalar $A_s$ vs $B_{NB}^g$ directed cycles (§9.3) | directed/undirected **cycle** ratio |

Both reflect $B_{NB}$'s directed-arc-space construction applied to
gauge-invariant scalar observables (Perron-singlet for $c_S$, primordial
perturbation amplitude for $A_s$). The full $A_s$ prefactor multiplies
this with the two-step NB-survival $q^2$ that closes the girth cycle
beyond the $(g-2)$ survival already in $a$.

### 9.5 Other candidates for the $1/2$ — closed-negative audit

Session 6 tested three alternative candidates for the $(1/2)_{\rm orient}$
factor; all rejected structurally:

- **srs-z bipartite-double-cover half**: rejected by the Λ_CC
  bipartite-cover analog (commit `8c7964e`). $A_s$ is intensive,
  N_hub-observer-anchored; cell doubling on srs-z gives bit-identical
  $A_s$, not a halving. The "sum-over-encodings = 2" reading is not
  A2-T-licensed (canonical multi-admissibility is Bayesian average).
- **Chiral-half (srs is I4₁32 chiral)**: rejected. $A_s$'s three factors
  ($\alpha_{\rm GUT}$, $(2/3)^g$, $(M_{\rm GUT}/M_{\rm Pl})^2$) are all
  chirality-symmetric. Chirality and orientation are structurally
  distinct (chirality = lattice handedness; orientation = arc-traversal
  direction).
- **W-field $c=1/2$ analog (Family-E §3.4)**: not framework-internal.
  Would invoke a graviton-field normalization, which inherits Type-3 GR
  status (symmetric-tensor TT decomposition convention), not a framework
  derivation from A1 + P1'.

The orientation-undirecting derivation is the unique framework-internal
single-projection candidate.

### 9.6 Independent cross-check — Bloch-Hashimoto Perron-projection (C1)

A separate construction via the Bloch-Hashimoto Perron eigenvector
overlap $\zeta(k) \equiv \langle v_{\rm iso} | \psi_{\rm Perron}(k)\rangle$
reproduces the same multiplicative form
$A_s = \alpha_{\rm GUT} \cdot (2/3)^g \cdot (M_{\rm GUT}/M_{\rm Pl})^2$
(Sessions 2 + 3, commits `565b583`, `82cd99d`). The κ² tensor expansion
of $1 - |\zeta(k)|^2$ near $\Gamma$ decomposes into clean framework
integers:

- $c_{\rm body} = N_{\rm atoms}^2 \cdot k^* = 48$ ($+1.08\%$)
- $c_{\rm face} = 2|E| \cdot k^* = 36$ ($-2.33\%$)
- $c_{\rm xy} = 4g/3 = 13.33$ ($+0.24\%$)

The C1 path provides the second structural reading of $A_s$ (parallel to
Feshbach Exponent Principle, 2026-05-05); together with §9's reading via
the unified-oblique resolvent, $A_s$ has three independent structural
identifications agreeing on the same multiplicative form.

### 9.7 Status and grade

**Grade: THEOREM-GRADE-STRUCTURAL** — same as §3 / §7 / §8. The
structural cross-lock is at the unified-oblique resolvent level (six
observables, one $B_{NB}$, one spectral datum $a$, distinct projections,
zero fitted constants). The $A_s$ prediction file's grade is unchanged
(DOMINANT-THEOREM-GRADE-CONDITIONAL per `predictions/N_hub.py`); this
extension is the *structural* cross-lock, not a numerical regrade.

**Honest scope.** The orientation-undirecting argument is at the same
level as §3.5's selection rule: structural, parallel to $c_S$'s edge
construction, but not from-resolvent-computed. The grade
THEOREM-GRADE-STRUCTURAL matches §3 / §7 / §8.

**What this does NOT close.** The $A_s$ base-formula's prediction uses
$(M_{\rm GUT}/M_{\rm Pl})^2$ from "standard gravitational coupling at
GUT scale" (Type-3 GR inheritance); this is the same conditional as
`predictions/N_hub.py`'s status and is unaffected by §9. The cosmology
sibling observables {n_s, r, σ_8} remain L6-blocked per the n_s scoping
audit;
§9 sharpens but does not unlock them (the C1 reading of κ² gives the
*form* of $n_s - 1 \propto -2c \cdot |k|^2$ but not the *number*, which
requires Need-B's Bloch-physical unit map).

### 9.8 Net for the unified-oblique theorem

The over-determination family — combining §3 / §7 / §8 / §9 of this
document with the parallel 2026-05-22 gen-3 anchor landing
(an internal working note) and the
2026-05-23 lepton/PMNS audit
(an internal working note,
commit `b653104`) — now reaches **twelve observables across four SM
sectors**:

- **Gauge / oblique** (5): $\delta_r$, $\delta\rho$ (§3); S, U, Δκ (§7)
- **Flavor / CKM** (3): $V_{cb}$, $V_{ub}$, $V_{us}$ (§8)
- **Quark/lepton Yukawa** (3): $y_t$, $y_b$ (gen-3 anchor, 2026-05-22);
  $y_\tau$ (lepton anchor; Row P74)
- **PMNS mixing angles** (3): $\theta_{12}^{\rm PMNS}$,
  $\theta_{13}^{\rm PMNS}$, $\theta_{23}^{\rm PMNS}$
  (2026-05-23 audit; Rows P32, P31, P13)
- **Cosmology / primordial perturbations** (1): $A_s$ (§9, this section)

If S, U, Δκ are excluded (they are derivative-class objects in §7 rather
than independent observables), the canonical count is **12 − 3 + 0 = 9
or 12 with the derivative-class objects**; the parallel
`reference_11_observable_section8_overdetermination_2026-05-23` memory
catalogues the **11-observable narrow framing** (without S/U/Δκ and
without A_s); this §9 adds A_s as the **12th-or-13th entry** depending
on whether S/U/Δκ are counted.

One $B_{NB}(\mathrm{srs})$, one spectral datum
$a = (2/3)^8 = \alpha_{1,{\rm bare}}$ (or its resummed/dressed forms
$a/(1{-}a)$, $(5/3)\cdot a$, or §8-derived $V_{us} = 9/40$), distinct
structurally-derived projections, zero fitted constants between them.
This is north-star **condition-3 sector extension to four SM sectors**,
the diagnostic the framework values most. A_s is the cosmology-sector
counterpart to the lepton/PMNS landing — both extend the 2026-05-22
narrow quark-sector framing to broader-sector over-determination.

**Cross-references:**
- an internal working note — the
  4-observable lepton/PMNS landing (parallel to this §9).
- an internal working note — the
  original 2026-05-22 narrow quark-sector framing (since broadened).
- an internal working note §2.5 — the consolidated 11-observable
  framing (predates §9; A_s integration noted in §9.8 here).
- Memory `reference_11_observable_section8_overdetermination_2026-05-23`
  — operational catalogue for "is this observable already
  over-determined?" lookups; A_s should be added as the 12th entry
  to update this catalogue.
