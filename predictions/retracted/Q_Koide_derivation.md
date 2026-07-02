# Q_Koide — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. See predictions/Q_Koide_v2.py and predictions/Q_Koide_v2_derivation.md for the post-A3 Born-rule re-derivation under the three-axiom framework (A1+A2+A3). Canonical axiom statement: docs/framework_axioms.md.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation's Step 5 (P2 application) identifies the srs C_3 irrep index `j ∈ {0, 1, 2}` with a generation index running over `{e, μ, τ}` via the formula `√m_j = √μ_triv + √μ_ω ω^j + √μ_ω̄ ω^{-j}`. B6 (`docs/theorem_B6_bridge.md`) proves the srs body-diagonal C_3 is the color-Z_3 of SU(3)_c via Spin(6)≅SU(4)→PS embedding, not a generation label. Under B6, the three C_3 irreps label color components within ONE Pati-Salam family, not three generations of one species. The coherent sum over C_3 irreps therefore aggregates across colors, not across generations — and the resulting "Q" is a color-sector arithmetic identity, not the charged-lepton Koide ratio.

**Re-derivation target**: Sprint 11 workstream B7.3 (mass operator on C³_gen; see `docs/master_plan.md` §Sprint 11). The framework has a separate structural path to three generations via the observer's minimum viable Hilbert space dimension C³_gen (MDL + Gleason's theorem → d_obs = 3). Re-derivation of Q_Koide from a mass operator on C³_gen is scoped for Sprint 11 B7.4.

**What survives as math**: the multiplicity identity

$$Q_{\text{color-sector}} \;=\; \frac{\mu_{\text{triv}} + \mu_{\omega} + \mu_{\omega^{2}}}{k^{*}\cdot\mu_{\text{triv}}} \;=\; \frac{4 + 2 + 2}{3 \cdot 4} \;=\; \frac{8}{12} \;=\; \frac{2}{3}$$

is a **standalone mathematical lemma** about the srs Bloch Hashimoto operator's Ramanujan (4, 2, 2) color-isotypic structure at the P-point. This lemma is label-agnostic and follows from `docs/theorem_BP_doubly_degenerate_h.md` Step 3 + the Ihara–Bass identity + the B6 color reading of the (4, 2, 2) multiplicities. It remains valid under all current theorems; only the physical identification with the charged-lepton Koide ratio is retracted.

## Specific failing step

Step 5 of the Derivation invokes **P2** ("generation-j mass amplitude = √μ-weighted coherent sum over C_3 irreps") with the reading that the index `j ∈ {0, 1, 2}` labels the three charged-lepton generations {e, μ, τ}. Quoting Step 5:

> "**P2** — $\sqrt{m_j} = \sqrt{\mu_{\text{triv}}} + \sqrt{\mu_{\omega}}\,\omega^{j} + \sqrt{\mu_{\omega^{2}}}\,\omega^{-j}$."

Here `j` was taken as a generation index. Under B6, the index `j` running over C_3 irreps runs over color components of a single PS generation, not three generations. The P2 coherent sum therefore aggregates amplitudes across color within one family; it does not produce three distinct generational mass eigenvalues. The subsequent Koide-ratio evaluation (Steps 6–7) is arithmetically correct as a statement about color-sector spectral data, but its identification with the observed {m_e, m_μ, m_τ} triple fails.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (PDG 2024) | Status |
|---|---|---|---|
| Q_Koide | 2/3 = 0.666666… | 0.666661 ± 0.0000068 | not explanatory under current framework |

The numerical match between the color-sector multiplicity ratio 8/12 = 2/3 and the observed charged-lepton Koide ratio is an empirical coincidence under the retracted reading. Whether it is re-derivable under the C³_gen framework — specifically as `Q = Tr(M)/(Tr√M)²` on a mass operator M defined on C³_gen — is the Sprint 11 B7.4 open question. The framework does not currently claim the 2/3 match as explanatory.

## Preserved original derivation (for reference; superseded)

The original derivation is retained below, marked as superseded. It establishes the multiplicity identity `Q = (μ_triv + μ_ω + μ_ω²)/(k*·μ_triv)` as a mathematical property of the srs Ramanujan subspace, which survives B6 as a standalone lemma. It is only the final identification of the C_3 index `j` with a generation label that is retracted.

---

# Derivation of the charged-lepton Koide ratio Q (SUPERSEDED, retained for reference)

## Abstract

We derive

$$Q \;:=\; \frac{m_e + m_\mu + m_\tau}{\left(\sqrt{m_e} + \sqrt{m_\mu} + \sqrt{m_\tau}\right)^{2}} \;=\; \frac{2}{3}$$

as a consequence of the $C_3$ irreducible-representation multiplicity structure of the Ramanujan subspace of the srs Bloch non-backtracking walk operator $B(\mathbf{k})$ at the $P$-point of the Brillouin zone, together with two adopted structural postulates of the framework (P1 Ramanujan selection; P2 $\sqrt{\textrm{multiplicity}}$ aggregation, both from `docs/W4_identification_catalog.md` §3). The multiplicity structure itself is a theorem — it follows from `docs/theorem_BP_doubly_degenerate_h.md` Step 3 together with the Ihara–Bass Bloch factorisation — so every step in the chain is either an upstream closed result, a cited mathematical theorem, explicit algebra, or an adopted postulate with its catalog reference.

## Framework axioms invoked

Inherited from upstream predictions files and theorems; no new axioms introduced here.

- **(A1)** Binary self-inverse toggle (`predictions/p_toggle.py`).
- **(A2)** MDL compression (`predictions/d_spatial.py`, `predictions/k_star.py`).

The framework's additional structural content used in this derivation:

- **Theorem** `docs/theorem_walker_dynamics.md` — walker dynamics on srs are non-backtracking walks; $B$ is the Hashimoto matrix on directed edges; $L$-step transition amplitudes are matrix elements of $B^{L}$.
- **Theorem** `docs/theorem_BP_doubly_degenerate_h.md` — at the $P$-point the Bloch Hashimoto operator $B(P)$ has the eigenvalue $h = (\sqrt{3} + i\sqrt{5})/2$ with multiplicity exactly 2, $C_3$-protected; its $\pm\sqrt{3}$ A-eigenspaces decompose under $C_3$ as $(\text{trivial}\oplus\omega)$ and $(\text{trivial}\oplus\omega^{2})$.
- **Postulate P1** (`docs/W4_identification_catalog.md` §3) — Ramanujan selection: physical mass amplitudes live on the 8-dimensional $\{h, h^{*}, -h, -h^{*}\}$ Ramanujan subspace of $B(P)$, not on the trivial $\pm 1$ tree subspace.
- **Postulate P2** (same §3) — $\sqrt{\textrm{multiplicity}}$ coherent aggregation: the generation-$j$ mass amplitude is the $\sqrt{\textrm{multiplicity}}$-weighted coherent sum over $C_3$ irreducible representations,
$$\sqrt{m_j} \;=\; \sqrt{\mu_{\text{triv}}} \;+\; \sqrt{\mu_{\omega}}\,\omega^{j} \;+\; \sqrt{\mu_{\omega^{2}}}\,\omega^{-j}, \qquad \omega = e^{2\pi i / 3}.$$

Per the catalog (§4): P1 and P2 are adopted additional structure beyond the two foundational axioms. Treating them as adopted lets the present derivation clear the rigor bar under the framework's Option-2 stance; treating them as targets for further reduction is the subject of the catalog's §5 (Reductions A, B, C).

## Derivation

### Step 1. Upstream: $k^{*} = 3$, $d = 3$, srs = I4_132 + Wyckoff 8a

From `predictions/k_star.py`, `predictions/d_spatial.py`, and `predictions/g_girth_derivation.md` §2, the MDL-optimal observer operates on a 3-regular 3D crystal net, and that net is the srs lattice in the standard I4_132 realisation with Wyckoff position 8a, $x = 1/8$ (Sunada 2012, *Notices AMS* **59**(2), 208–215; catalogued in the RCSR database as entry `srs`).

### Step 2. Walker dynamics: $B$ is the Hashimoto operator

By `docs/theorem_walker_dynamics.md` (Steps 1–7 in that document), the observer's dynamics on srs reduce to non-backtracking walks, with the Hashimoto matrix $B$ as the 1-step transition operator on the 12-dimensional directed-edge state space of the primitive cell. The $L$-step transition amplitude from one directed edge to another is the corresponding matrix element of $B^{L}$.

### Step 3. P-point A-eigenspace decomposition under $C_3$

By `docs/theorem_BP_doubly_degenerate_h.md` Step 3, the scalar Bloch adjacency $A(P)$ at the $P$-point is a Hermitian $4\times 4$ matrix with characteristic polynomial $(\lambda^{2} - 3)^{2}$, hence eigenvalues $\pm\sqrt{3}$ each with multiplicity exactly 2. The $C_3$ permutation $P_\sigma$ (induced by the body-diagonal rotation that fixes $P$) commutes with $A(P)$, and the two $A(P)$-eigenspaces decompose under $C_3$ as

- $+\sqrt{3}$-eigenspace $=$ $\text{span}\{|\omega\rangle, |{+}\sqrt{3}_{\text{triv}}\rangle\}$ — one $\omega$-irrep, one trivial irrep.
- $-\sqrt{3}$-eigenspace $=$ $\text{span}\{|\omega^{2}\rangle, |{-}\sqrt{3}_{\text{triv}}\rangle\}$ — one $\omega^{2}$-irrep, one trivial irrep.

### Step 4. Ihara–Bass lifts $C_3$-content to $B(P)$; 8-dim Ramanujan subspace has multiplicity structure $(4, 2, 2)$

The Ihara–Bass identity (Ihara 1966; Bass 1992; Terras 2011 §2.2) for the srs primitive cell ($|V| = 4$, $|E| = 6$, $k = 3$) in its Bloch form is
$$\det(I - u\,B(P)) \;=\; (1 - u^{2})^{2}\,\det\!\left((1 + 2u^{2})\,I - u\,A(P)\right).$$

Substituting the characteristic factorisation $(\lambda^{2} - 3)^{2}$ of $A(P)$,
$$\det\!\left((1 + 2u^{2})\,I - u\,A(P)\right) \;=\; \left(4u^{4} + u^{2} + 1\right)^{2} \;=\; \bigl(2u^{2} - \sqrt{3}\,u + 1\bigr)^{2}\bigl(2u^{2} + \sqrt{3}\,u + 1\bigr)^{2}.$$

Each inner quadratic gives two roots $u = (\sqrt{3}\pm i\sqrt{5})/4$ resp. $(-\sqrt{3}\pm i\sqrt{5})/4$; the corresponding B-eigenvalues $\mu = 1/u$ are $\mu \in \{h, h^{*}\}$ from the first quadratic and $\mu \in \{-h, -h^{*}\}$ from the second, each with multiplicity **2** from the square in the factorisation.

Because $B$ commutes with $C_3$ (which commutes with $A(P)$ at the fixed point $P$), each $A(P)$-eigenspace's $C_3$-content is inherited by the corresponding $B$-eigenspace pair. The $+\sqrt{3}$ A-eigenspace (2-dimensional, content $\text{trivial}\oplus\omega$) lifts to the 4-dimensional $\{h,h^{*}\}$ B-eigenspace pair, each B-eigenvalue of mult 2 carrying $C_3$-content $\text{trivial}\oplus\omega$. Summing over $h$ and $h^{*}$ gives content $2\cdot\text{trivial}\oplus 2\cdot\omega$ on the 4-dim $\{h,h^{*}\}$ block. Similarly the $-\sqrt{3}$ A-eigenspace lifts to the 4-dim $\{-h,-h^{*}\}$ block with content $2\cdot\text{trivial}\oplus 2\cdot\omega^{2}$.

The 8-dim Ramanujan subspace of $B(P)$ is the direct sum of these two blocks, and therefore has $C_3$ multiplicity structure
$$\mu_{\text{triv}} \;=\; 4, \qquad \mu_{\omega} \;=\; 2, \qquad \mu_{\omega^{2}} \;=\; 2.$$

This is a derivation, not a postulate: it follows from theorem_BP Step 3 plus the Ihara–Bass identity (cited mathematical theorem). It is further verified numerically in `explorations/bp_h_eigenspace_c3.py`. **Under B6, these multiplicities are color-isotypic counts for one PS generation × 8 species/chiralities, not per-generation counts.**

### Step 5. Apply postulates P1 and P2 [FAILING STEP under B6]

Adopt (per `docs/W4_identification_catalog.md` §3):

- **P1** — physical mass amplitudes live on the 8-dim Ramanujan subspace.
- **P2** — $\sqrt{m_j} = \sqrt{\mu_{\text{triv}}} + \sqrt{\mu_{\omega}}\,\omega^{j} + \sqrt{\mu_{\omega^{2}}}\,\omega^{-j}$.

**This is the step that B6 retires**: the index `j` running over C_3 irreps was interpreted as a generation index, but under B6 (`docs/theorem_B6_bridge.md` Step 7) the C_3 irreps label color components of one PS family.

Substituting the multiplicities from Step 4:
$$\sqrt{m_j} \;=\; 2 \;+\; \sqrt{2}\,\omega^{j} \;+\; \sqrt{2}\,\omega^{-j} \;=\; 2 \;+\; 2\sqrt{2}\,\cos\!\left(\tfrac{2\pi j}{3}\right).$$

In the standard Koide parametrisation $\sqrt{m_j} = \sqrt{M}\,(1 + \varepsilon\cos(2\pi j/3 + \delta))$, this corresponds to $\sqrt{M} = 2$ (so $M = 4$), phase $\delta = 0$, and amplitude $\varepsilon = \sqrt{2}$. Equivalently $\varepsilon^{2} = 4\mu_{\omega}/\mu_{\text{triv}} = 4\cdot 2/4 = 2 = 2(k^{*}-2)$, the amplitude relation used in `predictions/epsilon_Koide_derivation.md`.

### Step 6. Closed-form evaluation of $\Sigma\sqrt{m}$ and $\Sigma m$

Using $\sum_{j=0}^{2}\cos(2\pi j/3) = 0$ and $\sum_{j=0}^{2}\cos^{2}(2\pi j/3) = 3/2$ (or equivalently the $C_3$ orthogonality identity $\sum_{j}\omega^{jn} = 3\,\delta_{n\equiv 0\pmod 3}$):

$$\sum_{j=0}^{2}\sqrt{m_j} \;=\; 3\cdot 2 \;=\; 6.$$

$$\sum_{j=0}^{2} m_j \;=\; \sum_j \left(2 + 2\sqrt{2}\cos\tfrac{2\pi j}{3}\right)^{2} \;=\; \sum_j \left(4 + 8\sqrt{2}\cos\tfrac{2\pi j}{3} + 8\cos^{2}\tfrac{2\pi j}{3}\right)$$
$$\phantom{\sum m_j} \;=\; 12 + 8\sqrt{2}\cdot 0 + 8\cdot \tfrac{3}{2} \;=\; 12 + 12 \;=\; 24.$$

In the generic form (valid for arbitrary real multiplicities with $\mu_{\omega} = \mu_{\omega^{2}}$):
$$\sum_{j}\sqrt{m_j} \;=\; k^{*}\sqrt{\mu_{\text{triv}}}, \qquad \sum_{j} m_j \;=\; k^{*}\left(\mu_{\text{triv}} + \mu_{\omega} + \mu_{\omega^{2}}\right).$$

### Step 7. The Koide ratio

$$Q \;=\; \frac{\sum_{j} m_j}{\left(\sum_{j}\sqrt{m_j}\right)^{2}} \;=\; \frac{\mu_{\text{triv}} + \mu_{\omega} + \mu_{\omega^{2}}}{k^{*}\,\mu_{\text{triv}}} \;=\; \frac{4 + 2 + 2}{3\cdot 4} \;=\; \frac{8}{12} \;=\; \frac{2}{3}.$$

Numerically $2/3 = 0.666\overline{6}$. **This arithmetic is a valid color-sector spectral identity under B6; it is the interpretation as the charged-lepton Koide ratio that is retracted.**

### Remark. Why the compact form $Q = (k^{*}-1)/k^{*}$

Substituting the srs multiplicities $(\mu_{\text{triv}}, \mu_{\omega}, \mu_{\omega^{2}}) = (4, 2, 2)$ into the generic closed form gives $Q = 8/12 = 2/3 = (k^{*}-1)/k^{*}$ at $k^{*} = 3$. This matches the NB walker's "active edge fraction" at a $k^{*}$-regular vertex: both evaluate to $2/3$. The equality is a numerical coincidence at $k^{*} = 3$ with the specific srs multiplicities, not an independent combinatorial identity. The rigorous chain is the one above; the "active fraction" phrasing used in the older version of this file is a mnemonic that happens to land on the same number at this one parameter value.

## Result (color-sector arithmetic lemma only; generation identification retracted under B6)

$$Q_{\text{color-sector lemma}} \;=\; \tfrac{2}{3} \;=\; 0.666\overline{6}.$$

## References

- Bass, H. (1992). The Ihara–Selberg zeta function of a tree lattice. *Int. J. Math.* **3**, 717–797.
- Ihara, Y. (1966). On discrete subgroups of the two-by-two projective linear group over $p$-adic fields. *J. Math. Soc. Japan* **18**, 219–235.
- Koide, Y. (1983). A fermion–boson composite model of quarks and leptons. *Phys. Lett. B* **120**, 161–165.
- O'Keeffe, M., Peskov, M.A., Ramsden, S.J. & Yaghi, O.M. (2008). The Reticular Chemistry Structure Resource (RCSR) database. *Accts. Chem. Res.* **41**, 1782–1789. [Entry `srs`.]
- Sunada, T. (2012). Lecture on topological crystallography. *Notices AMS* **59**(2), 208–215.
- Terras, A. (2011). *Zeta Functions of Graphs: A Stroll through the Garden.* Cambridge University Press. §2.2 (Hashimoto matrix), §2.3 (Ihara–Bass identity).
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
