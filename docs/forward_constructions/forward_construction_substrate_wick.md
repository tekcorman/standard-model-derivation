# Substrate Wick contractions theorem — F3

**Date:** 2026-04-26 (PM, follow-on to F1 propagator closure).
**Status:** Theorem-grade closure of F3 from an internal note. Second concrete deliverable in the φ(x) cascade.
**Source:** `forward_construction_field_operator_phi_x.md` §7 next-step #4 (Wick-contraction structure on substrate).
**Predecessors:**
- `forward_construction_substrate_propagator.md` (F1, gives $G_F^{\text{sub}}$ used here).
- `../theorems/theorem_car_local_jordan_wigner.md` (CAR on substrate).
- `forward_construction_substrate_thermal_apparatus.md` §3 (vacuum).
- `forward_construction_field_operator_phi_x.md` §2.4 (B+C synthesis).

---

## Question

Does Wick's theorem — the central computational identity of perturbative QFT — hold for substrate fermion fields? Specifically: do n-point functions $\langle 0_F | T(\psi(g_1, t_1) \cdots \psi^\dagger(g_{2n}, t_{2n})) | 0_F \rangle$ decompose as the signed sum over pair-contractions, with each contraction equal to the substrate Feynman propagator $G_F^{\text{sub}}$?

If yes: substrate's perturbation theory inherits the standard Feynman-diagram apparatus directly. F4 (LSZ), F5 (S-matrix), F6 (Feynman rules) become concrete computational follow-ups rather than research-level open problems.

---

## Result (preview)

**Theorem (substrate Wick).** For substrate fermion fields $\psi(g_i, t_i), \psi^\dagger(g_j, t_j)$ in the free theory (Hamiltonian $H = D_{\text{sub}}$), and the fermionic vacuum $|0_F\rangle$:

$$\langle 0_F | T\big(\psi(g_1, t_1) \cdots \psi(g_n, t_n) \psi^\dagger(g_{n+1}, t_{n+1}) \cdots \psi^\dagger(g_{2n}, t_{2n})\big) | 0_F \rangle = \sum_{\sigma \in P_n} \mathrm{sgn}(\sigma) \prod_{k=1}^n G_F^{\text{sub}}(g_k, g_{n+\sigma(k)};\, t_k - t_{n+\sigma(k)})$$

where the sum runs over all pairings $\sigma$ of the $n$ $\psi$-fields with the $n$ $\psi^\dagger$-fields, and $\mathrm{sgn}(\sigma)$ is the Fermi sign of the permutation needed to bring contracted pairs adjacent.

**Why it works.** The standard Wick proof requires three ingredients, all theorem-grade on substrate:

| Ingredient | Substrate provenance |
|---|---|
| CAR algebra $\{\psi(g), \psi^\dagger(g')\} = \delta_{g, g'}$ | `../theorems/theorem_car_local_jordan_wigner.md` (JW + A4 + A1) |
| Free Hamiltonian bilinear in $c, c^\dagger$ | `forward_construction_substrate_propagator.md` §1.4 ($H = \sum_{\alpha, k} \varepsilon_\alpha(k) c^\dagger c$) |
| Fermionic vacuum with $c |0_F\rangle = 0$ | `forward_construction_substrate_propagator.md` §1.5 (Dirac sea) |

All three transpose directly from continuum-QFT to substrate. The standard Wick proof (Peskin-Schroeder §4.3, Weinberg §6.1) carries through verbatim.

---

## 1. Setup

### 1.1 Inputs

From `forward_construction_substrate_propagator.md` §1:

- **Fermion field at substrate vertex $g = (R, r)$:** $\psi(g) = (1/\sqrt V) \sum_{\alpha, k} u_\alpha(k, r) e^{ik \cdot R} c_{\alpha, k}$, with Bloch-mode CAR $\{c_{\alpha, k}, c_{\beta, k'}^\dagger\} = \delta_{\alpha\beta} \delta_{kk'}$, all other anticommutators zero.
- **Heisenberg evolution:** $c_{\alpha, k}(t) = e^{-i\varepsilon_\alpha(k) t} c_{\alpha, k}$, with $\varepsilon_\alpha(k)$ the eigenvalues of $D(k)$ (chirally symmetric; 16 positive, 16 negative).
- **Vacuum $|0_F\rangle$:** Dirac-sea state. $c_{\alpha, k} |0_F\rangle = 0$ for positive-$\varepsilon$ modes; $d_{\alpha, k} |0_F\rangle = 0$ for negative-$\varepsilon$ modes (where $d = c^\dagger$ for negative-$\varepsilon$).
- **Substrate Feynman propagator:** $G_F^{\text{sub}}(g, g'; \tau) = \langle 0_F | T(\psi(g, \tau) \psi^\dagger(g', 0)) | 0_F \rangle$, in closed form per F1 Theorem 3.2.

### 1.2 Time-ordering convention

For fermion fields:

$$T\big(A_1(t_1) A_2(t_2)\big) = \begin{cases} A_1(t_1) A_2(t_2) & t_1 > t_2 \\ -A_2(t_2) A_1(t_1) & t_2 > t_1\end{cases}$$

For products of $n$ fermion fields, the time-ordering produces an overall sign equal to the parity of the permutation that brings them in time-decreasing order. Equivalently:

$$T(A_1 \cdots A_n)(t_1, \ldots, t_n) = \mathrm{sgn}(\pi) A_{\pi(1)}(t_{\pi(1)}) \cdots A_{\pi(n)}(t_{\pi(n)})$$

where $\pi$ is the permutation with $t_{\pi(1)} > t_{\pi(2)} > \cdots > t_{\pi(n)}$.

### 1.3 Normal ordering

Define normal ordering $:\!\cdots\!:$ relative to the Dirac-sea vacuum $|0_F\rangle$: in any monomial of $c, c^\dagger$ (Bloch-mode CAR), move all annihilators (those that annihilate $|0_F\rangle$) to the right of all creators. Each anti-commutation past a non-zero anti-commutator picks up the standard fermion sign. Under this convention:

$$\langle 0_F | :\!A(c, c^\dagger)\!: | 0_F \rangle = 0$$

for any non-trivial normal-ordered monomial.

---

## 2. Wick contraction lemma

**Definition 2.1 (Wick contraction).** For two fermion-field operators $A$ and $B$ (each a $\psi$ or a $\psi^\dagger$ at some vertex and time), the *contraction* is:

$$\overbracket{A B} := T(A B) - :\!A B\!: \;= \langle 0_F | T(A B) | 0_F \rangle.$$

The first equality is a definition; the second equality follows because $T(AB) - :\!AB\!:$ is, by direct computation, a c-number (numerical scalar) — and any c-number equals its vacuum expectation value.

**Lemma 2.2 (contractions equal Feynman propagator).** The non-zero contractions among substrate fermion fields are:

$$\overbracket{\psi(g, t)\, \psi^\dagger(g', t')} = G_F^{\text{sub}}(g, g'; t - t')$$

$$\overbracket{\psi^\dagger(g, t)\, \psi(g', t')} = -G_F^{\text{sub}}(g', g; t' - t)$$

All other contractions ($\overbracket{\psi \psi}$, $\overbracket{\psi^\dagger \psi^\dagger}$) vanish.

*Proof.* By Definition 2.1, the contraction equals the vacuum expectation value of the time-ordered pair, which is the substrate Feynman propagator (F1 Theorem 3.2). The vanishing of $\overbracket{\psi\psi}$ and $\overbracket{\psi^\dagger \psi^\dagger}$ follows from CAR ($\{\psi(g), \psi(g')\} = 0$, similarly for daggers) plus the fermionic vacuum structure. The sign in $\overbracket{\psi^\dagger \psi}$ comes from the fermion time-ordering. $\square$

---

## 3. Main theorem: substrate Wick

**Theorem 3.1 (substrate Wick theorem).** For $n$ fermion fields $A_1, \ldots, A_n$ (each a $\psi$ or $\psi^\dagger$), with the constraint that $A_i$'s and $\psi$/$\psi^\dagger$ counts agree (i.e., charge conservation):

$$T(A_1 A_2 \cdots A_n) = :\!A_1 A_2 \cdots A_n\!: + \sum_{\text{pairings}} \overbracket{\cdots} \cdots :\!\cdots\!:$$

where the right-hand side is a sum over all ways to pair some subset of the operators (with appropriate fermion signs from anti-commutations to bring contracted pairs adjacent), each pair replaced by its contraction, and the remaining operators normal-ordered.

**Corollary 3.2 (vacuum n-point function).** Taking $\langle 0_F | \cdots | 0_F \rangle$:

$$\langle 0_F | T(A_1 \cdots A_n) | 0_F \rangle = \sum_{\text{full pairings}} \prod_{(i, j) \in \text{pairing}} \overbracket{A_i A_j} \cdot \mathrm{sgn}(\text{permutation})$$

since only fully-contracted terms survive (the normal-ordered residue has vanishing vacuum expectation value).

For an even number $n = 2N$ of fields with $N$ $\psi$-fields and $N$ $\psi^\dagger$-fields:

$$\langle 0_F | T\big(\psi(g_1, t_1) \cdots \psi(g_N, t_N) \psi^\dagger(g_1', t_1') \cdots \psi^\dagger(g_N', t_N')\big) | 0_F \rangle = \sum_{\sigma \in S_N} \mathrm{sgn}(\sigma) \prod_{k=1}^N G_F^{\text{sub}}(g_k, g_{\sigma(k)}';\, t_k - t_{\sigma(k)}').$$

For an odd number $n$ or unbalanced $\psi/\psi^\dagger$ count: vanishes by charge conservation.

*Proof of Theorem 3.1.* Standard Wick induction (see Peskin-Schroeder §4.3 or Weinberg §6.1 for the continuum-QFT version). The proof requires only:

1. **CAR:** $\{A_i, A_j\}$ are c-numbers (specifically, $0$ except for $\{\psi(g), \psi^\dagger(g')\} = \delta_{g, g'}$). Holds on substrate per `../theorems/theorem_car_local_jordan_wigner.md`.

2. **Heisenberg time evolution preserves the algebra:** $A_i(t)$ are still CAR operators (with $t$-dependent coefficients in the Bloch-mode basis). Holds because $H$ is bilinear in $c, c^\dagger$ (free theory; substrate Hamiltonian $H = D_{\text{sub}}$ is bilinear under JW second quantization).

3. **Vacuum $|0_F\rangle$ annihilated by all annihilators after the Bogoliubov decomposition** ($c$ on positive-$\varepsilon$ modes, $d$ on negative-$\varepsilon$ modes). Holds per F1 §1.5.

The induction step is the same as in continuum QFT: pull one operator out of the time-ordered product, anti-commute it through (gathering Wick contractions with the contraction lemma 2.2), and apply the inductive hypothesis to the residual normal-ordered product. The only change from continuum-QFT is that "spacetime point" $x$ becomes "substrate vertex + time" $(g, t)$, and the propagator is $G_F^{\text{sub}}$ instead of the continuum $G_F$. $\square$

---

## 4. Concrete examples

### 4.1 Two-point function

$N = 1$: $\langle 0_F | T(\psi(g, t) \psi^\dagger(g', t')) | 0_F \rangle = G_F^{\text{sub}}(g, g'; t - t')$.

This is the F1 result.

### 4.2 Four-point function

$N = 2$: $\langle 0_F | T(\psi(g_1, t_1) \psi(g_2, t_2) \psi^\dagger(g_1', t_1') \psi^\dagger(g_2', t_2')) | 0_F \rangle$.

By Corollary 3.2:

$$= G_F(g_1, g_1') G_F(g_2, g_2') - G_F(g_1, g_2') G_F(g_2, g_1')$$

(suppressing time arguments; first term identity pairing $\sigma = \mathrm{id}$, sign $+1$; second term swap pairing $\sigma = (12)$, sign $-1$ from fermion exchange).

This is the **substrate analog of the standard QFT free-fermion 4-point function**: identity-pairing minus exchange-pairing. Generates the Fermi-Dirac statistics under exchange.

### 4.3 Six-point function

$N = 3$: 6 pairings ($S_3$), with signs $\pm 1$ from permutation parity. Sum gives the substrate's free-theory 6-point function.

In general, $N$-point connected and disconnected diagrams emerge from the $N!$ pairing sum.

---

## 5. Substrate Feynman diagrams

### 5.1 Diagrammatic representation

Each pair-contraction $G_F^{\text{sub}}(g_i, g_j;\, t_i - t_j)$ is represented by a **propagator line** between substrate vertex $(g_i, t_i)$ and $(g_j, t_j)$.

A free-theory n-point function is a sum over **complete pair-matchings** of the $n$ external lines, with fermion-permutation signs. The graphical structure is identical to standard QFT free-fermion Feynman diagrams.

### 5.2 Interaction vertices

Beyond free theory: any interaction term in the substrate Hamiltonian $H_{\text{int}}$ that is polynomial in $\psi, \psi^\dagger$ generates a Feynman vertex via Dyson expansion:

$$\langle 0_F | T(\cdots e^{-i \int H_{\text{int}} dt}) | 0_F \rangle = \sum_{n \geq 0} \frac{(-i)^n}{n!} \int dt_1 \cdots dt_n\, \langle 0_F | T(\cdots H_{\text{int}}(t_1) \cdots H_{\text{int}}(t_n)) | 0_F \rangle$$

each term expandable by Wick into a sum of Feynman diagrams with **internal vertices** (insertions of $H_{\text{int}}$) and **external lines** (the $\cdots$ field operators).

This is the standard QFT perturbation-theory machinery, transposed verbatim to substrate. The substrate's specific interaction vertices come from non-bilinear terms in the substrate Hamiltonian — beyond F1's free theory, into F5/F6 territory.

### 5.3 Connected vs disconnected diagrams

Cluster-decomposition / linked-cluster theorem: connected $n$-point functions $\langle 0_F | T(\cdots) | 0_F \rangle_C$ obtained from full $n$-point functions by subtracting disconnected products. Standard combinatorial identity transposes directly to substrate.

---

## 6. Implications for QFT ontology and cascade

### 6.1 Direct ontology landings

| QFT-postulated object | Substrate grounding (this document) |
|---|---|
| **Wick's theorem** | Theorem 3.1 + Corollary 3.2; standard Wick induction transposes given F1 + CAR + free $H$. |
| **n-point function** | Sum over pair-contractions of $G_F^{\text{sub}}$ with fermion signs (Corollary 3.2). |
| **Feynman-diagram structure** | Substrate free-theory diagrams = pair-matchings; interaction vertices via Dyson expansion. |
| **Fermi-Dirac exchange statistics** (4-point) | Identity-pairing minus exchange-pairing (§4.2); derived from CAR + Wick. |

### 6.2 Cascade unblocked

Combined with F1 propagator, F3 Wick gives the full free-theory perturbative apparatus. Now tractable:

- **F4 LSZ reduction (~1–2 sessions):** in/out asymptotic projections + LSZ formula relating S-matrix elements to amputated $G_F^{\text{sub}}$ chains. Standard derivation applies given F3.
- **F5 substrate S-matrix at lowest order (~2–3 sessions):** $\psi\psi \to \psi\psi$ scattering; Feynman diagram = single propagator + interaction vertex + propagator. Concrete numerical computation given specific interaction Hamiltonian.
- **F6 Feynman rules (~2 sessions):** vertex expressions from substrate interactions + propagator + Wick rules. Generates graphical perturbation expansion.
- **F7 substrate renormalization (~3+ sessions):** still highest-leverage; Wick provides the perturbative apparatus for connecting bare and renormalized n-point functions under coarse-graining.

### 6.3 What does NOT close

- **Substrate-specific interaction vertices.** F3 covers the Wick *theorem* (free-theory + Dyson-expansion structure). The actual substrate interaction Hamiltonian terms (4-fermion, gauge couplings, Yukawa-like) are NOT enumerated here — these are F5/F6 territory.
- **Renormalization (F7).** F3 generates bare perturbation series; renormalization-as-coarse-graining requires connecting to the I-projection apparatus (Tier 1 op 1).
- **Continuum-spacetime form.** Substrate Wick is rigorous on substrate-discrete level. Continuum-QFT diagrams require §C smooth-manifold closure.
- **Anomalies / non-perturbative effects.** Standard Wick gives perturbation theory; non-perturbative substrate phenomena (instantons, confinement) require more.

---

## 7. Honest scope

1. **Theorem-grade closure of F3.** The substrate Wick theorem is rigorous given F1 + CAR + free Hamiltonian + Dirac-sea vacuum. The proof inducts on standard structure transposed verbatim from continuum QFT.

2. **No substantive new mathematics.** The novelty is structural: substrate's CAR + bilinear-$H$ structure makes Wick *apply*, but the proof itself is standard. The deliverable is the mapping continuum-QFT-Wick → substrate-discrete-Wick, with each ingredient's substrate provenance.

3. **Free theory.** F3 covers the free-theory Wick. Interactions are F5/F6 follow-up. The Dyson-expansion structure (§5.2) shows how interactions plug in but doesn't enumerate substrate-specific vertices.

4. **No new SM-prediction.** Like prior Tier 1/Tier 2 forward-construction results, F3 is structural ontology grounding (category-2 yield).

5. **Cluster decomposition / linked-cluster.** §5.3 notes these transpose; explicit formulation is a 1-session bounded follow-up.

---

## 8. Status

**Substrate Wick theorem: theorem-grade.** Closes F3 of an internal note. Together with F1, gives the substrate free-theory perturbative apparatus.

**Category:** category-2 yield (~3–4 ontology objects newly grounded: Wick theorem, n-point functions, Feynman-diagram structure, fermion exchange statistics).

**Effect on framework:**
- Substrate inherits standard QFT perturbation theory directly.
- F4 (LSZ), F5 (S-matrix), F6 (Feynman rules), F7 (renormalization) all become concrete computational follow-ups (1–3 sessions for F4–F6; 3+ for F7).
- Combined with F1 Lichnerowicz-substituted propagator and G2 substrate curvature, the substrate's free-fermion-field theory is now computationally complete at the substrate-discrete level.

**Effect on QFT ontology meta-doc:** §8 — Wick / n-point / Feynman-diagram entries grounded; "Time-ordered products" entry now closed for fermionic case.

---

## 9. Cross-references

- `forward_construction_substrate_propagator.md` — F1 (substrate Feynman propagator, used in §4 + Lemma 2.2).
- `../theorems/theorem_car_local_jordan_wigner.md` — substrate CAR (used in proof of Theorem 3.1).
- `forward_construction_substrate_thermal_apparatus.md` §3 — substrate vacuum.
- `forward_construction_field_operator_phi_x.md` — parent setup; §7 step #4 closed by this doc.

**Type 3 (cited published) references:**

- **Wick, G. C.** (1950). The evaluation of the collision matrix. *Phys. Rev.* 80, 268–272. (Original Wick theorem.)
- **Peskin, M. E. & Schroeder, D. V.** (1995). *An Introduction to Quantum Field Theory.* Westview, §4.3 (Wick's theorem; n-point functions).
- **Weinberg, S.** (1995). *The Quantum Theory of Fields, Vol. I.* Cambridge, §6.1 (Wick + Dyson expansion).
- **Streater, R. F. & Wightman, A. S.** (1964). *PCT, Spin and Statistics, and All That.* Benjamin (Wick's theorem in algebraic-QFT formulation).

---

## 10. Next forward-construction steps

1. **F4 LSZ reduction** (~1–2 sessions): standard LSZ formula on substrate using F3 Wick + F1 propagator. Asymptotic Bloch-mode states from positive-energy projection.
2. **F5 substrate S-matrix at lowest order** (~2–3 sessions): $\psi\psi \to \psi\psi$ via Feynman diagrams + a specific substrate interaction Hamiltonian. Cross-validate against perturbative-QFT answer.
3. **F6 substrate Feynman rules** (~2 sessions): full graphical perturbation expansion; specific vertex enumeration.
4. **F7 substrate renormalization** (~3+ sessions): RG flow as coarse-graining I-projection; connection to A2-T's information-theoretic apparatus.
5. **Cluster decomposition / linked-cluster** (~1 session, bounded): connected n-point functions vs full; standard combinatorial identity transposed.
