# H_multiway dim-count lemma — length-graded visible/dark decomposition of the F_inv(E) Hilbert space

**Date:** 2026-04-17
**Status:** Theorem (dim-count lemma) + open dispersion question.
**Verification:** sympy + brute-force enumeration (`predictions/H_multiway_dim_count.py`); foundation-side checks (`proofs/foundations/H_multiway_construction.py`).
**Companion:** `docs/theorem_H_multiway_construction.md` (full construction, Schur-complement analysis, open questions).
**Sub-target:** F.1-O3 of an internal working note (closest-to-existing-material attack on the Layer-1 multiway formalisation gap).

## Abstract

We construct the Layer-1 multiway Hilbert space $\mathcal{H}_{\text{multiway}}$ explicitly as the length-graded Hilbert-space lift of the free involutive monoid $F_{\text{inv}}(E)$ derived from axiom A1 of the framework (`../predictions/walker_dynamics_derivation.md` Steps 1-2). On each length-$L$ slice we exhibit the canonical decomposition

$$\mathcal{H}_{\text{multiway}}^{(L)} \;=\; \mathcal{H}_{\text{visible}}^{(L)} \,\oplus\, \mathcal{H}_{\text{dark}}^{(L)},$$

where $\mathcal{H}_{\text{visible}}^{(L)}$ is the span of length-$L$ reduced words (the canonical image of MDL canonicalization, A2-T) and $\mathcal{H}_{\text{dark}}^{(L)}$ is the span of length-$L$ strings containing at least one adjacent cancellable pair $e \cdot e$ (the kernel of the canonicalization map within the length-$L$ slice).

The dimensions admit the closed form

$$\dim \mathcal{H}_{\text{multiway}}^{(L)} = n^L,\quad
\dim \mathcal{H}_{\text{visible}}^{(L)} = R_L = n(n-1)^{L-1}\ \ (L\geq 1),\quad
\dim \mathcal{H}_{\text{dark}}^{(L)} = D_L = n\!\left[n^{L-1} - (n-1)^{L-1}\right]\ \ (L\geq 1),$$

with $n = |E| = 6$ for the srs primitive cell and $D_0 = 0$, $R_0 = 1$. The closed form is verified by sympy and by direct brute-force enumeration of all $n^L$ length-$L$ strings for $L = 0, 1, \dots, 7$.

The asymptotic dark fraction $D_L / n^L = 1 - ((n-1)/n)^{L-1}$ converges to 1 geometrically: almost all sufficiently long strings are dark.

This is the dim-count half of sub-target F.1-O3 of an internal working note. The companion question of whether the canonical operator structure on $\mathcal{H}_{\text{multiway}}$ produces a Schur-complement modification of the visible Bloch dispersion at small $q$ is **open**: the canonical block structure has $B_{VD} = 0$ (dark is absorbing in $F_{\text{inv}}(E)$), so the Schur complement reduces to $T_{\text{eff}} = B_{VV}$ and leaves $\gamma_{\text{phys}} = 1/16$ unmodified. See `docs/theorem_H_multiway_construction.md` §Check F for the precise statement of the remaining gap.

## Framework axioms invoked

- **(A1)** Binary self-inverse toggle: each edge $e$ carries a toggle operator $T_e$ with $T_e \cdot T_e = 1$.
- **(A2-T)** MDL: the observer selects models minimising total description length $L(M) + L(\text{data} \mid M)$ (derived theorem; see `docs/theorems/theorem_A2_mdl_from_finite_register.md`).

NOTE (post-2026-04-26 demotion): A2 and A3 are derived theorems; structural slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure chain referenced here is preserved; only the axiomatic-status labels change. The present dim-count lemma uses only A1 + A2-T; A3-T is not required for the combinatorial content of this file.

## Derivation

### Step 1 — Alphabet $E$ from upstream

By `predictions/d_spatial.py` and `predictions/k_star.py`, $d = 3$ and $k_* = 3$; together these select the srs lattice (space group I4$_1$32, Wyckoff 8a) as the MDL-optimal compressed graph (`predictions/g_girth_derivation.md` §2). The srs primitive cell has $|V| = 4$ vertices and $|E| = k_* \cdot |V| / 2 = 6$ undirected edges.

**The alphabet $E$ of the multiway substrate is therefore taken to be these 6 undirected edges per primitive cell.** Each edge $e \in E$ carries a toggle operator $T_e$ by axiom A1. The choice of "undirected edges" (rather than "directed edges") is forced by axiom A1: $T_e^2 = 1$ requires the toggle to be its own inverse, which is the property of an undirected edge under the involution that identifies $e$ with its reverse. (The directed-edge picture, with alphabet of size $2|E| = 12$, is the one used in the visible-side Hashimoto operator $B$, which acts on directed-edge causal states per `../predictions/walker_dynamics_derivation.md` Step 5; that picture is the projection of the present undirected-alphabet $F_{\text{inv}}(E)$ to the visible reduced words.)

### Step 2 — Length-graded $F_{\text{inv}}(E)$ Hilbert space

`../predictions/walker_dynamics_derivation.md` Step 1 establishes that, under axiom A1 plus the standard monoid-congruence definition, the toggle stream space is the **free involutive monoid**

$$F_{\text{inv}}(E) \;=\; E^* \;/\; \big(e \cdot e \sim \varepsilon : e \in E\big),$$

a quotient of the free monoid $E^*$ on $E$ (Serre 1980, *Trees*, §I.1 Prop. 4).

We now lift $E^*$ and $F_{\text{inv}}(E)$ to Hilbert spaces. For each $L \geq 0$, define the **length-$L$ unreduced multiway Hilbert space**

$$\mathcal{H}_{\text{multiway}}^{(L)} \;:=\; \mathbb{C}^{|E|^L} \;=\; \underbrace{\mathbb{C}^{|E|} \otimes \mathbb{C}^{|E|} \otimes \cdots \otimes \mathbb{C}^{|E|}}_{L\ \text{factors}}$$

with orthonormal basis $\{|w\rangle : w \in E^L\}$ and standard $L^2$ inner product $\langle w | w' \rangle = \delta_{w, w'}$. The full multiway Hilbert space is the length-graded direct sum

$$\mathcal{H}_{\text{multiway}} \;:=\; \bigoplus_{L \geq 0} \mathcal{H}_{\text{multiway}}^{(L)}.$$

Then

$$\boxed{\dim \mathcal{H}_{\text{multiway}}^{(L)} = n^L, \qquad n := |E| = 6.}$$

### Step 3 — Canonicalization map $\pi$ and reduced-word subspace

`../predictions/walker_dynamics_derivation.md` Step 2 shows that under A2-T the observer selects, from each equivalence class $[w] \in F_{\text{inv}}(E)$, the unique reduced (no two adjacent letters equal) representative $r(w) \in E^*$ (Serre 1980 §I.1 Prop. 4; Grünwald 2007 *The Minimum Description Length Principle* §5.1-5.3).

Define the **canonicalization map**

$$\pi : \mathcal{H}_{\text{multiway}}^{(L)} \;\longrightarrow\; \bigoplus_{L' \leq L,\ L - L' \text{ even}} \mathbb{C}^{R_{L'}},
\qquad \pi(|w\rangle) = |r(w)\rangle,$$

where $R_{L'} = \#\{w \in E^{L'} : w \text{ is reduced}\}$. The parity restriction $L - L' \in 2\mathbb{Z}_{\geq 0}$ holds because cancellation $e \cdot e \to \varepsilon$ removes letters in pairs.

**Visible Hilbert space at length $L$.** The image of $\pi$ restricted to length-$L$ reduced words (i.e., those with $r(w) = w$, equivalently those with no adjacent equal letters) embeds isometrically into $\mathcal{H}_{\text{multiway}}^{(L)}$ as the subspace

$$\mathcal{H}_{\text{visible}}^{(L)} \;:=\; \mathrm{span}\big\{|w\rangle : w \in E^L,\ w\ \text{reduced}\big\}.$$

This is the natural Hilbert-space lift of the visible (MDL-compressed) sector at length $L$.

**Dark Hilbert space at length $L$.** Define

$$\mathcal{H}_{\text{dark}}^{(L)} \;:=\; \mathrm{span}\big\{|w\rangle : w \in E^L,\ w\ \text{NOT reduced}\big\}.$$

Then $\mathcal{H}_{\text{multiway}}^{(L)} = \mathcal{H}_{\text{visible}}^{(L)} \oplus \mathcal{H}_{\text{dark}}^{(L)}$ as an orthogonal direct sum (the two spans are disjoint and exhaust the basis of $\mathcal{H}_{\text{multiway}}^{(L)}$).

This is the direct-sum decomposition stated in the abstract.

**Identification of $\mathcal{H}_{\text{dark}}^{(L)}$ as the kernel of canonicalisation within length $L$.** A length-$L$ string $w$ is reduced iff $r(w) = w$ iff $\pi$ acts trivially on $|w\rangle$ within length $L$ (no length reduction occurs). The dark space is exactly the orthogonal complement of the trivially-acted-upon subspace within $\mathcal{H}_{\text{multiway}}^{(L)}$. Equivalently, $\mathcal{H}_{\text{dark}}^{(L)}$ is the span of basis vectors that $\pi$ maps to a strictly shorter reduced word (i.e., "cancellation-event-bearing" strings). This matches the user-supplied informal heuristic in option F.1-O3 of an internal working note.

### Step 4 — Closed form for $R_L$ and $D_L$

**Reduced-word count $R_L$.** A reduced word of length $L$ is a string $(e_1, e_2, \dots, e_L)$ with $e_i \neq e_{i+1}$ for all $i$. By elementary counting:

$$R_0 = 1, \qquad R_1 = n, \qquad R_L = (n-1) \cdot R_{L-1} \quad (L \geq 2).$$

(At each step after the first, the next letter must differ from the previous, giving $n-1$ choices.) Solving the recursion:

$$\boxed{R_L = n (n-1)^{L-1} \quad (L \geq 1),\qquad R_0 = 1.}$$

This is the standard length-$L$ count for the free involutive monoid on $n$ generators (Serre 1980 §I.1 Prop. 4; equivalently the count of length-$L$ closed walks on the bouquet $K_{1, n}$ with self-loops removed; Terras 2011 §2.1 for the analogous NB-walk count on graphs).

**Dark-word count $D_L$.** Total minus reduced:

$$D_L \;=\; n^L - R_L \;=\; n^L - n(n-1)^{L-1} \;=\; n \cdot \big[n^{L-1} - (n-1)^{L-1}\big] \quad (L \geq 1),$$

with $D_0 = 0$.

The factored form follows by elementary algebra:

$$n^L = n \cdot n^{L-1}, \qquad n(n-1)^{L-1} = n \cdot (n-1)^{L-1},$$

so $D_L = n[n^{L-1} - (n-1)^{L-1}]$. (Verified symbolically by sympy in `predictions/H_multiway_dim_count.py`: `assert sp.expand(D_closed - D_factored) == 0`.)

**Numerical values for $n = 6$:**

| $L$ | $\dim \mathcal{H}_{\text{multiway}}^{(L)} = n^L$ | $R_L$ | $D_L$ | $D_L / n^L$ |
|---|---|---|---|---|
| 0 | 1 | 1 | 0 | 0.000 |
| 1 | 6 | 6 | 0 | 0.000 |
| 2 | 36 | 30 | 6 | 0.167 |
| 3 | 216 | 150 | 66 | 0.306 |
| 4 | 1 296 | 750 | 546 | 0.421 |
| 5 | 7 776 | 3 750 | 4 026 | 0.518 |
| 6 | 46 656 | 18 750 | 27 906 | 0.598 |
| 7 | 279 936 | 93 750 | 186 186 | 0.665 |

The ratio $D_L / n^L \to 1$ as $L \to \infty$ since $((n-1)/n)^{L-1} \to 0$. **Almost all sufficiently long strings are dark.**

### Step 5 — Brute-force verification

The script `predictions/H_multiway_dim_count.py` enumerates all $n^L$ length-$L$ strings for $L = 0, 1, \dots, 7$, classifies each as reduced or non-reduced via the standard linear-time check (`_is_reduced`), and confirms the closed-form $R_L = n(n-1)^{L-1}$ to exact integer equality. (See script output.)

This closes the dim-count half of the construction.

## Result

The length-graded multiway Hilbert space and its canonical visible/dark decomposition are constructed explicitly:

$$\mathcal{H}_{\text{multiway}} = \bigoplus_{L \geq 0} \mathcal{H}_{\text{multiway}}^{(L)}, \qquad \mathcal{H}_{\text{multiway}}^{(L)} = \mathcal{H}_{\text{visible}}^{(L)} \oplus \mathcal{H}_{\text{dark}}^{(L)},$$

with closed-form dimensions

$$\dim \mathcal{H}_{\text{visible}}^{(L)} = n(n-1)^{L-1}, \qquad \dim \mathcal{H}_{\text{dark}}^{(L)} = n^L - n(n-1)^{L-1}, \qquad n = |E| = 6.$$

For the srs primitive cell ($n = 6$):

- Per-step Jaynes-uniform branching (A2-T + walker_dynamics Step 4): probability $(n-1)/n = 5/6$ that an unreduced extension stays in the visible sector at the F$_\text{inv}$($E$) free-monoid level (or $(k-1)/k = 2/3$ at the on-graph level with $k = 3$); probability $1/n = 1/6$ (free-monoid) or $1/k = 1/3$ (on-graph) of cancellation, which removes the extending pair from the visible reduced word.

This is the formal Layer-1 multiway Hilbert space whose absence was the load-bearing blocker in:


**Status of the $n_s$ / $r$ / B7.3 / MS.1 unblocking:** the dim count alone does **not** unblock these. The companion question — whether the canonical operator structure on $\mathcal{H}_{\text{multiway}}$ produces a non-trivial Schur-complement modification of the visible Bloch dispersion at small $q$ — is **open**, and is shown in `docs/theorem_H_multiway_construction.md` to require structure beyond the canonical $F_{\text{inv}}(E)$ block decomposition (the canonical $B_{VD} = 0$ makes the Schur reduction trivial). See "Open questions" below for the precise statement.

## Comparison with experiment

This is a structural lemma, not an empirical observable. There is no PDG number to compare with. The "comparison" is internal:

- **Brute-force enumeration vs closed form:** exact agreement for $L = 0, \dots, 7$ (verified in the script).
- **Asymptotic dark fraction $D_L/n^L \to 1$:** consistent with the dark-matter-as-uncompressed-multiway-residue picture developed in an external research note (most multiway content is uncompressed = dark) and with $\Omega_{\text{DM}}/\Omega_m = 0.842$ (`predictions/Omega_DM_over_Omega_m.py` — already closed at theorem grade, derived from a different bounded-capacity argument). The two derivations are consistent (both predict dark > visible) but not identical (the $\Omega$ ratio uses the eddy-framework $N^*$ at finite capacity, not the asymptotic $L \to \infty$ ratio $D_L / n^L$); a precise quantitative bridge between the asymptotic $L$-graded dark fraction and the cosmological $\Omega_{\text{DM}}/\Omega_m$ remains an open question (see below).

## Open questions

1. **Schur-complement dispersion modification (load-bearing).** The canonical $F_{\text{inv}}(E)$ block decomposition of the per-step extension operator $B_{\text{full}}$ has the lower-triangular form

   $$B_{\text{full}} = \begin{pmatrix} B_{VV} & 0 \\ B_{DV} & B_{DD} \end{pmatrix},$$

   because dark strings are an absorbing class in $F_{\text{inv}}(E)$ (extending a dark string by any letter produces another dark string; cancellations only delete letters, never create them). Therefore the Schur complement on the visible side is trivial: $T_{\text{eff}}(E) = B_{VV} - 0 \cdot (E - B_{DD})^{-1} \cdot B_{DV} = B_{VV}$, and the visible Bloch dispersion $\gamma_{\text{phys}} = 1/16$ is unchanged. To obtain a non-trivial dispersion modification one would need a non-zero $B_{VD}$ (D → V "decompression"), which is not derivable from MDL + toggle alone. Per `docs/theorem_H_multiway_construction.md` §Check F, candidate sources include (i) a finite-truncation Wolfram rule with explicit decompression, (ii) a unitary (rather than Markov) walker on $\mathcal{H}_{\text{multiway}}$ whose adjoint provides the missing channel, or (iii) a Layer-0 supplementary axiom. None is currently derivable.

2. **Quantitative bridge to $\Omega_{\text{DM}}/\Omega_m$.** The asymptotic dark fraction $D_L/n^L \to 1$ as $L \to \infty$ is a leading-$L$ statement; the cosmological $\Omega_{\text{DM}}/\Omega_m = 0.842$ is a finite-ratio observable. Reconciling the two requires identifying the physical $L$ at which the cosmological ratio is read off (presumably a bounded compression capacity $L = N^*$ in the associated eddy-framework research note), at which $D_{N^*}/n^{N^*}$ should match $\Omega_{\text{DM}}/\Omega_m$. This is a separate quantitative target, not addressed by the dim count alone.

3. **Dark-sector Bloch fibre at $P$ and connection to F.1.** The construction does not directly compute $B_{\text{dark}}(P)$, the dark-side Bloch fibre at the P-point of the srs Brillouin zone, which was the proposed dark block in an internal working note. The canonical $B_{DD}$ at the F$_\text{inv}$($E$) level acts on the full $D_L$-dimensional dark space at length $L$; its Bloch decomposition (using lattice-translation symmetry inherited from the alphabet) gives a per-fibre operator whose dimension grows with $L$. Constructing the $L \to \infty$ limit (or finding a natural finite truncation) is a separate computation not undertaken here. This is option F.1-O3 of the flux-operator-at-P attempt; the dim-count side is closed but the Bloch-fibre construction at $P$ is left for the companion theorem doc and downstream sub-targets.

4. **Cl(6, 0) spinor / generation channel structure on the dark side.** Whether the dark sector carries a non-trivial $C_3$ representation distinct from the visible side's $(4, 2, 2)$ isotypic decomposition (relevant to the F.1 step 4 "$(4, 2, 2)$ emergence" question and to B7.3 Need A and Need D) is not addressed by the dim count. The companion theorem doc sketches this question and flags it as a separate sub-target.

These open questions are documented precisely in `docs/theorem_H_multiway_construction.md` §"Open questions"; see that doc for the full status of what closes and what remains.

## References

### Framework axioms / upstream

- `../predictions/walker_dynamics_derivation.md` Step 1 (F$_\text{inv}$($E$) from A1), Step 2 (MDL canonicalization from A2), Step 4 (Jaynes-uniform branching).
- `../predictions/B_P_doubly_degenerate_h_derivation.md` (visible $B(P)$ spectrum, $h$-eigenvalue mult 2).
- `predictions/k_star.py` ($k_* = 3$).
- `predictions/d_spatial.py` ($d = 3$).
- `predictions/g_girth_derivation.md` §2 (srs primitive cell, $|V| = 4$, $|E| = 6$).
- `predictions/srs_bloch_dispersion_gamma.py` ($\gamma_{\text{phys}} = 1/16$, used as the visible-side reference dispersion).
- `predictions/B_P_doubly_degenerate_h.py` (visible 12-dim Bloch fibre at $P$).
- `predictions/Omega_DM_over_Omega_m.py` (cosmological dark/luminous ratio reference).

### Cited mathematical theorems

- **Serre, J.-P.** (1980). *Trees.* Springer-Verlag. §I.1 Prop. 4 (free involutive monoid on $n$ generators; reduced-word canonical form; length-$L$ reduced-word count).
- **Terras, A.** (2011). *Zeta Functions of Graphs: A Stroll through the Garden.* Cambridge University Press. §2.1 (NB walks and reduced words on graphs).
- **Grünwald, P.** (2007). *The Minimum Description Length Principle.* MIT Press. §1.9 (universal coding), §5.1-5.3 (model equivalence and canonicalization).
- **Cover, T. M. & Thomas, J. A.** (2006). *Elements of Information Theory*, 2nd ed. Wiley-Interscience. Theorem 5.3.1 (Kraft inequality), Theorem 5.4.3 (optimal prefix codes) — used for the length-graded code interpretation.
- **Reed, M. & Simon, B.** (1978). *Methods of Modern Mathematical Physics, IV: Analysis of Operators.* Academic Press. §XIII.4 (Schur complement / Feshbach reduction; cited in companion doc).
- **Kato, T.** (1980). *Perturbation Theory for Linear Operators*, 2nd ed. Springer Grundlehren **132**. §II.2 (block-perturbation theory).

### Companion documents

- `docs/theorem_H_multiway_construction.md` — full construction including the operator-coupling analysis, Schur-complement computation, and open questions.
- `proofs/foundations/H_multiway_construction.py` — sympy + numpy verification of all six steps A-F of the construction.

### Negative-result siblings (related stalled attempts)


## Files referenced but NOT modified

Per task constraints: `results/parameters.csv`, `docs/parameters/derivations.md`, B3/B6 docs, `../predictions/walker_dynamics_derivation.md`, and existing scoping docs are NOT modified. No commits performed; no remote push.
