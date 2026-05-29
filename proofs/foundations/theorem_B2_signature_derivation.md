# Theorem B2 — Signature (6, 0) of the canonical quadratic form on K₄-edge space

## Abstract

Given that no canonical ordering of the 6-edge space of K₄ exists (Theorem B1.b),
the Clifford algebra Cl(V, Q) requires a non-degenerate quadratic form Q on the
6-dimensional undirected-edge space of K₄ (the cell quotient of the srs
primitive cell). We compare three natural graph-theoretic candidates and find
that only the P-point Ramanujan projector form (candidate 3) is non-degenerate
on all of R⁶. Its signature is **(6, 0)** — Euclidean Cl(6, 0). Candidates 1
(Edge Laplacian) and 2 (Hashimoto symmetric) are rank-3, hence incompatible
with a Clifford structure on the full 6-space.

## Framework axioms invoked

- **A1** (self-inverse toggle): srs NB-walk → Hashimoto operator → Bloch
  decomposition → B(P) spectrum.
- **A2** (MDL): selects invariant formulation (Theorem B1.b upstream).
- **Clifford non-degeneracy** (Lawson & Michelsohn 1989 Ch. I §1): Q must be
  non-degenerate for Cl(V, Q) to be a Clifford algebra; this is the
  selection criterion that distinguishes candidate 3.

## Derivation

### Step 1 — Candidate 1: Edge Laplacian (signature (3, 0, 3))

The oriented incidence matrix $B_{\rm inc} \in \mathbb{R}^{4 \times 6}$ of K₄
(head = +1, tail = −1) gives the edge Laplacian

$$
Q_1 = B_{\rm inc}^T B_{\rm inc}.
$$

For any graph, the nonzero spectra of $B B^T$ (vertex Laplacian $L_v$) and
$B^T B$ (edge Laplacian $L_e$) coincide (Biggs 1993 §4.3). For K₄, $L_v$ has
eigenvalues $\{0, 4, 4, 4\}$. Hence:

$$
\mathrm{spec}\, Q_1 = \{0, 0, 0, 4, 4, 4\}, \quad \text{signature } (3, 0, 3).
$$

The 3-dim kernel is the cycle space of K₄ (circuit rank = |E| − |V| + 1 = 3).
Candidate 1 is **rank-deficient** — cannot define a Clifford generator on all
of $\mathbb{R}^6$.

### Step 2 — Candidate 2: Hashimoto symmetric (signature (1, 2, 3))

The 12×12 Hashimoto non-backtracking operator $B$ on directed edges of K₄ is
non-symmetric. Its symmetric part restricted to the 6-dim undirected quotient
via the symmetrisation map $S$ (defined by $S(\hat{e}_j) = (\hat{e}_{\rm fwd(j)} + \hat{e}_{\rm rev(j)})/\sqrt{2}$):

$$
Q_2 = S^T \left(\frac{B + B^T}{2}\right) S.
$$

Numerical computation yields $\mathrm{spec}\, Q_2 = \{-1, -1, 0, 0, 0, 2\}$:

$$
\text{signature } (1, 2, 3).
$$

Candidate 2 is **indefinite and rank-deficient** — cannot define a Clifford
structure.

### Step 3 — Candidate 3: P-point Ramanujan projector (signature (6, 0))

Build $B(P)$: the 12×12 Bloch Hashimoto operator on the srs primitive cell at
$k = P = (\tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{4})$. Its spectrum
(Theorem BP) is

$$
\mathrm{spec}\, B(P) = \{h, h, h^*, h^*, -h, -h, -h^*, -h^*, +1, +1, -1, -1\},
\quad h = \frac{\sqrt{3} + i\sqrt{5}}{2}.
$$

Let $P_R$ be the orthogonal projector onto the 8-dim Ramanujan eigenspace
(eigenvalues $\{h, h^*, -h, -h^*\}$, mult 2 each). Construct via QR:
$P_R = Q_{\rm orth} Q_{\rm orth}^*$, where $Q_{\rm orth}$ orthonormalises the
Ramanujan eigenvectors. Verified: $P_R = P_R^*$, $P_R^2 = P_R$, ${\rm tr}(P_R) = 8$.

The 6-dim Hermitian form on the undirected-edge quotient:

$$
Q_3 = S^* P_R S, \quad S : \mathbb{C}^6 \to \mathbb{C}^{12}
$$

using the same symmetrisation $S$ adapted to the cell directed-edge labelling.
Its eigenvalues are:

$$
\mathrm{spec}\, Q_3 = \left\{
\frac{3 - \sqrt{3}}{6},\; \frac{3 - \sqrt{3}}{6},\;
\frac{3 + \sqrt{3}}{6},\; \frac{3 + \sqrt{3}}{6},\;
1,\; 1
\right\} \approx \{0.2113, 0.2113, 0.7887, 0.7887, 1, 1\}.
$$

All eigenvalues are strictly positive (closed-form error $< 5 \times 10^{-16}$):

$$
\text{signature } (6, 0, 0).
$$

Candidate 3 is **non-degenerate** — the Clifford algebra is **Cl(6, 0)**,
Euclidean.

### Step 4 — Why candidates 1 and 2 fail

Both K₄-structural forms are insensitive to the 3-dim cycle subspace of the
6-edge space (the null directions of candidate 1 coincide with the cycle space
of K₄). The P-point projector, by contrast, couples all 6 undirected edges
through complex Bloch phases and projects onto a subspace that is sensitive to
the full 6-dim structure. Non-degeneracy is therefore a consequence of the
k-space phase mixing at $k = P$.

## Result

Among the three candidates, **only candidate 3 (P-point Ramanujan projector)
yields a non-degenerate quadratic form on R⁶**, and its signature is **(6, 0)**
— the framework's Clifford algebra is **Cl(6, 0)**, Euclidean.

**Uniqueness caveat:** This result shows that among the three candidates
(Edge Laplacian, Hashimoto symmetric, P-point projector), only candidate 3
satisfies the non-degeneracy criterion. A full uniqueness proof for all
$S_4$-equivariant quadratic forms on $\mathbb{R}^6$ is an open task (B2.4).

## Comparison with experiment

N/A — foundational theorem. The signature (6, 0) determines the Clifford
algebra Cl(6, 0) used by downstream workstreams B3–B4. It makes no direct
numerical prediction; its physical consequence is that the framework operates
in a Euclidean (not Lorentzian) Clifford setting, consistent with the
spatial-lattice character of srs.

## Open questions

- **(B2.1)** The non-degeneracy selection criterion is Clifford-theoretic, not
  purely graph-theoretic. An MDL-internal argument ruling out candidates 1 and 2
  without invoking "we want a Clifford algebra" is not yet found.
- **(B2.2)** The Euclidean (6, 0) signature does not produce Lorentzian
  signature for relativistic spinors; the "+1 time" extension is an open gap.
- **(B2.5)** The closed-form eigenvalues $\{(3 \pm \sqrt{3})/6, 1\}$ are
  verified numerically but a pen-and-paper symbolic derivation from the Bloch
  Hashimoto matrix is an open clean-up task.

## References

- Biggs, N. (1993). *Algebraic Graph Theory* 2nd ed. Cambridge Univ. Press.
  §4.2–4.3 (edge Laplacian).
- Lawson, H.B. & Michelsohn, M.-L. (1989). *Spin Geometry.* Princeton.
  Ch. I §1 (non-degeneracy requirement).
- Terras, A. (2011). *Zeta Functions of Graphs.* Cambridge. Ch. 2 (Hashimoto).
- `../predictions/B_P_doubly_degenerate_h_derivation.md` — P-point spectrum of B(P).
