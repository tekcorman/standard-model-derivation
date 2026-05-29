# Derivation: Feshbach Coupling Strength alpha_1 = (2/3)^(g-2)

**Status:** THEOREM-GRADE under A1 + A2-T + A5(b) + Jaynes 1957.
(Updated 2026-04-19 session 2: A5(b) — the coupling clause of A5,
`docs/framework/framework_axioms.md` §5b — subsumes the previously-adopted
"I-Feshbach" identification.) Lemma 1 (tree NB walk survival)
remains theorem-grade combinatorially; the identification with
the physical coupling is now A5(b)-axiomatic.
**Verification:** `predictions/Feshbach_coupling_strength.py` (all assertions pass).
**Upstream:** `predictions/k_star.py` (k* = 3), `predictions/g_girth.py` (g = 10),
`../predictions/B_P_doubly_degenerate_h_derivation.md` (C_3-protected h-eigenspace at P-point).

## Relation to P5

Together with `predictions/uniform_Q_density.py` (part A: rho_Q uniform), this
document addresses part B of P5 from docs/framework/W4_identification_catalog.md §3:

| P5 sub-part | Status |
|-------------|--------|
| rho_Q uniform on Ramanujan circle | theorem (predictions/uniform_Q_density.py Part A) |
| Shape of resolvent integral (Im(h)/|h|^2 factor) | theorem (contour integral + uniform density) |
| Coefficient alpha_1 = (2/3)^(g-2) | Lemma 1 is theorem; I-Feshbach is ADOPTED |

## Lemma 1: Tree NB Walk Survival (THEOREM-GRADE)

**Statement.** On the universal covering tree of a k-regular graph, the probability
that an NB walker stays on the tree for L consecutive steps is ((k-1)/k)^L.

**Proof.** At each vertex, k-1 of the k incident edges are NB-admissible (the
incoming edge is excluded).  Under the Jaynes-uniform distribution (walker_dynamics
Step 4), the unconditional probability of taking an admissible NB step is (k-1)/k.
On the universal covering tree, no two NB walks reconverge, so steps are independent:

    p_tree(L) = product_{i=1}^{L} (k-1)/k = ((k-1)/k)^L.  QED.

## Corollary: srs with L = g-2

For srs with k* = 3, g = 10:

    alpha_1^bare = p_tree(g-2) = (2/3)^8 = 256/6561 ≈ 0.0390.

This quantity is theorem-grade.

## The Exponent Principle (ADOPTED structural theorem)

The Feshbach coupling alpha_1^bare is an instance of the Exponent Principle:
on a k-regular graph of girth g, the NB walk suppression exponent for a process
with n_fixed fixed external edges is g - n_fixed.

| Process | n_fixed | Internal length | Amplitude |
|---------|---------|----------------|-----------|
| Scattering (2 external) | 2 | g-2 = 8 | alpha_1^bare = (2/3)^8 |
| Transition (1 external) | 1 | g-1 = 9 | (2/3)^9 |
| Self-energy (0 external) | 0 | g = 10  | (2/3)^10 |

**Status of Exponent Principle:** numerically verified on K_4 and srs
(hashimoto_exponents.py, exponent_ladder.py); Feynman-rule analog motivated;
NOT yet proved independently at journal grade.  It is treated as an adopted
structural theorem at the same status tier as P1, P2 in W4_identification_catalog.md.

## ADOPTED: I-Feshbach Identification

**What is adopted:** the identity C_{g-2} = alpha_1^bare * (k-1)^(g-2), where
C_{g-2} = PB(QB)^{g-2}QP is the (g-1)-th term in the Schur-complement expansion.

**Precise gap (from ../predictions/Feshbach_coupling_strength_derivation.md §9):**
If P and Q are spectral projectors (eigenspace decomposition) of B, then
[B, P] = 0, so PBQ = 0 identically.  The finite K_4 matrix computation
CANNOT close I-Feshbach for this algebraic reason (confirmed numerically:
max |PBQ| < 1e-15 with Riesz spectral projectors).

**Corrected closure routes:**

- **Route A (Ihara-Bass Green's function on srs):** Write G(u) = (I-uB)^{-1}
  on the srs lattice (infinite periodic band operator).  The u^{g-2} coefficient
  counts NB walks of length g-2 between generation-changing edge pairs.  The
  Exponent Principle predicts the normalised count equals (2/3)^{g-2} times a
  girth-cycle orientation factor.  Proving this analytically closes I-Feshbach.

- **Route B (physical P/Q definition):** Define P = "visible sector" and
  Q = "dark sector" via the C_3-isotypic decomposition or bipartite sublattice
  decomposition in a way ORTHOGONAL to the eigenbasis of B (so PBQ != 0).
  Show the Feshbach self-energy has leading coefficient (2/3)^{g-2} by the
  Exponent Principle.

## Downstream Consequences

- predictions/V_us.py Route A (Feshbach): blocked by I-Feshbach (not by density shape).
- predictions/V_cb.py commensurate (1+alpha_1) correction: same status.
- Type D Class-1 and Class-2 (theta_23, theta_12, m_nu2, m_nu3): depend on
  the magnitude alpha_1, so they inherit I-Feshbach ADOPTED status.

## References

- Bass, H. (1992). Ihara-Selberg zeta function. Int. J. Math. 3, 717-797.
- Ihara, Y. (1966). J. Math. Soc. Japan 18, 219-235.
- Sunada, T. (2012). Topological Crystallography. Springer. §7-8.
- Terras, A. (2011). Zeta Functions of Graphs. Cambridge. §2.3.
- predictions/k_star.py, predictions/g_girth.py (upstream derived values).
- ../predictions/walker_dynamics_derivation.md (W4: NB uniform distribution).
