# Tree subspace of B(P) on srs: C_3 isotypic decomposition and flat dispersion

## Abstract

For the srs Bloch non-backtracking walker B(k) on the 4-vertex primitive
cell of the I4_132 + Wyckoff 8a embedding, the Ihara-Bass identity
(Ihara 1966; Bass 1992; Terras 2011 Theorem 3.1) gives a k-independent
factor (1 - u^2)^(|E| - |V|) in the secular polynomial det(I - u B(k)).
For srs primitive (|V| = 4, |E| = 6), this factor is (1 - u^2)^2,
contributing eigenvalues +/- 1 each with multiplicity exactly 2 at every
k. The corresponding 4-dimensional 'tree subspace' V_tree at the P-point
is C_3-invariant under the 432-point-group stabilizer of P. Character
orthogonality on C_3 = {e, sigma, sigma^2} gives the isotypic
decomposition

    V_tree = 0 . trivial  +  2 . omega  +  2 . omega-bar

where omega = exp(2 pi i / 3). Equivalently, each of the +1 and -1
eigenspaces decomposes as omega + omega-bar (one copy each). The tree
dispersion is identically flat: lambda_tree(P + q) = +/-1 for all |q|,
neither linear (Dirac) nor quadratic (Compton). Result is sympy-exact
(symbolic char poly factors as (u-1)^2 (u+1)^2 (4u^4 + u^2 + 1)^2; all
commutators vanish symbolically) and verified numerically (random-k
flat-band check at five k in [-1/2, 1/2]^3, max |mu_tree - (+/-1)| <
1.3e-15).

This is a strict-solid structural lemma using only Ihara-Bass and
character orthogonality. No Hilbert-space postulates (Lindblad, Born),
no mass postulates (P1 / P2), no observer-side bridge.

## Framework axioms invoked

- **A2-T (MDL).** Selects k* = 3, d = 3, srs = I4_132 Wyckoff 8a
  (predictions/k_star.py, predictions/d_spatial.py,
  predictions/g_girth_derivation.md sec 2).
- **A1 (binary self-inverse toggle).** Sets up the walker dynamics whose
  Hashimoto operator B is the 1-step transition on directed edges
  (../predictions/walker_dynamics_derivation.md, W1-W3).

No other axiom or postulate is used. In particular, no P1 or P2 mass
postulate, no Lindblad / Born structural reading, no parameter-bridge
gluing.

## Derivation

### Step 1: Build B(k) symbolically; verify char poly factor (1 - u^2)^2

Use the same primitive-cell bond list as
predictions/B_P_doubly_degenerate_h.py: 12 directed edges (es, et, cell)
on the 4-vertex bcc-primitive cell of srs.

The Bloch non-backtracking walker is

    B(k)[f, e] = [e.target = f.source] [f != reverse(e)] exp(-2 pi i k . e.cell)

acting on the 12-dim directed-edge space. At P = (1/4, 1/4, 1/4), sympy
evaluates B(P) symbolically (12x12 entries in {0, +/- 1, +/- i, +/- (1 +/- i)/...}).

The characteristic polynomial is

    det(I - u B(P)) = (u - 1)^2 (u + 1)^2 (4 u^4 + u^2 + 1)^2

as a polynomial in u over the rationals. Sympy verifies this exactly.

Dividing by (1 - u^2)^2 = (u - 1)^2 (u + 1)^2 gives quotient
(4 u^4 + u^2 + 1)^2 with zero remainder (sympy-exact).

The Ihara-Bass identity (Terras 2011 Theorem 3.1, Ihara 1966, Bass 1992)
for a k_star-regular graph with |V| vertices and |E| edges gives, for
ALL k:

    det(I - u B(k)) = (1 - u^2)^(|E| - |V|) det((1 + (k_star - 1) u^2) I - u A(k))

Here |E| - |V| = 6 - 4 = 2, so the (1 - u^2)^2 prefactor is structural
and k-independent. The +/-1 roots of B(k) appear with multiplicity
exactly 2 at every k in BZ_primitive.

### Step 2: Tree subspace V_tree

Define the tree subspace V_tree at k as the direct sum of the +1 and
-1 eigenspaces of B(k):

    V_tree(k) := ker(B(k) - I) (+) ker(B(k) + I)

By Step 1, dim V_tree(k) = 2 + 2 = 4 at every k. The complementary
'Ramanujan subspace' (eigenvalues h, h*, -h, -h* at P) has dim 8.

V_tree is the inflation to the directed-edge space of the cycle space
Z_1 of srs (the topological/homological subspace orthogonal to the
boundary map d^T). This is the standard interpretation of the tree
prefactor in Ihara-Bass.

### Step 3: C_3 action symbolically; commutator with B(P) vanishes

The C_3 element of the 432 point group is the 120-degree rotation about
the body diagonal (1, 1, 1). It induces:

- Vertex permutation sigma = (v_0)(v_1 v_3 v_2):
  v_0 -> v_0, v_1 -> v_3, v_3 -> v_2, v_2 -> v_1.
- Lattice cell permutation: (n_1, n_2, n_3) -> (n_3, n_1, n_2).
  (Derivation: a1 = (-1/2, 1/2, 1/2) -> (1/2, -1/2, 1/2) = a2 under
  (x, y, z) -> (z, x, y); so a_i -> a_{i+1} cyclically, hence
  n_i a_i -> n_i a_{i+1} = n_{i-1} a_i with index shift.)
- Reduced k-coordinate permutation: (k_1, k_2, k_3) -> (k_3, k_1, k_2),
  which fixes P = (1/4, 1/4, 1/4) exactly.

Build the directed-edge action C_3 in the 12-dim basis. For each edge
(es, et, ec), its image is (sigma[es], sigma[et], rotate(ec)). When the
image cell does not match the canonical cell representative in the
12-list, absorb the difference as a Bloch phase exp(-2 pi i P . shift).

Sympy verifies symbolically:

- C_3^3 = I (12x12, exact)
- [C_3, B(P)] = 0 (12x12, exact)

The first establishes that C_3 is a genuine order-3 unitary; the second
establishes that C_3 commutes with B(P) (P being a fixed point of C_3
on the BZ).

### Step 4: C_3 trace on V_tree; isotypic decomposition

Since [C_3, B(P)] = 0, C_3 commutes with the spectral projectors of
B(P) and therefore restricts to a unitary action on V_tree.

Compute the trace of C_3 on V_tree:

    tr(C_3 on V_tree) = -2 + 0 i

(numerically verified to 10 decimals; sympy + numpy via projector onto
the 4-dim +/-1 eigenspaces).

Apply character orthogonality on C_3 = {e, sigma, sigma^2}, |G| = 3
(Serre 1977 ch 2). The three irreducible characters are

    chi_trivial(g)   = (1, 1, 1)
    chi_omega(g)     = (1, omega, omega^2)
    chi_omega_bar(g) = (1, omega^2, omega)

with omega = exp(2 pi i / 3). For a representation R of dimension d
with character (d, tr R(sigma), tr R(sigma^2)) = (4, -2, -2):

    mult_chi = (1/3) sum_g conj(chi(g)) tr R(g)

- mult_trivial   = (1/3)(4 + (-2) + (-2))                                 = 0
- mult_omega     = (1/3)(4 + omega^2 (-2) + omega (-2))                   = (1/3)(4 - 2(omega + omega^2)) = (1/3)(4 - 2(-1)) = (1/3)(6)/3 ... let me redo. (4 - 2 omega^2 - 2 omega) = 4 + 2 = 6, divided by 3 = 2.
- mult_omega_bar = (1/3)(4 + omega(-2) + omega^2(-2))                     = (4 + 2)/3 = 2.

(Closed form: with (d, tr_sigma, tr_sigma2) = (4, -2, -2),
mult_trivial = (4 - 2 - 2)/3 = 0,
mult_omega = (4 - 2 omega^2 - 2 omega)/3 = (4 - 2(-1))/3 = 6/3 = 2,
mult_omega_bar = (4 - 2 omega - 2 omega^2)/3 = 6/3 = 2.)

Therefore

    V_tree = 0 . trivial + 2 . omega + 2 . omega-bar.

Uniqueness: the only triple (a, b, c) of nonnegative integers with
a + b + c = 4, b = c (forced by tr being real-valued for a complex-
self-conjugate representation), and a + b * 2 cos(2 pi / 3) =
a - b = -2 is (a, b) = (0, 2), i.e. (0, 2, 2).

### Step 5: Per-eigenspace decomposition

Each of the +1 and -1 eigenspaces is 2-dim and C_3-invariant
(since [C_3, B(P)] = 0). Compute the trace of C_3 on each:

- tr(C_3 on +1) = -1
- tr(C_3 on -1) = -1

(numerically verified). With (d, tr_sigma, tr_sigma2) = (2, -1, -1) on
each:

- mult_trivial = (2 - 1 - 1)/3 = 0
- mult_omega   = (2 - omega^2 - omega)/3 = (2 + 1)/3 = 1
- mult_omega_bar = 1

So each of the +1 and -1 eigenspaces decomposes as omega + omega-bar
under C_3. The 4-dim V_tree decomposition (0, 2, 2) is just the sum.

### Step 6: Tree dispersion is identically flat

By Step 1, the (1 - u^2)^2 prefactor of det(I - u B(k)) is
k-independent. Hence the +/-1 eigenvalues of B(k) appear with
multiplicity exactly 2 at every k. The 'tree-subspace dispersion' --
i.e. the eigenvalues of B(k) restricted to V_tree(k) -- is

    lambda_tree(P + q) = +/-1     for all q in BZ_primitive

Numerical verification: at five random k in [-1/2, 1/2]^3
(seeded numpy random state), the closest-to-+/-1 eigenvalues of B(k)
satisfy |mu - (+/-1)| < 1.3e-15 (machine precision).

This is a third dispersion type beyond the two tested in
an internal working note not Dirac-cone (linear in
|q|, like the Ramanujan +sqrt(3) eigenvalue at first-order degenerate
Rayleigh-Schrodinger) and not Compton-quadratic (like the Gamma-point
Perron eigenvalue with gamma_phys = 1/16). Tree subspace is FLAT to all
orders, by the Ihara-Bass identity itself, as a structural identity of
the secular polynomial.

## Result

The Ihara-Bass tree subspace V_tree at the P-point of B(k) on srs has
the closed-form C_3 isotypic decomposition

    (mult_trivial, mult_omega, mult_omega_bar) = (0, 2, 2)

with identically flat dispersion lambda_tree(P + q) = +/-1 for all q.

## Comparison with experiment

This is a structural lemma about a Bloch operator, not a phenomenological
parameter. It supplies an upstream constraint for any reading of B(P) as
a mass-source operator (see Open questions). Its 'observed value' is the
sympy + numerical confirmation of the algebraic claim.

## Open questions

1. **Q closure on tree multiplicities.** The construction-plan goal
   (recovering Q = 2/3 by P2 sqrt-coherent aggregation on tree
   multiplicities, in the wake of the photon-like reading of the h-mode
   per an internal working note) is NOT achieved by
   the (0, 2, 2) decomposition. With mu_trivial = 0 and mu_omega =
   mu_omega_bar = 2, the P2 formula gives sqrt(m_j) = 2 sqrt(2)
   cos(2 pi j / 3) (signs (+, -, -) for j = 0, 1, 2), with positive
   roots m_j = (8, 2, 2) yielding Q = 12 / (4 sqrt(2))^2 = 12/32 = 3/8.
   Or the signed sum sum sqrt(m_j) = 0, giving Q undefined. Neither
   matches Q = 2/3. See an internal working note for
   the full negative-result analysis.

2. **No-trivial-component obstruction.** The user's hypothesised
   mult_trivial != 0 needed for Q = 2/3 (via the (2, 1, 1) ratio
   variant) is NOT realised by the tree subspace at P. The trivial C_3
   component of the 12-dim directed-edge space at P resides entirely
   inside the Ramanujan subspace (consistent with
   ../predictions/B_P_doubly_degenerate_h_derivation.md Step 3, which decomposes the
   +/-sqrt(3) A-eigenspaces as (omega + trivial) and (omega-bar +
   trivial)).

3. **Other Bloch points.** The tree subspace exists at every k by
   Ihara-Bass, but its C_3 isotypic content at non-P points may differ
   (or C_3 may not stabilise the point). High-symmetry points with C_3
   stabilisers in bcc are Gamma and P; Gamma's tree-subspace
   decomposition is a separate calculation.

4. **What does 'flat tree dispersion' mean physically?** A flat band
   has infinite effective mass in the standard solid-state reading, or
   equivalently zero group velocity. Whether this is the framework's
   correct structural statement of fermion 'rest mass scale' (in which
   case the lemma Q = 2/3 would have to come from a different
   structural identity) is open. The Lindblad readout
   (predictions/lindblad_isotypic_at_P.py giving Q = 1/2), the H_eff
   route (an internal working note, also Q = 1/2), the parameter-
   bridge route (an internal working note),
   the mass-as-confinement route on h-eigenspace
   (an internal working note, Q = 1/3), and the
   present mass-on-tree-subspace route are five sibling stalls, all
   inheriting the same B7.3 Need A obstruction (no derived C_3 action
   on the abstract C^3_gen generation space).

## References

### Cited mathematical theorems

- **Ihara, Y.** (1966). "On discrete subgroups of the two by two
  projective linear group over p-adic fields." J. Math. Soc. Japan
  18, 219-235. (Ihara identity for regular graphs.)
- **Bass, H.** (1992). "The Ihara-Selberg zeta function of a tree
  lattice." Internat. J. Math. 3, 717-797. (Bass version of the
  Ihara-Bass identity in the matrix form used here.)
- **Terras, A.** (2011). *Zeta Functions of Graphs: A Stroll through
  the Garden.* Cambridge Studies in Advanced Math 128. Theorem 3.1
  (Ihara-Bass identity in the form
  det(I - u B) = (1 - u^2)^(|E|-|V|) det((1 + (k-1) u^2) I - u A);
  derives the k-independent tree prefactor used here).
- **Serre, J.-P.** (1977). *Linear Representations of Finite Groups.*
  Springer GTM 42. Chapter 2 (character orthogonality on a finite
  group, applied here to C_3).

### Upstream framework lemmas

- predictions/k_star.py -- k* = 3 (MDL).
- predictions/d_spatial.py -- d = 3 (MDL).
- predictions/g_girth_derivation.md sec 2 -- srs = I4_132 Wyckoff 8a
  (forced by k* = 3, d = 3).
- ../predictions/walker_dynamics_derivation.md W1-W3 -- B(k) is the Hashimoto
  Bloch walker on srs.
- predictions/B_P_doubly_degenerate_h.py -- B(P) Ramanujan-sector
  spectrum: h, h*, -h, -h* each multiplicity 2; |h|^2 = k* - 1 = 2.
  Same bond list used here.
- ../predictions/B_P_doubly_degenerate_h_derivation.md Step 3 -- C_3 isotypic
  decomposition of the +/-sqrt(3) A-eigenspaces (each = trivial +
  omega), used here as a cross-check for the trivial-component
  inventory.
  analysis on the h-eigenmode (concludes h is photon-like, fermion
  masses must come from elsewhere); the trigger for the present
  tree-subspace push.
  carrying through the Q = 2/3 closure attempt on tree multiplicities;
  reports it does NOT close.

## Files referenced but NOT modified

Per task constraints: results/parameters.csv, docs/parameters/derivations.md,
B3/B5/B6 docs, ../predictions/walker_dynamics_derivation.md,
../predictions/B_P_doubly_degenerate_h_derivation.md, and existing predictions/ files
are NOT edited. Only this new lemma + the companion stall doc are
produced.
