# Derivation: C_3-isotypic Lindblad on the visible Bloch fibre at P

**NOTE (post-A3, 2026-04-18):** Under the three-axiom framework (A1+A2+A3; docs/framework/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 Theorem 25 (predictions/observer_hilbert_space.py). The Lindblad-form derivation from A1+A2+A3 (vs adoption) and P1/P2 remain separately load-bearing.

**Companion file:** `predictions/lindblad_isotypic_at_P.py`
**Status:** lemma (closed-form three mass-flux values on the h-eigenspace
at P + closed-form direct Koide ratio Q_iso = 1/2; the framework's
Q_Koide = 2/3 of `predictions/Q_Koide.py` is NOT recovered by the direct
ratio formula on the three isotypic mass-fluxes; the structural reason is
identified in §"Why Q = 2/3 does not drop out").
**Predecessor:** `predictions/lindblad_steady_state_at_P.py` (directed-edge
basis dephasing). The present construction refines the dephasing channels
from twelve directed-edge projectors to three C_3-isotypic projectors;
the three resulting mass-flux quantities are distinct and informative,
but the maximally-mixed steady state is replaced by a 12-dim degenerate
manifold of steady states (no unique fixed point).

## Abstract

The directed-edge dephasing Lindblad of `predictions/lindblad_steady_state_at_P.py`
gives a single mass-scale m_h = 2/k* = 2/3 on the 2-dim h-eigenspace at
P with the maximally-mixed steady state rho_ss = I/12. That construction
collapses the three C_3-isotypic components of the Bloch fibre into a
single trace identity. The present refinement replaces the 12 directed-edge
jump operators with three C_3-isotypic jump operators

    L_alpha = sqrt(rate_alpha) * P_alpha,   alpha in {trivial, omega, omega-bar},

where P_alpha are the rank-4 orthogonal projectors onto the C_3-isotypic
sub-bundles of the 12-dim Bloch fibre (theorem `docs/theorem_B5_3_core.md`
Steps 1-4: full-fibre C_3-character is (12, 0, 0), giving multiplicities
(m_trivial, m_omega, m_omegabar) = (4, 4, 4)). The C_3-content of the
2-dim h-eigenspace is (1, 1, 0) (one trivial + one omega; theorem_B5_3_core
Step 5).

Two rate prescriptions are tested:
- (B.i)  rate_alpha = 1/k* (uniform across alpha)
- (B.ii) rate_alpha = mult_alpha / k* with mult_alpha = (4, 4, 4)

Both give:
- three closed-form mass-flux values (m_trivial_h, m_omega_h, m_omegabar_h)
  = ((1/k*)*1, (1/k*)*1, (1/k*)*0) under B.i, and ((4/k*)*1, (4/k*)*1, 0)
  under B.ii — both up to the overall rate factor;
- direct Koide ratio Q_iso = sum(m)/(sum sqrt(m))^2 = 1/2 in both
  variants (the rate_alpha factor cancels because all three multiplicities
  on the full fibre are equal);
- a 12-dim degenerate set of steady states (NOT a unique rho_ss) because
  H, U_C3, P_triv, P_omega, P_omegabar all mutually commute.

The Koide value Q = 2/3 of `predictions/Q_Koide.py` is **NOT recovered**
by the direct ratio formula sum(m)/(sum sqrt(m))^2 on the three Lindblad
mass-flux values. The structural reason is identified in §"Why Q = 2/3
does not drop out": Q_Koide = 2/3 in the standalone color-sector lemma
of `predictions/Q_Koide_derivation.md` is computed via the P2 sqrt-coherent
aggregation postulate (sqrt(m_j) = sqrt(mu_t) + sqrt(mu_o) omega^j +
sqrt(mu_ob) omega^-j), which is a *different* functional relationship
between the multiplicities (mu_triv, mu_omega, mu_omegabar) and the three
mass eigenvalues than what the Lindblad mass-flux trace identity supplies.
The present construction does not bridge the two functional forms.

## Framework axioms invoked

- **A1 (self-inverse toggle).** Each edge e of the srs primitive cell
  carries a toggle T_e with T_e * T_e = 1. (Used via the 12-dim directed-edge
  Bloch fibre and via the W4 cancellation channel rate 1/k* per step.)
- **A2 (MDL).** The observer encodes the toggle stream by its reduced
  word (Serre 1980 Trees §I.1). (Used via `../../predictions/walker_dynamics_derivation.md`
  Step 4 to derive the per-step W4 cancellation rate 1/k*.)

No further axioms; no physical observation enters.

## Upstream theorems (citable as closed)

- **U1.** `../../predictions/walker_dynamics_derivation.md` (W1-W4) — observer's MDL-compressed
  dynamics on srs are non-backtracking walks; B is the Hashimoto operator
  on the 12-dim directed-edge state space; W4 cancellation rate per step
  is 1/k*. Reading-Conventions section explicitly identifies (3) Open System
  / Lindblad as the framework's most accurate reading.
- **U2.** `../../predictions/B_P_doubly_degenerate_h_derivation.md` — at k = P, the Bloch
  Hashimoto B(P) has h = (sqrt(3) + i sqrt(5))/2 with multiplicity exactly
  2, C_3-protected. Step 3 (corrected): the h-eigenspace decomposes under
  C_3 as 1 trivial + 1 omega; -h-eigenspace decomposes as 1 trivial +
  1 omega-bar.
- **U3.** `docs/theorem_B5_3_core.md` — C_3-equivariant decomposition of
  the 12-dim Bloch fibre. Step 2: full-fibre C_3-character is (12, 0, 0),
  Schur orthogonality gives multiplicities (4, 4, 4). Step 5: at k = P,
  Ramanujan subspace (8-dim) decomposes as (4, 2, 2) and tree subspace
  (4-dim) as (0, 2, 2).
- **U4.** `predictions/lindblad_steady_state_at_P.py` — directed-edge
  basis dephasing Lindblad: rho_ss = I/12; m_h = 2/k* = 2/3.

## Cited mathematical theorems

- **Lindblad, G.** (1976). On the generators of quantum dynamical
  semigroups. *Communications in Mathematical Physics* **48**, 119-130.
  Defines the Lindblad master equation and its CPTP semigroup generators.
- **Gorini, V., Kossakowski, A., Sudarshan, E.C.G.** (1976). Completely
  positive dynamical semigroups of N-level systems. *J. Math. Phys.*
  **17**, 821-825.
- **Wolf, M.M.** (2012). *Quantum Channels and Operations: Guided Tour.*
  Theorem 6.1 (unital channels admit maximally mixed fixed point);
  §"Fixed-point sets of unital channels" (steady-state degeneracy
  classification under commuting Kraus operators).
- **Breuer, H.-P. & Petruccione, F.** (2002). *The Theory of Open
  Quantum Systems.* Oxford University Press. Ch. 3 §3.2.4 (steady states
  of unital Lindbladians); §3.4 (Hermitian symmetrisation companion).
- **Serre, J.-P.** (1977). *Linear Representations of Finite Groups.*
  §2.6 (Schur orthogonality, isotypic projectors). The projector formula
  P_alpha = (1/|G|) sum_g chi_alpha(g)^* g for cyclic G.

## Derivation

### Step 1. Visible Bloch fibre and C_3 action at P

By U1, the visible sector at P is the 12-dim Bloch fibre B(P) (the
fibre dim is 2|E| = 12 for srs primitive cell). By U2, the Hashimoto Bloch
operator B(P) has spectrum {h, h*, -h, -h*, +1, -1} with each eigenvalue
having multiplicity exactly 2. The body-diagonal 3-fold rotation C_3 fixes
the P point in reduced coordinates; its action on the directed-edge basis
is the k-independent 12 x 12 permutation matrix U_C3 with U_C3^3 = I and
[B(P), U_C3] = 0 (U3 Steps 1, 3).

### Step 2. C_3-isotypic projectors

By Schur orthogonality (Serre 1977 §2.6) for cyclic C_3 with characters
chi_trivial(c^k) = 1, chi_omega(c^k) = omega^k, chi_omegabar(c^k) = omega^{-k}
(omega = exp(2 pi i / 3)):

    P_alpha = (1/3) sum_{k=0}^{2} chi_alpha(c^k)^* U_C3^k.

Explicitly:

    P_trivial   = (I + U_C3 + U_C3^2) / 3
    P_omega     = (I + omega^* U_C3 + (omega^*)^2 U_C3^2) / 3
    P_omegabar  = (I + omega U_C3 + omega^2 U_C3^2) / 3

These are mutually orthogonal Hermitian idempotents summing to I_12. The
ranks (= traces) are (4, 4, 4) by U3 Step 2 (full-fibre C_3-character
(12, 0, 0)). All three commute with B(P) and with H = (B(P) + B(P)^dag)/2.

### Step 3. Hamiltonian

Same as in U4: H = (B(P) + B(P)^dag) / 2. Hermitian by construction;
||H - H^dag|| = 0 to machine precision. Because [B(P), U_C3] = 0 implies
[B(P)^dag, U_C3] = 0 (taking dagger of both sides; U_C3 is unitary as a
permutation), we have [H, U_C3] = 0 and hence [H, P_alpha] = 0 for each
alpha.

### Step 4. Isotypic jump operators

Variant B.i (uniform rate):

    L_alpha = sqrt(1/k*) P_alpha,   alpha in {trivial, omega, omega-bar}.

The rate 1/k* is the W4 cancellation rate per step (U1 Step 4) carried
over from U4. Distributing it uniformly across the three isotypic channels
preserves the total dissipation rate of U4: sum_alpha L_alpha^dag L_alpha
= (1/k*) sum_alpha P_alpha = (1/k*) I, identical to the unitality content
of U4. Variant B.i is the parsimonious choice: a single rate constant
1/k*, no additional structure.

Variant B.ii (multiplicity-weighted rate):

    L_alpha = sqrt(mult_alpha / k*) P_alpha,   mult_alpha = (4, 4, 4).

This folds the full-fibre C_3 multiplicity into the rate. Since mult_alpha
is the same across alpha (= 4 each), Variant B.ii rescales each L_alpha
uniformly by a factor of 2 relative to B.i. The total dissipation rate
becomes sum_alpha L_alpha^dag L_alpha = (4/k*) I (a factor of 4 larger).
This variant is the one connected to Q_Koide arithmetic where mu_alpha
enters as a weight; it is presented for completeness but does not change
the direct Koide ratio (Step 7).

### Step 5. Lindblad equation

With H from Step 3 and L_alpha from Step 4, the Lindblad equation is

    L(rho) = -i [H, rho]
             + sum_alpha (L_alpha rho L_alpha^dag - 0.5 {L_alpha^dag L_alpha, rho}).

Generates a CPTP semigroup on 12 x 12 density matrices (Lindblad 1976;
Gorini-Kossakowski-Sudarshan 1976).

### Step 6. Steady-state structure

Since H, L_trivial, L_omega, L_omegabar all mutually commute (they are
all functions of {I, U_C3}), the Lindblad superoperator L has a large
kernel: any density matrix block-diagonal in the C_3-isotypic decomposition
is invariant under L. Numerically the vectorised L (144 x 144) has
12 zero singular values and the maximally-mixed state I/12 is one of
infinitely many steady states.

This is a sharp departure from U4. There the dissipator was unital and
non-degenerate (the directed-edge projectors P_e do NOT all commute with
H, so the unitary part -i[H, .] picks out the unique I/12 steady state
on top of the unital dissipator). Here all jump operators commute with H,
so the unitary part vanishes on the entire isotypic block-diagonal
subspace, and so does the dissipator.

The lack of a unique steady state means the construction does NOT supply
the population-on-h identity Tr(P_h rho_ss) = 1/6 of U4. What it DOES
supply is the per-isotypic mass-flux trace identity, which depends only
on the structure of the jump operators and not on the steady state.

### Step 7. Mass-flux trace identity per isotypic channel

For each alpha, define the mass-flux on the h-eigenspace as

    m_alpha_h = sum_channel Tr(L_channel^dag L_channel * P_alpha P_h).

Using L_alpha = sqrt(rate_alpha) P_alpha so L_alpha^dag L_alpha = rate_alpha
P_alpha (idempotent), and orthogonality P_alpha P_beta = delta_{ab} P_alpha:

    m_alpha_h = sum_beta Tr(rate_beta P_beta * P_alpha P_h)
              = rate_alpha Tr(P_alpha P_h)            (only beta = alpha contributes)

The intersection-traces Tr(P_alpha P_h) are integers because P_alpha and
P_h commute (both commute with U_C3) and so the h-eigenspace decomposes
orthogonally into its C_3-isotypic components. By U3 Step 5:

    Tr(P_trivial P_h)  = 1
    Tr(P_omega P_h)    = 1
    Tr(P_omegabar P_h) = 0    (sum = 2 = Tr P_h, as required).

Hence:

  Variant B.i (rate_alpha = 1/k*):
    m_trivial_h  = 1/k* = 1/3
    m_omega_h    = 1/k* = 1/3
    m_omegabar_h = 0

  Variant B.ii (rate_alpha = mult_alpha/k* = 4/k*):
    m_trivial_h  = 4/k* = 4/3
    m_omega_h    = 4/k* = 4/3
    m_omegabar_h = 0

Both variants verified to machine precision in the script (variant B.i
diagonal entry 0.333333... and 0; variant B.ii diagonal entry 1.333333...
and 0).

### Step 8. Direct Koide ratio Q_iso

Apply the user-requested formula

    Q_iso = (m_trivial_h + m_omega_h + m_omegabar_h)
          / (sqrt(m_trivial_h) + sqrt(m_omega_h) + sqrt(m_omegabar_h))^2.

Variant B.i:

    sum m   = 1/3 + 1/3 + 0    = 2/3
    sum sqrt m = sqrt(1/3) + sqrt(1/3) + 0 = 2/sqrt(3)
    Q_iso = (2/3) / (4/3) = 1/2.

Variant B.ii:

    sum m   = 4/3 + 4/3 + 0 = 8/3
    sum sqrt m = sqrt(4/3) + sqrt(4/3) + 0 = 4/sqrt(3)
    Q_iso = (8/3) / (16/3) = 1/2.

The rate_alpha factor cancels because all three full-fibre multiplicities
are equal (4, 4, 4) and so rate_alpha is constant across alpha, leaving

    Q_iso = (Tr(P_t P_h) + Tr(P_o P_h) + Tr(P_ob P_h))
          / (sqrt(Tr(P_t P_h)) + sqrt(Tr(P_o P_h)) + sqrt(Tr(P_ob P_h)))^2
          = (1 + 1 + 0) / (1 + 1 + 0)^2 = 2/4 = 1/2.

So Q_iso = 1/2 is determined purely by the integer C_3-content (1, 1, 0)
of the h-eigenspace.

### Numerical verification

`predictions/lindblad_isotypic_at_P.py` constructs the 12 x 12 Bloch
matrix B(P) using the same I4_132 Wyckoff 8a primitive-cell bond list as
`predictions/B_P_doubly_degenerate_h.py`, the U_C3 permutation from
`proofs/foundations/theorem_B5_3_core.py`, the three rank-4 isotypic
projectors, and the 144 x 144 vectorised Lindblad superoperator. It
verifies:

- ranks(P_trivial, P_omega, P_omegabar) = (4, 4, 4) exactly;
- mutual orthogonality and idempotency of the projectors to ~1e-15;
- ||[H, P_alpha]|| = 0 (mutual commutation) to ~1e-15;
- (Tr(P_trivial P_h), Tr(P_omega P_h), Tr(P_omegabar P_h)) = (1, 1, 0) to ~1e-10;
- Variant B.i: (m_trivial_h, m_omega_h, m_omegabar_h) = (1/3, 1/3, 0) to ~1e-10;
- Variant B.ii: (m_trivial_h, m_omega_h, m_omegabar_h) = (4/3, 4/3, 0) to ~1e-10;
- Q_iso = 1/2 to ~1e-7 (the residual is the float floor on sqrt(0));
- Lindblad superoperator kernel dim = 12 (steady-state degeneracy).

## Result

Closed form on the visible Bloch fibre at P, under variant B.i (uniform
rate 1/k*):

    (m_trivial_h, m_omega_h, m_omegabar_h) = (1/3, 1/3, 0)        [exact rationals]
    Q_iso = (m_t + m_o + m_ob) / (sqrt(m_t) + sqrt(m_o) + sqrt(m_ob))^2
          = 1/2                                                    [exact rational]

Under variant B.ii (multiplicity-weighted rate, mult_alpha = (4, 4, 4)):

    (m_trivial_h, m_omega_h, m_omegabar_h) = (4/3, 4/3, 0)
    Q_iso = 1/2                                                    [unchanged].

The C_3-isotypic Lindblad gives three distinct mass-flux quantities,
matching the three C_3 irreps. **Q_iso = 1/2 != 2/3 = Q_Koide.** The
recovery of Q_Koide = 2/3 from the Lindblad mass-flux readout requires
additional structure not supplied by the direct trace identity; see the
focused scoping in §"Why Q = 2/3 does not drop out".

## Comparison with experiment

Not applicable. m_alpha_h are dimensionless framework-internal rates per
isotypic channel; converting to particle masses in MeV requires the
Bloch-to-physical-units map (Need B of an internal working note),
which is open. No quantitative observed-mass prediction is made here. The
Q_iso = 1/2 result is an internal mathematical statement about the
Lindblad mass-flux trace identities, not a comparison to the Koide ratio.

## Why Q = 2/3 does not drop out

The Q_Koide = 2/3 of `predictions/Q_Koide.py` is the "compact form"

    Q_Koide = (mu_triv + mu_omega + mu_omegabar) / (k* * mu_triv)

evaluated at the Ramanujan multiplicities (mu_triv, mu_omega, mu_omegabar)
= (4, 2, 2):

    Q_Koide = (4 + 2 + 2) / (3 * 4) = 8/12 = 2/3.

This formula does NOT come from applying

    Q = sum(m_j) / (sum sqrt(m_j))^2

directly to (mu_triv, mu_omega, mu_omegabar) = (4, 2, 2). Direct
evaluation gives

    Q_direct = 8 / (sqrt(4) + sqrt(2) + sqrt(2))^2 = 8 / (2 + 2 sqrt(2))^2
             = 8 / (12 + 8 sqrt(2))
             = 2 / (3 + 2 sqrt(2))
             ~ 0.343,

which is neither 2/3 nor 1/2. The Q_Koide.py compact form arises via
**postulate P2** (`docs/framework/W4_identification_catalog.md` §3) which prescribes
a *sqrt-coherent aggregation* over C_3 irreps to produce the three mass
eigenvalues:

    sqrt(m_j) := sqrt(mu_triv) + sqrt(mu_omega) omega^j + sqrt(mu_omegabar) omega^{-j},
    j = 0, 1, 2.

After that aggregation (with mu_omega = mu_omegabar so that the m_j are
real), C_3 orthogonality identities

    sum_j cos(2 pi j / 3) = 0,         sum_j cos^2(2 pi j / 3) = 3/2

give

    sum_j sqrt(m_j) = k* sqrt(mu_triv),
    sum_j m_j       = k* (mu_triv + mu_omega + mu_omegabar),

and hence Q_Koide = (mu_triv + mu_omega + mu_omegabar) / (k* * mu_triv).
At srs P with (mu_triv, mu_omega, mu_omegabar) = (4, 2, 2) on Ramanujan,
this gives 8/12 = 2/3.

The Lindblad mass-flux trace identity supplies (m_alpha_h) — the
mass-flux *per isotypic channel*, not the mass *per generation index*.
The two are related by completely different functional forms:

  Lindblad mass-flux:      m_alpha_h = rate_alpha * Tr(P_alpha P_h),
                           one value per alpha.
  P2 sqrt-coherent agg.:   sqrt(m_j) = sqrt(mu_triv) + sqrt(mu_omega) omega^j
                                                     + sqrt(mu_omegabar) omega^{-j},
                           one value per generation index j (of which there are k* = 3).

These two functions of the multiplicities (mu_triv, mu_omega, mu_omegabar)
in general give different Koide ratios. The Lindblad direct ratio is
sum(m)/(sum sqrt(m))^2 evaluated at (m_alpha) = (rate_alpha * mult_alpha).
The P2 aggregation Koide is the coherent-sum identity above, equivalent
to (mu_triv + mu_omega + mu_omegabar)/(k* mu_triv).

The framework's Q_Koide = 2/3 is the latter. To recover it from a
Lindblad-style readout, the construction would need to:

(i) Replace the diagonal trace identity m_alpha_h = rate_alpha *
    Tr(P_alpha P_h) by an aggregation that mixes the three isotypic
    channels coherently (off-diagonal in alpha, with sqrt-multiplicity
    coefficients); OR
(ii) Replace the m_alpha_h readout by k* = 3 distinct mass eigenvalues
    indexed by a separate "generation" label j, each obtained from a
    coherent C_3-Fourier sum over alpha.

Neither (i) nor (ii) is what the canonical Lindblad mass-flux on a
single Bloch fibre at P naturally supplies. The Lindblad readout is
intrinsically per-channel (per-alpha), not per-coherent-superposition
(per-j). Bridging the two requires additional axiomatic input — the P2
postulate is exactly that input — and the present Lindblad construction
does not derive it.

Under the reconciliation analysis of `../../docs/framework/B3_B6_reconciliation.md`,
the physical interpretation of the (4, 2, 2) Ramanujan multiplicities is
itself open: option (alpha) "C_3 = generation" (the pre-B6 reading), option
(beta) "pure algebraic SU(4) Cartan label" (the most honest fallback),
option (gamma) "non-PS SU(3) action". The Lindblad construction is
agnostic about that interpretation; it only computes the per-channel
trace identity and its direct Koide ratio.

## Open questions

1. **Bridge from Lindblad mass-flux to P2 aggregation.** The direct
   Lindblad readout m_alpha_h = rate_alpha * Tr(P_alpha P_h) is per-isotypic-channel;
   the Q_Koide = 2/3 derivation uses the P2 sqrt-coherent aggregation
   sqrt(m_j) = sqrt(mu_t) + sqrt(mu_o) omega^j + sqrt(mu_ob) omega^-j to
   produce per-generation masses m_j. No derivation is currently in repo
   that justifies the P2 form from MDL + toggle through a Lindblad-style
   construction. The framework's W4_identification_catalog §3 catalogs
   P2 as an "adopted postulate beyond the two foundational axioms."

2. **Choice of jump-operator basis.** The directed-edge basis dephasing
   of `predictions/lindblad_steady_state_at_P.py` gives a unique
   maximally-mixed steady state and a single mass scale m_h = 2/3. The
   C_3-isotypic basis dephasing of the present construction gives three
   mass-fluxes but loses uniqueness of the steady state. Neither basis
   is forced by MDL+toggle alone; both are consistent with the W4
   cancellation rate 1/k* per step. The framework currently has no
   theorem fixing which basis (or which combination) is the "canonical"
   open-system reading.

3. **What happens on the full Ramanujan subspace?** The h-eigenspace
   decomposes under C_3 as (1, 1, 0). The full Ramanujan subspace
   (8-dim, sum of {h, h*, -h, -h*} eigenspaces) decomposes as (4, 2, 2).
   Repeating Step 7 on the full Ramanujan projector P_Ram in place of
   P_h gives (m_triv_R, m_omega_R, m_omegabar_R) = (4/k*, 2/k*, 2/k*)
   under variant B.i, and the direct Koide ratio Q_iso_Ram = 8 /
   (sqrt(4) + sqrt(2) + sqrt(2))^2 = 8 / (2 + 2 sqrt(2))^2 ~ 0.343 — a
   third value, neither 1/2 nor 2/3. This confirms that the direct
   Koide-ratio formula is *not* invariant under choice of subspace
   (h vs Ramanujan) on which the trace identity is applied; the readout
   depends on the integer multiplicities of that specific subspace under C_3.

4. **Steady-state degeneracy as an open structural item.** The 12-dim
   kernel of L_super means 11 traceless steady states; under the
   detailed-balance reading of an internal working note,
   each could carry its own bidirectional flux. No physical principle
   currently selects one over another; the maximally-mixed state I/12
   is a default but not forced.

5. **Connection to predictions/lindblad_steady_state_at_P.py.** The
   directed-edge basis Lindblad has 12 jump operators (one per directed
   edge); the isotypic Lindblad has 3 (one per C_3 irrep). Both have
   the same total dissipation rate 12 * (1/k*) (B.i) = 12/k* in the
   directed-edge case and 3 * (1/k*) = 3/k* in variant B.i isotypic case
   — a factor-of-4 difference. The factor of 4 is the dim of each isotypic
   block; in B.ii it is restored explicitly by mult_alpha = 4. So B.ii
   is the dimension-faithful refinement of U4: same total dissipation
   rate, three rank-4 channels instead of twelve rank-1 channels. The
   trace identity on h gives the same total m_h:
   m_t_h + m_o_h + m_ob_h = (4 + 4 + 0)/k* = 8/3 under B.ii vs
   12 * (1/k*) * 2/12 = 2/3 under U4 — a factor-of-4 difference,
   not the same identity. The two readings answer different questions
   (per-channel vs per-channel-summed-with-rate-per-edge), and neither
   alone determines the framework's Q_Koide.

## References

### Memory / standards

  unverified identifications.
- `docs/parameters/parameter_linter.md` — hard quality gate.

### Cited mathematical theorems

- **Lindblad, G.** (1976). On the generators of quantum dynamical
  semigroups. *Communications in Mathematical Physics* **48**, 119-130.
- **Gorini, V., Kossakowski, A., Sudarshan, E.C.G.** (1976). Completely
  positive dynamical semigroups of N-level systems. *J. Math. Phys.*
  **17**, 821-825.
- **Wolf, M.M.** (2012). *Quantum Channels and Operations.* §"Fixed
  points of unital channels" + Theorem 6.1.
- **Breuer, H.-P. & Petruccione, F.** (2002). *The Theory of Open
  Quantum Systems.* Ch. 3 (Lindblad master equations, dephasing models,
  steady-state classification).
- **Serre, J.-P.** (1977). *Linear Representations of Finite Groups.*
  §2.6 (Schur orthogonality).
- **Sunada, T.** (2012). *Topological Crystallography.* §§5-6.

### Upstream framework theorems (closed)

- `../../predictions/walker_dynamics_derivation.md` — W1-W4: NB walker + Hashimoto + Bloch decomposition + Reading Conventions.
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` — h with multiplicity 2; Step 3 corrected isotypic decomposition.
- `docs/theorem_B5_3_core.md` — full-fibre C_3 multiplicities (4, 4, 4); Ramanujan (4, 2, 2); h-eigenspace (1, 1, 0).
- `predictions/B_P_doubly_degenerate_h.py` — 12-dim B(P) construction.
- `predictions/lindblad_steady_state_at_P.py` — directed-edge basis Lindblad (predecessor; m_h = 2/3, rho_ss = I/12).
- `predictions/k_star.py` — k* = 3.
- `predictions/d_spatial.py` — d_spatial = 3.

### Sibling / interpretation documents

- `predictions/Q_Koide.py` + `predictions/Q_Koide_derivation.md` — Q_Koide = 2/3 via P2 sqrt-coherent aggregation; status BLOCKED under B6 retraction.
- `../../docs/framework/B3_B6_reconciliation.md` — physical interpretation of (4, 2, 2) is open; default reading (β) "pure algebraic Cartan label."
- `docs/framework/W4_identification_catalog.md` §3 — P1 (Ramanujan selection), P2 (sqrt-coherent aggregation): adopted structural postulates, beyond MDL + toggle.
- an internal Markov-vs-unitary classification audit — Lindblad / Open System reading recommendation.

## Files referenced but NOT modified

Per task constraints: `results/parameters.csv`, `docs/parameters/derivations.md`,
B3/B5/B6 docs, `../../predictions/walker_dynamics_derivation.md`, and existing
`predictions/*.py` files are NOT edited. Only the two
`predictions/lindblad_isotypic_at_P*` files are produced (this file +
the accompanying `.py`). No edits to
`predictions/lindblad_steady_state_at_P.py` (the predecessor).

No commits performed; no remote push.
