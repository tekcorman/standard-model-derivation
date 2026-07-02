# Derivation: Yukawa-selective Lindblad on H_visible (x) S = 96-dim

**NOTE (post-A3, 2026-04-18):** Under the three-axiom framework (A1+A2+A3; docs/framework/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 Theorem 25 (predictions/observer_hilbert_space.py). The Lindblad-form derivation from A1+A2+A3 (vs adoption), Pati-Salam labeling, and P1/P2 remain separately load-bearing.

**Companion file:** `predictions/lindblad_yukawa_selective.py`
**Construction file:** `proofs/foundations/lindblad_yukawa_selective_construction.py`
**Status:** PARTIAL closure, CONDITIONAL on Conjecture C1 (Yukawa-like jump
operators NOT derived from MDL+toggle). The construction PRESERVES C_3
isotypic coherence on the visible side as designed (verified: max
||[L, U_C3 (x) I]|| = 0 for both jump-operator families) but does NOT
collapse the steady-state set to dimension 1 -- the dissipator's
C_3-symmetry yields kernel dim 96 by the conservation law of
an internal working note. The mass-flux readout
factorizes (species cancels in Koide ratio) because the rate operator
is unital and the readout projector is a Kronecker product. Result:
Q_charged-lepton = 1/2, NOT the framework-targeted 2/3.

**Predecessors:**
- `predictions/lindblad_steady_state_at_P.py` -- directed-edge dephasing
  Lindblad. Unique steady state I/12; one mass scale 2/k* = 2/3.
- `predictions/lindblad_isotypic_at_P.py` -- C_3-isotypic Lindblad.
  Three mass-flux values (1/3, 1/3, 0) per C_3 sector; kernel dim 12;
  Q_iso = 1/2.
- `predictions/lindblad_spinor_coupled.py` -- C_3-isotypic + visible
  edge x B-L species (C_3-BREAKING family). Kernel dim 32; Q_s = 1/2.

**Companion scoping:**
  conservation law: any C_3-symmetric jump operator gives a degenerate
  steady-state set by block-diagonalisation. The present construction
  satisfies the C_3-symmetry hypothesis of that conservation law and
  therefore inherits the steady-state degeneracy as predicted.

## Abstract

The framework's prior Lindblad pushes (directed-edge dephasing
`predictions/lindblad_steady_state_at_P.py`, C_3-isotypic dephasing
`predictions/lindblad_isotypic_at_P.py`, spinor-coupled
`predictions/lindblad_spinor_coupled.py`) all gave Q_charged-lepton =
1/2 under the canonical mass-flux readout. The structural pinpoint
: every constructed
dissipator in those pushes had a unital rate operator
sum_jump L^dag L proportional to the identity, which forces the
trace identity m_{s, alpha} = const * Tr(P_alpha P_h) * Tr(Pi_s) to
factorize species out of the Koide ratio, leaving Q = 1/2 determined
by the integer (1, 1, 0) C_3-content of the h-eigenspace alone.

The present construction tests whether a SELECTIVE dissipator with
both properties --

  (i)  C_3 isotypic coherence PRESERVED on the visible side, and
  (ii) species DECOHERENCE on the spinor side via Yukawa-like jumps
       L_{Y, s} = sqrt(gamma_2) I_visible (x) X_s with X_s the
       L<->R chirality swap on each species projector Pi_s --

can break the species cancellation and recover Q_Koide = 2/3.

The construction is well-defined and PRESERVES C_3 as designed (both
jump-operator families commute with U_{C_3} (x) I_S exactly). The
algebraic verification yields:

1. The maximally mixed state I_96/96 is a Lindblad steady state.
2. The 4 x 3 mass-flux table m_{s, alpha} = (1/72) * Tr(P_alpha P_h)
   (independent of species s, because the rate operator R =
   (gamma_1 + gamma_2) I_96 is unital and the readout projector is the
   Kronecker product P_alpha (x) Pi_s, so Schur orthogonality gives the
   factorization).
3. Per-species Q_s = 1/2 universally; in particular Q_charged-lepton =
   1/2, NOT 2/3.
4. The vectorized Lindblad superoperator's kernel has dimension 96
   (numerically verified via SVD on the 9216 x 9216 superoperator), the
   LARGEST steady-state degeneracy of any of the four Lindblad
   constructions tested. This is consistent with the
   an internal working note conservation law:
   C_3-symmetric jumps preserve the C_3 isotypic block-diagonal
   structure of density matrices, and the Yukawa-on-spinor jumps add
   their own conserved-charge structure on the spinor side.

CONCLUSION. The selectivity in the spinor sector is structurally
INSUFFICIENT to break the species factorization in the Koide ratio,
because each Yukawa swap operator X_s satisfies X_s^2 = Pi_s exactly,
giving sum_s X_s^2 = sum_s Pi_s = I_S, hence a unital rate operator.
The bridge from Lindblad mass-flux to Q_Koide = 2/3 still requires
the P2 sqrt-coherent aggregation postulate
(`docs/framework/W4_identification_catalog.md` §3), which is NOT supplied by
any canonical Lindblad mass-flux readout, including this
Yukawa-selective one.

## Conjecture C1 (status FLAGGED, NOT DERIVED)

The Yukawa-like jump operators of Family II,

    L_{Y, s} = sqrt(gamma_2) * I_visible (x) X_s,
    s in {e, nu, u, d},  X_s = |s_R><s_L| + |s_L><s_R| within Pi_s,

are NOT derived from MDL + binary self-inverse toggle. The motivation
is the physical Standard-Model Yukawa interaction (electroweak Higgs
mechanism), reframed as a Lindblad dissipator that mixes L and R
states of one species at a time.

The framework's existing W4-cancellation source of Lindblad jumps
(`../../predictions/walker_dynamics_derivation.md` Reading Conventions section, W4
discussion) supplies VISIBLE-SIDE jump operators only -- the
cancellation channel acts on the directed-edge basis of the visible
Bloch fibre. There is currently no MDL+toggle derivation of
SPINOR-SIDE jump operators, let alone species-selective L<->R swap
ones.

Possible future routes to deriving Conjecture C1 from MDL+toggle:

1. **MDL preference for C_3-preserving compression.** The framework's
   P2 postulate effectively encodes a coherent superposition of
   (mu_triv, mu_omega, mu_omegabar) Ramanujan amplitudes that pays
   for itself per MDL (one mass value per generation from C_3
   multiplicities). Mixing across species (lepton-quark, up-down)
   produces SU(2) or PS-violating states with no observed compressed
   pattern, so MDL would discard species-mixing information --
   decohering the species basis. This is a HEURISTIC for the form of
   the framework's preferred jumps, not a derivation. Carrying it
   through would require formalizing "MDL discards species-mixing
   information" as a precise statement about Lindblad jump operators,
   which has not been done.

2. **EW symmetry breaking + Higgs condensate.** In the Standard
   Model, the Higgs VEV breaks SU(2)_L x U(1)_Y to U(1)_em and
   Yukawa interactions then mix species (e.g., u_L with u_R, e_L with
   e_R). These are species-mixing jumps that are dynamically
   generated by the Higgs condensate. C_3 isotypic structure (in the
   visible Bloch fibre) is unaffected because the Higgs is a visible-
   sector singlet (per the framework's reading of color = visible C_3
   per `docs/theorem_B6_bridge.md`). This motivation has cleaner
   physical interpretation but is itself a downstream physics input,
   not derivable from MDL+toggle alone.

The lemma below is CONDITIONAL on Conjecture C1. If C1 holds, the
Lindblad construction is well-defined and gives the result derived
below. The lemma is a CANDIDATE consequence of C1.

## Framework axioms invoked

- **A1 (self-inverse toggle).** Each edge e of the srs primitive cell
  carries a toggle T_e with T_e * T_e = 1.
- **A2 (MDL).** The observer encodes the toggle stream by its reduced
  word.
- **C1 (CONJECTURE, FLAGGED).** Yukawa-like spinor-side jump operators
  L_{Y, s} = sqrt(gamma_2) I_visible (x) X_s exist as Lindblad
  dissipator channels in the framework, with rate gamma_2 = 1/k*
  matching the W4 cancellation rate per step.

A1, A2, and A3 are framework axioms (see docs/framework/framework_axioms.md).
C1 is a structural conjecture beyond A1+A2+A3, motivated by
Standard-Model Yukawa interactions but not derived from the framework's
three axioms.

## Upstream theorems (closed)

- **U1.** `../../predictions/walker_dynamics_derivation.md` (W1-W4) -- the observer's
  MDL-compressed dynamics on srs are non-backtracking walks; B is the
  Hashimoto operator on the 12-dim directed-edge state space; W4
  cancellation rate per step is 1/k*. Reading Conventions section
  identifies (3) Open System / Lindblad as the framework's most
  accurate reading.
- **U2.** `../../predictions/B_P_doubly_degenerate_h_derivation.md` -- B(P) has
  h = (sqrt(3) + i sqrt(5))/2 with multiplicity exactly 2,
  C_3-protected. Step 3 (corrected): h-eigenspace decomposes under
  C_3 as 1 trivial + 1 omega.
- **U3.** `docs/theorem_B5_3_core.md` -- C_3-equivariant decomposition
  of the 12-dim Bloch fibre. Step 2: full-fibre C_3-character is
  (12, 0, 0), Schur orthogonality gives multiplicities (4, 4, 4).
  Step 5: at k = P, h-eigenspace decomposes as (1, 1, 0).
- **U4.** `../../predictions/theorem_B3_spinor_fermion_derivation.md` -- the 8-dim Cl(6, 0)
  Dirac spinor S decomposes under Spin(4) x Spin(2) =
  SU(2)_L x SU(2)_R x U(1)_{B-L} as one Pati-Salam family with colour
  factored out. Step 3 establishes the chirality operator
  G_7 = -i Gamma_1 ... Gamma_6 with G_7^2 = I_S, G_7 = G_7^dag.
  Step 4 establishes the species labels {nu, e, u, d} x {L, R} on the
  8 weight states; each species projector Pi_s has rank 2, one L and
  one R chirality per species.
- **U5.** `predictions/lindblad_steady_state_at_P.py` -- directed-edge
  basis dephasing Lindblad: rho_ss = I/12; m_h = 2/k* = 2/3.
- **U6.** `predictions/lindblad_isotypic_at_P.py` -- C_3-isotypic
  Lindblad: m_alpha_h = (1/k*) * Tr(P_alpha P_h); kernel dim 12;
  Q_iso = 1/2.
- **U7.** `predictions/lindblad_spinor_coupled.py` -- spinor-coupled
  Lindblad with C_3-BREAKING family II (visible edge x B-L species);
  kernel dim 32; Q_s = 1/2.
- **U8.** an internal working note -- conservation
  law: any C_3-symmetric jump operator preserves the C_3 isotypic
  block-diagonal structure of density matrices, giving a degenerate
  steady-state set with one conserved population per block.

## Cited mathematical theorems

- **Lindblad, G.** (1976). On the generators of quantum dynamical
  semigroups. *Communications in Mathematical Physics* **48**, 119-130.
- **Gorini, V., Kossakowski, A., Sudarshan, E.C.G.** (1976). Completely
  positive dynamical semigroups of N-level systems. *J. Math. Phys.*
  **17**, 821-825.
- **Wolf, M.M.** (2012). *Quantum Channels and Operations: Guided Tour.*
  Theorem 6.1 (unital channels admit maximally mixed fixed point);
  §6 fixed-point set characterisation in terms of the noise commutant.
- **Breuer, H.-P. & Petruccione, F.** (2002). *The Theory of Open
  Quantum Systems.* Oxford University Press. Ch. 3 (master equations,
  unitality, steady-state classification).
- **Pati, J.C. & Salam, A.** (1974). Lepton number as the fourth
  colour. *Phys. Rev. D* **10**, 275-289 -- B-L = 4-colour
  decomposition.
- **Lawson, H.B. & Michelsohn, M.-L.** (1989). *Spin Geometry.*
  Princeton Math. Series 38. Ch. I, Theorem 5.7 (complex Cl(2k) ~
  M_{2^k}(C)).
- **Serre, J.-P.** (1977). *Linear Representations of Finite Groups.*
  §2.6 Schur orthogonality.

## Derivation

### Step 1 -- Total Hilbert space

By U1 the visible sector at P is the 12-dim Bloch fibre H_visible. By
U4 the spinor sector is the 8-dim Cl(6, 0) Dirac spinor S. The total
Hilbert space is the tensor product

    H_total := H_visible (x) S,        dim H_total = 12 * 8 = 96.

### Step 2 -- Hamiltonian

H_visible = (B(P) + B(P)^dag)/2 (Hermitian symmetrisation; same
choice as U5). H_spinor = 0 (no internal spinor dynamics). The total
Hamiltonian is

    H_full := H_visible (x) I_S + I_visible (x) H_spinor
            = H_visible (x) I_S    (since H_spinor = 0).

H_full is Hermitian by Hermiticity of H_visible.

### Step 3 -- Jump operators (Family I + Family II, both C_3-symmetric)

**Family I** (visible C_3-isotypic dephasing, on the spinor identity):

    L_{alpha, vis} := sqrt(gamma_1) * P_{alpha, vis} (x) I_S,
    alpha in {trivial, omega, omegabar}.

P_{alpha, vis} is the rank-4 C_3-isotypic projector on H_visible per
the Schur orthogonality formula
P_alpha = (1/3) sum_{j=0,1,2} chi_alpha(c^j)^* U_{C_3}^j
(with chi_trivial = (1, 1, 1), chi_omega = (1, omega, omega^2),
chi_omegabar = (1, omega^2, omega)). Family I is the same set of
jumps as U6 (`predictions/lindblad_isotypic_at_P.py`) lifted into the
96-dim Hilbert space.

**Family II** (Yukawa-like L<->R chirality swap, on the visible
identity):

    L_{Y, s} := sqrt(gamma_2) * I_visible (x) X_s,
    s in {e, nu, u, d}.

X_s is the L<->R chirality swap operator on Pi_s. Construction:

- By U4 Step 4, each species projector Pi_s (rank 2) contains
  exactly one G_7-eigenvector of eigenvalue +1 (the L state |s_L>)
  and one G_7-eigenvector of eigenvalue -1 (the R state |s_R>).
  This is verified in the construction-proof file.
- Define X_s := |s_R><s_L| + |s_L><s_R|.
- X_s is Hermitian: X_s^dag = (|s_R><s_L|)^dag + (|s_L><s_R|)^dag
  = |s_L><s_R| + |s_R><s_L| = X_s.
- X_s satisfies X_s^2 = Pi_s: direct calculation
  X_s^2 = (|s_R><s_L| + |s_L><s_R|)^2
        = |s_R><s_L|s_R><s_L| + |s_R><s_L|s_L><s_R|
          + |s_L><s_R|s_R><s_L| + |s_L><s_R|s_L><s_R|
        = 0 + |s_R><s_R| + |s_L><s_L| + 0
        = Pi_s
  (where we used <s_L|s_R> = 0 and <s_L|s_L> = <s_R|s_R> = 1).
- X_s is supported on Pi_s: Pi_s X_s Pi_s = X_s (verified
  numerically; follows from |s_L>, |s_R> in range(Pi_s)).

CONJECTURE C1 STATUS. The Yukawa-like operators L_{Y, s} are
postulated rather than derived; see the §Conjecture C1 section.

The Lindblad equation is

    dL/dt rho = -i [H_full, rho]
              + sum_{alpha} (L_{alpha, vis} rho L_{alpha, vis}^dag
                             - 0.5 {L_{alpha, vis}^dag L_{alpha, vis}, rho})
              + sum_{s} (L_{Y, s} rho L_{Y, s}^dag
                         - 0.5 {L_{Y, s}^dag L_{Y, s}, rho}).

This generates a CPTP semigroup on 96 x 96 density matrices (Lindblad
1976; GKS 1976).

### Step 4 -- Unitality

Family I (same as in U7's family I):

    sum_{alpha} L_{alpha, vis}^dag L_{alpha, vis}
        = gamma_1 sum_{alpha} P_{alpha, vis} (x) I_S
        = gamma_1 I_visible (x) I_S
        = gamma_1 I_96.

Family II:

    sum_{s} L_{Y, s}^dag L_{Y, s}
        = gamma_2 sum_{s} I_visible (x) X_s^2
        = gamma_2 I_visible (x) sum_s Pi_s         (Step 3: X_s^2 = Pi_s)
        = gamma_2 I_visible (x) I_S                (U4 Step 4: sum_s Pi_s = I_S)
        = gamma_2 I_96.

Total dissipator:

    sum L^dag L = (gamma_1 + gamma_2) I_96 = (2/k*) I_96   (g_1 = g_2 = 1/k*).

Unital. By Wolf 2012 Theorem 6.1, the maximally mixed state I_96/96
is a Lindblad steady state.

Numerical verification: ||sum L^dag L - (2/3) I_96|| ~ 1.1e-15
(machine precision).

### Step 5 -- Both families preserve C_3

The C_3 action on H_total is U_{C_3} (x) I_S where U_{C_3} is the
12 x 12 permutation matrix of U3 Step 1.

**Family I commutes with U_{C_3} (x) I_S** (by construction; same as
U6):

    [P_{alpha, vis} (x) I_S, U_{C_3} (x) I_S]
        = [P_{alpha, vis}, U_{C_3}] (x) I_S = 0,

since P_{alpha, vis} are the spectral projectors of U_{C_3}.

**Family II commutes with U_{C_3} (x) I_S** (trivially, since acts
as I on the visible side):

    [I_visible (x) X_s, U_{C_3} (x) I_S]
        = [I_visible, U_{C_3}] (x) X_s + I_visible (x) [X_s, I_S]
        = 0 + 0 = 0.

Numerical verification: max ||[L, U_{C_3} (x) I_S]|| = 0 across all
seven jump operators (machine precision).

Hence the dissipator PRESERVES the visible-side C_3 isotypic block
structure of density matrices, as designed.

### Step 6 -- Steady-state set dimension

By U8 (the C_3-symmetric-dissipator conservation law) the C_3-symmetric
dissipator preserves the C_3 isotypic block-diagonal structure of
density matrices. Each visible C_3 isotypic block (rank 4 per block,
three blocks) carries its own conserved population, contributing >= 3
diagonal-population zero modes.

The Yukawa-on-spinor jumps add their own conservation structure on
the spinor side. The chirality-conjugation operator G_7 anticommutes
with each X_s (since X_s swaps the +1 and -1 eigenstates of G_7
within Pi_s):

    G_7 X_s = G_7 (|s_R><s_L| + |s_L><s_R|)
            = -|s_R><s_L| + |s_L><s_R|
            != X_s

so G_7 is NOT a conserved quantity. However, each Pi_s commutes with
each X_s' (X_s' = 0 outside Pi_{s'}, so Pi_s X_s' = delta_{ss'} X_s,
likewise X_s' Pi_s = delta_{ss'} X_s) and so each Pi_s commutes with
the family II dissipator restricted to Pi_s. This gives 4 conserved
species populations per visible C_3 block, totalling >= 12 population
zero modes; plus off-diagonal coherences within each block that are
also conserved.

Numerical verification (SVD on the 9216 x 9216 vectorized Lindblad
superoperator):

    Lindblad kernel dim: 96
    Smallest singular value:     2.7e-17
    96th-smallest singular value: 2.5e-15
    97th-smallest singular value: 0.333  (well-separated)

The kernel is 96-dimensional, the LARGEST steady-state set of any of
the four Lindblad constructions tested in this work
(directed-edge: 1; isotypic: 12; spinor-coupled: 32;
Yukawa-selective: 96). This is consistent with the U8 conservation
law plus the additional spinor-side conservation from the X_s structure.

The maximally mixed state I_96/96 is in this 96-dim set (by Step 4
unitality) and serves as the canonical mass-flux readout. With this
choice, all trace identities are exact rationals.

### Step 7 -- Mass-flux trace identity per (species, generation channel)

For each species s in {charged-lepton, neutrino, up-quark, down-quark}
and each C_3 isotypic channel alpha in {trivial, omega, omegabar},
define the mass-flux on the h-eigenspace as

    m_{s, alpha} := Tr[ (P_{alpha, vis} P_h) (x) Pi_s
                       * (sum L^dag L) * rho_ss ].

Take rho_ss = I_96 / 96 (canonical default per Step 6) and use
sum L^dag L = (gamma_1 + gamma_2) I_96 = (2/k*) I_96 (Step 4). Then

    m_{s, alpha} = (2/k*) / 96 * Tr[ (P_{alpha, vis} P_h) (x) Pi_s ]
                 = (2/k*) / 96 * Tr(P_{alpha, vis} P_h) * Tr(Pi_s)
                 = (2/k*) / 96 * Tr(P_{alpha, vis} P_h) * 2
                 = (1 / (24 * k*)) * Tr(P_{alpha, vis} P_h).

(Used: Tr(A (x) B) = Tr(A) * Tr(B); Tr(Pi_s) = 2 for s in {e, nu, u, d}
since each is a rank-2 projector by U4 Step 4 on a single SU(2) doublet.)

By U3 Step 5 the C_3-content of the h-eigenspace is

    Tr(P_{trivial, vis} P_h)  = 1,
    Tr(P_{omega, vis} P_h)    = 1,
    Tr(P_{omegabar, vis} P_h) = 0.

With k* = 3:

    m_{s, trivial}  = 1/(24 * 3) * 1 = 1/72,
    m_{s, omega}    = 1/72,
    m_{s, omegabar} = 0,

for EVERY species s. The 4 x 3 table is a single column repeated four
times.

### Step 8 -- Per-species Koide ratio

For each species s, define

    Q_s := (m_{s, trivial} + m_{s, omega} + m_{s, omegabar})
         / (sqrt(m_{s, trivial}) + sqrt(m_{s, omega}) + sqrt(m_{s, omegabar}))^2.

Substituting the closed-form table:

    sum m   = 1/72 + 1/72 + 0       = 2/72 = 1/36
    sum sqrt m = sqrt(1/72) + sqrt(1/72) + 0 = 2 / sqrt(72)
    Q_s = (1/36) / (4 / 72) = (1/36) * (72/4) = 72 / (4 * 36) = 1/2.

Independent of species s. Q_charged-lepton = 1/2.

### Why selectivity does not break the species cancellation

The mass-flux factorizes as

    m_{s, alpha} = (constant) * (visible-trace) * (spinor-trace)
                 = (constant) * Tr(P_{alpha, vis} P_h) * Tr(Pi_s).

The denominator (sum_alpha sqrt(m_{s, alpha}))^2 picks up Tr(Pi_s) at
the level of factor. The numerator picks up Tr(Pi_s). The factors
cancel, leaving

    Q_s = sum_alpha Tr(P_{alpha, vis} P_h)
         / (sum_alpha sqrt(Tr(P_{alpha, vis} P_h)))^2
        = (1 + 1 + 0) / (sqrt(1) + sqrt(1) + sqrt(0))^2
        = 2 / 4 = 1/2.

Independent of the choice of species s. The species coupling on the
spinor side is "diagonal" with respect to the Koide ratio formula --
adding the spinor structure does NOT produce species-dependent Q
values under the canonical Lindblad mass-flux readout.

The structural reason is that each Yukawa swap X_s satisfies
X_s^2 = Pi_s exactly. This is a CONSEQUENCE of the chirality
structure on Pi_s (one |s_L> and one |s_R> per species) and the
swap nature of X_s. The sum sum_s X_s^2 = sum_s Pi_s = I_S then
forces the rate operator to be (g_1 + g_2) I_96 -- unital. With
unital R and Kronecker-product readout projector P_alpha (x) Pi_s,
the trace identity factorizes by Schur orthogonality
(Tr(A (x) B) = Tr(A) Tr(B)) and the species factor Tr(Pi_s) cancels
in the Koide ratio.

This is the same structural obstruction identified in U6's
"Why Q = 2/3 does not drop out" section: the direct ratio formula
applied to (m_alpha) with m_alpha proportional to Tr(P_alpha P_h)
gives a function of integer Tr-multiplicities, not of any
sqrt-coherent aggregation of multiplicities. The framework's Q_Koide
= 2/3 requires the P2 sqrt-coherent aggregation postulate
(`docs/framework/W4_identification_catalog.md` §3) which is NOT supplied by any
canonical Lindblad construction.

### Routes to bridging the obstruction

To produce species-dependent m_{s, alpha} that does NOT factorize via
Schur orthogonality, the Lindblad construction would need EITHER

(a) a non-unital rate operator (not proportional to I_96), OR

(b) a non-Kronecker-product readout projector (e.g., an entangled
    visible-spinor projector mixing P_alpha (x) Pi_s with
    P_alpha' (x) Pi_s'), OR

(c) a non-maximally-mixed steady state rho_ss != I_96/96 (impossible
    in the unital framework -- I/dim is always a steady state of a
    unital channel).

Route (a) requires jump operators that are NOT a sum of operators of
the form sqrt(rate) * P_a (x) Q_s with P_a, Q_s projectors summing
to the identity. The standard "tunneling" jumps L = a + a^dag (with a
a lowering operator) have R = (a^dag a + a a^dag) which can be a
non-trivial diagonal operator (like a number operator + 1). The
present X_s is a swap, not a tunneling, and X_s^2 = Pi_s makes the
corresponding R unital. A FUTURE construction might use lowering
operators a_s = |s_L><s_R| (not Hermitian), giving non-unital R.
This has not been attempted here because (i) the resulting jumps
are NOT C_3-symmetric obviously, requiring further analysis, and
(ii) the lowering-operator interpretation of "Yukawa-like" is one
notch further from the framework's existing W4 cancellation channel
than the Hermitian swap is. Both directions remain open.

Route (b) would require a structural reason to entangle the visible
and spinor readout projectors. The framework's existing structural
content (B3 Pati-Salam decomposition, B5.3-core C_3 isotypic
decomposition) does not supply such an entangled projector; doing so
would be a new structural input.

Route (c) is impossible by Wolf 2012 Theorem 6.1: any unital channel
admits I/dim as a steady state.

None of (a), (b), (c) is supplied by MDL+toggle in any current
formulation.

### Numerical verification

`predictions/lindblad_yukawa_selective.py` builds the 96 x 96 operators
via `proofs/foundations/lindblad_yukawa_selective_construction.py`, the
9216 x 9216 vectorised Lindblad superoperator, and computes:

- Family I: 3 jumps, all commuting with U_{C_3} (x) I_S
  (max ||commutator|| = 0 to machine precision)
- Family II: 4 jumps, all commuting with U_{C_3} (x) I_S
  (max ||commutator|| = 0 to machine precision)
- ranks(Pi_e, Pi_nu, Pi_u, Pi_d) = (2, 2, 2, 2) on the spinor side
- X_s^2 = Pi_s for every species (verified, error < 1e-8)
- sum_s X_s^2 = I_S (verified, error < 1e-8)
- ||sum L^dag L - (2/3) I_96|| = 1.1e-15 (unitality, machine precision)
- ||L(rho_ss = I/96)|| = 4.3e-19 (steady state, machine precision)
- (Tr(P_triv P_h), Tr(P_omega P_h), Tr(P_omegabar P_h)) = (1, 1, 0)
  to ~1e-10
- 96 smallest SVD singular values at machine zero ~ 2e-15 (kernel dim 96)
- 97th smallest SVD singular value: 0.333 (well-separated above kernel)
- Mass-flux table: every entry m_{s, alpha} = (1/72, 1/72, 0) within ~1e-10
- Per-species Q_s = 0.5 within ~1e-7 (residual from sqrt(0) numerical floor)

## Result

Closed form on H_total = H_visible (x) S = 96-dim, with H_spinor = 0,
family I C_3-isotypic + family II Yukawa L<->R swap (both at rate
1/k* per channel, both C_3-symmetric):

    Mass-flux table (every row identical):
                     trivial  omega   omegabar
        species s:    1/72    1/72    0
    Per-species Koide ratio Q_s = 1/2  for every s in {e, nu, u, d}.
    Lindblad steady-state set dimension: 96 (NOT unique).
    Maximally mixed state I_96/96 in the set; canonical readout.

In particular Q_charged-lepton = 1/2, NOT the observed Q_Koide = 2/3.

## Comparison with experiment

| Quantity | Predicted (this lemma) | Observed (PDG) | Deviation |
|----------|------------------------|----------------|-----------|
| Q_charged-lepton | 1/2 = 0.5000 | 2/3 ~ 0.6667 | -0.167 abs (~ 25% rel) |

(Per Wolfram MathWorld / Koide 1981: the observed charged-lepton mass
ratios m_e, m_mu, m_tau give Q ~ 0.6666 within experimental precision,
often quoted as exactly 2/3.)

The lemma's prediction Q_s = 1/2 is a STRUCTURAL value determined by
the integer C_3 multiplicities (1, 1, 0) of the h-eigenspace at P
alone; the Lindblad mass-flux readout does NOT see species identity in
this construction, despite the spinor-side selectivity in the Yukawa
jumps.

## Open questions

1. **The 96-dim steady-state set.** A direct analysis of the noise
   commutant (Wolf 2012 §6) for the family I + family II jump
   operators would give an analytic count of the kernel dimension. The
   conservation structure includes (i) the C_3 isotypic block-diagonal
   structure on the visible side (gives >= 12 population modes when
   crossed with the spinor structure), (ii) per-species population
   conservation on the spinor side (each Pi_s commutes with the full
   family II dissipator restricted to that block), and (iii)
   off-diagonal coherences. The numerically observed kernel dim 96 is
   consistent with a count of 3 visible C_3 blocks * 4 species
   populations + cross-coherence modes, but the rigorous analytic
   verification is open.

2. **Bridge from Lindblad to Q_Koide = 2/3.** Same structural
   obstruction as in U6, U7, this lemma: the Lindblad mass-flux
   readout produces values m_{s, alpha} = constant * Tr(P_alpha P_h)
   * Tr(Pi_s) by the Schur-orthogonality identity for unital channels.
   The direct ratio formula sum m / (sum sqrt m)^2 with such m gives
   1/2. The observed 2/3 arises only via the P2 sqrt-coherent
   aggregation. Bridging the two requires a non-Lindblad piece, which
   is not supplied by any canonical Lindblad construction including
   this Yukawa-selective one.

3. **Conjecture C1 derivation.** The Yukawa-like jump operators are
   currently postulated, not derived from MDL+toggle. The two
   motivations sketched in the §Conjecture C1 section (MDL preference
   for C_3-preserving compression; EW symmetry breaking) are
   heuristics, not proofs. A genuine derivation of C1 would require
   formalising "MDL discards species-mixing information" as a precise
   statement about Lindblad jump operators, OR constructing the EW
   Higgs mechanism as a downstream consequence of MDL+toggle (a major
   undertaking).

4. **Non-Hermitian Yukawa lowering operators.** Replacing the
   Hermitian swap X_s with a non-Hermitian lowering operator
   a_s = |s_L><s_R| (and including the Hermitian conjugate as a
   separate jump channel) would give non-unital R: a_s^dag a_s =
   |s_R><s_R|, and (a_s + h.c.)^dag (a_s + h.c.) = Pi_s, so the
   Hermitian-combination case reduces back to the swap (modulo a
   factor of 2). The truly non-Hermitian case (treating a_s and a_s^dag
   as separate jumps) gives R = sum_s (|s_L><s_L| + |s_R><s_R|)
   = sum_s Pi_s = I_S still -- unital. To get non-unital R one would
   need ASYMMETRIC rates between a_s and a_s^dag, breaking the
   detailed-balance condition of the Lindblad equation. This is an
   open construction direction, not attempted here.

5. **Per-species mass-flux factorisation.** The factorisation
   m_{s, alpha} = constant * Tr(P_alpha P_h) * Tr(Pi_s) is a direct
   consequence of (i) rho_ss = I_96/dim being in the steady-state set,
   (ii) sum L^dag L = constant * I_96 (unitality), and (iii) the
   readout projector being a Kronecker product P_{alpha, vis} (x) Pi_s.
   To produce species-dependent m_{s, alpha}, the construction would
   need to break one of (i)-(iii). All three remain open.

## References

### Memory / standards

  commits performed).
  This lemma is BLOCKED at full closure: the construction is well-
  defined and correctly preserves C_3 as designed, but (a) Conjecture
  C1 is not derived from MDL+toggle, and (b) the Q value Q_s = 1/2
  does not match the framework target Q_Koide = 2/3. The lemma is a
  CANDIDATE consequence of C1, not a framework theorem.
- `docs/parameters/parameter_linter.md` -- hard quality gate.

### Cited mathematical theorems

- **Lindblad, G.** (1976). On the generators of quantum dynamical
  semigroups. *Communications in Mathematical Physics* **48**, 119-130.
- **Gorini, V., Kossakowski, A., Sudarshan, E.C.G.** (1976). Completely
  positive dynamical semigroups of N-level systems. *J. Math. Phys.*
  **17**, 821-825.
- **Wolf, M.M.** (2012). *Quantum Channels and Operations.* §6 fixed
  points of unital channels; Theorem 6.1.
- **Breuer, H.-P. & Petruccione, F.** (2002). *The Theory of Open
  Quantum Systems.* Oxford University Press. Ch. 3.
- **Pati, J.C. & Salam, A.** (1974). Lepton number as the fourth
  colour. *Phys. Rev. D* **10**, 275-289.
- **Lawson, H.B. & Michelsohn, M.-L.** (1989). *Spin Geometry.*
  Princeton Math. Series 38. Ch. I §§1-5.
- **Koide, Y.** (1981). Fermion-boson two-body model of quarks and
  leptons. *Phys. Lett. B* **120**, 161-165.
- **Serre, J.-P.** (1977). *Linear Representations of Finite Groups.*
  §2.6 Schur orthogonality.

### Upstream framework theorems (closed)

- `../../predictions/walker_dynamics_derivation.md` -- W1-W4: NB walker; Hashimoto;
  Bloch decomposition; Open System reading.
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` -- h with multiplicity 2.
- `docs/theorem_B5_3_core.md` -- C_3 multiplicities (4, 4, 4) full
  fibre, (1, 1, 0) on h-subspace.
- `../../predictions/theorem_B3_spinor_fermion_derivation.md` -- Cl(6, 0) spinor with chirality
  G_7 and species partition Pi_e + Pi_nu + Pi_u + Pi_d = I_S.
- `predictions/B_P_doubly_degenerate_h.py` -- 12-dim B(P) construction.
- `predictions/lindblad_steady_state_at_P.py` -- directed-edge
  Lindblad lemma.
- `predictions/lindblad_isotypic_at_P.py` -- C_3-isotypic Lindblad
  lemma.
- `predictions/lindblad_spinor_coupled.py` -- spinor-coupled Lindblad
  lemma (C_3-BREAKING family II).
- `predictions/k_star.py` -- k* = 3.
- `predictions/d_spatial.py` -- d_spatial = 3.

### Sibling / interpretation documents

  identifies the conservation law that the present construction
  satisfies (and inherits the steady-state degeneracy from).
  the directed-edge Lindblad off P (n_s = 4).
- `docs/framework/W4_identification_catalog.md` §3 -- P1 (Ramanujan selection),
  P2 (sqrt-coherent aggregation): adopted structural postulates beyond
  MDL + toggle. The Lindblad construction does not derive P2.
- `predictions/Q_Koide.py` + `predictions/Q_Koide_derivation.md` --
  Q_Koide = 2/3 via P2 sqrt-coherent aggregation; status BLOCKED.
- an internal Markov-vs-unitary classification audit -- Lindblad / Open System reading
  recommendation.

## Files referenced but NOT modified

Per task constraints: `results/parameters.csv`, `docs/parameters/derivations.md`,
B3/B5/B6 docs, `../../predictions/walker_dynamics_derivation.md`, and existing
`predictions/*.py` files OTHER than the ones produced by this task are
NOT edited. Only the three files
`predictions/lindblad_yukawa_selective.py`,
`predictions/lindblad_yukawa_selective_derivation.md`, and
`proofs/foundations/lindblad_yukawa_selective_construction.py` are
produced.

No commits performed; no remote push.
