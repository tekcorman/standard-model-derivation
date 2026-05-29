# Derivation: Spinor-coupled Lindblad on H_visible (x) S = 96-dim

**NOTE (post-A3, 2026-04-18):** Under the three-axiom framework (A1+A2+A3; docs/framework/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 Theorem 25 (predictions/observer_hilbert_space.py). The Lindblad-form derivation from A1+A2+A3 (vs adoption), Pati-Salam labeling, and P1/P2 remain separately load-bearing.

**Companion file:** `predictions/lindblad_spinor_coupled.py`
**Construction file:** `proofs/foundations/lindblad_spinor_coupled_construction.py`
**Status:** PARTIAL closure. (i) Closed-form 4 x 3 mass-flux table with
exact rationals (1/72, 1/72, 0) per species s; (ii) per-species Koide
ratio Q_s = 1/2 universally, NOT the observed Q_Koide = 2/3; (iii) the
spinor-coupled Lindblad reduces the per-Hilbert-dim normalised steady-
state degeneracy relative to the pure C_3-isotypic Lindblad
(`predictions/lindblad_isotypic_at_P.py` kernel dim 12 on a 12 x 12 = 144
density-matrix space; this construction kernel dim 32 on a 96 x 96 =
9216 density-matrix space) but does NOT collapse to a unique steady
state; (iv) the maximally mixed state I_96/96 is in the steady-state
set (by unitality) and is the canonical default for the mass-flux
readout.

**Predecessors:**
- `predictions/lindblad_steady_state_at_P.py` — directed-edge dephasing
  Lindblad on the 12-dim Bloch fibre. Unique steady state I/12; one
  mass scale m_h = 2/k* = 2/3.
- `predictions/lindblad_isotypic_at_P.py` — C_3-isotypic Lindblad on
  the 12-dim Bloch fibre. Three mass-flux values per C_3 sector
  (1/3, 1/3, 0); 12-dim degenerate steady state; Q_iso = 1/2.

**Companion scoping:**
  C_3-symmetric jump operator gives a degenerate steady state by
  block-diagonalisation; the natural fix is to add jumps that break C_3.
  This document carries out that fix and reports honestly that it does
  not collapse the steady state to dim 1.

## Abstract

The directed-edge Lindblad
(`predictions/lindblad_steady_state_at_P.py`) gives a unique steady
state on the 12-dim visible Bloch fibre at P but produces only one
mass scale m_h = 2/3, with no species or generation discrimination.
The C_3-isotypic Lindblad (`predictions/lindblad_isotypic_at_P.py`)
exposes three mass-fluxes (1/3, 1/3, 0) per C_3 sector but its
dissipator commutes with the C_3 action, producing a 12-dim
degenerate steady-state set and no unique read-off. The
companion scoping doc an internal working note
establishes that ANY C_3-respecting jump operator produces this
degeneracy by construction. The natural remediation, identified in
that scoping doc as "structural option (a)," is to add jump operators
that break C_3 explicitly. The cleanest available C_3-breaking
content is the spinor B-L species labelling supplied by Theorem B3
(`../../predictions/theorem_B3_spinor_fermion_derivation.md`).

This document constructs that spinor-coupled Lindblad on the
H_total = H_visible (x) S = 12 (x) 8 = 96-dim Hilbert space. The
visible-side jump operators are the same C_3-isotypic ones (Family
I, on the spinor identity); a second jump family (Family II,
visible directed-edge tensored with the B-L species projector on
the spinor) is C_3-breaking on the visible side. Both families have
total dissipation rate (1/k*) I_96, so the combined dissipator is
unital with sum L^dag L = (2/k*) I_96.

We derive in closed form:

1. The maximally mixed state I_96/96 is a Lindblad steady state.
2. The 4 x 3 mass-flux table m_{s, alpha} = (1/72) * Tr(P_alpha P_h)
   (independent of species s by the Schur-orthogonality factorization
   of unital channels with mutually-commuting projector pairs);
   evaluating at the (1, 1, 0) C_3 content of the h-eigenspace gives
   the table

       m_{s, trivial}  = 1/72,    m_{s, omega}    = 1/72,    m_{s, omegabar} = 0

   for every s in {charged-lepton, neutrino, up-quark, down-quark}.
3. Per-species Koide ratio
        Q_s = (m_{s, t} + m_{s, o} + m_{s, ob})
            / (sqrt(m_{s, t}) + sqrt(m_{s, o}) + sqrt(m_{s, ob}))^2
            = 1/2   for every s.
   In particular Q_charged-lepton = 1/2, NOT the observed 2/3.
4. The Lindblad superoperator's vectorised kernel has dimension 32
   (numerical SVD on the 9216 x 9216 superoperator); the steady
   state is NOT unique. The C_3-breaking content of Family II
   removes some — but not all — of the kernel dimensions of the
   pure-isotypic Lindblad construction.

## Framework axioms invoked

- **A1 (self-inverse toggle).** Each edge e of the srs primitive cell
  carries a toggle T_e with T_e * T_e = 1.
- **A2 (MDL).** The observer encodes the toggle stream by its reduced
  word.

No further axioms; no physical observation enters.

## Upstream theorems (closed)

- **U1.** `../../predictions/walker_dynamics_derivation.md` (W1-W4) -- the observer's
  MDL-compressed dynamics on srs are non-backtracking walks; B is the
  Hashimoto operator on the 12-dim directed-edge state space; W4
  cancellation rate per step is 1/k*. Reading-Conventions section
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
  factored out. Step 2 constructs the B-L Cartan generator
  Y = Gamma_{56} / (2i); Step 4 establishes the species labels
  {nu, e, u, d} x {L, R} on the 8 weight states. The Y-eigenvalue
  signs partition S = 4 (Y_+) + 4 (Y_-) into the lepton-axis and
  quark-axis sectors.
- **U5.** `predictions/lindblad_steady_state_at_P.py` -- directed-edge
  basis dephasing Lindblad: rho_ss = I/12; m_h = 2/k* = 2/3.
- **U6.** `predictions/lindblad_isotypic_at_P.py` -- C_3-isotypic
  Lindblad: m_alpha_h = 1/k* * Tr(P_alpha P_h); kernel dim 12.

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
  colour. *Phys. Rev. D* **10**, 275-289 -- B-L = 4-colour decomposition
  used in U4.
- **Lawson, H.B. & Michelsohn, M.-L.** (1989). *Spin Geometry.*
  Princeton Math. Series 38. Ch. I, Theorem 5.7 (complex Cl(2k) ~
  M_{2^k}(C)).

## Derivation

### Step 1 — Total Hilbert space

By U1 the visible sector at P is the 12-dim Bloch fibre H_visible. By
U4 the spinor sector is the 8-dim Cl(6, 0) Dirac spinor S. The total
Hilbert space is the tensor product

    H_total := H_visible (x) S,        dim H_total = 12 * 8 = 96.

The framework reading of the spinor sector as the "species axis" of one
Pati-Salam family is U4's Theorem B3 statement.

### Step 2 — Hamiltonian

H_visible = (B(P) + B(P)^dag)/2 (Hermitian symmetrisation; same
choice as U5). H_spinor = 0 (no internal spinor dynamics; the Lindblad
dissipator carries all sector-mixing structure). The total Hamiltonian
is the Kronecker sum

    H_full := H_visible (x) I_S + I_visible (x) H_spinor
            = H_visible (x) I_S    (since H_spinor = 0).

H_full is Hermitian by Hermiticity of H_visible. The choice
H_spinor = 0 is the canonical "no extra structure" baseline of the
construction-plan (task plan Step B).

### Step 3 — Jump operators (Family I + Family II)

**Family I** (visible C_3-isotypic, on the spinor identity):

    L_{alpha, vis} := sqrt(1/k*) * P_{alpha, vis} (x) I_S,
    alpha in {trivial, omega, omegabar}.

P_{alpha, vis} is the rank-4 C_3-isotypic projector on H_visible, per
the Schur orthogonality formula
P_alpha = (1/3) sum_{j=0,1,2} chi_alpha(c^j)^* U_{C_3}^j
(with chi_trivial = (1, 1, 1), chi_omega = (1, omega, omega^2),
chi_omegabar = (1, omega^2, omega)). Family I is the same set of jumps
as U6 (`predictions/lindblad_isotypic_at_P.py`).

**Family II** (visible directed-edge tensored with the B-L species
projector on the spinor):

    L_{e, s} := sqrt(1/k*) * P_e (x) Pi_s,
    e in {0, ..., 11},  s in {Y_+, Y_-}.

P_e is the rank-1 directed-edge projector on H_visible. Pi_{Y_+} is
the rank-4 orthogonal projector onto the Y > 0 eigenspace of
Y = Gamma_{56}/(2i) on S; Pi_{Y_-} is the projector onto the Y < 0
eigenspace. By U4 Step 2 the Y eigenvalues are +/- 1/2 each with
multiplicity 4, so rank(Pi_{Y_+}) = rank(Pi_{Y_-}) = 4. By U4 Step 4
the (Z/2)^3 convention places Pi_{Y_-} on the lepton axis (B-L = -1
per Pati-Salam) and Pi_{Y_+} on the quark axis (B-L = +1/3 per PS,
colour collapsed). Both choices give the same Lindblad construction
since the Y-sign is precisely the (Z/2) convention freedom (b) of U4
Step 5.

The combined Lindblad equation is

    dL/dt rho = -i [H_full, rho]
              + sum_{alpha} (L_{alpha, vis} rho L_{alpha, vis}^dag
                             - 0.5 {L_{alpha, vis}^dag L_{alpha, vis}, rho})
              + sum_{e, s} (L_{e, s} rho L_{e, s}^dag
                            - 0.5 {L_{e, s}^dag L_{e, s}, rho}).

Generates a CPTP semigroup on 96 x 96 density matrices (Lindblad 1976;
GKS 1976).

### Step 4 — Unitality

Family I:

    sum_{alpha} L_{alpha, vis}^dag L_{alpha, vis}
        = (1/k*) sum_{alpha} P_{alpha, vis} (x) I_S
        = (1/k*) I_visible (x) I_S
        = (1/k*) I_96.

Family II:

    sum_{e, s} L_{e, s}^dag L_{e, s}
        = (1/k*) sum_e P_e (x) sum_s Pi_s
        = (1/k*) I_visible (x) I_S
        = (1/k*) I_96.

Total dissipator:

    sum L^dag L = (2/k*) I_96.

Unital. By Wolf 2012 Theorem 6.1, the maximally mixed state
I_96/96 is a Lindblad steady state.

### Step 5 — Family II jumps break C_3

The C_3 action on H_total is U_{C_3} (x) I_S where U_{C_3} is the
12 x 12 permutation matrix of U3 Step 1. Each Family I jump operator
P_{alpha, vis} (x) I_S commutes with U_{C_3} (x) I_S:

    [P_{alpha, vis} (x) I_S, U_{C_3} (x) I_S]
        = [P_{alpha, vis}, U_{C_3}] (x) I_S = 0.

(Verified numerically in the construction file: maximum ||commutator||
is exactly 0 to machine precision.)

Each Family II jump P_e (x) Pi_s does NOT commute with U_{C_3} (x) I_S:

    [P_e (x) Pi_s, U_{C_3} (x) I_S]
        = [P_e, U_{C_3}] (x) Pi_s,

and [P_e, U_{C_3}] = U_{C_3} P_e U_{C_3}^{-1} - P_e = P_{C_3 e} - P_e
which is non-zero whenever C_3 e != e (true for all 12 directed edges
of srs since the C_3 action has no fixed directed edges per U3 Step 2:
all four orbits have length 3). Verified numerically: maximum
||[L_{e, s}, U_{C_3} (x) I_S]|| > 1.6 across the 24 Family II jumps.

This is the structural C_3-breaking content of Family II.

### Step 6 — Steady-state set dimension

The vectorised Lindblad superoperator has dimension 9216 x 9216 (= 96^2).
By U6 the pure-isotypic dissipator has kernel dim 12 on the 144 x 144
visible-only superoperator (= one full block-diagonal density matrix
per C_3 isotypic block). The spinor coupling adds Family II, which
breaks C_3 and reduces the kernel; numerical SVD on the 9216 x 9216
superoperator gives kernel dim 32 (32 smallest singular values at
machine zero ~ 1e-16, 33rd singular value well-separated above 1e-3).

The 32-dim kernel is the "interaction algebra" of the noise (Wolf 2012
§6): operators commuting with both H_full and all jump operators. A
direct computation shows that operators of the form
A_e (x) B_s (with A_e diagonal in the directed-edge basis on visible
side and B_s diagonal in the (Pi_{Y_+}, Pi_{Y_-}) basis on spinor side)
commute with all Family II jumps; combined with the Family I commutant
(operators block-diagonal in visible C_3 isotypic blocks) and the
H_full commutant, the residual kernel has dim 32 (verified
numerically; the explicit algebra-counting argument for the value 32
is sketched in Open Question 1 below).

The maximally mixed state I_96/96 is in this 32-dim set (by Step 4
unitality) and serves as the canonical mass-flux readout below. With
this choice, all trace identities are exact rationals.

### Step 7 — Mass-flux trace identity per (species, generation channel)

For each species s in {charged-lepton, neutrino, up-quark, down-quark}
and each C_3 isotypic channel alpha in {trivial, omega, omegabar},
define the mass-flux on the h-eigenspace as

    m_{s, alpha} := Tr[ (P_{alpha, vis} P_h) (x) Pi_s
                       * (sum L^dag L) * rho_ss ].

Take rho_ss = I_96 / 96 (canonical default per Step 6) and use
sum L^dag L = (2/k*) I_96 (Step 4). Then

    m_{s, alpha} = (2/k*) / 96 * Tr[ (P_{alpha, vis} P_h) (x) Pi_s ]
                 = (2/k*) / 96 * Tr(P_{alpha, vis} P_h) * Tr(Pi_s)
                 = (2 * 2) / (96 * k*) * Tr(P_{alpha, vis} P_h)
                 = (1 / (24 * k*)) * Tr(P_{alpha, vis} P_h).

(Used: Tr(A (x) B) = Tr(A) * Tr(B); Tr(Pi_s) = 2 for s in {e, nu, u, d}
since each is a rank-2 projector by U4 Step 4 on a single SU(2)
doublet.)

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

### Step 8 — Per-species Koide ratio

For each species s, define

    Q_s := (m_{s, trivial} + m_{s, omega} + m_{s, omegabar})
         / (sqrt(m_{s, trivial}) + sqrt(m_{s, omega}) + sqrt(m_{s, omegabar}))^2.

Substituting the closed-form table:

    sum m   = 1/72 + 1/72 + 0       = 2/72 = 1/36
    sum sqrt m = sqrt(1/72) + sqrt(1/72) + 0 = 2 / sqrt(72)
    Q_s = (1/36) / (4 / 72) = (1/36) * (72/4) = 72 / (4 * 36) = 1/2.

Independent of species s. Q_charged-lepton = 1/2.

### Step 9 — Comparison with observation

Q_Koide observed = 2/3 (per `predictions/Q_Koide.py` which reports the
phenomenological compact form (mu_t + mu_o + mu_ob)/(k* mu_t) at
(mu_t, mu_o, mu_ob) = (4, 2, 2) — see U6 derivation §"Why Q = 2/3
does not drop out").

Q_charged-lepton (predicted, this lemma) = 1/2.

Deviation: |1/2 - 2/3| = 1/6 ~ 0.167 (~ 33%).

### Why Q does not depend on species

The mass-flux factorizes as

    m_{s, alpha} = (constant) * (visible-trace) * (spinor-trace)
                 = (constant) * Tr(P_{alpha, vis} P_h) * Tr(Pi_s).

The denominator (sum_alpha sqrt(m_{s, alpha}))^2 picks up sqrt(Tr(Pi_s))^2
= Tr(Pi_s) at the level of factor. The numerator picks up Tr(Pi_s).
The factors cancel, leaving

    Q_s = sum_alpha Tr(P_{alpha, vis} P_h)
         / (sum_alpha sqrt(Tr(P_{alpha, vis} P_h)))^2
        = (1 + 1 + 0) / (sqrt(1) + sqrt(1) + sqrt(0))^2
        = 2 / 4 = 1/2.

Independent of which species s we take. The species coupling on the
spinor side is "diagonal" with respect to the Koide ratio formula —
adding the spinor structure does NOT produce species-dependent Q values
under the canonical Lindblad mass-flux readout.

This is the same structural obstruction identified in U6's
"Why Q = 2/3 does not drop out" section: the direct ratio formula
applied to (m_alpha) with m_alpha proportional to Tr(P_alpha P_h)
gives a function of integer Tr-multiplicities, not of any sqrt-coherent
aggregation of multiplicities. The framework's Q_Koide = 2/3 requires
the P2 sqrt-coherent aggregation postulate
(`docs/framework/W4_identification_catalog.md` §3) which is NOT supplied by any
canonical Lindblad construction.

### Numerical verification

`predictions/lindblad_spinor_coupled.py` builds the 96 x 96 operators
via the construction file `proofs/foundations/lindblad_spinor_coupled_construction.py`,
the 9216 x 9216 vectorised Lindblad superoperator, and computes:

- ranks(Pi_{Y_+}, Pi_{Y_-}) = (4, 4) on the spinor side
- ranks(Pi_e, Pi_nu, Pi_u, Pi_d) = (2, 2, 2, 2) on the spinor side
- ||sum L^dag L - (2/3) I_96|| = 1.1e-16 (unitality, machine precision)
- max ||[L_family_I, U_C3 (x) I]|| = 0 (machine zero)
- max ||[L_family_II, U_C3 (x) I]|| = 1.633 (sharp C_3-breaking)
- (Tr(P_triv P_h), Tr(P_omega P_h), Tr(P_omegabar P_h)) = (1, 1, 0) to ~1e-10
- 32 smallest SVD singular values at machine zero ~ 1e-16 (kernel dim 32)
- 33rd smallest SVD singular value well-separated > 1e-3
- Mass-flux table: every entry m_{s, alpha} = (1/72, 1/72, 0) within ~1e-10
- Per-species Q_s = 0.5 within ~1e-7 (residual from sqrt(0) numerical floor)

## Result

Closed form on H_total = H_visible (x) S = 96-dim, with
H_spinor = 0, family I uniform-rate isotypic + family II uniform-rate
edge-by-species jump operators (both at rate 1/k*):

    Mass-flux table (every row identical):
                     trivial  omega   omegabar
        species s:    1/72    1/72    0
    Per-species Koide ratio Q_s = 1/2  for every s in {e, nu, u, d}.
    Lindblad steady-state set dimension: 32 (NOT unique).
    Maximally mixed state I_96/96 in the set; canonical readout.

In particular Q_charged-lepton = 1/2, NOT the observed Q_Koide = 2/3.

## Comparison with experiment

| Quantity | Predicted (this lemma) | Observed (PDG) | Deviation |
|----------|------------------------|----------------|-----------|
| Q_charged-lepton | 1/2 = 0.5000 | 2/3 ~ 0.6667 | -0.167 abs (~ 25% rel) |

(Per Wolfram MathWorld / Koide 1981: the observed charged-lepton mass
ratios m_e, m_mu, m_tau give Q = (m_e + m_mu + m_tau) /
(sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2 ~ 0.6666 within experimental
precision, often quoted as exactly 2/3.)

The lemma's prediction Q_s = 1/2 is a STRUCTURAL value determined by
the integer C_3 multiplicities (1, 1, 0) of the h-eigenspace at P and
nothing else; the Lindblad mass-flux readout does NOT see species
identity in this construction.

## Open questions

1. **The 32-dim steady-state set.** A direct analysis of the noise
   commutant (Wolf 2012 §6) for the family I + family II jump
   operators would give an analytic count of the kernel dimension. A
   sketch: family II jumps L_{e, s} = sqrt(gamma) P_e (x) Pi_s
   commute with operators of the form A (x) B where A is diagonal in
   the directed-edge basis (12 dimensions) and B is block-diagonal in
   the (Pi_{Y_+}, Pi_{Y_-}) decomposition (16 + 16 = 32 dimensions
   per visible block, but the structure is reduced when crossed with
   family I and H_full constraints). The numerical kernel dim 32 is
   consistent with such a count but the rigorous analytic verification
   is open. This is a Wolf-2012-§6-style algebra-of-fixed-points
   exercise.

2. **Bridge from Lindblad to Q_Koide = 2/3.** The Lindblad mass-flux
   readout produces values m_{s, alpha} = constant * Tr(P_alpha P_h)
   * Tr(Pi_s) by the Schur-orthogonality identity for unital channels
   (any factorised projector pair (P_a (x) Pi_s) acting against the
   unital rate operator gives a product of traces). The direct ratio
   formula sum m / (sum sqrt m)^2 with such m gives a quantity
   determined by the integer C_3 content of the h-eigenspace alone,
   which is 1/2. The observed 2/3 arises only via a different
   functional aggregation (the P2 sqrt-coherent aggregation of
   `docs/framework/W4_identification_catalog.md` §3). Bridging the two requires
   a non-Lindblad piece — the P2 postulate, or an analogous coherent
   aggregation rule that mixes the C_3 isotypic components on the
   visible side. No such bridge is supplied by the canonical Lindblad
   construction with H_spinor = 0 and unital dissipator.

3. **H_spinor = 0 vs. non-trivial.** Adding H_spinor = c_T_L T_L +
   c_T_R T_R + c_Y Y_BL would split the spinor 8 weight states by
   the Cartan eigenvalues. The unitary part -i [I_visible (x) H_spinor, .]
   would then mix off-diagonal spinor coherences. By construction
   this does NOT change rho_ss = I_96/96 (which is in the kernel of
   any -i[H, .] for any Hermitian H). It MIGHT reduce the steady-state
   set further by adding non-trivial constraints to the noise commutant
   on the spinor side. This is open and would be the natural next
   construction step. (The simplest test: with H_spinor proportional
   to Y_BL alone, the spinor sector is split into Pi_{Y_+}, Pi_{Y_-}
   blocks and the visible coherences within each block are unchanged
   — kernel reduction would come from off-diagonal Y-coherences in the
   spinor sector, which the H_spinor would lift the degeneracy of.
   Numerical experiment is straightforward but not done here.)

4. **n_s read-off (Step G of task plan).** Off-P extension along the
   Gamma-P axis is a substantial separate construction. With the present
   spinor-coupled Lindblad on a single Bloch fibre at P, the dissipator
   has a non-trivial kernel (dim 32) and the q-derivative of the
   spectral density of any observable is not uniquely defined (it
   depends on which steady state in the 32-dim set is chosen, or on
   how one defines a "thermal" ensemble within the set). The same
   structural obstruction as in an internal working note
   §S2 applies. Step G of the task plan was not attempted here.

5. **Per-species mass-flux factorisation.** The factorisation
   m_{s, alpha} = constant * Tr(P_alpha P_h) * Tr(Pi_s) is a direct
   consequence of (i) rho_ss = I_96 / dim being in the steady-state
   set, (ii) sum L^dag L = constant * I_96 (unitality), and (iii) the
   projector being a Kronecker product P_{alpha, vis} (x) Pi_s. To
   produce species-dependent m_{s, alpha}, the construction would
   need either a non-maximally-mixed steady state (impossible in the
   pure unital framework) or a non-Kronecker-product readout
   projector (e.g. an entangled visible-spinor projector). Neither is
   forced by MDL+toggle; both would be additional axiomatic input.

6. **Charge-conjugation convention (B3.2).** The (Z/2)^3 convention
   freedom of U4 Step 5 includes the Y-sign flip Pi_{Y_+} <-> Pi_{Y_-},
   which swaps the labels {lepton-axis, quark-axis}. Both choices give
   the same mass-flux table (each row is identical anyway), so this
   convention is NOT a freedom in the present readout.

## References

### Memory / standards

  commits performed).
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
  colour. *Phys. Rev. D* **10**, 275-289 -- B-L decomposition of
  Spin(6).
- **Lawson, H.B. & Michelsohn, M.-L.** (1989). *Spin Geometry.*
  Princeton Math. Series 38. Ch. I §§1-5.
- **Koide, Y.** (1981). Fermion-boson two-body model of quarks and
  leptons. *Phys. Lett. B* **120**, 161-165 -- the observed Koide value
  Q ~ 2/3.
- **Serre, J.-P.** (1977). *Linear Representations of Finite Groups.*
  §2.6 Schur orthogonality.
- **Sunada, T.** (2012). *Topological Crystallography.* §§5-6.

### Upstream framework theorems (closed)

- `../../predictions/walker_dynamics_derivation.md` -- W1-W4: NB walker; Hashimoto;
  Bloch decomposition; Open System reading.
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` -- h with multiplicity 2;
  Step 3 corrected isotypic decomposition (1, 1, 0).
- `docs/theorem_B5_3_core.md` -- C_3 multiplicities (4, 4, 4) full
  fibre, (1, 1, 0) on h-subspace.
- `../../predictions/theorem_B3_spinor_fermion_derivation.md` -- Cl(6, 0) spinor with
  Spin(4) x Spin(2) decomposition; B-L Cartan generator.
- `predictions/B_P_doubly_degenerate_h.py` -- 12-dim B(P) construction.
- `predictions/lindblad_steady_state_at_P.py` -- directed-edge basis
  Lindblad lemma (predecessor).
- `predictions/lindblad_isotypic_at_P.py` -- C_3-isotypic Lindblad
  lemma (predecessor; degenerate steady state).
- `predictions/k_star.py` -- k* = 3.
- `predictions/d_spatial.py` -- d_spatial = 3.

### Sibling / interpretation documents

  for the C_3-symmetric isotypic Lindblad off P; identifies the
  C_3-breaking remediation that this document carries out.
  the directed-edge Lindblad off P (n_s = 4).
- `../../docs/framework/B3_B6_reconciliation.md` -- physical interpretation of
  the C_3 isotypic multiplicities is open; default reading "pure
  algebraic Cartan label."
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
`predictions/lindblad_spinor_coupled.py`,
`predictions/lindblad_spinor_coupled_derivation.md`, and
`proofs/foundations/lindblad_spinor_coupled_construction.py` are
produced.

No commits performed; no remote push.
