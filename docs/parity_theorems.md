# CMB Parity Violation from the srs Vacuum

This document consolidates the parity-violation results established at
theorem grade for the chiral trivalent srs (Laves) lattice with space
group I4_1 32. It accompanies the P2 paper
"Cosmic parity violation from a chiral trivalent vacuum: a theorem
stack" and is the public-repo equivalent of docs/R_theorem.md for the
R = 228/7 result.

## Results

Four theorems + one strong conjecture (grade A-) control four CMB
parity-violating observables with zero free parameters.

    Theorem 1:  <(e . z^)^(2n)> = 1 / (3 * 2^(n-1))    (srs cubic moment)
    Theorem 2:  A = eps / k = 1/15 = 6.67%             (hemispherical)
    Theorem 3:  B(P) has h = (sqrt(3)+i sqrt(5))/2     (doubly degenerate,
                as a multiplicity-2 eigenvalue          C_3-protected)
    Theorem 4:  c_1(srs photon Hodge bundle) = 0       (trivial U(1)
                on every 2D slice of the BZ             topology)

    A- conjecture:
                beta = sin(arg h) * alpha_EM = 0.331 deg    (dark correction)

### Numerical matches

    A = 1/15 = 0.06667       obs: 0.065 +/- 0.02         0.08 sigma
    beta = 0.3306 deg        obs: 0.342 +/- 0.094 deg    0.12 sigma

Both use only quantities derived elsewhere in the framework
(eps = 1/5 from Bayesian update, h from the Ramanujan-saturated P-point
walk eigenvalue in P1). No adjustable parameters in either formula.

## Theorem 1: the srs cubic moment formula

    <(e . z^)^(2n)> = 1 / (3 * 2^(n-1))    for n >= 1, z^ any cubic axis

where the average is over the N_e = 24 directed edges of srs in the
conventional cubic unit cell.

**Proof.** By the 432 chiral cubic symmetry, the 24 directed edges split
into two subsets under projection onto any cubic axis: 8 edges lie
entirely in the plane perpendicular to z^ (contributing (e . z^)^2 = 0),
and 16 edges lie at 45 degrees to z^ (contributing (e . z^)^2 = 1/2).
This split follows from the body-diagonal orientation of the local C_3
axes and the equilateral-triangle geometry of edges in the plane
perpendicular to each C_3 axis. Therefore

    <(e . z^)^(2n)> = (16 * (1/2)^n) / 24 = 1 / (3 * 2^(n-1)).

For n = 1 this gives <(e . z^)^2> = 1/3 = 1/k, the rank-2 tensor
identity Sum_e e_a e_b = (N_e / 3) delta_ab of the 432 point group.
Numerical verification on the unit cell gives Sum_e e_a e_b = 8 * I
to machine precision.

Reference script: proofs/cosmology/A_dilution_derivation.py

## Theorem 2: A = 1/15 (hemispherical asymmetry)

    A = eps / k = 1/15 = 6.67%

where eps = 1/5 is the per-vertex Bayesian toggle asymmetry and
1/k = 1/3 is the n = 1 case of Theorem 1.

**Proof.** The CMB hemispherical power asymmetry is parameterized as
T(n^) = T_0 [1 + A (n^ . d^)] with observed A = 0.065 +/- 0.02. The
framework models it as

    A = eps * <(e . d^)^2>

where eps quantifies the per-vertex parity bias (a power-level quantity)
and <(e . d^)^2> = 1/k from Theorem 1 is the geometric weight that
couples per-vertex bias to the preferred-direction projection. The bias
eps is derived independently from a Bayesian Beta(1,1) -> Beta(2,1)
update on the toggle dynamics:

    P_create = 1/2,  P_persist = 1/3
    eps = (P_create - P_persist) / (P_create + P_persist) = 1/5

Combining: A = (1/5) * (1/3) = 1/15 = 0.06667. Match to observation:
0.08 sigma. Zero free parameters.

Reference script: proofs/cosmology/A_dilution_derivation.py

## Theorem 3: B(P) has h doubly degenerate, C_3-protected

Let B(k) be the Bloch non-backtracking walk operator on srs in the
4-vertex primitive cell, acting on the 12-dimensional directed-edge
space. At the P-point P = (1/4, 1/4, 1/4) of the primitive cubic BZ,
B(P) has

    h = (sqrt(3) + i sqrt(5)) / 2

as an eigenvalue of multiplicity exactly 2. The multiplicity is protected
by the C_3 stabilizer of P in the 432 point group.

**Proof sketch.** The scalar Bloch adjacency at P is

    A(P) = [[0, -i, -i, -i],
            [i,  0, -i,  i],
            [i,  i,  0, -i],
            [i, -i,  i,  0]]

with characteristic polynomial (lambda^2 - 3)^2. So A(P) has eigenvalues
+/- sqrt(3), each with multiplicity 2. The Ihara-Bass identity

    det(I - uB) = (1 - u^2)^(|E|-|V|) * det((1 + (k-1) u^2) I - u A)

for the srs primitive cell (|V|=4, |E|=6, k=3) becomes

    det(I - u B(P)) = (1 - u^2)^2 * det((1 + 2u^2) I - u A(P)).

Substituting A(P)'s characteristic factorization and solving gives the
walk eigenvalues mu = 1/u. The factor (2u^2 - sqrt(3) u + 1)^2 produces
walk eigenvalues {h, h*} each with multiplicity 2. The factor
(2u^2 + sqrt(3) u + 1)^2 produces {-h, -h*} each with multiplicity 2.
The prefactor (1 - u^2)^2 contributes eigenvalues +/- 1 each with
multiplicity 2. Total: 12 = 2|E|, matching the directed-edge dimension.

The C_3 element rotating by 120 degrees about the body diagonal
(1, 1, 1) fixes P and induces a vertex permutation that commutes with
A(P). The +/- sqrt(3) eigenspaces are therefore C_3-invariant
2-dimensional subspaces. A small C_3-preserving perturbation cannot
split the 2-fold degeneracy, and the Ihara-Bass translation transfers
this protection to the h-eigenspace of B(P). QED.

### Uniqueness at P

Among the high-symmetry points of the cubic BZ, only Gamma and P have
C_3 stabilizer. At Gamma the triplet A-eigenvalue -1 gives a complex
walk eigenvalue (-1 + i sqrt(7))/2 with |mu|^2 = 2 (Ramanujan-
saturated), but with multiplicity 3 reflecting the triplet A-eigenvalue.
At P, the walk eigenvalue h = (sqrt(3) + i sqrt(5))/2 has multiplicity
exactly 2. P is the unique high-symmetry point with a doubly-degenerate
complex walk eigenvalue at the Ramanujan bound. This singles out h as
the framework's canonical walk eigenvalue.

Reference script: proofs/cosmology/parity/srs_photon_bloch_primitive.py

## Theorem 4: c_1 = 0 on all photon-bundle slices

Let d(k): C^0 -> C^1 be the Bloch incidence matrix on the srs primitive
cell. Define the photon Hodge bundle as the transverse subspace
ker d^dag(k) (divergence-free 1-forms). Then for all k_l in [0, 1) and
any choice of axes (i, j, l):

    c_1^(ij)(k_l) = 0

i.e. the first Chern number on every 2D slice of the Brillouin zone
vanishes. The U(1) Abelian topology of the photon bundle is trivial.

**Proof.** The Bloch incidence matrix satisfies a generalized time
reversal symmetry d(-k) = d(k)* (element-wise complex conjugate). This
is verified symbolically via the scalar Laplacian identity A(-k) = A(k)*
with d^dag d = k * I - A. Under k -> -k, the photon Hodge bundle at -k
is the complex conjugate of the bundle at k. Complex conjugation flips
the Chern integer: c_1(bundle*) = -c_1(bundle). A 2D slice at fixed k_l
is mapped to the slice at -k_l by this symmetry. At the self-conjugate
slices k_l in {0, 1/2}, the bundle equals its own complex conjugate,
so c_1 = -c_1 forces c_1 = 0.

Between self-conjugate slices c_1(k_l) is piecewise constant (integer-
valued, changes only at topological band crossings). The photon bundle
has no band crossings of the charged type: Theorem 3 shows the
degeneracy at P is symmetry-protected (not accidental), and the single
rank-drop of the bundle occurs at Gamma where dim ker d^dag(Gamma) = 3
instead of 2. Direct sphere integration of the Berry curvature on a
small sphere around Gamma gives c_1^sphere -> 0 as the grid is refined,
demonstrating that the Gamma defect carries zero U(1) charge. Therefore
c_1(k_l) has no step jumps and remains 0 throughout. QED.

### Consequence for cosmic birefringence

A bulk axion angle theta * F_mu_nu tilde F^mu_nu on a vacuum with
c_1 = 0 everywhere contributes zero to cosmic birefringence (the term
is a total derivative in a homogeneous bulk with no boundaries).
Therefore any nonzero cosmic birefringence on the srs vacuum must come
from a dynamical mechanism, not from a topological bulk theta.

Reference scripts:
  proofs/cosmology/parity/srs_photon_berry.py          (slice Chern)
  proofs/cosmology/parity/srs_gamma_defect_charge.py   (sphere integration)

## Cosmic birefringence beta at A-

Theorem 4 forces beta to be dynamical. The framework has an established
dark-correction pattern from the neutrino sector: delocalized observables
receive corrections controlled by the chirality content of h, with two
distinct chirality invariants depending on whether the observable lives
at the amplitude level (first order in h) or the mass-squared level
(second order).

    Amplitude (first-order):  chi(h) = sin(arg h) = Im(h) / |h|
    Mass-squared (2nd-order): xi(h)  = tan^2(arg h) = Im^2(h) / Re^2(h)

For h = (sqrt(3) + i sqrt(5))/2:

    sin(arg h) = sqrt(5/8) ~ 0.7906
    tan^2(arg h) = 5/3 ~ 1.667

The tan^2(arg h) = 5/3 form controls delocalized mass-squared
observables (neutrino masses, PMNS theta_23). The sin(arg h) = sqrt(5/8)
form controls delocalized amplitude observables. Photon polarization
rotation is an amplitude observable, so the framework's dark correction
pattern predicts

    beta = sin(arg h) * alpha_EM = sqrt(5/8) / 137.036 = 0.3306 deg

Match to Eskilt 2022 (beta_obs = 0.342 +/- 0.094 deg): 0.12 sigma.

### Grade: A-, not theorem

This derivation rests on one framework-level assertion: that delocalized
amplitude observables receive dark corrections linear in sin(arg h) via
the standard QED coupling alpha_EM. This rule is the amplitude analog
of the tan^2(arg h) rule used for delocalized mass-squared observables.
It is empirically supported by the neutrino sector and structurally
consistent with Theorem 4 (no topological source is available, so any
beta must be dynamical, and the amplitude-linear form is the only
dimensionally consistent chirality invariant at leading order in
alpha_EM). A rigorous first-principles derivation from the walk
operator's one-loop contribution to the photon self-energy has not yet
been carried out, so we grade beta at A- rather than theorem.

### Hard cap

Since sin(arg h) is bounded by 1 in magnitude, the framework predicts

    |beta| <= alpha_EM ~ 0.418 deg

as a hard observational cap. Any future measurement exceeding this bound
would falsify Theorem 4's identification of beta as a dynamical
dark-correction observable. The framework currently sits at 79.1% of
this bound.

Reference script: proofs/cosmology/path_c_beta_verify.py

## Per-ell extensions

Theorem 1 extends to per-ell Legendre projections for the CMB parity
anomaly and parity-odd trispectrum. The framework predicts specific
rational values:

    <P_2> = 0                  (reverse: theorem via 3*(1/k) - 1 = 0 for k=3)
    <P_4> = -7/48 ~ -0.146
    <P_6> = -13/64 ~ -0.203
    <P_8> = +297/1024 ~ +0.290

and so on for higher even ell. The Land-Magueijo-style odd/even power
ratio R(ell_max ~ 28) ~ 1.002 from these amplitudes, in the right
direction compared to observed ~ few percent but underpredicting by
about an order of magnitude at current precision.

## Framework status and falsification

The parity-violation sector now has:

    - Theorem 1 (srs cubic moment)           verified
    - Theorem 2 (A = 1/15)                   verified (0.08 sigma)
    - Theorem 3 (B(P) doubly-degenerate h)   verified
    - Theorem 4 (c_1 = 0 on slices)          verified
    - beta = sin(arg h) * alpha_EM           A- grade (0.12 sigma)

controlled by two independent chirality couplings (never combined):

    eps = 1/5                from Bayesian toggle update
    chi(h) = sqrt(5/8)       from Hashimoto walk eigenvalue at P

Four CMB parity observables predicted, zero adjustable parameters.

### Falsification criteria

    1. |beta| > alpha_EM ~ 0.418 deg           --> falsifies Theorem 4
    2. A != 1/15 at future high precision       --> falsifies Theorem 2
    3. theta_23 != non-maximal with the         --> falsifies dark correction pattern
       tan^2(arg h) split
    4. Per-ell amplitudes != predicted         --> falsifies Theorem 1
       rational values at high precision

## Verification

    python3 proofs/cosmology/A_dilution_derivation.py      (Theorems 1, 2)
    python3 proofs/cosmology/path_c_beta_verify.py         (beta A-)
    python3 proofs/cosmology/parity/srs_photon_berry.py    (Theorem 4 slice Chern)
    python3 proofs/cosmology/parity/srs_gamma_defect_charge.py  (Theorem 4 Gamma charge)

## Key identities

    eps = (P_create - P_persist) / (P_create + P_persist) = 1/5
    Sum_e e_a e_b = (N_e / 3) delta_ab     (rank-2 tensor identity)
    <(e . z^)^(2n)> = 1 / (3 * 2^(n-1))    (srs cubic moment formula)
    A(P) char poly = (lambda^2 - 3)^2       (double eigenvalue +/- sqrt(3))
    |h|^2 = k - 1 = 2                       (Ramanujan saturation)
    sin(arg h) = sqrt(5/8) = Im(h)/|h|      (amplitude chirality)
    tan^2(arg h) = 5/3 = Im^2(h)/Re^2(h)    (mass^2 chirality)
    c_1 = 0 on all 2D slices                (trivial U(1) photon topology)
