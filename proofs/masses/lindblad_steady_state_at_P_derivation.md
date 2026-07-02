# Derivation: Lindblad steady state on the visible Bloch fibre at P

**NOTE (post-A3, 2026-04-18):** Under the three-axiom framework (A1+A2+A3; docs/framework/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 Theorem 25 (predictions/observer_hilbert_space.py). The Hilbert-space premise of Lindblad dynamics is therefore no longer assumed; the Lindblad-form derivation from A1+A2+A3 (vs adoption) remains a separate open workstream.

**Companion file:** `predictions/lindblad_steady_state_at_P.py`
**Status:** lemma (closed-form steady state on the visible 12-dim Bloch
fibre at the P-point of the srs Hashimoto walker; mass-scale and
bidirectional-flux trace identities at the h-eigenspace are exact rationals
in 1/k*).
**Companion scoping doc:** an internal working note (records
what does NOT close: the small-q dispersion of the dissipator's spectral
density is q-independent, hence does not move n_s off the |q|^2 stall of
an internal working note).

## Abstract

The framework's standard reading of axiom A1 (binary self-inverse toggle)
+ axiom A2 (MDL) at the level of the Hashimoto walker on the srs lattice
yields a single 12-dim Bloch fibre per primitive cell at the P-point
(`../../predictions/walker_dynamics_derivation.md` Steps 5-8;
`../../predictions/B_P_doubly_degenerate_h_derivation.md`). The previous attempts at
multiway formalisation (`docs/theorem_H_multiway_construction.md`,
internal working notes) tried to embed the dark
sector inside a single Hilbert space and stalled. The audit
an internal Markov-vs-unitary classification audit recommends an open-quantum-system
reading: the visible sector carries the Hilbert structure (Gleason on
the observer's compressed dynamics) and the dark sector enters as
decoherence channels (jump operators), in the standard Lindblad
1976 / Gorini-Kossakowski-Sudarshan 1976 framework.

This document constructs that Lindblad equation explicitly on the
visible Bloch fibre at P. The Hamiltonian is the Hermitian
symmetrisation of the Hashimoto operator; the jump operators are the 12
dephasing projectors onto directed edges, weighted by the W4
cancellation rate 1/k*. We derive in closed form:

1. The unique Lindblad steady state is rho_ss = I_12 / 12 (maximally
   mixed on the 12-dim fibre).
2. The steady-state population on the h-eigenspace is Tr(P_h rho_ss) =
   2/12 = 1/6.
3. The channel-summed jump rate at the h-eigenspace is m_h = 2/k* = 2/3.
4. The bidirectional flux on the h-eigenspace is Phi^bi_h = 4/k* = 4/3.

All four identities are exact rationals in 1/k*. They are the candidate
mass-as-flux readouts at the framework's only complex-Ramanujan-
saturated, C_3-protected Bloch eigenmode (h at P,
`../../predictions/B_P_doubly_degenerate_h_derivation.md`).

The companion scoping doc an internal working note
records that the small-q dispersion of the Lindblad dissipator's
spectral density is |q|^0 (q-independent), so this construction does
NOT move the cosmological n_s exponent off the |q|^2 stall of
an internal working note. It produces a clean
mass-scale identity but does not close the n_s problem.

## Framework axioms invoked

- **A1 (self-inverse toggle).** Each edge e of the srs primitive cell
  carries a toggle T_e with T_e * T_e = 1. (Used via the directed-edge
  basis of the Bloch fibre and via the W4 cancellation channel rate
  1/k* per step.)
- **A2 (MDL).** The observer encodes the toggle stream by its reduced
  word (Serre 1980 Trees §I.1). (Used via `../../predictions/walker_dynamics_derivation.md`
  Step 4 to derive the per-step W4 cancellation rate 1/k*.)

No further axioms. No physical observation enters.

## Derivation

### Step 1 -- Visible Bloch fibre at P

The srs Hashimoto walker has Bloch decomposition
`B = integral_BZ B(k) dk` (Sunada 2012 Topological Crystallography
§§5-6). At each k the fibre is the 2|E_primitive| = 12-dim space of
directed edges per primitive cell (`../../predictions/walker_dynamics_derivation.md`
Step 6, Step 8). At the P-point k = (1/4, 1/4, 1/4) the matrix B(P) is
the 12x12 explicit operator constructed in
`predictions/B_P_doubly_degenerate_h.py`. Its eigenvalues are
{h, h*, -h, -h*, +1, -1}, each with multiplicity 2, where
h = (sqrt(3) + i sqrt(5))/2 satisfies |h|^2 = k* - 1 = 2 (Ramanujan
bound saturated; `../../predictions/B_P_doubly_degenerate_h_derivation.md` Step 7).

### Step 2 -- Hamiltonian H on the fibre

B(P) is non-normal. To obtain a self-adjoint Hamiltonian that respects
the framework's amplitude reading of W3 (an internal Markov-vs-unitary classification audit
W3 row, "matrix identity is reading-invariant"), we take

    H := (B(P) + B(P)^dagger) / 2.

H is manifestly Hermitian. Numerically, ||H - H^dag|| = 0 to machine
precision (script log).

Alternative: H_log = -i log(U) where U is the unitary part of the
polar decomposition of B(P). H_log is also Hermitian but its spectrum
is logarithmic in the eigenvalues of B(P) and the connection to the
framework's spectral data is opaque. We take H_real because the
Hermitian symmetrisation preserves the algebraic spectral structure of
B(P) (its characteristic polynomial coefficients) and is the standard
Hermitian companion in open-quantum-system constructions
(Breuer-Petruccione 2002 §3.4, "anti-Hermitian shift").

### Step 3 -- Jump operators from W4 cancellation events

`../../predictions/walker_dynamics_derivation.md` Step 4 establishes that, under the
Jaynes-uniform NB-walk distribution, the per-step probability of the
next toggle being equal to the previous (a W4 cancellation event,
removing two edges from the reduced word) is exactly 1/k*. With k* = 3
on srs this is 1/3.

A W4 event at directed edge e erases the most-recent-edge information.
On the directed-edge fibre, this is a **dephasing** event localised at e:
the off-diagonal coherence between e and any other directed edge is
destroyed, while the population on e is unchanged. The standard
operator implementation is

    L_e := sqrt(1/k*) * P_e,

where P_e = |e><e| is the rank-1 projector onto the directed edge e.
We have one jump channel per directed edge, so 12 jump operators in
total. The prefactor sqrt(1/k*) ensures L_e^dag L_e = (1/k*) P_e and
hence

    sum_e L_e^dag L_e = (1/k*) sum_e P_e = (1/k*) I_12,

i.e. the dissipator is **unital** and probability-conserving (the total
cancellation rate per step, summed over all 12 directed edges, equals
12 * (1/k*) = 4 events of decoherence amplitude per unit time per
unit-trace state -- the relevant content is unitality, not the prefactor).

Remark on the alternative length-changing form. Encoding L_e as
a length-decreasing shift on a length-graded multiway Hilbert space
(L_e = sqrt(1/k*) sigma_- (x) P_e) is the natural F_inv(E) lift of
the cancellation event. But that route requires the multiway dimension
construction of `docs/theorem_H_multiway_construction.md`, which has
B_VD = 0 and gives a trivial Schur complement (visible dispersion
unchanged at gamma_phys = 1/16). The dephasing form of L_e is the
projection of that lift onto a single Bloch fibre and is what the
framework's "MDL erases the toggle sequence" interpretation directly
encodes.

### Step 4 -- Lindblad equation

With H from Step 2 and L_e from Step 3, the Lindblad equation
(Lindblad 1976, Gorini-Kossakowski-Sudarshan 1976; Breuer-Petruccione
2002 §3.2) reads

    d rho / dt = L(rho)
              := -i [H, rho]
                  + sum_e ( L_e rho L_e^dag - 0.5 {L_e^dag L_e, rho} ).

This generates a completely positive trace-preserving (CPTP) semigroup
on the 12 x 12 density matrices, by the GKS theorem (Gorini-Kossakowski-
Sudarshan 1976 Thm; Breuer-Petruccione 2002 Thm 3.2.5).

### Step 5 -- Steady state by unitality

For unital Lindbladians (sum_e L_e^dag L_e proportional to I), the
maximally mixed state is always a steady state (Wolf 2012 Quantum
Channels and Operations, Theorem 6.1; see also Breuer-Petruccione 2002
§3.2.4). Direct check:

    L(I/N) = -i [H, I/N] + sum_e (L_e (I/N) L_e^dag - 0.5 {L_e^dag L_e, I/N})
           = 0 - (1/N) sum_e (L_e^dag L_e - L_e L_e^dag)
           = 0,
    
where the first term vanishes because [H, I/N] = 0 and the second
because L_e L_e^dag = (1/k*) P_e = L_e^dag L_e.

**Uniqueness.** Vectorise the superoperator into a 144 x 144 complex
matrix (Roth's relation: vec(A X B) = (B^T (x) A) vec(X)) and compute
its singular spectrum. The smallest singular value is ~ 1.4e-15
(machine zero); the second-smallest is ~ 2.09e-1 (well-separated); the
null vector is exactly vec(I/12). Hence the unique Lindblad steady
state on the visible Bloch fibre at P is

    rho_ss = I_12 / 12.

### Step 6 -- Population on the h-eigenspace

Let P_h be the orthogonal projector onto the 2-dim h-eigenspace of B(P)
(constructed by QR-orthonormalising the two h-eigenvectors of B(P);
basis-independent since trace identities use Tr only). Tr(P_h) = 2.

Then

    Tr(P_h rho_ss) = Tr(P_h)/12 = 2/12 = 1/6.

This is the steady-state population (in the trace-class sense) of the
maximally-mixed state on the h-eigenspace. It is the framework's
analogue of "two-dimensional symmetry-protected channel out of twelve."

### Step 7 -- Channel-summed jump rate at the h-eigenspace

The jump rate of channel e at any subspace is Tr(L_e^dag L_e * P_h)
(the rate of dephasing events occurring within the projected state).
Summing over all 12 channels:

    m_h := sum_e Tr(L_e^dag L_e P_h)
        = (1/k*) sum_e Tr(P_e P_h)
        = (1/k*) sum_e (P_h)_{ee}
        = (1/k*) Tr(P_h)
        = 2/k*
        = 2/3.

The third equality uses sum_e (P_h)_{ee} = Tr(P_h) (the diagonal sum is
the trace, basis-independent identity). This is the candidate mass scale
of the Lindblad reading at the h-eigenmode. It is a *dimensionless* rate
in framework-internal units; converting to physical mass requires
Need B of an internal working note (Bloch-to-physical-units map),
which is open.

### Step 8 -- Bidirectional flux on the h-eigenspace

an internal working note defines the bidirectional
substrate flux as gain rate (D -> V) plus loss rate (V -> D). In our
unital dephasing Lindblad both directions have the same rate (gain =
loss for a steady state, by the detailed balance condition for unital
systems on a maximally mixed steady state; Breuer-Petruccione 2002
§3.4.3), so

    Phi^bi_h = 2 * m_h = 4/k* = 4/3.

This is the candidate mass-as-flux readout per the framing's
proportionality

    m_(k, alpha) ~ Phi^bi_(k, alpha)

at (k = P, alpha = h-channel).

### Numerical verification

The script `predictions/lindblad_steady_state_at_P.py` constructs the
12 x 12 matrix B(P) using the same I4_132 Wyckoff 8a primitive-cell
bond list as `predictions/B_P_doubly_degenerate_h.py`, the 144 x 144
vectorised Lindblad superoperator, and computes:

- ||H - H^dag|| = 0                 (Hermiticity, machine precision)
- ||sum_e L_e^dag L_e - (1/3) I|| = 0   (unitality, machine precision)
- smallest singular value of L_super = 1.4e-15  (rho_ss exists)
- second-smallest singular value     = 2.09e-1  (rho_ss unique)
- ||rho_ss - I/12|| = 1.8e-16         (closed form rho_ss = I/12)
- ||L(rho_ss)|| = 1.3e-16             (steady state by direct check)
- Tr(P_h rho_ss) = 0.166666...        (= 1/6)
- m_h            = 0.666666...        (= 2/3)
- Phi^bi_h       = 1.333333...        (= 4/3)

All four trace identities agree with the closed-form rationals 1/6,
2/3, 4/3 to machine precision.

## Result

Closed form on the visible Bloch fibre at P (12-dim, 2|E| = 12,
k* = 3, h-eigenspace mult 2):

    rho_ss = I_12 / 12                               (steady state)
    Tr(P_h rho_ss) = Tr(P_h) / 12 = 2/12 = 1/6       (population on h)
    m_h = 2 / k* = 2/3                                (mass scale on h)
    Phi^bi_h = 4 / k* = 4/3                           (bidirectional flux on h)

These are exact rationals in 1/k*. They depend only on
- k* = 3 (`predictions/k_star.py`),
- the multiplicity of h in B(P) (`predictions/B_P_doubly_degenerate_h.py`),
- the W4 cancellation rate 1/k* per step
  (`../../predictions/walker_dynamics_derivation.md` Step 4).

## Comparison with experiment

Not applicable. m_h and Phi^bi_h are dimensionless framework-internal
rates; converting to a particle mass in MeV requires the
Bloch-to-physical-units map (Need B of
an internal working note), which is open. No quantitative
prediction of an observed mass is made by this lemma.

## Open questions

1. **Bloch-to-physical-units map.** The constants 2/3 and 4/3 are
   dimensionless rates in lattice-step units. To compare with any
   observed particle mass requires fixing a physical time scale per
   walker step (Need B of an internal working note).

2. **Small-q dispersion of the dissipator (n_s readout).** The jump
   operators L_e = sqrt(1/k*) P_e are q-INDEPENDENT (they project onto
   directed-edge basis vectors, which are the same at every Bloch
   fibre). Hence the dissipator's spectral density at small q
   carries no |q| dependence. Under the standard cosmological
   identification <zeta(q) zeta(-q)> ~ |q|^{n_s - 4}, this yields
   n_s = 4 -- worse than the FDT-bridge stall n_s = 2 of
   an internal working note. The Lindblad
   construction does NOT supply the q-power needed to move n_s
   toward the observed 0.965. See
   an internal working note for the focused scoping of
   what does not close.

3. **Bidirectional flux interpretation.** The identification of the
   Lindblad mass-scale m_h = 2/k* with the framing's bidirectional
   flux Phi^bi_(k, alpha) of an internal working note
   requires the framing's proposed proportionality m ~ Phi^bi to hold
   structurally. The framing remains conjectural pending
   an internal working note Need A
   (multiway formalisation).

4. **Choice of H = H_real vs H = H_log.** Both produce Hermitian
   Hamiltonians on the 12-dim fibre. H_real (Hermitian part of B(P))
   preserves the algebraic spectral structure of B(P) and is the
   standard open-system companion. H_log (i log of polar unitary
   part) gives a different spectrum. The mass-scale identity m_h =
   2/k* is INSENSITIVE to this choice because the trace identity
   m_h = (1/k*) Tr(P_h) does not involve H. The choice affects only
   the unitary-evolution piece (-i [H, rho]) which vanishes on the
   maximally mixed steady state. So both H choices give the same
   rho_ss = I/12, the same population 1/6, the same m_h = 2/3, and
   the same Phi^bi_h = 4/3.

5. **Length-changing vs dephasing form of L_e.** The dephasing form
   used here projects the cancellation event onto a single Bloch
   fibre. The length-changing alternative (L_e on a length-graded
   multiway Hilbert space) requires the H_multiway construction
   (`docs/theorem_H_multiway_construction.md`), which has B_VD = 0
   and modifies neither the visible dispersion nor the steady state
   on a single fibre. The two forms give the same single-fibre result
   under the canonical Markov reading of F_inv(E).

## References

### Cited mathematical theorems

- **Lindblad, G.** (1976). On the generators of quantum dynamical
  semigroups. *Communications in Mathematical Physics* **48**, 119-130.
  Defines the Lindblad master equation and its generators.
- **Gorini, V., Kossakowski, A., Sudarshan, E.C.G.** (1976). Completely
  positive dynamical semigroups of N-level systems. *J. Math. Phys.*
  **17**, 821-825. Equivalent characterisation of CPTP semigroup
  generators in finite dimension.
- **Breuer, H.-P. & Petruccione, F.** (2002). *The Theory of Open
  Quantum Systems.* Oxford University Press. Ch. 3 (master equations,
  unitality, steady states, dephasing models).
- **Wolf, M.M.** (2012). *Quantum Channels and Operations: Guided
  Tour.* Available at https://www-m5.ma.tum.de/foswiki/pub/M5/Allgemeines/MichaelWolf/QChannelLecture.pdf .
  Theorem 6.1 (unital channels admit maximally mixed fixed point).
- **Sunada, T.** (2012). *Topological Crystallography.* Springer.
  §§5-6 (Bloch decomposition of periodic graphs).
- **Serre, J.-P.** (1980). *Trees.* Springer-Verlag. §I.1 (reduced
  words in the free involutive monoid).

### Upstream framework theorems (closed)

- `../../predictions/walker_dynamics_derivation.md` -- W1-W4: Hashimoto walker as
  observer's MDL-compressed dynamics; W4 cancellation rate 1/k*.
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` -- B(P) has h with
  multiplicity 2, C_3-protected.
- `predictions/k_star.py` -- k* = 3.
- `predictions/d_spatial.py` -- d_spatial = 3.
- `predictions/B_P_doubly_degenerate_h.py` -- 12-dim B(P) construction.

### Sibling open-system documents

- an internal Markov-vs-unitary classification audit -- audit recommending the open-
  system reading (the Lindblad bullet under "Markov / Lindblad
  route" in §5.I).
- `docs/theorem_H_multiway_construction.md` -- failed Markov H_multiway
  attempt; reusable dim-count lemma.
  attempt.
  stall; this Lindblad construction does NOT resolve it (see
  an internal working note).
  master-equation route (§5 Need-D scoping).

## Files referenced but NOT modified

Per task constraints: `results/parameters.csv`, `docs/parameters/derivations.md`,
B3/B6 docs, `../../predictions/walker_dynamics_derivation.md` (parallel agent
editing), and existing scoping docs are NOT edited. Only the two
predictions/lindblad_steady_state_at_P files plus the focused scoping
doc an internal working note are produced.

No commits performed; no remote push.
