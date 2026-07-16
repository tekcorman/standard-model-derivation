# Incomplete-equations to-do — the top-down completeness audit

## The law (why this file exists)
Every quantity in this framework is a **forced read of the complete spectrum of the one object**
(`D = B(srs ⊗ srs-z) ⊗ ∂_N`). Masses, couplings, mixings are **not targets to fit** — they *are* the spectrum.

**If a value does not fall out of the spectrum top-down, the DEFINING EQUATION OF THE OBJECT is incomplete
at that point.** The task is *never* to fit the residual or pattern-match a constant to a target (that is
numerology, and it is forbidden). The task is to (1) **locate the incompleteness**, (2) **add it here**, and
(3) **chase the math up to the complete top-level equation** — exactly as every closed result in this repo
was obtained. Think in **spectra and forced objects**, never in "mass" or "target parameter."

The question for any quantity is: *what is the equation whose solution it is, and is that equation complete?*

## ★ QUEUED FRESH STATIONS (2026-07-14 session residue) — pre-reg-ready work orders
Three scoped, dual-outcome station specs → internal research notes
(roadmap §3 SESSION-2026-07-14 block): **GEN-HOMES** (reconcile the 3 unreconciled generation
homes — C₃-windings [live] / Weyl m=−1,0,+1 bands / observer ℂ³_gen; the 2026-06-29 soft spot),
**CLK-GAP** (provenance + possible booking of β_gas = 8·ln(3/2) = 8× the edge-surprise gap, tying
S_fresh/S_disconfirm to the Layer-2 clock and the 16/15 root), **IDW-CvC** (is the orientation-
mechanism refutation a CLASS fact or a CONVENTION artifact — full goal-seek-guarded freeze, NO
−70 ppm comparison). DO-NOT-RETREAD: mirror-orientation generation splitting is REFUTED
(`build_dN_2026-06-30.md`: J-reality break + 565×; no-go theorem). None booked-forced without its
own freeze + verification.

## ★ GEN-IDENT-B (2026-07-15) — THE OBSERVER↔SUBSTRATE VERTEX COUPLING IS UN-BUILT
Freeze internal research notes; return
internal research notes; check
`proofs/foundations/genident_B_observer_residual_check_2026-07-15.py` (22/22 PASS). **THE NAMED
INCOMPLETE EQUATION:** the framework has never built a real, callable coupling between the observer
factor `ℂ³_obs` (`predictions/R3_observer_c3_generation.py`'s tensor factor / M1.B's
`M ⋊_α ℤ₃ ≅ M₃(ℂ) ⊗ M^α` crossed-product construction, `proofs/foundations/m1b_*.py`) and the vertex
functional `−κ·I(A;B)` (V1, `derivation_topdown/state/the_net.py` §11, operating on `H_hist ⊗ F`).
Source-level check: zero imports either direction, zero shared names, and `the_net.py:8769-8771`
carries an EXISTING standing disclaimer explicitly forbidding the conflation ("triad↔generations is
never in the verdict path"). GEN-IDENT-A (`docs/theorems/genident_A_offset_forced_2026-07-15.md`)
proved the substrate carries a FORCED rigid relative orientation between the winding axis `σ` and
the vertex-selected axis `W`; GEN-HOMES's mechanism scout showed that IF an observer were forced to
respect both, the joint commutant collapses by Schur (`⟨σ,W⟩=A4` irreducible on `ρ₃`, commutant
dim 3→1), killing the observer's continuous `U(1)²` label-freedom — but this is COUNTERFACTUAL: no
code path currently imposes `W` (or anything from the vertex) on `ℂ³_obs`. **Building that coupling
(a genuine numeric tie from `H_hist ⊗ F` to the observer's `ℂ³` factor, not a toy/symbolic one) is
the next construction target** if the ppm-identification wall is ever to be attacked via this route.
Discrete residual RESOLVED by the verification (calibrated: σ-alone reproduces the S₃ baseline
order 6): under a built collapse the surviving discrete labeling freedom is `Out(A4)≅ℤ₂` (order 2,
a SINGLE BIT), NOT the full `S₃` — the joint σ∧W normalizer-up-to-power leaves exactly the identity
plus one outer order-2 element (swaps σ's two nontrivial eigenspaces, fixes the identity-eigenspace).
So the counterfactual is sharp: building the coupling would reduce the entire generation-label
freedom to one binary datum — the minimal no-go input. NO mass/ppm/Koide/CKM/PMNS value entered this station's own
computation (goal-seek guard honored, verified by an explicit traced-source circularity check).

## ★ GEN-IDENT-β (2026-07-15, adjudicated 2026-07-16) — THE RUN-ENDPOINT `s` IS A ONE-BODY PHASE, INVISIBLE TO THE VERTEX (BLIND-BY-THEOREM; the DYNAMICAL route to pinning `s` is CLOSED)
Freeze internal research notes; return
internal research notes; verification `..._verification_2026-07-15.md`
(CONCUR); driver `proofs/foundations/genident_beta_endpoint_vertex_check_2026-07-15.py` (19/19 PASS); verdict
`docs/theorems/genident_beta_endpoint_vertex_2026-07-15.md`; accretion `beta_endpoint_vertex_read` in
`derivation_topdown/state/the_net.py`. **THE EQUATION THAT STAYS INCOMPLETE:** the generation labeling
(which winding-isotype = e/μ/τ) and the −70/−60.5 ppm subleading per-rep MAGNITUDE. Route β tested whether a
FORCED substrate functional pins the run-endpoint `s` (= the one free datum, `δ=φ·s`) spontaneously. Its
literal forms were already refuted in committed code (`explore_t12_observer_position.py` self-consistency +
MDL waterline; `gap_fixes_s_scratch.py` NJL; `endpoint_search` kinematic scan). The one untested object — the
substrate vertex `−κ·I(A;B)(s)` between the forced C₃-winding sectors on the `Λ•(ℂ³)=(4,2,2)` carrier — is
**EXACTLY CONSTANT in `s` for a structural reason**: `s` enters `c(s)=(1,e^{+iφs},e^{−iφs})` purely as a
per-mode phase ⟹ `|Ψ(s)⟩=U(s)|Ψ(0)⟩` with `U(s)=exp(iφs N_{ω¹})exp(−iφs N_{ω²})` a LOCAL (single-mode)
unitary (generator = a sum of single-mode number operators, local for EVERY mode cut) ⟹ bipartite
entanglement/mutual-information is invariant ⟹ no vertex functional can see `s`. NOT vacuity (entanglement
0.6–2.0 bits present; distinct mechanism — PHASE-LOCALITY — from GEN-IDENT-D2-leg-2's product-state `I≡0`).
verification killed the bipartition-triviality attack (flipping the phase-carrying mode `ω¹` onto side A still
gives exactly-constant entropy — Schmidt magnitudes are insensitive to a local phase regardless of which party
holds it; (4,2,2)/parity gradings are direct sums not tensor factorizations ⟹ no fourth forced cut). **NET:
`s` is a one-body/gauge-like phase, provably invisible to any bipartite substrate functional ⟹ the DYNAMICAL
route to the labeling is closed, joining the KINEMATIC closure (GEN-IDENT-D0→D2). THE BOUNDARY THEOREM is
earned: `s` is irreducibly the framework's one free Cauchy axis** (this + `explore_t12` + `gap_fixes_s` +
`endpoint_search` + the no-go). **Parameter impact NONE on −70 ppm / e-μ-τ** (goal-seek clean). REOPENER
(booked, not a loophole): a NON-bipartite one-body observable of the absolute winding phase could resolve `s`
— but that IS the single external observer datum the no-go requires; it would supply, not derive, and is a
NEW freeze. **The remaining number-moving generation target is now cleanly the ppm MAGNITUDE equation** (the
subleading per-rep correction), SEPARATE from the now-theorem-bounded labeling.

## ★ GEN-IDENT-C (2026-07-15) — THE OBSERVER FACTOR C³_obs HAS NO FORCED NUMERIC HOME ON THE CARRIER
Freeze internal research notes; return
internal research notes; check
`proofs/foundations/genident_C_coupling_check_2026-07-15.py` (24/24 PASS); **SEALED-CONCURRED +
BOOKED (architect adjudication 2026-07-15)** — verification `genident_C_verification_2026-07-15.md` /
`.py` (38/38 PASS) reproduced both load-bearing facts with fresh code (a sharper deterministic
non-uniqueness, cosines {0.7331, 0, 0}), ran FIVE independent attempts to force a unique anchor (all
failed: W-selection excludes every valid anchor by Schur; the sealed charge-conjugation K gives only
a real-dim-4 fixed set; two data-free functionals converge on the same forced-but-insufficient 2-dim
{vacuum, top-wedge} subspace; the moduli space = orbit of U(M^σ)≅U(4)×U(2)×U(2) with no fixed point),
and UPGRADED the Skolem–Noether obstruction from "supporting" to LOAD-BEARING. **THE NAMED INCOMPLETE
EQUATION, one level deeper than GEN-IDENT-B:**
before the observer↔vertex coupling GEN-IDENT-B named can even be attempted, `ℂ³_obs` itself needs a
FORCED numeric realization on the actual truncated `H_hist ⊗ F` carrier (M1.B's `M ⋊_α ℤ₃ ≅
M₃(ℂ) ⊗ M^α` crossed product has only ever been symbolic/sympy, never numerically instantiated on
the carrier; R3's ℂ³ is abstract/never-on-carrier). This station attempted it and found the
construction UNDERDETERMINED: (1) the ONLY carrier-tied σ-structure that already exists (F's level-1
subspace, `_a2c_level_rep(1)`) IS the substrate's own `ρ₃` exactly (residual 0.0) — using it would be
the forbidden C1-b identification; (2) granting the most-forced possible alternative (the exterior-
power tower lift `U_F` of σ onto ALL of F, order 3, self-consistent with `ρ₃` on level 1), the
crossed-product/matrix-unit recipe M1.B.c specifies does NOT single out a unique M₃(ℂ) block once
the fixed-point algebra `M^σ` is genuinely non-scalar (dim 24 on the real carrier — M1.B.c's own
script only ever verified its recipe in the degenerate toy case `M^α → ℂ`, which trivializes the
claim). Built out constructively: the candidate home is parametrized by a continuous moduli space (a
unit vector in each of `U_F`'s three eigenspaces, dims {4,2,2}); two independent, equally legitimate
anchor choices give PROVABLY DIFFERENT 3-dim subspaces of F (principal cosines {0.878, 0.852, 0.158},
not all 1). **Building a genuine forced M₃(ℂ) home for `ℂ³_obs` on the finite carrier — resolving
which point in this moduli space is not arbitrary, or showing the base algebra must instead be some
proper (non-full-matrix) subalgebra on which σ is genuinely OUTER (Skolem-Noether rules this out for
any full `B(H)` on a finite carrier) — is the next construction target upstream of GEN-IDENT-B's
coupling.** Consequence: C2 (the observer–vertex mediation test) is MOOT — there is no non-arbitrary
`ℂ³_obs` to couple. The ppm-identification wall via the observer-coupling route is now precisely
located one level deeper than GEN-IDENT-B found it: not "the coupling is un-built" but "the coupled
object's own carrier-realization is itself unforced." Route β (dynamical pin via a run fixed-point)
is untouched by this finding. NO mass/ppm/Koide/mass-ordering/CKM/PMNS value entered this station's
computation (goal-seek guard honored; explicit traced-dependency-chain circularity check, zero hits;
`m1b_c_basis_match.py` — whose lines 280–285 fix a label using Koide/mass-ordering — was never
imported).

## ★ GEN-IDENT-D (2026-07-15) — GATE D0: α IS PROPERLY OUTER ON THE TYPE-II₁ FACTOR M=L(F_inv(6))
Freeze internal research notes; theorem write-up
`docs/theorems/genident_D_outerness_2026-07-15.md`; return
internal research notes; driver
`proofs/foundations/genident_D_outerness_check_2026-07-15.py` (49/49 PASS, pure integer free-product
combinatorics, no numpy/sympy). **NOT YET verified — implementation pass output only, nothing booked.**
**VERDICT LEAF: D0 = OUTER-CONFIRMED.** GEN-IDENT-C proved the observer factor `ℂ³_obs` has no forced
numeric home on the FINITE `H_hist ⊗ F` carrier (Skolem–Noether: every automorphism of a full finite
matrix algebra is inner, so no properly-outer action, so no canonical `M₃(ℂ)` split). This station
tested whether that obstruction survives on the type-II₁ factor `M = L(F_inv(6)) ≅ L(𝔽₄)` where M1.B
always claimed the observer's Galois tower lived, by rigorously deciding gate D0: is the winding-C₃
automorphism `α` (induced by `σ=(1 2 3)(4 5 6)` on the 6 free `ℤ/2` generators) PROPERLY OUTER on `M`?
**Answer: yes, non-vacuously.** Four links, all closed self-contained: (i) `G=F_inv(6)` is ICC (proved
via exact word-length-growth under conjugation, `4n+|g|`) ⟹ `M` is a genuine II₁ factor; (ii) `σ̂`
moves `t_1↦t_2`, generators of distinct free factors, which are never conjugate ⟹ `σ̂` not inner on
`G`; (iii) — **the load-bearing link M1.B skipped** (its own "outer" claim rested on a MISAPPLIED
citation to Voiculescu's `Out(F_n)↪Out(L(F_n))` GROUP-embedding result, which says nothing about this
specific automorphism) — proved SELF-CONTAINED via a σ̂-twisted-conjugation/Fourier-coefficient
argument: every σ̂-twisted conjugacy class (including the identity's own, which needs and uses `σ`
being FIXED-POINT-FREE on all 6 generators) is infinite ⟹ no unitary `u∈ℓ²(G)` can implement `α` ⟹ `α`
not inner ⟹ (on a factor, not-inner ⟺ properly outer) properly outer; (iv) the identical argument for
`σ²` (also fixed-point-free) ⟹ both nontrivial `ℤ₃` elements properly outer ⟹ the action is FREE ⟹ the
standard Jones/Goodman–de la Harpe–Jones/Connes–Takesaki Galois theorem legitimately applies,
`M ⋊_α ℤ₃ ≅ M₃(ℂ) ⊗ M^α` with a CANONICAL `M₃(ℂ)` leg (no moduli freedom — the exact opposite of
GEN-IDENT-C's finite-carrier `U(4)×U(2)×U(2)` disease, because that disease was specifically a
consequence of the action being INNER there). **THE NAMED NEXT CONSTRUCTION TARGET (D1/D2, explicitly
NOT run by this gate-first station):** D1 must instantiate the `M₃(ℂ)` leg enough to pose D2 and prove
it carries no residual moduli (expected from outerness, not yet built); D2 is the genuinely new
cross-level mediation test between the canonical `M₃(ℂ)` (living on `M`, type II₁) and the vertex's
`W`-carrier (living on the finite `H_hist ⊗ F`) — a coupling between two different levels, not a
same-space contraction, and not assumed to be easy just because D0 landed. **Honest bound: D0 does NOT
label e/μ/τ, does NOT derive −70 ppm, and does not even touch whether the vertex forces `W` onto the
canonical home** — it only establishes that a canonical home CAN exist where GEN-IDENT-C proved it
could not on the finite carrier. NO mass/ppm/Koide/mass-ordering/CKM/PMNS value entered this station's
construction (goal-seek guard honored; driver's AST-based self-scan — not a self-referential substring
search — confirms zero physics-codebase imports and zero floating-point literals anywhere in the code).

## ★ GEN-IDENT-D1 (2026-07-15) — THE CANONICAL M₃(ℂ) HOME HAS NO RESIDUAL MODULI
Freeze internal research notes §2 "D1"; theorem write-up
`docs/theorems/genident_D1_canonical_home_2026-07-15.md`; return
internal research notes; driver
`proofs/foundations/genident_D1_canonical_home_check_2026-07-15.py` (33/33 PASS, exact sympy
algebraic arithmetic, no floating point). **NOT YET verified — implementation pass output only,
nothing booked.** Builds on the sealed D0 = OUTER-CONFIRMED (α properly outer on `M=L(F_inv(6))`).
**VERDICT LEAF: D1 PINS THE CANONICAL M₃(ℂ) — relative commutant `M' ∩ (M ⋊_α ℤ₃) = ℂ`, no moduli.**
Proven self-contained via a new short lemma (a finite factor + a nonzero element `w` solving
`x w = w α(x) ∀x` forces `α` inner, using that isometries in finite von Neumann algebras are
unitaries) applied twice, once per nontrivial `ℤ₃` grade, using D0-iii/iv's properly-outer results
as the contradiction hypothesis. **The exact place GEN-IDENT-C's `U(4)×U(2)×U(2)` (dim 24) moduli
fails to reappear, pinpointed:** GEN-IDENT-C's count came from the finite-carrier lift being INNER,
so its eigenspaces were bare unconstrained vector spaces (full unitary group `U(d_i)` available per
eigenspace); here, outerness forces each nontrivial ℤ₃-graded piece `Mu^n` to be a RANK-1
`M^α`-bimodule (the standard index-`|G|` fact for free actions), so there is no eigenspace of
dimension `>1` left to over-count — the commutant of `u` inside `M⋊ℤ₃` is computed directly and
equals exactly `M^α` (not an inflated matrix-block algebra). A second, independent, purely-arithmetic
confirmation: in a finite toy forced inner by Skolem–Noether, `dim(M⋊ℤ₃)=12` definitionally but the
clean tensor formula would need `18` — a flat dimensional impossibility, reinforcing that outerness
is structurally necessary, not merely nicer. Koide/DFT-stripped throughout (verified: AST self-scan,
zero physics-codebase imports, zero float literals; `m1b_c_basis_match.py` never referenced). **THE
NAMED NEXT CONSTRUCTION TARGET (D2, explicitly NOT run by this station):** the cross-level mediation
test between this canonical `M₃(ℂ)` home (living on `M`, type II₁) and the vertex's `W`-carrier
(living on the finite `H_hist⊗F`) — not assumed to be easy just because D1 landed. **Honest bound:
D1 does NOT label e/μ/τ, does NOT derive −70 ppm, and does NOT put `W` on the canonical home** — it
only establishes the home itself has zero residual moduli. **Rigor flagged as calibrated rather than
fully re-derived from first principles (does not appear to affect the verdict, but named for the
verifier):** the crossed product's "unique Fourier expansion" fact is used as standard/
definitional (verified concretely in the driver's covariant representation) rather than re-derived
from the von Neumann algebra axioms for the actual infinite-dimensional `M`; the rank-1-bimodule
explanation (the specific pinpoint of where GEN-IDENT-C's count fails) is stated as the standard
content of the free-action Galois theorem rather than proven line-by-line from bimodule first
principles — the verdict itself rests entirely on the self-contained §1 lemma proof, which does not
depend on this explanatory scaffolding.

## ★ GEN-IDENT-D2 (2026-07-15) — CLOSED: D2 = ORTHOGONAL-FORCED (the vertex does NOT put W on the home)
**Verdict doc `docs/theorems/genident_D2_orthogonal_verdict_2026-07-15.md`.** BOTH legs sealed ⟹ the
GEN-IDENT type-II₁ route is COMPLETE and CLOSED. **Leg 2 (adversarially sealed CONCUR):** no forced
vertex-mediated coupling — an independent try-hard-to-BUILD pass found every route collapses
(honest-crossed-product = arbitrary embedding into atomless II₁ M, or the (D)-trap; F_inv(6)↔walk =
forced (3,3) cycle-type match BUT no forced ∗-embedding between `L²(M,τ)` and `H_hist⊗F`; τ-GNS shadow
= maximally-mixed `I/3`, U(3)-blind; product state → I≡0 vacuous). Anchor driver
`genident_D2_leg2_no_forced_coupling_check_2026-07-15.py` (12/12); return
`working notes/GEN_IDENT_D2_leg2_verification_2026-07-15.md`. **ONE REOPENER (booked honest caveat):**
a future FORCED (not chosen) ∗-embedding `F↪M` via a canonical free-product projection would reopen
leg 2 — no candidate found (II₁ has no minimal projections), judged exhausted. **HAND TO ROUTE β**
(dynamical pin of run-endpoint s). **Parameter impact: NONE on −70 ppm / e-μ-τ** (the ℤ₂ Schur collapse
is NOT triggered via this route; labeling stays external, magnitude stays a separate incomplete eq).
Leg 1 detail ↓.

### GEN-IDENT-D2 LEG 1 (2026-07-15) — W DOES NOT DESCEND TO THE CANONICAL M₃(ℂ) HOME (SEALED)
Theorem write-up `docs/theorems/genident_D2_leg1_W_no_descent_2026-07-15.md`; verification return
internal research notes; driver
`proofs/foundations/genident_D2_half_descent_check_2026-07-15.py` (32/32 PASS, pure integer
combinatorics + exact free-product word arithmetic, no float). **SEALED-CONCURRED (CONCUR-WITH-
CORRECTION; both corrections applied).** Builds on sealed A/D0/D1.
**VERDICT LEAF: W does NOT descend to an automorphism of `M⋊_α ℤ₃`; the canonical M₃(ℂ) home
carries σ but has NO forced/canonical W-action.** Mechanism: `⟨σ⟩` is a **self-normalizing
Sylow-3 of A4** (`N_{A4}(⟨σ⟩)=⟨σ⟩`, `n₃=4`), and `⟨σ,W⟩=A4`, so `W σ W⁻¹ ∉ ⟨σ⟩` — W fails the
crossed-product descent criterion (`β` descends ⟺ `[βαβ⁻¹]∈⟨[α]⟩` in `Out(M)`). Transferred to
`M=L(F_inv(6))` via the **Generalized-Outerness Lemma** (proved self-contained: ANY nontrivial
generator-permutation induces a properly-outer automorphism — D0-iii's twisted-conjugacy/word-length
technique, fixed-point-freeness shown to be a convenience not a requirement; anchored in-driver for
the exact `τ_k=(WσW⁻¹)σ⁻ᵏ`, k=0,1,2, incl. τ₁'s two fixed points). Discriminates (only ⟨σ⟩'s 3
elements descend; σ,σ² positive controls). **This closes the AUTOMORPHISM route to D2=BUILT.**
**HONEST SCOPING:** denies only the *forced/functorial* W-action; an *unforced* inner unitary
`ρ₃(W)` on `M₃(ℂ)=B(ℂ³)` still exists (Skolem–Noether) — leg 1 says nothing forces it.
**LEG 2 (STILL OPEN, held for user reassessment):** whether a *vertex-mediated non-automorphism*
coupling `−κ·I(A;B)` across the level gap could transmit W. The D2-a construction sweep found it
**un-posable without the (D)-trap** (F's level-1 ≡ substrate ρ₃) or an unforced gluing (smuggled
datum) — i.e. no forced coupling; the un-built object is a forced `F_inv(6)↔walk` homomorphism
compatible with both σ-realizations. **Leg 1 + leg 2 = D2 ORTHOGONAL** (labeling external via the
type-II₁ route; hand to Route β). Leg 1 alone = the canonical route puts no W on the home.
**Parameter impact: none on −70 ppm / e/μ/τ** — the one-bit Schur collapse (GEN-IDENT-B) is NOT
triggered via this route.

## THE DEPENDENCY MAP (READ FIRST — added 2026-07-06)

The numbered items below are historically listed miss-by-miss, which hides the real structure:
**most of the open misses are not independent — they share a small number of GATES.** Attacking a
miss without knowing which gate it sits behind is how the program lost strategic direction (it drilled
one gate's worst face — §1 — for ~15 routes while banking grade, exactly the "keep drilling one number"
path `state_of_the_theory_and_strategy_2026-07-02_EOD.md` §6 told us to stop). The gates:

**THE M-TRACK — the missing STATE layer ω (2026-07-07). Strategic overlay: the framework built the
OPERATOR D; physics is (D, ω). The remaining wall inventory maps onto the state's three canonical
structures — M0 modular flow (κ, dynamical ∂_N), M1 sector category (−70 ppm keystone, B1 species),
M2 KMS thermodynamics (θ_*/Y_p/n_s cluster). See internal research notes
+ handoff internal research notes.
  • M0 (κ station) STARTED 2026-07-07 (a model), pre-reg blind `M0_modular_hamiltonian_kappa_prereg_2026-07-07.md`,
    probes `proofs/foundations/M0_convention_control_2026-07-07.py` + `M0_modular_hamiltonian_kappa_2026-07-07.py`,
    verify 65/65. BUILT: the one-bit vacuum's modular Hamiltonian at cell level (C=(I+iJ6)/2 = exact rank-3
    projector; S_ent bit-even). H-BIT PARTIAL-EXACT: the bit σ=J→−J is EXACT particle-hole C(−J)=I−C(J)
    ⟹ K_A→−K_A = the flow-reversing half of the modular conjugation. **κ STILL WALLED/OPEN** — one cell = one
    ΔS value (no slope); the naive S_ent=ln2·L bridge is too crude (MI≥0 vs signed ΔS); κ must go via the
    modular ENERGY (first law δ⟨K⟩). ~~CONTINUATION M0-2 supercell~~ **SUPERSEDED same day (architect): the
    supercell route is DEPRECATED for κ** (MI≥0 vs signed ΔS; J6 ≠ i·sign(D); a spatial slice is pure —
    temperature lives in the state of HISTORY). **⟲ REGRADED 2026-07-08 (ML0-6): this "spatial route
    DEPRECATED" flag is scoped to a FIXED-TICK spatial cut FOR κ ONLY. It does NOT deprecate regional
    subalgebras as such — the causal DIAMOND in (cell×tick) HISTORY is a DIFFERENT, live object (ML-0
    built its net; see below). The over-generalized reading is what deferred the locality layer.**
    **▶ THE κ COMPLETION = M0-2R (frozen contract 846573a,
    internal research notes): the run as the KMS state of the tick.**
    The reframe, verified on repo objects: the arrow (ρ_step = u·|h|max < 1) IS sub-criticality
    u < u_c = 1/(k−1) = 2^(−b_edge) of the multiway path gas; the currency principle (p=2^−L, E=κL)
    forces β·κ = ln2 (Landauer as CONSISTENCY); the Landauer point IS the path-gas critical point; by
    A1's own algebra κ = h/t_P (one Planck action quantum per tick). **THE NAMED INCOMPLETE EQUATION
    (T4): κ·t_P = A_tick — A1 uses A_tick = h = 2πħ; the 2π is un-derived (KMS-periodicity/BW-angle
    class).** Sessions: S1 = T2+T3 (arrow=sub-criticality; Landauer=criticality theorems, exact); S2 =
    T1 (KMS/FLOW-ID: is the run-state's modular generator affine in N̂ ⟹ thermal time = tick = Gate A's
    "dynamical ∂_N" at the state level) + T4 scoping. Named traps: α₁-vs-u_c (two temperatures),
    2π/ln2 pattern-match, purity overclaim, truncation mirage. Do NOT re-run M0-C/M0-1/M0-4.
    **▶ EXECUTED & COMPLETE (a model, 2026-07-07, commits e965d3d + e2c11fe, verify 65/65):**
    **S1 (T2+T3) THEOREMS LAND** — Ω_n=(k−1)ⁿ (3 proofs); Z(u) radius u_c=1/(k−1)=2^(−b_edge)
    (Ihara-Bass cross-check); BZ sup|h|=k−1 at Γ ⟹ the repo arrow ρ_step=u(k−1)<1 IS u<u_c;
    forward converges/backward diverges ⟹ delete>add COROLLARY. Currency consistency FORCES β·κ=ln2;
    per-tick factor=u_c ⟹ path-gas critical == currency per-tick == Landauer = ONE point. **ADOPTION
    NOTE (reported, register NOT edited): the ln2 in κ=k_B·T·ln2 is now DERIVED; currency premise + T
    remain adopted (partial A-IT3 downgrade — needs full protocol to book).**
    **S2 (T1) KMS-TICK, EXACT** — the equilibrium run state, restricted to the N̂/tick-count subalgebra
    (ω globally pure), has an EXACTLY geometric marginal (Born-2, ratio (u/u_c)²) and an EXACTLY affine
    modular generator −log p_n (residual 4e-14) ⟹ **THERMAL TIME = THE TICK** (Gate A's "dynamical ∂_N"
    at the state level), β_eff=2·log(u_c/α₁); **R-V "interior β free" DISSOLVES** (β derived; M_Z NOT
    reopened — report-only). Localized seed thermalizes at rate u_c=(Ramanujan gap)². Two-temperatures
    held. **T4-R (the 2π) DERIVED (a model, 2026-07-07, pre-reg e533558, commit 0c52815):** N̂ integer on
    consecutive integers ⟹ the modular flow is a compact U(1) of MINIMAL period exactly 2π (‖U(2π)−I‖=2e-14;
    no return at 2π/j) ⟹ Bohr–Sommerfeld number-phase: one tick = one full-loop action quantum = 2πħ = h.
    (2π = circle period, NOT pattern-matched; A1 = end cross-check.) **NET: κ's TEMPERATURE WALL CLOSED —
    κ = h/t_P, dimensionless content FULLY DERIVED (ln2 by currency-consistency + thermal-time=tick + 2π
    by tick-integrality); only t_P (standing anchor) + the currency ontology (E∝L) remain, neither an
    adopted external number. A-IT3 (Landauer) GRADUATED to framework-internal** (booked: OEF theorem §9,
    IT-stability axioms, adoption register). No scoreboard value moved (κ magnitude was already h/t_P in
    A1; this closes its DERIVATION). Do NOT re-run M0-C/M0-1/M0-4 or T1–T4.
  • M2(a) (KMS thermodynamics, the cosmology engine) EXECUTED 2026-07-07 (a model, pre-reg 342ed5e, commit
    22fc3f2, verify 65/65) → **EoS-DERIVED: c_s² = 1/3.** The walk-gas equation of state = the srs SPIN-1
    Weyl cone (triple point at Γ {−1,−1,−1,3}: 2 linear branches + 1 flat band). Relativistic sector:
    p=ρ/3 ⟹ **c_s² = 1/3** (statistics-robust + anisotropy-invariant) = the Tier-2 pressure mechanism the
    bias-function theorem (§9) names for θ_*. c_s²=1/3 REPORTED not fitted; raw velocity anisotropic 55%
    (coordinate feature; B3 emergent-SO(3) isotropises physical cone; scale deferred to M2c). **θ_* STAYS
    ❌ OPEN (overclaim guard): still needs M2c (native coasting acoustic scale replacing the log-divergent
    r_s) + M2b (fluctuation spectrum), CAN FALSIFY vs Planck. c_s²=1/3 is the INPUT, not the answer.** No
    value moved. `M2_walk_gas_eos_prereg_2026-07-07.md`. Y_p (B2.1a) still gated on B1 nucleon sector.
  • M2(b) (KMS fluctuation spectrum) EXECUTED 2026-07-07 (a model, pre-reg c267fd9, commit ad7322a, verify
    65/65) → **SPECTRUM-BUILT.** Native fluctuation spectrum S(q)=Σ_bands coth(β E_i/2)/(2 E_i) (KMS/FDT,
    E from the Weyl node λ_F=−1). **FLAT-BAND-DOMINATED** (spin-1 m=0 at E≈0 ⟹ coth/E divergent, flat/cone
    ~10⁴) = the clustering (matter) seed, vs the cone acoustic (radiation) ⟹ a two-component substrate
    fluid. Cone tilt blue (sign robust; exponent noisy from crude isolation). **n_s + σ_8 STAY ❌ OPEN**
    (overclaim guard): need the horizon-crossing map to the primordial CURVATURE (multi-session per the
    bias-function theorem) + trajectory T(z). Raw blue cone tilt is NOT n_s. c_s (M2a) + this spectrum =
    M2c's two inputs. No value moved. `M2b_fluctuation_spectrum_prereg_2026-07-07.md`. ▶ NEXT: **M2c**
    (native coasting acoustic scale → θ_* confront; DEEP, naive r_s log-diverges + naive fixes dead per
    B2 scoping, high-risk) OR **M1 DHR sectors** (−70 ppm/species keystone).
  • M2(c) (native acoustic scale → θ_*) EXECUTED 2026-07-07 (a model, pre-reg 95f9f7a, commit 2874677) →
    **DIAGNOSTIC; θ_* stays ❌ OPEN.** Native coasting (a∝t) ⟹ θ_*=11 rad (1057× Planck, ABSURD>2π;
    r_s≫D_C inverted hierarchy) ⟹ standard r_s/D_A INAPPLICABLE (confirms prior). NEW: a radiation era
    (a∝t^{1/2}, which M2b's derived cone/radiation component could source) CURES the r_s divergence →
    θ_*=5.3e-4 (finite, ~20× off, crude; a∝t^{1/2} flagged ASSUMPTION, NOT banked). **Two open pieces
    named: (i) native pre-recomb expansion (M2b radiation → a∝t^{1/2} vs theorem-coasting a∝t); (ii)
    bias-function/z_eff=1.916 extraction (θ_* likely a ΛCDM-fitter quantity, multi-session, same
    machinery that closed Ω_m/Ω_Λ/w_DE).** θ_* a genuine falsification exposure; formula inapplicable
    AND native pred open; no ΛCDM era imported. No value moved. `M2c_native_acoustic_scale_prereg_2026-07-07.md`.
  • **▶ THE MC-TRACK (architect diagnosis 2026-07-07, frozen contract 925f5b0,
    internal research notes): the OBSERVER-CLOCK layer — the θ_*
    category error is right-quantity-WRONG-CLOCK (same genus as κ's wrong-layer).** SMOKING GUN: coasting
    × horizon-thermal ⟹ T∝t^(−1/2) AND H∝T² = exactly the radiation-FRW laws; the frames differ ONLY in
    a(T) (lengths/angles) = precisely where θ_* broke; even z_rec=1089 was an import (native bath-clock
    1+z_rec≈1.2e6). THE PRINCIPLE: one history, many clocks — the Hubble tension = the derivative of the
    clock/bias map between anchors (sign = cheap kill-test); cosmic time (H₀t₀=1, −0.15σ) = the
    matter-clock reading, already shipped. Missing math: (1) the clock map (tick↔thermal/modular,
    conformal level — un-imports z_eq); (2) the phase-memory kernel (dissipation half of M2b's FDT —
    truncates the scale-free divergent sound integral; answers coherent-peaks); (3) the fitter map at
    perturbation level. **EXECUTION QUEUE: MC-0 frame-identity → MC-1 clock map → MC-2 kernel → MC-3 θ_*
    BLIND confront → MC-4 Hubble-tension sign→magnitude → then M1 DHR sectors (keystone; forks→architect) →
    MC-5 n_s (opt) → B2.1a Y_p → M3 last.** Poisons: no targets before blind confronts; kernel/map
    derived-or-dead; 1/48↔n_s FORBIDDEN. θ_*/Y_p/n_s/σ_8/−70 ppm all remain ❌ OPEN.**
  • **MC-TRACK EXECUTED (a model, 2026-07-07→08, verify 65/65): MC-0 IDENTITY-LOCKED · MC-1 MAP-FORCED ·
    MC-2 PARTIAL · MC-4 SIGN-PASS.** MC-0 (7ad4b14): frame identity theorem-grade (coasting & radiation-FRW
    share T(t),H(T); differ only in a(T)=lengths; dynamics frame-blind). MC-1 (4c6b60a): clock FORCED
    (M0-2R); z_rec=1100 photon-clocked; the clock does NOT make the era native ⟹ r_s divergence real→MC-2.
    MC-2 (b99f669): dissipation forced (Ramanujan gap ⟹ divergence CURABLE) but the QUANTITATIVE acoustic
    scale needs the collective-mode density-response/Lindhard build (γ_sound(q)) — **MC-3 θ_* BLOCKED**.
    MC-4 (55b6769): SIGN-PASS — the clock map + the theorem-grade 16/15 rate gap FORCE H_0^CMB<H_0^local
    (observed matches); UNIFIES the SHIPPED Hubble tension (=derivative of the map) with θ_* (=map on the
    acoustic sector) under one principle. **θ_* stays ❌ OPEN** (MC-3 blocked on γ_sound). ▶ NEXT: MC-2
    completion (density-response→γ_sound→MC-3 θ_* blind), OR M1 DHR sectors (−70 ppm keystone). Do NOT
    re-run MC-0/1/2/4.
  • **MC-2b REDIRECT (a model, 2026-07-08, commit c0d71a7, verify 65/65) — HONEST CORRECTION to the
    diagnosis.** DERIVED: the collective sound damping γ_sound(q)=(1/2)ν_s q², ν_s=c_s²τ (Maxwell,
    radiation cone; forced by M2a c_s + MC-2 τ) = the SILK/envelope damping. **UNITS CHECK KILLS the
    phase-memory-cures-r_s claim:** r_Silk=c_s τ≈2.6 tick-lengths is MICROSCOPIC (~60 orders below
    cosmological) ⟹ cosmological modes all coherent; damping does NOT set the sound horizon; the coasting
    r_s=c_s·η DIVERGENCE STANDS. **The r_s cure needs the NATIVE z_eq / FLUID-ONSET (the 3rd object)** —
    M2b's two-fluid supplies the crossover candidate but its native scale (flat-band gravitation? z_eq?)
    is UN-BUILT (Jacobson/entanglement-gravity or B1 masses). θ_* stays ❌ OPEN; MC-3 blocked on the
    z_eq build, NOT the kernel. ▶ NEXT: native-z_eq build (deep, maybe architect) OR M1 DHR sectors. Do NOT
    re-run MC-0/1/2/2b/4.
  • **MC-3a EXECUTED (a model, 2026-07-08, commit d213e72, verify 65/65) → TENSION: θ_* is a GENUINE ~9×
    coasting over-prediction, NOT a fitter artifact.** The perturbation-level clock/fitter map (obj 3a,
    the Hubble-tension analog) gives only ~16/15 (~7%) — far too small. Coasting θ_*(z_eq~3400)=0.093
    over-predicts Planck 0.0104 by **~8.9×**; the required onset (z~1248) is unphysically late. **θ_* is
    NOT a fitter artifact** (distinguishes it from H_0/Ω_m/w_DE which WERE and closed); **a native z_eq
    (3b) does NOT rescue it**; the ~9× is PHYSICS (coasting r_s too large vs D_C, the scale-free e-fold
    structure). **HELD OPEN as a QUANTIFIED ~9× FALSIFICATION EXPOSURE** — the two prior escapes
    (formula-artifact, fitter-artifact) are now BOTH eliminated. Escape routes named, neither built:
    (i) a non-coasting effective r_s for the acoustic sector (in TENSION with MC-1's forced coasting),
    (ii) the coasting acoustic prediction genuinely FAILS at θ_*. The θ_* thread has EXHAUSTED the
    MC-track's tractable routes. ▶ RECOMMEND: escalate the ~9× θ_* tension to architect (named open
    falsification-pressure; do NOT keep drilling — routes exhausted); pivot to **M1 DHR sectors** (−70
    ppm/species keystone). Do NOT re-run MC-0/1/2/2b/3a/4.
  • **▶ THE MG-TRACK (architect diagnosis 2026-07-08, frozen contract 6d5e11d,
    internal research notes): WHY θ_* isn't converging — THE REPO HAS
    TWO H's.** The coasting theorem is a COUNTING statement (N_hub.py:74: H = 1/(N·t_P) = the tick-rate
    SPINE, p-blind — any a∝N^p satisfies it); the thermal successes run on a∝N^{1/2},N^{2/3} ERAS
    (era_handoff:15); the distance successes use a=N at LATE times only. "a≡N at all epochs" = an
    unexamined LABEL (identification-layer lesson). θ_* = first observable with one leg in each regime ⟹
    the ~9×. Eras were "imported" because the framework has NO GRAVITY CLOSURE (nothing lets the fluid
    source ȧ/a) — ω never gravitates. INGREDIENTS ARRIVED via the κ arc: first law δ⟨K⟩=δS (M0-C exact) +
    KMS (M0-2R) + state counting = Jacobson's three inputs. **QUEUE: MG-0 two-H theorem (bounded; regrades
    MC-1's inherited label; does NOT relabel θ_* solved) → MG-1 Jacobson station (gravity from the state,
    derive-or-die; HARD; forks→architect) → MG-2 native z_eq (un-imports the dyadic-ladder seam) → MG-3 θ_*
    BLIND re-confront (CAN STILL FAIL). M1 DHR sectors interleavable.** θ_* stays ❌ OPEN; the MC-3a ~9×
    stays booked as pressure.
    **▶ MG-TRACK EXECUTED (a model, 2026-07-08, verify 65/65): MG-0 TWO-H-LOCKED · MG-1a CONTRADICTS-BY-4π ·
    MG-1c ERA-STRUCTURE-NATIVE.** The gravity closure (RG2b/Cai-Kim, promoted Friedmann H²∝ρ) RECONNECTED
    to the state layer. MG-0 (c05cd07): H_sub=Ṅ/N=1/(N t_P) forced+metric-blind; a(N) per era separate;
    "a=N global" over-identification; MC-1 regraded, MC-3a escape (i) dissolved. **MG-1a (b464e5b): the
    DERIVED κ=h/t_P=2π M_Pl is 4π off the closure's required M_Pl/2 ⟹ G_eff=G/(4π); CONFIRMS the panel's
    flag that κ=M_Pl/2 was goal-sought; Newton's G does NOT close parameter-free (4π=2π[h/ħ]×2[geometric]);
    magnitude OPEN.** MG-1c (9b43712): the two-source closure (record+M2b fluid) gives native
    radiation→matter→coasting era EXPONENTS = the two-H resolution (form level, un-imports the dyadic
    ladder); consistency needs κ/M_Pl<1 (derived 2π violates) ⟹ inherits MG-1a's 4π; flat-band gravitation
    = native dark-matter question, FROZEN. **NET: the 4π is now THE gravity question (does gravity see h or
    ħ; c_S=1 vs 2).** ▶ NEXT: MG-1b (c_S native via M0 modular first law + purity), then MG-2 (z_eq) / MG-3
    (θ_* blind). Do NOT re-run MG-0/1a/1c. G-magnitude/θ_*/Y_p/n_s/σ_8/−70 ppm remain ❌ OPEN.**
    **▶ MG-1b c_S=1-RATIFIED (f84e0cc) + MG-1d OPEN-MISS-AT-2π (b0e6337), verify 65/65.** MG-1b: M0-C's
    modular first law δ⟨K⟩=δS is entanglement/record machinery ⟹ c_S=1 (the c_S=2 MI hope NOT forced —
    costs the Clausius asset); reduces MG-1a's 4π to 2π. MG-1d (the Rindler-2π, disciplined): the
    framework's OWN derived inputs (κ=h/t_P, c_S=1) give **G_eff=G/(2π) — Newton's G is an OPEN MISS at
    exactly 2π** (parameter-free geometric miss). ħ/t_P WOULD close G_eff=G but requires an un-derived
    local Bisognano-Wichmann 2π; selecting it is goal-seeking (NOT done). See the INCOMPLETE EQUATION
    logged below.
    **⚠ INCOMPLETE EQUATION (NEW 2026-07-08, MG-1d): the emergent LOCAL causal-horizon (Unruh) temperature
    from the substrate.** Newton's G falls out of the gravity closure as G_eff=G/(2π) with the framework's
    derived κ=h/t_P and ratified c_S=1 — a sharp, parameter-free 2π MISS. The defining equation is
    incomplete at the horizon temperature: the gravitational Clausius (Jacobson) uses the LOCAL Unruh
    temperature T=a/(2π) (Bisognano-Wichmann K_mod=2π·K_boost), but M0/M0-2R derived only the GLOBAL tick
    modular flow (Bohr-Sommerfeld action-angle, one action quantum h per tick), NOT a LOCAL emergent
    Rindler boost. **Does the emergent local horizon temperature carry an independent BW 2π (⟹ gravity sees
    ħ/t_P ⟹ G_eff=G, closes) or reduce to the global tick κ=h/t_P (⟹ G_eff=G/(2π))?** CHASE: derive the
    continuum Rindler boost / modular flow of a causal DIAMOND from M0's discrete modular structure (the
    emergent-Lorentzian-metric asset, RG2b, is the substrate). Until then Newton's G is a 2π OPEN MISS —
    NOT to be relabeled "a convention" or closed by selecting ħ. `MG1d_rindler_2pi_prereg_2026-07-08.md`.**
    **⟲ SHARPENED by ML-1 (2026-07-08, commit 80b4b20): the causal-diamond modular flow WAS built (the_net.py);
    the emergent LOCAL Rindler boost EXISTS + is geometric (h_A NN-dominant ≈252) ⟹ the "only the global tick
    exists" horn is CLOSED. The 2π now reduces to ONE object: the srs emergent PROPER-DISTANCE METRIC
    (cell-layer ≠ geodesic hop, ~3 hops/cell). Concrete chase: re-read the modular slope on ML-0's exact
    geodesic distance. Newton's G stays a 2π OPEN MISS until that metric read.**
    **⟲ ML-1′ EXECUTED (2026-07-08, pre-reg 0b11a29 BEFORE read, verify 65/65) → CLEAN-NON-2π, G STAYS
    OPEN — but the 2π is now BRACKETED.** `proofs/foundations/ML1prime_geodesic_2pi_2026-07-08.py` (extends
    the_net.py). The cone-sector modular slope on the FORCED geodesic-hop metric (BFS; hops-per-cell =
    exactly 3.000, measured NOT tuned) = **0.44×2π** (near-horizon) — well BELOW 2π. With ML-1's CELL-layer
    reading **1.56×2π** (ABOVE), **2π is BRACKETED between the two combinatorial metrics (factor 3) ⟹
    NEITHER naive lattice distance (cell-layer NOR graph-hop) closes the 2π.** Values RAW, NOT
    pattern-matched (not π/any constant), NOT tuned to the bracket; ħ NOT selected. Caveat: nearest
    perpendicular bond is 3.5 hops deep ⟹ near-horizon slope under-resolved (concave interior, as on the
    benchmark parabola). **⟹ THE 2π DECIDER IS NOW DEFINITIVELY THE DERIVED EMERGENT-LORENTZ PROPER
    DISTANCE (the srs cone's velocity/metric structure), NOT any lattice-combinatorial distance — the
    sharpest form yet of MG-1d's incomplete equation: build the emergent-Lorentz spatial metric of the srs
    cone (from the dispersion/v_F), then the BW 2π is decided.** Newton's G stays ❌ OPEN at 2π.
    **⟲ ML-1″ EXECUTED (2026-07-08, pre-reg 3e6fe96 BEFORE probe, verify 65/65) → EMERGENT-METRIC DERIVED
    (the object MG-1d named) + 2π-CANDIDATE, G STILL ❌ OPEN.** `proofs/foundations/ML1pp_emergent_metric_
    2026-07-08.py` (extends the_net.py: `cone_velocity`). **FORCED (real object built):** the srs cone's
    emergent spatial metric is a CLEAN positive-definite quadratic form g^{ij} (velocities: axis 1/√2,
    face split 1/2 & √3/2, body 1/√3; off-diagonals ±1/4; **eigenvalues {1/4,1/4,1}** ⟹ velocity
    eigenvalues {1/2,1/2,1}), which predicts the body-diagonal velocity EXACTLY (0.5774 pred = meas) ⟹ a
    GENUINE anisotropic relativistic Dirac cone. The flat band is dispersionless (v=0). **This IS the
    emergent-Lorentz metric the 2π decider needs — now a forced read in the net.** **2π assessment
    (honest, NOT a closure):** because the metric is genuine, in its emergent-Lorentz (isotropised, B3)
    frame the cone is canonical Dirac ⟹ BW gives 2π ⟹ a CANDIDATE for G_eff=G, CONTROLLED by the
    emergent-Lorentz exactness = **B3's residual (the SAME control as the +6σ M_Z oblique floor — a new
    cross-link: gravity's 2π and M_Z's oblique floor share one control).** NOT closed: (i) BW-gives-2π is
    plausibility, NOT a computed G derivation; (ii) a resolved numerical confirmation (isotropised-frame
    slope=2π) is un-done (near-horizon under-resolved); (iii) the B3 lattice residual is the correction.
    ħ a candidate, NOT selected. **Newton's G stays ❌ OPEN at 2π. ▶ remaining: the isotropised-frame /
    operator-level (h_A=2π·K_boost) confirmation + the exact-emergent-Lorentz limit (B3).**
    • **▶ ML-2 EXECUTED (a model, 2026-07-08, pre-reg 50c64e7 BEFORE probe, verify 65/65) → SECTORS=SPECIES
      (structural); the −70 ppm KEYSTONE STAYS ❌ OPEN (the DR-force-vs-zero-bit choice = architect fork).**
      `proofs/foundations/ML2_dhr_sectors_2026-07-08.py` (extends the_net.py: `gauge_sector_category`).
      **FORCED (structural):** the DHR superselection sectors of the observable algebra A = F^G on ML-0's
      net COINCIDE with the species grading. Field algebra F = 8-dim Cl(6) Fock; gauge group G = **A4**
      (the forced J-covariance group), Fock rep SPINORIAL (cocycle takes −1) ⟹ the **double cover 2T**
      (binary tetrahedral; the double-cover Z2 = ML-0's fermion parity / Klein twist). **[U(g),N̂]=0**
      (species gauge-invariant) and each species eigenspace IRREDUCIBLE ⟹ Fock decomposes into exactly
      the species sectors **{ν:1, d:3, u:3, e:1}**. Whole-Fock A4-commutant dim = **8 = 2²+2²** ⟹ **2
      A4-irrep TYPES (a singlet + the triplet), each multiplicity 2, the two copies exchanged by the
      particle-hole Z2 (w↔3−w)** ⟹ species label = (A4 irrep) × (bit); **quark/lepton = triplet/singlet.**
      The winding deck {4,2,2} (unsigned screw U_π, U_π³=−I) is a SEPARATE spinorial grading CROSS-CUTTING
      these ([U_π²,N̂]=1.12≠0). Statistics = fermion parity {+,−,+,−} (Bose/Fermi; Cl(6)→KO-dim 6 = the KO
      2→6 residual's natural home, reported not interpreted). **⟹ the species labels ARE the net's
      superselection sectors, STRUCTURALLY — not an external assignment.** **architect FORK (booked, NOT
      adjudicated): (i) which sector = which physical particle / the 3 generations; (ii) DR-UNIQUENESS —
      does "sectors=species" + Doplicher-Roberts FORCE the species lift (paying WS1's 1.6300 bit/site
      adoption BY THEOREM) or is it still gauge (zero-bit)?; (iii) statistics/KO as a physical prediction.**
      The −70 ppm STAYS ❌ OPEN. Do NOT re-run ML-2.
    • **▶ ML-3 EXECUTED (a model, 2026-07-08, pre-reg b77f059 BEFORE probe, verify 65/65) → PARTIAL: the
      flat-band QUANTUM METRIC is BUILT (the daylight object); native two-fluid FORCED; z_eq NEEDS-ML-4;
      θ_* stays ❌ OPEN.** `proofs/foundations/ML3_flatband_weight_2026-07-08.py` (extends the_net.py:
      `band_quantum_metric`). **GROUNDING:** λ=−1 is NOT a global flat band — the m=0 branch is flat only
      to linear order at the node Γ (quadratic band-touching). **ML3-A (FORCED, real object):** the m=0
      band's QUANTUM METRIC — un-computed anywhere (every prior Berry read = the IMAGINARY part on
      DISPERSIVE bands) — is BUILT: **anisotropic, DIVERGENT ~C(n̂)/|k|² at the node**, C(n̂)={1.5(axis),
      3.0(body),2.66(gen),5.0(face)} (transverse geometry; longitudinal-along-flat-axis part vanishes);
      BZ-integrated weight FINITE (31.4, since ∫d³k/k²~∫dk). **Corrects the daylight hypothesis:** it
      DIVERGES (not a finite regulator) and does NOT simply "replace M2b's REG=1e-4" — the energy
      divergence and the wavefunction geometry are distinct objects. **ML3-B (FORCED):** the srs bands are
      a **NATIVE TWO-FLUID** — cone E~q (radiation, a⁻⁴) + m=0 heavy E~q² (matter, a⁻³) ⟹ **ρ_m/ρ_r ∝ a is
      forced by the dispersions** (the matter/radiation scaling, NOT imported); m=0-as-matter now grounded
      in E~q², not just clustering. **ML3-C:** the flat/cone fluctuation weight ratio is FINITE, O(10²–10³),
      but REGULATOR/grid-DEPENDENT (135/198/328 across grids, spread 2.4×) — NOT a clean forced constant
      (M2b's 10⁴ was REG-dependent too). **ML3-D (z_eq, BLIND; observed 3402 revealed only at the end):**
      native seed ~198 is order-consistent with observed z_eq but regulator-dependent + reference-epoch
      un-fixed ⟹ **NEEDS-ML-4** (regulator-independent weight + era integration). NOT pattern-matched, NOT
      closed. **architect FORK:** m=0-as-dark-matter ID; the reference-epoch/era integration (ML-4);
      scaling-as-physical-claim. **z_eq/θ_* STAY ❌ OPEN.** No scoreboard value moved. Do NOT re-run ML-3.
    • **▶ architect ADJUDICATED the 3 forks (2026-07-08 EVE) → continuation contract internal research notes
      2026-07-08_ML_fork_stations.md` (landed ebc43e6):** Fork A DR pays the FRAME not the WELD (winding
      cross-cuts ⟹ split verdict; the DR-canonical frame dissolves O4's lift-dependence ⟹ ε/ML-5 well-posed
      even with the weld unpaid; **A4-triplet = COLOR not generation**) → ML-2b. Fork B TWO confounds
      (metric AND state; BW is a VACUUM theorem) → ML-1‴. Fork C the DIAMOND is the regulator (δ⟨K_R⟩
      finite ∀R; crossing R_eq = native equality scale) → ML-3b. Order ML-1‴→ML-3b→ML-4, ML-2b
      interleavable, ML-5 gated on ML-2b=FRAME-FORCED.
    • **▶ ML-1‴ EXECUTED (a model, 2026-07-08, pre-reg b9595d5 BEFORE probe, verify 65/65) → CONVERGES-
      ELSEWHERE: the 2π miss RE-QUANTIFIED at ~1.07×2π; Newton's G stays ❌ OPEN (NOT closed).**
      `proofs/foundations/ML1ppp_computed_2pi_2026-07-08.py` (the_net.`emergent_metric`). Removed BOTH
      Fork-B confounds: **METRIC** (proper distance under the derived g^{ij}, factor 1/√(g^{00})=√2 per
      cell; FIXED first-principles, not tuned) + **STATE** (vacuum vs run-KMS separately). Benchmark
      calibration 0.9988×2π (pipeline trusted). **VACUUM (BW-relevant), finite-size extrapolated M→∞:
      slope = 1.068×2π ± 0.006** — stable across M=8/10/12, ~**7% ABOVE 2π** (11σ from the fit error).
      Removing the metric confound sharpened the miss from cell-1.56×2π / hop-0.44×2π to **~1.07×2π** (much
      closer, NOT 2π). The +7% is ABOVE (not the near-horizon parabola falloff, which is below) ⟹ a genuine
      feature; candidate sources = flat-band admixture at the Fermi surface / exact metric convention (NOT
      tuned away). **KMS state → 0.55×2π (−52% thermal shift)** ⟹ architect's STATE confound is real & large;
      the vacuum is the BW state; the thermal shift = the tick-thermal horizon correction (a named object).
      Operator test `h_A vs 2π·K_boost`: large residual (0.67) = the finite-region CC parabola
      2π·x·(1−x/W)·T₀₀ (not the pure boost) ⟹ contaminated, NOT a clean read; the near-horizon slope is
      primary. Cross-link: the +7% is the same ORDER as B3's +6σ M_Z oblique ~4% but does NOT match within
      2% (6.8 vs 4) — suggestive, NOT confirmed as one lattice correction. **2π MEASURED (extrapolated)
      never inserted; ħ NOT selected; local 2π does NOT retro-edit κ.** **Newton's G = an OPEN MISS,
      sharply re-quantified at ~1.07×2π (~7% high).** Do NOT re-run ML-1‴. ▶ NEXT: ML-3b (diamond
      δ⟨K_R⟩ → z_eq) then ML-4 θ_*, using ML-1‴'s decided ~1.07×2π normalization; ML-2b interleavable.
    • **▶ ML-3b EXECUTED (a model, 2026-07-08, pre-reg 5ca9b1a BEFORE probe, verify 65/65) → NO-CROSSING: the
      diamond regulates the weight (Fork C's valid fix) but there is NO static equality; z_eq is DYNAMICAL
      → ML-4. z_eq/θ_* STAY ❌ OPEN.** `proofs/foundations/ML3b_diamond_zeq_2026-07-08.py` (extends
      the_net.py: `diamond_modular_energy`). **FORK-C insight CONFIRMED valid:** the causal diamond IS the
      physical IR regulator — per-band δ⟨K_R⟩ (proper momentum cutoff q_min=π/R under the emergent metric,
      KMS modular energy) is **FINITE for every proper radius R with NO chosen regulator**, fixing ML-3's
      regulator-dependence (135/198/328). **BUT NO STATIC CROSSING:** the flat (matter, E~q²) band
      DOMINATES the cone (radiation, E~q) by **116×→26×** over R∈[8,256] cells, and δ⟨K_R⟩ SATURATES at
      large R (the diamond includes all modes) ⟹ the ratio never reaches 1; no reachable equality scale.
      **FORCED scaling exponents (the clean ML-4 input): cone δ⟨K_R⟩ ∝ R^{+0.94}, flat ∝ R^{+0.51}** —
      the two components DO scale differently with R (architect's prediction, structurally confirmed; cone
      grows faster), but they don't statically cross. ⟹ **the matter/radiation equality is DYNAMICAL (set
      by the redshift/era exponents), NOT a static-diamond crossing** — handed to ML-4 with the
      regulator-free δ⟨K_R⟩(R) + exponents. NO out-of-range R_eq extrapolated; NO z_eq pattern-matched;
      Planck 3402 NOT confronted (no native number yet). θ_* stays the booked ~9× exposure. No scoreboard
      value moved. Do NOT re-run ML-3b. ▶ NEXT: ML-4 (θ_* blind, using ML-1‴'s ~1.07×2π + ML-3b's exponents
      + the redshift/era machinery); ML-2b interleavable.
    • **▶ ML-2b EXECUTED (a model, 2026-07-08, pre-reg 318b45e BEFORE probe, verify 65/65) → FRAME-FORCED
      (conditional on TD-limit duality); ML-5 now POSABLE; −70 ppm STAYS ❌ OPEN.** `proofs/foundations/
      ML2b_dr_frame_2026-07-08.py` (extends the_net.py: `dr_frame_audit`). Confirms architect's Fork-A
      adjudication exactly. **ML2b-A CATEGORY COMPLETENESS:** the winding is NOT a gauge/DHR charge — the
      A4 gauge action fixes the vacuum ([U(g),N̂]=0, U(g)|0⟩∝|0⟩) so its irreps ARE the sectors, but the
      winding screw U_π does NOT (⟨0|U_π²|0⟩=i/2, |·|=0.5≠1; [U_π²,N̂]=1.12≠0) and is a GLOBAL geometric
      screw (not locally creatable) ⟹ it adds NO sectors ⟹ **category = the species (2T-irreps), NOT
      bigger** (conditional TD-limit). **ML2b-B:** DR ⟹ (F,2T) canonical given category+statistics+
      cell-duality, CONDITIONAL on the TD-limit twisted Haag duality (stated, not asserted). **ML2b-C the
      load-bearing result:** the species subspaces P_w are Schur-CANONICAL A4-isotypic components (residual
      freedom = the internal color-SU(3)/generation unitaries = PHYSICAL gauge, not a frame ambiguity) ⟹ a
      **CANONICAL FRAME with no lift-swap freedom ⟹ O4's 60% lift-dependence DISSOLVES ⟹ ML-5 ε readout is
      POSABLE in this frame.** **ML2b-D PAYMENT AUDIT:** the weld **H(w|t)=1.6300 bits SURVIVES unpaid**
      (winding cross-cuts, out of DR's reach); **DR pays the FRAME, NOT the weld** — confirming the fork was
      mis-priced ("1.63 bits or nothing"): the value is the canonical frame, not bit-reduction. **ML2b-E:**
      the A4-triplet = COLOR (gauge, = Cl(6)-Fock); the GENERATION label = the cross-cutting winding deck
      {4,2,2} (non-gauge) — two distinct 3's, do not conflate. **⟹ ML-5 is GATED-OPEN (posable), conditional
      on TD-limit duality; still derive-or-die, pre-registered, full poison set (2α₁⁵ excluded at 15σ_ε).
      No ε computed; −70 ppm STAYS OPEN.** No scoreboard value moved. Do NOT re-run ML-2b.
    • **▶ ML-5 EXECUTED (a model, 2026-07-08, pre-reg e9afdb1 BEFORE probe, verify 65/65) → CONSTRUCTION-GAP:
      the −70 ppm did NOT close and is NOT zero-bit; the canonical frame is NECESSARY but NOT SUFFICIENT.**
      `proofs/foundations/ML5_epsilon_2026-07-08.py` (reuses LOOP_E2a's interacting-run G_int; no scratch
      fork). Disciplined derive-or-die on ε=δ_eff−2/9=−1.7515e-7 rad; the value was **NOT computed** (the
      transport is under-determined ⟹ computing it would be tuning to the target). **ML5-A WELD-DEPENDENCE
      SETTLED → FORCED-CORRELATION-ONLY:** the ε-seed is the UNIVERSAL bit-odd deck channel
      (ν−e)/2=(d−u)/2=(0,±√3/6) (dev 4e-16, WS1) = the FORCED correlation **I(w;t)=0.18 bits**; the unpaid
      residual **H(w|t)=1.63 is NOT needed** ⟹ **ε is NOT zero-bit** (reconciles architect's Fork-A re the weld).
      **ML5-B TRANSPORT forced-or-gapped → GAP (computed, not asserted):** the interacting-run chiral
      carrier A(α₁)=tr(Q₁G_int)−conj·tr(Q₂G_int)=**8.8e-4 ≈ 0.6 α₁²** is FORCED and nonzero (vanishes free);
      but the natural FORCED map — the winding-phase shift free→interacting — is **~0, NOT ε** (and the free
      winding phase 0 ≠ the read's δ=2/9, which is the Wigner-d survival, a different functional). Mapping
      the O(α₁²) carrier to ε (O(α₁⁴⁻⁵)) needs a further ~α₁²⁻³ suppression + a lepton-slice projection with
      **NO forced selector** (E2a's K4a un-forced choice survives EVEN in the canonical frame). **⟹ the
      canonical frame (ML-2b) dissolves O4's 60% lift-dependence [NECESSARY] but does NOT supply the
      transport functional [NOT SUFFICIENT] — CORRECTS architect's Fork-A "frame ⟹ ML-5 well-posed".** **▶
      SHARPEST LOCALIZATION TO DATE: the −70 ppm needs ONE named un-built object — the FORCED lepton-slice
      transport functional of the interacting-run chiral asymmetry A(α₁) (projection + trace→phase +
      minus-leading + the α₁²⁻³ suppression).** −70 ppm STAYS ❌ OPEN. Do NOT re-run ML-5 (do NOT compute a
      value from the under-determined functional). No scoreboard value moved.
    • **▶ ML-5b EXECUTED (a model, 2026-07-08, pre-reg 96c2b31 BEFORE probe, verify 65/65) → COUPLING-GAP:
      the transport build did NOT close the −70 ppm, but located the wall VERY precisely; NOT tuned.**
      `proofs/foundations/ML5b_epsilon_transport_2026-07-08.py` (reuses read_phases + LOOP_E2a; no fork).
      **ML5b-A (clean forced simplification):** the run-phase LEVER **d(δ)/d(cosβ) = EXACTLY 1** at
      cosβ=1/3 (δ(1/3)=2/9 verified) ⟹ **ε = Δc**, the interacting CHIRAL correction to the band-edge
      overlap — **the whole −70 ppm reduces to ONE number Δc.** **ML5b-B (GAP, computed not asserted):**
      the natural FORCED coupling — the band-edge **Perron projection of the chiral asymmetry — is EXACTLY
      0** (3e-16): the chiral ω/ω̄ asymmetry is ORTHOGONAL to the real/non-chiral Perron band-edge. And the
      interacting chiral asymmetry A(α₁)=0.6α₁² is the WRONG ORDER (ε~α₁⁴⁻⁵), so a forced ~α₁²⁻³ suppression
      is needed and NO built object supplies it (the **D = B⊗∂_N tensor coupling does NOT force it**; the
      projection that could vanishes). **ML5b-C:** ε NOT computed (gap ⟹ a value = tuning); target NOT
      confronted. **⟹ SHARPEST LOCALIZATION EVER of the −70 ppm: ε = Δc (lever=1, forced); Δc = the FORCED
      α₁²→α₁⁴⁻⁵ suppression coupling the (off-band-edge) interacting chiral asymmetry to the band-edge
      overlap / lepton-slice phase — a SINGLE un-built object, orthogonal to the trivial band-edge
      projection.** −70 ppm STAYS ❌ OPEN. Do NOT re-run ML-5b (do NOT pick a Δc from the gapped coupling).
      No scoreboard value moved; nothing tuned or pattern-matched.
    • **▶ ML-4 EXECUTED (a model, 2026-07-08, pre-reg 09318cd BEFORE probe, verify 65/65) → θ_* FALSIFICATION
      DISCHARGED (the ~9× was a coasting artifact); θ_* NOT closed to precision (stays ❌ OPEN).**
      `proofs/foundations/ML4_theta_star_2026-07-08.py`. Assembled the native θ_* = r_s/D_C (H₀-independent,
      shape-only) from FORCED/built pieces: z_*=1100 (MC-1); the native two-fluid eras (ML-3: cone=radiation
      (1+z)⁴, m=0=matter (1+z)³) + the coasting-theorem late-time (1+z)² (replaces ΛCDM's Λ); c_s²=1/3
      (M2a). **RESULT (blind; Planck 0.0104109 only at the declared end):** native θ_* = **0.95×→1.36×
      Planck** over z_eq=1000→3400, and **within ~1.5× for the WHOLE native z_eq range [300,5000]** (0.60×
      → 1.49×). The PURE-COASTING control DIVERGES (r_s log-diverges, ~1092× here = M2c's 1057× / MC-3a's
      regularized ~9×). **⟹ the MC-3a ~9× was a COASTING ARTIFACT: the native pre-recombination era is
      RAD+MATTER (ML-3's two-fluid), which gives the small standard-like r_s ⟹ θ_* is O(1)×Planck robustly,
      NOT 9× off.** **The biggest falsification pressure on the cosmology is DISCHARGED — the framework
      SURVIVES the θ_* falsification test, and it is ML-3's own two-fluid that discharges it (satisfying
      internal consistency).** **HONEST LIMIT: θ_* is NOT closed to PRECISION** — a residual factor up to
      ~1.5× remains, sensitive to the UN-PINNED z_eq (ML-3b), the c_s baryon-loading (c_s→0.457 at recomb
      lowers θ_*), and the coasting fraction Ok. **θ_* STAYS ❌ OPEN for precision; the ~9× exposure is
      DISCHARGED (no longer a falsification threat).** Planck confronted only at the declared end; nothing
      tuned; no scoreboard value moved. Do NOT re-run ML-4 (do NOT tune z_eq/c_s/Ok to sharpen θ_*).
    • **▶ TRUNK-AND-GRAFTS CHARTER (2026-07-08, post-ML dynamics pivot) →
      internal research notes** — the final-push architecture: six adapter
      contracts (G1 Sunada / G2 Furey–Stoica / G3 NCG / G4 AQFT / G5 thermal-time / G6 zeta-gauge) bolting
      the VERIFIED-NATIVE structures (standard realization `explore_12`; **srs⊕srs-z forced doubling with
      Higgs = the inter-sheet connection** `explore_m04`/`explore_m08`; bass-zeta det(I−uB) identities
      `phase1_3_s1` + `ihara_unification`; Gilkey a₄ `d4_spectral_action`; Wilson quadratic
      `srs_wilson_action_quadratic`; **δ = holonomy** `delta_dynamical.py:542`; 2T/ℍ (2,2) bidoublet;
      tick-2π anchor) onto their mature homes. **INCOMPLETE EQUATIONS LOGGED:** (i) **the KO-dimension
      sign table (J,D,γ) NEVER EXECUTED and docs INCONSISTENT** (KO-2 crown_jewel:118 vs KO-4 CLEANROOM:38
      vs SM-needs-6) — forced arithmetic, contract G3a/R2; (ii) the log-det(I−uW) ↔ a₄ bridge (Bass =
      discrete heat kernel) un-built — G3b; (iii) M_W / the EW scale absent; (iv) DECISIVE: D1 the 2π
      Kotani–Sunada scaling limit (tick-2π vs BW-2π Wick-duality; ML-1‴'s 1.068×2π = candidate
      discrete-rotation defect — G may close or sharpen), D2 Higgs survival vs the Perez-Sanchez drop-out
      (our scalar is FORCED, theirs is decoration), D3 the confinement holonomy-disorder binary (⟨P⟩
      gate); (v) R1 the u⁴/u⁵ lepton-slice zeta coefficients = the −70 ppm's α₁⁴⁻⁵ order (coefficients =
      cycle counts, forced; PRE-REGISTER the functional BEFORE reading; NO selection toward −1.7515e-7).
      Discipline: every contract/D/R pre-registered before execution; adapters add ZERO physics; engine
      never forked; failures booked raw.
    • **▶ SYMPHONY-S1b CAMPAIGN COMPLETE (2026-07-09, four batches, all checks PASS; verify 72/72
      throughout) → THE PORTING IS DONE: coverage 21 → 98/161; the `engine-surface-missing` blocker is
      FULLY RETIRED (77 → 0).** Batches: 1-FLAVOR 18 rows (2d21e36) · 2-MASSES+HIGGS 14 rows (48f4266;
      M_persistence proven NOT load-bearing — the mass sector = v + y_τ + two anchors + ONE native Koide
      construction) · 3-COSMOLOGY 13 net rows (5abd931/9eed1e3; ε_CP ≡ read_clock().eps identity; the
      coasting suite free via N_hub; **N_eff DEMOTED by architect adjudication** — its mapping via
      read_flavor().gens re-performed a rejected forced pairing) · 4-GAUGE+MISC 32 rows (the RG chain
      reusing the engine's own derived β {33/5,1,−3}; M_Z/m_W/m_ν2/m_ν3 ported WITH their open misses
      carried 🟡 unchanged; **N_eff re-promoted via its TRUE ingredient** observer_dim_three, itself
      ported as a provenance-commented structural literal parallel to θ_QCD=0 — checker-adjudicated
      legitimate). **FINAL STATE: Tier A=88 (101 comparisons at ≤1e-9, ZERO mismatches across the whole
      campaign), Tier B=10 (all adoptions named), Tier C=63 = 30 orphans (decision station pending,
      user sign-off) + 14 declared-external + 19 physics-blocked (ML-2: 7, local-metric/D1b: 6, B2: 6).
      Zero source↔lock inconsistencies found in 77 ports. The one-object law's Layer-1 claim is now TRUE
      AT CODE LEVEL.** Discipline events: one demotion-and-proper-repromotion (N_eff); five identity
      discoveries (formulas already engine-native); values moved: NONE (by design — S1b is
      value-preserving consolidation; the exploitation stations are S2+).
    • **▶ R2b RESOLVED (2026-07-09, pre-reg 97702d1 [authority hierarchy + adjudication rule FROZEN
      BEFORE the literature read]; literature sweep + mechanical adjudication) → READ-AS-KO-6: the
      internal Cl(6) Fock carries KO-DIMENSION 6 (exotic presentation); 4+6 ≡ 2 (mod 8) — THE
      FRAMEWORK'S KO ANATOMY MATCHES THE NCG STANDARD MODEL, CONFIRMED.** The frozen rule's condition
      (a) satisfied at authority level 3 with EXACT-match tables: Dąbrowski–Dossena (IJGMMP 8 (2011)
      1833, arXiv:1011.4456): for even triples "there are always two real structure operators J, that
      differ by multiplication by the grading operator. None of them should be preferred" — their table's
      second presentation of n=6 is (−,−,−); Ćaćić (LMP 2013, arXiv:1209.4832, §2.2 + Table 2.2 "6−"):
      J→Jγ reversibly maps canonical↔exotic, "6−" = (−,−,−). Connes' canonical table (hep-th/0608226
      App. 7 Def. 7.2; van Suijlekom 2024 identical) has ε′=+1 at even n — our adapter implemented the
      canonical convention, which is why the forced (−1,−1,−1) printed no-row. CONSISTENCY IDENTITY
      verified in-code AND algebraically: J′ = J_F·γ_F carries (+1,+1,−1) = Connes' canonical KO-6 row
      (follows from Vsig·conj(Vsig)=−I and Vsig·P_F·Vsig†=−P_F — not a candidate-set coincidence).
      INTEGRITY CHAIN preserved: bug (caught) → anomaly (booked) → literature-first resolution (this) —
      never convention-picking. Adapter/README/verify updated; G3 ✅ GREEN; G3b (log-det ≡ a₄) remains
      the pending half. R2b CLOSED.
    • **▶ SYMPHONY-S2 (=G3b) INTEGRATED (2026-07-09, pre-reg 2da17ce; implementation + adversarial
      check PASS-WITH-NOTES; verify 72/72) → GREEN-CHAIN + AMPLITUDE-CONVERGENT: THE LAGRANGIAN BRIDGE —
      the certified zeta and the native spectral action are two reads of ONE machine-checked spectral
      chain.** Extends `adapters/ncg_spectral.py` (KO section byte-identical, checker-diffed). The
      chain: **LB-1** the Bass pencil's roots ≡ the adjacency spectrum via the certified Ihara map
      (1.5e-15; DISCLOSED pre-reg slip: the pencil solves the RECIPROCAL quadratic (k*−1)u²−λu+1=0 —
      sympy-verified both ways by the checker); **LB-2** the heat trace reconstructed FROM the zeta side
      (pencil roots via FFT-interpolated determinant + np.roots — no eigh) ≡ OMEGA_T1's direct
      computation (1.45e-12; route disjointness code-audited); **LB-3 THE WEYL AMPLITUDE: r(t) =
      A_measured/A_pred descends monotonically 1.0199 → 1.0024 over t ∈ [30,240] (40³ grid, 100³
      cross-check 2e-8); free-intercept extrapolation r_∞ = 0.9917 (inside the declared ±0.02) ⟹
      AMPLITUDE-CONVERGENT** — the lattice cone sector reproduces the continuum a₄'s Weyl amplitude
      predicted by the CERTIFIED Albanese data (V_alb=4, v=½) at the ~1% level (HONESTY CLAUSE: an
      intermediate-t scaling-window statement on a bounded spectrum, NOT a true t→0 law); **LB-4** the
      flat band exits as INDEX exactly (Str ≡ −2 = χ(K4), 1.8e-15 — flat = topology, never beta);
      **LB-5** β-rows ≡ the engine's b4d exactly ({33/5, 1, −3}). **STAYS OPEN (LB-6, the named
      import):** the universal Gilkey/Seeley–DeWitt coefficients (the −11/3, ⅔, ⅓ Dynkin structure =
      the ζ_{D₄}(0) frontier) remain a Type-3 import per the engine's own OPEN marker — NOT self-derived,
      NOT fitted from the lattice. BOOKED PATTERN (2nd instance): a model sweep-prose misquoted a prior-art
      number ("0.944" vs OMEGA_T1's actual 1.0097) — sweeps are INVENTORIES; numbers come only from
      running code. Label fix at integration (the fit criterion was gating per pre-reg; its label said
      otherwise — code was right, label wrong). **⟹ GATE PROGRESS: S1 ✓ S1b ✓ S2 ✓ — S3 (G7 quantum
      foundations) is the last gate station.**
    • **▶ SYMPHONY-S3 (=G7) INTEGRATED (2026-07-09, pre-reg acf6167; implementation + adversarial check
      PASS; verify 73/73) → GREEN: the quantum-mechanics tie-in — AND THE PUBLICATION GATE
      (S1 + S1b + S2 + S3) IS COMPLETE.** `derivation_topdown/adapters/quantum_foundations.py`.
      **QF-1 THE BORN RULE = a MEASURED THEOREM (conditional on A3, printed):** exponent == 2 at 1.7e-18;
      mechanism verified (Ramanujan ⊥ Perron at 4e-16 ⟹ the Hermitian norm's square is what survives);
      the falsification probe detonates at **14–18 orders** under a 5% exponent deformation — checker-
      verified genuine (the deformed marginal is a valid distribution, still self-consistently geometric;
      only the INDEPENDENT β_eff cross-check bites — no normalization artifact, and a decoy was survived).
      **QF-2 THE FIRST BELL READ = an HONEST NEGATIVE:** the declared 2-plane bilinear family on the
      Dirac-sea vacuum gives S_max 0.094/0.015 (FAR/NEAR) — NO-VIOLATION-IN-FAMILY, no Tsirelson breach;
      NOT evidence of classicality; ROOT CAUSE traced by the checker: the real (flux-free) hopping forces
      the Majorana covariance's same-parity blocks to ZERO, gutting simple-bilinear correlations ⟹
      ▶ NAMED FOLLOW-UP QF-2b (smeared/multi-mode observables; complex-flux sectors). The mandated
      two-route check caught a sign bug PRE-sweep. DISCLOSURE booked: a 3-plane family was scratch-
      explored during construction (all S ≪ 2); the shipped family is the pre-reg's exact 2-plane
      (checker-verified in code). **QF-3:** derived GKLS + record-superselection pointer + KMS
      thermalization at rate u_c — re-expressed formula-for-formula. Scope: no measurement-problem
      claim; no A3-independent Gleason; no interpretation. **⟹ THE GATE IS COMPLETE: S1 ✓ S1b ✓ S2 ✓
      S3 ✓ — the pause/full-public/publication decision point (Symphony charter) is REACHED.**
    • **▶ THE EXPLOITATION WAVE (2026-07-09, user-approved post-gate; kill list = the completeness
      review a31850d): S1d + G8-IF + D1b in parallel → B2 → QF-2b. Per-delivery physics summaries
      required (user directive).**
    • **▶ WAVE-S1d (the epoch API) INTEGRATED (2026-07-09, pre-reg 715b11d; implementation +
      adversarial check PASS; verify-wired) → GREEN: N_hub IS the time variable, at code level.**
      the_run.py's S1d section (append-only, 270 lines, 0 existing lines touched): `N_NOW()` (single
      source of truth via read_higgs_chain), the 103-row `N_DEPENDENCE` registry (70 independent /
      25 calibration-curve / 5 power / 3 composition; 0 untagged; manifest N-tag column live),
      `read_epoch(N, p_era=None)` (float or numpy N). **THE CALIBRATION FENCE (the physics
      adjudication): v_higgs/G_F and everything downstream of v (all fermion masses, M_Z, m_W,
      widths, tan β, T_e_ann — 19 quantities) are the G_F tether's OWN DEFINING CURVE, not epoch
      physics — structurally EXCLUDED from read_epoch (checker probed pathological N incl. arrays:
      ZERO leakage; the fence is structural, not a filter).** Native epoch reads: H_sub ~ N⁻¹, t ~ N,
      Λ ~ N⁻², m_ν ~ N^(−1/2). Era exponents ALWAYS explicit (rad 1/2 / mat 2/3 / rec 1 ≡ MG-1c's
      2/n; NO default cosmology anywhere); the repo's two disconnected N-clocks RECONCILED
      (scale_bridge ≡ the p=1/2 special case at rel 0; its local constant's +1.05e-08 drift booked,
      prior-art file untouched). Locks 107/107 bit-identical; checker independently recomputed all
      anchor values at rel 0.000e+00. **B2/CMB's epoch prerequisite (the guardrail's requirement) is
      DISCHARGED.** Named residue: WHICH era holds at WHICH N (the a(N) selection) = ML-3's open
      dynamical crossing — the API takes p explicitly BECAUSE that equation is incomplete.
    • **▶ WAVE-G8-IF (the inner-fluctuation sector, first build) EXECUTED (2026-07-09, pre-reg
      8cfb10b; implementation + adversarial check PASS-WITH-NOTES, one sub-verdict REVERSED at
      integration) → ONE STRUCTURAL THEOREM + TWO FORCED NEGATIVES + ONE NAMED MISSING OBJECT; no
      scoreboard value moved (by pre-reg).** `proofs/foundations/G8_IF_inner_fluctuation_2026-07-09.py`
      (station probe; adapter promotion deferred — the structure findings are negatives, no permanent
      contract earned yet). First-ever A = Σa[D_F,b] on the certified KO-6 Cl(6) Fock; A_F PRIMARY =
      the derived gauge-charge algebra (closure dim 38, Burnside-verified independently by the
      checker); D_F = the whole certified 4-dim space. **(1) THE FORCED-HIGGS THEOREM: Ω¹(D_F) is
      100% γ_F-ODD — the internal one-form space is FORCED entirely Higgs-like, never gauge-like**
      (A_F commutes with P_F, D_F anticommutes ⟹ oddness unconditional; checker derived it by hand +
      verified 2.5e-15; bonus forced fact: J_F = σ_M0 ANTICOMMUTES with P_F, so the symmetrized
      A + J_F A J_F⁻¹ stays odd — a KO-6 consequence). **Ω¹ is D_F-INDEPENDENT (forced structure):
      dim 20 (PRIMARY) / 32 (COMPARISON) for ALL 7 D_F's, max principal angle 9.1e-15** (integration
      fix, documented in-file: the arccos-near-1 method had inflated ULP noise to ~4e-8 → the literal
      booking was DEPENDENT; the checker re-measured sin-based at ~1e-15 → REVERSED to INDEPENDENT).
      **(2) IF-2 ANATOMY-OTHER: the fluctuation content is Q-charged color triplet⊕antitriplet +
      sextet + singlet — NO adjoint/octet component** (Casimir clusters {0×2, 16/3×6, 40/3×12},
      adjoint 12 absent at 7.4e-16; the implementation pass CAUGHT ITS OWN miscalibrated adjoint proxy that
      would have falsely declared ANATOMY-MATCHED — pipeline teeth). NOT the SM Higgs doublet anatomy
      (color-singlet); the D2-scalar⇔inner-fluctuation unification hypothesis NOT supported at cell
      level in this construction. **(3) IF-3 SHAPE-MISS, FORCED BY CONSTRUCTION: the internal second
      moment is a PSD quadratic form ⟹ u1/su3 ratio ≥ 0 always, but b₁/b₃ = −2.2 < 0 because b₃
      carries the VECTOR SELF-ENERGY −3C₂(G) — no matter/Higgs-loop quadratic form can carry the
      gauge row. THEOREM-SHAPED WALL: the gauge row canNOT come from inner fluctuations of D_F alone;
      it needs the vector sector's own self-interaction** (consistent with LB-6: the Gilkey import
      stays). **(4) IF-4 NOT-WELL-POSED, the missing object NAMED: read_obliques() is 100%
      cover/dart-space (checker-verified zero internal-Fock reference) ⟹ the oblique confront needs
      THE VERTEX/PROPAGATOR MAP — the embedding of the 8-dim internal Fock into the 2|E|-dim
      dart-space current-current machinery — which does not exist anywhere in the framework; inventing
      it ad hoc would be goal-seek. That map is now the NAMED next-station object for the M_Z oblique
      — and it is plausibly the SAME missing internal↔cover bridge as ML-5's −70 ppm coupling gap:
      two walls, one named object.** BASIS NOTE booked: the two adapters' conventions (furey_stoica
      raw vs ncg_spectral W-basis) were combined here for the FIRST time via W†N̂W = diag (1.8e-15;
      checker verified the W-transport is a clean algebra automorphism at 3.4e-16).
    • **▶ WAVE-B2-a (the density response χ(q,ω) — the Lindhard build) INTEGRATED (2026-07-10, pre-reg
      146c0bf; implementation + adversarial check PASS-WITH-NOTES, both major diagnoses independently
      CONFIRMED AND STRENGTHENED) → the response object is BUILT and PERMANENT (the_net.py §4d:
      lindhard_chi0 + mermin_chi); final verdict INSTRUMENT-LIMITED per the frozen R-5 override; TWO
      major structural findings; MC-2's incomplete equation SHARPENED into a two-part named object.**
      `proofs/foundations/B2a_density_response_2026-07-09.py` (full 67 s; --fast 9.2 s; deterministic;
      accretion 120 ins/0 del, §4b untouched; c_s/γ/β never adjusted — grep-verified).
      **FINDING 1 (R-2, checker-proven): M2b's S(q) IS NOT THE DENSITY STRUCTURE FACTOR.** The FDT
      bridge S(q) = −(1/π)∫coth(βω/2)Im χ₀ dω was independently re-derived (Wick + detailed balance)
      and independently re-coded (agreement 3.4e-15 vs the net's vectorized χ₀); the bridge CONVERGES
      onto a from-scratch direct fermionic S_ferm(q) = Σ|M|²f(1−f) (51.6%→1.6% as ω-range/η tighten) —
      while M2b's Σ coth(βE(q)/2)/(2E(q)) sits **6–7 orders of magnitude away**: it has NO BZ sum, NO
      Pauli blocking, NO density vertex — it is a per-mode BOSONIC-OSCILLATOR ansatz evaluated at q,
      its own object, and MUST NOT be read as the density structure factor. **CALIBRATED CORRECTION
      (no overreach): the TRUE fermionic flat/cone ratio is O(7–21×) at the probed q's — flat-band
      dominance is REAL but O(10), not M2b's O(10⁴) (that magnitude was the bosonic ansatz's
      regulator-bound 1/E² divergence). The two-fluid picture (flat = matter seed) SURVIVES on its
      other legs (ML-3 quantum metric, diamond modular energy) at honestly reduced spectral magnitude;
      ML-4's θ_* discharge is unaffected (it rides the era structure, not M2b's magnitude). M2b is
      DISQUALIFIED as the density seed for B2-b/B2-c — the correct seed object is the net's §4d χ₀.**
      **FINDING 2 (R-4/R-5, checker-proven structural): NO SOUND AT THIS CLOSURE — and that is a
      theorem-shaped fact, not a miss.** ω_peak ~ q^1.86 (diffusive), c_pole rising 0.25→0.83 (axis;
      anisotropy caveat: varies ~2× across directions, verdict robust in all tested); the density-only
      Mermin/RTA closure conserves number ONLY ⟹ in the collisional regime γ ≫ c_s·q (here γ_micro =
      (1/2)ln2 = 0.347 vs c_s·q ≤ 0.046) it structurally yields DIFFUSION, never propagating sound;
      the γ-reduction diagnostic (10×–1000×, scratch-only) found NO hidden pole (the peak tracks a
      γ-independent ~2.7–3.1 = the bare continuum edge — and no interaction term exists in this
      deliberately coupling-free build to split a zero-sound mode). The checker's own interband-only
      probe: NO interior peak at any q/direction ⟹ the collective structure is entirely the
      intraband/flat term (R-5's override was right, and sharper than shipped). **⟹ MC-2's "the sound
      pole" incomplete equation is SHARPENED to a TWO-PART NAMED OBJECT: (i) the momentum-conserving
      (two-moment/BGK) collisional closure — the substrate lives at γ ≫ c_s·q, so this is the primary;
      (ii) an interaction/self-consistency (RPA-type) term for the collisionless alternative. Neither
      exists; BOTH must be DERIVED (a hand-inserted coupling constant would be goal-seek — the station's
      own poison, kept).** R-1: χ₀(q) static table converged (32³/40³ drift ≤ 0.7%). The c_s² = 1/3
      confront stays OPEN (never run in a valid regime — the closure cannot express sound). No
      scoreboard value moved. B2-b (growth) remains gated on a valid response closure + the epoch API.
    • **▶ THE F1 FIX-STATION EXECUTED (2026-07-10, LIGHT — architect-direct, the banked-artifact
      correction queued by I-0a): BOUND_F1's missing MIN applied as the PRINCIPLED form** (L_opt =
      min(compound, independent) — the observer takes the cheaper description; dS clamped ≥ 0
      identically), with the RESOLUTION docstring citing I-0a (3695851), its checker, and the June
      lab-note ('the min is not optional'). **The impossible repulsive bindings are gone; the
      attraction theorem is restored; the file's own two-route identity checks pass over all 8,100
      pairs + 277,020 triples (err < 1e-12, ALL CHECKS PASS); the adversarial checker had
      independently verified this exact clamp yields 8100/8100 agreement with two_subsystem.**
      Downstream: no file imports BOUND_F1 (grep-verified); the ΔS ladder {1,2,3,4,6,13} is the
      POSITIVE spectrum, untouched by clamping negatives to 0. The I-0a station file remains the
      historical record of the bug as found (its embedded pre-fix reconstruction intentionally
      unchanged). ⟹ {MDL vertex, E_bind} now stand as ONE OBJECT in code as well as in theorem.
    • **▶ I-0a THE RECONCILIATION EXECUTED (2026-07-10, pre-reg 684d90b; implementation + adversarial
      check PASS-WITH-NOTES — the checker's attacks STRENGTHENED the core verdicts) → THE INTERACTION
      LAYER'S SHAPE IS RESOLVED: ONE interaction object + ONE connection object + ONE found bug.**
      `proofs/foundations/I0a_reconciliation_2026-07-10.py`. **VERDICT (checker-split): (1)
      {MDL vertex −κ·I(A;B), E_bind = −κ·ΔS} = ONE-OBJECT-MODULO-THE-BUG** — the same theorem-object;
      the 52% pairwise disagreement is 100% explained by ONE located defect (BOUND_F1's dS()/L()
      omits the independent-vs-compound MIN that two_subsystem takes and the June lab-note documents
      as the deliberate correct choice; the unclamped formula yields IMPOSSIBLE repulsive bindings on
      single-shared-edge topologies); **the checker applied the hypothetical fix: 8100/8100 EXACT —
      zero residual. ▶ F1 FIX-STATION queued (LIGHT-MED: apply the min, rerun the 285k verification,
      audit downstream consumers of ΔS/E_bind, refreeze).** The static-limit T(E→−∞)=3=|E_bind|
      "corroboration" DOWNGRADED per checker: definitionally consistent with the shared DS_MAX=3
      convention, not independent evidence. **(2) W_INT = DISTINCT, and ROBUSTLY: two GENERAL exact
      mechanisms** — Γ⁵-parity (STRUCTURAL, checker-generalized: the MDL functional is Clifford-free ⟹
      any non-arbitrary embedding acts as internal identity ⟹ Γ⁵-EVEN; W_INT's single-generator
      decoration is Γ⁵-ODD) and block support (Hashimoto's diagonal is EXACTLY zero at every k — a
      topological fact, no self-loops); the checker CONSTRUCTED the one both-mechanism-evading
      candidate and proved it is W_INT itself reweighted — escape requires re-importing W_INT's own
      structure. **(3) THE SYNTHESIS (checker-endorsed as 'strongly consistent with', not proven — no
      Leibniz identity checked): W_INT is the CONNECTION/KINETIC piece (grade-1, no Higgs leg;
      LOOP_E2a's own framing was 'the interacting PROPAGATOR'), the MDL object is THE INTERACTION —
      Lagrangian roles, not rival vertices. The three forgotten drafts were two parts of one
      Lagrangian plus one bug.** R-2: the MDL object is Γ⁵-EVEN (PASSES the O3 filter — it can pay
      the chiral walls); W_INT Γ⁵-ODD. R-3 (checker-softened): the MDL object is FIELD-CONTENT-FREE —
      outside the three classified SM vertex forms (like W_INT, but not chirality-excluded); it is an
      information-metric CONTACT coupling, neither Yukawa nor |H|⁴. **⟹ I-0b DESIGN CONSEQUENCE
      (binding): the Schwinger/a_e acceptance benchmark MUST be formulated for a contact-type
      coupling** (a pure contact term's contribution to an anomalous moment differs structurally from
      QED's vertex-type α/2π — whether it generates one at leading order at all is the design
      question; no benchmark may be run before this is settled in the I-0b pre-reg). R-4
      (declared-minimal): admissible on-site space dim 32 (A4/twisted-locality cuts NOT applied —
      upper bound); the MDL object inside, W_INT outside. Citation fix applied in-file (the I=−1 quote
      is from the lab-note, not the docstring). No scoreboard value moved; κ never adjusted; no
      fourth object.
    • **▶ THE α₃₁ RESOLUTION EXECUTED (2026-07-10, LIGHT PROCESS — architect-direct, per the new
      effort-scales-with-goal-seek-risk policy; no target exists to seek: m_ββ/α₃₁ are unmeasured) →
      m_ββ IS NOW SHARP: 3.5644 meV.** Executes the 2026-06-11 Majorana-panel's own finding, frozen
      since as an in-file INCONSISTENCY FLAG: the adopted M_R = |M_R|·diag(1, h_ω^g, h_ω²^g) anchors
      Majorana phases to eigenvalue 1 ⟹ α₃₁ = arg(h_ω²^g) = (−g·arg h) mod 360° = 360° − α₂₁ =
      **197.612°** (the conjugate C₃ channel; α₂₁ = 162.388° unchanged and self-consistent). The
      previously frozen 324.775° = 2g·arg(h) was (φ₂−φ₃), the 2-vs-3 relative phase — a DIFFERENT
      quantity, internally inconsistent with α₂₁'s own anchor; srs_unified_mixing §8 already used
      197.612°. EXPONENT FORK ALSO ADJUDICATED: the framework's α's are LITERAL eigenvalue arguments
      ⟹ they enter m_ββ exactly once (k=1); the k=2 reading demoted to a labelled diagnostic
      (3.1446 meV). Physical relative phase α₃₁−α₂₁ = 35.225°. Deliberate single-value re-freeze:
      alpha_31_PMNS 324.775→197.612; harvest m_bb conv1 1.539→3.564 (adjudicated), conv2 → 3.145
      (diagnostic); all other 131 locks untouched (diff-verified). FALSIFICATION STANCE UNCHANGED:
      3.56 meV is below nEXO/LEGEND-1000 reach; any positive detection still falsifies outright.
      ▶ THE TIMESTAMP DROP IS NOW UNBLOCKED (its m_ββ content is sharp).
    • **▶ RING-1 THE HARVEST INTEGRATED (2026-07-10, pre-reg f52b2f9; implementation + adversarial
      check PASS-WITH-NOTES — every number independently reproduced, incl. bit-identity of the
      original 107 locks vs git history) → THE FIRST NEW CONFRONTED NUMBERS IN MONTHS; coverage
      98→114; locks 107→134.** `proofs/foundations/R1_HARVEST_2026-07-10.py` (verify-wired) + the
      engine's appended R1 section + additive manifest/lock wiring. **m_ββ (NEW ROW, checker-language
      booking): 1.5–3.6 meV depending on phase-convention placement — TWO forks booked: (i) the ×2
      exponent-placement ambiguity (station-computed); (ii) THE DOMINANT FORK: the pre-existing
      UNRESOLVED α₃₁ inconsistency (flagged 2026-06-11 in predictions/alpha_31_PMNS.py: frozen
      324.775° vs adoption-consistent 197.612°) — m₁=0 makes only (α₃₁−α₂₁) physical, so THIS fork
      gates sharpness; ▶ named station: resolve α₃₁ before m_ββ is a point prediction. HONEST REACH
      (wording fixed in-file per checker): the band sits BELOW nEXO (5.7–17.7 meV) and LEGEND-1000
      (10–20 meV) projected sensitivity — consistent-with, not testable-by, next-gen; THE
      FALSIFICATION STANCE: any POSITIVE detection by nEXO/LEGEND falsifies the framework outright.**
      **Σm_ν = 59.4 meV — inside the Planck+BAO bound but AT the normal-ordering floor that DESI
      BAO+CMB combinations are actively squeezing (the contested zone): live falsification pressure,
      not a comfortable margin.** **Ω_ch² = +5.67σ vs Planck (checker-recomputed) — CONDITIONAL on
      the adopted z_eff (shares the Ω_DM row's root cause) BUT in physical-density units it is now,
      in its own right, a formally >5σ OPEN miss — booked OPEN/CONDITIONAL, foregrounded, never read
      as corroboration.** Coasting chain computed (q₀=0, w_eff=−1/3 exact; D_L(z=1)=6095.8 Mpc;
      Category-B genre throughout; the t₀ CMB-frame +23.7σ declared contrast carried raw). **δρ
      ADJUDICATION = ARTIFACT (checker re-ran and confirmed): the 2026-05-15 intra-vertex probe
      answers a DIFFERENT question (any custodial breaking at single-vertex level? — exact null,
      Δρ=0.0000%) — not a match to the +4.58% row, which is UNCHANGED and stays OPEN; the archaeology
      sweep's "sub-percent claim" characterization was the misreading (docstring skim). The
      cheapest-mover expectation is REVERSED — booked.** H-5 wiring: 6 WIRED (checker spot-ran all —
      no stretches) / 1 PARTIAL (gauge group: continuum promotion stays a declared adoption) / 5
      HONEST ORPHANS (Higgs rep, matter stability, low initial entropy, branch measure, observer
      Hilbert space — named wiring targets, not stretched). Minor bookkeeping per checker: the newly
      wired split is 14 Tier-A + 2 Tier-B (not 13+2); the aggregate 98→114 independently confirmed.
    • **▶ I-0b DESIGN NOTE WRITTEN (2026-07-10, architect adjudication) →
      internal research notes: THE BENCHMARK CATEGORY ERROR FIXED.** The
      Schwinger α/2π is the one-loop correction to the electron–PHOTON vertex — a CONNECTION-sector
      (W_INT/finite-k) observable, NOT a test of the MDL contact coupling (EFT lore: contact
      operators enter a_e only at (m/Λ)²-class suppression). REDESIGNED GATES: (1) I-0b-RATIO —
      binding-energy RATIOS from the ΔS ladder are κ-FREE (E_a/E_b = ΔS_a/ΔS_b), the vertex's
      acceptance test, MEDIUM, with the hadron/nucleus-ID mapping's EP-2-class adoption DECLARED;
      (2) I-0b-SCALE — the composite-scale κ bridge (the June arc's own walled magnitude), target
      B_d = 2.2245 MeV; (3) the Schwinger gate RE-SCOPED to the connection sector's future
      vector-sector program. **CONSEQUENCE: Paper IV partially REORDERS — IV-2 (composites) is the
      contact vertex's DIRECT first cascade and comes forward; a_e/a_μ + decay rates move behind the
      connection build; the contact-vertex and connection tracks are PARALLEL.** Next: I-0b-RATIO
      pre-reg. Poisons: no κ tuning; the ladder-ID adoption declared, never fitted; a RATIO-MISS is
      a result, not a failure to bury.
    • **▶ PAPER I DRAFTING TABLED + ARCHITECTURE REFRAMED (2026-07-10, USER DECISION — resume
      trigger: user says "ready to draft Paper I").** The claim-sheet
      (internal research notes — relocated from docs/papers_v2/, a directory-policy
      violation now repaired; paper material lives under `papers/`) was REJECTED AS ARCHITECTURE,
      retained as CONTENT: it re-presented the sector-scoreboard paper ("look at this structure"
      per physics sector — the SM's own org chart), which the bins memo already ruled out
      ("papers by LAYER not sector"). **THE REFRAMED FORM (user's direction, architect-endorsed):
      the RECOGNITION paper — mathematics first.** Movements: (1) the OBJECT defined + SELECTED as
      pure math — **SELECTION PER THE R-9 SUPERSESSION (2026-06-15, structural_residue_register.md),
      NOT Sunada-uniqueness**: Sunada strong-isotropy is cited as MOTIVATION/context only (his
      theorem stands as mathematics, but our own probes proved the old chain wrong — srs-z IS
      arc-transitive, strong isotropy carries no selection load); the live selection = the declared
      candidate class (V+E-transitive chiral 3D nets, 9 RCSR candidates) → structural-fingerprint +
      MDL-waterline study → **srs DOMINANT (survivors {srs, srs-c8, lou, lov} disclosed)** + the
      Laves L³/V=27/√2 extremal handle (unique attainment; common-cause caveat booked) → NB walk as
      canonical dynamics; zero physics vocabulary; (2) the FORCED INVARIANTS computed as theorems (Perron
      data, walker root (√3+i√5)/2, girth 10, rank-3 C-projector, Cl(6)/KO-6, A4 sectors, tick-2π);
      (3) THE RECOGNITION: one table, invariant ↔ measured dimensionless constant (25 within 1σ),
      the voilà EARNED because every LHS was verified as math before physics entered; (4) the AUDIT
      AFTER the reveal (MDL +168 bits, look-elsewhere, receipts — repositioned from face to
      honesty-appendix); (5) opens carried + forward falsification. GUARD: the reveal scoped to the
      DIMENSIONLESS SKELETON; the (D, ω, {A(O)}) three-layer split is the native disclaimer for
      magnitudes/dynamics living in Papers II–IV. The claim-sheet's 65-row accounting, opens
      section, and pre-mortem survive as §3/§5 content; the three claim-sentence variants are MOOT
      until the reframe is drafted. Timestamp drop v4 UNAFFECTED (user-approved, upload pending).
    • **▶ T0-NUCLEAR EXECUTED + INTEGRATED (2026-07-10; pre-reg c66fdf9; the CONSTRUCTION-MISMATCH
      stop-clause FIRED AT THE GATE — the discipline worked; architect re-run green; verify-wired)
      → NO 3-body number was ever produced; the RATIO-MISS stays booked unchanged. THE FINDING IS AN
      IDENTITY:** the frozen co-information functional gives I=5 on ALL 648 ground pairs and
      C₃=15 (with II₃=0) on ALL 216 ground triples, while the frozen rungs are 3 and 13 — **the gap
      is EXACTLY the branch-realization cost of 2 in both cases: rung = overlap-information − branch
      cost (3 = 5−2; 13 = 15−2).** Structurally: stage3a's ΔS = [Σ_e(mult−1) − Σ_v max(deg−2,0)]
      decomposes as (edge-overlap info) − (vertex-branching cost); the vertex file's co-information
      counts ONLY the edge-overlap half — its own honest-bounds note said so. **⟹ NUANCE TO I-0a:
      at composite scale the interaction functional and bare co-information are NOT identical — they
      differ by the vertex-degree term; the F1 THEOREM's object is ΔS (with the branch cost), so the
      binding potential must be built from the ΔS convention itself.** SECOND FINDING: **II₃ = 0 on
      every ΔS=13 ground triple — the 3-body ground state carries NO irreducible junction core**
      (pairwise-reducible; the departure from pairwise additivity is carried entirely by the branch
      cost). My pre-reg error owned: it froze two mutually inconsistent conventions (co-information
      V̂ + rung normalization). ▶ NAMED: **T0-NUCLEAR-2** — identical station with V̂ built from the
      ΔS convention verbatim (the stage3a formula as the depth function); the gate becomes
      rung-reproduction by construction; everything else (Jacobi solve, frozen boxes, mirror-mean
      confrontation, verdicts) carries over unchanged. The implemented-but-gated 3-body solver is
      reusable as-is.
    • **▶ MS-1a EXECUTED + INTEGRATED (2026-07-10; theorem-build, architect re-run green;
      verify-wired) → MS1a-THEOREM (milestone III.2's theorem half DONE; MS-1b operator/suppression
      half stays gated on the vertex).** `proofs/foundations/MS1a_fusion_grading_2026-07-10.py`,
      54 checks. **THE NO-ADDITIVE-CHARGE THEOREM, COMPUTED:** character tables from scratch
      (Dixon/Burnside, exact arithmetic), fusion rings verified; additive Z-valued charges on
      R(A4) and R(2T) have solution space {0} EXACTLY (rref over Fractions; nullities 0/0); the
      forcing chain is human-readable (3⊗3∋3 ⟹ q(3)=0; ...). **BONUS THEOREMS: fermion parity is
      the UNIQUE nontrivial Z₂ grading of the sector category (no R-parity-like second Z₂ exists);
      the center-even sub-ring of R(2T) ≅ R(A4); N̂ (hence 3Q) does NOT descend to the fusion ring —
      only torsion/parity data survives fusion.** Winding regression: ⟨0|U_π²⟩=i/2 bit-exact; honest
      extra finding: the screw also fails N̂-commutation (max|[U_π²,N̂]|=1.12), a SECOND independent
      gauge-test failure. Premises declared: TD-limit twisted-duality conditionality (ML-2b);
      single-cell HK-7 scope. ⟹ matter stability is now THEOREM-BLOCKED from exact conservation
      (not merely assessed) — the suppression statement (MS-1b) is the only remaining route,
      consistent with η_B's Sakharov requirement. ACCRETION-PASS addition: the fusion-ring +
      grading-enumeration machinery → the_net.py.
    • **▶ BRIDGE-T EXECUTED + INTEGRATED (2026-07-10; pre-reg 661806a frozen BEFORE impl; architect
      re-run 37 checks green; verify-wired) → VERDICT: ARROW-BLIND, THEOREM-GRADE (milestone II.2
      route B closed; the null is now a much stronger theorem).** `proofs/foundations/
      BRIDGE_T_2026-07-10.py`. T-1 PROVED: the two O(2) orbits pull back the conjugate cell vacua
      C(±J) — modular generators of OPPOSITE sign (M0-4b transported verbatim; the arrow question
      was well-posed). T-2: the state-level datum is genuinely non-R-definite (||RG∓G|| ≈ 1.42 both)
      — the escape hypothesis was real — BUT the datum evaluates to machine zero on both orbits, and
      the zero is FORCED by two new lemmas (both asserted): (i) A4 acts simply transitively on darts
      and commutes with {B, J_D, P_odd} ⟹ every seeded datum equals its own gauge average; (ii) THE
      SYMMETRIC-COMPRESSION LEMMA: F(u) = Uoᵀ(I−uB)⁻¹Uo is EXACTLY symmetric AND forward/reversed-
      invariant (Uoᵀ(I−uB)⁻¹Uo = Uoᵀ(I−uBᵀ)⁻¹Uo, dev 7e-18) ⟹ NO seed-anchored, state-level,
      all-orders-in-B two-point datum on the R-odd dart sector can EVER see the arrow or the orbit.
      Controls prove the kill is the symmetry's (generic antisym pairs at 0.093; breaking A4-only
      gives per-seed reads with zero average; breaking both gives forward≠reversed). Disclosed
      interpretation: the literal equal-time form is degenerately null by reality (real |G⟩ ⟹
      symmetric M); the pre-reg's "or equivalent" lag form declared PRIMARY before evaluation —
      both computed. **⟹ THE DISCRIMINATOR MUST LEAVE THE CLASS ENTIRELY: phase-bearing/Fock-level
      or GEOMETRIC. II.2's live route = BRIDGE-GEOM (Design C, the mirror-lattice theorem at finite
      k). ACCRETION-PASS addition: the symmetric-compression lemma + the A4-quotient lemma →
      the_net.py.** M_Z stays OPEN; no oblique quantity touched.
    • **▶ IV.4 T0-CLASS EXECUTED + INTEGRATED (2026-07-10; pre-reg 4db589f frozen BEFORE impl;
      architect check = independent re-run, verdict reproduced; verify-wired) → VERDICT: T1-FENCE +
      STAGE-0 NON-RIGID.** `proofs/foundations/IV4_T0_class_2026-07-10.py`. (1) **T₀ IS REAL AND
      DERIVED: T₀⁽²⁾ = U − B = 0.3311 (11.04% of U), box/grid-converged, from the frozen 2026-05-29
      H_rel machinery (regression <2e-4)** — the sealed A2's reading (b) now holds AT THEOREM GRADE
      in-framework, not just empirically. CALIBRATION: 11.04% is the SAME ORDER as the nuclear −16%
      (0.69×) ⟹ T0-NUCLEAR (3-body) stays LIVE, not under-powered. (2) **THE CLASS THEOREM: at the
      untuned operating point (U=3) the MDL contact vertex is DEEP-CONTACT class, B ≈ U − c·s with
      ∂lnB/∂lnμ = +0.12 (predicted 0.126 from the anatomy — matches); exponent +1 exists ONLY on a
      tuned near-critical locus. Internal coherence: STATIC(U,s) = EQUAL(U,s/2) EXACTLY (the
      reduced-mass mapping realized numerically). Raw B_static/B_equal = 1.0597, nowhere near 2.**
      ⟹ **H/Ps (linear-μ at 10.9 ppm) is ADJUDICATED OUT of the contact vertex INTO the CONNECTION
      sector (the finite-k photon, IV.7): the atomic block (r_p, H 1S–2S, Lamb, 21cm, Ps 1S–2S, Ps
      lifetimes) RE-HOMES IV.6→IV.7 in the milestone register.** This SHARPENS I-0a's anatomy: the
      contact vertex owns short-range binding (nuclear, T₀-corrected); long-range/Coulomb-class
      binding is the connection's. Disclosed adjudication: the working band = the 2026-05-29 file's
      own 32×32 lowest positive Dirac band (its U_c≈0.26); the 4-band scalar-Bloch band found
      pathological for this convention (inspected only, no number computed on it). ▶ NEXT: the
      T0-NUCLEAR pre-reg (3-body f_kin, target-blind conventions frozen from the 2026-05-29 file)
      + the IV.7 connection design note now carries the atomic block AND the Schwinger gate.
    • **▶ DISCIPLINE AUDIT (2026-07-10, user-prompted post-compression check) → THREE DRIFTS FOUND
      + CORRECTED:** (1) **ACCRETION DEBT booked:** today's station files (BRIDGE_LOCK's three lemmas
      + O(2) family; MS-1a's fusion-ring theorem when it lands; the I2/I2b CK-Toeplitz structure)
      are verify-wired standalone but NOT yet folded into the master modules — ▶ NAMED: THE
      ACCRETION PASS (one station: BRIDGE_LOCK lemmas + MS-1a fusion category → `the_net.py`
      [Layer-3 law]; I2 dictionary → `thermal_time.py` adapter; regression anchors preserved;
      verify entries repointed, not duplicated). Do NOT accrete un-checked station math — only
      integrated results. (2) **E_odd + T₀ ledger rows registered** (the 100%-disposition invariant
      requires every framework-named measurable/object on the ledger; milestone register subtotals
      to be bumped at the next register touch: IV gains 2 object-rows → universe 210). (3) **Full
      verify on main never run since the 3 new wirings (80th–82nd entries)** — running now,
      background. Standing answer to the one-math question: Layer-1 reads DID accrete (read_T_nu_dec
      in the_run.py, append-only); Layer-3 accretion is the debt named above; NO fork of a second
      program occurred (every station extends existing objects or confronts them).
    • **▶ BRIDGE-LOCK EXECUTED + INTEGRATED (2026-07-10; pre-reg 8df58a0 frozen BEFORE implementation;
      verify-wired) → VERDICT: LENS-NULL, THEOREM-GRADE (milestone II.2, route A closed cleanly).**
      `proofs/foundations/BRIDGE_LOCK_2026-07-10.py`. The A5-lock attachment functional canNOT
      discriminate the two O(2) orbits — and the null is FORCED by three machine-checked lemmas:
      (1) R = −Id on the whole R-odd sector (R·Uo = −Uo) ⟹ every transported subspace is
      R-invariant; (2) R·B·R = Bᵀ (reversal-transpose); (3) B real ⟹ the band-edge projectors are
      complex conjugates ⟹ Δ ≡ 0 identically for ANY R-parity-definite transported content (both
      branches attach 0.75/0.75; control: a non-R-definite vector gets 1.0/0.0 — the functional
      discriminates perfectly outside the R1-forced class). **The MASTER CHIRALITY LENS's predicted
      failure mode is now a theorem: W2-MAP's own R1 requirement (which carved out the O(2) family)
      is precisely what makes the family band-edge-blind.** FS-5iii discharged (deck-invariant
      subspace only); L-3 executed operationally — the null is NOT a convention artifact (the three
      lemmas contain no sign convention). The banked ν↔chir-7 lock is UNTOUCHED (orbit-discrimination
      failed, not the lock). ▶ II.2 HAND-OFF per the dossier: Design B (BRIDGE-T, the modular-arrow
      discriminator — orbit(2) pulls back the CONJUGATE vacuum C(−J), σ reverses modular flow, the
      arrow is derived; the datum must be second-order/state-level, NO linear-in-B functional) and/or
      Design C (BRIDGE-GEOM, the mirror-lattice theorem at finite k). M-2/M-3 stay gated; M_Z (+7.76σ)
      stays OPEN.
    • **▶ I-2 MATSUMOTO CONFRONT INTEGRATED (2026-07-10; impl + adversarial check PASS-WITH-NOTES,
      all 5 mandatory fixes applied; literature VERIFIED with sources; verify-wired) → VERDICT:
      SAME-OBJECT (QUALIFIED) — THE LANDAUER LOCK GAINS AN EXTERNAL UNIQUENESS THEOREM (one
      construction from bookable).** `proofs/foundations/I2_matsumoto_confront_2026-07-10.py`.
      THE CROWN IDENTITY (machine-checked all k): β·κ = h_top/b_edge = ln(k−1)/log₂(k−1) = ln 2 —
      our per-bit Landauer temperature IS the graph-algebra unique-KMS-at-topological-entropy point
      (Enomoto–Fujii–Watatani 1984) expressed per bit; unit-mediated, never numerology (S4a is pure
      change-of-base — booked as DISAMBIGUATION). Object-level: the SFT = the run's own history set
      (words = 12·2ⁿ⁻¹ triple-verified); their gauge circle = T4b's 2π; Parry fugacity 1/r = u_c;
      the Toeplitz KMS geography (aHLRS Thms 3.1+4.3, VERIFIED: no KMS below ln ρ; simplex above;
      UNIQUE at ln ρ factoring through O_A) = the sub-criticality arrow — the two-temperatures
      discipline gets an external name, and the SEED = the Toeplitz vacuum defect ΣSS* = 1−P_seed.
      WHAT IMPORT ADDS: UNIQUENESS (we proved consistency forces the point; they prove it is the
      ONLY one) + Parry-MME identification. FENCED framework-only: κ = h/t_P magnitude; the currency
      premise E=κL; the Born-2 layer (β_eff = 2(β_gas−h_top) is DEFINITIONALLY exact — dictionary
      entry, never corroboration). ▶ NAMED COMPLETION STATION (not run): represent S_d on H_hist,
      check ΣSS* = 1−P_seed, gauge-average/diagonal-restrict FIRST (the checker proved a naive
      off-degree KMS check FALSE-FAILS — every KMS state is flow-invariant, the run vector is not
      gauge-invariant), and pre-register the dart-algebra temperature β′ = β_eff + h_top =
      2β_gas − h_top = 5.7942945492 (machine-pinned S6a) — NOT β_eff, NOT β_gas. Until it lands,
      SAME-OBJECT is booked for the subshift/grading/critical-temperature/critical-diagonal only.
    • **▶ PAPER-III MEDIUM ASSESSMENTS INTEGRATED (2026-07-10; impl + adversarial cite-check
      PASS-WITH-NOTES ×2, all mandatory fixes applied in-doc with [CHECKER FIX] notes) →
      internal research notes;
      both ledger rows (⚙️) now carry PARTIALLY-FORCED-GAP-NAMED classifications.** (1) MATTER
      STABILITY: no exact symmetry in the derived inventory forbids p→e⁺π⁰ (Q conserves it; B−L
      adoption-conditional and conserving; winding Z₃ not DHR; finite sector category ⟹ no additive
      B̂; binding does NOT select color — the checker caught the draft INVERTING C4d's NEGATIVE) —
      consistent with η_B's Sakharov requirement ⟹ stability = a SUPPRESSION statement; build-task
      MS-1 named. (2) LOW INITIAL ENTROPY: arrow = theorem; the arrow's reduction-to-datum = booked
      DERIVED (the draft had INVERTED the CLEANROOM [F] legend — fixed); the datum itself + the
      register↔thermodynamic bridge (LE-2) + the gravitational/Weyl half remain named gaps; LE-1
      rebased on the b_edge=1 counting fact after the checker caught an S_fresh smuggle (the OEF's
      own §16 disclaimer). Checker lessons banked: probe results must be RUN not paraphrased;
      legend keys verified at source; composition premises named or the claim is demoted.
    • **▶ I-0b-RATIO EXECUTED + INTEGRATED (2026-07-10; pre-reg 13ec0ca + sealed A-RULE 3695bbe
      frozen BEFORE Stage B/C; scout→sealed-rule→implement→architect-check; verify-wired) →
      STATION VERDICT: RATIO-MISS + TWO STRUCTURAL FINDINGS + LADDER SUPERCELL(4)-STABLE.**
      `proofs/foundations/I0b_RATIO_stage_BC_2026-07-10.py`. (1) PRIMARY MISS (booked OPEN, no
      re-rule): the sealed max-ΔS rule's parameter-free 13/3 vs B(³H)/B_d = 3.8128 (−12.01%) and
      B(³He)/B_d = 3.4695 (−19.94%) — the miss falls on the law's completeness and/or the
      geometry→composite dictionary (both already-named opens); which leg fails is NOT adjudicated.
      (2) **INCOMPLETE EQUATION NAMED-AND-EXHIBITED (A2 → reading (b)): E_bind = −κ·ΔS is the
      TOPOLOGICAL FACTOR of an incomplete equation — H/Ps = 1.998933 equals the reduced-mass factor
      2/(1+m_e/m_p) to +10.9 ppm ⟹ the missing term is T₀(μ_eff), the relative-motion/inertial
      sector (binding ∝ μ_eff within one rung at leading order). THE completion chase: derive T₀
      from constituent dispersion on the lattice — this is the vertex program's next defining
      equation.** (3) **E_odd MEASURED (A3): B(³H)/B(³He) = 1.0990 vs forced 1 ⟹ E_odd = 0.381876
      MeV — the first measurement of the mirror-odd channel (the σ-odd sector the even ΔS invariant
      cannot carry; standardly the Coulomb displacement — for us an un-priced named gap, not
      retro-fitted).** R1: ladder {1,3}/{1,2,3,4,6,13} IDENTICAL on supercell(4) (13 ceiling + 2-body
      gap survive; the supercell(3) ΔS=−3 negative-tail bin was a wrap artifact — binding ladder
      unaffected). R2 sign drift fixed (2 docs); R3 EXP-B4/B7 category fix applied (mass≠binding;
      no 2-body ΔS=2). AME2020 digit deltas ≤1.7 eV (declared values kept). Poisons held: no κ, no
      scanning, no alternative rungs. ▶ NEXT: I-0b-SCALE (κ bridge) now needs T₀ FIRST — the scale
      station's design must price the inertial factor or it will mis-attribute the κ magnitude.
    • **▶ THE LIGHT BATCH EXECUTED (2026-07-10, architect-direct per the effort policy; booked
      honestly, no agents, no pre-reg — no goal-seek risk in any item):** (1) **T_ν_dec ENGINE
      SURFACE BUILT** — the_run.py `read_T_nu_dec()` (faithful predictions/T_nu_dec.py v2.0.0 port,
      α=1/2 instantaneous rate balance, calibration-curve family via the G_F tether), Tier-A-mapped
      in reads_manifest.py, EXACT float match to lock 0.8443997597588065 MeV; manifest now
      117/117 Tier-A, unmapped locks 6→5; the S1b orphan note superseded in place. (2) **BRANCH
      MEASURE + OBSERVER HILBERT SPACE WIRED TO THE QF ANCHORS** (two of R1's five honest orphans):
      ledger rows now cite QF-1's measured Born theorem (exponent 2 at 1.7e-18, conditional on A3)
      and the G7 suite's declared CDP-2011 premise consumption; observer_hilbert_space's
      lock-registration stays format-blocked (schema fact, not physics). Remaining named orphans:
      Higgs rep, matter stability, low initial entropy (the two MEDIUM Paper-III assessments queued).
      (3) **A_s 16/15 RE-AUDIT (Paper I dependency) → FINDING: FRAME-ASSIGNMENT INCONSISTENCY,
      BOOKED OPEN** (see the A_s ledger row note). The declared D2-extended rule prices the
      observer/substrate gap per power of the RATE (H¹→16/15, Λ=H²→(16/15)²); A_s is dimensionless
      with no written per-observable derivation of its power-1 assignment in
      `proofs/cosmology/As_feshbach_exponent_upgrade.py` (one-line "Item 1 closure" citation), AND
      the comparison frame conflicts with the H_0 row's own convention (H_0: substrate side vs
      Planck/CMB, observer side vs SH0ES local; A_s: OBSERVER-corrected value vs PLANCK). The −2.1σ
      is therefore CONDITIONAL on an unadjudicated frame assignment: bare/substrate −6.3σ; ×16/15
      −2.1σ; ×(16/15)² +2.4σ. This is the MC-track's right-quantity-wrong-clock genus (which frame
      Planck's A_s INFERENCE lands in is a property of the inference pipeline). NO value/status
      change (row stays 🟡; Paper I already carries A_s open — this audit RATIFIES that choice).
      ▶ Named resolution owner: the MC/ML clock-map layer (the same object as θ_*/z_eq), NOT a
      standalone A_s fix. Poison: do NOT pick the power that lands closest (that is the goal-seek).
    • **▶ WAVE2-GAUGE-A (the abelian a₂ magnetic-supercell difference trace) INTEGRATED (2026-07-10,
      pre-reg in 8ca645c/6e03d95, file landed in 3d791c4; implementation + adversarial check
      PASS-WITH-NOTES, full station independently re-run by the checker) → VERDICT:
      WINDOW-LIMITED-AT-PILOT (verified — the pre-reg's declared default), WITH THE NO-GO QUANTIFIED
      INTO A THEOREM-SHAPED STATEMENT + one standalone new result.**
      `proofs/foundations/W2_GAUGE_abelian_a2_2026-07-10.py` (--fast 0.6 s byte-stable, verify-wired).
      **THE INSTRUMENT IS THEOREM-TIGHT:** cell-periodic triviality gate 1.3e-15 (the cover
      gauge-triviality theorem re-verified in-code); folding check (B=0 supercell ≡ L× unit cell)
      3.1e-11; magnetic-Bloch construction from the DERIVED Albanese geometry (A23 = det(L)/|a1| =
      2/√3, checker-recomputed; field-axis WLOG by the σ=(123) Gram-preserving symmetry,
      checker-verified). **THE FLUX-QUANTIZATION NO-GO (the lasting content, checker-recomputed
      exactly): on a lattice, B is quantized from below at any supercell size; the weak-field
      Seeley–DeWitt window (B·t < 1) across t ∈ [30,240] requires L_CELLS ≳ 163–1306 ⟹ fiber
      dimensions ~1,600–13,000 — a NO-GO for exact diagonalization at reachable sizes. Every feasible
      B sits deep in the Landau-suppression regime (B⁴/B² = 17.6–20.0% vs the 10% gate; cone-sector
      suppression ~99.96% at t=240). SCOPE: NOT a no-go for the LINEAR-RESPONSE-IN-B route
      (∂²K/∂B² at B=0, Kubo-type, no supercell) — the named future route, which CONVERGES with the
      archaeology dig's proven Π_JJ Kubo engine (gauge_beta_from_substrate_kubo_probe family).**
      SIGN READ (checker-downgraded, fix applied in-file): ΔK < 0 everywhere = a diamagnetic-sign
      sanity check, NON-DISCRIMINATING in this regime (near-guaranteed by diamagnetic-inequality
      behavior; rules out sign bugs and a paramagnetic flat-band response, no more — never book as
      corroborating −B²/6's magnitude). **LB-4′ STANDALONE RESULT (checker-reproduced −16.000000 exact):
      THE FLAT BAND IS NOT LIFTED BY THE FIELD — the flat-mode count (rank of the incidence operator)
      is EXACTLY unchanged at B ≠ 0.** The falsifiable content is the RANK invariance (the Str = −2L
      identity is a McKean–Singer tautology at any B — never cite it as the evidence). Speculative
      flag (one line, labeled): the flat band = the framework's dark/matter component, and it is
      magnetically inert to machine precision — darkness as an index-adjacent statement; NOT banked.
      CHECKER FINDING FIXED IN-FILE: the Q2-periodicity check is tautologically 2π-periodic for ANY B
      (bare-phase construction) — a construction sanity check, NOT an independent verification of
      quantization (docstring corrected; quantization is guaranteed by definition via flux_B).
      Stopping-clause reading adjudicated DIRECTLY LICENSED (the pre-reg's parenthetical defines
      "feasible"). The symbolic target −B²/6 printed live from d4, never fit; ladder frozen; matter-a₄
      anchor printed. Non-abelian NOT attempted (out of scope, stands). No scoreboard value moved.
    • **▶ WAVE2-BGK (the two-moment conserving closure) INTEGRATED (2026-07-10, pre-reg in
      8ca645c/6e03d95; implementation + adversarial check PASS-WITH-NOTES — the checker reproduced the
      FULL station byte-identically on its own GPU run, 278/278 lines) → BGK-3 VERDICT: NO-SOUND,
      THEOREM-GRADE: even with quasi-momentum exactly conserved and ZERO free parameters, the derived
      bands produce NO propagating sound at the coupling-free level.** the_net.py §4d gained the
      velocity vertex + the two-moment conserving-RTA closure (velocity_operator, moment_chi0_matrix,
      closure_from_moments, two_moment_chi — permanent; single clean append, checker-verified).
      `proofs/foundations/W2_BGK_two_moment_2026-07-10.py` (full 2m37s GPU / --fast 2.6s CPU).
      **THE EVIDENCE CHAIN:** (i) the closure math independently re-derived by the checker (the
      Mermin two-moment closed form; conservation exact ~1e-15; scalar limit == B2-a's mermin_chi at
      2e-15; GL(4) covariance); (ii) the velocity vertex anchored to cone_velocity at 1.3e-10 (the
      checker's own fresh finite-difference, tighter than claimed); (iii) **THE POSITIVE CONTROL: the
      identical closure code on a classical Maxwell gas finds textbook isothermal sound (p=1.008,
      c=1.000–1.018 vs √T=1) and passes the T=4 cross-check (c→2.00–2.08 vs √4) — the instrument
      finds sound where sound exists**; (iv) the srs bands: no interior peak at q=0.01 in 3/4
      directions + dispersion exponent p=1.93–2.01 (diffusive) in all four (⟨100⟩/⟨110⟩/⟨111⟩/⟨210⟩),
      ω-ladder clean (≤1.53%), grid-converged (n_grid 32 vs 40 identical — the checker filled the D7
      prose-only gap itself). **THE MECHANISM (94.53%/13.88%, checker-reproduced): the flat band
      carries ~95% of the density weight but ~zero velocity content — THE MATTER COMPONENT IS
      IMMOBILE; conserved momentum has nothing to push** — a velocity-based, Fermi-statistics-clean
      corroboration of the two-fluid structure, INDEPENDENT of M2b's demoted bosonic ansatz. **D9
      AUDIT (git-archaeology-verified legitimate): the first run's verdict logic made the frozen
      "no-interior-peak ⟹ NO-SOUND" clause unreachable dead code; the fix restored the frozen text;
      zero measured numbers changed (diff-verified); the correction moved TOWARD the more falsifiable
      claim.** BOOKED PATTERN (3rd instance, new face): a docstring PROMISED a check (n_grid=40
      sanity) never implemented in code — promised checks must EXECUTE; prose-only checks are the
      sweep-misquote pattern's sibling. **BOOKINGS (checker-adjudicated language): (1) the acoustic
      story (θ_*'s c_s loading; M2c's radiation-era sound) requires the un-built INTERACTION term —
      the RPA/self-consistency object is the sole named remainder for sound; (2) the RPA term and
      ML-5's coupling gap are RELATED-BUT-DISTINCT — both named instances of the same structural
      gap-class (an interaction vertex beyond the free-quadratic theory), NOT shown to be the same
      term (W2-MAP's M-4 already proved W_INT symmetry-inequivalent) — a dedicated confront would be
      needed before claiming identity; (3) B2-b growth: the immobile flat-band reservoir does NOT need
      sound to cluster (pressureless matter clusters gravitationally) — what B2-b needs is the
      still-unbuilt GRAVITATIONAL/MODULAR SOURCE TERM (the δS=δ⟨K⟩ route), now its named gate.**
      MC2b's ν_s=c_s²τ ansatz remains unconfronted at response level (booked). c_s²=1/3 never entered
      construction; the two-routes confront stays OPEN awaiting the interaction term. No scoreboard
      value moved.
    • **▶ WAVE2-MAP (the vertex/propagator map, classification-first) EXECUTED (2026-07-10, pre-reg
      in 8ca645c/6e03d95; implementation + adversarial check PASS-WITH-NOTES, every number reproduced
      from scratch) → THE CLASSIFICATION IS DONE: AMBIGUOUS-BY-TWO-ORBITS (sharper than O(2)); ONE
      physical candidate map stands, pending one small named theorem; the ML-5 "two walls one object"
      hypothesis REFINED to PARTIAL with a proven negative half.**
      `proofs/foundations/W2_MAP_vertex_propagator_2026-07-10.py` (~3 s; 29 checks PASS; the_net.py
      untouched). **THE CLASSIFICATION (M-1a):** the full space of internal↔cover intertwiners under
      the DERIVED requirements {R1 antiunitary-compatibility, R2 A4-equivariance, R5 isometry}:
      dim Hom_A4(edge_rep, dart_rep) = 6 (checker re-derived by character theory + an independent
      Reynolds-projector method); the R-EVEN branch is EMPTY under isometry (rank exactly 3, proven
      obstruction); the R-ODD branch = Uo⊗End_A4 (dim 4) cut by isometry to TWO DISJOINT CIRCLES:
      **(1) the trivial gauge orbit — Φ_θ = Uo·e^{θJ6} is LITERALLY complex-scalar multiplication by
      e^{iθ} on the Witt +i eigenspace (1e-16) ⟹ ONE physical map under the ordinary phase
      convention; (2) a genuinely distinct orbit — Uo(cosφ S1 + sinφ S2) with S1,S2 exactly
      ANTICOMMUTING with J6 ⟹ these map the +i (particle) eigenspace into the −i (conjugate/
      antiparticle) eigenspace.** A chirality-based exclusion of orbit (2) is PLAUSIBLE (the proven
      no-improper-symmetry theorem) but NOT ESTABLISHED — it needs THE CHIRALITY BRIDGE: connecting
      the abstract commutant reflection to an actual orientation-reversing operation on the physical
      srs structure. **▶ THE NAMED NEXT PIECE (small, well-posed): prove or refute the chirality
      bridge ⟹ the map becomes FORCED-UP-TO-PHASE (and M-2/M-3 — the transport + the oblique
      insertion — unlock with Φ = Uo·e^{θJ6}), or the two-orbit ambiguity is fundamental.** M-1b: the
      dynamics-compatibility selection FAILED via an EXACT identity (⟨Beo·C, C′·J6⟩ = 0 over the whole
      commutant, ~1e-15, checker-reproduced from an independent basis) — CONFIRMED-EXACT,
      MECHANISM-PARTIAL (the symmetric part is a trace-parity triviality; the antisymmetric part
      extends the sweep's still-unnamed naive-candidate kill). M-2/M-3 GATED-SKIPPED correctly (no
      number forced through an ambiguous map). **M-4 (the ML-5 confront): PARTIAL — same ambient
      anchor, symmetry-inequivalent constructions: ⟨ŝ|G_int(α₁)|ŝ⟩/12 → c_S = 1/12 with deviation
      = −(1/3)α₁⁶ (three decades, checker-reproduced bit-identically) — ML-5's object DOES reduce to
      the oblique's singlet weight in the free limit; BUT W_INT's Clifford decoration = the UNSIGNED
      (Ue/R-even) convention, which the classification PROVES FAILS R2 outright** (not merely differs;
      dev exactly 2.0 from the legitimate R-odd convention) ⟹ the "two walls, one object" hypothesis
      survives only through the future forced map, NOT through W_INT as built. BOOKED SOFT SPOT (F7):
      the antiunitary companion τ_dart's sign convention is itself a choice (each sign forces one
      R-parity via R1 alone); σ_int/τ_dart are one-particle stand-ins, not a verified reduction of the
      certified Fock-level σ_M0 — named for any future formalization. BOOKED FLAG: the_net.py's
      docstring "R-even carries the vacuum J6/C" contradicts the computed intertwining (Uo yes, Ue no)
      — docstring fix deferred (the net is mid-accretion by concurrent stations). No scoreboard value
      moved; deck/Z3 never assumed; m_ν-scale same-FAMILY noted, not same-object.
    • **▶ WAVE2-D1c (the origin-constrained BW read) EXECUTED (2026-07-10, pre-reg aef6148;
      implementation + adversarial check PASS-WITH-NOTES) → V-0′ THE HARD INSTRUMENT GATE FAILED AND
      STOPPED THE STATION (the frozen contract worked exactly as designed: nothing on the lattice was
      read); the BW line's blocker is now named at its DEEPEST level — and THE LINE IS PARKED.**
      `proofs/foundations/W2_D1c_origin_constrained_bw_2026-07-10.py` (+ the_net.py accretion:
      bond_profile_slope origin_constrained/absolute_window params, defaults checker-verified
      bit-identical). **THE FINDING: THE LATTICE HORIZON COORDINATE IS AN UNDERIVED OBJECT.** The
      origin-constrained estimator (BW forces β(0)=0) exposed a HALF-BOND placement freedom in where
      x=0 sits that every previous instrument was structurally blind to: the free intercept ABSORBS
      constant x-shifts (proven algebraically — OLS slope depends only on centered x), the D1 ratio
      BAKED IT IN silently (retroactively explaining D1's ~2× ambiguity), origin-constraint exposes it
      fatally. On the 1D calibration chain: midway-threshold convention → w→0 = 0.500097×2π (the exact
      factor 2, checker-verified to 10 decimals at the single-bond level); edge-site convention →
      1.049×2π; the old benchmark's implicit convention (edge-site, single first-bond ratio) →
      0.9988×2π. **CHECKER REFINEMENT (booked): the 4.9% edge-convention residual is DOMINANTLY an
      extrapolation-form artifact** (the frozen ladder {1,1.5,2,3}·d_b reaches into the saturated
      regime + a degenerate duplicate window; a tight ladder inside the first bond recovers 0.9975×2π
      = 0.25%) — secondary to, and distinct from, the coordinate ambiguity. **THE PARK DECISION
      (architect, on the checker's recommendation): the checker's due-diligence probe of the cheapest
      derived-horizon candidate (the profile's self-consistent zero-crossing) is NEGATIVE-ON-1D —
      with ~1 bond per d_b the zero-crossing is unconstrained at exactly the half-bond scale in
      question; the entangling-surface placement on this discretization is plausibly a
      REGULARIZATION-SCHEME CHOICE, not extractable from bulk correlations. Three stations invested
      (D1 retired → D1b instrument-limited → D1c gate-stopped), diagnosis sharpened each time and now
      lands on a THEORY object, not an instrument fix. ⟹ THE BW LINE IS PARKED. G's 2π stays ❌ OPEN
      and UNQUANTIFIED. MG-1d's completion now requires THE DERIVED HORIZON COORDINATE (a
      Layer-3/modular-theory question: where does the net's own modular structure place the entangling
      surface?) — named, not scheduled.** OPTIONAL CHEAP PROBE booked for any future session (not
      run): does the 3D lattice's richer transverse geometry shrink the convention sensitivity
      (19–23% in the disclosed smoke test) with finer windows — distinguishing fundamental scheme
      ambiguity from 1D sparsity? NOTE booked: the delivered file's V-1 contract is pre-empted by the
      gate exit (never executed in-run); the checker exercised it independently (3.9025341659750197 ==
      the D1b log; net.self_test passes). Suggestive-but-unprovable context (NOT a result): every
      rigorous framing of the BW read keeps landing near 1×2π (0.85–1.01 first-bond lattice; 0.9975
      tight-ladder chain) with a new ambiguity exposed at each precision level — 2π is NOT confirmed
      and NOT quantified; the near-1 pattern is recorded as context only. No scoreboard value moved;
      ħ NOT derived.
    • **▶ WAVE2-QF-2c (the Bell completion) INTEGRATED (2026-07-10, pre-reg aef6148; implementation +
      adversarial check PASS-WITH-NOTES) → DOUBLE-NULL AGAIN; THE TERMINALITY CLAUSE FIRED:
      instrument-side Bell iteration is DECLARED CLOSED (binding; no QF-2d without a new physics
      object).** quantum_foundations.py QF-2c section (~480 lines; QF-0..2b byte-untouched; 228
      PASS/0 FAIL; ~1 s; checker reproduced all S_fixed values to the print floor). **LEG A (the
      natural per-mode family): NO-VIOLATION at all 7 instances (max S_fixed = 1.4223 at sep=1;
      classical bound 2) — and UPGRADED to a STRUCTURAL null (checker-proven): the four settings
      mutually commute ⟹ this family cannot violate CHSH for ANY state, not just this vacuum.** The
      mandated dense check caught pre-sweep that the natural basis's bilinears COMMUTE (disjoint
      Majorana pairs) ⟹ A(θ) non-dichotomic generically ⟹ the rotation-optimization is unlicensed;
      the implementation pass switched to the licensed fixed-settings CHSH (S_fixed, 4 odd-sign patterns) —
      checker verified the enumeration is COMPLETE (the even-sign patterns have classical bound 4,
      not 2: no licensed enlargement exists; the unlicensed Horodecki numbers printed transparency-
      only, max 1.93). **LEG B (r=3 quartics): NO-VIOLATION at the 3 instances reaching r_eff=3**
      (native BFS-ball S=0.0007; two chain controls 0.049/0.011); FAR/NEAR/ladder STRUCTURALLY
      EXCLUDED at r=3 (2 sites/side — declared, not skipped); dense validations exact (Q²=I,
      {Q1,Q2}=0, wick_general==dense ~1e-17). TERMINALITY WORDING TIGHTENED at integration
      (checker mandate; claims strictly less — coverage-accurate: leg B's native coverage = ONE
      geometry). The two PHYSICS doors remain the only path: the derived J6-compatible patch ω;
      the finite-k/tick sector. No thresholds moved; crisis branch intact; no scoreboard value moved.
    • **▶ WAVE-QF-2b (the smeared Bell read) INTEGRATED (2026-07-10, pre-reg 13bcb03; implementation +
      adversarial check PASS-WITH-NOTES) → DOUBLE-NULL, SCOPED — and one sub-result upgraded to a
      PROVEN LEMMA; the flux question CLOSED by the repo's own theorems; QF-2c NAMED with two legs.**
      `derivation_topdown/adapters/quantum_foundations.py` +~470 lines (QF-0..4 byte-untouched; 136
      PASS/0 FAIL; 0.9 s; checker reproduced every S_max to 8 decimals from scratch). **F-0 THE FLUX
      ADJUDICATION (machine-checked): max|Im C| == 0.0 exactly — the real flux-free hopping IS the
      derived object; any added phase is pure gauge (cover gauge-triviality) or a chosen finite-k
      sector (holonomy triviality) ⟹ THE COMPLEX-FLUX BRANCH IS PERMANENTLY DEAD as an instrument fix;
      two doors named out-of-scope: a DERIVED J6-compatible non-product patch ω (new derivation
      station; naive product extension provably Bell-dead) + the finite-k/tick sector.** **THE NULL:
      NO-VIOLATION-IN-FAMILY everywhere** — control chain(400) 0.146/0.006; FAR 0.026; NEAR 0.719;
      BFS-ball 0.133; ladder sep{1,3,9,27} = {1.253, 0.040, 0.533, 0.026} (raw, orientation-mixing
      declared); no Tsirelson breach. **THE PROVEN LEMMA (checker-established): for the declared
      shared-pivot family A(θ)=cosθ(iγ₀γ₁)+sinθ(iγ₀γ₂), the CHSH tensor T is EXACTLY rank ≤ 1 on ANY
      real free-fermion covariance, independent of mode selection** (Clifford-parity: every T entry but
      T₀₀ contains a same-parity Γ block ≡ 0; verified on 60 random-mode + 200 generic-covariance
      trials, all exact) ⟹ r=2 could never beat r=1 (S identical to 8 decimals) BY CONSTRUCTION, and
      the null for THIS family class is theorem-grade. **SCOPE CORRECTION (binding): this does NOT
      cover all 2-mode families — the natural per-mode-bilinear basis genuinely reaches rank 2
      (diagnostic S_max ~1.62 < 2, a 60-trial sample not a sweep).** **F-3: the pre-reg's quartic pair
      is ALGEBRAICALLY IMPOSSIBLE at r=2 — PROVEN (dim(even Cl(4))=8 ⟹ exactly ONE quartic = the
      region parity = CENTRAL; both declared constructions commute exactly) — an ARCHITECT design flaw
      (r∈{1,2} frozen without the dimension count), owned; the quadratic-obstruction question is
      therefore UNANSWERED, not answered-negative.** New permanent instrument: wick_general (2n-point
      Pfaffian, dense-validated 4.8e-17). **▶ QF-2c NAMED (not run), two independent legs, both
      checker-validated as well-posed: (i) r=3 quartic pairs (anticommuting independent quartics EXIST
      at 6 Majoranas — verified abstractly; necessarily overlapping support); (ii) the natural
      per-mode-bilinear family (breaks the rank-1 lemma; unexplored).** BOOKED SENTENCE: the
      double-null is an OBJECT-LEVEL statement for the covered families — NOT classicality, NOT
      terminal for vacuum-Bell (the two named doors + QF-2c's two legs survive). No thresholds moved;
      crisis branch intact; no scoreboard value moved.
    • **▶ WAVE-D1b (the controlled BW read) EXECUTED (2026-07-09→10, pre-reg 2411540; implementation +
      adversarial check PASS-WITH-NOTES) → V-4 VERDICT: INSTRUMENT-LIMITED (per the frozen contract;
      G's 2π stays ❌ OPEN and UNQUANTIFIED); the instrument's defect is now NAMED SHARPLY and the
      machinery is proven sound.** `proofs/foundations/D1b_controlled_bw_2026-07-09.py` (full ladder
      M∈{8..16} dense, 3 directions, 17.8 min); the_net.py §4b gained vertex_position +
      bond_profile_slope + Patch(skip_pair_bfs) (clean accretion, independently re-verified bit-identical).
      V-0 benchmark 0.998751×2π ✓; V-1/V-2 ✓; V-3 axis razor-thin fail (0.0503 vs 0.05, noise-level on a
      57%-residual fit); Lorentz gate FAIL (driven by the window/depth interaction, see below); declared
      extrapolations slightly NEGATIVE (booked raw). **THE NAMED RESIDUAL DEFECT (checker-established,
      replaces any generic wording): WINDOW-DILUTION / FREE-INTERCEPT ARTIFACT — β(x) rises within 1–2
      bond-lengths of the cut then SATURATES to a plateau; the curvature is already present in the 1D
      free-fermion calibration case itself (β/x falls 0.999→0.404×2π by x=7.5) ⟹ BW's linear law is an
      x→0 ASYMPTOTIC, not a wide-window law — NOT an srs artifact. The free-intercept affine fit lets the
      intercept absorb the near-horizon rise (~1.6–2.3× further dilution vs origin-constrained). The
      window = fraction-of-each-direction's-OWN-depth makes axis read at half the absolute cutoff of
      ⟨111⟩/⟨110⟩; a COMMON ABSOLUTE window brings the three directions to 7–13% agreement (the
      "Lorentz failure" was mostly the window artifact, not broken isotropy). The M-ladder extrapolation
      inherits all of this (absolute window grows with M ⟹ the fit tracks shrinking dilution, not an
      M→∞ constant).** **THE MACHINERY IS SOUND: the first-bond diagnostic recovers axis 0.85–0.90×2π /
      ⟨111⟩ ~1.01×2π — same ballpark as the retired D1 single-bond read** ⟹ the
      covariance/entanglement-Hamiltonian/position-bridge chain is validated; the miss is confined to the
      window-averaging design (an architect-owned pre-reg choice, frozen before numbers — the contract
      worked as intended). **▶ NAMED FUTURE BUILD D1c (pre-registerable, NOT run): origin-constrained
      β=a·x fit (BW forces β(0)=0) over the FIRST 1–3 bond layers with a COMMON ABSOLUTE window across
      directions + an M-ladder at fixed absolute window** — all three fixes are checker-quantified.
      COVERAGE-GAP FLAG (booked): benchmark_bw_2pi does NOT exercise bond_profile_slope (hand-rolled
      first-bond ratio) — any future multi-bond extractor must add a benchmark leg through the actual
      extractor (a window-sweep on the 1D chain would have surfaced this pathology in seconds).
      V-5: geodesic 0.1908×2π vs Cartesian 0.4056×2π (expected disagreement, adjudication 2). No
      scoreboard value moved; ħ NOT derived (only the BW-2π-CONFIRMED branch could have); the retired
      +7% never used.
    • **▶ I0-G4 INTEGRATED (2026-07-08, pre-reg e09aa0e; implementation a model + adversarial check PASS;
      verify 66/66) → GREEN: `derivation_topdown/adapters/aqft_net.py` — the net {A(O)} is a
      machine-checked Haag–Kastler INSTANTIATION at declared scope.** HK-0..HK-6 all pass at frozen
      tolerances (exact-zero light cone through T=4/M=5; Z³ covariance 0.0 ×3 directions; all 62 duality
      subsets, worst 1.1e-10; DHR = {ν:1,d:3,u:3,e:1} + 2T). Checker independently verified the HK-5
      spectrum mask is derivation-sound and non-load-bearing. **STAYS OPEN (HK-7, declared):** TD-limit
      Haag duality (ML2b's DR frame remains CONDITIONAL on it); local DHR transporters/braiding;
      past-cone/intersection closure. No scoreboard value moved; zero physics added.
    • **▶ I1-G2 INTEGRATED (2026-07-08, pre-reg e09aa0e; implementation a model + adversarial check
      PASS-WITH-NOTES; verify 67/67) → GREEN: the Furey–Stoica labeling dictionary HOLDS on the engine's
      Cl(6) Fock, basis-vector by basis-vector — THE LABELING IS COMPLETE + INDEPENDENTLY CORROBORATED.**
      `derivation_topdown/adapters/furey_stoica_labels.py`. FS-0..FS-4 at frozen tolerances: the forced J6
      frame IS a Witt basis (nilpotency 7e-16, first explicit check); 8 ladder states = minimal left
      ideal, N-grading == species projectors (≤2e-15); **Q = N̂/3 = the FIRST DERIVED charge operator in
      the repo — spectrum {0,⅓×3,⅔×3,1} (7e-16), grading == species; dictionary {ν, d̄, u, e⁺} under ONE
      global ideal convention** (conjugate ideal {ν̄,d,ū,e⁻}); color su(3) = ladder bilinears re-verified
      as contract. **FS-5 (dual-outcome, computed) → INDEPENDENT/CROSS-CUTTING: the generation ℤ₃ (deck)
      is NOT in the A4 gauge action** — all 8 order-3 elements fail all 6 t-permutations (residuals
      0.455–0.622); σ₃∈A4 abstractly but U_π ⊥ U(σ₃) exactly (HS-orthogonal, checker-verified
      phase-independent) ⟹ **generation = a FOURTH mechanism (winding deck), distinct from Furey/triality
      — the framework's distinctive claim, now machine-checked.** ADJUDICATED (architect error owned): the
      pre-reg wrote "γ₅²=I"; Euclidean Cl(6,0) FORCES γ₅²=−I (ω²=(−1)¹⁵) — contract = the involution with
      sign disclosed (checker confirmed math + pre-existing docstring + prior art). STAYS OPEN (FS-6):
      hypercharge/weak-isospin from the Fock (ℍ edge-qubit, G3+); conjugate-ideal build. Zero physics
      added; no scoreboard value moved. **⟹ I1 COMPLETE: two green contract suites (G4, G2).**
    • **▶ I2-G5a INTEGRATED (2026-07-08, pre-reg cbe42e3; implementation a model + adversarial check PASS
      with FALSIFICATION PROBE; verify 68/68) → GREEN: the tick sector is a machine-checked CONSTRUCTIVE
      CONNES–ROVELLI thermal-time instantiation.** `derivation_topdown/adapters/thermal_time.py`.
      KMS-0..KMS-5 at frozen tolerances: run marginal exactly geometric (rel std 2e-16); modular generator
      AFFINE IN THE TICK (residual 4e-14), slope = derived β_eff = 2·log(u_c/α₁) = **5.1011473686**
      (triple-confirmed: formula + adapter + M0-2R's own fit; an earlier sweep-prose "2.94" was a a model
      arithmetic slip, nowhere in the repo); **Gibbs identification ρ_run = e^(−β_eff·N̂)/Z at 1.9e-22 ⟹
      the modular flow of the run state IS the physical tick flow**; two-point KMS w.r.t. the TICK
      generator at 1.2e-13 (100 frozen pairs) — **checker falsification probe: a non-Gibbs perturbation
      FAILS by ~13 orders (worst 1.83, 37/100 pairs; wrong-β control 49.0) ⟹ the contract is
      content-bearing, NOT the ρ-tautology**; β·κ = ln2 exact (symbolic). STAYS OPEN (KMS-6): the vN TYPE
      of the tick algebra (G5b); the crossed-product/observer (G5c); spatial KMS; TD-limit. Zero physics;
      no scoreboard value moved.
    • **▶ I2-G1 INTEGRATED (2026-07-08, pre-reg cbe42e3; implementation a model + adversarial check
      PASS-WITH-NOTES; verify 69/69) → GREEN + THE ISOTROPIZATION WELD: the Kotani–Sunada standard
      realization is EXACTLY the frame in which the emergent light cone is isotropic.**
      `derivation_topdown/adapters/sunada_geometry.py`. SR-0..SR-3 at machine precision: **b₁(K4)=3 == the
      Z³** (space's dimensionality IS the first Betti number, machine-checked); the geometry IS the
      standard realization (input-free: harmonic 5e-16, bond isotropy 5e-16, 120°, bcc 3.000000000000);
      chirality's geometric seat (C₃ ⟨111⟩, no improper symmetry); **BZ == Jacobian torus
      H₁(K4,ℝ)/H₁(K4,ℤ)** (cotree↔deck bijection exact; hashimoto exactly periodic 2.5e-15). **SR-4 →
      ISOTROPIZED (dual-outcome, frozen logic): g_cart = L·g_frac·Lᵀ eig [0.500018,0.500045,0.500068],
      spread 1.0e-4 (vs 4:1 fractional); checker: ONLY the k-duality-derived transform isotropizes
      (inverse-conjugates are MORE anisotropic, spread 2.5) ⟹ SHARP.** Corroborates B3's isotropic cone
      oblique from real-space harmonic geometry. Isotropic speed 0.70714 ≈ 1/√2 = √(v_Hodge·v_adj)
      (OMEGA_Q0 dictionary; report-only, tied to the LCLᵀ=I convention — ISOTROPY is the invariant).
      **FINDINGS (booked raw):** (i) architect tolerance error #2: SR-4(i)'s 1e-6 regression on g_frac eig
      {¼,¼,1} FAILED at 1.3e-4 — set below cone_velocity's finite-difference precision (eps-probe: dev
      shrinks LINEARLY with eps to the roundoff floor ⟹ true eigenvalues {¼,¼,1}; the engine's own atol is
      1e-2); the FAIL is printed, not hidden. (ii) **PROSE-LABEL CORRECTION (repo-wide): the engine's H1
      frame is the TRUE CYCLE SPACE (d₀·H1≈0); B1 = svd(d0)[2][:3] is the COBOUNDARY/row-space frame** —
      prior prose ("B1 = cycle basis") had them SWAPPED; checker traced ALL engine uses (the_net, WS1,
      E2a, d4_spectral_action): only complementary-orthonormal-frame properties are used ⟹ mislabel NOT a
      bug; future cycle-space work MUST use H1 (or ker d₀), never B1. STAYS OPEN (SR-5): the heat-kernel
      scaling limit + the 2π (D1, decisive wave); flat-band geometry. Zero physics; no scoreboard value
      moved. **⟹ I2 COMPLETE: four green suites (G4, G2, G5a, G1); the trust wave is DONE.**
    • **▶ I3-G6ab INTEGRATED (2026-07-08, pre-reg 88a9433; implementation a model + adversarial check PASS
      with THEOREM ADJUDICATION; verify 70/70) → GREEN + A NEW THEOREM.**
      `derivation_topdown/adapters/zeta_gauge.py`. **ZG-1 the Bass identity HOLDS on our B** (vs the
      engine's own ihara_zeta_inv — two objects in the repo since 2026-06, never confronted; worst
      1.4e-15). **ZG-2 cover-girth selection EXACT** (m_L<3.4e-15 for L≤9; m₁₀=120=2×10×6 integer-exact
      vs the Wilson enumeration). **ZG-4 det(I−uW_INT) computed for the FIRST time: the loop-expansion
      identity −log det(I−uW) = Σ u^L/L Tr W^L holds at 1.2e-17 (ρ(W_INT)=√2 exactly) — THE "ONE
      GENERATING FUNCTION" STATEMENT IS NOW MACHINE-CHECKED** (charter §0 thesis, first hard test passed).
      **ZG-3 → STRUCTURED-MISMATCH upgraded to a CONFIRMED THEOREM (checker-verified): MAXIMAL-ABELIAN-
      COVER GAUGE TRIVIALITY** — every cover-closed walk is null-homologous ⟹ zero net signed visit on
      EVERY edge (all 120 girth-10 cycles exact; cycle-space→Z³ injective ⟹ ALL lengths; k-integrated
      response invariant under large random A, per-k changes O(10²)) ⟹ **cell-periodic signed U(1) is pure
      gauge on the cover; the zeta's zero-momentum gauge response vanishes at ALL orders ⟹ the physical
      Wilson/photon bridge is intrinsically FINITE-k — a NAMED OPEN CONTRACT (G6b′, feeds D3).** PRIOR-ART
      FLAG: `srs_wilson_action_quadratic.py` docstring says "signed ±1" but the code implements an
      UNSIGNED per-dart indicator (internally consistent as its own convention; text/code mismatch —
      future users must not assume signed). Zero physics; no scoreboard value moved.
    • **▶ I3-R2/G3a INTEGRATED (2026-07-08, pre-reg 88a9433; adversarial check FAIL → mechanical fix →
      re-check PASS; verify 71/71) → THE KO SIGN TABLE EXECUTED: verdict KO-OTHER (ANOMALOUS) — NOT
      KO6-FOUND; the reconciliation with Connes' SM anatomy is NOT confirmed.**
      `derivation_topdown/adapters/ncg_spectral.py`. **KO-1 (spacetime factor): (−1,+1,+1) → KO-dim 4**
      (m06's computation, now a machine contract). **KO-2/3 (the internal Cl(6) Fock, first execution):**
      the graded pairs with C_ideal/K_g6 leave ε′ UN-FORCED (leaks 0.22/0.29); the particle-hole pair
      (P_F, σ_M0) — σ_M0 = the genuine aᵢ↔aᵢ† lift, gated by its defining relation |Σ·aᵢ·Σ⁻¹ − aᵢ†| =
      2e-15 — forces **(ε,ε′,ε″) = (−1,−1,−1): ANOMALOUS, matching NO even Connes row ⟹ KO-OTHER.** The
      internal KO-dim is NOT established. **INTEGRITY RECORD (the pipeline's teeth):** the first
      implementation reported KO6-FOUND ((+1,+1,−1) → 6, "4+6≡2 confirmed"); the adversarial check exposed
      it as an operator-ORDERING BUG (the constructed Vsig failed the defining relation at 2.0; the
      corrected operator — independently derived by the checker, match 0.0 exact — reverses the verdict).
      The false positive died at step 4, before booking/wiring; retracted in full. **R2b NAMED OPEN
      QUESTION (literature-first, pre-register before recomputing):** (−,−,−) is the exact J↔Jγ
      convention-image of KO-6's (+,+,−) for even triples (checker-verified: J′=Jγ ⟹ ε′ flips, ε″
      preserved, J′²=ε·ε″=+1); whether the standard literature convention makes our forced reading KO-6
      is NOT decided here (Jγ is outside the frozen candidate set; deciding by convention-picking would
      be goal-seek). Γ₅ ≡ −P_F exactly (one effective grading). STAYS OPEN: the internal KO-dim (R2b);
      G3b (log-det ≡ a₄); first-order condition; full axiom audit. Zero physics; no scoreboard value
      moved. **⟹ I3 COMPLETE: G6ab GREEN + G3a/R2 EXECUTED-HONEST; six suites wired, verify 71/71.**
    • **▶ I4-D1 EXECUTED (2026-07-08, pre-reg b06c93f; implementation + adversarial check PASS-WITH-NOTES)
      → INCONCLUSIVE per the frozen criteria — AND the check re-grades the ENTIRE BW measurement line:
      Newton's G 2π residual STAYS ❌ OPEN, and its prior "+7%" QUANTIFICATION IS WITHDRAWN AS
      UNRELIABLE (instrument-limited, not resolved).** `proofs/foundations/D1_bw_canonical_2026-07-08.py`.
      D1-1 reproduced ML-1‴ exactly (1.068344×2π ± 0.006). **FINDING 1 (identity, checker-confirmed
      exact): the canonical Kotani–Sunada normalization ≡ the fractional proper-distance normalization
      for EVERY cut direction** (isotropization ⟹ LᵀL = v_iso²·g_frac⁻¹ ⟹ s(n̂)/v_iso ≡ PROPER(n̂)) ⟹ the
      +7% was NEVER a frame artifact; the D1 falsifier was tautological (architect design flaw, owned).
      **FINDING 2 (the instrument defect, checker bug-hunt): the frozen ML-1‴ axis recipe hardcodes
      "1.0 cell" per first-bond gap, but the selected bond spans 0.5 cell in full vertex-position terms
      (gaps {0, 0.25, 0.5} exist) ⟹ the ABSOLUTE distance convention carries a ~2× AMBIGUITY** (axis
      1.07×2π ↔ ~2.14×2π; diagonal 1.28×2π ↔ 0.72×2π under the two conventions) **that dwarfs the quoted
      ±0.6%.** **FINDING 3 (fit fragility): the raw axis slopes PLATEAU at ~1.10 for M≥8** (sensitivity:
      M≥8-linear 1.104, constant 1.102, M≥10 1.102) — the 1.068 intercept was M=6-driven; the 5-point
      ladder drifts UP (1.077). **FINDING 4: finite-M cut-direction dependence is real (axis 1.068 vs
      ⟨111⟩ 1.277, frame-independent, like-for-like NOT yet established due to Finding 2; diagonal ladder
      still falling at M=12 ⟹ its M→∞ NOT established.** **⟹ THE HONEST STATE: the near-horizon first-bond
      methodology (ML-1‴/D1, M≤14, linear-1/M) CANNOT currently produce a controlled 2π comparison — the
      G-2π residual is OPEN and UNQUANTIFIED (the defining equation of the READ is incomplete: the
      lattice near-horizon proper-distance convention is un-derived). Do NOT quote "+7%" or "1.07×2π" as
      the miss.** **▶ NAMED BUILD D1b (the controlled BW read):** (a) full-vertex-position bond distances
      with the convention DERIVED (bond-midpoint proper distance, no hardcoded cell counts); (b) slope
      from the full near-horizon PROFILE (multi-bond/multi-layer fit, not first-bond); (c) M≥8-anchored
      extrapolation with mandatory fit-form sensitivity; (d) like-for-like multi-direction covariance as
      the acceptance criterion (direction-independence = the Lorentz check); (e) sparse solvers for
      M≥20 if needed. No scoreboard value moved; ħ NOT selected; nothing tuned. Do NOT re-run D1 as-is.
    • **▶ I5-D2 EXECUTED (2026-07-09, pre-reg 3cfcde1/9c2646b; implementation + adversarial check
      PASS-WITH-NOTES) → VERDICT MIXED (frozen logic) — AND THE PHYSICS READING IS ESTABLISHED: the
      forced srs↔srs-z scalar IS a genuine propagating field with the textbook one-loop Yukawa-scalar
      profile; the "survival" question is re-posed as RELATIVE (D2b).**
      `proofs/foundations/D2_higgs_survival_2026-07-08.py`. **ESTABLISHED (checker-verified):**
      (1) **THE MEXICAN HAT, computed for the first time**: V(m;g) = m²/g − ∫DOS·[√(ε²+m²)−|ε|]dε
      (prefactor FORCED by m08's own Ifun — the unique form whose stationarity is m08's gap equation;
      pre-reg's m²/(2g) was an architect error, corrected on-screen as instructed); argmin V ==
      solve_gap(g) at 1e-7 across the g-ladder; **V″(m*) > 0 everywhere (the radial/Higgs mode is
      massive)**; exact U(1)_A degeneracy (the {0,π} pinning = m08's crystallographic result, cited).
      (2) **THE SCALAR PROPAGATES**: first response computation in the repo (k-space bubble on the cone
      fiber; χ-formula brute-force-verified at 2e-7): Z > 0 in all 24 combos, grid-stable 0.03%,
      window-stable 1.2%, **ISOTROPIC to 0.001%** (another independent isotropy corroboration).
      (3) **QUALITATIVELY DISTINCT FROM DECORATION**: the un-forced identity vertex gives Z_dec < 0 with
      the opposite node-fraction profile (checker: genuine, no sign theorem violated) ⟹ NOT the clean
      Perez-Sanchez death. (4) **Z ~ c·log(Λ/m) QUANTITATIVELY** (checker: Λ-scan b=0.0241 vs
      resolution-consistent m-scan b′=0.0256, ratio 1.06; log beats power/linear fits) — the textbook
      one-loop wavefunction renormalization of a Yukawa-coupled scalar. **FAILED (the MIXED member):**
      the pre-reg's node-domination criterion (f_node negative O(1); stiffness is off-node/UV-fed) — but
      the criterion itself embodied a DIMENSIONAL ERROR (architect: the static 3D gapped-Dirac bubble is
      LOG-divergent, not ~1/m — checker's independent analysis concurs; the SM Higgs itself would fail
      that criterion). CORRECTIONS OWNED: the V-prefactor; the 1/m criterion; the probe's reported
      m-slope −0.015 was under-resolved — the resolution-consistent slope is ≈ −0.21 (do NOT quote
      −0.015). **⟹ THE HONEST STATE: the framework's Higgs-analog is a REAL FIELD (massive radial mode
      on a computed potential + positive isotropic log-running stiffness + not decoration); whether it
      SURVIVES the continuum limit in the Perez-Sanchez sense is the RELATIVE question Z_φ vs Z_gauge
      under continuum scaling — ▶ NAMED BUILD D2b, sharing its finite-k gauge-response machinery with
      G6b′.** NO EW-scale claim (g = the layer's irreducible input; scale stays ❌ OPEN). No scoreboard
      value moved. Do NOT re-run D2 as-is (the m-ladder needs the composite-grid Z throughout).
    • **▶ I6-R1 EXECUTED (2026-07-09, pre-reg 94efdfc; implementation + adversarial check PASS) →
      LOCATION-REFUTED: the u⁴/u⁵ location hypothesis for the −70 ppm is DEAD as posed; the −70 ppm stays
      ❌ OPEN at ML-5b's wall — with the sharpest structural characterization of the channel to date.**
      `proofs/foundations/R1_zeta_order_reading_2026-07-09.py`. ONE frozen functional (the exact per-order
      decomposition A_L of the already-forced chiral asymmetry; identity Σα₁^L·A_L == A(α₁) at 7e-16; no
      selector anywhere — checker-audited). **THE READ: tail₄₅ = α₁⁴A₄ + α₁⁵A₅ = −2.6764e-6 — 15.3× off ε
      (>2× threshold) AND nearer the α₁⁴ poison (1.15×) than ε ⟹ REFUTED on both conditions
      independently.** **STRUCTURAL FINDINGS (checker-verified EXACT):** (1) the forced asymmetry is a
      power series in α₁² with EXACT algebraic coefficients: A₂=A₆=−1/√3, A₄=A₈=−2/√3, A₁₀=−17/√3 (all
      ≤6e-16; the 17 found by the checker); (2) **odd orders vanish IDENTICALLY** — mechanism diagnosed
      (checker): each W_INT hop injects one grade-1 Clifford generator ⟹ W^L has grade parity (−1)^L; the
      vacuum is a Γ-eigenvector ⟹ odd-grade expectations vanish — the O0 graded-blindness principle,
      independently rediscovered; (3) **THE CHANNEL IS REAL-VALUED AT EVERY ORDER** (Im at noise floor
      ∀L≤10) ⟹ the dart-level winding asymmetry carries NO PHASE CONTENT at ANY loop order — ε is a
      PHASE, and this channel provably has none to give ⟹ the transport to ε requires structure this
      functional LACKS (deepens ML-5/ML-5b: the missing suppression coupling is not a re-weighting of
      this series). The −70 ppm's one named object (the forced lepton-slice transport / α₁²→α₁⁴⁻⁵
      suppression) remains OPEN and is now known to need a PHASE-BEARING channel. No scoreboard value
      moved; nothing tuned; the target appeared only at the declared end. Do NOT re-run R1.
    • **▶ I6-D3 EXECUTED (2026-07-09, pre-reg 0d70dd2; implementation + adversarial check PASS) →
      VERDICT PAIR (RIGID, DEGENERATE): a clean double negative for confinement-from-matter at this
      order — AND THE HOLONOMY-TRIVIALITY THEOREM (the build's third theorem, same root as the
      gauge-triviality).** `proofs/foundations/D3_confinement_binary_2026-07-09.py`.
      **THE THEOREM (checker-confirmed EXHAUSTIVELY, 192/192 classes at L=10,12,14): every cover-closed
      non-backtracking cycle's Cl(6) matter holonomy is EXACTLY +I** — cover-closed ⟹ null-homologous ⟹
      every quotient edge traversed an EVEN number of times (patterns: all-2s at girth; 2s and 4s at
      L=12,14) ⟹ the path-ordered product of anticommuting involutions reorders to a CENTER element —
      and empirically always +I, never −I. **⟹ ONE ROOT, TWO COROLLARIES: the maximal abelian cover
      TRIVIALIZES ALL CELL-LEVEL HOLONOMY — the U(1) gauge response vanishes (G6ab's theorem) AND the
      matter cannot decohere cycle holonomy (D3). Gauge/confinement dynamics on this object is
      intrinsically a FINITE-k / non-vacuum / tick-sector phenomenon.** D3-1 RIGID: g = 1+0j exactly, all
      classes, all sizes (no matter-induced disorder mechanism at cycle level). D3-3 DEGENERATE: Δf
      sign-flips across the frozen windows (+0.0094/−0.0065); both sector growth rates == log√2 (the
      pre-named ergodicity mechanism confirmed); r_q = 0.48/0.67 raw — no static-charge suppression.
      ANCHOR ADJUDICATION (architect wording slip #5, surfaced by the implementation pass, checker-confirmed
      defensible): the exact identity is the CONJUGATE-PAIR winding equality ‖Q₁v‖²≡‖Q₂v‖² (kills
      Im⟨P⟩; algebraic consequence of P3 real) — NOT the three-way equality (t=0 carries the Perron
      mode); the raw 21-combo table printed. **STAYS OPEN: confinement itself (area law, string tension,
      mass gap — now knowably NOT from cell-level matter decoherence; the arena is finite-k [G6b′] /
      non-vacuum states / the tick sector — named D3b).** No scoreboard value moved. Do NOT re-run D3.
      **⟹ I6 COMPLETE: R1 LOCATION-REFUTED + D3 (RIGID, DEGENERATE) + the holonomy-triviality theorem.**
    • **▶ SYMPHONY-S1 INTEGRATED (2026-07-09, charter 3b3d619, pre-reg 911809e; implementation +
      adversarial check PASS; verify 72/72) → THE READS MANIFEST: the instrument's honest coverage
      baseline is 21/161 — and the gap is measured, named, and mostly PORTING, not physics.**
      `derivation_topdown/adapters/reads_manifest.py` + generated `docs/parameters/reads_manifest.md`.
      **THE COVERAGE NUMBERS (the scoreboard's first feed): Tier A = 15 rows ENGINE-MATCHED (18/18
      comparisons at ~1e-16, ZERO mismatches — where the engine plays, it plays exactly), Tier B = 6
      COMPOSED (adoptions listed), Tier C = 140 RESISTING, decomposed: 77 engine-surface-missing (real ✅
      derivations living in standalone predictions/*.py never ported into the master object — PURE
      PORTING WORK), 30 ledger orphans (no lock — registration decisions needed), 14 Bin-X external,
      7 ML-2 species-lift, 6 local-metric (ML-3/4/D1b), 6 response (B2).** **STANDING-CLAIM CORRECTION
      (booked): "~95 ✅ live in the_run.py" was FALSE at code level — the engine surface natively produces
      ~21; the rest live scattered in predictions/. The one-object law's Layer-1 claim is aspirational
      until S1b executes.** BONUS FINDINGS (walled off from tier counts, checker-verified): the forced
      read_masses() reproduces m_μ/m_e and m_τ/m_e at 1.8e-14 (no lock names the ratios — reported, not
      paired); read_generation's free axis s is CALIBRATED on m_μ/m_e in the demo (flagged
      non-predictive); eta_lattice(1/12) vs c_S(1/12) is a numeric coincidence of DIFFERENT derivations
      (deliberately unmapped); 95 engine outputs have no ledger row (unplayed notes = candidate
      unregistered predictions, incl. the_net's entire Layer-3 apparatus). **⟹ GATE RESCOPED (charter
      amendment): publication gate = S1 + S1b + S2 + S3, where ▶ S1b = THE PORTING CAMPAIGN: fold the 77
      into the_run.py as accreted reads (faithful transcription of the predictions' closed forms — NO
      re-fits; engine self-test + full verify gate every batch; the manifest re-run per batch as the
      progress meter, 21 → ~98+), plus the 30 orphan registration decisions (architect+user).** No
      scoreboard value moved; predictions/ untouched; misses stay misses (M_Z carried 🟡 through Tier-A).
  • **▶ THE ML-TRACK (architect diagnosis 2026-07-08 LATE, frozen contract
    internal research notes): the LOCALITY layer of ω — the ONE un-built
    object behind ALL THREE terminal blockers.** The convergence a model missed: (1) MG-1d's logged chase
    (causal-diamond modular flow / BW 2π), (2) M1 DHR sectors (defined ONLY relative to a net of local
    algebras — un-posable without it; M1 never had a frozen contract, hence deferred ~6×), and (3) MG-2's
    flat-band gravitational weight (= per-band contribution to the LOCAL modular energy δ⟨K_diamond⟩) are
    all functions of the SAME missing layer: **physics is (D, ω, {A(O)}) — the net was never built.**
    The "spatial route DEPRECATED" flag was over-generalized (its true scope: fixed-tick cut for κ only;
    M0-2R proved temperature lives in HISTORY ⟹ the local object is the causal diamond in cell×tick
    history; B nearest-neighbor ⟹ STRICT light cone ⟹ diamonds canonical). Tractable NOW: Peschel
    convention locked <1e-9 (M0-convention-control), exact covariance C=(I+iJ6)/2, run KMS state (M0-2R
    T1), emergent cone (A5b/RG2b) — a Gaussian diamond modular Hamiltonian is an EXACT finite computation.
    **QUEUE: ML-0 history net (axioms + twisted duality + regrade the deprecation flag) → ML-1 diamond
    modular flow (the 2π decider: BW-LANDS ⟹ G closes / TICK-REDUCES ⟹ G/(2π) confirmed / NON-GEOMETRIC
    ⟹ obstruction named, likely the flat band; benchmark control FIRST, 2π MEASURED never inserted) →
    ML-2 = M1 DHR keystone (species=DR-forced lift paying WS1's 1.6300 bits, OR the zero-bit theorem;
    forks → architect) → ML-3 = MG-2 flat-band local modular weight → native z_eq → ML-4 = MG-3 θ_* BLIND
    (CAN STILL FAIL) → ML-5 ε readout (optional, LAST, trap-densest).** Poisons: never goal-seek
    ħ/G_eff=G/species=sectors; BW literature = pipeline control, not target; a local 2π does NOT
    retro-edit κ; WS2's cycle kill binding in its regime (sector question is a different quantifier);
    do-not-re-run list in the contract. G(2π)/−70 ppm/θ_*/Y_p/n_s/σ_8 all remain ❌ OPEN.**
    • **▶ ML-TRACK GEOMETRIC-ROUTE CORROBORATION (a model, 2026-07-08 LATE, daylight audit — NO probe run,
      NO value moved): architect's algebraic diagnosis (the net {A(O)}) and an independent GEOMETRIC diagnosis
      converge on the SAME object.** a model, hunting for a fresh entry point, arrived at the missing locality
      layer from the OTHER direction: the **quantum geometric tensor Q_ij(k)=Tr[P ∂_iP ∂_jP] of the srs
      bands** (P(k)=the Bloch projector=the exact covariance C=(I+iJ6)/2). The QGT is the MOMENTUM-SPACE
      content of ML-1's position-space Peschel h_A=log((1−C_A)/C_A) — same projector, two diagnostics ⟹ a
      STRONGER blind 2π test (the quantum-metric integral is a gauge-invariant scalar BW-geometricity must
      relate to the boost normalization). **Three concrete refinements to fold into the ML-1/ML-3 pre-reg
      (amend, do NOT fork a competing track):**
      (1) EVERY Berry-type computation in the repo (srs_bloch_ckm, c1_photon_bundle, arg_h g-2, srs_photon_berry)
          computes only the IMAGINARY part (Berry curvature/Chern) of DISPERSIVE bands. The REAL part
          (quantum metric g_ij) and the FLAT BAND (m=0) were NEVER computed — the exact object ML-1 Q2
          needs per-band and ML-3 needs for the weight.
      (2) THE LOCATED INCOMPLETENESS: `M2b_fluctuation_spectrum_2026-07-07.py:63` closes the flat-band
          divergence with a HARDCODED `REG=1e-4` ("residual flat-band dispersion / substrate floor,
          characterized downstream") — that characterization was never done. The flat-band QUANTUM METRIC
          is the forced finite object standing in for REG (a flat band's physics lives in its quantum
          geometry, not its energy). ⟹ ML-3's "flat-band local modular weight" has a CONCRETE forced
          candidate; de-risks ML-3 from "does it gravitate?" to "compute its quantum metric."
      (3) HONEST CORRECTION (a model self-catch, discipline): a probe-3 claim that "the local 2π is a MASTER
          KEY cascading to binding+hadron+Y_p" was OVER-REACH — architect's poison "a local 2π does NOT
          retro-edit κ" is correct: binding magnitudes route through the GLOBAL κ=h/t_P (already derived,
          M0-2R T4), NOT the open local Rindler 2π; they are different layers. The ONLY valid residue is a
          DOCUMENTATION LAG: `theorem_binding_energy_functional.md:6` (dated 2026-07-04) still calls κ "the
          walled M_Z-pole mystery," which is STALE post-M0-2R (κ=h/t_P derived) — a bookkeeping fix, not an
          unlock. Book this so the conflation cannot corrupt ML-1's blind confront.
      **NET: independent geometric route corroborates architect's ML-track (raises confidence this is THE next
      build); adds a k-space diagnostic + a forced flat-band-weight candidate + a located REG placeholder;
      corrects one a model over-reach. The ML-track stands as the frozen contract; ML-0 next.**
    • **▶ ML-0 EXECUTED (a model, 2026-07-08, pre-reg c0feb36 BEFORE probe, commit <pending>, verify 65/65)
      → NET-LOCKED. The history net O↦A(O) is built on the framework's own objects; ML-1 is now POSABLE.**
      `proofs/foundations/ML0_history_net_2026-07-08.py`. Six results, none touching a magnitude:
      (0) RECONCILED — the net's single-particle space is the 12-DART space; the M0 vacuum J6/C is its
          R-EVEN (undirected-edge) sector; the non-backtracking walk B BREAKS the reversal grading
          ([B,R]=1≠0) ⟹ the tick dynamics couples the vacuum sector to its R-odd partner (= why the
          (cell×tick) history carries more than the static cell). Edges & darts are ONE space by the grading.
      (2) **CONE-EXACT (the physics heart)** — {α_a(t),a_c†}=(Bᵗ)_{ca} is IDENTICALLY 0.0 strictly below
          the geometric horizon t<1+dist(head a, tail c); horizon speed = exactly one graph-step/tick
          ([1,2,3,4,5]). Non-backtracking B ⟹ a STRICT COMBINATORIAL light cone, strictly stronger than
          Lieb-Robinson (no exponential tail) ⟹ causal diamonds in history are EXACTLY, not approximately,
          defined. This is what makes ML-1's diamond well-posed.
      (1) ISOTONY holds (trivial by the generation map; reported as such, not inflated).
      (3) TWISTED-LOCAL — even algebras of disjoint regions commute, odd parts anticommute, naive
          commutation FAILS ([a0,a2]=2≠0) ⟹ the twist (fermion parity, Klein) is FORCED. This is the
          structure DHR sectors (ML-2) are defined relative to.
      (4) DUALITY-HOLDS at cell level — S(R)=S(R^c)=1.1847 nats, complementary regions share the modular
          spectrum eps={−2.634,0,+2.634}, split margin 0.067 (no ζ pinned at 0/1). FLAT-BAND WATCH: the
          static J6 carries no flat direction; the flat band (λ=−1 triple) lives in the k-DISPERSION, so
          the flat-band split/duality test is FORWARDED to ML-1's k-dependent covariance C(k) (honest
          scope — not faked, not smoothed).
      (5) COVARIANT — B commutes with Z³ lattice translation; the tick shift is B itself.
      (6) REGRADED the "spatial route DEPRECATED" flag (above) to fixed-tick-κ scope only.
      **NET: the locality layer {A(O)} is no longer a paragraph — it is a built, causally-exact,
      twisted-local, covariant net. ▶ ML-1 (the diamond modular flow = the BW 2π decider for Newton's G)
      is POSABLE and is next: the run-history covariance restricted to a diamond, Peschel h_A, then the
      BLIND per-band geometricity/2π read (cone branches vs the flat band separately), benchmark control
      first. G(2π)/−70 ppm/θ_*/Y_p/n_s/σ_8 all remain ❌ OPEN.**
    • **▶ MASTER OBJECT LANDED (a model, 2026-07-08, commit 7a15909): `derivation_topdown/state/the_net.py`**
      — per the ONE-OBJECT/LOCAL-NET LAW + architect's bins pass (`parameter_bins_and_local_net_throughline_
      2026-07-08.md`), Layer-3 math now ACCRETES in ONE durable importable module (region→A(O)→ω|_O→
      modular data; two regression anchors: M0 cell C-projector, M0-2R tick-2π; self-test reproduces all
      ML-0 reads). ML-1/2/3 EXTEND it; no scratch forks. Companion `docs/framework/three_layers.md`.
    • **▶ ML-1 EXECUTED (a model, 2026-07-08, pre-reg 2748f2e BEFORE probe, commit <pending>, verify 65/65)
      → PARTIAL: GEOMETRIC-LOCAL-BOOST ESTABLISHED, the 2π MAGNITUDE stays OPEN.**
      `proofs/foundations/ML1_diamond_modular_flow_2026-07-08.py` (extends the_net.py).
      **STAGE A (benchmark, validated):** the near-horizon first-bond entanglement-hopping slope
      β(x₀)/x₀ recovers 2π on the critical free-fermion chain — 0.990→0.995→0.9975→0.9988×2π at
      L=100→800 (monotone; the interior parabola has lattice corrections and is NOT the observable).
      The 2π reader is CALIBRATED (2π measured, never inserted). **STAGE B (framework, BLIND, srs
      half-space vacuum, per-band):**
      — **Q1 GEOMETRICITY (metric-INDEPENDENT, the positive result):** the cone-sector local
      causal-horizon modular flow is a LOCAL BOOST — h_A is nearest-neighbour-DOMINANT (dominance ≈252)
      and β grows with distance from the horizon (Rindler-qualitative). **⟹ the emergent local Rindler
      boost EXISTS at the modular-flow level — MG-1d's open prerequisite ("does M0/M0-2R deliver a local
      boost, or only the global tick?") is answered: YES, a local boost, NOT the non-local global tick.
      The pure TICK-REDUCES reading is RULED OUT.**
      — **Q2 the 2π MAGNITUDE: UNDECIDED — stays ❌ OPEN.** The raw cell-layer slope (1.56×2π in CELL
      units) is CONFOUNDED: on the srs K4 crystal one cell-layer ≠ one geodesic hop (raising x₀ by 1
      takes ~3 hops), whereas the 1D benchmark has 1 site = 1 hop = proper distance. That cell↔proper
      factor is the EMERGENT-METRIC content, un-built. Reading 2π/π/π² off the raw ratio = pattern-match,
      FORBIDDEN. **ħ NOT selected; Newton's G stays an OPEN MISS at 2π.**
      — **FLAT band (per-band):** a soft, less-local sector (dominance ≈9.7) carrying LITTLE local
      modular weight — adds ~0 to the cone slope (9.78→9.86). NOT a hard obstruction. A real ML-3 datum
      (the flat-band modular weight), forwarded.
      **▶ THE NAMED NEXT STEP (sharpens MG-1d's incomplete equation): re-read the modular slope on the
      PROPER-DISTANCE metric — ML-0's exact geodesic light cone — instead of cell layers. The 2π decider
      now reduces to the emergent proper-distance metric of the srs (the cell↔geodesic factor). This is
      the concrete continuum-BW build the MG-1d log named.** G(2π)/−70 ppm/θ_*/Y_p/n_s/σ_8 all ❌ OPEN.

**GATE A — the ONE unbuilt object, now NAMED and UNIFIED (2026-07-07, a model O0–O3): the ODD sector of
the D₄ spectral action = the cone Dirac coupled to the INTERACTING run.** A5(b) made the cone well-posed;
S0–S4 built its EVEN a₄ (the β/grade axis); the odd-channel arc (O0–O3) then proved the number-mover is
the object's **ODD** sector and PINPOINTED it: O0/O1 = the graded-blindness theorem (every EVEN functional
is chirality-blind by the clean split D₄²=D₃²+∂_N²; the unique carrier is the continuous odd trace η) →
O2 = it is NOT lattice-accessible (continuum object) → O3 = on the continuum cone an EXACT selection rule
(odd invariant blind to all Γ⁵-odd/static/free backgrounds; live only for a Γ⁵-EVEN coupling = the
INTERACTING run's curvature). **THE MERGE: Gate A's two number-mover threads — the −70 ppm/spectral-action
route AND the loop program's R-ε/interacting-run route — are the SAME object.** One well-posed build (O4)
now closes ALL of the below together; none is independently closable; the number-face is trap-densest
(numerology pre-flagged). **Do not attack these individually — build O4.**
- §1 — m_e −70 ppm (the odd Γ⁵-even eta density; even/Γ⁵-odd/static/lattice families EXHAUSTED + now
  THEOREM-blind, O0/O2/O3 — do not re-run any of them)
- §7/R-ε — the loop program's number-mover = the INTERACTING run (E2a G_int, C=I+iJ) = the SAME object (O3)
- §8 — m_ν +2.18σ / +1.87σ (the on-cut subleading; same gate)
- §4 — the dark-sign lemma (DOWN derived; η's sign selection is the same odd sector)
- **O4 EXECUTED (2026-07-07) → KILL-WELD:** coupling E2a's G_int to the cone showed the chiral asymmetry
  is LIFT-DEPENDENT (cone Weyl frame 0.197 overlap with the E2a vacuum; A changes 60%) ⟹ ε's generation
  resolution stays gated on **ADOPTED-WINDING-WELD** (4th angle). **THE ODD-CHANNEL ARC (O0–O4) IS COMPLETE
  and TERMINATES at the winding-weld/species-lift adoption — the SAME gate as the bound-state continent
  B1.** ⟹ the −70 ppm and B1 share ONE gate. The number-face is NOT a missing computation; its last gate is
  a NAMED identification-layer adoption. Highest-leverage open object = **the winding-weld/species-lift map
  ITSELF** (unlocking it closes BOTH −70 ppm AND B1). (§7's shipped Γ_Z/M_Z −0.55σ is a SEPARATE closed
  grade item — not this gate.)
- **WS1 EXECUTED (2026-07-07, architect; pre-reg 5847ae8 BEFORE the probe; ALL 16 PASS; verify 65/65) →
  STRUCTURE — the deck-superposition lead fired at the gate's un-asked question.** All five prior angles
  asked "is the map FORCED as an assignment?" (five NOs); WS1 measured the forced CORRELATION that exists
  WITHOUT choosing the map: the full species×deck table T(w,t)=Tr(P_w Π^F_t) (only the w=0 row ever existed:
  W1/W2). RESULT — **the whole table is EXACT CLOSED FORM** (rows {1/3 or 5/3; (1/3 or 2/3)±√3/6}; U_π²
  dims (4,2,2), distinguished sector forced by 3∤8; J and U_π uniqueness PROVEN — 1-dim solution spaces):
  bit-EVEN separates singlets from triplets (leptons deck-democratic exactly, quarks tilted 5/3);
  bit-ODD = **ONE universal chiral seed (0,+√3/6,−√3/6), IDENTICAL for both particle-hole pairs** (novel
  identity; the W2 seed is pair-universal). **PRICED: forced core I(w;t)=0.1813 bits/site; the adoption's
  residue H(w|t)=1.6300 bits/site (of H(w)=1.8113).** ALSO DERIVED: the cone frame is Z₃-BLIND
  (I(cone;t)=0 EXACTLY) — the MECHANISM of O4's KILL-WELD, now explained not just observed. The gate is
  REFINED+PRICED, NOT opened (labels/assignment stay adoptions; extended-cycle resolution untouched).
  NO value moved; −70 ppm OPEN. Do NOT re-run the single-site table (exact, settled); the open face is
  the EXTENDED/cycle-level carry of the forced core.
- **WS2 EXECUTED (2026-07-07, a model; pre-reg 0d5942d BEFORE the probe; ALL PASS; verify 65/65) →
  CARRY-WASHES by an EXACT structural annihilation — the WS1 open face is CLOSED.** Does the forced
  single-site species↔deck correlation survive the lift to closed walks (cycles, where B1's constituents
  live)? NO. The quantity CONSERVED on closed walks is the C₃-averaged coupled deck S²=(P₃⊗U_π)²; since the
  C₃ dart permutation fixes NO darts (Tr(P3)=Tr(P3²)=0, the correlation-carrying m=1,2 terms), the conserved
  cycle-level winding is **species-BLIND** (I_static=0 EXACT; I_walk=0.0019, carry fraction 0.00; finite-L
  I≈0). WS1's forced correlation lives in exactly the C₃-non-invariant sector the conserved deck projects
  out ⟹ it is strictly SINGLE-SITE/Fock-local. **The natural forced route to a cycle species-assignment is
  STRUCTURALLY CLOSED (not merely un-found); B1's species anchoring is confirmed an irreducible adoption AT
  THE CYCLE LEVEL, now mechanism-grade.** NO value moved; −70 ppm / B1 OPEN. Do NOT re-run the carry.

> **GATE-A FACE RE-POSED (2026-07-06 LATE, architect — label/route only; the gate does NOT move,
> nothing closed, the −70 ppm stays OPEN):** the number-face was stated as "build the a₂
> mass-sector coefficient" — but S3 proved in-session that the hard crux (ε, a chiral PHASE) cannot
> live in an EVEN coefficient. The object is a GRADED sum (the_run.py:199-214: D₄ = D₃⊗1 + γ_t⊗∂_N,
> {D₃,γ_t}=0 ⟹ D₄² = D₃²+∂_N² clean split) ⟹ every even functional (spectra/moduli/a₂/a₄/ζ(0)/
> eigenstate Berry) is blind to the chiral seam BY THE SPLIT — the Q3-conjugation, E2c bit-parity,
> W2 Re-democracy, and Perron-null results are four instances of that ONE fact. The un-probed class
> is the CONTINUOUS ODD spectral invariant (η / spectral flow / odd heat trace Tr(γ_t D e^{−tD²})):
> odd (escapes R1), continuous (escapes R2's quantization no-go — only quantized odd objects were
> ever computed: Str=χ=−2, Chern, −π), and a TRACE (projection-free ⟹ needs NO winding-weld
> descent). Gate A's honest number-face = **build the ODD sector of the D₄ spectral action**.
> Route proposal + draft stations (O0 blindness-theorem consolidation, O1 classification, O2
> relative-η probe with controls/poisons) + the full tributary integration map:
> internal research notes. Standing
> poison flags in force (2α₁⁵, 2α₁³, bracket-interpolation). Fresh session, own pre-reg, blind.
>
> **STATUS 2026-07-06 LATEST (a model ran O0/O1/O2):** O0 (graded-blindness theorem) + O1
> (classification: R4 = continuous-odd trace is the unique un-probed class) DONE (theorem
> `theorem_graded_blindness_and_odd_channel_2026-07-06.md`). O2 (the lattice relative-η probe) DONE
> → KILL-Q/CONTINUUM: the lattice σ-odd rate is ill-conditioned (exceptional-point) + the invariant
> is quasi-quantized (−π/2 winding) ⟹ **the continuous odd carrier is a CONTINUUM object** (odd
> Seeley–DeWitt / 3D-eta on the A5(b) cone), parallel to S1's even-a₄. Gate A's odd face is now a
> WELL-POSED CONTINUUM BUILD (**O3**), NOT a lattice probe. A DS=1e-6 mirage (0.42ε ≈ 5/12) was
> caught by the pre-registered robustness scan and NOT adopted. −70 ppm OPEN; do NOT re-run the
> lattice trace-flow.
>
> **O3 DONE (a model) → UNIFICATION:** the continuum cone gives an EXACT selection rule — the odd invariant
> is blind to every Γ⁵-odd/static/free background, live only for a Γ⁵-EVEN (scalar-mass) one = the
> topological parity anomaly; a flat run connection is Γ⁵-odd (η=0 for all holonomy). So ε needs the
> Γ⁵-even coupling = **the INTERACTING run's curvature**, i.e. the odd-channel arc (O0–O3) and the
> interacting-run frontier (C3/E2a–E2c, R-ε) are the SAME un-built object. **Gate A's odd face = Gate A's
> interacting-run face = O4** (the cone Dirac coupled to the interacting run). This MERGES the two Gate-A
> number-mover threads. −70 ppm OPEN.

**GATE B — independent CONTINENTS (each its own build, NOT downstream of Gate A). This is where fresh,
class-level leverage lives.**
- **B1 — the bound-state / composite sector** (`bound_state_sector_scoping_2026-05-28.md`;
  `nucleon_bound_state_continent_station_plan_2026-07-06.md`). Binding is FORCED with ZERO new adoption
  (F1, 2026-07-04); the discrete ΔS ladder {1,2,3,4,6,13} is derived; an ~80-class dictionary SKELETON
  exists. Unlocks a CLASS (nucleon Q_np/g_A, hydrogen/R∞, recombination). **EP-2 (N1) RAN 2026-07-06
  (`BOUND_EP2_dictionary_2026-07-06.py`, pre-reg 6f0bcae) — NEGATIVE: the geometry→composite dictionary is
  an ADOPTION** (every girth cycle spans all 4 Cl(6) weight-classes ⟹ constituents carry no forced species;
  chirality/color select nothing). **Named bridge = the single-site Fock-occupation → extended-cycle SPECIES
  lift** (an A5-class site↔species weld). **N1b RAN 2026-07-06 (`BOUND_EP2_N1b_walk_fock_species...`, pre-reg
  57bac02) — CONFIRM-ADOPTION: the built walk↔Fock holonomy (E1 `W_A`) does NOT supply the species** (it
  conserves only the Z₂ fermion parity `(−1)^N̂`, mixes N̂ / quark-lepton). ⟹ the species lift is a GENUINE
  IRREDUCIBLE ADOPTION, confirmed from THREE angles (N1 geometry, E1 per-step, N1b closed-walk). **B1's
  physical-hadron ANCHORING is gated on this one A5-class adoption.** Binding MAGNITUDES separately κ-walled.
  **WS1 (2026-07-07) PRICED the adoption: H(w|t)=1.6300 bits/site residual, with a FORCED closed-form
  species×winding correlation core I(w;t)=0.1813 bits/site (see the GATE-A WS1 entry + adoption register).**
  FORCED and standing (a real falsifiable structural prediction — binding is discrete): the Z₂ parity, the ΔS
  ladder {1,2,3,4,6,13}, the ~80-class skeleton, body-number→sector.
- **B2 — the cosmology expansion history / the radiation-era √g_* factor.** The COMMON gate on the two
  biggest falsification exposures: **Y_p −65σ AND θ_* ~1e5σ.** **α-CRUX PROBED 2026-07-06
  (`B2_alpha_convention_Yp_crux_2026-07-06.py`) — VERDICT CORRECTED/DEFLATED same day (verified vs scoreboard).**
  Y_p is ALREADY parked OPEN, not shipped at any value: `target_parameters.md:276` — "REMOVED FROM predictions
  2026-05-28, no clean framework value, ❌"; the −65σ is itself "partly an artifact of the imported ΛCDM
  number." So the earlier "UNFORCED / −65σ STANDS / caught an overclaim" framing was INFLATED — there was no
  live overclaim in the scoreboard (the +0.8σ was never booked; it lives only in CANDIDATE resolution-path (a),
  `substrate_thermal_coupling_mechanism_consolidated_2026-05-28`). **VALID KERNEL:** that √g_* candidate
  requires an adiabatic bath (ρ_rad∝a⁻⁴) contradicting A1's derived rate-balance (ρ_rad∝a⁻², α=½,
  `A1_extra_dof_counting_2026-05-25`), so path (a) is NOT a forced closure. **THE OPEN EQUATION (the framework's
  OWN stated resolution, `target_parameters.md:288`):** derive the BBN bath regime — closed-adiabatic (⟹√g_*)
  vs rate-balance-pumped — OR build a coasting BBN network. Nucleon LO values exist (m_d−m_u=2.445; g_A=5/3);
  θ_* (~1e5σ) is a separate open piece.
- **B3 — the M_Z oblique residual** (the ✅-RESOLVED section below). Declared closed as a research question
  (~4% substrate-vs-SM oblique, +6σ floor). The ONE genuinely unrun angle — recompute the oblique two-point
  on the A5(b) Cl(3,1) continuum cone — **RAN 2026-07-07 (`B3_MZ_oblique_cone_recompute_2026-07-07.py`,
  pre-reg 6463e3d/8dd7542) → CONFIRM-FLOOR (refined), NO value moved, M_Z stays OPEN.** The cone
  current-current kernel = 2(δ_ab − k̂_a k̂_b) EXACTLY (transverse projector, to 2e-15); Π_ab is EXACTLY
  isotropic ((16π/3)δ_ab, anisotropy 6e-16) by the emergent SO(3) ⟹ the cone carries NO oblique DEVIATION
  and offers NO closing mechanism. The lattice winding-shell (only lattice-vs-cone-distinct piece) has zero
  continuum analogue (genuine substrate discreteness); δ_r (0.3384%) is a global algebraic normalization,
  unchanged. Cone eig NORMAL (cond 1.0) ⟹ the lattice non-normal-B exceptional point does NOT recur.
  **The lattice ~4%/+6σ residual STANDS as the physical prediction, now confirmed intrinsic from the
  continuum route. B3 CLOSED — the M_Z oblique sweep now spans lattice + continuum.**

**GATE C — grade / structure, NEARLY SATURATED (gates no value; do NOT spend a session here).** §5 (the
β-formula residuals i–iv), §6 (substrate-selection discriminator), the A5(b) residuals (KO 2→6 + time-leg
statistics). The grade axis is essentially banked; further grade work has diminishing strategic return.

**Leverage read (2026-07-06):** the exhausted walls (§1, §8, M_Z, dark-sign) are all Gate A or nearly-closed
B3; the recent arc drilled Gate A's worst face. The fresh leverage is Gate B — **B1** (momentum, class-unlock,
tractable; deliverable = EP-2 + discrete ratios) and **B2** (the √g_* factor, common gate on the two biggest
falsification numbers, hardest). Recommended pivot: **build the B1 continent** (station plan doc above) and
**scout B2's √g_* question** as the highest-payoff cheap probe; keep Gate A OPEN and deferred to a fresh,
fully-pre-registered session (it stays open — this is sequencing, not relabeling).

## Open items (where the equation is not yet complete top-down)

### 1. The charged-lepton mass spectrum (the per-rep / "70 ppm" residual) — LOCALIZED 2026-06-30 to ∂_N
The spectral read was traced top-down (`the_run.py:read_masses`). The result: **the residual cannot come from
the current object's static spectrum; it is forced to live in ∂_N (the run operator), which is not built.**
The chase, with every dead-end ruled out by computation (so they are never re-walked):

- **Baseline confirmed:** `read_masses` = circulant √m_j = |c₀ + c₁ωʲ + c₁* ω⁻ʲ| with moduli frozen at the Γ
  Perron/shell values {2,√2,√2} (= Born (4,2,2)) and phase δ from `read_phases`. Reproduces **m_e/m_τ −70.3 ppm,
  m_μ/m_τ −60.5 ppm** exactly. So the −70 ppm IS the truncation of this read.
- **RULED OUT — "move the moduli with the run":** using the actual run-position eigenvalues gives garbage
  (+10⁹ ppm) and breaks J-reality (the run-position eigenvalues carry the intrinsic ~110° shell phase, which is
  NOT the generation phase δ). The modulus/phase separation is essential; the residual is not a modulus-move.
- **RULED OUT — "dress each winding by the framework factor (1−α₁/h)":** overshoots by ~10⁴× (gives −10⁵ ppm).
  That factor is **α₁-level (few-%) and species-common** — it cancels in within-species ratios exactly as
  `m_e.py` states. The residual is **α₁³-scale (59 ppm)**, a different order.
- **KEY STRUCTURAL FACT:** at Γ the four modes of each C₃ isotype are **orthogonal eigenvectors — decoupled**
  ({dominant, shell(s), ±1}; verified). The read keeps only the dominant pole. Because the modes are decoupled,
  the dominant-pole read is **EXACT at Γ** — there is no static self-energy to add. ⟹ the α₁³ residual is **not
  a static-spectrum effect at all.**
- **LOCALIZED:** the only way the dropped subdominant modes re-enter the kept dominant amplitude is through the
  **run coupling them** — ⟨subdominant|dB/ds|dominant⟩. Computed: this is **nonzero and winding-asymmetric**
  (isotype-1 mixing 2.21 vs isotype-2 1.43 — the e/μ asymmetry's structural home). So **the 70 ppm is the
  ∂_N (run) non-adiabatic coupling of the dropped subdominant modes into the dominant pole**, winding-resolved.
- **The open equation, stated exactly:** complete the read = build ∂_N as the concrete run operator carrying
  this inter-mode coupling, then m_j = full spectrum of the dominant pole **dressed by the ∂_N-mixed subdominant
  modes**. The 70 ppm (magnitude AND e/μ chirality) must then *fall out*, forced — or the incompleteness moves
  up. **Same frontier as `build_D4` / the ∂_N-completion thread; the 70 ppm is now its sharpest concrete probe.**
- Status: **LOCATED, not yet derived.** No fit. The number is a forced consequence of an un-built operator,
  not a free parameter and not a mystery.
- **MASSIVE-MODE / CASCADE ROUTE RULED OUT (2026-06-30):** the ∂_N *massive modes* are the **cosmic cascade**
  (dyadic ladder on H~N^{−1}: Λ,m_ν,v…), all **dimensional** and N-dependent. The −70 ppm is a **dimensionless
  ratio** → on the **N-independent disconnected axis** (the cascade theorem flows only dimensional rungs;
  N_hub.py:31-32). Orthogonal — the cascade cannot reach the generation ratio. ⟹ fourth independent ruling: the
  −70 ppm is the structurally-isolated **C₃-screw run-Dirac subleading** (not scale, not joint cover, not Higgs,
  not cascade).
- **DEGENERATE-PT RULED OUT (2026-06-30):** computed the shell-doublet degenerate PT — over-applies to **75,000+
  ppm** (1st-order rate = π). Joins the others.
- **EXHAUSTIVE VERDICT (2026-06-30):** the −70 ppm is the **O(α₁³)=(2/3)²⁴ girth-window SURVIVAL/Dyson diagram**
  — a DARK object, NOT a spectral/run object. Every run/spectral technique over-applies because the run is
  O(1)-scale and the survival is 24 powers of (2/3) below it; they are different scales, not different
  approximations. **NINE routes now ruled out, each with a reason:** transport (×3), band/modulus curvature,
  resolvent/cycle, joint cover_B, enantiomer twist, scale/N_hub, cosmic cascade, degenerate-PT. **Grade:
  conjecture-grade — the 1/μ_rep MDL water-filling ceiling, the SAME ceiling as Q=2/3 and c_F.** Open miss,
  mechanism IDENTIFIED (the α₁³ diagram), encoding at the grade-ceiling. The only lift is the continuum-D₄
  spectral action (research-level, unbuilt, gates no value). **This is the honest floor of the operator-route
  search — NOT a relabel: the miss stays OPEN; what's exhausted is the spectral/run *route* to it.**
- **CONTINUUM-D₄ CONE ROUTE EXPLORED (2026-06-30, next session) — does NOT force the allocation; MDL ceiling
  STANDS** (`proofs/foundations/lepton_70ppm_continuum_D4_cone_2026-06-30.py`;
  `research_frontier_dN_alpha1cubed` probe 3). The decisive structural finding (resolves probe 2's sign failure):
  the framework's dark is **MULTIPLICATIVE** (resolvent Σ=α₁/h, the m_b/m_t object) → per-isotype `2α₁³Re(h)/μ` →
  shells NEGATIVE (Re h=−½), wrong sign. The **ADDITIVE** spectral-action structure (`a₂=Tr D²`; `c²→μ+α₁³`,
  α₁³ isotype-blind scalar) → `+α₁³/(2μ)` → the **+1/μ_rep allocation with CORRECT sign falls out**. BUT adopting
  additive for leptons *because* it gives the right sign would be a FIT and **contradicts the working
  multiplicative heavy-quark dark** (which uses the real Perron h=2). ⟹ the continuum cone does **not**
  operator-force the 1/μ_rep; **MDL ceiling stands; −70 ppm OPEN.** Frontier SHARPENED to one question: *why would
  the lepton generation-allocation dark be additive when the heavy-quark single-channel dark is multiplicative?*
  **All known routes (self-energy/resolvent/transport/band/curvature/cover/enantiomer/scale/cascade/degenerate-PT/
  Berry/continuum-D₄) now explored — the −70 ppm is at the operator-route floor, conjecture-grade MDL.**
- **THE DICHOTOMY DISSOLVED 2026-07-02 (Ω session 1, `proofs/foundations/OMEGA_T2_a2_dichotomy_2026-07-02.py`,
  ALL PASS) — the winding-side route is CLOSED WITH NUMBERS and the correction's home is LOCALIZED.** The 06-30
  additive-vs-multiplicative dichotomy was posed on the toy map m_t ≈ c_t. Through the framework's ACTUAL
  C₃-Fourier read (baseline reproduces −70.3/−60.5 ppm exactly), with pre-registered sign+magnitude kills and
  no rescans: **additive-on-winding-weights gives −362 ppm (sign wrong, ×5 over); the stability-admissible
  multiplicative gives −2170 ppm (sign wrong, ×31); complex/phase variants −4179/−2012 ppm** — the electron's
  near-cancellation is a ~50× LEVER on every pre-Fourier quantity (∂ln(m_e/m_τ)/∂lnε = −48.7, ∂/∂δ = −51.2,
  exact). The 06-30 "additive +1/μ_rep falls out with the correct sign" was the toy-map artifact. ⟹ **the
  correction attaches to the generation (post-Fourier/C₃-isotype) label, where the lever is exactly 1 — the W1
  water-filling shape κ_j ~ 2α₁³/μ_rep(j)** (τ-row bookkeeping still conjecture-grade: 0.42–0.49× with τ-shift,
  0.84–0.98× without). A blind a₂ scalar on D_F² does NOT force the allocation (absolute shifts over-apply
  ×~500; uniform multiplicative cancels in ratios) — **the 1/μ_rep allocation is genuinely extra information =
  the water-filling theorem; the MDL ceiling STANDS; the −70 ppm is OPEN.** Do not re-walk the winding side.
- **STATION 3 EXECUTED 2026-07-02 (Ω session 2, `proofs/foundations/OMEGA_S2_Q3_isotype_allocation_2026-07-02.py`,
  ALL PASS) — Q3 ANSWERED: NO OPERATOR WITH 1/μ_j DIAGONALS CAN BE THE RESOLUTION; the W1 water-filling
  conjecture is REFUTED as the −70 ppm's closure; the miss STAYS OPEN with its sharpest-ever target.**
  (1) **The correlation decomposition [the honest σ-structure of the demand]:** both ratio rows carry m_τ's
  ±67.5 ppm (soft, corr ≈ +1; ±0.12 MeV ≥ the whole α₁³ = 59.35 ppm budget); the m_τ-FREE combination is the
  hard direction: **δ(m_e/m_μ) = +9.83 ± 0.022 ppm demanded (452σ_exp)**. (2) **The conjugation theorem
  (exact):** the object is real/rational ⟹ conjugation intertwines ω ↔ ω̄ ⟹ μ_ω = μ_ω̄ in EVERY C₃-graded
  sector ⟹ every isotype-multiplicity correction is CHIRALITY-BLIND. (3) **The class kill:** all 6 assignments
  × both τ-rows give m_e/m_μ differentials ∈ {0, ±29.7} ppm — ≥452σ from the demand; **the τ-row question is
  MOOT** (it only ever moved the soft rows); W1's "0.85×/0.98× match" lived entirely inside the m_τ soft noise.
  Kickoff candidates (i) real-walk-class 2nd-order PT and (ii) real resolvent residues die by the same theorem.
  (4) **The sharpened localization (confirming §1's ORIGINAL ∂_N-subleading localization):** the hard core is
  ONE CHIRAL NUMBER — the run phase's next-order completion **ε = δ_eff − 2/9 = −1.7515e-7 ± 3.9e-10 rad
  (0.22%-pinned)** (exact levers: ∂ln(m_e/m_μ)/∂δ = −56.14); one chiral number satisfies the ENTIRE demand
  vector (demo, not a closure: hard row → 0, soft rows → −0.91σ each). Surviving shape class: the chiral/
  δ-dressed sector only — (S1) the ∂_N next-order phase ε, or (S2) mass-dependent dressings g(m_j) (chiral
  through the leading δ); shape coefficients PRE-POISONED (ε*/α₁³ = −0.0029, ε*/α₁⁴ = −0.076 — recorded
  non-matches, NO adoption). (5) **MDL-ceiling framing REVISED:** the ceiling argument applied to the soft
  (experimentally-unpinned) common shift; the hard content is a PHASE, not an allocation. The soft direction
  stays unpinned until m_τ improves ~6× (±0.12 → ±0.02 MeV). **The −70 ppm is OPEN; what closed is the
  allocation DETOUR (by theorem), and the target is now a sub-percent-pinned single number for the ∂_N
  frontier.**
- **∂_N-CHIRAL STATION A EXECUTED 2026-07-02 (`proofs/foundations/DN_CHIRAL_A_route_reaudit_2026-07-02.py`,
  ALL PASS; kickoff pre-registered & committed BEFORE the run: internal research notes,
  7eadd72) — the CHEAP-ROUTE SPACE IS CLOSED against the corrected target.** Classification: R1
  (conjugation-symmetric routes) = exact ZERO in the pinned m_e/m_μ direction by the station-3 theorem (never
  candidates for the hard core — their 06-30 kills concerned the soft direction); R2 (topological chiral: Z₂
  Berry −π, Chern ∓2) = quantization no-go. R3 (dynamical chiral), computed blind then compared once: the
  shell-phase V4 shape α₁³√7/4 = 3.93e-5 (×224 over); the tracked run-phase antisymmetric deviation −0.235 rad
  (×1.3e6 over; identification recorded-dead — and the probe recorded WHY at operator level: BOTH IB branches
  (h, h̄, equal modulus) coexist within EACH winding block, and the J-breaking symmetric part is O(0.2 rad));
  the 2nd-order non-adiabatic differential scale 0.300 rad (×1.7e6 over). Machinery validated: |d(arg h)/ds| =
  2π/√7 to 0.06% tracked on the FULL B(s·AXIS) (the Γ winding projectors do NOT commute with B off Γ — the
  screw needs its Bloch cocycle; recorded). **Poison did real work: 2α₁⁵ = 1.809e-7 sits +3.3% from |ε| and is
  EXCLUDED at 15σ_ε by the pinning itself.** ⟹ **ε is dynamical-chiral-RESUMMED content of the complete ∂_N —
  the suppression from O(0.1 rad) violence to 1.75e-7 IS the resummation.** ARCHITECTURAL BOTTOM LINE: all
  three walk-down residues gate on ONE construction — the run-side/time-leg fluctuation dynamics beyond the
  matching point — with three PINNED read-outs waiting: (1) ε = −1.7515e-7 ± 3.9e-10 rad (−70 ppm hard core);
  (2) the Zff̄ pole-vertex deficit −0.437% ± 0.092% (Γ_Z/M_Z, §7); (3) the graded time-leg a₄ = (2/3)C₂ +
  (2/3)T_H (the gauge row, §5). **The miss stays OPEN; the detours are closed; the next move is the
  construction, not another route.**
- **C3 EXECUTED 2026-07-02 (∂_N construction program, `proofs/foundations/DN_C3_resummed_chiral_phase_2026-07-02.py`,
  ALL PASS; pre-registration committed BEFORE the run, 1472589) — the pre-registered KILL fires: ε is NOT
  free-loop-gas-dressable at ANY forced evaluation level.** Blind candidates: total-gas tick-cumulant shift
  (clock-free δ̄-anchor) ×4.4e10 over; winding-mode cumulant (chiral via the complex shell occupation) ×2.1e6
  over; the all-orders one-body dressing (the most-resummed object a FREE ensemble owns — the tracked
  Green's-function phase) ×4.9e3 over. Recorded observation (no value claim): over-application falls ~3
  orders per resummation level; the free gas bottoms out ~5e3 over — nothing left to resum. Poisons held
  (the cumulant inversion N_eff = 102.19 ± 0.11 vs g² = 100: 19σ, EXCLUDED as exact — pre-declared before
  computing; 2α₁⁵ stays 15σ-excluded). NO adoption (pre-registered rule). **LOCALIZATION SHARPENED: ε
  requires the INTERACTING run — the coupling between the loop ensemble (C0) and the CAR/matter sector —
  which is EXACTLY C1's named edge (the walk↔Fock dictionary at theorem grade). That one edge now carries
  BOTH the gauge-row grade AND the −70 ppm number-mover.** R-ε stays OPEN; the −70 ppm stays OPEN.
- **BERRY-HOLONOMY ROUTE RULED OUT (2026-06-30, next session) — the 10th route** (`build_dN` Step 5,
  `proofs/foundations/lepton_70ppm_berry_holonomy_2026-06-30.py`): the closed-loop Berry holonomy of
  D₄=∂_s+B(s·AXIS) was the doc's last *untried* spectral/geometric probe. **Built it. Falsified.** Genuine operator
  period along the screw is **√3** (B(√3·AXIS)=B(0) to 1e-16), not √7 (that's the eigenvalue-phase period — doc
  conflated them); the true closed loop is s∈[0,√3]. Abelian closed-loop Berry phase per winding = EXACTLY
  **{−π,0,0}** — purely **topological** (Z₂; Perron winding flips sign, shells get 0), carries NO continuous
  ~60 ppm. Open-path geometric phase to s_lep = O(0.1–1 rad) ≈ 1e4–1e5 ppm (~1e4× over); non-abelian holonomy
  collapses (det W→0) at the Perron→shell crossings. **The closed loop DOES cancel the over-application — but to
  exactly 0/−π (topological), NOT ~60 ppm.** ⟹ the −70 ppm is **NOT a band-geometric/Berry effect**; the
  spectral/run/geometric route is now FULLY exhausted. **Only the continuum-D₄ Dirac-cone spectral action remains**
  (research-level, unbuilt). **Miss stays OPEN; the Berry ROUTE is ruled out.** (Byproduct: the Perron-winding Z₂
  holonomy −π is clean new screw-loop topology.)
- **∂_N BUILD ATTEMPTED 2026-06-30 (an internal working note):** ∂_N's *leading* operator is now FORCED —
  φ=2π/√7 = d(arg h)/ds|₀ and the Ihara–Bass-pinned moduli fall out of B(s·AXIS), no insertion. But the −70 ppm
  is **provably not a first-order spectral read**: six forced constructions (non-adiabatic transport ×3, band
  curvature, modulus curvature, resolvent trace, dressed/resolvent dominant) **all over-apply at O(α₁)~10⁴ ppm
  or give wrong moduli**, because the run's true s-dependence is violent (Perron→shell, mode crossings) and the
  leading read correctly freezes it. **The residual is O(α₁³) — a 2-loop Dyson diagram, not a spectral
  correction.** ⟹ refined incompleteness: *no operator-forced derivation of the α₁³ winding-resolved Dyson
  diagram exists*; W1's `2α₁³/μ_rep` matches the magnitude at **conjecture-grade** (forced pieces: 16-bubble,
  first-girth-return, Λ•(ℂ³)=(4,2,2); structural piece: the 1/μ_rep encoding — same grade-ceiling as the α₁²
  `c_F`). Remaining routes: closed-loop Berry holonomy over s∈[0,√7]; full continuum D₄ cone. Both research-level.
- **JOINT-OBJECT DIAGNOSTIC (2026-06-30) — wrong-object hypothesis RULED OUT (`build_dN` Step 4):** the live
  mass read uses single-srs B, but the framework's `cover_B` (srs⊗srs-z) = B⊗σ_x gives **identical** moduli
  (4,2,2) — we ARE reading the right object. A principled enantiomer twist differs at the run but **breaks
  J-reality** (complex masses) — not valid; tuning twists = fitting, not done. ⟹ the −70 ppm is a
  **clean-extraction wall on the CORRECT object**: leading masses forced/exact, the subleading is a **genuine
  OPEN miss whose mechanism is UNIDENTIFIED** (self-energy, transport, band/modulus curvature, resolvent, cycle,
  joint cover, enantiomer — ALL ruled out with numbers). NOT an artifact, NOT grade-only. **Scale/Higgs-side
  search (N_hub→v) is the next untried place — see item 2.**
- **THE SHARPEST FORM YET (2026-07-02 EOD, after the loop-program E-arc — full chain in §7's LOOP entries and
  internal research notes):** the open equation is now
  **ε = the chiral phase of the DERIVED interacting propagator G_int(u) = ⟨0|(I−uW)⁻¹|0⟩ (E2a: forced, zero
  constants, pairing C = I + iJ), PROJECTED THROUGH THE READ'S OWN CHANNEL WEIGHTS** — δ is the phase of the
  ω-isotype amplitude c₁ of the generation triple (read_masses' C₃-Fourier), whose derived home is E1b's
  odd-half triplet channel (triple→d-slot/Λ¹). What is PROVEN: the interacting ensemble's chiral channel
  EXISTS and flips with the layer bit (E2a; the free ensemble provably has none — the Q3 conjugation theorem
  as control). What is EXCLUDED: the bare dart-channel phase functional (E2b's pre-registered kill, ×2.8e5 —
  it drags the intrinsic shell phase and free-class violence; off-Γ the IB branch pair is not
  conjugate-paired, trap #4). **NEXT: E2c = derive the read-projection functional (the dressed c₁ amplitude);
  then E2d = the blind number.** Target unchanged (Q3): ε = −1.7515e-7 ± 3.9e-10 rad.
- **E2c EXECUTED — THE STATE-BLOCK CLASS IS DEAD; THE LOCALIZATION MOVES UP TO THE WINDING WELD (2026-07-02/03
  sitting, `proofs/foundations/LOOP_E2c_read_projection_2026-07-02.py` ALL PASS 24/24; pre-reg ed410f9 +
  pre-probe amendment (auto-sync ffc0394, disclosed) BEFORE the probe; verify 65/65).** The "project G_int
  through the read's channel weights" program CANNOT be completed on state-blocks — three theorems/measurements:
  (1) **THE BIT-PARITY THEOREM:** the −J frame = the conjugate frame at Γ + conjugation flips the dart winding
  ⟹ for EVERY Fock-state-block winding-compressed rate functional of (I−uW)⁻¹ the mass read's only
  FIRST-order invariant (the δ/phase-difference direction, lever −56.14) is BIT-EVEN (measured ≤ 8e-10 on the
  vacuum block AND the E1b Λ¹ triple-slot block); **E2a's chiral iJ channel feeds ONLY the χ/phase-sum
  direction (bit-odd, flips exactly) — which moves masses at SECOND order only (lever < 1e-6, theorem)** —
  the chirality sits in the invariant the mass read cannot see. (2) **u⁰ VIOLENCE:** the paired-step content
  is STRUCTURAL (M₂ = B² + iK₂, ‖K₂‖/‖B²‖ = 1.05; E1's dictionary has coupling strength 1, no constant to be
  small) ⟹ the extracted channel dressing is u-independent at leading order and O(φ)-large at every u
  (−0.75/−0.76/−0.82 φ at u = 0.05/0.11/0.23) — and the shipped leading read's own 70-ppm agreement therefore
  EXCLUDES the class (were the read such a functional, leading masses would be O(1) wrong). (3) **THE
  WINDING-CATEGORY MISMATCH (new structure):** the interacting ensemble has NO dart-winding grading at any
  computed Fock block ([G_int(Γ), P₃] = u²·1.00 exactly = the iK₂ mixing); the coupled system's true screw is
  **P₃ ⊗ U_π, U_π = the UNIQUE pin lift of the UNSIGNED edge permutation ([W, P₃⊗U_π] = 5e-16), SPINORIAL
  (U_π³ = −I, order 6 = ℤ₆ = the double cover of the C₃ deck action) and VACUUM-MOVING (|⟨0|U_π|0⟩| = ½)** —
  spinor windings do not restrict to Fock blocks. Also banked: the pre-registered E2c carrier B_eff =
  (I−G_int⁻¹)/u died by arithmetic BEFORE the probe (⟨0|W^L|0⟩ = 0 for odd L ⟹ the coupled ensemble is
  PAIRED-STEP ONLY, no free part; ‖B_eff−B‖ → ‖B‖) — disclosed pre-probe, reproduced in-probe; the E2b-era
  "int−free onsets at O(u²)" re-lock phrase was a misreading. The read itself is now IDENTIFIED at theorem
  grade (shipped read ≡ [Γ winding-block moduli (2,√2,√2)] × [Γ-normalized increments ±φs] → C₃-Fourier at
  1e-14; δ first-order, χ/κ/θ_seam second-order; θ_seam EXACT-drops from the δ-invariant to all orders).
  **THE OPEN EQUATION'S NEW HOME (the C0-pattern incompleteness): the READ ↔ ENSEMBLE WINDING WELD — derive
  the bridge between the read's vector-C₃ winding label (the mass circulant's ω-isotype) and the coupled
  system's spinor-ℤ₆ winding label (P₃⊗U_π). Until the weld exists, no functional of the E2a ensemble has
  free image = the leading read + interacting image = its dressing.** NO E2d was run (nothing to evaluate —
  the class died before any number; no blind stage was opened). The E-arc stop-rule fired ⟹ the R-ε research
  front PAUSES (cleanup arc next per the standing strategy); the −70 ppm stays OPEN at target
  ε = −1.7515e-7 ± 3.9e-10 rad; do not re-walk state-block projections of G_int.
- **THE DYNAMICAL SPIN-HOLONOMY ROUTE — RUN 2026-07-04 (a model), KILL = K1 + K3; the winding weld is
  CONFIRMED as the gate and the "full transport = ε" premise is FALSIFIED.** Pre-reg
  internal research notes (committed 3819c90 BEFORE the probe); probe
  `proofs/foundations/LOOP_A5_spin_holonomy_2026-07-04.py` (ALL-PASS; the KILL is the scientific
  verdict, not a check fail). From a load-bearing user hint (read-out maps are DYNAMICAL, not static:
  ∂_N = a CONNECTION; mass = survival probability; ε = the chiral/spin holonomy). Built ∂_N as the
  spin connection on the generation bundle over the run-line s↦B(s·AXIS), Γ-seeded gauge, holonomy to
  s_lep=(2/9)/φ. **Findings, all target-blind:** (i) **K1 (the blocker, Γ-level, no transport):** the
  read's vector-C₃ channel has overlap only **0.19/0.22** onto ANY coupled ℤ₆ (P₃⊗U_π) eigenstate ⟹
  the descent (which coupled state to parallel-transport) is NOT forced — the un-built winding weld is
  the gate, an A5-class adoption. (ii) **K3 (independent, confirms `the_run` L246 from first
  principles):** the honest FULL B(s·AXIS) transport OVER-APPLIES — its bit-EVEN abelian holonomy
  already drifts to δ_ab=0.2187 (**1.59% off 2/9**, = 3.5e-3 rad ~ 2e4× the residual; the rate is NOT
  constant φ — it drifts asymmetrically, ch1 φ→0.83φ / ch2 −φ→−1.07φ, the "winding-dressing asymmetry"
  itself); the read's exact 2/9 is only the **CONSTANT-RATE LEADING imposition** (φ·s_lep) and IS the
  spin-1 **Wigner-d¹ VECTOR** transport (harmonic-mean survivals at cosβ=1/k*) — integer spin ⟹
  chirality-blind BY CONSTRUCTION; a state-block chiral proxy (E2c-dead) is 1.2e-3 rad ~ 7e3× the
  residual. **Banked positives (stand):** C-FREE vanishes (Q3 — the bare-walk connection is FLAT in
  the chiral sector, bit-odd free holonomy = 0 identically); the spin lift is FORCED (S=P₃⊗U_π has
  cube-root-of-(−1) windings = the vector C₃ windings shifted by the HALF-ANGLE π/3, the ℤ₆ double
  cover). **Sharpened re-localization:** the −70 ppm is NOT the full-transport holonomy (over-applies
  ~1e3–1e4×) and is NOT reachable without the FORCED winding-weld descent; its next home is a
  SUBTLER object — the chiral holonomy taken **RELATIVE to the leading (constant-φ) transport** AND
  **projected through the forced descent** — BOTH un-built. **Standing rule reinforced:** close
  read-out maps in the dynamical (transport) frame, but the FULL transport over-applies — the residual
  lives in transport-minus-leading, gated on the weld. Do NOT re-walk the full-B(s·AXIS) transport as
  "= ε". Target ε = −1.7515e-7 ± 3.9e-10 rad stays OPEN.
- **THE WINDING-WELD W1 SUBSPACE TEST — RUN 2026-07-04 (a model): the descent is NOT a forced bijection;
  K1 HARDENED (weld = irreducible adoption at the subspace level too); ONE new forced handle banked.**
  Pre-reg internal research notes (committed 3f23407 before probe); probe
  `proofs/foundations/LOOP_A5_winding_weld_W1_2026-07-04.py` (ALL-PASS; NO ε; verify 65/65). Tested the
  RIGHT object the A5 K1 measured too coarsely (single-eigenstate overlap 0.19/0.22): the SUBSPACE/deck
  descent from the coupled ℤ₆ (S=P₃⊗U_π) to the read's C₃. **Structure (banked):** S³=−I uniform (the ℤ₂
  double-cover sign is GLOBAL); S² = the C₃ DECK, labels {0,1,2} each 32-dim; present spectrum = odd ζ₆
  (descent bijective on present sectors; the vector windings shifted by the half-angle π/3). **The
  tension:** [W,S²]=0 (deck preserved) but [W,P₃]≠0 (the read's DART label is not a coupled good quantum
  number). **CORE finding:** the coupled DECK grading is modulus-UNIFORM — free deck sectors all = 2
  (Perron), interacting all = √2 (the O(1) dressing collapses 2→√2) — so it is NOT the read's dart
  (2,√2,√2) grading; the read's fine structure is a dart-P₃ quantity invisible to the deck. The read's
  Cl(6) companion is a CHOICE (the frame-vacuum lift; deck-weight peak 0.622, not concentrated). ⟹ the
  pre-registered PASS (forced bijection reproducing the read) is FALSIFIED; the weld is an irreducible
  adoption. **BANKED POSITIVE (new, sharper than A5):** the read channel's descent to the deck sectors is
  a FORCED COVARIANT superposition — weights **{1/3, 1/3 ± √3/6}** = {0.045, 0.333, 0.622}, one set
  cyclically permuted in t — NOT random spreading. So the read↔coupled relation HAS forced structure; what
  is unforced is (a) the Cl(6) companion (the lift) and (b) the grading identity (deck modulus-uniform,
  read not). **NET:** the −70 ppm and the composite dictionary stay pinned on the weld = an irreducible
  adoption; the ε-home does NOT localize through a forced descent (transport-minus-leading does not open).
  The one new handle for a FUTURE attack = the forced deck-superposition {1/3, 1/3±√3/6} (attack the
  companion/grading-identity through it). Do NOT re-run the deck-modulus/bijection test — it's settled NO.
- **THE WELD W2 CHIRAL-SEED TEST — RUN 2026-07-04 (a model): FOLLOWED THE {1/3,1/3±√3/6} LEAD TO ITS
  TERMINUS — the chiral (ε) SEED is FORCED; the route REOPENS; the −70 ppm is now SHARPLY localized to a
  single un-forced piece (a sub-ppm suppression).** Pre-reg internal research notes
  (committed 5c40ffb before probe); probe `proofs/foundations/LOOP_A5_winding_weld_W2_2026-07-04.py`
  (ALL-PASS; NO ε; verify 65/65). The {1/3,1/3±√3/6} superposition = the VACUUM's Cl(6)-deck content.
  **FORCED source (exact): ⟨0|U_π²|0⟩ = +i/2** (Re=2e-16, Im=0.5000). Its **REAL part = 0** ⟹ the bit-EVEN
  deck content is democratic {1/3,1/3,1/3} EXACTLY (WHY the read is chirality-blind); its **IMAGINARY part
  = ½** ⟹ the entire bit-ODD asymmetry ±√3/6 = |⟨0|U_π|0⟩|/√3. Free-vanishing (bit-average → democratic;
  carried by the iJ, not the I — the kinematic face of E2a's C = I+iJ). **UPGRADE:** the W1 "irreducible
  adoption" was confined to the bit-EVEN modulus grading, where ε does NOT live; the bit-ODD channel where
  ε DOES live (E2c) is FORCED. **So the weld's CHIRAL channel is forced; only the ε-irrelevant bit-even
  modulus identity is the adoption. HONEST TERMINUS: what is FORCED is the chiral SEED (⟨0|U_π²|0⟩=i/2,
  O(1)); what is NOT done is the ε NUMBER = a sub-ppm transport-minus-leading functional of this O(1) seed
  — and A5 already showed the naive transport OVER-APPLIES (bit-odd ~1e-3 rad, ~7e3×); the forced
  corrections are O(1)/α₁-level, both over-applying 1.75e-7. A ~α₁⁵ suppression of the forced seed is NOT
  in hand, and 2α₁⁵≈2e-7 is FLAGGED POISON (a coincidence, not a derivation).** ⟹ THE OPEN EQUATION,
  NEWLY SHARPEST: everything in the −70 ppm chiral channel is now FORCED except ONE piece — the sub-ppm
  suppression carrying the O(1) forced seed ⟨0|U_π²|0⟩=i/2 to ε = −1.7515e-7±3.9e-10 rad. Locate that
  suppression top-down (do NOT insert a power of α₁; do NOT pattern-match 2α₁⁵). The −70 ppm stays OPEN.
- **A5-DISCRETE CLOSED (Step 1 of the A5-unlock program) — RUN 2026-07-04 (a model): the imported lepton
  CHIRALITY assignment ν↔chir-7 / e↔chir-5/3 (`the_run.read_selection` L335 `⚠A5`) is now DERIVED.**
  Pre-reg internal research notes (committed 2cc385a before probe); probe
  `proofs/foundations/LOOP_A5_discrete_chirality_2026-07-04.py` (ALL-PASS; NO mass/ε; verify 65/65). Forced
  chain (reverse-excluding): (1) chir-7 (λ=−1) IB-root −½+i√7/2 IS the cover_B **√−7 enantiomer band-edge**
  (the J-bit band); chir-5/3=√−5, off it; both |h|²=2. (2) **J — where the forced seed ⟨0|U_π²|0⟩=i/2
  lives — is the A4 3-irrep**, and A4's only 3-dim irrep sits at the 3-fold-degenerate adjacency eigenvalue
  **λ=−1** (K₄ spectrum {3,−1,−1,−1}); so the seed lives in the chir-7 band (forced by rep theory). (3)
  **ν=n=0=the Fock vacuum (grade-even, weight 1.000) carries the seed** ⟹ ν→chir-7 FORCED + CHIRAL (seed
  flips +i/2→−i/2 with J) + reverse EXCLUDED (ν→chir-5/3 puts the even-grade seed's 3-irrep band on the
  odd-grade species e). (4) e-leg by COMPLEMENTARITY+REALITY (the other singlet → Perron sector; complex
  chiral root ⟹ √3), disclosed as the weaker leg. **CLOSES the discrete face of A5** (the framework's
  oldest conditional); the `⚠A5` import → a derivation. STAYS OPEN: the −70 ppm MAGNITUDE (Step 2, the
  sub-ppm seed suppression). the_run flag change is USER-gated. **Grade move: A5 chirality import
  "IMPORTED" → "DERIVED (ν from the seed, e by complementarity); magnitude open."**
- **A5-MAGNITUDE (Step 2, the −70 ppm) — ATTEMPTED 2026-07-05 (a model): WALL at ~11× over-application; the
  MECHANISM is confirmed (closest any route has reached), the state-block χ is ~3.4× too large.** Pre-reg
  internal research notes (committed eb079c3 before probe); probe
  `proofs/foundations/LOOP_A5_magnitude_2026-07-05.py` (ALL-PASS = checks ran; verdict WALL; verify 65/65;
  NO tuning). Construction (FROZEN): the forced W2 seed's bit-odd holonomy **χ = 1.161×10⁻³ rad** (the
  A5/E2c coupled shell-rate ½(χ(+J)−χ(−J))·s_lep) fed into `read_masses`, where it enters at **SECOND order
  via cos χ** (a forced, non-inserted suppression = read_masses' own structure). Result: **+795 ppm on
  m_e/m_τ vs observed ~70 ppm ⟹ factor ≈ 11.3× over.** **KEY PROGRESS:** the second-order cos χ mechanism
  reaches the RIGHT SCALE (hundreds of ppm from χ~10⁻³) — vs the naive first-order phase over-application
  (~7×10³×). From orders-of-magnitude to ~11×. Controls held (C-FREE Q3=2.5e-10; C-LEADING 4e-15). **NOT
  tuned:** the ×11 is reported not fixed; **2α₁⁵=1.809×10⁻⁷ ≈ |ε_target|=1.7515×10⁻⁷ is POISON — flagged,
  NOT invoked**; no α₁ power inserted. Sign: cos χ is sign-fixed (χ² even), moves m_e/m_τ UP; todo phrasing
  (read under-shoots) ⟹ direction plausibly CORRECT (magnitude wall, not sign exclusion; convention to pin).
  **SHARPENED OPEN (→ architect): the state-block χ (1.16×10⁻³) is ~3.4× (=√11.3) too LARGE (A5: state-block
  over-applies); compute χ via the PROPER transport (geometric/Berry, made robust), target ~3.4×10⁻⁴ rad.
  IF that ~3.4× suppression is FORCED (not inserted), the −70 ppm lands.** The −70 ppm stays OPEN.
- **PROPER (spinor Berry) TRANSPORT — ATTEMPTED 2026-07-05 (a model, post-A5(b)): WALL — the pure Berry
  OVERSHOOTS; the answer is now BRACKETED (11th documented route on the hard core).** Pre-reg
  internal research notes (df0108d, BEFORE the probe); probe
  `proofs/foundations/LOOP_A5_magnitude_proper_transport_2026-07-05.py` (ALL-PASS; verdict WALL). Using
  the A5(b)-derived SPINOR object + the robust FHS gauge-invariant Berry method (C-GAUGE dev 1e-12; the
  fragility A5 couldn't beat is BEATEN), the eigenVECTOR Berry connection of the SAME coupled resolvent
  (C-CONSISTENCY: eigenVALUE-rate reproduces χ_state=1.16e-3 exactly) gives **χ_proper = 5.5×10⁻⁵ =
  χ_state/21** → **+1.8 ppm (×39 too SMALL; direction UP correct).** So the pre-registered "Berry lands
  √11 smaller" is REFUTED (it overshoots to 21× down). **BRACKET (the deliverable):** needed χ~3.4×10⁻⁴
  ∈ (eigenVECTOR-Berry 5.5×10⁻⁵ [geometric only, ×39 small], eigenVALUE-rate 1.16×10⁻³ [dynamical, ×11
  big]). The physical χ is a specific INTERMEDIATE of the same operator — NOT the geometric mean (2.5×10⁻⁴,
  1.35× off), NOT to be interpolated (tuning). POISON flagged: shift ≈ 20α₁⁵ (mantissa coincidence, NOT
  invoked). Open equation now: "which FORCED object gives the intermediate χ" — derived, not tuned. −70 ppm OPEN.
- **RELATIVE-to-PERRON BERRY — ATTEMPTED 2026-07-06 (a model): NULL → the TRANSPORT FAMILY is EXHAUSTED
  (12th route; 3rd/last transport variant, pre-declared).** Pre-reg `A5_magnitude_relative_berry_prereg_2026-07-05.md`
  (0d52a9f, BEFORE probe); probe `LOOP_A5_magnitude_relative_berry_2026-07-05.py` (ALL-PASS). read_masses'
  χ = ½(arg a₁+arg a₂) − arg a₀ is relative to the Perron c₀; the forced correction (subtract the Perron's
  run-Berry) is **χ_Perron = 0 EXACTLY** — `perron_frame(+J)==perron_frame(−J)` to machine precision ⟹ the
  **Perron (ω⁰, democratic) carries ZERO chiral holonomy** (master lens confirmed; the chiral χ lives entirely
  in the shell). So χ_rel = χ_shell, same ×39 wall. **ALL THREE natural transport objects now documented, none
  lands:** eigenVALUE-rate (dynamical) ×11 big / eigenVECTOR-Berry shell (geometric) ×39 small / rel-to-Perron
  (NULL). **NO further transport variants (pre-declared).** The −70 ppm's χ is NOT a simple diagonal transport
  of the resolvent modes; the next forced principle must differ in KIND — leading candidate = the OFF-DIAGONAL
  non-adiabatic coupling ⟨sub|dB/ds|dom⟩ (this §1's own localization, the mode-MIXING, distinct from the
  diagonal Berry phase) — a FRESH arc. −70 ppm OPEN.
- **OFF-DIAGONAL / NON-ABELIAN GEOMETRIC — ATTEMPTED 2026-07-06 (a model): WALL → GEOMETRIC FAMILY EXHAUSTED;
  KEY NEW FINDING = the diagonal Berry is the RIGHT α₁³ scale (13th route).** Pre-reg
  `A5_magnitude_nonabelian_offdiag_prereg_2026-07-06.md` (7fce515); probe
  `LOOP_A5_magnitude_nonabelian_offdiag_2026-07-06.py` (ALL-PASS). **Scale reconciliation (durable):** the
  DIAGONAL geometric Berry (the read_masses U(1) χ) = 5.54e-5 = **0.93 α₁³** — the FIRST RIGHT-SCALE result
  (localizes 06-30's "α₁³ Dyson" to the geometric sector) — but ×39 short (6.2× in χ); the eigenVALUE-rate
  is α₁² (wrong scale, ×11 over). The OFF-DIAGONAL mode-mixing ⟨sub|dB/ds|dom⟩ (the 2.21/1.43 class) =
  **O(α₁)** — one power LARGER, over-applies ×1e4 as a χ; a DIFFERENT (SU/off-diagonal) sector, EITHER
  orthogonal to the diagonal χ (×39 short stays) OR over-applies (×1e4). NEITHER supplies the forced ~6.2×
  at α₁³. **⟹ the −70 ppm's α₁³ coefficient is NOT any geometric-phase object** (diagonal α₁³ ×39-short,
  off-diagonal α₁ ×1e4-over, no forced intermediate). GEOMETRIC FAMILY EXHAUSTED (adds to the transport
  family). Remaining lift = the continuum-D₄ Dirac-cone spectral action (the 3rd-girth-winding
  α₁³=(2/3)²⁴ survival Dyson diagram; A5(b)-enabled) — a DIFFERENT forced principle, NOT a mode phase.
  POISON held (2α₁³/2α₁⁵ NOT invoked). −70 ppm OPEN.
- **D₄ SPECTRAL-ACTION S3 RAN 2026-07-06 (`D4_S3_alpha1cubed_isotype_2026-07-06.py`, pre-reg f441caf; §6 +
  §7 CORRECTION in internal research notes) — WALL/non-closure; the crux
  RE-POINTED, then self-corrected same-day.** The continuum-D₄ a₄ machine is now BUILT (S1) — but the a₄ is
  the GAUGE sector; the −70 ppm lives in the MASS sector. S3 confirmed the amplitude/shell route is
  generation-viable (a σ-isotype-BLIND additive c₀ shift STILL moves m_e/m_τ via the C₃-Fourier shell
  cos(δ+2πj/3) — so isotype-blindness is NOT the obstruction; the refuted 1/μ_rep is not needed). **BUT
  reconciling with Q3 (STATION 3 below): the experimentally-PINNED hard direction is m_τ-FREE
  (δ(m_e/m_μ)=+9.83 ppm, 452σ) and is a chiral PHASE (ε=δ_eff−2/9), NOT an additive amplitude** — the
  additive/a₂-additive route moves only the SOFT (m_τ ±67.5 ppm noise ≥ budget) direction. So the hard crux
  is UNCHANGED = the chiral phase ε. Its (S1) ∂_N-phase candidate = this session's EXHAUSTED χ; the ONE
  un-exhausted candidate is **(S2) the mass-dependent δ-dressing g(m_j)** (chiral through the leading δ) —
  the sharpened S3b target. −70 ppm OPEN; poison untouched; no value moved.
- **O0 + O1 EXECUTED 2026-07-06 LATEST (a model; `ODD_O0_graded_blindness_theorem_2026-07-06.py`, ALL PASS;
  theorem `docs/theorems/theorem_graded_blindness_and_odd_channel_2026-07-06.md`; CONSOLIDATION — no ε
  computed, NO value moved).** The GRADED-BLINDNESS THEOREM is proved on the srs objects: D₄ = D₃⊗1 +
  γ_t⊗∂_N with {D₃,γ_t}=0 (D₃ = Hodge-Dirac, γ_t = form grading — verified exact) ⟹ D₄² = D₃²+∂_N² (clean
  split, no cross term, holds for EVERY ∂_N) ⟹ the bit σ (A↦A, B↦−B) leaves D₄² EXACTLY invariant ⟹ **every
  even functional (spectrum/moduli/a_k/ζ(0)/resolvent/Berry-of-D²) is chirality-BLIND by the object's own
  split**, and the chiral bit lives ONLY in the σ-ODD carrier Tr((γ_t⊗∂_N)g(D₄²)) — REAL, generically
  nonzero (channel LIVE). **The four scattered walls are ONE theorem's corollaries** (C1 Q3-conjugation =
  the EVEN clause; C2 E2c bit-parity = the even/odd split of Tr(D₄ g); C3 W2 seed Re=0/Im=½ = the theorem
  on one matrix element; C4 Perron-null = the lemma "σ-odd operator ⟹ zero expectation in a σ-invariant
  sector", proved in-probe). ⟹ the ~15 exhausted routes were all even or D²-eigenstate functionals = blind
  BY THE SPLIT, not by accident. **O1 classification (proved):** {R1 even DEAD / R2 quantized-odd DEAD /
  R3 state-projected-odd weld-gated / **R4 continuous-odd spectral TRACE (η / spectral flow) UN-PROBED**} —
  R4 is the UNIQUE remaining class (σ-odd, continuous, projection-free ⟹ no winding-weld descent). Inventory
  re-verified: no repo probe computes a continuous odd invariant (19 probes compute QUANTIZED odd objects
  Str=χ=−2/Chern/−π; η named as un-run sequel at `b4_a4_dirac_index_probe.py:344`). NEXT = O2 (relative-η
  probe, own pre-reg BEFORE the probe, blind). −70 ppm OPEN.
- **O2 EXECUTED 2026-07-06 LATEST (a model; `ODD_O2_relative_eta_2026-07-06.py` ALL PASS = controls; pre-reg
  internal research notes, committed f2cd79b BEFORE the probe; blind;
  NO value moved) — VERDICT: KILL-Q / CONTINUUM (the pre-declared most-probable outcome).** Built the σ-odd
  spectral TRACE (R4) of the run family on the A5 coupled machinery (same S-0 as `LOOP_A5_magnitude`; the
  ONLY change = the top-2-eigenvalue PROJECTION → the all-modes TRACE `d/ds arg det C_shell`). Findings,
  all robust: (1) the parameter-free lattice invariant is quasi-QUANTIZED — the spectral-flow winding over
  [0,s_lep] = **−π/2** (clean quarter, 400-pt unwrap); (2) the continuous RATE is numerically ILL-CONDITIONED
  on the lattice — the DS-scan gives −2.19e-7 → −7.28e-8 → **+1.36e-5** (200× spread, SIGN-FLIP) = the same
  non-normal exceptional-point pathology the repo flagged for M_Z; (3) the odd heat trace has NO forced scale
  (t-drift, UV cutoff). ⟹ the continuous σ-odd invariant carrying ε is NOT robustly lattice-accessible — it
  is a **CONTINUUM** object (the odd Seeley–DeWitt / 3D-eta density on the A5(b) cone), exactly parallel to
  S1's even-a₄ ("lattice heat-kernel DEAD END, D₃ bounded"). This explains uniformly why ALL ~15 prior LATTICE
  routes walled and LOCATES Gate A's odd face as a well-posed continuum build. **CAUGHT MIRAGE (poison
  discipline worked):** at the single DS=1e-6 value the trace read −7.28e-8 = 0.4156·ε (same sign/order,
  0.4156 ≈ **5/12** to 0.25%) — looked like the closest-ever bracket; the pre-registered DS robustness scan
  KILLED it as a finite-difference artifact (200× unstable, sign-flipping). Recorded, NOT adopted; NO poison
  invoked. Controls PASS (C-FREE 1.1e-8; C-LEADING 4e-15; bit-odd removes the bit-even leading). **NEXT = O3
  (a fresh CONTINUUM build, NOT a lattice probe): the odd sector of the D₄ spectral action on the A5(b)
  Fock-Dirac cone (`d4_spectral_action.py`), own pre-reg, blind vs ε.** Do NOT re-run the lattice trace-flow
  (ill-conditioned — settled). −70 ppm OPEN.
- **O3 EXECUTED 2026-07-06 LATEST (a model; `ODD_O3_continuum_odd_action_2026-07-06.py` ALL PASS; pre-reg
  internal research notes, committed ccc72e4 BEFORE the probe; blind;
  NO value moved) — VERDICT: KILL-ANOMALY/SUB-LEADING → UNIFICATION.** Built the FORCED continuum 4D cone
  Dirac on the A5(b) cone: g5=−iγ¹γ²γ³ has g5²=1, **[g5,gD]=0** (labels the two 3D parity-irreps), so the
  4th anticommuting gamma γ⁰ is the unique other Hermitian involution in the gD-anticommutant and Γ⁵=i·g5·γ⁰;
  D₄=k·gD+k_t γ⁰ has {Γ⁵,D₄}=0 ⟹ **massless eta=0 exactly** (T0). **EXACT SELECTION RULE (T1):** η(D₄+X)≠0
  ONLY for a Γ⁵-EVEN X — BLIND to every Γ⁵-odd background (vector shift η=0; chiral mass m·Γ⁵ η=0), LIVE only
  for the Γ⁵-EVEN scalar mass m·I (η≠0, UV/grid-growing = the topological parity anomaly). **(T2)** a
  flat/static run connection is Γ⁵-ODD ⟹ η=0 for EVERY holonomy (C-FREE, θ=0.1/2·9/1.0) ⟹ ε is NOT any
  static-cone read and NOT the leading anomaly. The only source of a Γ⁵-EVEN coupling = dynamical mass
  generation = the run's CURVATURE = **the INTERACTING run**. **⟹ THE UNIFICATION (the deliverable): the
  odd-channel arc (O0–O3) and the standing interacting-run frontier (C3/E2a–E2c, the loop program's R-ε
  number-mover) are the SAME un-built object — the cone Dirac coupled to the interacting run, whose
  sub-leading Γ⁵-even eta density IS ε.** Two independently-pursued frontiers converge on ONE gate; this is
  why every static/free/lattice route was blind (Γ⁵-odd or even-functional, by the selection rule + O0's
  graded-blindness theorem). NO value computed (no forced magnitude on the static cone; none invented); no
  poison invoked; controls PASS; analytic (no O2 ill-conditioning). **NEXT = O4 = build the INTERACTING-RUN
  connection on the cone (the un-built C0–C3/E2a object) and read its odd Γ⁵-even eta density — doubly
  motivated (closes BOTH the odd channel AND R-ε).** Do NOT re-run static-cone/lattice odd reads. −70 ppm OPEN.
- **O4 EXECUTED 2026-07-07 (a model; `ODD_O4_interacting_run_cone_2026-07-07.py` ALL PASS; pre-reg
  internal research notes, committed 8cf0abc BEFORE the probe; blind;
  NO value moved) — VERDICT: KILL-WELD (the odd-channel arc's TERMINUS).** Coupled E2a's forced interacting
  run G_int to the A5(b) cone. **FORCED (S1):** the chiral asymmetry A(u)=Tr(Q₁G_int)−conj Tr(Q₂G_int)
  scales as **u^2.09** (matches E2a's u²-grading) ⟹ **A(α₁)=−8.82e-4** (bit-odd; C-FREE 3e-16, C-BIT to
  9e-16) — a clean forced number, but O(α₁²), ×5000 ABOVE ε (O(α₁⁵)): ε is a deep sub-leading residue of A.
  **DECISIVE (S2, C-WELD):** the cone's forced Weyl frame overlaps E2a's canonical Fock vacuum by only
  **0.197**, and A changes **60%** under the admissible frame swap ⟹ the generation-resolved chiral phase is
  **LIFT-DEPENDENT** ⟹ **ε's generation resolution stays GATED on ADOPTED-WINDING-WELD** (4th independent
  angle, after EP-2/N1, N1b, W1). **⟹ THE CONSOLIDATION: the odd-channel arc (O0 theorem → O1 → O2 continuum
  → O3 unification → O4 terminus) ends at the SAME identification-layer adoption that gates the bound-state
  continent B1. The −70 ppm and the B1 nucleon sector SHARE ONE GATE: the winding-weld/species-lift map.**
  ε is NOT a missing computation — its last gate is a NAMED adoption (confirms the standing identification-
  layer lesson at the deepest level). NO poison invoked (A at α₁², one power above 2α₁³; ε's 2α₁⁵ untouched).
  **The odd-channel arc is COMPLETE. Highest-leverage open object now = the winding-weld/species-lift adoption
  ITSELF (gates BOTH −70 ppm AND B1); the one un-exhausted lead = W1's forced deck-superposition
  {1/3,1/3±√3/6}.** Do NOT re-run odd-channel lattice/cone reads (O0–O4 settle them). −70 ppm OPEN.
- **THE ODD-CHANNEL REFRAME — SYNTHESIZED 2026-07-06 LATE (architect; STRUCTURAL PROPOSAL ONLY, nothing run,
  no number computed, no value moved).** The route-exhaustion history has ONE common structure: the object
  is a graded sum (D₄ = D₃⊗1 + γ_t⊗∂_N, clean split D₄² = D₃²+∂_N², `the_run.py:199-214`) ⟹ **every even
  functional is blind to the chiral seam by the split itself** — the Q3 conjugation theorem, the E2c
  bit-parity theorem, W2's Re⟨U_π²⟩=0 democracy, and the Perron-null relative Berry are four proved
  instances of one graded-blindness fact; the "chirality lives in the iJ, never the spectrum" lens is its
  slogan. The DN_CHIRAL_A classification (R1 even / R2 quantized-topological / R3 dynamical state-projected)
  has a GAP: the **continuous odd spectral invariant** (η-invariant / spectral flow of the run family /
  odd heat trace Tr(γ_t D e^{−tD²})) — never computed in this repo (only the QUANTIZED odd objects exist:
  Str e^{−tD₃²}=χ=−2, Chern ∓2, Berry −π; `b4_a4_dirac_index_probe.py:344` even NAMES η as its un-run
  sequel, May 2026). It is odd (escapes R1), continuous (escapes R2), and a projection-free TRACE (needs
  no winding-weld descent — the weld gates state-projections, not traces); its variation carries BOTH a
  local eigenvalue-velocity term AND crossing/spectral-flow content, i.e. it is a forced object of exactly
  the "specific intermediate between the ×11-over dynamical and ×39-short geometric endpoints" KIND the
  bracket demands (⚠ plausibility-of-kind, NOT a magnitude claim). Also: the 06-30 non-abelian holonomy
  collapse (det W→0) sat exactly at the Perron→shell crossings — where spectral flow concentrates.
  **Draft program (a model commits its own pre-reg): O0 = prove the graded-blindness theorem once (subsumes
  the four corollaries — pure consolidation); O1 = the classification-completion theorem-let; O2 = the
  relative-η probe** (η(full run family) − η(constant-φ leading family) on the generation sector; controls
  C-FREE/C-BIT/C-EVEN/C-LEADING; poisons standing: 2α₁⁵, 2α₁³, no inserted power, no bracket
  interpolation; blind vs ε = −1.7515e-7 ± 3.9e-10). A KILL (odd trace quantized-or-zero) is decisive:
  it closes the whole continuous-odd class and leaves only the interacting-run build + B2. Full spec +
  tributary integration map: internal research notes.
  The −70 ppm stays OPEN.

### 2. The Higgs/scale sector vs the electron mass — SEARCHED 2026-06-30, scale route RULED OUT
Hypothesis (user): a small Higgs-side correction, shared via N_hub→v, brings in the electron mass and makes
N_hub consistent. **Searched (v_higgs.py, m_H.py, lambda_higgs.py, m_e.py); result: DISFAVORED for the dominant
−70 ppm, with numbers.**
- Higgs sector is fully dark-corrected and **closes**: m_H = √(2λ)·v, λ carries its own Family-D (−4α₁², from 4
  Higgs legs) → m_H **−0.05σ**. The one asymmetry: the v_higgs Family-D (−α₁²) is **absorbed into N_hub** (not
  applied) via the G_F round-trip — but that touches the scale/cosmology.
- **The framework's own `m_e.py` decomposes the residual:** −70 ppm **Koide RATIO** (scale-independent, ~84%) +
  −13 ppm **m_τ absolute scale** (Higgs/N_hub-touchable). **VERIFIED:** the ratio (f_min/f_max)² is *invariant*
  under a +1000 ppm v/N_hub shift ⟹ a Higgs/v/N_hub correction **CANCELS in it and cannot supply the −70 ppm**.
- ⟹ the −70 ppm is the per-rep **δ-gap** (δ for m_e/m_τ-exact = 0.2222208 vs 2/9), scale-independent,
  lepton-Yukawa-side — confirming item 1. The Higgs/scale route is **ruled out** alongside the joint-object route.
- The ONLY Higgs/scale-touchable part is the **−13 ppm m_τ absolute scale** (16%), the v←N_hub←G_F circular
  calibration (Gap G1) — small, separate from the −70 ppm, and mostly y_τ.
- **Doc-lag corrected:** the memory line "m_e 70 ppm = the N_hub over-determination residual" was WRONG (it's the
  scale-independent ratio; the over-determination is the −13 ppm). Fixed in
  `memory/N_hub-overconstrained-higgs-vs-electron-2026-06-29`.

### 3. N_hub calibration omits a derived correction → H_0 is flattered (over-determination, 2026-06-30)
The N_hub value is pinned by the G_F round-trip using **only the Class-C (5/12) dark correction on v**.
`N_hub.py` (lines 120-129) explicitly instructs: *"if a higher-order Feshbach analog on v is later derived
(above and beyond 5/12), N_hub should be recomputed."* The **Family-D (−α₁²) on v IS that analog** — derived
theorem-grade in `v_higgs.py` — and N_hub was **never recomputed** with it (it is "absorbed").
- **Consistent treatment (apply the derived Family-D):** N_hub −0.61% (×(1−α₁²)^V) ⟹ H_0 ∝ 1/N_hub shifts +0.61%.
  - CMB/substrate side: **68.18 (+1.56σ) → 68.60 (+2.39σ)** vs Planck 67.4 — **WORSE.**
  - observer side: 72.72 (−0.30σ) → 73.17 (+0.12σ) vs SH0ES 73.04 — slightly better; but Planck dominates.
- **Finding (AM):** thought the reported H_0 +1.56σ was flattered by omitting a derived Family-D (consistent
  value +2.39σ). **CORRECTED PM by the full audit (an internal working note):** the Family-D
  **does not belong on v** (condensate ≠ legged scattering vertex; not actually applied; its "absorption check"
  is a by-design tautology). ⟹ omitting it is **correct**, **H_0 = +1.56σ stands** (the +2.39σ assumed a
  correction that shouldn't be applied). The real error is the *opposite*: `v_higgs.py`/`N_hub.py` **over-claim a
  v Family-D that shouldn't exist.** (Genuine dispute: the framework asserts a v "1H+0F vertex"; category
  grounds say no → +1.56σ.)
- **BIGGER finding (the audit's bottom line): v/N_hub is an ASSEMBLED, CALIBRATED form — NOT a forced top-down
  read.** v = v_obs to 0σ **by construction** (N_hub inverted from G_F ≡ from v_obs). Effectively ONE free input
  (N_hub, Gap-G1, band-B) + M_P unit + ~4–5 non-forced modeling choices. The "UNIQUE/THEOREM-GRADE/forced"
  labels overclaim; v's 0σ is a calibration artifact, not a prediction. Defensible core: 1/√2 overlap, −1/4
  finite-size exponent, 5/12 count, one adopted N_hub. **Corrections owed are listed in the honest-grade doc.**
- Does NOT touch the −70 ppm electron ratio (scale-independent, item 2).

### 4. The dark-correction SIGN — DOWN is derived (rate framing); the standalone formal lemma is CLOSED as characterization+impossibility (D3, 2026-07-04)
**Status: sign is DERIVED DOWN (foundation-conditional); the CAS-closeable lemma is the open equation.** The dark
self-energy Σ=α₁/h is magnitude-forced (Re/−Im to 1e-12) and the sign is **DOWN** via the framework's foundational,
user-confirmed **mass = dynamical recurrence RATE** definition: the rate/velocity reading (cycle-takers waste steps
→ delayed) gives mass×(1−α₁/h) = DOWN, reproducing the framework's value (`theorem_dark_self_energy_unified §3`).
- **Already forced:** the vertex-dark sign (y_τ, c_F) is rigorously DOWN (Peskin −1, a separate closed loop, §2.5).
- **The open equation:** a *standalone* CAS-checkable lemma that "mass=recurrence-rate ⇒ reading (2) [rate→DOWN]
  over (1) [amplitude→no-change] and (3) [return-amplitude→UP]" does NOT yet close (`§3 correction 2026-06-29`).
  The mass=rate *definition* selects DOWN; a from-nothing formal derivation of that selection is un-built.
- ⚠ The sign itself is NOT undetermined — it is DOWN. Do not relabel this as "empirical/open sign"; what is owed
  is only the formalization of a settled result.

### 5. The gauge-β FORMULA (ζ_{D₄}(0)) — GRADE AXIS COMPLETE 2026-07-06; residuals NARROWED (not fully closed)
**Status (2026-07-06): the β-FORMULA GRADE AXIS is COMPLETE — the SM-physics import is REMOVED — but β is
NOT fully closed end-to-end; the residuals are NARROWED (below).** The β VALUES {33/5, 1, −3} are derived
(`read_gauge_running`; the hardcoded "target" removed 2026-07-01; +4 completion derived, not injected). The
FORMULA is now **{the derived A5(b) cone's own heat-kernel a₄ (S1) × native (2s_z)² spin content, all 3
fields (S2) × native group factors (SU(3)_c, SU(2)_L)}**, with only the pure-math **Gilkey a₄ theorem**
imported. **NARROWED RESIDUALS — what "fully closed end-to-end" still needs** (all GRADE, gate no value):
  (i) the **Gilkey a₄ THEOREM itself** — that the heat trace HAS (1/12)trΩ²+(1/2)trE² — is a pure-math
      import (like Ihara–Bass), not re-derived from the object;
  (ii) the **VECTOR/SCALAR rows (−11/3, +1/3) on a NATIVE cone** — currently the universal helicity rule
       (pure-math Gilkey), NOT a native-cone re-derivation the way the fermion +2/3 was (S1);
  (iii) **U(1)_Y's "which-U(1)" D_F selection** — a stated framework_axioms adoption, not derived;
  (iv) the **+4 completion's named framework-class steps** — the walk↔Fock dictionary (A5-class), the
       |s−½| minimal-spin selection (A2-class), and the KO 2→6 form-parity↔statistics identification.
Detailed derivation trail (S1/S2, the group factors, the completion) below.
- **GROUP FACTORS de-imported (2026-07-05/06) — the Dynkin/Casimir INPUTS to the β sums are now NATIVE, not
  table lookups:** SU(3)_c T(3)=½, C₂(adj)=3 as traces over the Cl(6)-Fock so(6)-bivectors
  (`NATIVE_a4_color_su3_2026-07-05`, probe 1); SU(2)_L T(2)=½, C₂(adj)=2 as traces over the T-ID2 commutant
  su(2) (`NATIVE_a4_su2L_2026-07-06`, probe 3); U(1)_Y native (Y off the Hamming weight; norm 3/5 ↔ native
  sin²θ_W=3/8) **modulo ONE stated adoption** — the C₃-breaking "which-U(1)" D_F selection (framework_axioms),
  NOT a de-import. So the β VALUES are native down to {native group factors} + {the declared Seeley–DeWitt
  SPIN dictionary}; `gauge_dynkin`'s T3/T2 tables are now the working lookup for values that fall out as traces.
  This is ORTHOGONAL to the open β-FORMULA axis below (the spin rows / ζ_{D₄}(0)).
- **ζ_{D₄}(0) β-FORMULA axis — D₄ SPECTRAL-ACTION S1 DONE 2026-07-06 (`D4_S1_native_a4_machine_2026-07-06.py`,
  ALL PASS; pre-reg e55c7c1; program `D4_spectral_action_program_kickoff_2026-07-06.md`).** The native
  continuum-a₄ machine is BUILT on the A5(b) Fock-Dirac cone (the continuum unlock the lattice D₃ lacked):
  H²=|k|² (continuum a₀); **E=−2F·S computed NATIVELY from the A5(b) γ commutators**; the orbital
  (1/12)trΩ²=−B²/6 COMPUTED (the Bt/sinh(Bt) Landau trace on the cone); the **FERMION row +2/3 native**
  on the cone; b_i={33/5,1,−3} assemble from the object's own a₄ × the native group factors. **UPGRADE:
  the SM-physics flavor ("one-loop QFT β formula") is REMOVED — ζ_{D₄}(0) = {the DERIVED cone's own
  heat-kernel a₄} × {native group factors}, with only the pure-math GILKEY a₄ THEOREM imported (same
  status as Ihara–Bass).** The a₄ machine is REUSABLE infrastructure (`derivation_topdown/bridge/d4_spectral_action.py`,
  importable, self-test ALL PASS) for S3 (the α₁³/−70 ppm, trap-dense, LATER) and S4 (the CAR-KMS loop).
  **S2 DONE 2026-07-06 (`D4_S2_native_spin_rows_2026-07-06.py`, ALL PASS; pre-reg 0b4a5d4): the a₄ SPIN
  CONTENT is now NATIVE for ALL THREE fields** — fermion (2s_z)²=1 (S1); VECTOR (2s_z)²=4 (the emergent
  band spin-1 rep S_a, Casimir 2, via (1/2)trE² over the transverse s_z=±1 pair); SCALAR 0 (the Higgs).
  The helicity rule → {+2/3,−11/3,+1/3}; b_i={33/5,1,−3}. **⟹ the β-FORMULA GRADE AXIS is COMPLETE:
  ζ_{D₄}(0) = {derived cone's a₄} × {native (2s_z)² spin content} × {native group factors}, only pure-math
  Gilkey imported.** RESIDUAL (named, = statistics/completion, NOT the spin rows): the +4 completion's
  KO 2→6 form-parity↔statistics + the flat/Higgs time-leg shadow (DN_C1). No value moved; the −70 ppm STAYS OPEN.
- **[SUPERSEDED 2026-07-06 by S1/S2 above — kept as the 2026-07-01 trail] The (then-)open equation:** the
  one-loop β FORMULA (the −11/3, ⅔, ⅓ Dynkin structure) was still standard-QFT typed (Layer-2). Its native
  top-down form is **ζ_{D₄}(0)** — the spectral zeta of D₄ = B⊗∂_N (the 4D Dirac-cone completion, KO-dim 2→6).
  **NOW BUILT: the continuum Dirac-cone a₄ machine (S1) + native spin content (S2)** replaced this;
  the lattice heat-kernel dead-end was confirmed. Residual = the header's (i)–(iv). GATES NO VALUE.
- **Within-repo caution to resolve:** `O_native_beta_eliminate_mssm_adoption:32` argues only the +2 scalar-half has
  a clean substrate home, so "the full +4 is FORCED (not merely reproduces the values)" is not yet unanimous —
  closing ζ_{D₄}(0) is what settles whether the completion is forced.
- **STATION 2 EXECUTED 2026-07-02 (Ω session 2, `proofs/foundations/OMEGA_S2_Q2_internal_a4_gauge_row_2026-07-02.py`,
  ALL PASS) — Q2 ANSWERED (computed, not inherited); THE β-FORMULA LAYER CLOSES; the gauge row localizes to the
  time-leg complex.** (1) **The (−11/3, 2/3, 1/3) row structure is DERIVED** from the heat kernel's two universal
  Seeley–DeWitt coefficients a₄ ⊃ (1/12)trΩ² + (1/2)trE² with the magnetic-moment endomorphism E = −2F·S —
  validated on exact spectra (torus θ-normalization Poisson-exact; Landau trace tB/sinh(tB): the t² coefficient
  IS (1/12)trΩ² = −B²/6); per helicity pair b = −(−1)^{2s}[(2s_z)² − 1/3]: {+1/3 complex scalar, +2/3 Weyl,
  −11/3 vector+ghost} with ONE unit normalization and TWO forced outcomes; component-level ghost bookkeeping
  agrees (+2/3 − 4 − 1/3 = −11/3); b_2HDM re-assembles {21/5, −3, −7} with no per-row tuning (matter-row
  regression: the Weyl row IS the 06-25 cone result's content). **Seeley–DeWitt replaces "one-loop QFT" as the
  declared Type-3 import — the β FORMULA now lives in the same spectral-action layer as ζ_{D₄}(0); the Layer-2
  tag upgrades accordingly** (wording user-gated). (2) **The graded theorem (exact):** opposite-statistics
  pairing cancels the orbital −1/3's pairwise; only paramagnetic content survives: vector pair = −3, chiral/
  Higgs pair = +T ⟹ b_graded = −3C₂ + T_f + T_H, and the completion's add ≡ the shadow rows exactly
  ((1/3)T_f sfermion + (2/3)T_H higgsino + (2/3)C₂ gaugino). (3) **The object-side pairing (the Q2 decision):
  D₃ IS the supercharge** — every nonzero mode's even/odd components are isospectral D₃²-pairs with trivially
  commuting internal charges ⟹ the multiplet reading is DERIVED for all massive/cone content, conditional on
  ONE named identification (form-parity ↔ statistics = the KO 2→6 step). **The FLATS are D₃-UNPAIRED** (parity-
  definite zero modes — the same fact as the index/β separation) ⟹ the spatial complex supplies NO shadow for
  the gauge sector ⟹ **the remaining open equation, sharpened: build the TIME-LEG (γ_t∂_N) fluctuation complex
  for the flat/Higgs sector; its graded a₄ must supply (2/3)C₂ + (2/3)T_H.** β values unchanged; nothing shipped.
- **C0+C1 EXECUTED 2026-07-02 (∂_N construction program; `DN_C0_run_measure_2026-07-02.py` +
  `DN_C1_timeleg_graded_a4_2026-07-02.py`, both ALL PASS; hypothesis pre-registered in C0's committed probe
  BEFORE C1 ran) — THE TIME-LEG COMPLEX IS BUILT; the shadow rows are DERIVED-CONDITIONAL.** C0: the run
  direction's fluctuation measure is FORCED = the object's own loop ensemble (free energy ln ζ(u) =
  −Tr ln(I−uB), Ihara–Bass verified per fiber; propagator = the Q1 fugacity-phase resolvent; subcritical at
  α₁; occupations Bose-form with entropic energies, COMPLEX on the shell ⟹ signed/interference on modes,
  positive on paths); the matter sector's CAR-KMS(β=1) is independently forced; **the Bass exponent
  |E|−|V| = b₁−1 = 2 = the flat count — the gauge sector's fluctuation determinant is the (1−u²)² prefactor.**
  C1: **the graded pairing is the TICK-LATTICE MATSUBARA DOUBLING** — (1−u²e^{2iω})^{b₁−1} =
  (1−ue^{iω})^{b₁−1}(1−ue^{i(ω+π)})^{b₁−1} exactly (periodic + antiperiodic sectors per mode, identical
  internal content); the antiperiodic sector is FERMIONIC by A4/CAR through the walk↔Fock dictionary (tick
  parity = Fock parity; the even sector's u²-quanta = fermion bilinears; parity period = p_toggle = 2). ONE
  rule (antiperiodic partner, statistics flipped, spin |s−½| by A2-minimal selection, station-2 row
  dictionary) reproduces **all three completion rows (sfermion 1/3·T_f, higgsino 2/3·T_H, gaugino 2/3·C₂)
  with the matter row as the no-tuning control**; per group add = {12/5, 4, 4}, b_2HDM + add = {33/5, 1, −3}.
  **ζ_{D₄}(0) status now: β FORMULA derived (Seeley–DeWitt, station 2) + completion CONTENT derived-conditional
  (here). The remaining research edge, stated exactly: theorem-grade the two named framework-class steps —
  (i) the walk↔Fock dictionary (A5-class), (ii) the |s−½| minimal-content selection (A2-class).** Shadows are
  loop content, NOT sparticles (standing note). No value moved.
- **SHARPENED 2026-07-02 (Ω session 1, `proofs/foundations/OMEGA_T1_zeta_D4_gauge_row_2026-07-02.py`, ALL PASS;
  post-Q0 — see §7 for the Q0 correction):** mechanism 1 (the D₄ heat kernel) is VALIDATED: the factorization
  Tr e^{−tD₄²} = (4πt)^{−1/2}·Tr_band is exact; the band trace's cone sector obeys the Albanese dictionary
  (A·t^{−3/2}, v = 1/2, V_alb = 4, verified on a 40³ grid); and **the index/β separation is a per-fiber
  IDENTITY**: Str e^{−tD₃²}(k) = χ(K₄) = −2 for all k, t (γ_t = (−1)^F, D₃ = the supercharge; the H¹ flats are
  the index density — "flat band → index not β" is now exact). **What the completion IS:** b_4d ≡ −3C₂(G) +
  T_f + T_H (the N=1 index/holomorphic form), exactly, all three groups — the "+4 shadows" are the D₄
  complex's own grading partners, and what ζ_{D₄}(0) must produce is −3C₂ + ΣT, not the raw −11/3 list.
  **Localization (the open equation, sharpened):** the band sector CANNOT carry the gauge row — the band-side
  gauge fields (H¹ flats) generate the deck U(1)³, abelian by construction (C₂ ≡ 0); the non-abelian charges
  live in the Cl(6)-Fock internal space ⟹ **the gauge row = the a₄ of the internal (D_F) fluctuation sector
  against the D₄ cone, un-built.** Grade frontier still open; no value gated.
- **D1 PIECE-1 NATIVE-a₄ ARC — PROBES 1+2 EXECUTED 2026-07-05 (a model; arc+prereg
  internal research notes, committed before each probe).** The
  ζ_{D₄}(0) gauge-row de-import now PARTITIONS cleanly. **CRUX resolved (from the live `the_run.py`):
  color SU(3)_c lives in the Cl(6)-Fock sector (Λ•(ℂ³)=1⊕3⊕3̄⊕1, quark=Λ¹), NOT Object A's ℂ[A₄]
  M₃ (a FAMILY su(3); `explore_m07` conflates them).** (i) **PROBE 1 (`NATIVE_a4_color_su3_2026-07-05.py`,
  ALL PASS):** color su(3) = the 8 traceless mode-bilinears (grade-2 Cl(6) bivectors) on the k*=3
  edge-modes; **T(3)=½, C₂(adj)=3 fall out as TRACES over the object** (= `gauge_dynkin`'s hardcoded
  values, de-imported); native b₃=−7/−3. GROUP factors NATIVE (modest: the forced connection, not the
  Casimir arithmetic). (ii) **PROBE 2 (`NATIVE_a4_spin_rows_2026-07-05.py`, WALL, pre-declared):** the
  substrate λ=−1 cone is a SPIN-1 multifold (Chern −2,0,+2; flat middle band); the FERMION row (+2/3)
  is native (cone=1 Weyl per cone, 06-25) ⟹ with probe 1 the whole (2/3)T_f is native; but the VECTOR
  (−11/3) & SCALAR (+1/3) SPIN coefficients are UNLOCKED on the spin-1 cone (a₄ counts 4/1/2 ≠ Dirac
  2/2/0) ⟹ walled on A5(b) = the spin-1→spin-½ Clifford locking / PS-embedding split (OMEGA_T4's
  open Target 4). LEDGER: {SU(3) group factors + fermion row = NATIVE} vs {vector/scalar spin =
  walled on A5(b)}. No value moved; no PDG.
  (iii) **A5(b) CLOSED 2026-07-05 (`A5b_closure_kahler_dirac_reduction_2026-07-05.py`, LOCK; pre-reg
  `A5b_closure_prereg_2026-07-05.md` @0eb7444):** the spin-1→spin-½ Clifford locking is DERIVED —
  the band spin-1 cone and the Fock spin-½ Dirac are the VECTOR and SPINOR reps of ONE emergent SO(3)
  (Q0's Albanese rotation, shared momentum index; C1 S_a spin-1, C2 J_a spin-½ same struct-const);
  the physical fermion (Fock Dirac, T-ID2 γ^{h_a}) + the Clifford current LOCKS the a₄ counts:
  timelike 4→**2.000** (sigma_shell), topological −2→**0** (Chern), spacelike →2 (γ^ρ = genuine
  Lorentz 4-vector; the emergent Cl(3,1) IS the locking mechanism). Every ingredient a cited theorem;
  nothing chosen for 2/2/0. **The −11/3/+1/3 wall's PREMISE (unlocked cone) is LIFTED** — the locked
  Dirac ⟹ standard Seeley–DeWitt applies (the declared Type-3 import, same as Ihara–Bass); the probe
  does NOT recompute −11/3 from scratch. Residual: facet (c) collapses to A5(a) "matter=Fock" — not a
  new adoption; A5 stays irreducible. STILL OPEN (unrelated to A5(b)): U(1)_Y (which-U(1)) — **SU(2)_L now
  NATIVE** (probe 3 `NATIVE_a4_su2L_2026-07-06.py`, T(2)=½/C₂=2 via the T-ID2 commutant su(2); gauge-row
  group-factor column COMPLETE). **Piece 1's A5(b) open equation is CLOSED.**

### 6. The substrate-selection discriminator — srs is DOMINANT among waterline survivors, not uniquely forced (ruled 2026-07-01)
**Status: the selection equation is incomplete top-down.** Per the R-9 SUPERSESSION
(`docs/audits/registers/structural_residue_register.md`, 2026-06-15, probe-backed ×4) and the 2026-07-01
ruling accepting it: the operative substrate closure is the RCSR structural-fingerprint study — srs is
**DOMINANT in an MDL-waterline superposition of survivors {srs, srs-c8, lou, lov}**, discriminated by PDG
observables. The "(A) → arc-transitive → Sunada → srs unique" chain is retained as provenance only
(`arc_transitivity_ground_truth.py`: srs-z IS arc-transitive, so arc-transitivity does not hard-gate it;
strong isotropy is a true Sunada-certified property of srs but was shown 4 independent ways not to carry
the selection load).
- **The open equation, stated exactly:** what structural functional of the one object, computable WITHOUT
  data, separates srs from {srs-c8, lou, lov}? The register's srs-z study shows all 14 prediction
  differences trace to the single fact of cell-doubling (extensive quantities; the intensive spectrum is
  bit-identical) — so the candidate discriminator is extensive/topological (cell size, cover degree), and
  its MDL cost must be *forced*, not asserted. Until it is derived, the substrate selection step consumes
  data, and the claim is scoped accordingly.
- **Honest scoping (not a weakening of the physics):** a survivor superposition is the NATIVE A2 reading —
  the waterline retains every representation that saves bits; observation discriminates. The same
  selective-retention logic used for chirality and triality applies to the substrate itself. But per the
  top-down law the *discrimination* must eventually fall out of the object, not the data — that is this
  open equation.
- **No predicted value changes** (all live values are srs reads). This is the claim-level honesty of the
  selection step; front-door language updated accordingly (A1 pass, 2026-07-01).

### 7. Widths/lifetimes — the frequency-RESOLVED self-energy Σ_X(ω) with thresholds is un-built (F4, located 2026-07-02)
**UPDATE 2026-07-06 — the Γ_Z/M_Z RATIO was landed by the R-V radiative layer; D₄ S4 characterized its loop
coefficient's grade (do NOT conflate with the broader open equation).** The specific Γ_Z/M_Z ratio was landed
by the R-V/EW radiative layer (`predictions/ew_width_layer.py`, V1/V2 2026-07-02) — it SHIPS at **−0.55σ**
(the "+4.8σ_exp OPEN" below is the RAW pre-layer assembly). **D₄ S4** (`D4_S4_carkms_loop_coefficient_2026-07-06.py`,
ALL PASS; pre-reg 72907fa) then characterized THAT layer's loop COEFFICIENT: the UV/RG side IS the S1 a₄
(native, K-rational); the FINITE remainder is a continuum-loop **TRANSCENDENTAL** (bubble −2+√3π/3; vertex
π²/12−ln2²/2), the SAME Type-3 class as the golden-rule 1/(48π) ⟹ it does NOT admit a clean forced spectral
read; the ceiling is **FUNDAMENTAL** (now UNDERSTOOD, not a missing derivation). This is GRADE work on a
SHIPPED value — NOT relabeling an open miss. **DISTINCT and STILL OPEN:** the BROADER §7 equation — the full
ω-RESOLVED Σ_X(ω) FUNCTION (thresholds, the 61-decade width span across ALL species) — remains un-built; the
R-V layer landed one matching-point ratio, not the width function.

**Status: LOCATED by the pre-registered over-application audit
(`proofs/foundations/F4_width_math_verification_2026-07-02.py`, 20/20; scoping doc
internal research notes).** A width is −2·Im of a pole of an
energy-resolved resolvent fed by OPEN channels only. The framework's Im structure at the matching
point (Σ(h)=α₁/h, √5/4, √7/4, the 1/√2 step) is **transport/dephasing content** (δρ's verified
use), NOT a particle width: measured Γ/m spans >61 decades (e→t) while every matching-point read
is one constant (Γ_e=0 kill-test; μ over-applied ×1.5e16; gauge bosons ×1.6 = the only right-order
regime).
- **The open equation, stated exactly:** the map X → (channel, pole frequency ω_X in the band
  variable, open final-state set) and the girth-window embedding Σ_X(ω) evaluated THERE — the
  Feshbach theorem gives one constant Σ(h); the width needs the FUNCTION, with thresholds =
  the framework's own dressed masses (Γ_e = 0 by channel-emptiness exactly; top closed at the Z
  by its own m_t read). The E-resolution already exists in the object (Im g_cavity = π·DOS on-cut,
  exact; Bloch F = Im λ/|λ|² over the BZ); what is un-built is Σ at a specified ω off the
  matching point.
- **In-reach sub-question ANSWERED 2026-07-02 (same day, S2a) — the band route is the KILL branch**
  (`proofs/foundations/F4_cone_spectral_function_2026-07-02.py`, ALL PASS; results in the session
  doc §5-RESULT): the substrate cones (adjacency Γ/R and Hodge-Dirac Γ) are ~~chirally warped
  spin-1-like multifolds (non-metric under the cubic little group ⇒ un-isotropizable), C =
  2.76×(1/12π) in mean-v units — NOT the Dirac value~~ **[CORRECTED same day by Q0, next bullet:
  the velocities (v₁₀₀=1/√2, v₁₁₀=1/2, v₁₁₁=1/√3, k·p-verified) are right, but the "non-metric/
  chirally-warped/non-universal-C" interpretation was a coordinate artifact]**; the direct
  pair-creation channel is **q²-DARK**, all low-ω weight flowing through the two exact H¹ flat
  bands (the gauge sector) — these parts stand. ⟹ **the 1/(12π) phase space is NOT band-geometric;
  it is Clifford-kinematic** — still true, in the sharpened LOCKING sense of the Q0 bullet. The
  Σ_X(ω) equation needs: Clifford-trace vertex kinematics + band-side thresholds/content.
  Converges on the SAME continuum-D₄/Clifford keystone as the −70 ppm.
- **Q0 ANSWERED 2026-07-02 (Ω session 1, `proofs/foundations/OMEGA_Q0_albanese_isotropy_2026-07-02.py`,
  ALL PASS; session doc internal research notes): (a) — ISOTROPY
  RESTORATION DERIVED.** The S2a "non-metric" verdict applied O(3) in homology Bloch coordinates;
  the actual little-group action is GL(3,Z) preserving the H₁ cycle Gram (all 24 automorphisms
  computed; invariant form unique by irreducibility). Exact results: both cones are perfect metric
  cones at leading order (sympy char-poly identities); **Q_adj⁻¹ = Gram_H₁ = 3I+C exactly (no free
  scalar)** = the Kotani–Sunada standard-realization/Albanese metric read off the object's own
  H¹/gauge sector; in Albanese momentum **v_adj = 1 and v_Hodge = 1/2 exactly**; the substrate
  cone constant is the **universal isotropic spin-1 value 1/(6π)** (pipeline re-run, +0.15%); the
  S2a anomaly 0.0733 = 2⟨v⟩·(1/6π) postdicted to 0.11% including the two-object coincidence
  (Q_h = Q_a/4). Bands are even in q at every order, so nothing about the cone was "chiral"
  (chirality lives in eigenvectors: adjacency Γ-triple Chern = (−2,0,+2), R = (+2,0,−2) conjugate,
  Hodge pair REAL/Chern-0 — `OMEGA_T4_clifford_12pi_2026-07-02.py`). **a₂/a₄ now mean:** cone-sector
  Seeley–DeWitt coefficients w.r.t. the Albanese volume (V_alb = 4/cell, v = 1/2 explicit); the
  flats are a separate 1D index sector (Str ≡ −2 = χ per fiber, exact).
- **The 1/(12π) layer moved 2026-07-02 (Ω session 1, `OMEGA_T4_clifford_12pi_2026-07-02.py`, ALL
  PASS):** the per-Weyl unit 1/(24π) and the per-Dirac **(v²+a²)/(12π) are DERIVED exactly** from
  the Clifford trace + the Q0 metric (symbolic phase-space integral + calibrated-pipeline
  cross-check at −0.00%). The band cannot supply them — **the LOCKING VIOLATION:** the multifold's
  three "Weyl counts" are pairwise unequal (timelike 4, spacelike 1, topological 2); only a Lorentz
  (Clifford) channel locks all three to its content. Named residual (not absorbed): that the
  physical EW current is the spinor current γ^μ(v−aγ⁵) — P3 FORM derived, PS-embedding split =
  the identified Type-3-conditional step (Clause 10c upgrade argued, prediction-file wording
  user-gated).
- **T-ID2 SITTING 1 EXECUTED 2026-07-02 (`proofs/foundations/TID2_A_split_and_J_2026-07-02.py`, ALL PASS;
  kickoff pre-committed 5d46928) — THE SPACETIME/INTERNAL SPLIT THEOREM CHAIN LANDS.** The Cl(6) generator
  space = the edge space R⁶; its decomposition H¹ ⊕ B¹ is **UNIQUE** (the two inequivalent 3-dim S₄ irreps;
  Hom_{S₄} = 0 — no invariant alternative exists); spacetime = H¹ (Q0/Albanese). Cl(3)_{H¹} exact; equal
  chiralities; commutant M₂ ⊕ M₂. **THE J-THEOREMS: no S₄-invariant complex structure exists; Hom_{A₄} = 1 ⟹
  THE A₄-canonical J (unique up to ±) ⟹ the CAR/Fock quantization FORCES S₄ → A₄ — the framework's ℂ[A₄] is
  the stabilizer of the complex structure quantization requires; every odd permutation flips J → −J ⟹ the ±J
  pair IS the enantiomer pair (srs ↔ srs-z; the joint object = both quantizations).** The canonical modes
  satisfy the CAR exactly, form an A₄ TRIPLET, and reproduce the Hamming/species grading — **read_species'
  Fock structure now derives from THE canonical J, not a pairing convention.** ~~Recorded for sitting 2: N̂'s
  commutant fraction 0.8125~~ **CORRECTED SITTING 2 (`TID2_B_current_form_2026-07-02.py`, ALL PASS): the
  fraction is 3/4 EXACTLY** (exact identity N̂ = 3/2 + D̂/2; the recorded 0.8125 was a vec-convention bug,
  predicted in the pre-registration and verified). **SITTING-2 OUTCOME: the commutant = even-Cl(3)_{B¹} ⊗
  {1, ω₃} exactly = an internal su(2) PER CHIRALITY (all-doublet, Casimir ≡ 3/4, ω₃ 4+4 — four internal
  labels = four species slots per spatial spinor); the dipole remainder D̂ is A₄-invariant and purely mixed =
  the HIGGS-DIRECTION candidate (CLEANROOM §5, flagged). K4 FIRED: the 'charges-internal' half of the
  candidate DIES — the parity (−1)^N anticommutes with the spatial Clifford (split-odd ω₆-class) ⟹ Q̂ has
  ZERO internal component and the Hamming ladder's internal shadow is trivial; the species ladder (1+3+3+1)
  and the split labels (2⊗2 per chirality) are TWISTED by exactly the parity sector. The split-uniqueness and
  J-theorems STAND. **SITTING 3 EXECUTED
  (`TID2_C_lorentzian_assembly_2026-07-02.py`, ALL PASS): THE LORENTZIAN ASSEMBLY LANDS — Cl(3,1) exact with
  γ⁰ = the internal B¹-VOLUME (the (−,+,+,+) signature EMERGES from the split; γ⁰² = −1, nothing inserted);
  γ⁵ = −ω₆ = the existing cl6_chirality (P3's grading IS the assembled 4D chirality — consistency lock); γ⁰
  is A₄-invariant and flips under every odd permutation (the time orientation rides the enantiomer choice);
  the Cl(3,1) commutant = EXACTLY ONE su(2) (chirality-preserving) ⟹ the site-local Fock space = (4-comp
  DIRAC SPINOR) ⊗ (su(2) DOUBLET) — one Dirac doublet per site; and the sitting-2 obstruction's exact
  identity: the Fock parity (−1)^N = −i·ω₆ ∝ γ⁵ — the species (−1)ⁿ factors are CHIRALITY/axial factors; the
  U(1)/color content is deck/winding-sector (read_gauge's own home).** **SITTING 4 EXECUTED
  (`TID2_D_chirality_bit_2026-07-02.py`, ALL PASS) — T-ID2's PLANNED CORE COMPLETE (4/4 sittings, every
  pre-registration git-witnessed): the mirror is T-LIKE (odd permutations: det R|_{H¹} = +1 — SPACE
  orientation PRESERVED; det R|_{B¹} = −1 — time gamma and 4D chirality FLIP); THE ONE-BIT THEOREM: one Z₂
  datum (the enantiomer) coherently carries {quantization sign J, time orientation γ⁰, chirality γ⁵, dart
  handedness e₁e₂} ⟹ a CHIRAL coupling needs no import — L-vs-R IS the srs-vs-srs-z choice; the read's
  T₃-pattern operator (−1)^N/2 = (i/2)γ⁵ EXACTLY (the weak-isospin READ is the chirality grading — the
  sitting-2 twist fully named); the SM-form T̂₃ = (iK₃)P_L (both factors previously derived) has spectrum
  {±½ ×2, 0 ×4} = the one-generation doublet pattern; P_L ↔ P_R rides the bit.** **T-ID1 SITTING 1 EXECUTED
  (`TID1_A_coupling_rule_2026-07-02.py`, ALL PASS; kickoff pre-committed 37b3310): THE COUPLING RULE EXISTS —
  one function rule(channel, disc-class, projection, order) reproduces ALL EIGHT worked dark-sector instances
  exactly from forced inputs (the case law is ONE LAW; the Ihara–Bass discriminant = the coherence criterion:
  disc > 0 ⟹ resummed windings, disc ≤ 0 ⟹ leading-only component-wise-real per S2b; pole dressings rank 1/2
  by the L-rule; order-2 leg counts; c_v and ½-EW flags KEPT). R2 first computation, OUTCOME (i): the mirror
  DISTINGUISHES the factors — the deck U(1) charge is C-conjugated (axis-preserving mirror: W → −W exact),
  the su(2) is SELF-CONJUGATE (bivector Λ², det ≡ +1, exact inner automorphism) ⟹ CANDIDATE per-factor rule:
  charge-flippable ⟹ vector-like; self-conjugate ⟹ chiral (P_L on exactly the su(2)); NAMED TENSION: SM
  hypercharge chirality via the Pati–Salam decomposition (Y = T₃R + B−L mix) — sitting 2 must DERIVE the rule
  and resolve it. R3 (the rate clause) stated with S2b/S6/Q1/C2 pedigree; the loop program's entry form is
  fixed (c-weighted CAR-KMS EW loop).** **T-ID1 SITTING 2 EXECUTED
  (`TID1_B_per_factor_rule_2026-07-02.py`, ALL PASS; pre-reg 1e4edfa): the SECOND su(2) = Cl(0,2)'s
  quaternion factor (the dart qubit) — the PATI–SALAM PAIR su(2)_{B¹} × su(2)_{02} survives the Lorentzian
  assembly (16-dim commutant, commuting pair verified); the dart swap is inner and flips ω₀₂, bit-locked with
  χ ⟹ the LR-MIRROR (joint object LR-symmetric, each enantiomer chiral — PS realized by the mirror pair);
  the HYPERCHARGE TENSION RESOLVES by exact rational arithmetic on the read's own table: **B−L =
  (−1)ⁿ(2n−k*)/k*** (a clean Fock read), **B−L = 2Q − (−1)ⁿ** (the charge's vector/chirality split = the
  split T-ID2 s2 measured on Q̂), **Y_L = (B−L)/2 and Y_R = T₃ᴿ + (B−L)/2 on all 8 states** — hypercharge
  chiral only through T₃ᴿ; the one-unit principle at A2-class with a 4/4 consequence table.** **T-ID1 SITTING 3 EXECUTED
  (`TID1_C_vertex_selector_2026-07-02.py`, ALL PASS; pre-reg 509fc36) — T-ID1 COMPLETE AT MAXIMAL GRADE.**
  The real structure built explicitly (C₈ = γ¹γ³γ⁵∘conj fixes the gammas, C₈² = −1, flips χ; C₂ = σy∘conj
  quaternionic — Cl(0,2) ≅ ℍ confirmed at the real-structure level; su(2)_L generators C-real). The survival
  table's kill fired UPWARD into an **IMPOSSIBILITY THEOREM: the real structure is CHIRALITY-BLIND by
  construction (identical vector/axial antisymmetry columns, cellwise) ⟹ the vertex-level chirality selection
  is NECESSARILY the layer/enantiomer bit = THE ARROW OF TIME (T-ID2 s4's one-bit theorem) ⟹ the SM's
  L-selection adds ZERO description length — it is the already-counted arrow.** Deriving L-vs-R further would
  contradict the joint object's mirror symmetry. Bonus recorded: the temporal/spatial survival split is
  factor-typed (charge-type = densities; su(2)-type = spatial currents; site-local — the cover propagates).
  **T-ID1 SCOREBOARD: R1 (one law, 8/8) · R2 (classification + PS pair + hypercharge on the read's table +
  the bit = the arrow + the impossibility closure) · R3 (rate clause) — THE LOOP PROGRAM'S PROJECTIONS ARE
  FULLY SPECIFIED.** Front-door interpretations user-gated.
- **The ω-resolved VERTEX class established 2026-07-02 (Ω session 1,
  `OMEGA_T3_width_vertex_omega_class_2026-07-02.py`, ALL PASS) — the first width-side object with
  the demanded SIGN.** S6 pins the winding amplitudes z-flat (topological uⁿ, Z_res = 1); the
  forced DURATION (a winding = g ticks) gives the probe-frequency response W(ω) = Σuⁿe^{ingω}.
  **Sign lemma (sympy, exact): W(0) − Re W(θ) = u(1+u)(1−cosθ)/[(1−u)(1−2ucosθ+u²)] ≥ 0** — the
  class can ONLY reduce the pole vertex (DOWN = the demand's direction; the residue class was
  UP-only: the algebra is now bracketed). Pattern-type matches S4 by construction (width-only,
  Γ_W/Γ_Z-cancelling, pole positions untouched, Γ_e = 0 preserved); magnitude range covers the
  demand (max 2u/(1−u²) = 7.8% raw, 0.65% under c_S — c_S re-use stays POISONED). **Value NOT
  claimed (±21% rule): the remaining §7 core = (i) the pole-phase map X → θ_X (one dimensionless
  phase per particle), (ii) the forced vertex projection.** θ-inversions recorded as poisoned
  comparisons only (1.85/1.16/0.44 rad). Γ_Z/M_Z stays OPEN (+4.8σ).
- **C2 EXECUTED 2026-07-02 (∂_N construction program, `proofs/foundations/DN_C2_vertex_loop_class_2026-07-02.py`,
  ALL PASS; pre-registration committed BEFORE the run, 2188fbe) — R-V's CLASS SELECTED: the CAR-KMS matter
  loop.** The q²-dark admixture lead RETIRED (pair-darkness re-verified: |M|² ratio 4.00, 6.8e-6 at q = 0.03;
  the admixture fraction has no forced nonzero home — 0 under the P3 identification, (E/E_sub)²-suppressed
  otherwise; its sign argument was right, its magnitude structurally empty). **The demand in the loop's own
  natural unit (α₂/4π = 0.2690%, fresh from the g₂ leaf): +0.89 ± 0.33 (G_F^v-form) / −1.62 ± 0.34 (α-form) —
  the FIRST O(1)-coefficient candidate class in the entire F4→Ω→∂_N chain** (all others orders-off or excluded
  by pedigree/sign/theorem; the S3 frozen accounting independently attributed the residual to exactly this
  layer). All four S4/falsification surfaces hold (common part cancels in Γ_W/Γ_Z; differential ~0.08% ≪ ±2%;
  pole positions untouched; Γ_e = 0). **REDUCTION: conditional on the P3/PS current identification the loop's
  content is standard EW ⟹ R-V = SM-REPRODUCTION-CONDITIONAL (the 1/(12π) grade family); the from-scratch
  coefficient requires the interacting sector coupling.** ⟹ **PROGRAM-PHASE CLOSE: with C1 (R-G derived-
  conditional) and C3 (R-ε → interacting run), ALL remaining content hangs on ONE keystone — theorem-grade the
  identification layer (the walk↔Fock dictionary / P3-PS current split, the framework's single A5-class
  seam).** Γ_Z/M_Z stays +4.8σ OPEN as shipped. Named-not-acted user-gated option: a NEW registered assembly
  variant importing the EW radiative layer as declared Type-3 (like 48π/1.409) would close Γ_Z/M_Z numerically
  at SM-reproduction grade — a registration decision, not a derivation.
- **LOOP V1 EXECUTED 2026-07-02 (`proofs/foundations/LOOP_V1_car_kms_calibration_2026-07-02.py`, ALL PASS
  32/32; pre-registration committed BEFORE the probe, a5287f4; verify 65/65) — the R-V loop machinery is
  CALIBRATED and the EVALUATION RULE IS DERIVED.** Calibration at/beyond the S2a standard: the Veltman doublet
  Δρ ≡ (N_c g²/(64π²m_W²))F(m₁², m₂²) SYMBOLICALLY (per-log-atom residues exactly 0), Ward/custodial/decoupling
  and Q_u/s²/μ²-independence exact, optical-vs-symbolic Im at 3e-13%, dispersion rebuild ~1e-12%, sub-threshold
  absorptive parts exactly zero (the Γ_e = 0 structural fact), **the massless lock Im Π_T = s(v²+a²)/(12π) at
  1e-14 — the T4 Clifford unit as the machinery's own optical theorem** (1/(48π) = (1/12π)×(g/2c)²-norm).
  **RED: the KMS loop family has exactly TWO parameter-free evaluations (β→0 dead / β→∞ vacuum); interior β =
  a forbidden continuous input (CLEANROOM §7 + §6 III₁); a derived clock is Q1-excluded; the ARROW (the
  already-counted one bit) selects the VACUUM loop ⟹ the EW radiative layer = standard EW one-loop with
  framework inputs — C2's reduction with the evaluation rule now forced; thermality enters as statistics only
  (C1's parity doubling). NEW conditionals: none.** No framework number touched; the target appears nowhere in
  the probe. Γ_Z/M_Z stays +4.8σ OPEN. **NEXT: V2 (fresh session) — pre-registration must freeze the scheme
  (framework couplings in their validated MS-bar-analog roles), pre-decide the known α_s×x_t two-loop-Δρ import
  question (band-relevant), and gate all four surfaces; single marked comparison.**
- **LOOP V2 EXECUTED 2026-07-02 (`proofs/foundations/LOOP_V2_rv_blind_evaluation_2026-07-02.py`, ALL PASS
  12/12; pre-registration committed BEFORE the probe, d37a679, incl. the frozen scheme + tier rule) —
  R-V LANDS: the EW radiative layer on the α-form golden rule = −0.4864% = −1.81 loop units vs the
  pre-registered demand −0.437% ± 0.092% = −1.62 ± 0.34 (pull −0.54, LANDING tier).** Γ_Z/M_Z closes
  **+4.76σ → −0.55σ BY DERIVATION** — equal to the SM's own −0.53σ residual (SM-REPRODUCTION grade doing
  exactly what it says). Method: the certified PDG-2024 worked example (Table 10.6; sums pass at 0.03 MeV;
  the α-form W channel reproduces Γ(W→eν) = 226.29 ± 0.04 MeV at +0.010%) extracted against the SHIPPED
  α-form tree at the PDG MS̄ point; applied at framework leaves with all input-drift sensitivities bounded
  (|ΔS| < 0.012 loop units, 30× under band; scheme legitimacy = the repo's own scheme convention §7: the
  RG-endpoint couplings are MS̄-at-M_Z by declaration). Surfaces: Γ_W/Γ_Z −0.06σ → +0.14σ sub-σ HOLDS (the
  kickoff's "differential ≲0.1%" size-estimate MISSED, actual +0.41% — the κ̂/b-vertex content has no W
  analog; disclosed, not relabeled); poles untouched; Γ_e = 0. Disclosed pre-reg calibration miss: the
  blanket ±2% per-channel gate fires on the b-row (−2.45% = exactly its named content ρ_t −1.25% + κ̂_b
  −0.18% + b-mass −0.41% + common −0.63%; certified independently by the Eq.-10.55 structure check,
  residual −0.41% in-window). **GRADE: SM-REPRODUCTION-CONDITIONAL (C2's reduction + V1's derived vacuum-
  loop evaluation rule; standing conditional = the P3/PS identification). NO VALUE SHIPPED: the +4.8σ
  header STANDS until the user gates the registration step (Clause 10; scope includes the Γ_W/Γ_Z
  companion header). NEXT: R-ε — the γ⁵-graded sector of the INTERACTING run; no worked example exists;
  genuinely from-scratch (C3 killed all free-gas evaluations).**
- **REGISTRATION EXECUTED same day (USER GATE): Γ_Z/M_Z = 0.027350 (−0.55σ, Clause 8c PASS) and
  Γ_W/Γ_Z = 0.83802 (+0.14σ) now SHIPPED with the derived layer** via the new single-source leaf
  `predictions/ew_width_layer.py` (Clause 9b bridge tag explicit; [external] certified PDG-2024 worked
  example in-file; anti-drift welds; 10b tripwire asserts the pre-layer deficit's presence). Value lock
  re-frozen deliberately (103 → 104; the designed FAIL fired on exactly the intended 2 drifts + 1 new);
  DAG 114/0; verify ALL-PASS; MDL ledger margin unchanged (+168.0 — widths are not parameter rows).
  **THE GRADE CEILING IS THE REMAINING §7 CONTENT: the native O(1) coefficient (the interacting sector
  coupling / walk↔Fock dictionary at theorem grade) — the row can never pass bridge-conditional until it
  lands. The M_Z pole oblique (+6σ-class) is NOT touched (rates only, R3 clause) and remains the pole-side
  open item. R-ε (−70 ppm hard core) remains the loop program's open number-mover.**
- **LOOP E1 EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E1_walk_fock_dictionary_2026-07-02.py`, ALL PASS
  16/16; pre-reg witnessed e82ee62 BEFORE the probe) — the walk↔Fock dictionary's OPERATOR LAYER IS DERIVED;
  the A5 seam narrows to ONE weld.** (D1) The Ihara–Bass identity derived in-probe over the dart-reversal
  involution: det(I−uB) = [Π_edges(1−u)(1+u) = the dart-qubit swap eigen-split, T14-welded (swap flips ω₀₂)]
  × [site-cavity determinant with the u²(D−1) backtrack self-energy = cavity_gf's structure; its quadratic's
  two roots = the IB branch pair]; exact at Γ + generic k, real + complex fugacity; all blocks S4-covariant.
  The pair sector's u-quanta are SINGLE pair-mode excitations ⟹ **tick parity = Fock parity is now a THEOREM
  on the pair/flat sector — C1's conditional (i) upgrades there (wording user-gated; R-G grade improves).**
  (D2, the sharpest result) **the step lift is FORCED OUTRIGHT: of 16 A₄-equivariant parity-odd edge-covariant
  families, the vacuum-block discriminator leaves ZERO freedom — X_a = γ_a unique ⟹ ε cannot live in any
  operator deformation; it lives in the STATE coupling of the two forced measures** (C0's walk ensemble ×
  CAR-KMS) across the site↔species seam = the one named remaining conditional. Naive matter-Fock transfer
  candidates measured (Fock traces select ℤ₂-even/cycle classes, not NB): non-matches recorded. **E2 gate:
  force the seam OR show its freedom doesn't touch the mirror-odd sector — pre-register exactly one. The
  −70 ppm stays OPEN.**
- **LOOP E1b EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E1b_seam_parity_2026-07-02.py`, ALL PASS 18/18;
  pre-reg e185e0e) — (b-PASS): THE E2 GATE OPENS.** In the DERIVED frame (the Fock functor Γ(V) = the CAR
  structure's own lift, vacuum-canonical by construction; δ = det V_g ≡ 1, mode rep ⊂ SU(3)): vertex ℂ⁴ ≅
  F_even ≅ F_odd ≅ trivial⊕triplet ⟹ the seam ambient = 4 Schur-forced channels with the derived table
  (even: Perron→vacuum, triple→u-slot; odd: Perron→e-slot, triple→d-slot; all images Hamming-pure). The
  statistics theorem (E1+C1) kills the even half ⟹ **admissible seams = odd-half isometries with ONE
  physical relative phase θ_seam** (recorded bonus: = the (c₀, c₁e^{iδ}) relative-phase class of the mass
  read — question (a)'s remaining content). **The mirror maps the entire admissible set out of itself
  (clean half-exchange) ⟹ ZERO in-layer mirror-odd seam freedom — the mirror on admissibility = the pure
  layer swap = the already-counted bit ⟹ ε's odd channel factors through DERIVED structure only; E2 (blind
  ε) is WELL-POSED with the seam quarantined.** Sitting disclosures: E1 ERRATUM (conj-nullspace bug in
  lift_U; E1 re-run corrected — the D2 zero-freedom verdict REPRODUCES; conclusions stand) + two new trap
  entries propagated to the spine §1 (conj(Vh) nullspaces; phase-incoherent lifts vs the derived Γ(V)
  frame). The −70 ppm stays OPEN; NEXT = E2 (fresh sitting, own pre-reg: T10-bit odd projection frozen,
  lepton-slice point, resummation protocol vs C3's ladder, 4 surfaces, single marked comparison).
- **LOOP E2b EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E2b_blind_epsilon_2026-07-02.py`, run exactly as
  pre-registered 361da9f, banked AS-RUN with its FAILs visible) — THE PRE-REGISTERED TIER-KILL FIRES:
  ε_raw(as-registered) = −4.97e-2 rad = ×2.8e5 over; R-ε stays OPEN; no adoption, no relabeling.**
  Post-mortem (the kill's information): (1) the frozen functional was MIS-FRAMED (my design error, exposed
  by the probe's own control): the difference-form ½(arg g_h − arg g_h̄) equals the intrinsic SHELL PHASE on
  conjugate pairs — the exact non-δ trap the pre-reg cited; the conjugation-ODD invariant is the phase-SUM /
  modulus-DIFFERENCE form (E2a's own A = t₁ − conj(t₂) structure); (2) the line component (+8.9e-4) sits at
  C3-c's killed scale (bare-channel free-class violence); (3) off Γ the IB branch pair is not
  conjugate-paired (trap #4) ⟹ no functional patch suffices. **RE-LOCALIZATION (pre-named, now confirmed
  twice): the READ-PROJECTION LAYER — δ is the phase of the ω-isotype amplitude of the GENERATION TRIPLE
  (E1b's odd-half triplet channel, triple→d-slot/Λ¹); the dressing of δ = the phase of the DRESSED
  triplet-channel amplitude, not the bare dart-channel expectation. The −70 ppm's open equation is now: the
  interacting triplet-channel amplitude's chiral phase (E2a's forced G_int projected through the READ's own
  channel weights).** The interacting chiral channel itself remains a THEOREM (E2a). NEXT: E2c = derive the
  read-projection functional (fresh sitting, own pre-reg); only then a further blind evaluation.
- **LOOP E2a EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E2a_interacting_form_2026-07-02.py`, ALL PASS
  12/12; pre-reg committed before the probe) — THE INTERACTING FORM IS FORCED; THE CHIRAL CHANNEL IS OPEN.**
  The vacuum pairing on the derived Fock structure is **C = I + iJ EXACTLY** (Wick/Pfaffian certified
  in-probe); the interacting walk propagator G_int(u) = ⟨0|(I−uW)⁻¹|0⟩ with W = ΣB_{d'd}γ_{e(d')}⊗E_{d'd}
  (all pieces forced: B, the E1-rigid step lift, the V1 vacuum, the canonical J; γ→1 reduction = the free
  ensemble exactly; = the Wick-weighted path sum order-by-order; odd u-orders vanish = the u²/bilinear
  grading, C1-consistent). **SELECTION RULES: the mirror flips the iJ part exactly; at Γ the free ensemble
  reproduces Q3's conjugation theorem (μ_ω = μ_ω̄ at 1e-16, the control) while the INTERACTING ensemble
  carries a NONZERO ω-vs-ω̄ asymmetry flipping with the layer bit (A(+J) = −conj(A(−J))) — the chiral
  channel the −70 ppm requires EXISTS, is FORCED, and evades the conjugation theorem exactly through the
  iJ pairing.** C3's free-gas over-application ladder explained structurally (the free ensemble had NO
  chiral channel; the free candidates borrowed violence from the wrong sector). K4a did not fire (no
  un-forced choice anywhere). **E2b (the blind ε number) FROZEN in the E2a banner: the winding-chiral phase
  of G_int along the screw line to the lepton slice with the Bloch cocycle (trap #5) frozen in E2b's own
  pre-reg; resummation = the resolvent; 4 surfaces; single marked comparison; tier rule; C3 ladder as
  reference. The −70 ppm stays OPEN until E2b lands or kills.**
- **Q1 DECIDED 2026-07-02 (Ω session 2 station 1, `proofs/foundations/OMEGA_S2_Q1_which_clock_2026-07-02.py`,
  ALL PASS — the pre-registered S1-KILL): the ω-vertex VALUE claim is FALSIFIED; the winding layer
  is excluded TWO-SIDEDLY.** The ω-extension is the fugacity phase (forced, operator-level: one
  tick per NB step). The Z channel in the tick frame is REAL, SUBCRITICAL (2u = 0.078 < 1 — the
  same fact as the arrow) and OVERDAMPED: spectrum max at ω = 0, no real-frequency resonance, the
  only pole purely imaginary (−i·2.55) ⟹ **the channel has no frequency to hand the winding
  interferometer.** Every III₁-admissible phase candidate is trivial (0, 2π) or Γ/M-sized (max
  0.17% raw — out of band); an absolute clock gives QUADRATIC triviality (deficit ∝ θ², sympy —
  even a 100× hierarchy is 40× below band; the framework's ladder ⟹ zero — and this same fact
  PROTECTS all shipped matching-point pole reads); the gap continuation is UNFORCED (+iκ diverges
  4.7e9; the physical retarded depth ≈ 0; −κ full-kill = a by-hand choice, its c_S pairing 0.3384%
  band-edge-outside anyway). **With S6: z-side (residue UP-only, amplitudes waterline-flat) +
  ω-side (zero response at EW poles) ⟹ the −0.437% ± 0.092% is NOT winding-layer content in ANY
  slot.** The T3 sign lemma survives as structure (the two lemmas bracket and exclude the layer).
  S1a (current-projected winding content) SUPERSEDED — no value-slot remains to project onto.
  **The §7 core moves UP: the pole-vertex deficit lives in the INTERNAL (Cl(6)/Clifford) EW-loop
  vertex layer — the framework-native ρ_f/s̄²_eff analogs, genuinely UN-BUILT** (only formally-
  signed existing class: per-leg Family-D c_F u², ~11× too small — S6). Leading sign-correct
  successor candidate, NAMED not built: the q²-DARK band-side admixture of the physical vertex
  (timelike darkness can only REMOVE pole weight ⟹ DOWN forced; magnitude = the vertex's
  band-orbital admixture fraction, requiring the P3/PS-embedding current identification of
  `OMEGA_T4`). Γ_Z/M_Z stays OPEN (+4.8σ); nothing shipped, no falsification surface touched.
- **Lemma PROVED 2026-07-02 (S2b, CAS: `proofs/foundations/F4_S2b_width_ratio_dark_lemma_2026-07-02.py`):**
  (L1) a REAL multiplicative dressing leaves Γ/M invariant EXACTLY and cancels identically in
  Γ_W/Γ_Z when common — and the gauge sector's matching-point dark reads the exactly-real Perron
  channel ⇒ the known dark sector cannot touch widths; (L2) a complex dressing would shift Γ/M by
  (2/(1−Σ_r))(1+(Γ/2M)²)Σ_i = 0.0444 for every shell fermion; (L3) that is over-applied ×1.6e16
  (μ) and contradicts Γ_e = 0 ⇒ EXCLUDED ⇒ **the dark map's component-wise REAL usage is FORCED
  BY STABILITY, not a convention** — the fermion pole stays real at matching-point order; widths
  live in Σ_X(ω) only.
- τ_μ is NOT a target: G_F is calibrated FROM τ_μ (`predictions/G_F.py`) — only the 192π³ rate
  structure could ever be honest content.
- **S3 EXECUTED 2026-07-02 — first width observables shipped as class-(b) assemblies**
  (`predictions/Gamma_W_over_Gamma_Z.py` −0.06σ PASS; `predictions/Gamma_Z_over_M_Z.py`
  **+0.44% = +4.8σ_exp OPEN residual**; registered per parameter_linter.md, value lock 103 PASS).
- **S4 EXECUTED 2026-07-02 — the "EW radiative layer" DECOMPOSED against the framework's own
  oblique set** (`proofs/foundations/F4_S4_width_oblique_decomposition_2026-07-02.py`, ALL PASS;
  diagnostic only, no closure, residual stays OPEN). Findings, each computed not asserted:
  1. **The width assembly is parametrization-consistent** — α-form ≡ G_F^tree-form×(1+δρ) is an
     exact tree identity of the framework's own quantities ⟹ the +0.44% is invariant content.
  2. ~~The G_F TRIANGLE GAP: +0.410%, "wired into nothing" (7b)~~ — **RETIRED same day (S5,
     chase-the-math-up): it is an EXACT IDENTITY of the existing oblique pair.**
     `proofs/foundations/F4_S5_GF_triangle_identity_2026-07-02.py` (ALL PASS): symbolically,
     within the framework's own chain (M_Z = √π·v·√(α₂+α_Y)(1−δ_r), m_W = M_Z·c·√(1+δρ),
     c² = α₂/(α₂+α_Y), g₂² = 4πα₂, G_F^v = 1/(√2v²)):
     **G_F^v/G_F^tree = (1−δ_r)²(1+δρ)** — the "+0.410% gap" = δρ − 2δ_r + O(δ²) = +0.408%,
     DERIVED content (live slack +0.0028%, located: the g₂-leaf vs M_Z-iteration α₂ rounding).
     The S4 framing "wired into nothing" was WRONG — it was fully determined; the identity had
     not been noticed. Corollary: α-form width ≡ G_F^v-form/(1−δ_r)² exactly.
  3. **Sub-equation (7a) ATTACKED AND LOCALIZED (S6, same day) — the residue route is
     SIGN-EXCLUDED; 7a merges into §7-proper as its sharpest numerical target.**
     `proofs/foundations/F4_S6_width_residue_no_go_2026-07-02.py` (ALL PASS):
     - **Sign lemma (sympy, exact):** for the Z-channel dressed pole with per-winding profile
       φ(z) = u(z₀/z)^a, Z_res − 1 = a·c_S·u/[(1−u)² − a·c_S·u] ≥ 0 for ALL a ≥ 0 (the sign
       class fixed by the PROVEN shell z-structure Σ(z) = α₁/z, decreasing). The residue can
       only dress the width UP: {a=0: 0, a=1: +0.353%, a=g: +3.65%}; the demand is DOWN
       (−0.437% ± 0.092%). **Residue route excluded regardless of coefficient** — decisive
       precisely because the demand band is ±21% in coefficient units (1/12, 1/9, 1/8 would
       all "pass" a magnitude test; only the sign/class argument is honest).
     - **Waterline theorem-let:** the framework's own reading (windings = A2-T topological
       classes, the axioms' 2026-04-21 NOTE — explicitly NOT a dynamical resummation) forces
       the profile a = 0 ⟹ **Z_res = 1 exactly: the framework predicts NO oblique-residue
       dressing of the width.**
     - **Taxonomy sweep (argument, not fit):** singlet c_S re-use = the mass-shift projection,
       no derivation for coupling re-use, POISONED (would "pass" at 1.1σ — the trap); vertex
       Family-D O(u²) 8× too small; channel c=1 9× too large; democratic 5/12 4× too large;
       custodial δρ wrong sign and 2.5× too large; S3 omissions ≤0.05% each; combining = fit.
     ⟹ **the −0.437% width normalization is genuine ω-resolved VERTEX content — the Σ_X(ω)
     equation of this §7, now carrying its sharpest target: the Zff̄ effective vertex at the
     pole must come out 0.437% ± 0.092% below the α-form tree assembly (and simultaneously fit
     the S4 pattern across M_Z/m_W/Γ_W/Γ_Z).** Mass-side reads (δ_r, δρ) are pole-POSITION
     content; the width's normalization is not — the matching-point program for widths is
     complete and honestly closed. Γ_Z/M_Z stays OPEN (+4.8σ).
  4. **VERDICT (V2): MULTI-COMPONENT.** The residual vector {M_Z +0.018%, m_W +0.040%,
     Γ_Z/M_Z +0.438%, Γ_W/Γ_Z −0.120%} is NOT collinear with a common ρ̄ direction (single-scalar
     test fails on 3 of 4 rows). Any candidate derivation of the layer must reproduce the
     PATTERN (three distinct directions: width-ρ̄, Δr̄/triangle, δ_r-completion), not one number
     — a much sharper falsification surface. Γ_W/Γ_Z is layer-insensitive (stays sub-σ), as
     shipped.
  5. **☠ PRE-POISONED NUMEROLOGY (declared upon computation, UNUSABLE without forced
     derivations):** demand −0.401×δρ ≈ −2/5·δρ (0.25% apart); the 0.599 ≈ 3/5 complement;
     the 7a candidate-dressing list (item 3). **POISON RESOLVED BY DERIVATION (S5): the
     "+0.410% ≈ (3/8)·δρ" proximity was accidental — the true object is the IDENTITY
     δρ − 2δ_r (item 2). Case study: the poison discipline worked — the coincidence was
     quarantined instead of adopted, and the real algebra arrived one session later.**

### 8. The neutrino absolute scale (m_ν₃ +2.18σ, m_ν₂ +1.87σ) — an ON-CUT subleading-spectral miss = the SAME gate as §1; NOT an N_hub wall (SCOPED 2026-07-05)
`m_ν₃ = (k*·N_atoms)·M_Pl·N_hub^(−1/2) = 12·M_Pl/√N_hub = 50.57 meV` is a **forced read** missing **+0.87% HIGH (+2.18σ)**; m_ν₂ = m_ν₃/√R and m_ν₁ = 0 hang off its scale. Forced/fine: **R = 228/7** (Ihara, 1.4σ), **m_ν₁ = 0** (rank-2 seesaw). Scoping verdict (a model; CORRECTS a recurring category error):
- **NOT walled behind N_hub / Gap-G1 — do NOT regenerate this (the recurring WRONG conclusion).** N_hub is the framework's ONE adopted GLOBAL dimensional input, pinned to ppm and shared by EVERY dimensional read (m_e, m_τ, M_Z, v, G_F; `predictions/N_hub.py:6-11`); it cannot move for the neutrino without breaking all of them ⟹ it is NOT a free leg of any "degeneracy," and "derive N_hub" is a category error (= Gap G1, which the framework does not attempt). The direct spectral-gap read is FORCED given the global N_hub.
- **NOT a y_ν adoption.** The seesaw re-reading `m_ν₃ = y_ν²·v²/M_R` (y_ν=1) is a re-parameterization; the DIRECT read `12·M_Pl/√N_hub` carries no y_ν. Attributing the residual to "adopted y_ν=1" is a framing choice, not the located incompleteness.
- **DARK-DC route CLOSED (do not re-walk):** m_ν₃ already bakes in the Feshbach residue-at-h; applying the universal DC template (1−√5/4·α₁/(1−α₁)) on top double-counts → 49.42 meV (−1.4%, −3.5σ, WORSE) (`predictions/m_nu3.py:74-84`, Open-Q0); the forced (0H+2F) Majorana-vertex subleading is +α₁²/6 ≈ +0.025% (too small).
- **CORRECT localization (the framework's OWN derived §7.6 selection rule; `docs/parameters/parameter_uniqueness_ledger.md:1079`, `proofs/foundations/selection_rule_reaudit_2026-05-16.py`):** **m_ν₃ is ON the McKay cut (disc≤0)** — the SAME cut as δρ. The derived criterion **FORBIDS closing an on-cut residual by geometric resummation**; the residual must be a **sub-leading-spectral (multi-insertion) sum** — an un-built ∂_N object. ⟹ the +2.18σ is an OPEN on-cut subleading-spectral miss of **exactly the §1 (−70 ppm) / δρ (+4.58%) class**, gated on the SAME un-built ∂_N sub-leading-spectral machinery. It is NOT a separate wall — it is the **neutrino face of the one open ∂_N-subleading frontier.**
- **Located sub-question (the concrete locus):** the live leading normalization 1/(k*·N_atoms)=1/12 is the Γ-Perron OFF-support object, while §7.6 places m_ν₃ ON-cut (the on-cut scale channel gives ≈196 meV, 4× off; `predictions/m_nu3_derivation.md:14-18`). Whether the 1/12 leading channel is itself forced (or an off-support coincidence) is the read-out-channel question to chase up — NOT a resummation.
- Status: **OPEN, correctly localized — same gate as §1.** No fit; no value moved. Standing rule: do NOT re-conclude "walled behind N_hub" (category error, regenerated ≥3×); do NOT re-walk the dark-DC route (closed); the residual is an on-cut sub-leading-spectral sum.

## ✅ RESOLVED 2026-06-30 — M_Z via the BZ-integrated Z-current vacuum polarization: 0.810 does NOT fall out; M_Z is a forced oblique residual
**The planned attack was carried out. The `0.810` does NOT fall out of the BZ integral (`R = 0.2046`, not 0.810).
M_Z is confirmed to be a FORCED substrate-vs-SM oblique difference — a real ~4%-relative residual — exactly the
honest prior. Complete honest result, not a failure, not a fit.** Deliverable:
`proofs/foundations/M_Z_BZ_integrated_vacuum_polarization_2026-06-30.py`; theorem:
`docs/theorems/theorem_M_Z_BZ_vacuum_polarization_2026-06-30.md`.

**What was built (forced, basis-free).** On `directed_edges()`'s own ordering ([B(0),P]=0 verified), the C₃ dart
permutation `P` and winding operator `W=(P−P²)/(i√3)` (eigs {0,±1}; Tr W²=8). Reproduced the Γ split exactly:
Perron Σw²=0, shell √2 **Σw²=4** (chiral half=2), |λ|=1 Σw²=4. Then the genuine BZ integral
`<Σw²·F>_BZ = ∫_BZ Σ_{Im λ>0} |⟨l|W|r⟩|²·Im(λ)/|λ|²` (chirality = the Im λ>0 hemisphere; at Γ the two
hemispheres cancel, h:+2·√7/4, h̄:−2·√7/4). Ratio to the Γ template `[Σw²·F]_Γ=2·√7/4`.

**Result: `R = 0.2046`** (converged ngrid 12→44; basis-free `½Σ|F|` cross-check identical; ENTIRELY shell-band,
|λ|=1 gives 0.000). The Γ "bracket" was an **artifact of evaluating F at its BZ maximum (Γ)** — the genuine
BZ-integrated shell is ~5× smaller and does **not** bracket:

| oblique | value | M_Z |
|---|---|---|
| δ_r (Perron singlet, **LIVE**) | 0.3384% | **+8.1σ** under |
| + chiral shell **@Γ** (artifact, F at BZ max) | 0.3614% | −1.9σ over |
| + **BZ-integrated** shell (R·0.0230% = +0.0047%) | **0.3431%** | **+6.1σ** (NOT closed) |
| SM tree→pole target | 0.3570% | substrate UNDER-predicts by 3.9% rel |

⇒ the substrate's Q-current vacuum polarization, integrated honestly over the Brillouin zone, predicts the EW
oblique to **~4% relative** (0.343% vs SM 0.357%) → M_Z **+6σ**. The framework's **intrinsic precision floor on
the oblique**. The live single-term δ_r (+8.1σ) stands; the forced next term (BZ shell) improves to +6.1σ but does
not close. **`0.810` is NOT forced by T₃−s²Q** — it was the ratio of the BZ maximum to the BZ average, a
coincidence (we did NOT pattern-match it; we built the integral and it came out 0.205). Robustness: the full
two-propagator bubble (interband, genuine field-theory correlator) gives R≈0.57 but does **not** converge
(exceptional-point ill-conditioning of the non-normal B) and is still **not** 0.810 — no natural definition reaches
it. **LESSON BANKED:** a k-point template at a high-symmetry point over-estimates a BZ integral (Γ = spectral
extremum); an apparent Perron-vs-Γ-shell "bracket" can be an evaluation artifact — integrate before bracketing.

**M_Z is now CLOSED as a research question:** it is the framework's last σ-lever and it bottoms out as a genuine
~few-% substrate-vs-SM oblique residual, as the honest prior predicted. No closure exists without un-forcing the
substrate spectrum.

## Standing audit task
Walk the repo for **every** place a value is fit, pattern-matched, calibrated, or graded
"structural-conditional" rather than a **forced read of the object's spectrum**. Each such place is an
incomplete equation → list it here, with the precise statement of *which equation* and *where it stops being
forced*. (**STARTED 2026-06-30** — first verified instances logged below; audit remains open.)

**▶ SYSTEMATIC OVERCLAIM SWEEP 2026-07-06 → `docs/audits/overclaim_audit_2026-07-06.md` — MOSTLY RETRACTED SAME DAY (the audit over-flagged).** 4 parallel Explore auditors pattern-matched "observation"/"import" vocabulary WITHOUT the repo's rigor framework and produced false positives. Corrections (see the audit doc's RETRACTION header): η_B/β = legitimate linter-sanctioned `channel_select` (channel fixed structurally BEFORE observation; smuggle already reframed May 2026) — de-gradings REVERTED; Γ_Z/M_Z = user-gated 2026-07-02, already honestly graded SM-reproduction/bridge-conditional — framing REVERTED; "44 vs native <39" = already disclosed in `honest_sigma_count_2026-06-22`; A_s/16-15 = unverified, likely also legitimate. SURVIVOR (partial, scope-clarification not overclaim): named the 3 identification-layer adoptions in the adoption_register. **LESSON: an agent sweep on vocabulary produces false positives — verify against deep history BEFORE acting.** Independent of this sweep, the B2 √g_*/Y_p contradiction STANDS (read from primaries).

### Logged fit-instances (δ=2/9 / generation-splitting; verified 2026-06-30 by two background audits + spot-check)
The recurring soft spot: **δ=2/9 (the generation splitting) reverse-engineered to the observed value and dressed
as derived.** The VALUE 2/9 is forced three ways (Q(1−Q), Wigner-HM, φ·s); but the *splitting itself* is adopted
(the substrate's leading construction is phase-degenerate, δ≡0) and the −70 ppm subleading is OPEN. Instances
where a file presents a *fit* as a *derivation*:
- **`proofs/foundations/delta_dynamical.py:1176,1477`** — hardcodes `target = 2/9`, scans ~10 measures against it
  (most printed as failing: "≠2/9", "that's 1/3 not 2/9"…), selects "harmonic mean = 2/9", prints
  **"VERDICT: the dynamical derivation is SUCCESSFUL."** Target-driven; the "success" is selection-against-target,
  not a forced read. (The HM route gives the VALUE, but the 1559-line search framing is reverse-engineering.)
- **`proofs/_scratch/O_generation_phase_is_born_invariant_not_phase_2026-06-17.py:23-25`** — `assert
  abs(pat[s]/obs[s]-1) < 0.01` against **observed** δ `{0.22227, 0.1102, 0.0744}`, then prints "δ is a forced A4
  Born-invariant." A fit-to-data asserted as an invariant (the −1% tolerance is the tell).
- **`proofs/_scratch/O_generation_angle_from_fiber_eigenvalue_2026-06-18.py`** — self-labeled "VERIFIED (reproduces
  all 3 sector δ) but **NOT DERIVED**"; needs δ "ADOPTED." Honest tag, but lives as a candidate-derivation.
- **`proofs/foundations/V_Ram_Cl6_iso_all_yukawas_2026-05-26.py:153`** — `delta = 2/9` hardcoded into the LIVE
  Yukawa generator (repo flags it "stale artifact to RETIRE"; persists). The down/up δ {1/9, 2/27} are likewise
  empirical extractions (`BR4_cyclic_toeplitz_koide_reframe` G4/G5 = "EMPIRICAL EXTRACTION, NOT a derivation").
- **Route A vs Route B (the generation derivation is two inconsistent objects):** `derive_generation_spectrum.py`
  (the "forced" ∂_N spectrum, ε=2, **not wired into any prediction**) derives a one-parameter SHAPE that cannot be
  set to the leptons; the LIVE masses adopt empirical Koide (ε=√2, δ=2/9). Scope note added to the file
  (2026-06-30). Reconciling them (or retiring Route A's "FORCED MASS" framing) is the open structural item.
- **`predictions/delta_Koide.py`** — graded δ=Q(1−Q)=2/9 as a "[THEOREM] identity of the Koide parametrisation";
  **corrected 2026-06-30** (it is NOT a parametric identity — δ and Q are independent; CAS verifies only the
  arithmetic; value forced 3 ways, splitting adopted, subleading OPEN). No value change.

## Retraction log (so the same mistake is not re-made)
- **2026-06-30:** the lepton e/μ chirality was claimed "forced" as `δ/k* = 2/27` (ratio 29/25). It is a FIT —
  the operator gives κ_e/κ_μ = 4.15, not 1.16. Retracted. Lesson: a number matching the data is *not* a
  derivation; only the object's spectrum producing it is. This file exists because that lesson kept being lost.

#### D3 PRE-REGISTRATION (2026-07-04, committed BEFORE the probe) — the dark-sign characterization lemma
The prior CAS attempt failed by chasing an IMPOSSIBLE target (force the sign "from nothing"). This
pre-registers the honest, closeable target: a CHARACTERIZATION + IMPOSSIBILITY theorem. Probe:
`proofs/foundations/DARK_sign_lemma_D3_2026-07-04.py`. Scope: NO fit; NO mass VALUE moved; sympy-exact.
- **Q-D3:** formalize the settled dark sign as a standalone lemma at its HONEST grade.
- **Claim (to prove, sympy-exact):** the three readings of the first-girth-return dark object
  Σ = α₁/h are the RECIPROCAL TRIPLE `{ r₁ = 1 (fixed-L amplitude), r₂ = 1 − u (rate = 1/mean-length),
  r₃ = 1/(1 − u) (total resolvent) }`, `u = α₁/h`, with `r₂·r₃ = 1` and `r₁ = 1` the geometric mean —
  giving signs {NO-CHANGE, DOWN, UP}. They are three DISTINCT functionals of the SAME object.
- **Corollary (the impossibility):** since the sign varies across the three functionals, it CANNOT be
  forced from "mass = dynamical recurrence" alone — the from-nothing lemma is IMPOSSIBLE (proven, not
  "attempt failed"). DOWN is forced by the framework's INDEPENDENT commitment to **mass = recurrence
  RATE** (reading r₂; committed for the generation/∂_N sector, not chosen for the sign); readings r₁/r₃
  are the amplitude/resolvent functionals the framework does NOT use for mass.
- **Consistency gate:** r₂ reproduces the framework's shipped value `mass × (1 − α₁/h)` (DOWN) exactly.
- **PASS** = the triple + signs + reciprocity proven, the impossibility established, DOWN pinned to the
  rate commitment. **KILL** = the readings do not give three distinct signs, OR r₂ ≠ the shipped form.
- **Grade delivered:** the sign moves from "settled DOWN, standalone lemma OPEN/FAILED" to "settled
  DOWN, formalized as CONDITIONAL-on-the-rate-foundation with the unconditional version PROVEN
  impossible." **Honesty (todo law):** this does NOT claim the sign is forced from nothing (it is not);
  it formalizes a settled result and proves the unconditional claim impossible. Poison: no fit; no value
  moved; the vertex-dark sign (Peskin −1, already forced) is cited, not re-derived.


> **D3 OUTCOME (2026-07-04, `proofs/foundations/DARK_sign_lemma_D3_2026-07-04.py`, ALL PASS; pre-reg
> 57a5e71 before the probe; sympy-exact; NO fit, NO value moved): the standalone formal lemma is
> CLOSED — as a characterization + impossibility, the honest form.** The three readings of Σ=α₁/h are
> the RECIPROCAL TRIPLE `{r₁ = 1 (fixed-L amplitude), r₂ = 1−u (rate), r₃ = 1/(1−u) (resolvent)}`,
> `u = α₁/h`, with **r₂·r₃ = 1 exactly** and r₁ the geometric mean — giving signs **{NO-CHANGE, DOWN,
> UP}** (all sympy-exact). Three DISTINCT functionals ⟹ the sign is functional-dependent ⟹ **the
> from-nothing lemma is IMPOSSIBLE (now a proven no-go, not "an attempt failed").** DOWN is forced by
> the framework's INDEPENDENT commitment to mass = recurrence RATE (reading r₂), which reproduces the
> shipped `mass × (1 − α₁/h)` EXACTLY (consistency gate passed); r₁/r₃ are the amplitude/resolvent
> functionals the framework does not use for mass. **Grade move:** "settled DOWN; standalone lemma
> OPEN/FAILED" → "settled DOWN; FORMALIZED as conditional-on-the-rate-foundation with the
> unconditional version PROVEN impossible." The shared dark sign under m_t/m_b/M_Z/m_W/m_ν is
> HARDENED (fixed by a foundation committed elsewhere, not a per-correction choice), not re-opened.
> Honest per this file's law: the sign stays settled DOWN; what closed is the FORMALIZATION + the
> impossibility of the from-nothing version. Probe-fix disclosed: sympy `summation` returns a
> convergence-guarded Piecewise; the closed form 1/(1−u) on the physical domain 0<u<1 (verified equal
> to the sum) is used downstream — no gate semantics moved.    • CS-0b (IV.7) 2026-07-10: W_INT re-decoration SURVIVES both branches (chiral asymmetry decoration-robust; magnitude convention-dependent ~2.7x; branches differ by exact sign = the O(2) ambiguity is a chirality-sign; graded-blindness intact). File committed unwired; full integration deferred (token throttle). Stopped mid-run (resume later, cheap models): BRIDGE-GEOM, T0-NUCLEAR-2, accretion, LE-1, I2b, X.2.
    • ACCRETION PASS DONE-AND-VERIFIED (2026-07-10): the_net.py §7 (35a3d4f, pure-append 649+/0−) carries map_commutant/map_family/map_null_lemmas + symmetric_compression + fusion_ring/additive_charge_nullity/z2_gradings w/ 16/16 self-test; HK suite + manifest --fast green; a model verification report internal research notes
    • X.2 SWEEP ADJUDICATED (2026-07-10, architect): dossier internal research notes ORDER FORCED: X.2-b FIRST (flat-band flatness: measure-zero directions vs finite solid angle — gates whether the two-fluid ratio theorem is well-posed; the band is anisotropically flat: exact along [100]/[110], 3.29 along [111]) → X.2-a (pure rho_flat(beta)/rho_cone(beta) crossing theorem, native units, NO z-conversion, dual-outcome) → z-conversion stays gated on the clock map (IV.8 owns beta_eff↔T(N); do NOT invent it in X.2 — the A_s frame lesson). Adoption DECLARED not derived: flat band = matter (frozen MG-1c; EXP-C1 prediction-only). HYGIENE: (1) ML3-B print bug — "9.86" = 3.29 inflated 3x by unnormalized direction vector; fix note needed in that file; (2) stale z_eq registered verdict at target_parameters ~:220 + declared Planck value discrepancy (3402±26 on file vs 3387±21) — pin the declared value BEFORE any confrontation. X.2-b ready to launch (1 a model agent) on user approval.
    • T0-NUCLEAR-2 ADJUDICATED (2026-07-10, architect; files landed via cron 9a84398): gate PASSED (dS convention reproduces 3/13 on all reference configs) but VERDICT = KIN-WRONG-WAY booked RAW (B3=19.598, R_kin=7.343 vs mirror mean 3.641, +102% OFF). ROOT CAUSE = DOMAIN INCOMPLETENESS, named: the solver collapsed onto exact 3-body COINCIDENCE (raw dS=20 > certified rung 13) — but the certified ladder lives on stage3a's DISTINCT-CYCLE configurations; the potential at coincidence is UNDEFINED by the framework (2-body file capped it to 3; my pre-reg used it raw; BOTH unforced — pre-reg error #2 owned). Kinetic sector sane (zero-point ~0.40 above coincidence depth, same scale as T0(2)=0.331). CANDIDATE FORCING: coincidence exclusion = exchange statistics — connect to MS-1a's fermion-parity-unique-Z2. ▶ GATED NEXT (not launched): T0-NUCLEAR-3 = (i) DOMAIN ADJUDICATION theorem first (what configuration space does the ladder certify; is coincidence Pauli-excluded by the framework's own statistics), THEN (ii) the 3-body solve on that domain (no cap, no raw-20, by construction). NO third V-hat guess without the domain theorem. RATIO-MISS stays OPEN; E_odd untouched. Verify-wire T0_NUCLEAR2 at next [MECH] pass.
    • LE-1 ADJUDICATED (2026-07-10): LE-1-COMPOSED (milestone III.3 theorem half DONE). S_register(N) <= N*b_edge = N bits; N(0)=1 gives S=0 exactly (purity cross-checked Tr(rho^2)=1); THE COLD-START QUANTIFIED: at u=alpha_1 register entropy saturates ~0.03 bits (5e-4 of envelope at N=60) — the whole sub-critical run is register-cold, not just the boundary. Three premises explicit (clock identification via S1d; register != thermodynamic entropy — LE-2 bridge stays open; T1 purity knife-edge). Honest mid-run fix documented (seed-consistency in check c). Report internal research notes Verify-wire at next [MECH] pass (batch w/ T0_NUCLEAR2).
    • I2b ADJUDICATED (2026-07-10, architect; milestone II.5 -> DONE-WITH-NOTE): THE COMPLETION CONSTRUCTION LANDED — S_d represented on the TRUE word-Fock H_hist (with an algebraic proof the reduced 12-dim propagator CANNOT carry CK generators — factor-of-q obstruction); Toeplitz defect Sum S S* = 1 − P_seed EXACT (0.0 dev) at two truncations ⟹ the framework's algebra is CONSTRUCTED and the external uniqueness imports (EFW critical point; aHLRS geography) attach to OUR OWN OBJECT. THE beta' RESOLUTION: the run's diagonal is SHARPLY KMS at beta_natural = 2*beta_gas = 6.4874417297 (not the pre-pinned beta' = 5.7943); exact relation beta_natural = beta' + h_top ⟹ the pre-pin was the PER-SHELL (degeneracy-weighted) reading, the run is PER-PATH — the two-temperatures family resolved by computation, not fitted. GEOGRAPHY COHERENT: 2*beta_gas > ln2 = above-critical, exactly where aHLRS puts seed-anchored states ⟹ the full picture: the Landauer lock = the unique CRITICAL CK point of our constructed algebra; the run = a specific ABOVE-CRITICAL Toeplitz KMS state at the Born-squared gas temperature. Two temperatures, one algebra, both theorems about a BUILT object. I2 table row [6]: UNDETERMINED -> VERIFIED-AS-TOEPLITZ. Verify-wired (90 entries) w/ T0_NUCLEAR2 + LE1 + CS0b this pass.
    • BRIDGE-GEOM ADJUDICATED (2026-07-10, architect; verify-wired 91st) → K-DEPENDENT + THE ENANTIOMER-BLINDNESS THEOREM ⟹ MIRROR-REQUIRED EXCLUDED BY PROOF; T-LIKE-REFRAME supported; THE BRIDGE PROGRAM (LOCK/T/GEOM) IS EXHAUSTED WITH THREE THEOREMS. Findings: both O(2) branches = exact intertwiners at Gamma/H, BOTH DIE at P/N (the map is a per-sector object; Gamma/H carries it — a design fact for the Fock-level build); M_inv = M_srs exactly, s_inv = −s_srs exactly ⟹ the mirror lattice's problem = the exact conjugate at EVERY k for the whole real O(2) family ⟹ spatial lattice chirality provably invisible to this classification class; the load-bearing bit is TID2-D's TEMPORAL S4-coset flip. The master-closure conjecture ("conjugate orbit requires the mirror lattice") is REFUTED — booked raw. ⟹ II.2's orbit decision RE-HOMES to the Fock-level/phase-bearing construction (= IV.7 CS-0/CS-1's object; also II.4's coupling home): ONE construction now carries M_Z's gate, the ppm channel, AND the connection sector. II.3 stays gated on it. Also: implementation had to build the k-dependent dart bundle from the PHYSICAL embedding (the abstract K4 voltage graph is screw-blind) — 24/24 rotations cross-verified, machine-derived 4_1↔4_3 flip under inversion.
    • CS-1 NAMED INCOMPLETE (2026-07-11): the renormalization/scheme identification for the induced Maxwell coefficient — which power of u is 'tree level' for the induced kinetic term. Until derived top-down, the transverse k² coefficient (pi_2 = −5.0227e-08 at u=α₁, u⁸-scaling) is NOT on a comparable absolute scale to any coupling; NO confrontation with α_EM is permitted. Owner: IV.7/CS-3.
    • FOCK-0 NAMED GATE (2026-07-11, V4): the dart-side/history-space modular conjugation is UNBUILT — the Tomita J of the accreted I2b algebra (the_net.py §7b) in ω_diag at β_natural = 2β_gas, per-sector on the graded Fock layer (§8). The frozen DR-map class (pinned by intertwining modular conjugations) is UNTESTED until it exists (FOCK-0b). Booked as lemma instead: generator-route Hom_A4(dart_rep, F) = {0} exact (projectivity mismatch — reinforces the M-1b linear-intertwiner null; kills the naive 2T-lift on the dart carrier). Conditional on ML-2b TD-limit duality throughout.
    • THE WELD (named incomplete equation, 3rd independent appearance, 2026-07-11): the identification of the history/first-quantized path space H_hist with the field/second-quantized Cl(6) Fock F (species sector dims {1,3,3,1}) is UNDERIVED. It gates M_Z (behind the W2 proven fence), the ppm four (II.4), the vertex cascade's statistics/domain (T0-N-3's D2 NO-EXCLUSION — CAR is extrinsic on darts, not forced), and the per-observable clock rule. Architect hypothesis TO BE FROZEN as A2 (Push 2): the weld = the second-quantization/Fock functor of the run (histories exponentiate into fields; shell n ↔ n-excitation sector; λ_n = n·λ_1 signature). Cite internal research notes + `docs/framework/BOOTCAMP.md` §8.
    • THE EMBER (booked, not adjudicated, 2026-07-11): per-shell/single-shell DR maps EXIST at the solved λ* = c₁/ε = 2.463 (96-dim kernel at shell 1, 0 at shell 2) — the W2 obstruction (FOCK-0d) is only against ONE global rate shared across shells, not against per-shell maps individually. The per-shell/affine clock-relation class = the next architect freeze (A1 = FOCK-0e, running; A2 = Push 2). Cite internal research notes §§5-7. [UPDATE 2026-07-12: A1/FOCK-0e PROVED the history clock linear (c_n = n·c₁, algebraic identity) ⟹ the ember's per-shell rate is DERIVED (λ_n = n·λ₁, structure); the remaining open is A2, the graded MAP itself, not its rate. Cite internal research notes.] [UPDATE 2 (2026-07-12, A2 adjudicated AF-3): the LEVEL-PRESERVING J-pinned functor is a PROVEN dead class (grading-parity mismatch: field J = the antiparticle level-swap 0↔3/1↔2, history J shell-preserving; verified, φ₁=0 forced 0/384). The named incomplete equation is now: the J-compatible GRADED weld class — J-orbit-pair freedom exists at shell 1 ONLY (144/384; the 3rd shell-1-only ember). Candidate for the A2b freeze (user conversation open): the CONJUGATE-PAIR WELD — J pairs the weld Φ with its antiparticle conjugate J_F∘Φ∘J_hist instead of pinning Φ to itself; the tower is pinned by the K-flow at λ_n = n·λ₁. Level-additivity vs level-reversal (naive pair-codomain dies at shell 2) = A2b's lemma-0 to prove. Cite internal research notes + `A2_check_2026-07-12.md`.] [UPDATE 3 (2026-07-12, A2b adjudicated B3): the R/F_bit REVERSAL↔FERMION parity dictionary is REFUTED for the graded class (theorem: pair-block survivor is reversal-even, level-1 is fermion-odd; verified 144→24→0) — for a graded functor, fermion parity = word-LENGTH parity AUTOMATICALLY. The bare conjugate-pair class has exactly 24 real dims of freedom (mapped). Named candidate pin for A2c: level-1 A4-equivariance (an HONEST rep there — the projectivity wall was full-F only; Hom(shell-1→level-1) = 2 complex dims in the tensor-square reading, 3 in the Frobenius reading — both computed). Banked lemmas: L0a (all level-additive self-J welds die beyond shell 1) · L0b (region clocks can never pin the tower; N̂ at rate c₁ is the ONLY exactly-intertwined field clock — explains the shell-1-only embers) · L0c. FLAW NOTE: `_field_algebra_a4_rep`/`spin_lift` (the_net §8, superseded generator machinery) carries a phase ambiguity — scoped, NO adjudicated result routes through it, do NOT reuse without fixing. Cite `A2b_return_2026-07-12.md` + `A2b_check_2026-07-12.md`.] [UPDATE 4 — THE WELD ARC CLOSED (2026-07-12, A2c → C3 + A2d → D2): the pair-block pin fell to a Schur/character-orthogonality theorem (4th obstruction; its J-free justification was inheritance alone + it encoded species pairing as INPUT = D4 tension). In the minimal justified class (grading + level-1 equivariance), THE WELD EXISTS — a mapped ℂP² family over the 3-dim multiplicity space of the 3-irrep in shell-1 (nullity 6 real vs allowance 2; both solver routes; A2c-checker-independent numbers) covering levels 1⊕2⊕3 of F with the conjugate weld exactly at level 2 and generic coupling to every region-clock eigenspace (±ε included). ⟹ THE FINAL NAMED INCOMPLETE EQUATION OF THE ARC: **THE MULTIPLICITY SELECTOR** — what structure selects the weld's direction in the 3-dim multiplicity space. FOCK-2 (numeric confrontation) stays GATED on it: confronting an unselected direction = goal-seek, forbidden. Stopping rule holds: no further weld stations; the selector waits for a theorem-grade source (candidates: the region/modular program = Push 3; the vertex/interaction program). Generation-resemblance of the 3-space = outlook only. Cite `A2c_return/check_2026-07-12.md` + `A2d_return_2026-07-12.md`.]
    • THE W2 THEOREM'S NAMED FREEDOMS (carried, not resolved, 2026-07-11): (i) the K_F↔F identification is by dimension+basis-label match only (Named Residual Freedom #1 in FOCK0d — a differently-principled embedding of a region's local complex-fermion Fock space onto F could change the spectrum and hence the verdict; not excluded by proof); (ii) the C_A-eigenvalue-1/2 mechanism (odd-dimension-region Schmidt pairing forcing exactly one self-dual mode) is verified to machine precision at all 4 A4-orbit-inequivalent 3-edge regions but has not been re-derived as a standalone lemma from first principles. Cite internal research notes §§8/10.
    • THE CONNECTION SCALE BRIDGE (extends the CS-1 NAMED INCOMPLETE bullet above; CS-2 adjudication 2026-07-12, not a duplicate): CS-2 proved the class flip to the linear-μ/Coulomb class (exponent → +1, B_static/B_equal → 2) is a WEAK-COUPLING-REGIME property, not a shape property — swapping the contact vertex for a Coulomb-shape kernel at the inherited deep operating point gives exponent +0.1118 (CLASS-MISS), but the same kernel's exponent climbs monotonically to +0.71 at g=0.3 on the frozen g-grid, never crossing the acceptance band at any scanned g in the deep regime. The scale bridge (which power of u / what normalization makes the connection's coupling O(1)-comparable — same object as CS-1's declined scheme identification) is now the SINGLE named blocker for all four CS-2 faces: the class flip itself, the atomic block (BLOCKED-BY-NAMED-AMBIGUITY), the quantitative E_odd = 0.381876 MeV confront (SIGN-CONSISTENT qualitative only), and any real Δα value. Found identity (machine-checked exact): B_static(g,s) = B_equal(g,s/2) — the ratio and the exponent acceptance criteria are two reads of one B(μ) curve, not independent tests. Cite internal research notes §§2-4.
