# `derivation_topdown/adapters/` — the grafts

An **adapter** is a verification contract, not a new derivation. Each adapter file imports the
one engine (`derivation_topdown/bridge/the_run.py` for the global spectrum, `state/the_net.py`
for the state and the net {A(O)}), and asserts — on the object the framework already built — the
defining axioms of a mature external theory (algebraic QFT, spectral geometry, thermal time,
zeta functions, ...). An adapter adds **zero physics**: no new constant, no refit, no engine
edit. A green contract suite means "this object satisfies these axioms / reproduces this anchor
theorem — run it and see." **The claim is instantiation, not equivalence**: a passing adapter
does not assert the framework's object equals, or is the unique instance of, the external
theory; a failing contract is a booked finding (a discovery about the object), never a bug to be
tuned away by loosening a tolerance or shrinking a region family. Per the Trunk-and-Grafts
Charter (internal research notes) and the Build Ops Protocol
(internal research notes), every adapter's contract list is pre-registered
and frozen before implementation, and `verify.py` gains one backbone entry per adapter at
integration time.

## Ledger (as of 2026-07-09; build increments I1–I6 complete; verify 71/71)

| Suite | Theory | Status | Headline |
|---|---|---|---|
| G4 `aqft_net` | Haag–Kastler / DHR | ✅ GREEN | exact light cone; 62-region duality; sectors ≡ species |
| G2 `furey_stoica_labels` | division-algebra SM | ✅ GREEN | Q = N̂/3 first derived charge; dictionary {ν, d̄, u, e⁺}; generation ⊥ A4 |
| G5a `thermal_time` | Connes–Rovelli | ✅ GREEN | ρ_run = e^(−β_eff·N̂)/Z at 1.9e-22; falsification-probed |
| G1 `sunada_geometry` | Kotani–Sunada | ✅ GREEN | b₁ = 3; BZ ≡ Jacobian; the isotropization weld |
| G6a,b `zeta_gauge` | Bass/Ihara zeta | ✅ GREEN | −log det(I−uW) ≡ loop expansion at 1e-17; + the cover gauge-triviality theorem |
| G3a `ncg_spectral` | Connes KO table | ✅ GREEN | spacetime KO-4 ✓; internal (−1,−1,−1) = KO-6 (exotic presentation, R2b) ✓; 4+6≡2 CONFIRMED |
| G7 `quantum_foundations` | Born / Bell / decoherence | ✅ GREEN | Born exponent = 2 MEASURED (probe-armed, 14–18 order blow-up); CHSH honest-negative (family stated); GKLS + pointer re-expressed |

**Theorems found by this build** (each born from a contract refusing to pass, then adversarially
verified): the **isotropization weld** (G1/SR-4); **cover gauge-triviality** (G6/ZG-3: cell-periodic
signed U(1) is pure gauge on the maximal abelian cover); **holonomy triviality** (D3, station probe:
every cover-closed cycle's Cl(6) holonomy is exactly +I, 192/192 classes) — jointly: *the maximal abelian
cover trivializes all cell-level holonomy; gauge/confinement dynamics is intrinsically finite-k /
non-vacuum / tick-sector.*

**Withdrawn by the pipeline's own checks:** the BW "+7% (1.068×2π)" gravitational-residual
quantification (instrument-limited; G's 2π OPEN, unquantified → D1b); a KO6-FOUND artifact (ordering
bug, caught pre-booking, reversed).

**Named open builds:** D1b (controlled BW read) · D2b (Z_scalar/Z_gauge continuum scaling) · D3b
(finite-k/tick confinement) · G6b′ (finite-k
Wilson bridge). Decisive-wave station probes live in `proofs/foundations/` (D1, D2, D3, R1); all
bookings in `docs/incomplete_equations_todo.md`. (R2b — KO J↔Jγ convention — RESOLVED 2026-07-09,
READ-AS-KO-6; see G3 section below.)

## G1 `sunada_geometry.py`
One-line purpose: verify the srs cover as a Sunada/Kotani-Sunada standard realization (harmonic
equilibrium, Albanese embedding/metric vs the engine's own emergent metric, BZ == Jacobian torus
of K4, and the existence of a heat-kernel scaling limit).

**Status: ✅ GREEN (integrated 2026-07-08; pre-reg cbe42e3; adversarial check PASS-WITH-NOTES).**
SR-0..SR-3 pass at machine precision: **b₁(K4) = 3 == the Z³** (the dimensionality of space IS the first
Betti number); the geometry IS the Kotani–Sunada standard realization (harmonic 5e-16; bond-block isotropy
5e-16; 120° angles; bcc Gram ratio 3.000000000000); the chirality has its geometric seat (C₃ ⟨111⟩ screw,
no improper symmetry); **BZ == the Jacobian torus H₁(K4,ℝ)/H₁(K4,ℤ)** (cotree cycles ↔ deck basis exact;
hashimoto(k) exactly periodic, 2.5e-15). **SR-4 verdict: ISOTROPIZED — the Kotani–Sunada frame is exactly
the frame in which the emergent light cone is isotropic** (g_cart eig [0.500018, 0.500045, 0.500068],
spread 1.0e-4 vs 4:1 fractional; checker: ONLY the k-duality-derived transform isotropizes — the
inverse-conjugates are MORE anisotropic; L used as-is, verdict logic frozen). Isotropic speed 0.70714 ≈
1/√2 = √(v_Hodge·v_adj), consistent with OMEGA_Q0's Albanese dictionary (report-only; the value is tied
to the LCLᵀ=I normalization convention — the ISOTROPY is the invariant claim). **Findings (booked, honest):**
(i) SR-4's g_frac regression at the pre-reg's 1e-6 FAILED at 1.3e-4 — an architect tolerance error (set
below cone_velocity's finite-difference precision; eps-probe confirms the deviation shrinks linearly with
eps ⟹ true eigenvalues {¼,¼,1}); (ii) prose-label correction: the engine's **H1 frame is the true cycle
space** (d₀H1≈0) and **B1 is the coboundary/row-space frame** — prior prose had them swapped; all engine
uses need only complementary orthonormal frames (mislabel, NOT a bug). NOT claimed (SR-5): the heat-kernel
scaling limit / any 2π statement (D1); the flat-band sector geometry.

## G2 `furey_stoica_labels.py`
One-line purpose: verify the Furey-Stoica minimal-left-ideal / Witt-ladder construction on the
framework's own Cl(6) Fock space (charge quantization Q = (1/3)(N1+N2+N3), ideal <-> ML-2 species
sectors, Z3 <-> the deck order-3 element).

**Status: ✅ GREEN (integrated 2026-07-08; pre-reg e09aa0e; adversarial check PASS-WITH-NOTES).**
FS-0..FS-4 pass at frozen tolerances: the engine's forced J6 frame IS a Witt basis (nilpotency 7e-16 —
first explicit check); the 8 ladder states = a minimal left ideal whose N-grading EQUALS the repo's
species projectors (≤2e-15); **Q = N̂/3 — the first charge operator ever DERIVED in this repo (spectrum
{0, ⅓×3, ⅔×3, 1} at 7e-16)** — dictionary under the one global ideal convention {ν, d̄, u, e⁺} (conjugate
ideal = {ν̄, d, ū, e⁻}); color su(3) = ladder bilinears (closure, [T,N̂]=[T,Q]=0, 3 on N=1, 3̄ on N=2).
**FS-5 verdict (dual-outcome, computed): INDEPENDENT/CROSS-CUTTING** — no order-3 A4 element reproduces
the deck's t-cycling (residuals 0.455–0.622, none near 0); σ₃ ∈ A4 as an abstract permutation, yet the
deck operator U_π is exactly Hilbert–Schmidt orthogonal to A4's representation of it ⟹ the generation
triple is a FOURTH mechanism (the winding deck), distinct from Furey-program color/triality. Adjudicated
note: the pre-reg's "γ₅² = I" wording was an architect error — Euclidean Cl(6,0) forces γ₅² = −I
(ω² = (−1)¹⁵, basis-independent); the contract tests the involution with the sign disclosed. NOT claimed
(FS-6): hypercharge/weak isospin from the Fock (ℍ edge-qubit sector, G3+); the conjugate-ideal build;
any triality-based 3-generation claim.

## G3 `ncg_spectral.py`
One-line purpose: verify the noncommutative-geometry spectral-triple axioms on the framework's own
(J6, D, gamma) — including the KO-dimension sign table (J^2, JD-/+DJ, J gamma-/+gamma J) and the
identity of the log-det(I-uW) loop expansion with the discrete Gilkey a4 spectral action.

**Status: ✅ GREEN — G3a/R2/R2b RESOLVED (integrated 2026-07-08; pre-reg 88a9433; adversarial check
FAIL → mechanical fix → re-check PASS; R2b adjudication 2026-07-09) — verdict KO6-FOUND (EXOTIC
PRESENTATION); and G3b/S2 — THE LAGRANGIAN BRIDGE — GREEN (2026-07-09): the certified zeta and the
native spectral action are two reads of ONE machine-checked spectral chain. LB-1 the Bass pencil's
roots ≡ the adjacency spectrum via the certified Ihara map (1.5e-15; the pencil solves the RECIPROCAL
quadratic — a pre-reg citation slip, algebra checker-verified); LB-2 the heat trace reconstructed from
the zeta side ≡ the direct computation (1.5e-12, genuinely disjoint code routes); LB-3 THE WEYL
AMPLITUDE: the cone-sector t^(−3/2) amplitude vs the continuum prediction from the certified Albanese
data — r(t) descends monotonically 1.0199 → 1.0024 over t ∈ [30,240], extrapolation 0.992 ± (within
the declared ±0.02) ⟹ AMPLITUDE-CONVERGENT (honesty clause: an intermediate-t scaling-window
statement on a bounded spectrum, not a true t→0 law); LB-4 the flat band exits as INDEX exactly
(Str ≡ −2 = χ(K4), 1.8e-15); LB-5 β-rows ≡ the engine's b4d exactly. NOT claimed (LB-6): the
self-derivation of the universal Gilkey coefficients (the ζ_{D₄}(0) → Dynkin-row frontier — the
framework's named Type-3 import, still OPEN); any Higgs-potential claim; any true t→0 limit.**
The KO sign table, executed for the first time: KO-0 the Connes table matches m06's in-code table;
KO-1 the spacetime factor re-verified: (−1,+1,+1) → **KO-dim 4** (m06's computation, now a contract).
KO-2/KO-3 the internal Cl(6) Fock: over the frozen candidate sets, the graded pairs with C_ideal/K_g6
leave ε′ UN-FORCED (leaks 0.22/0.29); the particle-hole pair (P_F, σ_M0) — where **σ_M0 is the genuine
a_i ↔ a_i† lift, gated by its defining relation at 2e-15** — gives a cleanly FORCED **(ε,ε′,ε″) =
(−1,−1,−1).**

**INTEGRITY RECORD (preserved verbatim):** the first implementation reported KO6-FOUND; the adversarial
check exposed it as an operator-ordering BUG (the defining relation failed at 2.0), the fix was applied
with the defining relation as sole arbiter, and the corrected arithmetic reversed the verdict — the
pipeline caught a false positive before booking. The corrected signs then stood, honestly, as
ANOMALOUS/KO-OTHER (matching no row of Connes' canonical table) pending a literature-first follow-up.

**R2b adjudication (2026-07-09, internal research notes): READ-AS-KO-6.**
The literature documents a genuine convention freedom for EVEN real spectral triples — J and J′ = Jγ
are both admissible real structures, "perfectly on the same footing" (Dąbrowski–Dossena, Int. J. Geom.
Methods Mod. Phys. 8 (2011) 1833, arXiv:1011.4456, Introduction + Table 1: the second presentation of
n=6 is (−,−,−)); the replacement J → Jγ maps the canonical row to the "exotic" row reversibly (Ćaćić,
Lett. Math. Phys. 2013, arXiv:1209.4832, §2.2 + Table 2.2, column "6−" = (−,−,−)). Connes' own
canonical table (hep-th/0608226 App. 7 Def. 7.2, citing the 1995 J. Math. Phys. paper; van Suijlekom
2024 Tables 3.1/5.1 identical) has ε′=+1 at even n and does not discuss the freedom — our implemented
convention matched Connes' canonical table, which is why the raw computation printed no row under that
convention alone.

**Therefore the internal Cl(6) Fock's forced (−1,−1,−1) IS KO-dimension 6 in the EXOTIC presentation**;
the canonical-presentation partner J′ = J_F·γ_F carries (+1,+1,−1) = Connes' KO-6 row EXACTLY —
verified in-code (`R2b-verify`; matrix computations unchanged, this is the one new computation added).
The internal KO-dimension is 6. **The 4+6≡2 (mod 8) reconciliation is CONFIRMED (R2b, literature-first):**
"CLEANROOM's KO-4" (m06, spacetime), "SM-needs-6" (this file's internal Fock reading, exotic
presentation), and "crown-jewel's KO-2" (the total) are three DIFFERENT, mutually consistent readings
of the SAME Connes anatomy — resolved never by convention-picking, only by the pre-registered authority
hierarchy. NOT claimed (KO-4): first-order condition; full Connes axiom audit; G3b.

## G4 `aqft_net.py`
One-line purpose: verify the Haag-Kastler / DHR axioms of algebraic quantum field theory on the
net {A(O)} built in `state/the_net.py`.

**Contract list (frozen in internal research notes):**
- **HK-0 ANCHORS (regression):** `anchor_cell_projector()` and `anchor_tick_2pi()` both hold.
- **HK-1 ISOTONY:** nested forward diamonds O1 subset O2 (same base, depth d<d') give
  A(O1) subset A(O2) as mode-sets; >=3 base points, depths 1..3, on `Patch(M=4)`.
- **HK-2 EXACT CAUSAL LOCALITY:** {alpha_a(t), a_c^dagger} = (B^t)_ca is identically zero
  strictly below the geometric horizon; T in {1,2,3} on `Patch(M=4)` (mandatory), T in
  {1,2,3,4} on `Patch(M=5)` (optional, time-budget permitting).
- **HK-3 TWISTED (KLEIN) LOCALITY:** even-even commute, even-odd commute, odd-odd anticommute
  (all < 1e-12); naive (untwisted) commutation FAILS (the twist is forced); the Klein-twisted
  commutant commutes (< 1e-12).
- **HK-4 Z^3 COVARIANCE:** B[b,a] = B[T_e b, T_e a] identically (< 1e-13) on interior dart pairs,
  for all three lattice directions e1, e2, e3.
- **HK-5 CELL-LEVEL TWISTED HAAG DUALITY (full family):** for every region R of the 6 cell edges
  with 1 <= |R| <= 5 (all 62 subsets): S(R) = S(R^c) within 1e-8, and the nontrivial
  single-particle modular spectra of R and R^c agree within 1e-8.
- **HK-6 DHR SECTORS == SPECIES:** `gauge_sector_category()` gives species_sector_dims ==
  {0:1, 1:3, 2:3, 3:1}, double_cover_2T == True, sectors_are_species == True, fermion_parity ==
  {0:+1, 1:-1, 2:+1, 3:-1}.
- **HK-7 SCOPE DECLARATION (printed, not computed, never gates PASS/FAIL):** explicitly NOT
  claimed — (i) thermodynamic-limit (infinite-lattice) Haag duality (HK-5 is cell-level only);
  (ii) local DHR charge transporters / braiding at general regions; (iii) past-cone /
  diamond-intersection closure beyond forward diamonds. These remain OPEN.

**Status: ✅ GREEN (integrated 2026-07-08).** All HK-0..HK-6 pass at the frozen tolerances: exact-zero
light cone (0.0 through T=4 on M=5), Z³ covariance 0.0 in all three directions (854 dart-pairs each),
full 62-subset duality family (worst entropy diff 1.1e-10; spectra to 5.3e-15), DHR category
{ν:1, d:3, u:3, e:1} with the 2T double cover. Adversarial check PASS (independent rerun; the HK-5
spectrum mask verified derivation-sound and non-load-bearing — cutoff swept across the full spectral
gap with identical results; thresholds matched to the engine's own self-test; zero smuggles). Pre-reg
e09aa0e; wired into `verify.py` as suite "adapters". HK-7 items remain OPEN as declared.

## G5 `thermal_time.py`
One-line purpose: verify the KMS analyticity condition on the tick state (M0-2R), determine the
von Neumann type of the tick algebra, and check the crossed-product-with-observer construction
against the emergent horizon entropy.

**Status: ✅ GREEN — G5a (the tick-KMS / Connes–Rovelli contract) integrated 2026-07-08 (pre-reg
cbe42e3; adversarial check PASS incl. falsification probe).** KMS-0..KMS-5 pass at frozen tolerances:
the run marginal is exactly geometric (ratio (α₁/u_c)², rel. std 2e-16); the modular generator is AFFINE
IN THE TICK (−log p_n = β_eff·n + c, residual 4e-14) with β_eff = 2·log(u_c/α₁) = 5.1011473686;
**the Gibbs identification ρ_run = e^(−β_eff·N̂)/Z holds at 1.9e-22** — the modular flow of the run state
IS the physical tick flow (a constructive Connes–Rovelli thermal-time instantiation); the two-point KMS
boundary condition w.r.t. the TICK generator holds at 1.2e-13 over the frozen 100-pair observable set —
and the checker's falsification probe confirmed the contract is content-bearing (a non-Gibbs perturbation
fails by ~13 orders of magnitude; wrong-β control fails at 49.0); β·κ = ln 2 re-derived exactly
(symbolic, no floats). NOT claimed (KMS-6): the von Neumann TYPE of the tick algebra (G5b, open); the
crossed-product/observer construction (G5c, open); spatial-region KMS; any TD-limit statement.

## G6 `zeta_gauge.py`
One-line purpose: verify the Bass/Ihara zeta identity on the framework's own non-backtracking
walk B, its matrix-weighted (Cl(6)-holonomy) generalization against the Matsuura-Ohta form, the
Wilson quadratic action as its truncation, and the Polyakov-loop confinement binary.

**Status: ✅ GREEN — G6a,b integrated 2026-07-08 (pre-reg 88a9433; adversarial check PASS incl. the
theorem adjudication).** ZG-1 **the Bass identity holds on our B** (det(I−uB(k)) == the engine's own
`ihara_zeta_inv`, worst 1.4e-15, 360 samples). ZG-2 **cover-girth selection exact**: the k-integrated
moments m_L vanish for L=1..9 (<3.4e-15) and m₁₀ = 120 = 2×10×6 — integer-exact against the Wilson file's
cycle enumeration. ZG-4 **the matter-weighted zeta det(I−uW_INT) computed for the first time**:
scalar-reduction control exact (== det(I−uB)⁸); **the loop-expansion identity −log det(I−uW) =
Σ u^L/L·Tr W^L holds at 1.2e-17** (ρ(W_INT) = √2 exactly) — the "one generating function" statement,
machine-checked. ZG-3 verdict: STRUCTURED-MISMATCH, **upgraded by the checker to a CONFIRMED THEOREM —
maximal-abelian-cover gauge triviality**: every cover-closed walk is null-homologous ⟹ zero net signed
visit on EVERY edge (all 120 girth cycles verified; the cycle-space→Z³ map proven injective ⟹ holds at
ALL lengths; k-integrated response invariant under large random A while per-k traces change by O(10²)) ⟹
**cell-periodic signed U(1) holonomy is pure gauge on the cover; the zeta's zero-momentum gauge response
vanishes at all orders — the physical Wilson/photon bridge is intrinsically finite-k** (named open
contract, feeds D3). Prior-art flag: the original `srs_wilson_action_quadratic.py` docstring says "signed
±1" but its code implements an UNSIGNED per-dart indicator (internally consistent as its own convention;
text/code mismatch booked). NOT claimed (ZG-5): Matsuura–Ohta large-N structure; confinement (⟨P⟩,
holonomy disorder — G6c,d/D3); the a₄ reading (G3b); the finite-k Wilson recovery (open).

## G7 `quantum_foundations.py`
One-line purpose: the quantum-mechanics tie-in — the Born rule as a measured theorem of the run
measure, the first Bell/CHSH read on the net's actual local algebras, and decoherence/pointer
structure as re-expressed theorems.

**Status: ✅ GREEN (integrated 2026-07-09; pre-reg acf6167; adversarial check PASS).**
**QF-1 THE BORN RULE — a THEOREM-CHECK, conditional on A3 (the framework's adopted purification axiom;
conditionality printed every run):** the run measure's exponent is MEASURED as exactly 2 (per-tick ratio
== (α₁/u_c)² at 1.7e-18; the mechanism verified — the Ramanujan modes are orthogonal to the Perron
vector at 4e-16, so the Hermitian norm's square is what survives; modular slope-ratio = 2.000000000000).
The in-contract falsification probe: deforming the exponent to 2.1 breaks the affine/Gibbs
identifications by **14–18 orders of magnitude** (3.2e14 / 7.4e18) against the independently-derived
β_eff — and the checker verified the probe is genuine (the deformed marginal is a valid distribution
whose own best fit is still geometric; only the independent cross-check bites).
**QF-2 CHSH/TSIRELSON — the first Bell read on the object: an HONEST NEGATIVE.** Two disjoint regions
(FAR sep=27, NEAR sep=1) on the Patch Dirac-sea vacuum; the declared 2-plane Majorana-bilinear family;
Wick/Pfaffian correlations validated against dense many-body expectations at 5e-17 BEFORE the sweep
(the mandated two-route check — it caught a sign-convention bug pre-sweep). S_max = 0.094 (FAR) / 0.015
(NEAR): NO-VIOLATION-IN-FAMILY; no Tsirelson breach. NOT evidence of classicality — the checker traced
a structural root cause: the real (flux-free) hopping matrix forces the Majorana covariance's
same-parity blocks to vanish, killing most of the correlation structure available to simple bilinears
(consistent with free-fermion lore: violations need smeared/cleverer operators). ▶ NAMED FOLLOW-UP
QF-2b: smeared/multi-mode observables (and/or complex-flux sectors) for a genuine vacuum-violation
attempt. **QF-3 DECOHERENCE/POINTER:** the derived GKLS form (Choi −2.5e-15), the record-superselection
pointer basis (step-isometry identities at 6e-16), and thermalization to KMS at the predicted rate u_c
(0.5019 vs 0.5000) — re-expressed formula-for-formula from the phase-3/M0-2R proofs. NOT claimed (QF-4):
the measurement problem; A3-independent Gleason; event-separation language; any interpretation.
DISCLOSURE (from the implementation report, recorded here): a wider 3-plane observable family was
explored in scratch during construction (all S ≪ 2); the shipped family is exactly the pre-reg's
declared 2-plane — verified by the checker in code.

**QF-2b (integrated 2026-07-10, pre-reg 13bcb03; check PASS-WITH-NOTES): the smeared Bell read —
DOUBLE-NULL (scoped) + one PROVEN LEMMA + the flux question CLOSED.** F-0 machine-checks the flux
adjudication (Im C == 0 exactly; any added phase = pure gauge or a chosen finite-k sector, by the
repo's own gauge/holonomy-triviality theorems — the complex-flux "fix" is permanently dead; the two
legitimate doors named out-of-scope: a derived J6-compatible patch state, and the finite-k/tick
sector). SVD-of-X_AB smeared modes (knob-free rule), control-first on chain_vacuum(400),
FAR/NEAR/BFS-ball + a separation ladder: NO-VIOLATION-IN-FAMILY everywhere (max S = 1.253 at sep=1),
no Tsirelson breach. **THE LEMMA (checker-proven): the declared shared-pivot family's CHSH tensor is
exactly rank ≤ 1 on ANY real free-fermion covariance — a Clifford-parity fact — so this family can
never exceed S = 2|T₀₀|; the null for this family class is theorem-grade.** Scope: does NOT cover the
natural per-mode family (genuinely rank-2; unexplored — QF-2c leg ii). F-3's quartic pair proven
ALGEBRAICALLY IMPOSSIBLE at 4 Majoranas/region (unique central quartic — a pre-reg design flaw,
owned); the quadratic-obstruction question stays OPEN → QF-2c leg i (r=3 quartics, verified viable).
New permanent instrument: wick_general (2n-point Pfaffian, dense-validated).
