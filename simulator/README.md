# simulator/ — enumerate-then-MDL-gate architecture

**Status:** pipeline scaffolding wired (2026-05-12) — **S0 axioms** (the
{(A),(B),(I),A5-mass} slate + derived theorems + adoptions + the no-privilege
chain), **S1 menus** (Coxeter Axis A; `crystal_nets` Axis B w/ vendored RCSR
snapshot; vertex/edge algebras; readings; `gauge_tuples` — Tasks A-E gauge zoo,
framework tuple ⟹ SU(4)×SU(2)_L×SU(2)_R = Pati-Salam; `matter` — PS generation
(theorem-grade) + the adopted MSSM extension ≡ R-9's residue), **S2 gating** (mdl + cooling +
`observer` — the Gleason d=3⇒k*=3⇒|E|=3 bridge + (A)⟹arc-transitive⟹Sunada⟹srs;
`waterfilling` — A2-T channel ensembles, post-R-9 degenerate for chiral channels),
**S3** (kernel + observables — framework slice → full physics catalog),
`cayley` (the abstract Coxeter-GROUP graph — NOT the substrate), **`frontier`**
(the ~11 genuine open gaps; R-9 is CLOSED, not among them). The framework
substrate is **srs**, forced structurally (R-9 closure). Remaining: the
option-(c) Axis-B logic absorb (recompute arc-transitivity in-house + absorb
`assess_net`/`dl_comparison`-as-consistency-check/A2-T-waterfilling); the S3
COMPUTE absorb (`proofs/{flavor,masses,gauge,lorentz}` + `lorentz_sig_*` →
`physics/`); `cosmology.py`; the match-layer substrate-source swap; the rename
`simulator/`→`simulator/`.
`channel_select` (Stage-2 MDL selection) replaces the retired `mdl_select`
(argmin). Validation probe:
`proofs/foundations/simulator_validation.py` (316/316).

**Purpose.** The architectural rebuild target for the simulator — enumerate
the candidate space and let MDL gate it, rather than starting from a
hardcoded srs and a curated prediction list. Lives in `simulator/`
(not `simulator/`) so it can be built up without colliding with the live
cherry-picked dominant-slice simulator (`simulator/` + `match/`); the
eventual swap replaces `simulator/` with this once the remaining layers land.

**What's wired so far (`simulator/`, importable, tested):**
- `axioms.py` (S0) — the {(A) self-containment, (B) finite observer, (I) active
  reading, A5-mass} slate (`slate()`); the derived theorems (`derived_theorems()`:
  A1, A2, A3, A4, P1', substrate-agnosticism, ℂ-field-selection, Gleason-d=3);
  the declared adoptions (`adoptions()`: N_hub — the one adopted dimensional input, "which universe" (G_F is a PREDICTION, not an anchor — `axioms.n_hub_pivot()`); MSSM matter ≡ R-9's
  residue); and `no_privilege_consequences()` — the (A)⟹no-privilege chain
  {uniform substrate measure, absent inter-generator commutation, arc-transitive
  ⟹ Sunada ⟹ srs, d=3} that the gating/substrate layers use. A thin faithful
  index of `docs/framework/framework_axioms.md` §10-11 — nothing re-derived.
- `frontier.py` — the simulator's boundary: the ~11 genuine open gaps, each a
  `Gap` (title, blocker, status, residue-register R-N, `proofs/**` clusters,
  what it affects) + a per-gap stub that `raise NotImplementedError` with the
  precise blocker. `list_gaps()` / `get_gap(key)` / `gaps_affecting_load_bearing_content()`.
  (R-9 is NOT here — CLOSED structurally; its residue ≡ the MSSM-as-adoption gap.)
- `menus/coxeter.py` — `enumerate_{finite,affine,hyperbolic,free,full_menu}`
  + `srs_equivalent`. 282-system menu (33 finite |E|=2..8 incl. E_6/E_7/E_8/
  F_4/H_3/H_4; 11 affine; 231 Path-B multi-gen cells; 7 free baselines),
  mirroring `sector_coxeter_*_audit.py` + `sector_path_B_multi_gen_audit.py`.
- `menus/vertex_algebras.py` — Cl(2k,0), Cayley-Dickson ℝ/ℂ/ℍ/𝕆/sedenion/…,
  Tits-Freudenthal magic square. `menus/edge_algebras.py` — Cl(0,p) + 𝕆 edge.
  `menus/fibers.py` — srs high-symmetry k-points (Γ, P-Ramanujan, N, H,
  Γ-cone, P-cone); other Coxeter fiber tables TODO.
- `gating/mdl.py` — Stage 1: `L_elias`, `description_length`,
  `free_word_log_count`, `compression_value`, `freq_factor`, `n_attest`,
  `combined_weight`, `slice_combined_weight` — machine-precision parity with
  `sector_coxeter_freq_weighted_audit.py`. Stage 2: `channel_select`,
  `canonical_encoding` (parity with `simulator/kernel.py`).
- `gating/cooling.py` — `cooling_profile`, `retained_at`, `saturated_zoo`,
  `dominant_slice`, `subdominant_zoo`, `srs_slice`, `cooling_cascade_table`.
- `gating/observer.py` (S2 — the Axis-A↔Axis-B bridge) — `hilbert_dimension()`
  (Gleason 1957 + MDL minimum-cost ⟹ 3), `spatial_dimension()/vertex_coordination()/alphabet_size()`
  (= 3), `isotropy_requirement()` (the (A)⟹arc-transitive⟹Sunada⟹srs chain),
  `condition_coxeter_menu()` (collapse the 282-system menu to the |E|=3 region),
  `condition_crystal_net_menu()` (collapse the net candidates to the unique
  arc-transitive one = [srs]), `conditioned_substrate()` (the d/k*/|E|/srs slice
  + the chain refs). NOT a substrate-only gate — it consumes the substrate-only
  menus and returns the framework's conditioned slice. Probes: `sector_C1_gleason_genericity_audit`,
  `observer_hilbert_space_construction`, `theorem8_*`, the M1-M7 Layer-1-escape audits.
- `gating/waterfilling.py` (S2) — A2-T Boltzmann-weighted channel ensembles.
  `boltzmann_weight(dl) = 2^-dl`, `channel_contributors(channel)`, `channel_ensemble_weights`,
  `waterfilled_value(channel, per_realization_values)`, `lattice_axis_shift(channel)`.
  Post-R-9: chiral-dependent channels {C1,C2,C3,C5,C6} have a SINGLE contributor
  (srs forced) ⟹ zero lattice-axis shift; only C4 (dark/cosmo) is nontrivial —
  ths/dia + the d>3 placeholders (`frontier.d_gt_3_substrates`) + the dim-count
  dark partition contribute at sub-σ weight (only Ω_DM/Ω_m computed: +0.002).
  Absorbs `substrate_lattice_waterfilling_*` + `beta_c1_waterfilling_audit` +
  `substrate_a2t_waterfilling_program.md` (whose §(l) "naive ensemble breaks
  PDG" finding is RETRACTED by the R-9 closure — the substrate is forced, not an ensemble).
- `menus/gauge_tuples.py` (S1 — Tasks A-E gauge zoo) — `GaugeTuple` (substrate,
  vertex-alg, edge-alg) → gauge group. `framework_gauge_tuple()` = (srs, Cl(6,0),
  Cl(0,2)≅ℍ) ⟹ SU(4)×SU(2)_L×SU(2)_R (Pati-Salam), N_attest = 59049 computed
  from the live menus. `enumerate_tuples()` (12), `subdominant_tuples()`,
  `layer1_escape_tuples()` (5: G_2/F_4/E_6/E_7/E_8 vertex algebras — audited
  barren, `frontier.layer1_escapes`), `cooling_cascade_order()`. Mirrors
  `sector_cooling_cascade_audit.py` + `k4_pati_salam_cl8.py` + `srs_so10_embedding.py`.
- `menus/matter.py` (S1) — the matter content. `pati_salam_generation()` =
  (4,2,1)⊕(4̄,1,2) per gen from Cl(6,0) Fock @ the trivalent srs vertex (Wedderburn,
  P3 §4) — theorem-grade; `n_generations()` = 3 from C_3/Galois-ℤ_3 (theorem-grade).
  `mssm_adoption()` = the ONLY adopted piece (the MSSM superpartner content +
  2-loop RG) ≡ R-9's residue (srs-z = the bipartite double cover; "is the cover
  forced?" — Path E blocked, per-sector-β closed-negative, Path E'/M6 open),
  load-bearing for gauge unification. `derived_matter()`/`adopted_matter()`/`is_adopted_matter()`.
  Refs `p3_wedderburn_vertex_classification`, `theorem_generation_C3_bridge`,
  `mssm_matter_content_required`, the adoption register, `frontier.mssm_as_adoption`, `axioms.adoptions()`.
- `zoo.py` — `enumerate_all_slices`, `saturated_zoo`, `dominant_slice`,
  `framework_slice`, `subdominant_zoo`, `cooling_cascade_table`, `_demo`
  (`python -m simulator.zoo`).
- `substrate.py` — `Substrate` dataclass; `framework_default`, `dominant_at`,
  `from_names`, `from_tuple`; Bloch ops delegate to `simulator.srs_substrate`
  for the framework slice (`srs_bridge.py`).
- `cayley.py` — `Cay(W(M), S)` builder (reflection-rep BFS for finite W;
  radius-R ball for infinite W) + `structural_catalog` (|V|, degree, girth,
  diameter, adjacency spectrum, closed-walk counts). ⚠️ This is the abstract
  Coxeter-GROUP graph — a structural-invariants/sanity tool, NOT the framework
  substrate. The framework substrate is a crystal net; see "two axes" below.
- `menus/crystal_nets.py` + `menus/data/` — Axis-B substrate-realization menu.
  Curated `CANDIDATE_NETS`: the 9 V+E-transitive 3-c chiral 3D cubic candidate
  nets (srs, srs-z, srs-c4, srs-c8, srs-c27, lou, lov, okw, hcb-c4) + the
  achiral / hexagonal 3-regular nets the A2-T program references for the
  non-chiral channels (ths, ths-z, eta, utj) + non-3-regular reference nets for
  DL/coordination comparison (qtz, dia, pcu, nbo, …), each with static metadata
  (SG, k*, |V|/|E|, girth, bipartiteness, DL_struct, channel compatibility)
  mirroring `rcsr_candidate_sweep_2026-05-01.md` + `substrate_a2t_waterfilling_program.md`.
  **DATA: vendored** — `data/rcsr_candidates_snapshot.json` is a date-stamped
  parsed snapshot (31 nets; SHA-256 of the source recorded) so per-net
  fingerprints work with NO network/`/tmp` dependency; regenerate with
  `data/_refresh_rcsr_snapshot.py`. **LOGIC: delegated (for now)** — the
  fingerprint/DL computation stays in the mature `proofs/foundations/` apparatus
  (`rcsr_net_assessment.assess_net`, `dl_comparison.py`, …); `crystal_nets`
  loads the vendored data and calls into them via `_backend_*` seams. (`assess_net`
  targets the cubic 3-regular set, so for hexagonal/non-cubic entries the live
  fingerprint isn't computable — graceful: `available: False` + the raw parsed
  entry stands in.) The `_backend_*` seams are the option-(c) absorb point — see
  the ABSORB TARGET block at the top of `crystal_nets.py`.
- `kernel.py` — `CountingKernel(substrate=…)`: MDL primitives delegate to
  `gating.mdl`; counting primitives delegate to the live `simulator.CountingKernel`
  for the framework slice, raise `NotImplementedError` for other slices.
- `observables.py` — Axis A: `all_substrate_outputs(substrate)` — full physics
  catalog (delegated to `simulator.observables`) for the framework slice;
  abstract Coxeter-GROUP-graph invariants + algebra facts + a
  `not_a_spatial_substrate` note (pointing to `menus.crystal_nets`) for any
  other zoo slice. Axis B: `crystal_net_catalog(name)` / `crystal_net_dl_comparison()`
  → the per-net fingerprint / DL comparison via `menus.crystal_nets`.
  `compare_slices(a, b)` tabulates two zoo slices' Coxeter-group-graph invariants.

**TWO SUBSTRATE AXES — and what each one's MDL picks.**

- **Axis A — Coxeter-quotient relation structure** (`menus/coxeter.py` +
  `gating/` + `zoo.py`): which quotient of F_inv(|E|) compresses the toggle
  stream. Raw substrate-only MDL at framework scale picks a HIGH-|E| exceptional
  quotient (`zoo.dominant_slice()` ≈ A_8 = S_9 × Cl(16,0)-region, ~10^61.4) —
  NOT srs / |E|=3 (at N_hub the frequency penalty is inactive for any m ≤ ~30
  system, so Φ dominates and larger |E| compresses more). This reproduces
  `sector_coxeter_full_menu_ranking_audit.py`'s verdict: **k*=3 is observer-
  side, not substrate-only-MDL-dominant on Axis A** — exactly the "skeptical
  bridge probe" finding. The H_3 (|E|=3) slice IS retained (plurally) — ranked
  ~4300 of ~6000 — just not the argmax.

- **Axis B — crystal-net spatial realization** (`menus/crystal_nets.py` →
  the RCSR / `dl_comparison.py` / `substrate_lattice_waterfilling_batch.py` /
  `uniqueness_audit_v2` apparatus): the framework's ACTUAL substrate
  determination. The substrate is **srs** (the (10,3)-a / Laves / K₄ net,
  I4_132, girth 10), forced **STRUCTURALLY** — NOT by a DL tiebreak (**R-9
  CLOSED — STRUCTURAL, 2026-05-12**; see `crystal_nets.framework_substrate_selection()`
  / `observables.substrate_selection()`):
    (A) self-containment ⟹ no privileged spatial direction/edge-orientation
    ⟹ the walker's directed-edge causal state ⟹ the observer's model treats
    all directed edges as equivalent ⟹ the model is strongly isotropic
    (arc-transitive) ⟹ by substrate-agnosticism the SUBSTRATE is strongly
    isotropic ⟹ (Sunada 2012) srs is the UNIQUE strongly-isotropic 3-reg
    3-conn ℝ³ crystal net. With k*=3, d=3 ⟹ srs.
  The DL comparison is now a **consistency check** (srs is also DL-minimum /
  uniquely symmetry-specifiable; the M2a +3.25-bit and γ.2 Wyckoff-x≈0.6607
  refinements are RETRACTED). srs-z is **NOT a competing substrate** — it's the
  bipartite **double cover** of srs (the χ̃/Witten-SUSY-QM layer), so **R-9 ≡
  the MSSM-adoption question** (quotient vs cover). `zoo.framework_slice()` =
  srs × Cl(6,0) × Cl(0,2) ≅ ℍ (Cl(2k,0) at k=3 ⇒ Cl(6,0); G_2 edge-qubit
  theorem ⇒ Cl(0,2)) — what the match layer consumes.

The two axes meet at the observer-side conditioning d_spatial = 3 (Gleason,
the `kernel.mdl_select_hilbert_dimension` primitive) ⇒ vertex coordination
k* = 3 ⇒ |E| = 3, which collapses Axis A's high-|E| argmax onto the |E|=3
region; and within the 3-regular ℝ³ crystal nets the same no-privilege that
gives the uniform substrate measure (= (A)) forces arc-transitivity ⟹ (Sunada)
srs. Both axes' "answer" is observer-side conditioning applied — Axis A makes
that explicit ("k*=3 is observer-side"), Axis B's srs-selection *is* (A)'s
no-privilege applied to spatial labels.

The live simulator (`simulator/` + `match/` on main) is a cherry-picked
dominant-slice computer: it assumes `(srs, Cl(6), Cl(0,2))` and queries
the substrate for outputs. The user-correct target is **enumerate ×
MDL-gate (Stage 1) × channel-select (Stage 2)** where the dominant slice
EMERGES from MDL gating rather than being assumed, and each observable's
value EMERGES from channel-selecting the matching above-waterline candidate
rather than being a curated lookup.

**MDL cleanup status (2026-05-10 → 12, user-led, in progress on main +
`simulator-enumerate-and-gate-skeleton`):** `mdl_above_waterline` (the
Stage-1 threshold) had zero prediction-layer callers as of 2026-05-10 —
being wired now. `channel_select` (the Stage-2 selection) was added
(`a5fc696`, `3c4f41e` on the skeleton branch; `b2824c1` on main) and now
covers all 11 `match/sm_predictions/` files. `mdl_select` (argmin) RETIRED.

## The pattern (already worked out elsewhere)

The framework's existing apparatus has the right shape in several pieces:

- `proofs/foundations/sector_coxeter_freq_weighted_audit.py` —
  Coxeter quotient menu × frequency-weighted MDL gate.
- `proofs/foundations/sector_coxeter_full_menu_ranking_audit.py` —
  ranks the full Coxeter menu at N_hub.
- `proofs/foundations/sector_coxeter_E{2,3,4_to_8}_compressibility_audit.py` —
  per-|E| Coxeter enumeration.
- `proofs/foundations/sector_cooling_cascade_audit.py` — Task D:
  combined-tuple N_attest profile across substrate × vertex × edge.
- `proofs/foundations/sector_path_B_multi_gen_audit.py` — Path B
  multi-generator Coxeter audit.
- Tasks A/B/C/D/E (commits `2c2a624 7748658 a648f98 51edbc8 d5bdc45`,
  2026-05-07) — vertex / edge / combined / cooling / framework-apparatus
  connection.

These are one-off audits scattered in `proofs/foundations/`. The skeleton
proposes consolidating the enumerate-then-gate pattern into the
simulator's core.

## The skeleton

```
simulator/                          (rebuild target on main, replacing current)
├── menus/                          enumerate candidates at each layer
│   ├── coxeter.py                  Coxeter quotient menu (|E|=2..8)
│   ├── vertex_algebras.py          Cl(2k,0), Cayley-Dickson, magic-square Lie
│   ├── edge_algebras.py            Cl(0,p) per p
│   ├── fibers.py                   high-symmetry k-points per Coxeter system
│   └── readings.py                 reading classes R1-R7 + walk classes W1-W10 (+ channel labels)
├── gating/                         MDL waterline (Stage 1) + channel selection (Stage 2)
│   ├── mdl.py                      L(M), Φ(M, N), freq_factor, N_attest, W(M, N) [Stage 1]
│   │                               + channel_select, canonical_encoding [Stage 2 wrappers]
│   └── cooling.py                  cooling cascade across N samples (Stage 1 retention)
├── substrate.py                    Substrate(coxeter, vertex_alg, edge_alg) dataclass
├── kernel.py                       counting primitives + MDL primitives, over a Substrate
├── observables.py                  outputs for a specified slice
└── zoo.py                          full zoo: enumerate × Stage-1-gate at N → retained slices

match/                              consumes simulator outputs; Stage-2 channel_select lives here
                                    (each observable's channel string fixed by its substrate def)
```

The kernel still has its counting primitives; what changes is (a) they take
a `Substrate(coxeter=…, vertex_alg=…, edge_alg=…)` parameter rather than
hardcoding srs, and (b) the MDL primitives finally have callers — Stage 1
(`mdl_above_waterline` / `combined_weight`) filters substrate-menu candidates
at observation length N; Stage 2 (`channel_select`) picks which retained
candidate each observable reads. The full retained zoo EMERGES from
`zoo.saturated_zoo(N_hub)`; the raw-MDL top slice is `zoo.dominant_slice(N_hub)`
(a high-|E| exceptional system — see HONEST FINDING above); the framework's
empirical slice is `zoo.framework_slice()` (srs, via the observer-side bridge).

**Two-stage MDL gating (post-2026-05-12 cleanup):**
- Stage 1 — WATERLINE THRESHOLD: `combined_weight(M, N) ≥ 0` ⇒ candidate
  retained. ALL retained candidates are PHYSICALLY REALIZED — no single
  minimum-cost winner. (`kernel.mdl_above_waterline`, `gating.cooling.retained_at`.)
- Stage 2 — CHANNEL SELECT: for ONE observable, the structural channel
  string (fixed by its substrate definition, BEFORE candidates are
  enumerated) picks the matching retained candidate. K-equivalent matches
  resolve to the min-bit-cost canonical representative. (`kernel.channel_select`.)
- The retired `mdl_select` (argmin over total bit cost) wrongly collapsed
  Stages 1+2 — RETRACTED 2026-05 per
  `feedback_waterline_not_minimum_canonical_distinction`. Kept only for
  backwards-compat of audit checks.

## Acceptance criteria for the rebuild

1. ✅ `simulator.menus.coxeter.enumerate_full_menu()` returns 282
   Coxeter systems (33 finite |E|=2..8, 11 affine, 231 Path-B multi-gen, 7
   free) matching the prior `sector_coxeter_*` + `sector_path_B_*` audits.
2. ✅ `simulator.gating.mdl.{combined_weight, compression_value,
   freq_factor, description_length, max_relation_length, n_attest, L_elias}`
   reproduce `sector_coxeter_freq_weighted_audit.py` at machine precision
   (`simulator_validation.py`, 316/316).
3. ⚠️ REVISED. `simulator.zoo.saturated_zoo(N_hub=8.4e60)` returns
   the full plurally-retained zoo (~6000 slices, Stage 1). The RAW-MDL top
   slice (`zoo.dominant_slice()`) is a HIGH-|E| exceptional system, NOT srs
   — substrate-only MDL prefers larger |E| once the frequency penalty is
   inactive at framework scale (this reproduces
   `sector_coxeter_full_menu_ranking_audit.py`'s verdict). The framework's
   empirical slice `(srs ~ H_3, Cl(6,0), Cl(0,2) ≅ ℍ)` is
   `zoo.framework_slice()` — singled out by the *observer-side bridge*
   (Gleason d=3 ⇒ |E|=k*=3; crystal-net edge-transitivity Sunada-uniqueness;
   Cl(2k,0) at k=3; G_2 edge-qubit theorem), NOT by raw substrate-only MDL.
   The original criterion ("srs is the top-ranked entry") was wrong; what's
   true is that srs's |E|=3 region IS retained (plurally, deep in the zoo)
   and IS the framework slice via the bridge.
4. 🔶 PARTIAL. `simulator.observables.all_substrate_outputs(slice)`
   returns the full physics catalog for the framework slice (delegates to
   `simulator.observables`) and the abstract Coxeter-GROUP-graph invariants (+
   a `not_a_spatial_substrate` note pointing to `menus.crystal_nets`) for any
   other Axis-A zoo slice — it does NOT crash, and does NOT silently return the
   srs catalog. An arbitrary |E| Coxeter quotient simply has no crystal-net
   realization (and the framework's substrate is fixed on Axis B, not by
   ranking Coxeter quotients, so that's fine — see "two axes" above). The
   Axis-B realization candidates (srs, srs-z, …) DO have per-net catalogs:
   `crystal_net_catalog(name)` / `crystal_net_dl_comparison()` bridge to the
   live RCSR / `dl_comparison.py` apparatus. `compare_slices` shows different
   Axis-A zoo slices ⇒ different structural numbers.
5. ⬜ TODO. `match/` migration to consume `zoo.framework_slice()` instead of
   the hardcoded srs; the live `match/sm_predictions/` files already use
   `kernel.channel_select` (Stage 2), so this is a substrate-source swap, not
   a logic change. Must preserve the existing PDG match (373/373 tests).
6. 🔶 PARTIAL. The axioms + counting primitives + MDL primitives + utility
   classes generate everything — for the framework slice the rebuild
   enumerates Axis A (Coxeter-quotient menu), Stage-1-gates it, bridges to
   Axis B (`crystal_nets` → RCSR / `dl_comparison.py` / A2-T waterfilling),
   and exposes the framework slice (= srs × Cl(6,0) × Cl(0,2)) via the
   observer-side conditioning d=3 ⇒ k*=3. Remaining: deeper `crystal_nets`
   ↔ RCSR-probe integration (vendor a parsed snapshot? absorb the fingerprint
   + A2-T logic?) and the `match/` substrate-source swap.

## What stays the same

- The counting primitives (`walk_count`, `orbit_count`, `equiv_class_count`,
  `branch_measure`, `toggle_markov`, `bloch_taylor_at_gamma`).
- The MDL primitives — but now WITH CALLERS: `mdl_above_waterline` (Stage-1
  threshold), `channel_select` (Stage-2 selection), `canonical_encoding`
  (K-equivalent encoding canonicalization). `mdl_select` (argmin) RETIRED.
- The 5 utility modules
  (`SpectralUtility`, `AlgebraicUtility`, `GroupOrbitUtility`,
  `GeometricPhaseUtility`, plus moved `PatiSalamUtility` in `match/`).
- The reading classes R1-R7, walk classes W1-W10 — now carrying explicit
  `channel` labels (see `menus/readings.py`).
- The match-layer API (V_us, m_τ, etc. still computable, still match-tested
  against PDG) — all 11 `match/sm_predictions/` files now enumerate
  above-waterline candidates + channel_select.
- The 11 validation probes (they call into the framework slice via the
  new zoo API once the swap lands).

## What changes

- `simulator.kernel.CountingKernel` no longer hardcodes `SrsSubstrate`.
  It takes a `Substrate` parameter; default is `Substrate.framework_default()`
  (the srs slice, via the observer-side bridge — NOT the raw-MDL argmax
  `Substrate.dominant_at(N_hub)`, which is a high-|E| exceptional system).
- The MDL primitives finally have callers — `simulator.gating.{mdl, cooling}`
  uses `mdl_above_waterline` to threshold every substrate-menu candidate
  (Stage 1); the prediction layer (match package) uses `channel_select` to
  pick the observable-matching candidate from the above-waterline set
  (Stage 2). As of 2026-05-10 these had ZERO callers.
- `mdl_select` (argmin) is RETIRED — it conflated `canonical_encoding` with
  `channel_select` and discarded above-waterline channels.
- `simulator.observables.all_substrate_outputs(slice=...)` parameterizes
  over zoo slices.
- New top-level entrypoint `simulator.zoo.saturated_zoo(N_hub)` returns
  ALL plurally-retained slices at N_hub (Stage 1), not just the dominant one.
- The honest claim becomes verifiable: of the N retained slices in the
  zoo, only ONE produces the SM observable pattern. The other slices
  produce DIFFERENT substrate output catalogs. Whether ANY observed
  physics matches a subdominant slice is a downstream question for
  `match/`.

## Files

- `menus/coxeter.py`           — CoxeterSystem dataclass + enumerators ✅
- `menus/vertex_algebras.py`   — VertexAlgebra dataclass + enumerators ✅
- `menus/edge_algebras.py`     — EdgeAlgebra dataclass + enumerators ✅
- `menus/fibers.py`            — Fiber dataclass; srs fibers ✅, other Coxeter TODO
- `menus/readings.py`          — ReadingClass + WalkClass enums (with channel labels) ✅
- `gating/mdl.py`              — Stage 1: L(M), Φ(M, N), freq_factor, combined_weight,
                                 slice_combined_weight, above_waterline, retained_above_waterline;
                                 Stage 2: channel_select, canonical_encoding ✅
- `gating/cooling.py`          — cooling_profile, retained_at, saturated_zoo, dominant_slice,
                                 subdominant_zoo, srs_slice, cooling_cascade_table ✅
- `substrate.py`               — Substrate dataclass; framework_default / dominant_at / from_names ✅;
                                 Bloch ops delegate to simulator.srs_substrate (framework slice) ✅
- `srs_bridge.py`              — bridge to live simulator.srs_substrate.SrsSubstrate ✅
- `cayley.py`                  — abstract Coxeter-GROUP-graph builder (reflection-rep BFS for
                                 finite W; radius-R ball for infinite W) + structural_catalog
                                 (|V|, degree, girth, diameter, adjacency spectrum, closed-walk
                                 counts); `_demo` (`python -m simulator.cayley`). ⚠️ A
                                 structural-invariants tool, NOT the framework substrate ✅
- `menus/crystal_nets.py`      — Axis-B substrate-realization menu: curated `CANDIDATE_NETS`
                                 (9 chiral cubic + ths/ths-z/eta/utj + non-3-regular reference nets),
                                 static metadata mirroring the RCSR sweep + A2-T docs; vendored data
                                 (`data/rcsr_candidates_snapshot.json`, 31 nets) → rcsr_fingerprint /
                                 dl_comparison call into the live proofs/foundations/ probes via
                                 `_backend_*` seams (option-(c) absorb point). THIN BRIDGE, no network dep ✅
- `menus/data/`                — `rcsr_candidates_snapshot.json` (vendored, date-stamped, SHA-256 of
                                 source recorded) + `_refresh_rcsr_snapshot.py` (regenerator + the
                                 "how to refresh" doc) ✅
- `kernel.py`                  — CountingKernel(substrate=…): MDL primitives → gating.mdl ✅;
                                 counting primitives → live simulator.CountingKernel for the
                                 framework slice ✅, NotImplementedError for other slices
- `observables.py`             — Axis A: all_substrate_outputs(substrate) — framework slice →
                                 full physics catalog (simulator.observables) ✅; other zoo slice
                                 → Coxeter-GROUP-graph invariants + `not_a_spatial_substrate` note ✅.
                                 Axis B: crystal_net_catalog(name) / crystal_net_dl_comparison()
                                 → menus.crystal_nets ✅. compare_slices(a, b) ✅
- `zoo.py`                     — enumerate_all_slices, saturated_zoo, dominant_slice (raw-MDL),
                                 framework_slice (observer-side-conditioned), subdominant_zoo,
                                 cooling_cascade_table, `_demo` (`python -m simulator.zoo`) ✅

Validation: `proofs/foundations/simulator_validation.py` (316/316) — menu
counts vs the prior audits, machine-precision MDL parity vs the freq-weighted
audit, channel_select parity vs `simulator/kernel.py`, Axis-A zoo ranking (incl.
the honest "raw MDL ≠ srs" finding), framework-slice structural counts, kernel
delegation, Coxeter-GROUP-graph invariants (|V|=|W|, |S|-regular, girth analytic↔BFS,
Perron=degree, closed-walks; E_8 build-capped → analytic only; affine truncated),
compare_slices (different zoo slices ⇒ different structural numbers), Axis-A
observables (`not_a_spatial_substrate` note for non-srs slices), Axis-B crystal_nets
menu + vendored snapshot (date-stamped, SHA-256 recorded, ≥30 nets; srs = framework
substrate; srs-z DL ties srs = R-9; channel gating; rcsr_fingerprint sourced from the
vendored snapshot — srs spectrum [-3,-1,-1,-1,1,1,1,3]; graceful for hexagonal nets),
`_refresh_rcsr_snapshot.py` importable. `channel_select` and `mdl_above_waterline`
already had reference implementations in `simulator/kernel.py`; `gating.mdl` mirrors them.

## Two axes — and why `cayley.py` is NOT the substrate

The framework's substrate is fixed on **Axis B (crystal-net realization)**, not
Axis A (Coxeter-quotient relation structure) and definitely not the Coxeter
*group*'s Cayley graph:

- `cayley.py` builds `Cay(W(M), S)` — the Cayley graph of the abstract Coxeter
  GROUP. It's a useful structural-invariants tool (girth, spectrum, closed-walk
  counts of W(M)), but it is *not* a substrate at any level — and srs is NOT
  `Cay(W(H_3), S)` (srs is the (10,3)-a net, an infinite 3-periodic graph with
  girth 10; `Cay(W(H_3), S)` is a finite 120-vertex graph with girth 4).

- The substrate-realization candidate set (Axis B) is the **RCSR crystal nets**
  — the 9 V+E-transitive 3-c chiral 3D 3-regular nets (srs, srs-z, srs-c4,
  srs-c8, srs-c27, lou, lov, okw, hcb-c4) plus the centrosymmetric (ths, dia)
  and d>3 alternatives for the non-chiral channels. That layer is **already
  mature**: `proofs/foundations/rcsr_net_assessment.py` (parse RCSR DB, build
  the nets), `rcsr_per_substrate_fingerprint.py` (per-net fingerprint),
  `rcsr_candidate_sweep.py` (χ̃/bipartiteness sweep), `dl_comparison.py` +
  `srs_vs_srs_z_dl_audit.py` + `qtz_vs_srs_dl_comparison.py` + `lov_dl_audit.py`
  (DL minimization → srs), `substrate_lattice_waterfilling_batch.py` +
  an internal working note (A2-T Boltzmann-
  weighted-MDL gating, channel-by-channel), an internal working note
  (the Row-4 k* / R-9 program). `menus/crystal_nets.py` is the rebuild's thin
  INDEX of / bridge to that layer — it does not reimplement it.

- So there is no "research-level realization gap" in the way an earlier draft of
  this README implied. The framework's substrate-realization candidate set IS
  enumerated, fingerprinted, DL-compared and A2-T-waterfilled. The framework
  substrate is srs — and **R-9 is now CLOSED — STRUCTURAL (2026-05-12)**: srs is
  forced by (A) ⟹ no privileged spatial direction ⟹ arc-transitive substrate
  model ⟹ (Sunada 2012) srs is the unique strongly-isotropic 3-reg 3-conn ℝ³
  crystal net. "Strong isotropy" is (A)'s no-privilege applied to spatial labels
  — derived, not adopted; on par with the no-privilege that gives the uniform
  substrate measure. The DL comparison (srs ties srs-z at 12.17 bits, M2a-only)
  is a *consistency check*, not the selector; the earlier "M2a +3.25-bit
  refinements / γ.2 Wyckoff-x≈0.6607 polynomial encoding close the 2.56-bit gap"
  framing is RETRACTED (the closure doesn't use the data term). The *residue*
  R-9 characterized: srs-z (the bipartite **double cover** of srs) carries the
  Witten-SUSY-QM χ̃ grading — the substrate-level home of the framework's
  adopted SUSY/MSSM structure ⟹ **R-9 ≡ the MSSM-adoption question** (quotient
  vs cover); that's the only thing still open here, and it's a *declared
  adoption* (`menus/matter.py`), not a substrate-uniqueness hole. (Distinct from
  the Axis-A "k*=3 is observer-side" finding, which is *expected* — that's the
  skeptical-bridge result, not a gap.) Front-end derivation:
  `walker_dynamics_derivation.md` Step 4b + `g_girth_derivation.md` Step 2.

So: `observables.all_substrate_outputs` returns the full physics catalog for the
framework slice; for any *other* Axis-A zoo slice (an arbitrary |E| Coxeter
quotient) it returns the abstract Coxeter-GROUP-graph invariants + a
`not_a_spatial_substrate` note — because such a slice has no crystal-net
realization, and that's fine: the framework's substrate is fixed on Axis B, not
by ranking Coxeter quotients. For the Axis-B per-net catalogs (the srs/srs-z/…
realization candidates) use `observables.crystal_net_catalog(name)` /
`crystal_net_dl_comparison()` — which bridge to the live RCSR / `dl_comparison.py`
/ A2-T-waterfilling probes. This is the honest content of "subdominant slices
produce their own catalogs": structurally yes for Axis-A zoo slices (and
`compare_slices` shows it); for the Axis-B realization candidates the per-net
catalogs already exist in `proofs/foundations/` and `crystal_nets` indexes them.

## Remaining work (to finish the rebuild and swap `simulator/`)

1. `menus/crystal_nets` ↔ RCSR-probe integration. **(b) DONE** — a vendored,
   date-stamped parsed snapshot (`data/rcsr_candidates_snapshot.json`, 31 nets,
   source SHA-256 recorded; regenerator `data/_refresh_rcsr_snapshot.py`) so
   Axis-B fingerprints are network-independent; the fingerprint/DL *logic* still
   delegates to `proofs/foundations/` via the `_backend_*` seams. The
   **framing-correction** sub-task of (c) is also DONE — `crystal_nets.py` now
   describes srs as forced *structurally* (R-9 closure: (A)⟹arc-transitive⟹Sunada),
   srs-z as the *double cover* ≡ MSSM-adoption, DL as a consistency check
   (`framework_substrate_selection()`, `arc_transitive` field), 316/316.
   **(c) — now UNBLOCKED (R-9 CLOSED 2026-05-12) and simplified**: absorb (i) the
   arc-transitivity check (recompute it in-house from the parsed nets), (ii) the
   `assess_net` per-net fingerprint computation, (iii) `dl_comparison.py` **as a
   consistency check** (NOT the retracted M2a/γ.2 refinements), (iv) the A2-T
   channel-waterfilling into `simulator`; the `proofs/foundations/`
   probes become thin wrappers. See the ABSORB TARGET block at the top of
   `crystal_nets.py`. (Coordinate with whatever the bg linter is in
   `proofs/foundations/` doing next — post-R-9 it moved to M_R waterfilling.)
2. Other-Coxeter fiber tables in `menus/fibers.py` — only meaningful for slices
   that have a crystal-net realization (Axis B); not needed for arbitrary |E|
   Coxeter quotients (Axis A). srs fibers are wired; the other RCSR nets'
   high-symmetry k-points come from `rcsr_per_substrate_fingerprint.py`.
3. Match-layer migration: `match/` consumes `zoo.framework_slice()` instead of
   hardcoded srs (substrate-source swap; `channel_select` already in place).
4. Then: replace `simulator/` with `simulator/`, repoint the 11
   validation probes + `match/` imports, confirm 373/373.
