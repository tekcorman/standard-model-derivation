# Substrate State — T2.1 Live Executor

**Status:** First deliverable complete 2026-04-27. Implements 3 lean ops (involutivity, conjugation, abelianization) as live functions on F_inv(E) word partitions. Verified against analytic counts; surfaces corrections to bookkeeping simulator.

**Code:** `proofs/wave_engine/substrate_state.py`

## What this is

The architectural shift from accountant to executor. Every Φ value the bookkeeping simulator looks up from a template is now COMPUTED via actual partition arithmetic on F_inv(E) configurations.

State holds:
- `classes`: list of equivalence classes; each is a frozenset of words
- `tags`: assumption stack (matches bookkeeping)
- `objects`: derived objects emitted as ops fire
- `Phi_total`, `L_total`: accumulators
- `refinements`: tuple of refinements applied

Each op is a function `op(state) → state` that:
1. Refines the partition by collapsing classes via an equivalence function
2. Computes `Φ_marg = log₂(n_classes_before / n_classes_after)` from the actual class counts
3. Adds the relevant refinement label
4. Emits the appropriate derived object

## Reference scale

The bookkeeping simulator uses `(E=6, n=10)` with class counts ~60M → 11.7M → 977K → 64. The live executor cannot hold 60M words in memory; uses `(E=6, n_max=4)` for verification, where:

- raw = 6⁴ = 1296
- reduced = ?
- cyclic = ?
- abelian = ?

These are computed live, not formula-looked-up.

## Verification result (2026-04-27)

Live cascade trace at E=6, n_max=4:

| step | classes | Φ_marg | bookkeeping classes | Δ% |
|---|---|---|---|---|
| initial (raw words length-N) | 1296 | — | 1296 | 0% |
| after 0.4 involutivity | **781** | 0.731 | 750 | **+4.13%** |
| after 1.8 conjugation (cyclic) | **181** | 2.109 | 165 | **+9.70%** |
| after 1.10 abelianization | **31** | 2.546 | 64 | **−51.56%** |
| **total Φ** | — | **5.39** | (bookkeeping: 4.34) | **+1.05 bits** |

**The live executor disagrees with the bookkeeping simulator at every step.** Three corrections surfaced:

1. **Bookkeeping under-counts reduced classes by ~4%.** Bookkeeping formula `E·(E-1)^(N-1) = 750` counts length-N reduced words only, missing shorter reductions reachable via cancellation. Live count 781 = 750 (length-4 reduced) + 30 (length-2 reduced reachable from length-4 raw) + 1 (empty word). Asymptotic ratio: 25/24 ≈ 4.17%.

2. **Bookkeeping under-counts cyclic classes by ~10%** for the same reason — shorter reductions form valid cyclic classes too.

3. **Bookkeeping over-counts abelian classes by ~2×.** Bookkeeping used `2^E = 64` for the abelian image. But:
   - Abelianization of length-N raw word has parity constraint: total parity = N mod 2. Only `2^(E-1) = 32` even-parity elements appear (at even N).
   - At finite N, some even-parity elements aren't reachable. At n=4, the element `(1,1,1,1,1,1)` requires length ≥ 6 (one of each generator, all odd parity), so unreachable from length-4 raw → 31 reachable, not 32.

## Asymptotic behavior — T2.1h scaling verification (2026-04-27)

Live executor scaled to n_max=7 (~280K words, 0.6s):

| n_max | raw | reduced | cyclic | abelian | time | red ratio (live/bookkeeping) |
|---|---|---|---|---|---|---|
| 4 | 1,296 | 781 | 181 | 31 | 0.00s | 1.0413 |
| 5 | 7,776 | 3,906 | 670 | 32 | 0.01s | 1.0416 |
| 6 | 46,656 | 19,531 | 2,816 | 32 | 0.07s | 1.0417 |
| 7 | 279,936 | 97,656 | 11,830 | 32 | 0.62s | 1.0417 |

**Asymptotic predictions verified:**
- **Reduced ratio → 25/24 ≈ 1.0417** (converged by n_max=6)
- **Abelian → 2^(E-1) = 32** stabilizes from n_max=5 onward (parity-restricted image)
- Cyclic ratio decreases slowly (~1.06 at n=7, still converging)

For the lean cascade at n=10 (bookkeeping reference), live Φ ≈ bookkeeping_Φ + 1 bit. Net for full wave: −134.85 instead of bookkeeping's −135.85. Marginal improvement, but the rigor floor is now established.

**Memory scaling.** n_max=7 ran in 0.62s with ~280K words. n_max=8 (~1.7M words) is feasible in pure Python with patience; n_max=9+ would need numpy/sparse representations. The executor is sufficient for verification at the framework's working scales.

## Why this matters

The bookkeeping simulator's Φ values were *closed-form approximations* — convenient for fast computation but systematically off at finite N (and even asymptotically off for the abelian step). The live executor:

1. **Computes Φ from actual partition arithmetic** — verifiable against any reference standard, no hidden approximations.
2. **Generalizes to non-lean ops** — once we add live implementations of Bloch decomposition, Hashimoto, JW, etc., their Φ values are computed from substrate structure, not template lookup.
3. **Eliminates the L encoding ambiguity at the substrate level** — once ops are explicit functions, L is the bits-of-implementation, mechanically derivable.

This is the foundation T2.1 lays. The lean (3-op) verification proves the architecture works; expanding to all 195 ops is incremental.

## Live ops implemented

### Word-partition layer (first deliverable)

| op | live function | partition refinement |
|---|---|---|
| 0.4 involutive cancellation | `op_0_4_involutive` | `reduced_form(w)` via stack-based cancellation |
| 1.8 conjugation | `op_1_8_conjugation` | `cyclic_class(w)` via lex-min rotation |
| 1.10 abelianization | `op_1_10_abelianization` | `abelianization(w, E)` via parity counts |

Each is ~3 lines of code wrapping a generic `refine_partition` helper.

### Graph layer (T2.1e — added 2026-04-27)

The substrate has a second aspect: the srs primitive-cell graph (4 atoms, 12 directed edges, BCC primitive lattice vectors). It populates a separate `GraphLayer` field on `SubstrateState` and is consumed by the spectral / Bloch / Hashimoto ops.

| op | live function | output |
|---|---|---|
| 4.21 srs quotient | `op_4_21_srs_quotient` | populates `GraphLayer` with primitive-cell connectivity from `proofs/common.py:find_bonds()` |
| 2.15 adjacency at Γ | `op_2_15_adjacency` | `A_at_Gamma` 4×4 + spectrum [−1, −1, −1, 3] (Perron = k* = 3) |
| 2.18 Hashimoto at Γ | `op_2_18_hashimoto` | `B_at_Gamma` 12×12 + spectrum [2, 1×3, −1/2 ×6, −1×2] (Perron = k*−1 = 2) |

**Verifications** (passing in `python3 proofs/wave_engine/substrate_state.py`):
- h_max(Γ) = 2.000000 = k*−1 (matches the Ihara prediction at the Perron eigenvalue, consistent with the high-precision result in `proofs/foundations/lorentz_sig_hashimoto_d4_iso.py`)
- Ihara factorization u² − λu + (k*−1) = 0 verified live for all four adjacency eigenvalues at Γ: each λ produces (u₊, u₋) with u₊·u₋ = 2
- Closed-walk counts Tr(B^n) for n=1..7: `[0, 0, 24, 24, 0, 96, 168]` — Tr(B^3) = 24 matches K_4-quotient triangle counting (4 vertices × 6 NB closed 3-walks per vertex / 3 starting points = 8 distinct triangles × 3 visits = 24)

**Φ correction surfaced.** The bookkeeping simulator's `BLOCH_SRS` template assigns op 2.18 Φ = log₂(8) = 3 bits. The live executor disentangles 2.18 from the Bloch-decomposition compression and computes the per-step NB compression rate from actual walk counts:

| length n | all walks | NB walks | log₂(ratio) |
|---|---|---|---|
| 1 | 3 | 3 | 0.000 |
| 2 | 9 | 6 | 0.585 |
| 3 | 27 | 12 | 1.170 |
| 4 | 81 | 24 | 1.755 |
| 5 | 243 | 48 | 2.340 |
| 6 | 729 | 96 | 2.925 |

Per-step Φ converges to log₂(k*/(k*−1)) = log₂(3/2) ≈ **0.585 bits**, not 3 bits. This surfaces an additional ~2.4-bit overcount per Hashimoto firing in the lookup-table accounting, on top of the ~1-bit correction at the word-partition layer.

## Next deliverables (T2.1 expansion)

- **T2.1d Live Bloch decomposition** (~3 sessions). Implement op 4.17 as actual eigendecomposition of A(k) on the K_4-quotient. Returns per-k fibers; Φ from |input dim|/|max k-fiber dim|. Will reuse `GraphLayer` infrastructure built for T2.1e.
- **T2.1e Live Hashimoto.** ✅ DONE 2026-04-27. Spectrum + closed-walk counts at Γ verified; Φ = log₂(3/2) per step; bookkeeping disentangled from Bloch compression.
- **T2.1f Live JW + Cl(6;ℂ) spinors** (~3 sessions). Implement ops 5.6, 5.7, 5.9 as explicit operator construction. Returns 8-dim spinor data per node.
- **T2.1g Integrate with main simulator** (~2 sessions). Replace `PHI_TEMPLATE` lookup in `simulator.py` with live computation when SubstrateState is sufficiently populated.
- **T2.1h Scale to n_max ≥ 6.** ✅ DONE — n_max=7 runs in 0.62s; asymptotic ratios converge.

Total T2.1 expansion remaining: ~8–10 sessions for live executor across the remaining high-impact catalog ops.

## Connections

- **T1.1 marginal-Φ template dedupe** (`mechanism.md`): bookkeeping simulator already accumulated 95-bit overcount via template-dedupe-naive accounting. T2.1 SURFACES additional ~1-bit corrections from finite-N partition arithmetic that bookkeeping missed.
- **T1.2 formal L encoding** (`cost_methodology.md`): once ops are live functions, L = bits-of-actual-implementation, mechanically derivable. Subsumes the "formal L" question.
- **T2.5 the author's separate private derivation DAG integration**: post-T2.1, each live op's output can be cross-linked to the author's separate private derivation-DAG nodes carrying `information_characteristics`. Observable-side Φ becomes derivable from the author's separate private derivation metadata + framework prediction.

## Caveats

- **n_max=4 is small.** Verification at n_max=4 is sufficient to surface bookkeeping discrepancies, but live numbers don't directly compare to bookkeeping's n=10 reference. Asymptotic extrapolation is the bridge.
- **Memory scaling.** n_max=10 is ~60M words; not tractable in this Python implementation. T2.1h scaling would require either smaller n_max with asymptotic extrapolation, or sparse-representation optimization, or moving compute-intensive parts to C/numpy.
- **Currently 3 of 195 ops** are live. The architectural shift is demonstrated; the bulk of the work is incremental op-by-op implementation.
