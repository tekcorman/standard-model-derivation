#!/usr/bin/env python3
"""
M7 — f_3 quantification at subdominant substrate retentions |E| ∈ {4, 5, 6, 7}.

Companion to theorem9_f3_quantification_on_srs.py (mechanism M7 of the
revised Theorem 9 §2 access mechanisms list).

Question
--------
Theorem 9's PARTIAL closure argues that at the k*=3 dominant slice, the
MDL-preferred Fano-line embedding sends 3 substrate toggles into a single
Fano line of the seven octonionic imaginary units.  All triple products
within a Fano line associate (the line spans an ℍ ⊂ 𝕆 subalgebra), so
the observer's worldline products stay in ℍ and direct octonion content
is structurally suppressed.

At SUBDOMINANT slices |E| ∈ {4,5,6,7}, the |E| toggles cannot all live
on a single Fano line (Fano lines have exactly 3 points), so AT LEAST
one 3-letter window in any worldline must lie OFF a Fano line and produce
a non-associative associator [a,b,c] ≠ 0.  The Hurwitz theorem moreover
forbids any closed associative subalgebra of 𝕆 of dimensions 5, 6, 7
(only ℝ, ℂ, ℍ at dims 1, 2, 4 are normed-division-associative).

So at |E| ≥ 4 substrate retentions, octonion content is DIRECTLY accessible
to the observer's worldline, with a positive fraction f_3 of 3-letter
windows producing non-associative outputs.

This probe enumerates and tabulates f_3 per |E|.

Method
------
1. Fix the standard Fano plane on imaginary octonion units {1,…,7}
   with seven 3-point lines (Cayley/Lounesto convention).
2. For each |E| ∈ {4,5,6,7}, enumerate all C(7,|E|) embeddings of the
   |E| substrate toggles into 3-of-7 octonion units.
3. For each embedding S, count the number of 3-element subsets T ⊆ S
   that coincide with one of the seven Fano lines (= "associative triples").
4. f_3(S) = 1 − (Fano triples in S) / (total triples C(|E|,3)).
5. MDL-preferred embedding = argmin f_3 (max Fano fraction = max
   associative content = lowest worldline encoding cost in associative
   words).  Compute f_3 at the MDL-preferred embedding for each |E|.
6. Combine with Theorem 8 substrate-side suppression for net access weight
   at framework scale N_hub ∼ 10^60.

Result is purely combinatorial; no new framework structure proposed.

DAG: pure structural audit (companion to mechanism M7 of Theorem 9 §2).
No theorem / prediction / ledger modification.
"""

import math
from collections import Counter
from itertools import combinations


# ----------------------------------------------------------------------------
# Standard Fano plane on octonion imaginary units 1..7
# (Cayley/Lounesto 2001 §23.1 convention)
# ----------------------------------------------------------------------------

FANO_LINES = [
    frozenset({1, 2, 3}),
    frozenset({1, 4, 5}),
    frozenset({1, 6, 7}),
    frozenset({2, 4, 6}),
    frozenset({2, 5, 7}),
    frozenset({3, 4, 7}),
    frozenset({3, 5, 6}),
]

ALL_UNITS = frozenset(range(1, 8))


def verify_fano():
    """Each pair in exactly one line; each point in 3 lines; 7 lines, 28 non-Fano triples."""
    pair_count = Counter()
    point_count = Counter()
    for L in FANO_LINES:
        assert len(L) == 3
        for p in L:
            point_count[p] += 1
        for pair in combinations(sorted(L), 2):
            pair_count[pair] += 1
    assert len(FANO_LINES) == 7
    assert all(c == 3 for c in point_count.values()) and len(point_count) == 7
    assert len(pair_count) == 21 and all(c == 1 for c in pair_count.values())
    total_triples = len(list(combinations(range(1, 8), 3)))
    assert total_triples == 35
    non_fano = total_triples - len(FANO_LINES)
    assert non_fano == 28


# ----------------------------------------------------------------------------
# Per-embedding Fano-line counting
# ----------------------------------------------------------------------------

def fano_lines_inside(subset):
    """Count Fano lines fully contained in subset."""
    S = frozenset(subset)
    return sum(1 for L in FANO_LINES if L <= S)


def f3_for_embedding(subset):
    """f_3 = fraction of 3-element subsets of `subset` that are NOT Fano lines."""
    n = len(subset)
    if n < 3:
        return 0.0
    total = math.comb(n, 3)
    fano_inside = fano_lines_inside(subset)
    non_fano = total - fano_inside
    return non_fano / total


# ----------------------------------------------------------------------------
# MDL-preferred embedding per |E|
# ----------------------------------------------------------------------------

def mdl_preferred_per_E(E_size):
    """
    Enumerate all C(7, E_size) embeddings; report:
      - distribution of Fano-line counts
      - MDL-preferred embedding (max Fano lines = min f_3)
      - all minima (degeneracy class)
    """
    embeddings = list(combinations(range(1, 8), E_size))
    counts = [(emb, fano_lines_inside(emb)) for emb in embeddings]
    max_fano = max(c for _, c in counts)
    min_fano = min(c for _, c in counts)
    mdl_class = [emb for emb, c in counts if c == max_fano]
    distribution = Counter(c for _, c in counts)
    return {
        'E_size': E_size,
        'num_embeddings': len(embeddings),
        'distribution': dict(sorted(distribution.items())),
        'max_fano_lines': max_fano,
        'min_fano_lines': min_fano,
        'mdl_class': mdl_class,
        'f3_mdl': 1.0 - max_fano / math.comb(E_size, 3),
        'f3_worst': 1.0 - min_fano / math.comb(E_size, 3),
        'total_triples': math.comb(E_size, 3),
    }


# ----------------------------------------------------------------------------
# Suppression at framework scale
# ----------------------------------------------------------------------------

def substrate_suppression_log_bits(E_size):
    """
    Theorem 8: free-monoid substrate F_inv(E) gives F_total ~ N · log₂(|E|−1).
    Per-step substrate suppression vs |E|=3 dominant baseline:
      ΔF_per_step = log₂(|E|−1) − log₂(2) = log₂((|E|−1)/2).
    At N_hub ~ 10^60 steps, total bit-cost penalty:
      ΔF_total = N_hub · log₂((|E|−1)/2).
    Returns (per-step, total_at_N_hub) in bits.
    """
    per_step = math.log2((E_size - 1) / 2.0)
    N_hub = 1e60
    total = N_hub * per_step
    return per_step, total


def access_weight(E_size, N_windows=1):
    """
    Direct-octonion-access net weight:
      log₂(weight) = − ΔF_substrate − N_windows · log₂(1/f_3) (proxy for
      effective associator activity at the embedding).
    More carefully: the *direct-access* component scales as f_3 (probability
    a given 3-letter window produces non-associative output), so a single
    window contributes log₂(f_3); total access weight at N windows is
    N · log₂(f_3) − ΔF_substrate.  We report per-window quantities.
    """
    info = mdl_preferred_per_E(E_size)
    f3 = info['f3_mdl']
    per_step, _ = substrate_suppression_log_bits(E_size)
    # Per-window net access bits relative to baseline (log scale, more negative = more suppressed)
    # log₂(f3) is non-positive; substrate cost per_step adds.
    per_window_bits = math.log2(f3) - per_step
    return {
        'f3_mdl': f3,
        'substrate_per_step_bits': per_step,
        'per_window_access_bits': per_window_bits,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    verify_fano()
    print("=" * 95)
    print(" M7 — f_3 quantification at subdominant substrate retentions |E| ∈ {4,5,6,7}")
    print("=" * 95)
    print()
    print(" Standard Fano plane on octonion imaginary units {1,...,7} (Cayley/Lounesto):")
    for L in FANO_LINES:
        print(f"   {sorted(L)}")
    print(f"   Verified: 7 lines, each pair in exactly 1 line, 28 non-Fano triples.")
    print()

    # ----- per-|E| enumeration -----
    print("=" * 95)
    print(" §1. Per-|E| enumeration of embeddings into 7 octonion imaginary units")
    print("=" * 95)
    print()
    print(f"   {'|E|':>4} {'C(7,|E|)':>10} {'C(|E|,3)':>10} {'distribution of Fano-lines-inside':<55}")
    print("   " + "-" * 88)

    results = {}
    for E_size in [4, 5, 6, 7]:
        info = mdl_preferred_per_E(E_size)
        results[E_size] = info
        dist_str = ', '.join(f"{k}:{v}" for k, v in info['distribution'].items())
        print(f"   {E_size:>4} {info['num_embeddings']:>10} {info['total_triples']:>10}   {dist_str}")
    print()

    # ----- MDL-preferred embedding per |E| -----
    print("=" * 95)
    print(" §2. MDL-preferred embedding per |E| (max Fano lines = max associative content)")
    print("=" * 95)
    print()
    print(f"   {'|E|':>4} {'max_Fano':>9} {'min_Fano':>9} {'f_3 (MDL)':>11} {'f_3 (worst)':>12} {'#MDL-class':>12}")
    print("   " + "-" * 75)
    for E_size in [4, 5, 6, 7]:
        info = results[E_size]
        print(f"   {E_size:>4} {info['max_fano_lines']:>9} {info['min_fano_lines']:>9}"
              f" {info['f3_mdl']:>11.6f} {info['f3_worst']:>12.6f} {len(info['mdl_class']):>12}")
    print()

    # Spell out MDL-class for |E|=4 and |E|=5 (small, illustrative)
    print(" Example MDL-preferred embeddings (for inspection):")
    for E_size in [4, 5]:
        info = results[E_size]
        sample = info['mdl_class'][:3]
        print(f"   |E|={E_size}: max={info['max_fano_lines']} Fano-lines-inside; "
              f"first {len(sample)} of {len(info['mdl_class'])} MDL embeddings:")
        for emb in sample:
            inside = [sorted(L) for L in FANO_LINES if L <= frozenset(emb)]
            print(f"     S = {sorted(emb)}  →  Fano lines inside: {inside}")
    print()

    # ----- Monotonicity check -----
    print("=" * 95)
    print(" §3. Monotonicity of f_3 at MDL-preferred embedding")
    print("=" * 95)
    print()
    print("   |E|=3 (dominant): MDL embedding IS a Fano line, f_3 = 0/1 = 0.000")
    print("                    (Theorem 9 PARTIAL closure baseline.)")
    print("   |E|=4..7 (subdominant): tabulated below.  Δf_3 vs |E|=3:")
    print()
    print(f"   {'|E|':>4} {'f_3 (MDL)':>11} {'note'}")
    print("   " + "-" * 80)
    print(f"   {3:>4} {0.0:>11.6f}   single Fano line; ℍ-closure (Theorem 9 PARTIAL)")
    for E_size in [4, 5, 6, 7]:
        info = results[E_size]
        note = ''
        if E_size == 7:
            note = "all 35 triples present; 7 Fano + 28 non-Fano (Lounesto §23.1)"
        elif E_size == 6:
            note = "1 unit excluded; 4 of 7 Fano lines survive (3 killed)"
        elif E_size == 5:
            note = "2 units excluded; 2 of 7 Fano lines survive (5 killed)"
        elif E_size == 4:
            note = "3 units excluded; ≤ 1 Fano line fits in 4 points"
        print(f"   {E_size:>4} {info['f3_mdl']:>11.6f}   {note}")
    print()
    monotone = all(results[a]['f3_mdl'] <= results[b]['f3_mdl']
                   for a, b in zip([4, 5, 6], [5, 6, 7]))
    print(f"   Monotone non-decreasing in |E|: {monotone}")
    print()

    # ----- Cross-check with Lounesto identity -----
    print("=" * 95)
    print(" §4. Cross-check at |E|=7 with Lounesto 2001 §23.1 identity")
    print("=" * 95)
    print()
    info7 = results[7]
    expected_f3_at_7 = 28 / 35
    print(f"   |E|=7: f_3 = (35 − 7) / 35 = 28/35 = {expected_f3_at_7:.6f}")
    print(f"   computed:                                     {info7['f3_mdl']:.6f}")
    assert abs(info7['f3_mdl'] - expected_f3_at_7) < 1e-12
    print(f"   Match: PASS")
    print()

    # ----- Suppression at framework scale -----
    print("=" * 95)
    print(" §5. Suppression at framework scale N_hub ∼ 10^60")
    print("=" * 95)
    print()
    print(" Theorem 8 substrate suppression: F_total ~ N · log₂(|E|−1).")
    print(" Baseline |E|=3 (dominant): per-step cost = log₂(2) = 1 bit.")
    print(" Subdominant |E|: per-step penalty ΔF = log₂((|E|−1)/2) bits.")
    print(" Net per-window direct-octonion-access weight (log₂):")
    print("   per_window_bits = log₂(f_3)  −  ΔF_substrate")
    print()
    print(f"   {'|E|':>4} {'f_3 (MDL)':>11} {'log₂(f_3)':>11}"
          f" {'ΔF/step (bits)':>16} {'per-window net (bits)':>22}")
    print("   " + "-" * 80)
    for E_size in [4, 5, 6, 7]:
        aw = access_weight(E_size)
        lf3 = math.log2(aw['f3_mdl']) if aw['f3_mdl'] > 0 else float('-inf')
        print(f"   {E_size:>4} {aw['f3_mdl']:>11.6f} {lf3:>11.6f}"
              f" {aw['substrate_per_step_bits']:>16.6f}"
              f" {aw['per_window_access_bits']:>22.6f}")
    print()
    print(" Total suppression at N_hub = 10^60 windows (log₂):")
    print(f"   {'|E|':>4} {'ΔF_total (bits)':>20} {'log₂(weight) total':>22}")
    print("   " + "-" * 60)
    N_hub = 1e60
    for E_size in [4, 5, 6, 7]:
        aw = access_weight(E_size)
        delta_total = aw['substrate_per_step_bits'] * N_hub
        # Direct access weight at N_hub (substrate cost only): −ΔF_total bits.
        # Per-step f_3 weight is positive evidence (log₂(f_3) ≤ 0) but per-window;
        # at N_hub windows total log-weight = N_hub * (log₂(f_3) − ΔF_step).
        log_total = N_hub * aw['per_window_access_bits']
        print(f"   {E_size:>4} {delta_total:>20.4e} {log_total:>22.4e}")
    print()
    print(" All four |E|=4..7 subdominant substrates are astronomically suppressed at")
    print(" N_hub ∼ 10^60: net log₂(weight) ≲ −5·10^59 bits ⇒ direct-octonion-access")
    print(" via subdominant substrate is structurally inactive at framework scale.")
    print()

    # ----- Implications -----
    print("=" * 95)
    print(" §6. Implications for Theorem 9 (PARTIAL, non-closing)")
    print("=" * 95)
    print()
    print(" (a) f_3 jumps from 0 (at |E|=3 dominant Fano line) to 0.7500 (|E|=4) and")
    print("     plateaus at 0.8000 for |E|=5,6,7 at MDL-preferred embeddings.")
    print("     Direct octonion content via non-Fano triples is present at ALL")
    print("     subdominant slices |E| ≥ 4.")
    print()
    print(" (b) Within MDL-preferred embeddings (computed in §2):")
    print("       |E|=4:  3/4   = 0.7500   ≤ 1 Fano line fits in 4 points")
    print("       |E|=5:  8/10  = 0.8000   2 Fano lines survive deletion of 2 points")
    print("       |E|=6:  16/20 = 0.8000   4 Fano lines survive deletion of 1 point")
    print("       |E|=7:  28/35 = 0.8000   all 7 Fano lines present (Lounesto §23.1)")
    print()
    print("     The Fano plane's symmetric incidence (each point on 3 lines, each pair")
    print("     on exactly 1 line) makes Fano-line-density invariant under point-")
    print("     deletion in this regime — explaining the |E|=5,6,7 plateau at 0.8.")
    print()
    print(" (c) Substrate suppression:")
    print("     |E|=3: per-step 1 bit (baseline).")
    print("     |E|=4: per-step log₂(3/2) ≈ 0.585 bit MORE than baseline.")
    print("     |E|=7: per-step log₂(3) ≈ 1.585 bits MORE than baseline.")
    print("     At N_hub ~ 10^60, all subdominant substrates carry > 10^59 bit penalty.")
    print()
    print(" (d) Theorem 9 PARTIAL non-closure verified: direct octonion access EXISTS")
    print("     at |E|=4..7 subdominant slices (f_3 > 0), but is structurally suppressed")
    print("     at framework scale by Theorem 8.  This is consistent with the revised")
    print("     Theorem 9 statement: dominant Cl-direct retention preserves room for")
    print("     subdominant 𝕆 retention without making it observable at framework scale.")
    print()
    print(" (e) Mechanism M7 is QUANTIFIED but NEGATIVE for direct framework-scale")
    print("     observability. Subdominant-substrate access remains a Layer-1 escape")
    print("     candidate ONLY in transient (small-N) cosmological regimes — connecting")
    print("     to mechanism M4 (cooling-cascade transients) for joint analysis.")
    print()

    # ----- Honest scope flags -----
    print("=" * 95)
    print(" §7. Honest scope flags")
    print("=" * 95)
    print()
    print(" • This probe is COMBINATORIAL, not dynamical.  f_3 counts FRACTIONS of")
    print("   3-letter triples that lie off Fano lines, given a uniform-walker prior")
    print("   over substrate toggles.  Actual associator amplitudes ([a,b,c] structure")
    print("   constants of 𝕆) are NOT computed; non-Fano triples generate associators")
    print("   of varying magnitude depending on octonion structure constants.")
    print()
    print(" • The Fano-plane labeling (which 7 triples are 'lines') is CONVENTION-")
    print("   DEPENDENT modulo the action of G_2 = Aut(𝕆).  The combinatorial count")
    print("   f_3 is invariant under G_2 (G_2 permutes Fano lines), so the table is")
    print("   convention-independent.  But IDENTIFYING substrate toggles with specific")
    print("   octonion units requires an embedding map that the framework does not yet")
    print("   derive — this is part of the Theorem 9 PARTIAL non-closure: the embedding")
    print("   itself is plurally retained.")
    print()
    print(" • 'MDL-preferred embedding' here means MAX-FANO-CONTENT under the structural")
    print("   argument that associative triples carry lower per-window encoding cost.")
    print("   The ACTUAL bit-cost is not rigorously derived; the heuristic is the same")
    print("   as Theorem 9's load-bearing assumption A1 (assumption-list of the")
    print("   companion probe).  This probe inherits A1; it does not strengthen it.")
    print()
    print(" • Theorem 8 substrate suppression assumes uniform free-monoid F_inv(E)")
    print("   model.  Under Coxeter-quotient saturation (per memory 2026-05-06+2),")
    print("   the substrate menu is broader; the per-step cost log₂(|E|−1) is the")
    print("   FREE upper bound, with quotients giving smaller cost.  The suppression")
    print("   bounds here are conservative (substrate cost is upper-bounded so penalty")
    print("   may be smaller; net log-weight may be less negative than tabulated).")
    print()
    print(" • No Theorem 8 / Theorem 9 / ledger / prediction modifications.  This probe")
    print("   is a closed combinatorial appendix to mechanism M7 of Theorem 9 §2.")
    print()

    print("=" * 95)
    print(" SUMMARY TABLE")
    print("=" * 95)
    print()
    print(f"   {'|E|':>4} {'C(7,|E|)':>10} {'f_3 (MDL)':>11}"
          f" {'ΔF/step (bits)':>16} {'log₂(weight) at 10^60':>23}")
    print("   " + "-" * 80)
    print(f"   {3:>4} {35:>10} {0.0:>11.6f} {0.0:>16.6f} {0.0:>23.4e}")
    for E_size in [4, 5, 6, 7]:
        aw = access_weight(E_size)
        log_total = 1e60 * aw['per_window_access_bits']
        print(f"   {E_size:>4} {results[E_size]['num_embeddings']:>10}"
              f" {aw['f3_mdl']:>11.6f} {aw['substrate_per_step_bits']:>16.6f}"
              f" {log_total:>23.4e}")
    print()
    return 0


if __name__ == "__main__":
    main()
