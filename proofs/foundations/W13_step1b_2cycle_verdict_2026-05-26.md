# Step 1b verdict — 24-cycle 2-cycle decomposition

**Date:** 2026-05-26
**Probe:** `proofs/flavor/hashimoto_24cycle_2cycle_decomp_2026-05-26.py`
**Status:** **HONEST NEGATIVE** — no clean 2-cycle decomposition captures the 24-cycle space.

## Context

After W12 falsified the triple-girth m=3 closed-bubble hypothesis (12.2% decomposition rate), the user suggested testing "girth + 14-cycle composition" as an alternative. The arithmetic showed that 24 = 10 + 14 requires s=0 (disjoint), which is topologically invalid. The closest valid 2-cycle compositions are:

| Configuration | L_A + L_B − 2s | Topologically valid? |
|---|---|---|
| girth + 16-cycle, s=1 | 10 + 16 − 2 = 24 | ✓ |
| two 14-cycles, s=2 | 14 + 14 − 4 = 24 | ✓ |
| 14-cycle + 16-cycle, s=3 | 14 + 16 − 6 = 24 | ✓ |
| two 16-cycles, s=4 | 16 + 16 − 8 = 24 | ✓ |
| girth + 14-cycle, s=0 | 10 + 14 − 0 = 24 | ✗ (disjoint) |

## What was tested

`hashimoto_24cycle_2cycle_decomp_2026-05-26.py` enumerated 16884 distinct undirected length-24 NB cycles on H(srs) (in an 8³ supercell). For each, tested whether any of the four valid 2-cycle decompositions yield the cycle as a symmetric difference of two component NB cycles of the specified lengths sharing exactly the specified seam length.

## Results

| Decomposition | Hits | Rate |
|---|---|---|
| (a) girth + 16-cycle, s=1 | 3687 | **21.8%** |
| (b) two 14-cycles, s=2 | 148 | 0.9% |
| (c) 14-cycle + 16-cycle, s=3 | 1777 | 10.5% |
| (d) two 16-cycles, s=4 | 3529 | 20.9% |
| **ANY of (a)–(d)** | **6959** | **41.2%** |
| (previous) triple-girth at m=3 | (3441 sample) | 12.2% |

## Reading

Even combining all natural 2-cycle decompositions covers only 41% of length-24 NB cycles on H(srs). The 24-cycle space is **structurally richer** than the 14-cycle and 16-cycle spaces, both of which decompose at 100%.

The "girth + 16-cycle, s=1" is the best single 2-cycle decomposition at 21.8% — but still far from theorem-grade structural identification.

## What this means cumulatively

**The α₁³ rep-resolved Family-D extension has NO substrate mechanism among the natural cycle-decomposition candidates tested.** Both the triple-girth (W12) and the 2-cycle (W13) decomposition hypotheses are falsified at the structural level.

Specifically, the framework's m=2 closed-bubble derivation of c_H^(α₁²) = α₁² **does not extend cleanly to m=3 = α₁³**. The 16-cycle and 14-cycle "two girth glued at 2- or 3-edge seam" patterns are SPECIAL — they don't generalize to a clean 24-cycle = "k cycles glued" pattern.

## Possible remaining paths (unexplored in this session)

1. **Higher-order multi-cycle decompositions** (4+ cycles): combinatorially expensive; likely captures more of the 24-cycle space but at the cost of structural simplicity.

2. **Cycle-EMBEDDING decompositions** rather than cycle-COMPOSITION: the 24-cycle might admit a girth-10 sub-walk (not sub-cycle) plus a 14-step "exterior" walk, but these aren't standard symmetric-difference decompositions.

3. **Spectral approach**: compute `tr(B^L)` traces directly to characterize the 24-cycle count in terms of spectral data — but this gives counts, not decompositions.

4. **Abandon the α₁³ Family-D extension entirely**: accept that the framework's substrate machinery at α₁² order is genuinely SPECIAL (m=2 closed-bubble is the unique clean structure), and the α₁³ rep-resolved correction lives in some OTHER mechanism family (Family A Berry-phase direct application, Family C counting at sub-leading order, or research-level closure via C³_gen / Need-D-3).

## Honest cumulative position after Steps 1 + 1b

The α₁³ rep-resolved Family-D mechanism for closing m_e and m_μ Koide-ratio residuals is **structurally unsupported**. The numerical pattern that matched m_μ at 98% via (2/μ_rep)·α₁³ remains a real empirical observation, but it has **no theorem-grade substrate derivation** within the framework's existing α₁² Family-D extension protocols.

The framework's existing grade for m_e/m_μ predictions — **mathematically-complete-conditional** with un-derived sub-leading Feshbach analog per master doc §8b — stands and should not be modified.

The W4-W10 work is preserved as exploratory **research notes**, NOT theorem construction. Predictions DAG is correctly unchanged.

## Tasks status update

- W11 honest theorem inventory: established 2 rigorous meta-theorems (Born rule + non-factorization)
- W12 (Step 1): falsified triple-girth m=3 decomposition (12.2%)
- W13 (Step 1b): falsified 2-cycle decompositions (max 21.8% individual; 41.2% combined)
- Remaining: research-level work (multiple sessions) — outside this session's scope
