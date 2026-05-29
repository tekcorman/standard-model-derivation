# Step 1 verdict — 24-cycle decomposition on H(srs)

**Date:** 2026-05-26
**Probe:** `proofs/flavor/hashimoto_24cycle_decomposition.py`
**Status:** **HONEST NEGATIVE** — Conjecture B falsified in simple form.

## The conjecture under test

W11 Conjecture B: *every length-24 NB cycle on H(srs) decomposes as three girth-10 NB cycles via symmetric difference*, analogous to the m=2 closed-bubble decomposition at length 16 (which holds at **100%**).

If TRUE, this would give the substrate-side derivation of `c_H^(α₁³) = α₁³` for Family-D Route C extension at m=3, supporting the broader α₁³ rep-resolved Family-D mechanism for closing m_e/m_μ Koide-ratio residuals.

## What the computation found

| Quantity | Value | Comment |
|---|---|---|
| Distinct length-24 NB cycles enumerated | 3441 | from BFS on 8³ supercell, starting from center |
| Girth-10 directed cycles enumerated | 196 | from BFS, starting from center cells |
| Tested cycles | 3441 / 3441 | full coverage of enumerated set |
| **Decomposable (≥1 triple)** | **420** | **12.2%** |
| **Undecomposable (0 triples)** | **2523** | **73.3%** |
| Degenerate (repeated undirected edges) | 498 | 14.5% |

For comparison, the **16-cycle decomposition at m=2 gives 100%** decomposability (1344 of 1344, verified by `hashimoto_16cycle_decomposition.py`).

The 24-cycle pattern is qualitatively different: most 24-cycles do NOT arise as three-girth-cycle symmetric differences.

### Topology of the decomposable subset

Of the 420 decomposable cycles, the topology histogram is dominated by:
- Chain topology with pairwise sharings (s_AB, s_BC, s_AC) ∈ {permutations of (2, 1, 0)}: ~3184 decompositions across permutations
- Triple-intersection topology with (s_AB+s_BC+s_AC, s_ABC)=(5, 1): ~344 decompositions
- Symmetric triangle (1, 1, 1) with no triple: only 64 decompositions

The dominant topology is **asymmetric chain (2-edge seam + 1-edge seam)**, NOT a clean symmetric extension of the 16-cycle's 2-edge seam.

## What this falsifies

**Conjecture B (Route-C extension at m=3)** as a straightforward analog of m=2: FALSIFIED.

The 24-cycle space on H(srs) is structurally MUCH RICHER than "three girth cycles glued at seams." The dominant 24-cycles (73%) arise from some other combinatorial mechanism — possibly:
- Girth + 14-cycle compositions (where 14-cycle is the V_cb host per `hashimoto_14cycle_decomposition.py`)
- Higher-order multi-cycle structures (more than 3 component cycles)
- Non-symmetric-difference compositions

The framework's α₁² Route C derivation (m=2 closed-bubble) is **specific to L=16** and does NOT extend cleanly to L=24 = m=3·(g−2).

## Consequences for the α₁³ rep-resolved Family-D extension

**The W6 derivation of c_H^(α₁³) = α₁³ via Route H extension to length 24 = 3(g−2) is no longer supported on the Route-C-extension side.** It's still possible the joint walker at length 24 (Route H) gives α₁³ via a 3-way joint walker mechanism, but:

(a) The framework currently nominates only ONE Sunada-cospectral dark-sector partner for srs (which is srs-z, per R-9 closure). A 3-way joint walker would require a THIRD partner from {srs-c4, srs-c8, srs-c27} or elsewhere — not currently established.

(b) Without Route C or Route H supporting c_H^(α₁³) = α₁³, this conjecture is structurally **unsupported**.

**Without c_H^(α₁³)** there is no trivial-rep Yukawa cancellation at α₁³, and the W6/W5 mechanism for closing m_τ at α₁³ = 0 also has no support.

## Honest position on the full α₁³ extension proposal

| W11 Conjecture | Status after Step 1 |
|---|---|
| **A**: c_F_amp^(α₁³)_rep_j = −α₁³/μ_rep_j | unsupported (no derivation; would need its own substrate mechanism even if B held) |
| **B**: c_H^(α₁³) = α₁³ via m=3 closed-bubble | **FALSIFIED via Route C extension** (12.2% rate, not 100%) |
| **C**: Trivial-rep Yukawa cancellation | depends on A + B — both unsupported |
| **D**: ω/ω̄ asymmetry mechanism | undrived (separately conjectural) |

## The remaining theorem-grade content

The two RIGOROUS theorems from W11 stand:

**Theorem 1 (Born rule action).** $\delta m/m = 2\cdot(\delta amp/amp)$ for real $|\varepsilon|\ll 1$. Trivial calculus.

**Theorem 2 (Non-factorization).** Rep-dependent corrections cannot be re-absorbed into rep-universal coupling-level coefficients; must act at amplitude level. Trivial proof by contradiction.

These remain useful structural meta-theorems but do NOT close the α₁³ rep-resolved Family-D mechanism.

## What's true about the data

The numerical observation stands:
- m_μ Koide residual is +60.5 ppm = +α₁³ (Family-D-α₁²-extrapolated form, A_mass=2)
- m_e Koide residual is +70.3 ppm = +α₁³ + ω/ω̄-asymmetric piece
- m_τ residual is −13 ppm = α₁⁴-scale, within master doc §8b ~0.5% Yukawa systematic budget

These are real residual patterns. But they do NOT have a theorem-grade substrate derivation, and the proposed α₁³ Route C extension is FALSIFIED.

## What this means for next steps

1. **The α₁³ rep-resolved Family-D extension as I formulated it is structurally unsupported.** The numerical pattern that matched m_μ at 98% has no theorem-grade mechanism.

2. **No predictions/ changes can be proposed** based on the W4-W10 sketches — they were grounded in a falsified conjecture.

3. **Alternative mechanism search** would be needed to close m_e/m_μ Koide ratios at sub-percent precision. Candidates include:
   - 14-cycle composition mechanism (girth + V_cb-host cycle)
   - Berry-phase Family-A direct application (not via the proposed α₁³ rep-resolution)
   - C³_gen / Need-D-3 substrate derivation of y_e, y_μ separately (same wall as light quark masses)
   - Honest acknowledgment that the residuals are within ~0.5% Yukawa systematic budget per master doc §8b, with no specific α₁³ explanation

4. **The honest grade** of m_e/m_μ predictions remains what it was BEFORE this session: mathematically-complete-conditional with un-derived sub-leading Feshbach analog (master doc §8b). The α₁³ extension I proposed does not improve this grade.

## What was achieved by this session

- Identified that the m_e and m_μ Koide-ratio residuals have a **per-C₃-rep structure** not captured by the prior `predictions/m_e.py` "δ-lever-amplified" annotation (which is genuinely falsified by the m_μ data being comparable rather than 14× smaller).
- Established two rigorous meta-theorems (Born rule action + non-factorization).
- Tested and FALSIFIED the natural α₁³ Route C extension (m=3 closed-bubble = three girth cycles).
- Confirmed the predictions DAG is correct to leave unchanged.

The user's discipline — predictions only via the linter, no overclaim — is validated by this result. The α₁³ extension that looked promising at sketch grade does NOT survive rigorous structural test.
