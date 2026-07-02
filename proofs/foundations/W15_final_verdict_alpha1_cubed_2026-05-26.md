# W15 — Final verdict on α₁³ Family-D Route H extension after spectral/waterline analysis

**Date:** 2026-05-26
**Status:** **HONEST NEGATIVE** — the framework's R-9 closure structurally rules out the 3-way joint walker that the α₁³ extension would require.

## The spectral framing (W14)

After the cycle-decomposition tests (W12, W13) returned negative, I reformulated per user direction: the framework's α₁ is a **spectral expectation over the waterline-included ensemble**, not a single closed-walk count.

In this spectral framing:

```
α₁_bare = q_NB^(g−2)              # 1-substrate walker survival
α₁²    = (q_NB · q_NB)^(g−2)      # 2-way joint walker (srs × srs-z)
α₁³    = (q_NB)^(3(g−2))           # 3-way joint walker survival required
```

For α₁³ Family-D Route H to be theorem-grade by the same mechanism as α₁², we need a **3-way joint NB walker** on three cospectral substrates: srs + two distinct cospectral partners.

## What R-9 closure says (2026-05-12)

an internal working note — **R-9 CLOSES STRUCTURAL** with substrate = srs **alone**.

Key conclusions:

1. **Picture (A) — above-waterline ensemble** is the WRONG interpretation. The framework's `feedback_compressibility_weighting_for_substrate_audit.md` is explicit: superposing alternatives weighted by MDL bit-cost is "misapplication of channel_select." Competing whole-substrate hypotheses don't form an ensemble — they get MAP-selected.

2. **Picture (B) — MAP hypothesis** is the correct picture. The substrate is the unique full-DL-minimal hypothesis, with both `DL_model` (a few bits favoring srs) AND `DL_data|model` (≥ 20,000 bits favoring srs) selecting srs as the unique substrate. The alternatives are DOMINATED, not summed-over.

3. **Cospectral alternatives and their fates** (per R-9 closure table):

| Candidate | Cell | Status | Excluded by |
|---|---|---|---|
| **srs** | 4 atoms | THE substrate | (A) → arc-transitivity → Sunada 2012 picks srs |
| srs-z | 8 atoms (double cover) | rejected | non-minimal cell + not arc-transitive |
| srs-c4 | 4 atoms | rejected | V+E-transitive but NOT arc-transitive |
| srs-c8 | 4 atoms | rejected | same |
| srs-c27 | 4 atoms | spectrally ≡ srs | redundant (same net) |
| hcb-c4 | 8 atoms | rejected | disconnected primitive cell |
| lou/lov/okw | 12 atoms | rejected | non-minimal cell |

## Tension with master doc §3 D

Master doc §3 D Family-D Route H (2026-05-15) was written **after** R-9 closure (2026-05-12) and uses (srs × srs-z) joint walker for c_H = α₁². The interpretation: srs-z is rejected **as a substrate** but appears as a **dark-sector alternative** in the disruption mechanism — these are different roles.

But for **α₁³ Family-D Route H extension** via 3-way joint walker, we'd need a SECOND distinct cospectral partner beyond srs-z. The R-9 closure tells us:
- srs-c4, srs-c8 are V+E-transitive but NOT arc-transitive → structurally distinct in the SAME way (just like srs-z)
- srs-c27 is redundant with srs (same net) → not useful
- hcb-c4 / lou / lov / okw are non-minimal cell → excluded structurally

So potential second partners are {srs-c4, srs-c8}. But:
- All have `q_NB = 2/3` per step (same NB-walker structure as srs)
- All predict the SAME cell-extensive observables as srs (so they're NOT structurally distinct in the dark-disruption sense — that's why srs-z was chosen, being the only one with DIFFERENT cell-extensive predictions via its bipartite double-cover structure)

## The structural blocker

For a 3-way joint walker (srs × srs-z × srs-cX) to provide α₁³ Family-D Route H, we need srs-cX to be **structurally distinct from BOTH srs AND srs-z** in the sense of providing additional dark-sector disruption beyond what srs-z already provides.

But per R-9 closure analysis: srs-c4/c8 are cell-extensively REDUNDANT with srs (predict the same observables). They don't provide additional disruption at the α₁³ level.

**Conclusion: the framework's R-9 closure structurally rules out a 3-way joint walker mechanism for α₁³ Family-D Route H extension.** There is no second distinct cospectral partner available.

## Alternative reading: 2-way at extended length

Numerically, `(q_NB · q_NB)^(3(g−2)/2) = (4/9)^12 = α₁³`. This is 2-way joint walker on (srs × srs-z) at 12 joint steps instead of 8.

But 12 = 1.5·(g−2) is not an integer multiple of (g−2). The Feshbach Exponent Principle specifies (g−2) joint steps as the canonical "girth-cycle interior" length with endpoint pinning n_fixed = 2. Extending to 1.5(g−2) requires going BEYOND the girth-cycle interior — outside the Feshbach principle's domain.

This is a research-level structural question: can the Feshbach Exponent Principle be extended to non-integer multiples of (g−2)? Not currently established.

## Cumulative verdict on the α₁³ rep-resolved Family-D extension

After Steps 1 (W12), 1b (W13), and 1c (W14+W15):

| Layer | Status |
|---|---|
| Born rule factor 2 (W11 Theorem 1) | Rigorous, trivial calculus |
| Non-factorization (W11 Theorem 2) | Rigorous, trivial proof |
| α₁³ form via 3-way joint walker (Route H extension) | **STRUCTURALLY BLOCKED** by R-9 closure |
| α₁³ form via 2-way at extended length (12 steps) | Beyond Feshbach Exponent Principle — RESEARCH-LEVEL |
| α₁³ via Route C 24-cycle decomposition (W12) | FALSIFIED (12.2% rate) |
| α₁³ via 2-cycle 24-cycle decompositions (W13) | FALSIFIED (max 21.8% individual; 41.2% combined) |
| Shape c_F_rep ∝ α₁³/μ_rep at amp level | NO derivation; analogy to α₁² doesn't extend |
| Trivial-rep Yukawa cancellation | depends on c_H — unsupported |
| ω/ω̄ asymmetry mechanism | undrived |

## What's actually established

1. **The m_e and m_μ Koide-ratio residuals are real per-C₃-rep observations** (not single-walk lever amplification as the prior `predictions/m_e.py` annotation claimed).

2. **The framework has NO substrate-level mechanism** at the α₁³ Family-D extension level to close these residuals at theorem grade.

3. **The framework's existing grade** for m_e/m_μ predictions (`mathematically-complete-conditional`, with un-derived sub-leading Feshbach analog per master doc §8b) stands and **should not be modified**.

4. **The predictions DAG is correctly unchanged** through this entire investigation. The original m_e/m_μ predictions remain valid at their stated grade.

## What the proper closure path looks like

To close m_e/m_μ Koide residuals at theorem grade, the framework needs one of:

1. **C³_gen / Need-D-3 substrate derivation** of y_e and y_μ separately (currently the framework only has y_τ; e and μ are Koide-phenomenology completions). Same wall as light quark masses. Multi-session research.

2. **Extension of R-9 closure** to identify a third structurally-distinct cospectral partner, enabling a 3-way joint walker. This would require finding a (k=3, g=10)-class structure that provides ADDITIONAL dark-sector disruption beyond srs-z.

3. **Feshbach Exponent Principle extension** to non-integer multiples of (g−2), enabling 2-way joint walker at length 1.5(g−2). Research-level structural question.

4. **Different mechanism family entirely** — e.g., direct Berry-phase Family A at α₁³ rep-resolved (which `predictions/m_e.py` and Q_Koide.py framework allows in principle).

None of these is within bounded probe scope for this session.

## Honest position

The α₁³ rep-resolved Family-D extension I attempted to construct in W4–W10 was a **sketch that did not survive its first rigorous test** at the structural mechanism level. The numerical match (κ_ω̄ at 98% via 2·α₁³/μ_rep_j) is a real empirical observation, but it has **no theorem-grade substrate derivation** in the current framework.

The user's discipline — predictions only via the linter, no overclaim — is validated by this entire investigation. The dressed-up "theorem-grade" claims I made earlier in the session did not survive rigorous test.

Predictions DAG remains correctly unchanged. m_e and m_μ keep their existing `mathematically-complete-conditional` grade per master doc §8b.
