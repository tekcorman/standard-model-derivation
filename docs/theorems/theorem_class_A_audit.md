# Class A audit — rigor check on the unified spectral dark theorem

**Status:** Audit-grade revision of `theorem_unified_spectral_dark.md`. Identifies two members where the "spectral identification" is a numerical coincidence at k=3 rather than an algebraic unification, and confirms four members where the spectral and non-spectral routes give the same formula in k.

**Written:** 2026-04-28.

## Audit findings summary

| coefficient | spectral formula | non-spectral formula | algebraic identity in k? | status |
|---|---|---|---|---|
| q_NB | (k−1)/k | (k−1)/k (Markov forward/total) | **YES** ✓ | unified |
| α_1_bare | q_NB^(g−2) | q_NB^(g−2) (cumulative Markov) | **YES** ✓ | unified |
| α_1_full | q_NB^(g−2)/(1−q_NB^(g−2)) | same (geometric series) | **YES** ✓ | unified |
| c (= 5/12) | (\|V\|(k−2)+1)/(\|V\|k) | n_g/(\|V\|·k²) where n_g = \|V\|·k(k−2)+k | **YES** ✓ | unified |
| **ε_CP** | **1/(2k−1)** | **(k−2)/(k+2)** Bayesian | **NO** — agree ONLY at k=3 | **coincidence** |
| **A_hemispherical** | inherits ε_CP/k* | inherits ε_CP/k* | inherits ε_CP's caveat | **coincidence** |

**Headline:** 4 of 6 Class A members are algebraically unified across spectral and non-spectral routes (same formula in k). 2 of 6 are numerical coincidences specific to k = 3 — the routes agree at the framework's coordination but would diverge for k ≠ 3.

## Detailed audit

### Member 1: q_NB = 2/3 — UNIFIED

**Spectral:** q_NB = λ_max(B)/λ_max(A) = (k−1)/k for k-regular graphs.
**Markov:** q_NB = (forward choices)/(total choices) = (k−1)/k.

Same formula (k−1)/k. Both give 2/3 at k=3. Both routes agree for ANY k.

Status: ✓ algebraic unification.

### Member 2: α_1_bare = (2/3)^8 — UNIFIED

**Spectral:** α_1_bare = (λ_B/λ_A)^(g−2) = q_NB^(g−2).
**Markov:** α_1_bare = q_NB^(g−2) (cumulative survival over a girth window).

Same formula. Both routes agree for any (k, g).

Status: ✓ algebraic unification.

### Member 3: α_1_full = V_cb = 256/6305 — UNIFIED

**Spectral:** α_1_full = q_NB^(g−2) / (1 − q_NB^(g−2)).
**Geometric series (existing derivation):** α_1_full = α_1_bare / (1 − α_1_bare) = ∑_n α_1_bare^n.

Algebraically the same formula. Spectral framing is a reformulation, not a separate derivation chain.

Status: ✓ algebraic unification.

### Member 4: c = 5/12 — UNIFIED

**Spectral (this session):** c = (2(\|E\|−\|V\|)+1)/(2\|E\|) = (\|V\|(k−2)+1)/(\|V\|·k) for k-regular.
**Cycle counting (existing):** c = n_g/(N_atoms·k²) where n_g = \|V\|·k(k−2)+k for srs (verified numerically).

Algebraically:
$$\frac{n_g}{|V| k^2} = \frac{|V| k(k-2) + k}{|V| k^2} = \frac{|V|(k-2) + 1}{|V| k}$$

Same formula. Both routes give 5/12 at (|V|=4, k=3). For other (|V|, k), both give the same value via this identity.

Status: ✓ algebraic unification.

### Member 5: ε_CP = 1/5 — NUMERICAL COINCIDENCE

**Spectral (this session):** ε_CP = (λ_A − λ_B)/(λ_A + λ_B) = 1/(2k−1) for k-regular.
**Bayesian (existing, Row P28):** ε_CP = (P_+ − P_−)/(P_+ + P_−) where P_+ = 1/2 (binary toggle), P_− = 1/k (uniform among k options). Algebraically: ε_CP = (k−2)/(k+2).

Comparing formulas:
| k | Spectral 1/(2k−1) | Bayesian (k−2)/(k+2) | agree? |
|---|---|---|---|
| 2 | 1/3 | 0 | ✗ |
| **3** | **1/5** | **1/5** | **✓** |
| 4 | 1/7 | 1/3 | ✗ |
| 5 | 1/9 | 3/7 | ✗ |

Algebraic check: (k−2)(2k−1) = k+2 ⇒ 2k² − 5k + 2 = k + 2 ⇒ 2k² − 6k = 0 ⇒ k(k−3) = 0 ⇒ **k = 0 or k = 3**.

The two formulas agree only at k = 3 (and trivially k = 0). For k = 4 (a hypothetical 4-coordinated substrate), the Bayesian route would give 1/3 while the spectral route would give 1/7 — they're predicting different observables.

**Honest reading:** The "spectral identification" of ε_CP is a NUMERICAL COINCIDENCE at the framework's specific k* = 3, not an algebraic unification of the two derivation chains. The Bayesian and spectral routes are computing *different physical quantities* that happen to have the same value at k = 3.

This is meaningfully weaker than the c = 5/12 case where the cycle and spectral routes give the *same formula* in (|V|, k). Calling ε_CP "spectrally derived" overstates the structural over-determination.

Status: ⚠ numerical coincidence at k = 3; primary derivation remains Bayesian (Row P28).

### Member 6: A_hemispherical = 1/15 — INHERITS ε_CP's CAVEAT

A_hemispherical = ε_CP × ⟨(ê·ẑ)²⟩ = ε_CP/k*.

If ε_CP's spectral identification is a numerical coincidence (member 5), then A_hemispherical's spectral content is also a coincidence. The 1/k* factor is structural (Class E-style), but ε_CP's spectral form is the questionable piece.

Status: ⚠ inherits ε_CP's caveat.

## Ihara cross-validation re-examination

The earlier claim: u(k) = u'(k) at λ = k holds only for k = 1 (trivial) and **k = 3**. The framework's k* = 3 is also selected by Brown 1986 information bound. I claimed this as "independent cross-validation".

**Re-examined:** The Ihara identity u(k) = u'(k) at the Perron is genuinely an algebraic identity, not a coincidence:
- u(k) = k − 1 (always)
- u'(k) = (k−1)/(k−2) for k > 2

Setting them equal: k − 1 = (k−1)/(k−2), so (k−1)(k−3) = 0, giving k = 1 or k = 3.

The Brown 1986 argument (Fisher rank gives k* = d = 3) and the Ihara identity (u(k) = u'(k) at k = 3) ARE structurally independent — they use different mathematical machinery (information theory + crystal-net periodicity vs Hashimoto zeta function geometry).

But: I claimed they "converge on k = 3". This is true numerically but I don't have a *deeper structural reason* connecting the two. The convergence could be:
1. **Genuine cross-validation:** independent structural arguments both forcing k = 3 reflect a deeper truth about why k = 3 is special.
2. **Numerical coincidence:** the two arguments happen to give the same answer at k = 3 because of unrelated mathematical features.

Without a structural argument linking Fisher rank to Hashimoto zeta function geometry, I can't distinguish (1) from (2). The honest framing is:

> Two structurally independent arguments (Brown 1986 information bound + Ihara value-gradient merger) both pick k = 3. Whether this is genuine cross-validation or a non-trivial numerical coincidence is open.

This is similar to the ε_CP situation: a non-trivial numerical agreement at k = 3 that could be deeper or could be coincidence. I'll keep the Ihara observation in the unified theorem doc but reframe with honest uncertainty.

## Revised Class A summary

After audit:

| route | members algebraically unified | members coincidentally agreeing at k=3 |
|---|---|---|
| spectral ↔ non-spectral | q_NB, α_1_bare, α_1_full, c | ε_CP, A_hemispherical |

**4 members are algebraically unified** (same formula in k between routes). These are robustly Class A.

**2 members are k=3-specific coincidences** (different formulas in k that agree at k=3). These are still Class A in the *taxonomic* sense (their primary derivations are different — Bayesian for ε_CP) but the spectral route is a coincidence-level identification, not a structural over-determination.

The unified Class A theorem (`theorem_unified_spectral_dark.md`) should be edited to reflect this.

## Recommended edits to `theorem_unified_spectral_dark.md`

1. Move ε_CP and A_hemispherical to a "spectral coincidence at k=3" subsection.
2. Add the table from this audit showing which members are algebraically unified vs coincidentally agreeing.
3. Reframe the Ihara cross-validation as "two independent arguments converge on k=3 — open whether this is structural or coincidence".
4. Reduce the headline claim from "6 framework constants spectrally derived" to "4 spectrally derived (algebraic unification) + 2 coincidentally agreeing at k=3".

## Implications

1. **Class A is still substantive but smaller in rigor than originally claimed.** 4 algebraically unified members is still a major result — α_1_full = V_cb spectrally derived is the most important new finding (CKM mixing as a Hashimoto Perron-power geometric series).

2. **The framework's substrate is still over-determined for c = 5/12** — both routes give the same formula in (|V|, k), and this is the strongest cross-class identity the framework has.

3. **The "Ihara cross-validation of k = 3" stands** but with honest uncertainty about whether it's structural or coincidence.

4. **Ω_DM/Ω_m and ε_CP have been correctly identified as Class D** (Bayesian / max-entropy). My earlier framing "ε_CP is also Class A" was over-stating the over-determination.

## Closure status

- Audit complete.
- Updates needed to `theorem_unified_spectral_dark.md`: revise the headline + add this audit's table.
- Class A master theorem stands at the 4-member level (q_NB, α_1_bare, α_1_full, c=5/12).
- ε_CP and A_hemispherical's primary derivations remain Bayesian (Class D); their spectral coincidence at k=3 is a notable but weaker observation.

The framework's master-theorem coverage now stands at:
- Class A: **4 algebraically-unified parameters** (down from claimed 6).
- Class B: per-coefficient k·p framework; numerical G_sub blocked.
- Class C: no master theorem (taxonomy only).
- Class D: 3 parameters via Bayesian/max-entropy master theorem.
- Class E: 3 parameters via combinatorial master theorem.

Total robust closure: **10 parameters** under master theorems + many more under per-row derivations.
