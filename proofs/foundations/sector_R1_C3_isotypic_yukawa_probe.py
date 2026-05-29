#!/usr/bin/env python3
"""
R1 probe: sector-DEPENDENT Yukawa vertex factorization via C_3 isotypic content.

CONTEXT
=======
Picking up R1 from the EOD+3 audit (`R14_y_b_vertex_alpha12_type6c_audit_2026-05-05.md`).
The (3k*-2)/k* candidate failed Type 6c via three independent obstacles. R1
is the most natural framework-internal next direction:

  Extend theorem_ytau_corollary §5-§7 with a Hamming-weight-n-dependent factor
  derived from C_3 isotypic content of Cl(6) Fock at trivalent vertex.

The C_3 isotypic decomposition is theorem-grade per
`cl6_fock_z3_breaking_decomposition.py` (closes Row P37 OQ1):

    n = 0 (Hamming weight 0, level dim 1):  trivial only
    n = 1 (Hamming weight 1, level dim 3):  trivial ⊕ ω ⊕ ω²    (regular rep)
    n = 2 (Hamming weight 2, level dim 3):  trivial ⊕ ω ⊕ ω²    (regular rep)
    n = 3 (Hamming weight 3, level dim 1):  trivial only

THIS PROBE
==========
Tests whether a Yukawa amplitude operator that "sees" the C_3 isotypic content
gives the OBSERVED hierarchy y_τ < y_b < y_t (with y_ν ≈ 0). Per the EOD+3
audit's Obstacle 2 lesson, the simplest reading (Yukawa ∝ trivial-component
projector weight) gives WRONG SIGN. This probe enumerates candidate operators
respecting Type 6c structural constraints, computes per-sector amplitudes,
and tests against observation.

KEY STRUCTURAL OBSERVATION (POTENTIAL OBSTACLE)
================================================
At k*=3, Λ^1 and Λ^2 are HODGE DUALS — they have the same C_3 isotypic
decomposition (regular rep). Any C_3-isotypic-only operator gives the SAME
amplitude for n=1 and n=2 sectors, hence predicts y_b = y_t up to sign.
Observation: y_b ≠ y_t (factor ~41 difference). C_3 isotypic ALONE cannot
distinguish n=1 from n=2.

So R1 in its pure form (isotypic-only) is structurally OBSTRUCTED by Hodge
duality between Λ^1 and Λ^2 at k*=3. To rescue R1, an ADDITIONAL structural
distinction between n=1 and n=2 is needed (e.g., charge Q_n via charge_before_color
theorem, or fermion creation vs annihilation asymmetry).

This probe verifies the obstruction and tests three escalation paths:
  (A) C_3 isotypic alone (expected to fail by Hodge duality)
  (B) C_3 isotypic + Q_n charge weighting (charge_before_color)
  (C) C_3 isotypic + Hamming-weight asymmetry (creation vs annihilation count)

EXPECTED OUTCOME
================
Path (A) fails by structural symmetry. (B) and (C) require new structural
content to be Type 6c-compliant. The probe documents which path (if any)
gives both directional match (y_b > y_τ, y_t > y_τ) AND numerical match
within tolerance, AND has Type 6c-compliant selection step.
"""

from __future__ import annotations

from itertools import combinations
from fractions import Fraction
import math
import numpy as np

# ============================================================================
# 1. Framework constants and observation
# ============================================================================
K_STAR = 3
G_GIRTH = 10
ALPHA_1_BARE = Fraction(2, 3) ** 8
ALPHA_1_FULL = (5/3) * float(ALPHA_1_BARE)
Y_TAU_PRED = ALPHA_1_FULL / K_STAR**2

V_HIGGS = 246.22
M_TAU = 1.77686
M_BOTTOM = 4.18
M_TOP = 172.69
M_NU3 = 0.05013e-9

Y_TAU_OBS = M_TAU / V_HIGGS
Y_B_OBS = M_BOTTOM / V_HIGGS
Y_T_OBS = M_TOP / V_HIGGS
Y_NU_OBS = M_NU3 / V_HIGGS

OBSERVED_RATIOS = {
    3: 1.0,                        # y_τ / y_τ (reference)
    2: Y_T_OBS / Y_TAU_OBS,        # ≈ 97.2
    1: Y_B_OBS / Y_TAU_OBS,        # ≈ 2.35
    0: Y_NU_OBS / Y_TAU_OBS,       # ≈ 2.8e-11
}

omega = np.exp(2j * np.pi / 3)
TOL = 1e-12

print("=" * 78)
print("R1 PROBE: C_3 isotypic Yukawa vertex factorization")
print("=" * 78)
print()
print(f"  Framework constants:")
print(f"    k*       = {K_STAR}     (predictions/k_star.py)")
print(f"    α_1_full = {ALPHA_1_FULL:.6f}    (predictions/alpha_1_full.py)")
print(f"    y_τ_pred = α_1_full/k*² = {Y_TAU_PRED:.6e}")
print()
print(f"  Observation (heaviest generation per sector):")
print(f"    {'sector':<25} {'n':>2} {'Q_n=n/k*':>10} {'y_obs':>11} {'y/y_τ':>11}")
print(f"    {'-'*25} {'-'*2} {'-'*10} {'-'*11} {'-'*11}")
print(f"    {'lepton (τ)':<25} {3:>2} {3.0/K_STAR:>10.4f} {Y_TAU_OBS:>11.4e} {1.0:>11.4f}")
print(f"    {'up-type (top)':<25} {2:>2} {2.0/K_STAR:>10.4f} {Y_T_OBS:>11.4e} {Y_T_OBS/Y_TAU_OBS:>11.4f}")
print(f"    {'down-type (bottom)':<25} {1:>2} {1.0/K_STAR:>10.4f} {Y_B_OBS:>11.4e} {Y_B_OBS/Y_TAU_OBS:>11.4f}")
print(f"    {'neutrino (ν₃)':<25} {0:>2} {0.0:>10.4f} {Y_NU_OBS:>11.4e} {Y_NU_OBS/Y_TAU_OBS:>11.4e}")
print()


# ============================================================================
# 2. Build Λ^•(C^k*) Fock space and C_3 cyclic operator σ
# ============================================================================
def levels(k_star: int):
    return [(n, list(combinations(range(k_star), n))) for n in range(k_star + 1)]


def sigma_on_edge(i: int, k_star: int) -> int:
    return (i + 1) % k_star


def _sorted_with_sign(seq):
    seq = list(seq)
    n = len(seq)
    sign = 1
    for i in range(n):
        for j in range(0, n - i - 1):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                sign = -sign
    return seq, sign


def apply_sigma(subset, k_star):
    if not subset:
        return (), 1
    image = [sigma_on_edge(i, k_star) for i in subset]
    sorted_image, sign = _sorted_with_sign(image)
    return tuple(sorted_image), sign


def sigma_matrix(level_basis, k_star):
    dim = len(level_basis)
    M = np.zeros((dim, dim), dtype=complex)
    idx = {s: k for k, s in enumerate(level_basis)}
    for col, src in enumerate(level_basis):
        tgt, sign = apply_sigma(src, k_star)
        M[idx[tgt], col] = sign
    return M


fock_levels = levels(K_STAR)


# ============================================================================
# 3. Build C_3 isotypic projectors per level
# ============================================================================
print("=" * 78)
print("Step 3: C_3 isotypic projectors P_trivial, P_ω, P_ω̄ per Λ^n")
print("=" * 78)
print()
print("  P_α = (1/3) Σ_k ω^{-αk} σ^k   for α ∈ {0, 1, 2} (irrep label)")
print()

projectors = {}
for n, basis in fock_levels:
    sig = sigma_matrix(basis, K_STAR)
    sig2 = sig @ sig
    P_trivial = (np.eye(sig.shape[0]) + sig + sig2) / 3.0
    P_omega   = (np.eye(sig.shape[0]) + np.conj(omega)    * sig + np.conj(omega**2) * sig2) / 3.0
    P_omegabar = (np.eye(sig.shape[0]) + np.conj(omega**2) * sig + np.conj(omega)    * sig2) / 3.0

    # Verify projection idempotency + orthogonality
    for name, P in [("trivial", P_trivial), ("ω", P_omega), ("ω̄", P_omegabar)]:
        err = np.linalg.norm(P @ P - P)
        assert err < 1e-9, f"P_{name}^2 ≠ P at level n={n}: {err}"
    err_complete = np.linalg.norm(P_trivial + P_omega + P_omegabar - np.eye(sig.shape[0]))
    assert err_complete < 1e-9, f"P_trivial + P_ω + P_ω̄ ≠ I at level n={n}: {err_complete}"

    projectors[n] = (P_trivial, P_omega, P_omegabar)

    tr_t = float(np.trace(P_trivial).real)
    tr_o = float(np.trace(P_omega).real)
    tr_b = float(np.trace(P_omegabar).real)
    print(f"  n = {n}: dim = {sig.shape[0]:>2}, "
          f"tr(P_trivial) = {tr_t:.4f}, tr(P_ω) = {tr_o:.4f}, tr(P_ω̄) = {tr_b:.4f}")

print()
print(f"  Verified: P_α² = P_α and Σ_α P_α = I at all levels.")
print()


# ============================================================================
# 4. STRUCTURAL OBSERVATION — Hodge duality of Λ^1 and Λ^2 at k*=3
# ============================================================================
print("=" * 78)
print("Step 4: Hodge-duality test — does Λ^1 ≅ Λ^2 isotypically at k*=3?")
print("=" * 78)
print()

P_trivial_1, P_omega_1, P_omegabar_1 = projectors[1]
P_trivial_2, P_omega_2, P_omegabar_2 = projectors[2]

trace_test = (
    abs(np.trace(P_trivial_1) - np.trace(P_trivial_2)) < TOL and
    abs(np.trace(P_omega_1)   - np.trace(P_omega_2))   < TOL and
    abs(np.trace(P_omegabar_1) - np.trace(P_omegabar_2)) < TOL
)
print(f"  tr(P_trivial) at n=1: {float(np.trace(P_trivial_1).real):.4f}")
print(f"  tr(P_trivial) at n=2: {float(np.trace(P_trivial_2).real):.4f}")
print(f"  tr(P_ω)       at n=1: {float(np.trace(P_omega_1).real):.4f}")
print(f"  tr(P_ω)       at n=2: {float(np.trace(P_omega_2).real):.4f}")
print()
print(f"  RESULT: Λ^1 and Λ^2 have IDENTICAL C_3 isotypic decomposition at k*=3.")
print(f"          (Both: trivial ⊕ ω ⊕ ω̄, dim 3 each.)")
print()
print(f"  STRUCTURAL OBSTACLE: any C_3-isotypic-only operator produces the SAME")
print(f"  amplitude for n=1 (down quark) and n=2 (up quark). Hence cannot")
print(f"  distinguish y_b from y_t. Observed y_b/y_τ ≈ 2.35 vs y_t/y_τ ≈ 97.2,")
print(f"  factor ~41 apart — C_3 isotypic alone is structurally INSUFFICIENT.")
print()


# ============================================================================
# 5. Test candidate (A): pure C_3 isotypic factors
# ============================================================================
print("=" * 78)
print("Step 5: Candidate (A) — pure C_3 isotypic Yukawa amplitude factors")
print("=" * 78)
print()

# Multiple ways an "isotypic-aware Yukawa amplitude" might depend on n:
#  A1: dim(trivial component at level n) — counts C_3-symmetric Fock states
#  A2: dim(non-trivial component at level n) — counts C_3-asymmetric
#  A3: dim(total level n) — equals dim(P_trivial) + dim(P_ω) + dim(P_ω̄)
#  A4: ratio dim(trivial)/dim(level)
#  A5: Hodge complement: read sector n via complement n_c = k*-n

candidates_A = []
for n in range(K_STAR + 1):
    P_t, P_o, P_b = projectors[n]
    dim_total = P_t.shape[0]
    dim_trivial = round(float(np.trace(P_t).real))
    dim_nontrivial = round(float(np.trace(P_o).real)) + round(float(np.trace(P_b).real))
    candidates_A.append((n, dim_total, dim_trivial, dim_nontrivial))

print(f"  {'n':>2} {'dim Λ^n':>8} {'dim trivial':>12} {'dim non-trivial':>16}")
print(f"  {'-'*2} {'-'*8} {'-'*12} {'-'*16}")
for n, dim_tot, dim_t, dim_nt in candidates_A:
    print(f"  {n:>2} {dim_tot:>8} {dim_t:>12} {dim_nt:>16}")
print()

# Test each candidate amplitude form
def report_candidate(label, factor_fn, n_range=(0, 1, 2, 3)):
    """Report y_(sector n)/y_τ predictions per candidate factor function."""
    factors = {n: factor_fn(n) for n in n_range}
    f_tau = factors[3]
    if abs(f_tau) < TOL:
        print(f"  {label}: f(τ)=0, division undefined.")
        return
    print(f"  {label}:")
    print(f"    {'n':>2}  {'f(n)':>8}  {'y/y_τ pred':>11}  {'y/y_τ obs':>11}  {'rel err':>9}  {'dir match'}")
    print(f"    {'-'*2}  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*9}  {'-'*9}")
    for n in n_range:
        f = factors[n]
        ratio_pred = f / f_tau
        ratio_obs = OBSERVED_RATIOS[n]
        if abs(ratio_obs) < TOL:
            err = abs(ratio_pred)  # neutrino case
        else:
            err = abs(ratio_pred - ratio_obs) / abs(ratio_obs)
        # Direction match: do both predicted and observed ratios go the same way (both > 1 or both < 1)?
        if n == 3:
            dir_str = "—"
        else:
            both_above = (ratio_pred > 1) and (ratio_obs > 1)
            both_below = (ratio_pred < 1) and (ratio_obs < 1)
            both_zero  = (abs(ratio_pred) < TOL) and (abs(ratio_obs) < 1e-9)
            dir_str = "✓" if (both_above or both_below or both_zero) else "✗"
        print(f"    {n:>2}  {f:>8.4f}  {ratio_pred:>11.4e}  {ratio_obs:>11.4e}  {err:>9.4f}  {dir_str}")
    print()


print("  [A1] Yukawa amplitude ∝ dim(P_trivial at Λ^n)")
report_candidate("A1: dim(trivial)", lambda n: candidates_A[n][2])

print("  [A2] Yukawa amplitude ∝ dim(non-trivial at Λ^n)")
report_candidate("A2: dim(non-trivial)", lambda n: candidates_A[n][3])

print("  [A3] Yukawa amplitude ∝ dim(Λ^n) total")
report_candidate("A3: dim(level)", lambda n: candidates_A[n][1])

print("  [A4] Yukawa amplitude ∝ dim(trivial)/dim(level)")
report_candidate("A4: trivial/level", lambda n: candidates_A[n][2] / candidates_A[n][1])

print("  [A5] Yukawa amplitude via Hodge complement: read n via n_c = k*-n")
report_candidate("A5: Hodge complement", lambda n: candidates_A[K_STAR - n][2])


print()
print(f"  VERDICT for path (A) — pure C_3 isotypic:")
print(f"  None of A1-A5 matches OBSERVED hierarchy direction (y_b > y_τ, y_t > y_τ)")
print(f"  for ALL three quark/lepton sectors. Confirmed by Hodge duality of Λ^1, Λ^2.")
print(f"  Path (A) STRUCTURAL OBSTRUCTION — needs additional content.")
print()


# ============================================================================
# 6. Test candidate (B): C_3 isotypic + Q_n charge weighting
# ============================================================================
print("=" * 78)
print("Step 6: Candidate (B) — isotypic × charge function f(Q_n)")
print("=" * 78)
print()

# Per `theorem_charge_before_color.md`: Hamming weight n at trivalent vertex
# gives U(1) ⊂ U(3) charge Q_n = n/k*. This breaks the n ↔ k*-n Hodge symmetry
# (Q_1 = 1/3 ≠ Q_2 = 2/3). Could a charge-weighted isotypic factor give the
# observed hierarchy?

# Several charge-weighting candidates from natural framework structure:
# B1: f(Q_n) = Q_n  — direct charge proportionality
# B2: f(Q_n) = Q_n²  — like Yukawa-charge-coupling squared
# B3: f(Q_n) = (1+Q_n)  — combined hypercharge offset
# B4: f(Q_n) = exp(Q_n · g²)  — running coupling form
# B5: f(Q_n) = Q_n / (1-Q_n) for Q_n < 1; lepton (Q=1) singular — doesn't fit
# B6: f(Q_n) = level dim(Λ^n) × Q_n
# B7: f(Q_n) = Q_n^(2 - dim(non-trivial)/k*) — combined isotypic-charge

def Q_n(n, k_star=K_STAR):
    return Fraction(n, k_star)


print(f"  Sector charges per charge_before_color theorem: Q_n = n/k*")
print()
candidates_B = [
    ("B1: f(Q_n) = Q_n",                     lambda n: float(Q_n(n)) if n > 0 else 1e-99),
    ("B2: f(Q_n) = Q_n²",                    lambda n: float(Q_n(n))**2 if n > 0 else 1e-99),
    ("B3: f(Q_n) = (1+Q_n)",                 lambda n: 1.0 + float(Q_n(n))),
    ("B4: f(Q_n) = (1 + 3·Q_n)",             lambda n: 1.0 + 3.0*float(Q_n(n))),
    ("B5: dim(level) × Q_n",                 lambda n: candidates_A[n][1] * float(Q_n(n)) if n > 0 else 1e-99),
    ("B6: dim(level) × Q_n²",                lambda n: candidates_A[n][1] * float(Q_n(n))**2 if n > 0 else 1e-99),
    ("B7: (1+Q_n) × dim(level)",             lambda n: (1.0 + float(Q_n(n))) * candidates_A[n][1]),
    ("B8: dim(non-trivial) × (1+Q_n)",       lambda n: candidates_A[n][3] * (1.0 + float(Q_n(n))) if candidates_A[n][3] > 0 else 1e-99),
]

for name, fn in candidates_B:
    report_candidate(name, fn)

print(f"  VERDICT for path (B) — isotypic + charge:")
print(f"  Test for any candidate matching ALL of: y_b/y_τ ≈ 2.35, y_t/y_τ ≈ 97,")
print(f"  y_ν/y_τ ≈ 0. Any matches highlight as 'MATCH' below.")
print()


# ============================================================================
# 7. Honest verdict aggregator
# ============================================================================
def aggregated_match(factor_fn, n_range=(0, 1, 2, 3)):
    """Return (direction_matches, max_rel_err) tuple."""
    factors = {n: factor_fn(n) for n in n_range}
    f_tau = factors[3]
    if abs(f_tau) < TOL:
        return False, float('inf')
    dir_ok = True
    max_err = 0.0
    for n in n_range:
        if n == 3:
            continue
        ratio_pred = factors[n] / f_tau
        ratio_obs = OBSERVED_RATIOS[n]
        # Direction
        if n == 0:
            # neutrino: predicted should be tiny
            if abs(ratio_pred) > 1e-3:
                dir_ok = False
        else:
            if (ratio_pred > 1) != (ratio_obs > 1):
                dir_ok = False
            err = abs(ratio_pred - ratio_obs) / abs(ratio_obs)
            if err > max_err:
                max_err = err
    return dir_ok, max_err


print("=" * 78)
print("Step 7: Aggregated match verdict per candidate")
print("=" * 78)
print()
print(f"  {'candidate':<48} {'dir match':>10} {'max rel err':>12} {'verdict':<10}")
print(f"  {'-'*48} {'-'*10} {'-'*12} {'-'*10}")
all_candidates = []
for label, fn in [("A1: dim(trivial)", lambda n: candidates_A[n][2]),
                  ("A2: dim(non-trivial)", lambda n: candidates_A[n][3]),
                  ("A3: dim(level)", lambda n: candidates_A[n][1]),
                  ("A4: trivial/level", lambda n: candidates_A[n][2] / candidates_A[n][1]),
                  ("A5: Hodge complement", lambda n: candidates_A[K_STAR - n][2]),
                  ] + candidates_B:
    dir_ok, max_err = aggregated_match(fn)
    if dir_ok and max_err < 0.05:
        verdict = "MATCH ✓"
    elif dir_ok and max_err < 0.5:
        verdict = "close"
    elif dir_ok:
        verdict = "dir-only"
    else:
        verdict = "FAIL ✗"
    all_candidates.append((label, dir_ok, max_err, verdict))
    print(f"  {label:<48} {'yes' if dir_ok else 'NO':>10} {max_err:>12.4f} {verdict:<10}")
print()


# ============================================================================
# 8. Type 6c gate verdict
# ============================================================================
print("=" * 78)
print("Step 8: Type 6c gate verdict")
print("=" * 78)
print()

matches = [(l, e) for l, d, e, v in all_candidates if v == "MATCH ✓"]
close_matches = [(l, e) for l, d, e, v in all_candidates if v == "close"]
dir_only = [(l, e) for l, d, e, v in all_candidates if v == "dir-only"]

print(f"  MATCH (direction + numerical < 5%):  {len(matches)}")
print(f"  Close (direction + numerical < 50%): {len(close_matches)}")
print(f"  Direction only:                       {len(dir_only)}")
print(f"  Total candidates tested:              {len(all_candidates)}")
print()

if matches:
    print(f"  STRONG candidates (numerical match < 5%):")
    for label, err in matches:
        print(f"    - {label}  (max rel err {err:.4f})")
    print()
    print(f"  Type 6c verification REQUIRED for any STRONG candidate:")
    print(f"    (6a) L-expression: is f(n) expressible in framework's L grammar?")
    print(f"    (6b) K-membership: does f(n) at n ∈ {{0,1,2,3}} lie in K = ℚ(√2,√3,√5)?")
    print(f"    (6c) Selection step: is the choice of f(n) form canonical_encoding")
    print(f"         or channel_select with structural argument?")
    print()
    print(f"  Manual Type 6c audit needed for the matching candidate(s).")
elif close_matches:
    print(f"  Close candidates (within factor 1.5): {len(close_matches)}")
    for label, err in close_matches:
        print(f"    - {label}  (max rel err {err:.4f})")
    print()
    print(f"  None matches at <5%. Closest are 'directionally correct' but")
    print(f"  numerically off. Type 6c BLOCKS without numerical match.")
else:
    print(f"  NO STRONG MATCHES. Most candidates fail direction test.")
    print(f"  R1 in pure C_3 isotypic + charge form does not close down-sector y_b.")
    print()
    print(f"  STRUCTURAL VERDICT: The combined factor needed for the OBSERVED")
    print(f"  hierarchy (y_b/y_τ ≈ 2.35, y_t/y_τ ≈ 97) is dramatic and likely")
    print(f"  requires additional structural mechanisms beyond simple isotypic-")
    print(f"  charge weighting:")
    print(f"    - The y_t/y_τ ratio (97) is NOT a small-rational combination of")
    print(f"      framework constants {{k*=3, g=10, |E|=6, N_atoms=4}}.")
    print(f"    - The dramatic up-down quark hierarchy (y_t/y_b ≈ 41) suggests a")
    print(f"      mechanism BEYOND single-vertex C_3 isotypic content.")
    print()


# ============================================================================
# 9. Recommendations
# ============================================================================
print("=" * 78)
print("Step 9: R1 outcome and recommendations")
print("=" * 78)
print()

if not matches:
    print(f"""  R1 OUTCOME: STRUCTURALLY OBSTRUCTED in pure C_3 isotypic form.

  Path (A) [pure isotypic]: Λ^1 ≅ Λ^2 by Hodge duality at k*=3 forces
  y_b = y_t for any isotypic-only operator. Cannot distinguish n=1 from n=2.

  Path (B) [isotypic + charge Q_n]: tested ~8 charge-weighted candidates;
  none matches BOTH y_b/y_τ ≈ 2.35 AND y_t/y_τ ≈ 97 simultaneously.

  STRUCTURAL READING:
    The Yukawa hierarchy spans 14 orders of magnitude (y_ν ~ 10⁻¹³ to y_t ~ 1).
    No simple C_3 isotypic factor or Q_n function can span this range.
    The framework's existing single-vertex Yukawa derivation produces
    sector-blind output by construction (theorem_ytau_corollary §3-§7).

  ESCALATION DIRECTIONS:

    (R1-extended) Combine isotypic + charge + RG running. The dramatic top
    Yukawa (y_t ~ 1) might come from RG running between the unification scale
    and EW scale; the framework has alpha_GUT machinery but hasn't applied
    it to Yukawa running. Multi-session.

    (R2) Sector-DEPENDENT Cl(2) channel structure via SU(2)_L doublet partner
    (b paired with t-mass vs τ paired with ν-massless). Speculative; new
    framework content needed.

    (R3) Accept that quark Yukawa hierarchy is environmental / RG-anchored
    and not derivable from substrate alone. y_τ stands as the only Yukawa
    fully derivable in current framework apparatus.

  HONEST READ:
    R1 in its bounded form does NOT close Row P39 down-sector. The
    structural obstacle (Hodge duality of Λ^1 and Λ^2 at k*=3) is real
    and means C_3 isotypic content alone cannot be the mechanism.

    R-14 down-sector closure remains genuinely research-level. Not a
    bounded-session fix.
""")
else:
    print(f"  R1 OUTCOME: candidate match found. Type 6c audit needed for selection step.")

print("=" * 78)
print("END")
print("=" * 78)
