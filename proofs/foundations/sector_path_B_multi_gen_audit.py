#!/usr/bin/env python3
"""
Path B: multi-generator relation compressibility + frequency-weighted retention audit.

SCOPE
-----
Path A enumerated PAIRWISE (Coxeter) relations T_e T_f T_e ... = T_f T_e T_f ...
of cyclic length 2·m_ef, giving the standard Coxeter classification (finite /
affine / hyperbolic) of quotients of the free involutive monoid F_inv(|E|).
See:
  - proofs/foundations/sector_coxeter_E2_compressibility_audit.py
  - proofs/foundations/sector_coxeter_E3_compressibility_audit.py
  - proofs/foundations/sector_coxeter_E4_to_E8_compressibility_audit.py
  - proofs/foundations/sector_coxeter_freq_weighted_audit.py

Path B (this file) enumerates MULTI-GENERATOR relations: relations that
involve K ≥ 3 distinct generators in a single relator, going BEYOND Coxeter
pairwise. These are not captured by the Coxeter ADE / affine / hyperbolic
classification.

The canonical concrete Path B family is the symmetric K-cycle relation
  R(K, m): (T_1 T_2 ... T_K)^m = id
imposed atop the involutivity relations T_e^2 = id (already required by F_inv).
For each (|E|, K, m) triple with K ≤ |E|:

  L_r = K · m            (relator word length; encoding cost contribution)
  N_attest = |E|^L_r     (length-N stream at which rarest relator is 1× attested)
  freq_factor(N) = log₂(N) − L_r · log₂(|E|)
  L(M)  = encoding cost over all imposed multi-gen relations (Elias gamma style)
  Φ(M, N) = log₂(|F_inv(E) words ≤ N|) − log₂(|W(M) elements ≤ N|)
            (only computable cleanly when |W(M)| is known/finite)
  Combined Bayesian weight:
    W(M, N) = Φ(M, N) − L(M) + min(freq_factor(M, N), 0)

FRAMEWORK-SCALE CUTOFF (per commit 30b4bd7 / freq-weighted audit):
  K · m · log₂(|E|) ≤ log₂(N_hub) ≈ log₂(10^60) ≈ 200
Beyond that bound, the relation is frequency-suppressed even at framework scale.

HONEST SCOPE NOTE
-----------------
Unlike Path A, the multi-generator menu is NOT cleanly classified by an
ADE-style theorem. Most K-gen-only relations imposed atop bare F_inv (without
further pairwise relations) yield INFINITE, generally non-classifiable
quotients. Only a small subset of (K, m, |E|) combinations gives a quotient
whose order can be computed in closed form via the present probe; the rest are
listed with |W| = "infinite / not classified" and only the L / freq cutoff is
evaluated.

The probe therefore reports two things per row:
  (a) cutoff-conformity: does K·m·log₂(|E|) ≤ 200?  (always computable)
  (b) compression margin Φ − L: only when |W(M)| is known finite.

This matches the freq-weighted audit's framing of Part B as a frequency-cutoff
analysis rather than a finite-group enumeration.

NO STRUCTURAL CLAIMS, NO THEOREMS, NO LEDGER UPDATES. This is bookkeeping
matching the methodology of the committed Path A probes.

Smuggle-hazard hygiene (per an internal working note
and feedback_no_smuggling_observer_outputs_into_substrate.md):
  - |E| varied over {2, 3, 4, 5, 6, 7, 8}; nothing fixes |E| = 6.
  - K ranges 3..|E| within each |E|; nothing fixes K = 3.
  - No appeal to srs / Cl(6) / observer-side outputs anywhere.
"""
import math


# -------------------------------------------------------------------------
# Encoding cost helpers (matching freq-weighted audit conventions)
# -------------------------------------------------------------------------

def L_elias(m):
    """Elias-gamma cost for positive integer m. m=∞ encoded as 1 bit (special token)."""
    if m == float('inf'):
        return 1.0
    if m < 1:
        return float('inf')
    return 1 + 2 * math.floor(math.log2(m))


def L_relation(K, m):
    """Encoding cost of a K-gen cyclic relation at exponent m.

    Encoding fields per relation:
      - K (which generators appear; here we charge L_elias(K) for the count)
      - m (cyclic exponent; charge L_elias(m))
      - generator-tuple selection from |E| (handled separately at L_M level)
    For consistency with freq-weighted audit (which charged only L_elias(m)
    per pairwise relation, since the unordered pair (i,j) was implicit in
    the matrix layout), we charge L_elias(K) + L_elias(m) per multi-gen
    relation. The (i_1,...,i_K) tuple selection is bounded by the matrix
    layout in the same way the pairwise (i,j) was.
    """
    return L_elias(K) + L_elias(m)


# -------------------------------------------------------------------------
# F_inv(|E|) word counts (matching prior probes)
# -------------------------------------------------------------------------

def F_inv_log_count(E, N):
    """log₂(# reduced F_inv(E) words of length ≤ N).
    No two adjacent letters equal (involutive cancellation), so for E ≥ 2
    the count of length-L words is E·(E−1)^(L−1) for L ≥ 1, plus the empty
    word at L = 0. For large N: ≈ N·log₂(E−1) + log₂(E/(E−2)) (E ≥ 3).
    """
    if N == 0:
        return 0.0
    if E == 1:
        return 1.0 if N >= 1 else 0.0
    if E == 2:
        return math.log2(2 * N + 1) if N > 0 else 0.0
    return N * math.log2(E - 1) + math.log2(E / (E - 2))


# -------------------------------------------------------------------------
# Frequency support (matching freq-weighted audit)
# -------------------------------------------------------------------------

def freq_factor(E, L_r, N):
    """log₂ of expected count of rarest relation in length-N substrate stream."""
    if N <= 0:
        return float('-inf')
    return math.log2(N) - L_r * math.log2(E)


def N_attest(E, L_r):
    """Threshold N at which freq_factor crosses 0 (rarest relator 1× attested)."""
    return E ** L_r


def cutoff_check(E, K, m, N_hub_log2=200.0):
    """Framework-scale cutoff: K·m·log₂(|E|) ≤ log₂(N_hub) ≈ 200."""
    return K * m * math.log2(E) <= N_hub_log2


# -------------------------------------------------------------------------
# Known finite-quotient cases (the small enumerable subset)
# -------------------------------------------------------------------------
#
# In general a K-gen cyclic relation (T_1...T_K)^m = id imposed atop bare
# F_inv(E) involutivity does NOT give a finite group. Below we list the
# small subset where |W(M)| is computable in closed form.
#
# Entries: (E, K, m, name, order, notes)
#
# Note on the symmetric-group reading:
# Coxeter A_{n−1} = S_n is presented by:
#   (s_i)^2 = id,
#   (s_i s_{i+1})^3 = id  (pairwise braids),
#   (s_i s_j)^2 = id  for |i − j| ≥ 2  (pairwise commuting).
# That is a PAIRWISE Coxeter system, fully covered by Path A. It is NOT a
# K-gen single-relator system in the Path B sense.
#
# A genuinely K-gen single-relator example: (T_1 T_2 ... T_K) = c, c^m = id.
# For K-gen with NO pairwise relations imposed (so the underlying group is
# a quotient of F_inv(K) = (Z/2)*^K of |E| generators by a single K-gen
# cyclic relation), the quotient is generically infinite.
#
# The few (K, m) tuples giving finite quotients require imposing additional
# structural identities (commuting, or multiple K-gen relators). Without
# those, the L/freq audit is the only computable component.
#
# For this enumeration we therefore primarily report L/freq margins. For
# the few cases where a finite-quotient |W| is identified in the literature
# (e.g., Coxeter-style braid groups B_K modulo central element) we list it
# as a comparison row.

KNOWN_FINITE_KGEN_CASES = [
    # (E, K, m, name, order, notes)
    # Trivial baseline: K-gen with m=1 gives (T_1...T_K) = id, equivalent
    # to expressing T_K = T_{K-1}...T_1, reducing to F_inv(K-1) (still infinite for K≥3).
    # Listed for completeness; no finite quotient.
    #
    # K=3 m=2 atop bare F_inv: (abc)^2 = id with a^2=b^2=c^2=id alone
    # gives an infinite Coxeter-like group. NO finite |W| in closed form.
    #
    # The genuinely finite cases that arise from a single K-gen cyclic
    # relator atop F_inv generally do NOT exist for K ≥ 3 without extra
    # pairwise structure. We note this honestly and only enumerate
    # frequency/cutoff margins below.
]


# -------------------------------------------------------------------------
# Main enumeration: (|E|, K, m) sweep with cutoff and freq margins
# -------------------------------------------------------------------------

print("=" * 130)
print("Path B: multi-generator relation compressibility + frequency audit")
print("=" * 130)
print()
print("Single-relator family R(K, m): (T_1 T_2 ... T_K)^m = id atop F_inv(|E|).")
print()
print("For each (|E|, K, m) with K ≥ 3 and K ≤ |E|, this audit computes:")
print("  L_r       = K·m                       (relator length)")
print("  N_attest  = |E|^L_r                   (1×-attestation threshold)")
print("  freq@10^60 = log₂(10^60) − L_r·log₂(|E|)  (framework-scale freq factor)")
print("  cutoff    = PASS iff K·m·log₂(|E|) ≤ 200")
print()
print("L(M) (single-relator system): L_elias(K) + L_elias(m), in bits.")
print("Φ(M, N) reported only when |W(M)| is known finite (column = '—' otherwise).")
print()

NHUB_LOG2 = 200.0  # log₂(N_hub) ≈ log₂(10^60) ≈ 199.3 → round to 200 per commit 30b4bd7.

E_VALUES = [2, 3, 4, 5, 6, 7, 8]
K_VALUES = [3, 4, 5, 6, 7, 8]
M_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]

# Hard-assert the cutoff is the live framework value
assert abs(NHUB_LOG2 - 200.0) < 1e-9, "Framework cutoff must remain at log₂(N_hub) ≈ 200"

print(f"{'|E|':>3} {'K':>2} {'m':>3} {'L_r':>5} {'N_attest':>14} "
      f"{'L(M)':>5} {'freq@10^60':>11} {'cutoff':>7} "
      f"{'|W|':>14} {'Φ@10^60':>9} {'margin':>9} {'verdict':>9}")
print("-" * 130)

rows = []
for E in E_VALUES:
    for K in K_VALUES:
        if K > E:
            continue  # K-gen relator needs K distinct generators
        for m in M_VALUES:
            L_r = K * m
            n_attest = N_attest(E, L_r)
            L_M = L_relation(K, m)  # Single-relator system
            ff_60 = freq_factor(E, L_r, 10**60)
            cutoff_pass = cutoff_check(E, K, m, NHUB_LOG2)

            # |W| / Φ: not classifiable in closed form for generic K-gen.
            # Mark as "infinite / unclassified".
            W_str = "infinite/unclas"
            Phi_60 = float('nan')
            margin = float('nan')
            margin_str = "—"
            Phi_str = "—"

            # Combined verdict at framework scale: PASS iff cutoff_pass
            # (frequency support clears) AND compressibility doesn't rule out.
            # Without |W|, we report frequency-cutoff verdict only.
            if cutoff_pass and ff_60 >= 0:
                verdict = "FREQ-OK"
            elif cutoff_pass and ff_60 < 0:
                verdict = "BORDERLINE"  # below cutoff but still rare
            else:
                verdict = "SUPPRESSED"

            rows.append({
                'E': E, 'K': K, 'm': m, 'L_r': L_r, 'N_attest': n_attest,
                'L_M': L_M, 'ff_60': ff_60, 'cutoff_pass': cutoff_pass,
                'W_str': W_str, 'Phi_str': Phi_str, 'margin_str': margin_str,
                'verdict': verdict,
            })
            print(f"{E:>3} {K:>2} {m:>3} {L_r:>5} {n_attest:>14.2e} "
                  f"{L_M:>5.1f} {ff_60:>+11.2f} {('PASS' if cutoff_pass else 'FAIL'):>7} "
                  f"{W_str:>14} {Phi_str:>9} {margin_str:>9} {verdict:>9}")


# -------------------------------------------------------------------------
# Summary / aggregations
# -------------------------------------------------------------------------

print()
print("=" * 130)
print("SUMMARY — framework-scale (N_hub ~ 10^60) retention by (|E|, K)")
print("=" * 130)
print()
print("Per (|E|, K), the largest m at which the cutoff K·m·log₂(|E|) ≤ 200 still passes:")
print()
print(f"{'|E|':>3} {'K':>2} {'log₂(E)':>9} {'m_max (cutoff)':>16} {'N_attest@m_max':>18}")
print("-" * 60)
for E in E_VALUES:
    for K in K_VALUES:
        if K > E:
            continue
        m_max = int(NHUB_LOG2 / (K * math.log2(E)))
        n_at_mmax = E ** (K * m_max)
        print(f"{E:>3} {K:>2} {math.log2(E):>9.3f} {m_max:>16} {n_at_mmax:>18.2e}")

print()
print("=" * 130)
print("FREQUENCY-CLEARING cells (cutoff PASS AND ff@10^60 ≥ 0)")
print("=" * 130)
print()
freq_clear = [r for r in rows if r['verdict'] == 'FREQ-OK']
print(f"Total cells enumerated: {len(rows)}")
print(f"FREQ-OK cells:          {len(freq_clear)}")
print(f"BORDERLINE cells:       {len([r for r in rows if r['verdict']=='BORDERLINE'])}")
print(f"SUPPRESSED cells:       {len([r for r in rows if r['verdict']=='SUPPRESSED'])}")
print()

# Where the cutoff bites: largest L_r = K·m for which cutoff still passes
print("Where the cutoff bites: smallest m at which cutoff FAILS, per (|E|, K):")
print()
print(f"{'|E|':>3} {'K':>2} {'first-fail m':>14} {'L_r':>5}")
print("-" * 35)
for E in E_VALUES:
    for K in K_VALUES:
        if K > E:
            continue
        first_fail_m = None
        for m in M_VALUES:
            if not cutoff_check(E, K, m, NHUB_LOG2):
                first_fail_m = m
                break
        if first_fail_m is not None:
            print(f"{E:>3} {K:>2} {first_fail_m:>14} {K*first_fail_m:>5}")
        else:
            print(f"{E:>3} {K:>2} {'(none in sweep)':>14} {'—':>5}")


# -------------------------------------------------------------------------
# Reading guide
# -------------------------------------------------------------------------

print()
print("=" * 130)
print("Reading the table")
print("=" * 130)
print("""
1. Each row is a SINGLE-RELATOR multi-generator system R(K, m) over |E|
   binary involutive generators. The relation is (T_1...T_K)^m = id; the
   underlying free involutive monoid F_inv(|E|) provides the rest.

2. L_r = K·m is the relator's word length. N_attest = |E|^L_r is the length
   of substrate stream at which the rarest relation is expected to occur 1×
   on average.

3. cutoff column reports K·m·log₂(|E|) ≤ 200 (the framework-scale freq cutoff
   from commit 30b4bd7 / sector_coxeter_freq_weighted_audit.py lines 225-246).

4. |W| / Φ are reported as "—" because a generic single K-gen cyclic relator
   atop bare F_inv(|E|) does NOT close onto a finite group:
     - K=3 m=2:  (abc)^2 = id with a²=b²=c²=id alone is an infinite group
                 (related to the von Dyck (2,2,2,...) family but with no
                 pairwise braid relations bounding it).
     - K≥3 m≥2 generically: infinite, not classifiable in closed form by
                 the present probe.
   Compression margin Φ−L is therefore not a meaningful retention criterion
   for these systems; the live retention criterion is the freq cutoff alone.

5. Verdicts:
     FREQ-OK     = cutoff PASS and freq_factor@10^60 ≥ 0 (rarest relation
                   well-attested at framework scale).
     BORDERLINE  = cutoff PASS but freq_factor@10^60 < 0 (technically below
                   200-bit cutoff but still under-attested at N_hub).
     SUPPRESSED  = cutoff FAIL (K·m·log₂(|E|) > 200): system frequency-
                   suppressed even at framework scale.

6. Practical reading at framework scale N_hub ~ 10^60:
     - For |E| = 2 (log₂ = 1):       m_max ≈ 200/K. K=3 allows m≤66; K=8 m≤25.
     - For |E| = 3 (log₂ ≈ 1.585):   K=3 allows m≤42; K=8 m≤15.
     - For |E| = 8 (log₂ = 3):       K=3 allows m≤22; K=8 m≤8.

   Higher |E| compresses the m-budget faster because each generator letter
   carries more bits.

7. Honest scope:
   - This audit ENUMERATES the (|E|, K, m) cells against a frequency cutoff.
   - It does NOT classify the resulting quotients.
   - It does NOT assert any specific multi-gen system is the substrate's
     "right" reading.
   - It does NOT propose new framework structure.
   - Per A2-T plural retention: ALL frequency-clearing systems are
     simultaneously retained at framework scale, in the same sense as
     Path A's Coxeter menu.

8. Multi-relator systems (multiple K-gen relators imposed simultaneously)
   are NOT enumerated here. Each additional relator adds L_elias(K)+L_elias(m)
   to L(M), and the cutoff applies to the LONGEST relator's L_r. Some
   multi-relator systems (e.g., specific finite simple group presentations
   like Mathieu M_11 = ⟨a, b, c | a²=b²=c²=(ab)¹¹=(bc)³=(ac)⁴=(abc)⁴=...⟩)
   could in principle be enumerated, but each requires a hand-checked finite
   |W| and is therefore out of scope for this combinatorial sweep.

9. Comparison to Path A: in Path A, the |W(M)| of the Coxeter quotient was
   computable in closed form for finite, affine, and (with growth-rate
   approximation) hyperbolic systems. Path B does not enjoy that classification
   for generic K-gen relators. The HONEST scope of Path B is therefore the
   freq-cutoff axis only — a strictly weaker retention criterion than
   Path A's Φ−L+freq combined audit.
""")

print("=" * 130)
print("PATH B MULTI-GENERATOR FREQUENCY-CUTOFF AUDIT COMPLETE")
print("=" * 130)
