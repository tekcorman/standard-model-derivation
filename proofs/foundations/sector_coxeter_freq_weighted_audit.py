#!/usr/bin/env python3
"""
Frequency-weighted Coxeter compressibility audit.

Compression-only audit (sector_coxeter_E*_compressibility_audit.py) computed
Φ − L per Coxeter system. At framework scale all retained systems showed
margins ~10^60 bits, suggesting all are equally retained.

This is incomplete. Per multiway-as-brute-force interpretation: each Coxeter
relation r of length L_r has expected occurrence count in a uniform-random
length-N stream of approximately N · |E|^(−L_r). Below the threshold where
the rarest relation appears ≥ 1× on average, the system is FREQUENCY-
SUPPRESSED — its distinguishing features aren't yet attested in the data.

Combined Bayesian weight per Coxeter system M at observation length N:
  Φ(M, N) − L(M) + freq_factor(M, N)
where:
  freq_factor = min over relations r in M of log₂(N · |E|^(−L_r))
              = log₂(N) − max(L_r) · log₂(|E|)

If freq_factor < 0: model's rarest relation occurs less than once per stream.
                    Bayesian weight is suppressed by |freq_factor| bits.
If freq_factor ≥ 0: rarest relation well-attested. Compression-only ranking applies.

For each system we compute:
  - L(M)
  - max(L_r) (length of the longest/rarest defining relation)
  - threshold N at which freq_factor crosses 0: N_attest = |E|^max(L_r)
  - Φ at multiple N values
  - Combined weight at multiple N values

Prediction (per user insight): simple systems (low |E|, short relations)
have low N_attest. Complex systems (high |E|, long relations) have high
N_attest. At small N, simple dominates; at framework scale all retained.
"""
import math


def L_elias(m):
    if m == float('inf'):
        return 1.0
    return 1 + 2 * math.floor(math.log2(m))


def L_M(E, m_pairs):
    total = 0.0
    for i in range(1, E+1):
        for j in range(i+1, E+1):
            m = m_pairs.get((i, j), 2)
            total += L_elias(m)
    return total


def F_inv_log_count(E, N):
    if N == 0 or E == 0:
        return 0.0
    if E == 1:
        return 1.0 if N >= 1 else 0.0
    if E == 2:
        return math.log2(2 * N + 1) if N > 0 else 0.0
    return N * math.log2(E - 1) + math.log2(E / (E - 2))


def Phi_finite(E, order, N):
    """Compression value for a finite Coxeter group of given order."""
    f_log = F_inv_log_count(E, N)
    w_log = math.log2(order)
    return max(0.0, f_log - min(f_log, w_log))


def max_relation_length(m_pairs):
    """Longest defining relation length: 2 * max(m_ij), since (T_i T_j)^m has length 2m.
    For m=2 (commuting): length 4 (T_i T_j T_i T_j = id). For m=∞: no relation."""
    max_m = 2
    for (i, j), m in m_pairs.items():
        if m == float('inf'):
            continue
        if m > max_m:
            max_m = m
    return 2 * max_m


def freq_factor(E, max_L_r, N):
    """Frequency support: log₂ of expected count of rarest relation at length N."""
    if N <= 0:
        return float('-inf')
    return math.log2(N) - max_L_r * math.log2(E)


def N_attest(E, max_L_r):
    """Threshold N at which freq_factor crosses 0 (rarest relation 1× attested)."""
    return E ** max_L_r


def combined_weight(Phi, L, freq):
    """Bayesian combined weight (compressibility + frequency support)."""
    return Phi - L + min(freq, 0.0)


# Coxeter systems sample
systems = [
    # |E|=2 (just I_2(p) varying m)
    {'E': 2, 'name': 'V_4 = (Z/2)² (m=2)', 'm_pairs': {(1,2): 2}, 'order': 4, 'class': 'finite'},
    {'E': 2, 'name': 'S_3 = D_3 (m=3)', 'm_pairs': {(1,2): 3}, 'order': 6, 'class': 'finite'},
    {'E': 2, 'name': 'D_4 (m=4)', 'm_pairs': {(1,2): 4}, 'order': 8, 'class': 'finite'},
    {'E': 2, 'name': 'D_8 (m=8)', 'm_pairs': {(1,2): 8}, 'order': 16, 'class': 'finite'},
    {'E': 2, 'name': 'D_∞ (m=∞)', 'm_pairs': {(1,2): float('inf')}, 'order': None, 'class': 'free'},
    # |E|=3
    {'E': 3, 'name': '(Z/2)³ (all m=2)', 'm_pairs': {(1,2):2, (1,3):2, (2,3):2}, 'order': 8, 'class': 'finite'},
    {'E': 3, 'name': 'A_3 = S_4', 'm_pairs': {(1,2):3, (2,3):3}, 'order': 24, 'class': 'finite'},
    {'E': 3, 'name': 'B_3 octahedral', 'm_pairs': {(1,2):4, (2,3):3}, 'order': 48, 'class': 'finite'},
    {'E': 3, 'name': 'H_3 icosahedral', 'm_pairs': {(1,2):5, (2,3):3}, 'order': 120, 'class': 'finite'},
    # |E|=4
    {'E': 4, 'name': 'A_4 = S_5', 'm_pairs': {(1,2):3,(2,3):3,(3,4):3}, 'order': 120, 'class': 'finite'},
    {'E': 4, 'name': 'F_4 (rank 4 exceptional)', 'm_pairs': {(1,2):3,(2,3):4,(3,4):3}, 'order': 1152, 'class': 'finite'},
    {'E': 4, 'name': 'H_4 (rank 4 icosahedral×)', 'm_pairs': {(1,2):5,(2,3):3,(3,4):3}, 'order': 14400, 'class': 'finite'},
    # |E|=6: E_6
    {'E': 6, 'name': 'A_6 = S_7', 'm_pairs': {(1,2):3,(2,3):3,(3,4):3,(4,5):3,(5,6):3}, 'order': 5040, 'class': 'finite'},
    {'E': 6, 'name': 'E_6 (exceptional)', 'm_pairs': {(1,2):3,(2,3):3,(3,4):3,(4,5):3,(3,6):3}, 'order': 51840, 'class': 'finite'},
    # |E|=7: E_7
    {'E': 7, 'name': 'E_7 (exceptional)', 'm_pairs': {(1,2):3,(2,3):3,(3,4):3,(4,5):3,(5,6):3,(3,7):3}, 'order': 2903040, 'class': 'finite'},
    # |E|=8: E_8
    {'E': 8, 'name': 'A_8 = S_9', 'm_pairs': {(1,2):3,(2,3):3,(3,4):3,(4,5):3,(5,6):3,(6,7):3,(7,8):3}, 'order': 362880, 'class': 'finite'},
    {'E': 8, 'name': 'E_8 (THE exceptional)', 'm_pairs': {(1,2):3,(2,3):3,(3,4):3,(4,5):3,(5,6):3,(6,7):3,(3,8):3}, 'order': 696729600, 'class': 'finite'},
]


print("=" * 130)
print("Frequency-weighted Coxeter compressibility audit")
print("=" * 130)
print()
print("Combined weight per system M at observation length N:")
print("  W(M, N) = Φ(M, N) − L(M) + min(freq_factor(M, N), 0)")
print()
print("freq_factor = log₂(N) − max(L_r) · log₂(|E|)")
print("If negative: rarest relation expected < 1× per stream → support penalty")
print("If positive: well-attested → compression-only ranking applies")
print()
print(f"{'system':<32} {'|E|':>4} {'L':>5} {'max(L_r)':>9} {'N_attest':>14}", end="")
for N in [10, 100, 10000, 10**6, 10**60]:
    print(f"  W@N=10^{int(math.log10(max(N,1))):<2}", end="")
print()
print("-" * 130)


def compute_row(sys):
    E = sys['E']
    L = L_M(E, sys['m_pairs'])
    max_L_r = max_relation_length(sys['m_pairs'])
    n_attest = N_attest(E, max_L_r)
    print(f"{sys['name']:<32} {E:>4} {L:>5.1f} {max_L_r:>9} {n_attest:>14.2e}", end="")
    for N in [10, 100, 10000, 10**6, 10**60]:
        if sys['class'] == 'finite':
            Phi = Phi_finite(E, sys['order'], N)
        else:
            Phi = 0.0
        ff = freq_factor(E, max_L_r, N)
        W = combined_weight(Phi, L, ff)
        # Format compactly
        if abs(W) > 1e15:
            mag = math.copysign(math.log10(abs(W)), W)
            print(f"  {('+' if W > 0 else '-')}10^{int(mag):>3}", end="")
        else:
            print(f"  {W:>+9.2f}", end="")
    print()


for sys in systems:
    compute_row(sys)


print()
print("=" * 130)
print("Reading the table")
print("=" * 130)
print("""
KEY: N_attest = |E|^max(L_r) is the threshold where the rarest relation
becomes attested ≥ 1× per length-N stream. Below N_attest, the model is
frequency-suppressed; above, it's compression-dominated.

THE NATURAL HIERARCHY (per user's insight, frequency-corrected):

|E| = 2 (dihedral menu):
  V_4: max(L_r)=4, N_attest = 2^4 = 16. Attested early.
  S_3: max(L_r)=6, N_attest = 2^6 = 64. Slightly later.
  D_4: max(L_r)=8, N_attest = 256.
  D_8: max(L_r)=16, N_attest = 65536.
  Higher m → exponentially higher N_attest.

|E| = 3:
  (Z/2)³: max(L_r)=4, N_attest = 3^4 = 81.
  A_3=S_4: max(L_r)=6, N_attest = 729.
  B_3: max(L_r)=8, N_attest = 6561.
  H_3: max(L_r)=10, N_attest = 59049.

|E| = 8 (E_8):
  E_8: max(L_r)=6, N_attest = 8^6 = 262144 ≈ 2.6×10^5.
  A_8 = S_9: max(L_r)=6, N_attest = 262144.
  Both attested at the same threshold (both have m=3 braids).

NATURAL STOPPING POINT:

At any fixed observation length N, only systems with N_attest ≤ N are
fully supported. Systems with N_attest > N are frequency-suppressed.

Threshold N at framework scale N = 10^60:
  All systems with max(L_r) ≤ 60·log₂(|E|)/log₂(|E|) = 60 are attested.
  Equivalently max(m) ≤ 30 for any |E|.
  Coxeter systems with m_ij > 30 anywhere become frequency-suppressed
  even at framework scale.

This puts a natural ceiling on the menu: m_ij ≤ ~30 for framework-scale
retention. Beyond that, the relations are too rare to support the
quotient model.

For finite exceptional Coxeter (F_4, H_3, H_4, E_6, E_7, E_8):
  All have max m ≤ 5 (H_3, H_4 have m=5). N_attest ≤ |E|^10 << 10^60.
  All ARE retained at framework scale, with margins growing as N
  exceeds N_attest.

For affine and hyperbolic with m_ij ≤ 6: also retained.
For hyperbolic with very large m_ij: NOT retained even at framework
scale, because the rare-relation frequency penalty exceeds compression.

BOUNDS PART B (multi-generator relations):

Multi-generator relations (T_1 T_2 ... T_k)^m = id have effective L_r = km
(or longer). For k generators with m=2, L_r = 2k. N_attest = |E|^(2k).

For |E|=8, k=8 (single relation involving all 8 generators), m=2:
  L_r = 16, N_attest = 8^16 = 2.8×10^14.
  At framework scale 10^60: easily attested.
  But at observation length N=10^14: not yet supported.

For multi-gen relation involving k generators each, m=3 (longer):
  L_r = 3k, N_attest = 8^(3k). For k=8, m=3: N_attest = 8^24 ≈ 5×10^21.
  Approaches framework scale; hyperbolic-like dilution.

Bottom line: multi-generator relations have MUCH higher N_attest than
pairwise. Their natural cutoff occurs at observation lengths near or
above 10^21 for k=8. At framework scale 10^60 they're still retained
but barely.

The natural stopping point on Part B is determined by N_attest crossing
N_observed. For framework-scale N_hub ~ 10^60, multi-gen relations up to
k=8 with m≤3 are retained; beyond that they're frequency-suppressed.
""")

print("=" * 130)
print("FREQUENCY-WEIGHTED AUDIT COMPLETE")
print("=" * 130)
