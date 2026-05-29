#!/usr/bin/env python3
"""
Canonical prediction file for the C_3-isotypic Lindblad construction on the
visible Bloch fibre at the P-point of the srs Hashimoto walker.

NOTE (post-A3, 2026-04-18): Under the three-axiom framework (A1+A2+A3;
docs/framework/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 Theorem 25
(predictions/observer_hilbert_space.py). The Lindblad-form derivation from
A1+A2+A3 (vs adoption) remains a separate open workstream.

Setup. Refines predictions/lindblad_steady_state_at_P.py
(directed-edge basis dephasing) by replacing the 12 directed-edge jump
operators L_e = sqrt(1/k*) P_e with three C_3-isotypic jump operators
L_alpha = sqrt(rate_alpha) P_alpha (alpha in {trivial, omega, omega-bar}),
where P_alpha are the rank-4 orthogonal projectors onto the C_3-isotypic
sub-bundles of the 12-dim Bloch fibre at P (theorem B5.3-core,
docs/theorem_B5_3_core.md, Step 2: full-fibre C_3-character is
chi(e, c, c^2) = (12, 0, 0), giving multiplicities (4, 4, 4)).

Three mass-flux quantities are computed:
    m_alpha_h = sum over channel L of Tr(L^dag L * P_alpha P_h)
where P_h is the orthogonal projector onto the 2-dim h-eigenspace at P
(predictions/B_P_doubly_degenerate_h.py + theorem_BP_doubly_degenerate_h
Step 3 corrected). The h-eigenspace decomposes under C_3 as 1 trivial +
1 omega + 0 omega-bar (theorem_B5_3_core Step 5 verified).

Two rate prescriptions are tested:
    (B.i)  rate_alpha = 1/k* (uniform across alpha; W4 cancellation rate per
           Markov step, walker_dynamics Step 4)
    (B.ii) rate_alpha = mult_alpha / k* with mult_alpha = (4, 4, 4) the
           full-fibre C_3 multiplicity (the dimension-weighted alternative)

Both are computed and reported. Variant B.i is the parsimonious choice
(single rate constant 1/k*); variant B.ii folds the full-fibre multiplicity
into the rate.

Result (closed form, both variants).

  Variant B.i (uniform rate 1/k*):
      m_trivial_h  = (1/k*) Tr(P_triv P_h)     = 1/3
      m_omega_h    = (1/k*) Tr(P_omega P_h)    = 1/3
      m_omegabar_h = (1/k*) Tr(P_omegabar P_h) = 0
      Q_iso = (m_t + m_o + m_ob) / (sqrt(m_t) + sqrt(m_o) + sqrt(m_ob))^2
            = (2/3) / (2/sqrt(3))^2 = (2/3) / (4/3) = 1/2.

  Variant B.ii (multiplicity-weighted rate):
      m_trivial_h  = (4/k*) Tr(P_triv P_h)     = 4/3
      m_omega_h    = (4/k*) Tr(P_omega P_h)    = 4/3
      m_omegabar_h = (4/k*) Tr(P_omegabar P_h) = 0
      Q_iso = (m_t + m_o + m_ob) / (sqrt(m_t) + sqrt(m_o) + sqrt(m_ob))^2
            = (8/3) / (4/sqrt(3))^2 = (8/3) / (16/3) = 1/2.

Both variants give Q_iso = 1/2, NOT 2/3. The factor (rate_alpha) divides
out of the Koide-style ratio in both cases, leaving a quantity that
depends only on the integers (Tr(P_triv P_h), Tr(P_omega P_h),
Tr(P_omegabar P_h)) = (1, 1, 0), and that ratio is 2/2^2 * (1/1)... = 1/2.

Steady-state structure. Because [H, U_C3] = 0 (theorem_B5_3_core Step 4)
and the jump operators L_alpha = sqrt(rate) P_alpha all commute with H
and with each other, the Lindblad superoperator preserves the C_3
isotypic block-diagonal structure of the density matrix. The kernel of
the vectorised Lindblad superoperator is 12-dimensional (numerically
verified): the dynamics never mix isotypic blocks. The maximally mixed
state I/12 is in this kernel, but is NOT the unique steady state. Any
density matrix block-diagonal in the C_3-isotypic decomposition with
non-negative spectrum and unit trace is a valid steady state.

Conclusion. The C_3-isotypic Lindblad construction is mathematically
well-defined and gives three distinct mass-flux values (m_trivial_h,
m_omega_h, m_omegabar_h) that are exact rational multiples of the
intersection-trace integers (Tr(P_triv P_h), Tr(P_omega P_h),
Tr(P_omegabar P_h)) = (1, 1, 0). The direct Koide-style ratio
sum(m)/(sum sqrt m)^2 evaluates to 1/2, NOT to 2/3. The Q_Koide value
2/3 of predictions/Q_Koide.py emerges instead from the P2 sqrt-coherent
aggregation postulate applied to the (4, 2, 2) Ramanujan multiplicities,
which is a different functional relationship between (mu_triv, mu_omega,
mu_omegabar) and (m_e, m_mu, m_tau) and not what the Lindblad mass-flux
trace identity supplies. See the companion derivation doc for the
focused scoping of what additional structure would be needed to recover
Q = 2/3 from a Lindblad mass-flux readout.
"""

# ============================================================
# PARAMETER: lindblad_isotypic_at_P
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Three closed-form mass-flux rationals on the h-eigenspace,
#              under variant B.i (rate_alpha = 1/k*):
#                m_trivial_h  = 1/3
#                m_omega_h    = 1/3
#                m_omegabar_h = 0
#              Direct Koide ratio Q_iso = 1/2.
# Source:      Structural prediction of the C_3-isotypic Lindblad on the
#              Bloch fibre at P. "Observation" = numerical kernel of the
#              144-dim vectorised Lindblad superoperator + trace identities.
# PDG edition: n/a

# --- PREDICTED VALUE -----------------------------------------
# Value:       (m_trivial_h, m_omega_h, m_omegabar_h) = (1/3, 1/3, 0)
#              Q_iso = sum(m)/(sum sqrt(m))^2 = 1/2
# Deviation:   exact rationals; numerical residual ~ 1e-15 (machine precision)
#              Q_iso = 1/2 differs from Q_Koide = 2/3 (P2-aggregation result
#              of predictions/Q_Koide.py) because the direct ratio formula
#              is functionally different from the P2 sqrt-coherent
#              aggregation that Q_Koide invokes.

# --- DERIVED FORMULA -----------------------------------------
# Full proof in predictions/lindblad_isotypic_at_P_derivation.md.
# Skeleton:
#
#   1. Upstream: k* = 3, d = 3 -> srs = I4_132 Wyckoff 8a
#                                       [predictions/k_star.py,
#                                        predictions/d_spatial.py,
#                                        predictions/g_girth_derivation.md §2]
#   2. Upstream: B(P) is the 12x12 Hashimoto Bloch operator with
#      h-eigenspace of multiplicity 2.
#                                       [predictions/B_P_doubly_degenerate_h.py;
#                                        ../../predictions/B_P_doubly_degenerate_h_derivation.md]
#   3. Upstream: U_C3 is the 12x12 k-independent permutation matrix
#      representing the body-diagonal C_3 on directed edges. [B_P, U_C3] = 0
#      at k = P (and on the entire Gamma-P fixed axis).
#                                       [docs/theorem_B5_3_core.md Steps 1-4;
#                                        proofs/foundations/theorem_B5_3_core.py]
#   4. The three C_3-isotypic projectors are
#         P_alpha = (I + chi_alpha(c)^* U + chi_alpha(c^2)^* U^2) / 3,
#         alpha in {trivial: chi=(1,1,1),
#                   omega:   chi=(1,omega,omega^2),
#                   omegabar:chi=(1,omega^2,omega)}.
#      Each is rank 4 (full-fibre C_3 character (12, 0, 0) gives
#      multiplicities (4, 4, 4); theorem_B5_3_core Step 2).
#   5. Hamiltonian H = (B(P) + B(P)^dag)/2 (Hermitian, by construction).
#   6. Jump operators (variant B.i):
#         L_alpha = sqrt(1/k*) P_alpha,
#      with the W4 cancellation rate 1/k* of walker_dynamics Step 4
#      distributed UNIFORMLY across the three isotypic channels (one
#      decoherence channel per C_3 irrep, in place of the directed-edge
#      basis decoherence channels of predictions/lindblad_steady_state_at_P.py).
#   7. Probability conservation: sum_alpha L_alpha^dag L_alpha
#         = (1/k*) sum_alpha P_alpha = (1/k*) I_12.
#      Identical unitality content as the 12-channel variant; the
#      reorganisation into three rank-4 channels does not change the
#      total dissipation rate.
#   8. Mass-flux trace identity on the h-eigenspace:
#         m_alpha_h = sum_channel Tr(L_channel^dag L_channel * P_alpha P_h)
#                   = (1/k*) Tr(P_alpha^2 P_h)
#                   = (1/k*) Tr(P_alpha P_h)
#         where the second equality uses P_alpha^2 = P_alpha (idempotent).
#      The intersection-traces Tr(P_alpha P_h) are integers because
#      P_alpha and P_h commute (both commute with U_C3): the h-eigenspace
#      decomposes orthogonally into its C_3-isotypic components.
#   9. Symbolic / numerical computation of the integers:
#         Tr(P_triv P_h)     = 1
#         Tr(P_omega P_h)    = 1
#         Tr(P_omegabar P_h) = 0
#      matching theorem_B5_3_core Step 5 (h-eigenspace = trivial + omega).
#  10. Direct Koide ratio:
#         Q_iso = (m_t + m_o + m_ob) / (sqrt(m_t) + sqrt(m_o) + sqrt(m_ob))^2
#               = (1/3 + 1/3 + 0) / (sqrt(1/3) + sqrt(1/3) + 0)^2
#               = (2/3) / (4/3)
#               = 1/2.
#      The Koide value 2/3 is NOT recovered by this direct ratio; see the
#      companion derivation doc for the structural reason (Q_Koide of
#      predictions/Q_Koide.py uses the P2 sqrt-coherent aggregation
#      postulate, which is a different functional relationship).
#
# Variant B.ii (rate_alpha = mult_alpha/k* with mult_alpha = (4, 4, 4))
# rescales the three masses uniformly by 4 and gives the same Q_iso = 1/2.
# The rate cancels out of Q_iso because rate_alpha is the same across
# alpha (full-fibre multiplicities are equal, (4, 4, 4)).

# --- INPUTS --------------------------------------------------
# symbol      | value             | status    | predictions/ file                            | meaning
# ------------|-------------------|-----------|----------------------------------------------|--------
# k_star      | 3                 | [derived] | predictions/k_star.py                        | coordination number; W4 cancellation rate = 1/k*
# d_spatial   | 3                 | [derived] | predictions/d_spatial.py                     | spatial dimension; selects 3D srs
# srs embed   | I4_132 Wyckoff 8a | [derived] | predictions/g_girth_derivation.md §2         | space group + bond list
# B(P)        | 12x12 complex     | [derived] | predictions/B_P_doubly_degenerate_h.py       | Hashimoto Bloch at P
# h, mult 2   | (sqrt3+i sqrt5)/2 | [derived] | predictions/B_P_doubly_degenerate_h.py       | h-eigenspace dim
# U_C3, perm  | 12x12 perm        | [derived] | docs/theorem_B5_3_core.md Step 1             | C_3 action on directed edges
# (4, 4, 4)   | full-fibre mult   | [derived] | docs/theorem_B5_3_core.md Step 2             | C_3 character (12, 0, 0) Schur orthogonality
# (1, 1, 0)   | h-eigenspace mult | [derived] | docs/theorem_B5_3_core.md Step 5             | C_3 content of h-eigenspace
# Lindblad    | gen. quantum dyn. | [cited]   | Lindblad 1976; GKS 1976                      | unitarity-preserving CP semigroup

# --- IMPLEMENTATION ------------------------------------------
# Numerical construction of the 144-dim vectorised Lindblad superoperator
# with three C_3-isotypic jump operators, plus closed-form verification of
# the three mass-flux trace identities.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp

from k_star import predict_k_star
from d_spatial import predict_d_spatial

# Upstream framework values.
d = predict_d_spatial()
k_star = predict_k_star(d)

# Reuse the directed-edge construction and U_C3 build from the
# B5.3-core proof (they are themselves built on proofs/common.py
# find_bonds + the documented C_3 vertex permutation).
from proofs.common import find_bonds  # noqa: E402
from proofs.foundations.theorem_B5_3_core import (  # noqa: E402
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
)
import functools

bonds = find_bonds()
directed = build_directed_edges(bonds)
N = len(directed)
assert N == 12, f"Unexpected directed-edge count {N}, expected 12 for srs primitive cell."

# B(P), Hamiltonian, and C_3 permutation U.
P_pt = (0.25, 0.25, 0.25)
B_P = bloch_hashimoto(P_pt, directed)
U = build_c3_on_directed_edges(directed)
H = (B_P + B_P.conj().T) / 2

# C_3 isotypic projectors (Schur orthogonality).
omega = np.exp(2j * np.pi / 3)
I_N = np.eye(N, dtype=complex)
P_triv = (I_N + U + U @ U) / 3
P_omega = (I_N + np.conj(omega) * U + np.conj(omega) ** 2 * (U @ U)) / 3
P_omegabar = (I_N + omega * U + omega ** 2 * (U @ U)) / 3

# Sanity: idempotent, Hermitian, orthogonal, sum to I.
for name, P_a in [('triv', P_triv), ('omega', P_omega), ('omegabar', P_omegabar)]:
    assert np.linalg.norm(P_a @ P_a - P_a) < 1e-10, f"{name} not idempotent"
    assert np.linalg.norm(P_a - P_a.conj().T) < 1e-10, f"{name} not Hermitian"
assert np.linalg.norm(P_triv + P_omega + P_omegabar - I_N) < 1e-10, "projectors do not sum to I"
for n1, P1 in [('triv', P_triv), ('omega', P_omega), ('omegabar', P_omegabar)]:
    for n2, P2 in [('triv', P_triv), ('omega', P_omega), ('omegabar', P_omegabar)]:
        if n1 != n2:
            assert np.linalg.norm(P1 @ P2) < 1e-10, f"{n1}/{n2} not orthogonal"

# Each projector has rank exactly 4 (full-fibre multiplicities (4, 4, 4)).
for name, P_a in [('triv', P_triv), ('omega', P_omega), ('omegabar', P_omegabar)]:
    rk = int(round(np.trace(P_a).real))
    assert rk == 4, f"{name} rank {rk}, expected 4"

# h-eigenspace orthogonal projector.
ev_B, V_B = np.linalg.eig(B_P)
h_target = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
mask_h = np.abs(ev_B - h_target) < 1e-6
h_indices = np.where(mask_h)[0]
assert len(h_indices) == 2, f"h multiplicity {len(h_indices)}, expected 2"
V_h_raw = V_B[:, h_indices]
V_h, _ = np.linalg.qr(V_h_raw)
P_h = V_h @ V_h.conj().T
assert abs(np.trace(P_h).real - 2.0) < 1e-10, f"Tr(P_h) = {np.trace(P_h).real}"

# Verify P_h commutes with each P_alpha (h-eigenspace decomposes
# orthogonally into C_3 isotypic components).
for name, P_a in [('triv', P_triv), ('omega', P_omega), ('omegabar', P_omegabar)]:
    err = np.linalg.norm(P_a @ P_h - P_h @ P_a)
    assert err < 1e-10, f"[{name}, P_h] norm {err}"

# Intersection-traces (the integer C_3 content of the h-eigenspace).
d_th = float(np.trace(P_triv @ P_h).real)
d_oh = float(np.trace(P_omega @ P_h).real)
d_obh = float(np.trace(P_omegabar @ P_h).real)
# Closed-form expectations (theorem_B5_3_core Step 5: h-eigenspace = trivial + omega).
assert abs(d_th - 1.0) < 1e-10, f"Tr(P_triv P_h) = {d_th}, expected 1"
assert abs(d_oh - 1.0) < 1e-10, f"Tr(P_omega P_h) = {d_oh}, expected 1"
assert abs(d_obh - 0.0) < 1e-10, f"Tr(P_omegabar P_h) = {d_obh}, expected 0"

# Variant B.i: rate_alpha = 1/k* (uniform).
gamma_i = 1.0 / k_star
L_ops_i = [np.sqrt(gamma_i) * P_triv, np.sqrt(gamma_i) * P_omega, np.sqrt(gamma_i) * P_omegabar]

# Probability conservation: sum L^dag L = (1/k*) I.
S_check = sum(L.conj().T @ L for L in L_ops_i)
assert np.linalg.norm(S_check - gamma_i * I_N) < 1e-10, \
    f"Variant B.i: sum L^dL != gamma I"

# Mass-flux on h-eigenspace per isotypic channel (rates cancel symmetric
# but we keep the explicit channel sum to make the structure transparent).
m_t_i = sum(np.trace(L.conj().T @ L @ (P_triv @ P_h)).real for L in L_ops_i)
m_o_i = sum(np.trace(L.conj().T @ L @ (P_omega @ P_h)).real for L in L_ops_i)
m_ob_i = sum(np.trace(L.conj().T @ L @ (P_omegabar @ P_h)).real for L in L_ops_i)

# Closed-form: each L_alpha^dag L_alpha = (1/k*) P_alpha; the channel
# sum picks out only the diagonal alpha block of P_h. So
#   m_alpha_h = (1/k*) Tr(P_alpha P_h).
m_t_i_expected = gamma_i * d_th
m_o_i_expected = gamma_i * d_oh
m_ob_i_expected = gamma_i * d_obh
assert abs(m_t_i - m_t_i_expected) < 1e-10
assert abs(m_o_i - m_o_i_expected) < 1e-10
assert abs(m_ob_i - m_ob_i_expected) < 1e-10

# Direct Koide-style ratio Q_iso = sum(m_alpha) / (sum sqrt(m_alpha))^2.
sum_m_i = m_t_i + m_o_i + m_ob_i
sum_sqrt_m_i = np.sqrt(m_t_i) + np.sqrt(m_o_i) + np.sqrt(m_ob_i)
Q_iso_i = sum_m_i / sum_sqrt_m_i ** 2

Q_iso_i_expected = sp.Rational(1, 2)
# Tolerance is 1e-7 to allow for sqrt(0) numerical floor on the
# Tr(P_omegabar P_h) = 0 channel (eig solver injects a tiny float into
# the numerically-zero intersection-trace, which propagates into the
# square-root sum).
assert abs(Q_iso_i - float(Q_iso_i_expected)) < 1e-7, \
    f"Variant B.i: Q_iso = {Q_iso_i}, expected {Q_iso_i_expected}"

# Variant B.ii: rate_alpha = mult_alpha / k* with mult_alpha = (4, 4, 4).
mult = {'triv': 4, 'omega': 4, 'omegabar': 4}
L_ops_ii = [
    np.sqrt(mult['triv'] / k_star) * P_triv,
    np.sqrt(mult['omega'] / k_star) * P_omega,
    np.sqrt(mult['omegabar'] / k_star) * P_omegabar,
]

m_t_ii = sum(np.trace(L.conj().T @ L @ (P_triv @ P_h)).real for L in L_ops_ii)
m_o_ii = sum(np.trace(L.conj().T @ L @ (P_omega @ P_h)).real for L in L_ops_ii)
m_ob_ii = sum(np.trace(L.conj().T @ L @ (P_omegabar @ P_h)).real for L in L_ops_ii)

sum_m_ii = m_t_ii + m_o_ii + m_ob_ii
sum_sqrt_m_ii = np.sqrt(m_t_ii) + np.sqrt(m_o_ii) + np.sqrt(m_ob_ii)
Q_iso_ii = sum_m_ii / sum_sqrt_m_ii ** 2

# Variant B.ii closed form: m_alpha_h = (mult_alpha/k*) Tr(P_alpha P_h)
# = (4/3, 4/3, 0). Q_iso_ii = (8/3) / (4/sqrt(3))^2 = (8/3)/(16/3) = 1/2.
Q_iso_ii_expected = sp.Rational(1, 2)
assert abs(Q_iso_ii - float(Q_iso_ii_expected)) < 1e-7, \
    f"Variant B.ii: Q_iso = {Q_iso_ii}, expected {Q_iso_ii_expected}"

# Vectorised Lindblad superoperator (variant B.i) for steady-state analysis.
def _vectorise_lindblad(L_ops_local, H_local, N_local):
    Iloc = np.eye(N_local, dtype=complex)
    Lsup = -1j * (np.kron(Iloc, H_local) - np.kron(H_local.T, Iloc))
    for Lk in L_ops_local:
        LdL = Lk.conj().T @ Lk
        Lsup = Lsup + np.kron(Lk.conj(), Lk) - 0.5 * (np.kron(Iloc, LdL) + np.kron(LdL.T, Iloc))
    return Lsup

L_super_i = _vectorise_lindblad(L_ops_i, H, N)
sv_i = np.linalg.svd(L_super_i, compute_uv=False)
n_kernel_i = int((sv_i < 1e-10).sum())

# Steady state is NOT unique under the C_3-isotypic dephasing because H
# and the L_alpha all commute (they share the U_C3 eigenbasis structure
# block-by-block) and so the entire isotypic block-diagonal subspace of
# density matrices is a steady-state set. Numerically n_kernel = 12.
# (= 3 isotypic blocks * 4 dim each, restricted to block-diagonal density
#  matrices: 3 * 4 = 12 diagonal degrees of freedom.)
assert n_kernel_i >= 9, f"Expected >=9-dim kernel, got {n_kernel_i}"

print(f"k* = {k_star}, d = {d}")
print(f"Bloch fibre dim N = {N}")
print()
print("C_3-isotypic projectors at P (each rank 4):")
print(f"  Tr(P_triv)     = {np.trace(P_triv).real:.6f}")
print(f"  Tr(P_omega)    = {np.trace(P_omega).real:.6f}")
print(f"  Tr(P_omegabar) = {np.trace(P_omegabar).real:.6f}")
print()
print("h-eigenspace decomposition under C_3 (theorem_B5_3_core Step 5):")
print(f"  Tr(P_triv P_h)     = {d_th:.6f}  (expected 1)")
print(f"  Tr(P_omega P_h)    = {d_oh:.6f}  (expected 1)")
print(f"  Tr(P_omegabar P_h) = {d_obh:.6f}  (expected 0)")
print()
print("Variant B.i: rate_alpha = 1/k* (uniform):")
print(f"  m_trivial_h  = {m_t_i:.6f}  (expected 1/3)")
print(f"  m_omega_h    = {m_o_i:.6f}  (expected 1/3)")
print(f"  m_omegabar_h = {m_ob_i:.6f}  (expected 0)")
print(f"  Direct Koide ratio Q_iso = sum(m)/(sum sqrt m)^2")
print(f"      = {sum_m_i:.6f} / ({sum_sqrt_m_i:.6f})^2 = {Q_iso_i:.10f}  (expected 1/2)")
print()
print("Variant B.ii: rate_alpha = mult_alpha/k* with mult_alpha = (4, 4, 4):")
print(f"  m_trivial_h  = {m_t_ii:.6f}  (expected 4/3)")
print(f"  m_omega_h    = {m_o_ii:.6f}  (expected 4/3)")
print(f"  m_omegabar_h = {m_ob_ii:.6f}  (expected 0)")
print(f"  Direct Koide ratio Q_iso = {Q_iso_ii:.10f}  (expected 1/2)")
print()
print("Lindblad superoperator (variant B.i):")
print(f"  shape {L_super_i.shape}; kernel dim = {n_kernel_i}")
print("  Steady state is NOT unique: any density matrix block-diagonal in the")
print("  C_3-isotypic decomposition is invariant under the dynamics (because")
print("  H, P_triv, P_omega, P_omegabar all mutually commute).")
print()
print("Conclusion: the C_3-isotypic Lindblad gives three distinct mass-flux")
print("values (m_trivial_h, m_omega_h, m_omegabar_h) = (1/3, 1/3, 0) under B.i.")
print("Direct Koide ratio Q_iso = 1/2, NOT 2/3. The Q_Koide = 2/3 of")
print("predictions/Q_Koide.py uses the P2 sqrt-coherent aggregation postulate,")
print("which is a different functional relationship between (mu_triv, mu_omega,")
print("mu_omegabar) and the three mass eigenvalues. See companion .md.")


# --- PURE FUNCTION -------------------------------------------
# Inputs: k_star and the three C_3 multiplicities of the h-eigenspace
# (all forced by upstream theorems). The pure function rebuilds the
# Bloch fibre, the C_3 projectors, the Lindblad superoperator, and
# returns the three mass-flux values plus the direct Koide ratio.

@functools.lru_cache(maxsize=None)
def predict_lindblad_isotypic_at_P(k_star,
                                   mult_h_trivial,
                                   mult_h_omega,
                                   mult_h_omegabar):
    """
    C_3-isotypic Lindblad construction on the 12-dim Bloch fibre at P.

    Computes three mass-flux values m_alpha_h = (1/k_star) * Tr(P_alpha P_h)
    on the h-eigenspace, where alpha runs over the three C_3 irreps
    (trivial, omega, omegabar) and P_alpha is the rank-4 C_3-isotypic
    projector on the 12-dim Bloch fibre. Variant B.i (uniform rate
    1/k_star across alpha) is the canonical choice.

    Returns the three mass-flux values and the direct Koide-style ratio
    Q_iso = sum(m_alpha) / (sum sqrt(m_alpha))^2.

    Parameters
    ----------
    k_star : int
        Coordination number. Theorem established for k_star = 3 (srs).
    mult_h_trivial : int
        C_3 trivial multiplicity on the h-eigenspace; 1 for srs/h
        (theorem_B5_3_core Step 5).
    mult_h_omega : int
        C_3 omega multiplicity on the h-eigenspace; 1 for srs/h.
    mult_h_omegabar : int
        C_3 omegabar multiplicity on the h-eigenspace; 0 for srs/h.

    Returns
    -------
    dict with keys:
        'm_trivial_h', 'm_omega_h', 'm_omegabar_h' : float
            Mass-flux values on the h-eigenspace per isotypic channel.
        'Q_iso' : float
            Direct Koide-style ratio sum(m)/(sum sqrt(m))^2.
        'lindblad_kernel_dim' : int
            Number of zero singular values of the vectorised Lindblad
            superoperator (the steady-state degeneracy).
    """
    if k_star != 3:
        raise ValueError(
            f"lindblad_isotypic_at_P established for k_star = 3 only. "
            f"Got k_star = {k_star}."
        )
    if (mult_h_trivial, mult_h_omega, mult_h_omegabar) != (1, 1, 0):
        raise ValueError(
            f"h-eigenspace C_3 multiplicities for srs at P are (1, 1, 0) per "
            f"theorem_B5_3_core Step 5. Got "
            f"({mult_h_trivial}, {mult_h_omega}, {mult_h_omegabar})."
        )

    import sys as _sys
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    repo = _os.path.dirname(here)
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    from proofs.common import find_bonds as _find_bonds
    from proofs.foundations.theorem_B5_3_core import (
        build_directed_edges as _bde,
        bloch_hashimoto as _bh,
        build_c3_on_directed_edges as _bc3,
    )

    bonds_local = _find_bonds()
    directed_local = _bde(bonds_local)
    n_local = len(directed_local)
    if n_local != 12:
        raise RuntimeError(f"Unexpected directed-edge count: {n_local}")

    B_loc = _bh((0.25, 0.25, 0.25), directed_local)
    U_loc = _bc3(directed_local)
    H_loc = (B_loc + B_loc.conj().T) / 2

    om_loc = np.exp(2j * np.pi / 3)
    I_loc = np.eye(n_local, dtype=complex)
    P_t = (I_loc + U_loc + U_loc @ U_loc) / 3
    P_o = (I_loc + np.conj(om_loc) * U_loc + np.conj(om_loc) ** 2 * (U_loc @ U_loc)) / 3
    P_ob = (I_loc + om_loc * U_loc + om_loc ** 2 * (U_loc @ U_loc)) / 3

    # h-eigenspace projector.
    h_loc = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    ev_loc, V_loc = np.linalg.eig(B_loc)
    mask = np.abs(ev_loc - h_loc) < 1e-6
    idx = np.where(mask)[0]
    if len(idx) != 2:
        raise RuntimeError(f"h-eigenspace mult = {len(idx)} (expected 2).")
    Vq, _ = np.linalg.qr(V_loc[:, idx])
    Ph_loc = Vq @ Vq.conj().T

    # Variant B.i jump operators.
    gamma_loc = 1.0 / k_star
    L_t = np.sqrt(gamma_loc) * P_t
    L_o = np.sqrt(gamma_loc) * P_o
    L_ob = np.sqrt(gamma_loc) * P_ob

    # Mass-flux trace identities.
    m_t_loc = float(sum(
        np.trace(L.conj().T @ L @ (P_t @ Ph_loc)).real for L in [L_t, L_o, L_ob]
    ))
    m_o_loc = float(sum(
        np.trace(L.conj().T @ L @ (P_o @ Ph_loc)).real for L in [L_t, L_o, L_ob]
    ))
    m_ob_loc = float(sum(
        np.trace(L.conj().T @ L @ (P_ob @ Ph_loc)).real for L in [L_t, L_o, L_ob]
    ))

    sum_m = m_t_loc + m_o_loc + m_ob_loc
    sum_sqrt_m = np.sqrt(max(m_t_loc, 0)) + np.sqrt(max(m_o_loc, 0)) + np.sqrt(max(m_ob_loc, 0))
    Q_iso_loc = float(sum_m / sum_sqrt_m ** 2) if sum_sqrt_m > 0 else float('nan')

    # Vectorised Lindblad kernel dim (steady-state degeneracy).
    Iloc = np.eye(n_local, dtype=complex)
    Lsup = -1j * (np.kron(Iloc, H_loc) - np.kron(H_loc.T, Iloc))
    for Lk in [L_t, L_o, L_ob]:
        LdL = Lk.conj().T @ Lk
        Lsup = Lsup + np.kron(Lk.conj(), Lk) - 0.5 * (np.kron(Iloc, LdL) + np.kron(LdL.T, Iloc))
    sv = np.linalg.svd(Lsup, compute_uv=False)
    kernel_dim = int((sv < 1e-10).sum())

    return {
        'm_trivial_h': m_t_loc,
        'm_omega_h': m_o_loc,
        'm_omegabar_h': m_ob_loc,
        'Q_iso': Q_iso_loc,
        'lindblad_kernel_dim': kernel_dim,
    }


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl = {
        'm_trivial_h': m_t_i,
        'm_omega_h': m_o_i,
        'm_omegabar_h': m_ob_i,
        'Q_iso': Q_iso_i,
    }
    pure = predict_lindblad_isotypic_at_P(k_star, 1, 1, 0)

    print()
    print("Implementation:")
    for kk in ['m_trivial_h', 'm_omega_h', 'm_omegabar_h', 'Q_iso']:
        print(f"  {kk}: {impl[kk]:.10f}")
    print("Pure function:")
    for kk in ['m_trivial_h', 'm_omega_h', 'm_omegabar_h', 'Q_iso']:
        print(f"  {kk}: {pure[kk]:.10f}")
    print(f"  lindblad_kernel_dim: {pure['lindblad_kernel_dim']}")

    for kk in ['m_trivial_h', 'm_omega_h', 'm_omegabar_h', 'Q_iso']:
        diff = abs(impl[kk] - pure[kk])
        # Tolerance 1e-7 admits the float floor on the tiny m_omegabar_h
        # channel (Tr(P_omegabar P_h) is exactly 0; np.linalg.eig returns
        # a ~1e-16 float that propagates into the channel-summed trace and,
        # via sqrt, into Q_iso at ~1e-8 relative precision).
        assert diff < 1e-7, f"Mismatch for {kk}: {impl[kk]} vs {pure[kk]}"
    print()
    print("OK: outputs agree.")
    print(f"Closed form (variant B.i): m = (1/3, 1/3, 0); Q_iso = 1/2 (NOT 2/3).")
