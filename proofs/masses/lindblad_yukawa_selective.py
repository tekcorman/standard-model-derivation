#!/usr/bin/env python3
"""
Canonical prediction file for the Yukawa-SELECTIVE Lindblad on the
H_total = H_visible (x) S = 12 (x) 8 = 96-dim Hilbert space.

NOTE (post-A3, 2026-04-18): Under the three-axiom framework (A1+A2+A3;
docs/framework/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 Theorem 25
(predictions/observer_hilbert_space.py). The Lindblad-form derivation from
A1+A2+A3 (vs adoption), Pati-Salam labeling, and P1/P2 remain separately
load-bearing.

GOAL. The framework's prior Lindblad pushes (directed-edge dephasing
predictions/lindblad_steady_state_at_P.py, C_3-isotypic dephasing
predictions/lindblad_isotypic_at_P.py, spinor-coupled
predictions/lindblad_spinor_coupled.py) all gave Q_charged-lepton = 1/2
under the canonical mass-flux readout. The structural pinpoint
(an internal working note): every constructed
dissipator in those pushes had a unital rate operator
sum_jump L^dag L proportional to the identity, which forces the trace
identity m_{s, alpha} = const * Tr(P_alpha P_h) * Tr(Pi_s) to factorize
species out of the Koide ratio, leaving Q determined by the integer
(1, 1, 0) C_3-content of the h-eigenspace and giving 1/2.

The present construction tests a SELECTIVE dissipator with both
properties:

  (i)  C_3 ISOTYPIC COHERENCE PRESERVED on the visible side: every jump
       operator commutes with U_{C_3} (x) I_S, so the dissipator does
       NOT decohere across the (trivial, omega, omegabar) C_3 isotypic
       blocks of the visible Bloch fibre at P. This is the property
       the prior pushes lacked (the directed-edge family II of
       predictions/lindblad_spinor_coupled.py BREAKS C_3).

  (ii) SPECIES DECOHERENCE on the spinor side: a second jump family
       acts on the spinor sector by L<->R chirality swap within each
       species s in {e, nu, u, d}, modeling Yukawa-like dynamics that
       mix L and R states of one species at a time.

Construction (per construction-proof file
proofs/foundations/lindblad_yukawa_selective_construction.py):

  H_total  = H_visible (x) S, dim = 96
  H_full   = H_visible (x) I_S    (with H_spinor = 0)
  H_visible = (B(P) + B(P)^dag) / 2

  Jump operators (two C_3-symmetric families):
    Family I  (visible C_3-isotypic dephasing, on spinor identity):
       L_{alpha, vis} = sqrt(gamma_1) * P_{alpha, vis} (x) I_S,
       alpha in {trivial, omega, omegabar}.
    Family II (Yukawa L<->R swap, on visible identity):
       L_{Y, s}       = sqrt(gamma_2) * I_visible (x) X_s,
       s in {e, nu, u, d}, where X_s = |s_R><s_L| + |s_L><s_R| is the
       Hermitian L<->R chirality swap on the rank-2 species projector
       Pi_s (chirality eigenstates of G_7 = -i Gamma_1 ... Gamma_6 within
       Pi_s).

  Total dissipator: sum L^dag L = (gamma_1 + gamma_2) I_96 (unital;
  derived from sum_alpha P_{alpha, vis} = I_visible and sum_s X_s^2 =
  sum_s Pi_s = I_S).

CONJECTURE C1 (PHYSICAL MOTIVATION FLAGGED, NOT DERIVED FROM MDL+TOGGLE).

The Yukawa-like jump operators L_{Y, s} of Family II are NOT derived
from the framework's axioms (A1 binary self-inverse toggle + A2 MDL + A3
partial-trace purification; see docs/framework/framework_axioms.md).
The motivation is the physical Standard-Model Yukawa interaction
(electroweak Higgs mechanism), reframed as a Lindblad dissipator that
mixes L and R states of one species at a time. This is a structural
conjecture beyond MDL+toggle; the lemma below is CONDITIONAL on this
conjecture being correct, i.e. on the framework being able to derive
species-internal L<->R swap as a Lindblad jump channel from MDL+toggle
in some future workstream. Until that derivation is supplied, this
file's prediction is a CANDIDATE consequence of conjecture C1, not a
framework theorem.

Numerical procedure:
  1. Build the 96 x 96 matrices via the construction file (imported).
  2. Vectorize the Lindblad superoperator (9216 x 9216 complex).
  3. SVD; report kernel dimension.
  4. Use rho_ss = I_96 / 96 (canonical default; in the steady-state
     set by unitality).
  5. Compute the 4 x 3 mass-flux table m_{s, alpha} = Tr[(P_alpha P_h)
     (x) Pi_s * (sum L^dL) * rho_ss].
  6. Compute per-species Koide ratio Q_s = sum(m_s) / (sum sqrt(m_s))^2.

RESULT (closed form, derived below; numerically verified):

  Steady-state set: kernel dim = 96 (verified numerically; see HONEST
  READOUT below). The maximally mixed I_96/96 is in the set (by
  unitality) and is the canonical default.

  Mass-flux table: m_{s, alpha} = (gamma_1 + gamma_2)/96 * Tr(P_a P_h)
                                   * Tr(Pi_s)
                                = (2/k*)/96 * Tr(P_a P_h) * 2
                                = (1/72) * Tr(P_a P_h)
  for every species s (independent of s).

  Per-species Koide ratio Q_s = 1/2 universally (NOT 2/3).

  HONEST READOUT. The construction PRESERVES C_3 isotypic coherence on
  the visible side as designed (verified: max ||[L, U_C3 (x) I]|| = 0
  for both families) but does NOT collapse the steady-state set to
  dimension 1 -- the dissipator's C_3-symmetry produces a steady-state
  degeneracy by the conservation-law of
  an internal working note (any C_3-symmetric jump
  preserves the C_3 isotypic block-diagonal structure of density
  matrices). With the unital rate operator sum L^dag L proportional to
  I_96, the mass-flux readout factorizes into (visible-trace) *
  (spinor-trace) by Schur orthogonality for unital channels with
  Kronecker-product readout projectors, and Q is determined by the
  integer C_3 content (1, 1, 0) of the h-eigenspace alone, giving
  Q_s = 1/2 for every species.

  The selectivity in the SPINOR-side decoherence (Yukawa L<->R swap
  per species) is NOT enough to break the species factorization in the
  Koide ratio: each X_s satisfies X_s^2 = Pi_s, so sum_s X_s^2 = I_S
  exactly cancels the per-species discrimination at the level of the
  rate operator. The species enter the trace identity ONLY through the
  Pi_s projector at the readout step, which factorizes against the
  unital rate operator.

CONCLUSION. Q_charged-lepton = 1/2 under the Yukawa-selective
construction with rho_ss = I_96/96. The construction does NOT achieve
the framework's prediction Q_Koide = 2/3. The structural reason is
identical to the prior Lindblad constructions: the unital rate
operator forces the species factor to drop out. The selective C_3
preservation does NOT help because the C_3-symmetric dissipator has
the steady-state degeneracy of an internal working note

The bridge to Q_Koide = 2/3 still requires the P2 sqrt-coherent
aggregation postulate (docs/framework/W4_identification_catalog.md §3), which is
NOT derivable from MDL+toggle through any canonical Lindblad mass-flux
readout, including this Yukawa-selective one.

References:
  Lindblad 1976; Gorini-Kossakowski-Sudarshan 1976; Wolf 2012 Theorem
  6.1 (unital channels admit maximally mixed fixed point) and §6
  fixed-point characterization; Breuer-Petruccione 2002 Ch. 3.
  Theorem B3 (proofs/foundations/theorem_B3_spinor_fermion.py;
  ../../predictions/theorem_B3_spinor_fermion_derivation.md) for the chirality operator G_7
  and species labels on the 8-dim Cl(6, 0) spinor.
  Theorem B5.3-core (docs/theorem_B5_3_core.md) for (1, 1, 0) C_3
  content of h-eigenspace.
  Predecessor lemmas: predictions/lindblad_steady_state_at_P.py,
  predictions/lindblad_isotypic_at_P.py,
  predictions/lindblad_spinor_coupled.py.
  Companion stall: an internal working note
"""

# ============================================================
# PARAMETER: lindblad_yukawa_selective
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       For each species s in {e, nu, u, d}, the per-C_3-isotypic-
#              channel mass-flux on the h-eigenspace is identical, equal
#              to (gamma_1 + gamma_2)/96 * Tr(P_a P_h) * Tr(Pi_s):
#                m_{s, trivial}  = 1/72
#                m_{s, omega}    = 1/72
#                m_{s, omegabar} = 0
#              Per-species Koide ratio Q_s = 1/2 for every species.
#              Lindblad steady-state SET dim: large (NOT unique).
#              The maximally mixed state I_96/96 is in the set and serves
#              as canonical readout.
# Source:      Structural prediction of the Yukawa-selective Lindblad,
#              CONDITIONAL on Conjecture C1 (Yukawa L<->R swap as a
#              framework-derivable jump channel). "Observation" =
#              numerical SVD of the 9216-dim vectorized Lindblad
#              superoperator + Schur-orthogonality trace identities for
#              unital channels.
# PDG edition: n/a

# --- PREDICTED VALUE -----------------------------------------
# Value:       4 x 3 mass-flux table (all rows identical):
#                                trivial   omega    omegabar
#                charged-lepton:  1/72     1/72     0
#                neutrino:        1/72     1/72     0
#                up-quark:        1/72     1/72     0
#                down-quark:      1/72     1/72     0
#              Per-species Q_s = 1/2 for every s.
#              Q_charged-lepton (predicted, this construction) = 1/2.
#              Q_Koide (observed) = 2/3.
# Deviation:   |Q_pred - Q_obs| = |1/2 - 2/3| = 1/6 ~ 0.167 (~ 25% rel).
#              The construction PRESERVES C_3 isotypic coherence as
#              designed but does NOT reach Q_Koide = 2/3 because the
#              unital rate operator causes the species factor to cancel
#              out of the Koide ratio.

# --- DERIVED FORMULA -----------------------------------------
# Full proof in predictions/lindblad_yukawa_selective_derivation.md.
# Skeleton:
#
#   1. Upstream: H_visible = 12-dim Bloch fibre at P with C_3 isotypic
#      content (4, 4, 4) full / (1, 1, 0) on h-subspace.
#                                       [predictions/B_P_doubly_degenerate_h.py;
#                                        docs/theorem_B5_3_core.md;
#                                        ../../predictions/B_P_doubly_degenerate_h_derivation.md]
#   2. Upstream: S = 8-dim Cl(6, 0) spinor with chirality operator
#      G_7 and species partition Pi_e + Pi_nu + Pi_u + Pi_d = I_S
#      (each Pi_s rank 2).
#                                       [proofs/foundations/theorem_B3_spinor_fermion.py;
#                                        ../../predictions/theorem_B3_spinor_fermion_derivation.md]
#   3. H_total = 12 * 8 = 96.
#   4. H_full = H_vis (x) I_S (H_spinor = 0).
#   5. Family I jump operators (3): L_{a, vis} = sqrt(gamma_1) P_{a, vis} (x) I_S.
#      Sub-dissipator: sum_a L^dL = gamma_1 I_96.
#      Each commutes with U_{C_3} (x) I_S (visible-side C_3 preserved).
#   6. Family II jump operators (4): L_{Y, s} = sqrt(gamma_2) I_vis (x) X_s,
#      where X_s = |s_R><s_L| + |s_L><s_R| is the L<->R chirality swap
#      within Pi_s. Each X_s satisfies X_s^2 = Pi_s, so
#      sum_s X_s^2 = sum_s Pi_s = I_S.
#      Sub-dissipator: sum_s L^dL = gamma_2 I_vis (x) sum_s X_s^2 = gamma_2 I_96.
#      Each L_{Y, s} commutes with U_{C_3} (x) I_S (acts trivially on visible).
#   7. Total dissipator: sum L^dL = (gamma_1 + gamma_2) I_96 = (2/k*) I_96
#      (unital with k* = 3 and gamma_1 = gamma_2 = 1/k*).
#   8. Wolf 2012 Theorem 6.1: I_96/96 is in the steady-state set.
#   9. Steady-state structure. Both families commute with U_{C_3} (x) I_S,
#      so the dissipator preserves the C_3 isotypic block-diagonal
#      structure of density matrices (theorem_isotypic_lindblad_q_attempt.md
#      conservation law). The kernel dim of the vectorized Lindblad
#      superoperator is large (numerically verified > 12), reflecting
#      the C_3-symmetry-induced steady-state degeneracy. Selecting
#      rho_ss = I_96/96 as canonical readout (in the set by unitality).
#  10. Mass-flux per (species, generation channel) on h-subspace:
#         m_{s, alpha} = Tr[(P_a P_h) (x) Pi_s * (sum L^dL) * rho_ss]
#                      = (g1+g2) / 96 * Tr((P_a P_h) (x) Pi_s)
#                      = (g1+g2) / 96 * Tr(P_a P_h) * Tr(Pi_s)
#                      = (2/k*) / 96 * Tr(P_a P_h) * 2
#                      = (1 / (24 * k*)) * Tr(P_a P_h)
#                      = (1/72) * Tr(P_a P_h)   for k* = 3.
#  11. With Tr(P_a P_h) = (1, 1, 0): every row is (1/72, 1/72, 0).
#  12. Per-species Q_s = (1/72 + 1/72 + 0) / (sqrt(1/72) + sqrt(1/72) + 0)^2
#                     = (2/72) / (2/sqrt(72))^2 = (2/72) * 72/4 = 1/2.
#      Q_charged-lepton = 1/2, NOT 2/3.

# --- INPUTS --------------------------------------------------
# symbol      | value             | status      | predictions/ file                            | meaning
# ------------|-------------------|-------------|----------------------------------------------|--------
# k_star      | 3                 | [derived]   | predictions/k_star.py                        | coordination; W4 cancellation rate
# d_spatial   | 3                 | [derived]   | predictions/d_spatial.py                     | spatial dim; selects 3D srs
# srs embed   | I4_132 Wyckoff 8a | [derived]   | predictions/g_girth_derivation.md §2         | space group + bond list
# B(P)        | 12x12 complex     | [derived]   | predictions/B_P_doubly_degenerate_h.py       | Hashimoto Bloch at P
# h, mult 2   | (sqrt3+i sqrt5)/2 | [derived]   | predictions/B_P_doubly_degenerate_h.py       | h-eigenspace dim
# U_{C_3}     | 12x12 perm        | [derived]   | docs/theorem_B5_3_core.md Step 1             | C_3 on directed edges
# (4, 4, 4)   | full-fibre mult   | [derived]   | docs/theorem_B5_3_core.md Step 2             | C_3 character on full fibre
# (1, 1, 0)   | h-eigenspace mult | [derived]   | docs/theorem_B5_3_core.md Step 5             | C_3 content of h-eigenspace
# Cl(6, 0) S  | 8-dim spinor      | [derived]   | proofs/foundations/theorem_B3_spinor_fermion.py | one PS family
# G_7         | chirality op      | [derived]   | ../../predictions/theorem_B3_spinor_fermion_derivation.md Step 3     | Spin(6) chirality
# Pi_s ranks  | (2, 2, 2, 2)      | [derived]   | ../../predictions/theorem_B3_spinor_fermion_derivation.md Step 4     | per-species projector
# X_s         | L<->R swap        | [conjecture C1] | (this file)                              | Yukawa-like jump
# Yukawa rate | 1/k*              | [conjecture C1] | (this file; matched to W4)               | Family II rate
# Lindblad    | gen. quantum dyn. | [cited]     | Lindblad 1976; GKS 1976; Wolf 2012 Thm 6.1   | unital CP semigroup -> rho_ss = I/dim

# --- IMPLEMENTATION ------------------------------------------
# 96 x 96 operators built via the construction-proof file. 9216 x 9216
# vectorised Lindblad SVD with the closed-form trace identities
# verified numerically.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np

from k_star import predict_k_star
from d_spatial import predict_d_spatial

d = predict_d_spatial()
k_star = predict_k_star(d)

# Pull all 96 x 96 operators from the construction file.
from proofs.foundations import lindblad_yukawa_selective_construction as construction
import functools

H_full = construction.H_full
L_all = construction.L_all
DIM = construction.DIM
N_VIS = construction.N_VIS
P_h = construction.P_h
P_vis_triv = construction.P_vis_triv
P_vis_om = construction.P_vis_om
P_vis_omb = construction.P_vis_omb
I_S = construction.I_S
I_VIS = construction.I_VIS
Pi_e = construction.Pi_e
Pi_nu = construction.Pi_nu
Pi_u = construction.Pi_u
Pi_d = construction.Pi_d
gamma_1 = construction.gamma_1
gamma_2 = construction.gamma_2
U_C3_vis = construction.U_C3_vis

assert DIM == 96
assert H_full.shape == (96, 96)

# ---- Vectorise the Lindblad superoperator (9216 x 9216) ----
print(f"Building vectorised Lindblad superoperator: {DIM**2} x {DIM**2}")
I_TOT = np.eye(DIM, dtype=complex)
L_super = -1j * (np.kron(I_TOT, H_full) - np.kron(H_full.T, I_TOT))
for L in L_all:
    LdL = L.conj().T @ L
    L_super = L_super + np.kron(L.conj(), L) - 0.5 * (np.kron(I_TOT, LdL) + np.kron(LdL.T, I_TOT))

print(f"L_super shape: {L_super.shape}")

# ---- Steady state via SVD ----
print("Computing SVD of vectorised Lindblad superoperator (this takes a moment)...")
U_sv, S_sv, Vh_sv = np.linalg.svd(L_super)
n_kernel = int((S_sv < 1e-9).sum())
print(f"Smallest singular value:     {S_sv[-1]:.3e}")
print(f"2nd-smallest singular value: {S_sv[-2]:.3e}")
print(f"3rd-smallest singular value: {S_sv[-3]:.3e}")
print(f"n_kernel-th smallest sv (last zero):     {S_sv[-n_kernel]:.3e}")
print(f"(n_kernel+1)-th smallest sv (1st nonzero): {S_sv[-n_kernel-1]:.3e}")
print(f"Lindblad kernel dim (zero modes < 1e-9): {n_kernel}")
assert n_kernel >= 1, "No steady state found"

# ---- Use closed-form rho_ss = I/96 (canonical default; in steady-state set
# by unitality of the dissipator)
rho_ss = np.eye(DIM, dtype=complex) / DIM

# Verify L(rho_ss) = 0
L_of_rho = (
    -1j * (H_full @ rho_ss - rho_ss @ H_full)
    + sum(
        L @ rho_ss @ L.conj().T - 0.5 * (L.conj().T @ L @ rho_ss + rho_ss @ L.conj().T @ L)
        for L in L_all
    )
)
err_L = np.max(np.abs(L_of_rho))
print(f"||L(rho_ss = I/96)|| = {err_L:.3e}  (should be machine zero -- I/96 is in steady-state set by unitality)")
assert err_L < 1e-12, "I/96 is not in the steady-state set"

# ---- Compute the rate operator R = sum_jump L^dag L ----
R = sum(L.conj().T @ L for L in L_all)
gamma_tot = gamma_1 + gamma_2
err_R = np.max(np.abs(R - gamma_tot * I_TOT))
assert err_R < 1e-9, f"R not equal to (g1+g2) I: ||residual|| = {err_R}"
print(f"R = sum L^dL = ({gamma_1} + {gamma_2}) I_96 = {gamma_tot:.6f} I_96 (residual {err_R:.3e})")

# ---- Verify Family I and Family II BOTH commute with U_C3 (x) I_S ----
U_C3_full = np.kron(U_C3_vis, I_S)
max_comm_all = max(np.linalg.norm(U_C3_full @ L - L @ U_C3_full) for L in L_all)
print(f"max ||[L, U_C3 (x) I]|| over all jumps: {max_comm_all:.3e}  (== 0 means C_3 PRESERVED)")
assert max_comm_all < 1e-10, "Some jump operator breaks C_3 -- construction broken"

# ---- Mass-flux table m_{s, alpha} ----
species_dict = {
    'charged-lepton (e)': Pi_e,
    'neutrino (nu)': Pi_nu,
    'up-quark (u)': Pi_u,
    'down-quark (d)': Pi_d,
}
isotypic_dict = {
    'trivial': P_vis_triv,
    'omega': P_vis_om,
    'omegabar': P_vis_omb,
}

mass_flux_table = {}
for sname, Pi_s in species_dict.items():
    for aname, P_a in isotypic_dict.items():
        proj_full = np.kron(P_a @ P_h, Pi_s)
        m_val = float(np.real(np.trace(proj_full @ R @ rho_ss)))
        mass_flux_table[(sname, aname)] = m_val

print()
print("Mass-flux table m_{species, alpha} = Tr[(P_alpha P_h) (x) Pi_s * R rho_ss]:")
print(f"{'species':<22} | {'trivial':>12} | {'omega':>12} | {'omegabar':>12}")
print("-" * 72)
for sname in species_dict:
    row = "  ".join(f"{mass_flux_table[(sname, a)]:>12.6f}"
                    for a in ['trivial', 'omega', 'omegabar'])
    print(f"{sname:<22} | {row}")

# Closed-form check
expected_per_row = [1.0 / 72, 1.0 / 72, 0.0]
for sname in species_dict:
    for i, aname in enumerate(['trivial', 'omega', 'omegabar']):
        m_val = mass_flux_table[(sname, aname)]
        assert abs(m_val - expected_per_row[i]) < 1e-10, \
            f"{sname}/{aname}: {m_val} vs expected {expected_per_row[i]}"

print()
print("Closed-form check: every row is (1/72, 1/72, 0) -- OK.")

# ---- Per-species Koide ratio ----
print()
print("Per-species Koide ratio Q_s = sum(m_s)/(sum sqrt(m_s))^2:")
Q_per_species = {}
for sname in species_dict:
    ms = [mass_flux_table[(sname, a)] for a in ['trivial', 'omega', 'omegabar']]
    s_m = sum(ms)
    s_sm = sum(np.sqrt(max(m, 0)) for m in ms)
    Q = s_m / s_sm ** 2 if s_sm > 0 else float('nan')
    Q_per_species[sname] = Q
    print(f"  {sname:<22}: Q_s = {Q:.10f}  (closed form: 1/2 = 0.5)")

Q_charged_lepton = Q_per_species['charged-lepton (e)']
for sname, Q in Q_per_species.items():
    assert abs(Q - 0.5) < 1e-7, f"{sname}: Q = {Q}, expected 1/2"

print()
print(f"Q_charged-lepton (predicted) = {Q_charged_lepton:.10f}")
print(f"Q_Koide observed             = 2/3 = 0.6666666...")
print(f"Deviation                    = |1/2 - 2/3| = 1/6 ~ 0.1667 (~ 25% rel)")
print()
print(f"Steady-state status: kernel dim = {n_kernel}.")
print("  -- The Yukawa-selective Lindblad PRESERVES C_3 isotypic coherence")
print("     on the visible side as designed (max ||[L, U_C3 (x) I]|| ~ 0)")
print("     but the C_3-symmetry of the dissipator yields a steady-state")
print("     degeneracy by the conservation law of")
print("     an internal working note The maximally mixed")
print("     state I_96/96 is in the set (by unitality) and is the canonical")
print("     readout used here.")
print()
print("Conclusion: the Yukawa-selective Lindblad (CONJECTURE C1) gives")
print("Q_charged-lepton = 1/2, NOT 2/3. The selectivity in the spinor sector")
print("(Yukawa L<->R swap per species) is NOT enough: each X_s satisfies")
print("X_s^2 = Pi_s exactly, so sum_s X_s^2 = I_S and the rate operator is")
print("R = (g1+g2) I_96 -- unital. With unital R and Kronecker-product")
print("readout projector P_alpha (x) Pi_s, the trace identity factorizes")
print("by Schur orthogonality (Tr(A (x) B) = Tr(A) Tr(B)) and the species")
print("factor Tr(Pi_s) cancels in the Koide ratio. Q is then determined")
print("by the integer C_3 content (1, 1, 0) of the h-eigenspace alone,")
print("giving Q = 2/(2)^2 = 1/2 universally.")
print()
print("The bridge to Q_Koide = 2/3 still requires the P2 sqrt-coherent")
print("aggregation postulate (docs/framework/W4_identification_catalog.md §3),")
print("which is NOT supplied by any canonical Lindblad mass-flux readout.")


# --- PURE FUNCTION -------------------------------------------
# Inputs: k_star, gamma_1, gamma_2, and the C_3 multiplicities of the
# h-eigenspace. The pure function rebuilds the visible side, the spinor
# side with Yukawa-like jumps, and returns the 4 x 3 mass-flux table,
# the per-species Koide values, and the Lindblad kernel dim.

@functools.lru_cache(maxsize=None)
def predict_lindblad_yukawa_selective(k_star,
                                       gamma_1,
                                       gamma_2,
                                       mult_h_trivial,
                                       mult_h_omega,
                                       mult_h_omegabar):
    """
    Yukawa-selective Lindblad on H = H_visible (x) S = 96-dim.

    Computes the mass-flux table m_{s, alpha} for s in {charged-lepton,
    neutrino, up-quark, down-quark} and alpha in {trivial, omega,
    omegabar} (the C_3 isotypic decomposition of the h-eigenspace at P).
    Returns the per-species Koide ratios and the Lindblad superoperator
    kernel dim.

    Construction uses two C_3-symmetric jump-operator families:
      Family I:  L_{a, vis} = sqrt(gamma_1) P_{a, vis} (x) I_S   (3 jumps)
      Family II: L_{Y, s}   = sqrt(gamma_2) I_vis (x) X_s        (4 jumps)
    where X_s is the L<->R chirality swap within species projector Pi_s.

    Closed form (independently of any species s, by Schur orthogonality
    + h-eigenspace C_3 content (mult_h_trivial, mult_h_omega,
    mult_h_omegabar) = (1, 1, 0)):
        m_{s, alpha} = (gamma_1 + gamma_2)/96 * mult_h_alpha * 2
                     = (gamma_1 + gamma_2)/48 * mult_h_alpha
    With gamma_1 = gamma_2 = 1/k_star and k_star = 3:
        m_{s, alpha} = (2/(3*48)) * mult_h_alpha = mult_h_alpha / 72
    Per-species Q_s = (mult_h_t + mult_h_o + mult_h_ob)
                    / (sqrt(mult_h_t) + sqrt(mult_h_o) + sqrt(mult_h_ob))^2
    For (1, 1, 0): Q_s = 2/(2)^2 = 1/2 for every species.

    NOTE. The Yukawa-like jump operators of Family II are CONJECTURAL
    (Conjecture C1 in the file header); not derived from MDL+toggle.
    The lemma is conditional on C1.

    Parameters
    ----------
    k_star : int
        Coordination; theorem established for k_star = 3.
    gamma_1 : float
        Family I rate (visible C_3 dephasing). Canonical: 1/k_star.
    gamma_2 : float
        Family II rate (Yukawa L<->R swap). Canonical (conjectural): 1/k_star.
    mult_h_trivial, mult_h_omega, mult_h_omegabar : int
        C_3 isotypic multiplicities on the h-eigenspace; (1, 1, 0) for srs/h
        per theorem B5.3-core Step 5.

    Returns
    -------
    dict with keys:
        'mass_flux_table' : dict[(species, alpha) -> float]
        'Q_per_species'   : dict[species -> float]
        'lindblad_kernel_dim' : int
    """
    if k_star != 3:
        raise ValueError(
            f"lindblad_yukawa_selective established for k_star = 3 only. "
            f"Got k_star = {k_star}."
        )
    if (mult_h_trivial, mult_h_omega, mult_h_omegabar) != (1, 1, 0):
        raise ValueError(
            f"h-eigenspace C_3 multiplicities for srs at P are (1, 1, 0) per "
            f"theorem B5.3-core Step 5. Got "
            f"({mult_h_trivial}, {mult_h_omega}, {mult_h_omegabar})."
        )

    import sys as _sys
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    repo = _os.path.dirname(here)
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    from proofs.foundations import lindblad_yukawa_selective_construction as cstr

    H_full_local = cstr.H_full
    DIM_local = cstr.DIM
    P_h_local = cstr.P_h
    P_vis = {
        'trivial': cstr.P_vis_triv,
        'omega': cstr.P_vis_om,
        'omegabar': cstr.P_vis_omb,
    }
    species = {
        'charged-lepton (e)': cstr.Pi_e,
        'neutrino (nu)': cstr.Pi_nu,
        'up-quark (u)': cstr.Pi_u,
        'down-quark (d)': cstr.Pi_d,
    }

    # Rebuild jumps with the requested rates (the construction file fixes
    # gamma_1 = gamma_2 = 1/k_star; for the pure-function contract we
    # rebuild with the passed-in values).
    I_VIS_loc = cstr.I_VIS
    I_S_loc = cstr.I_S
    L_all_local = []
    for P_a in [cstr.P_vis_triv, cstr.P_vis_om, cstr.P_vis_omb]:
        L_all_local.append(np.sqrt(gamma_1) * np.kron(P_a, I_S_loc))
    for X_s in [cstr.X_e, cstr.X_nu, cstr.X_u, cstr.X_d]:
        L_all_local.append(np.sqrt(gamma_2) * np.kron(I_VIS_loc, X_s))

    # Vectorise Lindblad
    I_loc = np.eye(DIM_local, dtype=complex)
    L_sup = -1j * (np.kron(I_loc, H_full_local) - np.kron(H_full_local.T, I_loc))
    for L in L_all_local:
        LdL = L.conj().T @ L
        L_sup = L_sup + np.kron(L.conj(), L) - 0.5 * (np.kron(I_loc, LdL) + np.kron(LdL.T, I_loc))

    sv = np.linalg.svd(L_sup, compute_uv=False)
    kernel_dim = int((sv < 1e-9).sum())

    rho_ss_loc = I_loc / DIM_local
    R_loc = sum(L.conj().T @ L for L in L_all_local)

    mass_flux_loc = {}
    for sname, Pi_s in species.items():
        for aname, P_a in P_vis.items():
            proj = np.kron(P_a @ P_h_local, Pi_s)
            mass_flux_loc[(sname, aname)] = float(np.real(np.trace(proj @ R_loc @ rho_ss_loc)))

    Q_loc = {}
    for sname in species:
        ms = [mass_flux_loc[(sname, a)] for a in ['trivial', 'omega', 'omegabar']]
        s_m = sum(ms)
        s_sm = sum(np.sqrt(max(m, 0)) for m in ms)
        Q_loc[sname] = float(s_m / s_sm ** 2) if s_sm > 0 else float('nan')

    return {
        'mass_flux_table': mass_flux_loc,
        'Q_per_species': Q_loc,
        'lindblad_kernel_dim': kernel_dim,
    }


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl = {
        'mass_flux_table': mass_flux_table,
        'Q_per_species': Q_per_species,
        'lindblad_kernel_dim': n_kernel,
    }
    pure = predict_lindblad_yukawa_selective(k_star, gamma_1, gamma_2, 1, 1, 0)

    print()
    print("Implementation:")
    print(f"  lindblad_kernel_dim: {impl['lindblad_kernel_dim']}")
    print(f"  Q_charged-lepton: {impl['Q_per_species']['charged-lepton (e)']:.10f}")
    print("Pure function:")
    print(f"  lindblad_kernel_dim: {pure['lindblad_kernel_dim']}")
    print(f"  Q_charged-lepton: {pure['Q_per_species']['charged-lepton (e)']:.10f}")

    for key in mass_flux_table:
        diff = abs(impl['mass_flux_table'][key] - pure['mass_flux_table'][key])
        assert diff < 1e-8, \
            f"Mismatch for mass_flux {key}: {impl['mass_flux_table'][key]} vs {pure['mass_flux_table'][key]}"
    for sname in Q_per_species:
        diff = abs(impl['Q_per_species'][sname] - pure['Q_per_species'][sname])
        assert diff < 1e-7, \
            f"Mismatch for Q_{sname}: {impl['Q_per_species'][sname]} vs {pure['Q_per_species'][sname]}"
    assert impl['lindblad_kernel_dim'] == pure['lindblad_kernel_dim'], \
        f"kernel_dim mismatch: {impl['lindblad_kernel_dim']} vs {pure['lindblad_kernel_dim']}"
    print()
    print("OK: outputs agree.  Conjecture C1 (Yukawa L<->R swap channels): "
          "Q_charged-lepton = 1/2, NOT 2/3.")
