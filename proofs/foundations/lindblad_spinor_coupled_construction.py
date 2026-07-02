#!/usr/bin/env python3
"""
Construction proof for the spinor-coupled Lindblad on the 96-dim Hilbert
space H = H_visible (12-dim Bloch fibre at P) tensor S (8-dim Cl(6,0) spinor).

This script is the foundational construction file: it builds the 96 x 96
operators (Hamiltonian, jump operators, projectors) and verifies their
algebraic properties (commutation, unitality, projector idempotency,
species/isotypic intersections) symbolically/numerically before the
prediction layer (predictions/lindblad_spinor_coupled.py) consumes them
to compute the steady state and mass-flux table.

Construction (per task plan steps A-E):

  Step A. H_total = H_visible (x) S, dim = 12 * 8 = 96.
          H_visible: 12-dim Bloch fibre at P-point of srs Hashimoto walker
                     (predictions/lindblad_steady_state_at_P.py;
                      ../../predictions/walker_dynamics_derivation.md).
          S: 8-dim Cl(6,0) Dirac spinor with Spin(4) x Spin(2) =
             SU(2)_L x SU(2)_R x U(1)_{B-L} Cartan structure
             (proofs/foundations/theorem_B3_spinor_fermion.py;
              ../../predictions/theorem_B3_spinor_fermion_derivation.md).

  Step B. H_full = H_visible (x) I_S + I_visible (x) H_spinor.
          We take H_spinor = 0 (no internal spinor dynamics; the Lindblad
          dissipator carries all sector-mixing structure). H_full is
          Hermitian by Hermiticity of H_visible.

  Step C. Jump operators (two channel families, both unital, the second
          C_3-breaking on the visible side):

          Family I (visible C_3-isotypic, on the spinor identity):
              L_{alpha, vis} = sqrt(1/k*) * P_{alpha, vis} (x) I_S,
              alpha in {trivial, omega, omegabar}.
              These commute with U_{C_3} (x) I_S; same isotypic content as
              the prior degenerate-steady-state construction
              (predictions/lindblad_isotypic_at_P.py).

          Family II (visible directed-edge, on the spinor B-L species):
              L_{e, s} = sqrt(1/(k* * 2)) * P_e (x) Pi_s,
              e in {0, ..., 11}, s in {Y_+, Y_-}.
              P_e is the rank-1 directed-edge projector on the visible side
              (does NOT commute with U_{C_3}). Pi_s is the rank-4 spinor
              projector onto the B-L = +1 (Y_+) or B-L = -1 (Y_-) eigenspace
              of the spinor Cartan generator Y = Gamma_{56}/(2i)
              (theorem_B3_spinor_fermion Step 2, Step 4: Y eigenvalues
              partition the 8 spinor states into 4 + 4 by B-L sign,
              corresponding to {quark axis, lepton axis} of one PS family).
              The rate prefactor 1/(k* * 2) keeps the family II total
              dissipation rate equal to family I's: sum_{e,s} L^dag L =
              sum_{e,s} (1/(2 k*)) P_e (x) Pi_s = (1/(2k*)) (sum_e P_e)
              (x) (sum_s Pi_s) = (1/(2k*)) I_V (x) I_S = (1/(2k*)) I_96
              -- wait, this equals (1/(2 k*)) I, i.e. the family II rate
              constant on the identity is 1/(2 k*) = 1/6, while family I's
              is 1/k* = 1/3. We rescale family II to also give 1/k* * I,
              i.e. use sqrt(1/k*) prefactor on family II as well; then
              total dissipator is (2/k*) I_96.

          The C_3-breaking comes from: U_{C_3} (x) I_S permutes the 12
          directed edges, so [U_{C_3} (x) I_S, P_e (x) Pi_s] != 0 for
          individual e (the union sum_e P_e is U_{C_3}-invariant but
          individual edges are not). Hence the dissipator has Family II
          jump operators that do NOT preserve the C_3-isotypic block-
          diagonal subspace of density matrices, lifting the steady-state
          degeneracy of the pure-isotypic Lindblad
          (predictions/lindblad_isotypic_at_P.py kernel dim 12).

  Step D. Vectorize the Lindblad superoperator (96^2 = 9216 dim) and
          compute its kernel via SVD. Report kernel dimension (1 = unique
          steady state desired).

  Step E. For each species s (we report B-L = + and B-L = -, i.e. quark
          axis and lepton axis) and each C_3 isotypic sector alpha
          (trivial, omega, omegabar) of the h-eigenspace, compute
              m_{s, alpha} = Tr[(P_{alpha, vis} P_h) (x) Pi_s)
                              * sum_jump L^dag L rho_ss]
          giving a 2 x 3 mass-flux table.

  Step F. Per-species Koide check:
              Q_s = (m_{s, t} + m_{s, o} + m_{s, ob})
                  / (sqrt(m_{s, t}) + sqrt(m_{s, o}) + sqrt(m_{s, ob}))^2.

This script does only the algebraic construction and small-dim verifications.
The 9216 x 9216 SVD and the steady-state computation are done in the
companion prediction file predictions/lindblad_spinor_coupled.py.

References:
  Lindblad 1976 (CMP 48, 119); Gorini-Kossakowski-Sudarshan 1976
  (J. Math. Phys. 17, 821); Wolf 2012 "Quantum Channels and Operations"
  Theorem 6.1 (unital channels) and §6 fixed-point set characterization;
  Breuer-Petruccione 2002 Ch. 3.

Prints "OK:" on success.
"""

from __future__ import annotations

import sys
import os
import itertools

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

from proofs.common import find_bonds  # noqa: E402
from proofs.foundations.theorem_B5_3_core import (  # noqa: E402
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
)

TOL = 1e-10

# --------------------------------------------------------------------------
# Step A.1 Visible side: 12-dim Bloch fibre at P with C_3 permutation
# --------------------------------------------------------------------------

bonds = find_bonds()
directed = build_directed_edges(bonds)
N_VIS = len(directed)
assert N_VIS == 12, f"visible fibre dim {N_VIS}, expected 12"

P_pt = (0.25, 0.25, 0.25)
B_P = bloch_hashimoto(P_pt, directed)
U_C3_vis = build_c3_on_directed_edges(directed)
H_vis = (B_P + B_P.conj().T) / 2
assert np.allclose(H_vis, H_vis.conj().T, atol=TOL), "H_vis not Hermitian"

# C_3-isotypic projectors on visible side
omega = np.exp(2j * np.pi / 3)
I_VIS = np.eye(N_VIS, dtype=complex)
P_vis_triv = (I_VIS + U_C3_vis + U_C3_vis @ U_C3_vis) / 3
P_vis_om = (I_VIS + np.conj(omega) * U_C3_vis + np.conj(omega) ** 2 * (U_C3_vis @ U_C3_vis)) / 3
P_vis_omb = (I_VIS + omega * U_C3_vis + omega ** 2 * (U_C3_vis @ U_C3_vis)) / 3

for name, P_a in [('triv', P_vis_triv), ('omega', P_vis_om), ('omegabar', P_vis_omb)]:
    assert np.linalg.norm(P_a @ P_a - P_a) < TOL, f"vis {name} not idempotent"
    assert np.linalg.norm(P_a - P_a.conj().T) < TOL, f"vis {name} not Hermitian"
    rk = int(round(np.trace(P_a).real))
    assert rk == 4, f"vis {name} rank {rk}, expected 4"
assert np.linalg.norm(P_vis_triv + P_vis_om + P_vis_omb - I_VIS) < TOL

# h-eigenspace projector on visible side
h_target = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
ev_B, V_B = np.linalg.eig(B_P)
mask_h = np.abs(ev_B - h_target) < 1e-6
h_indices = np.where(mask_h)[0]
assert len(h_indices) == 2, f"h-eig multiplicity {len(h_indices)}, expected 2"
V_h_raw = V_B[:, h_indices]
V_h, _ = np.linalg.qr(V_h_raw)
P_h = V_h @ V_h.conj().T
assert abs(np.trace(P_h).real - 2.0) < TOL, f"Tr(P_h) = {np.trace(P_h).real}"

# --------------------------------------------------------------------------
# Step A.2 Spinor side: 8-dim Cl(6, 0) with B-L species projectors
# --------------------------------------------------------------------------
# Brauer-Weyl Pauli construction (matches proofs/foundations/theorem_B3_spinor_fermion.py)

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Gamma = [None] * 7
Gamma[1] = kron(sx, I2, I2)
Gamma[2] = kron(sy, I2, I2)
Gamma[3] = kron(sz, sx, I2)
Gamma[4] = kron(sz, sy, I2)
Gamma[5] = kron(sz, sz, sx)
Gamma[6] = kron(sz, sz, sy)

I_S = np.eye(8, dtype=complex)


def biv(a, b):
    return 0.5 * (Gamma[a] @ Gamma[b] - Gamma[b] @ Gamma[a])


# Verify Clifford relations
for a, b in itertools.product(range(1, 7), repeat=2):
    lhs = Gamma[a] @ Gamma[b] + Gamma[b] @ Gamma[a]
    rhs = 2.0 * (1.0 if a == b else 0.0) * I_S
    assert np.allclose(lhs, rhs, atol=TOL), f"Clifford relation fails {a},{b}"

# Cartan generators
T1 = biv(1, 2) / (2j)
T2 = biv(3, 4) / (2j)
Y_BL = biv(5, 6) / (2j)  # B-L Cartan generator (Spin(2) factor)
G7 = -1j * Gamma[1] @ Gamma[2] @ Gamma[3] @ Gamma[4] @ Gamma[5] @ Gamma[6]

for name, M in [('T1', T1), ('T2', T2), ('Y_BL', Y_BL), ('G7', G7)]:
    assert np.allclose(M, M.conj().T, atol=TOL), f"{name} not Hermitian"

# B-L species projectors: eigenspaces of Y_BL.
# Y_BL has eigenvalues +/- 1/2 (Cartan generator of Spin(2)); half the spinor
# has Y = +1/2, half has Y = -1/2. Per B3 Step 4 + Pati-Salam reading
# (with the (Z/2)^3 convention), these label the {quark axis, lepton axis}
# of one PS family. We take Pi_+ = projector onto Y > 0 eigenspace,
# Pi_- = projector onto Y < 0 eigenspace.

ev_Y, vec_Y = np.linalg.eigh(Y_BL)
mask_pos = ev_Y > 0
mask_neg = ev_Y < 0
n_pos = int(mask_pos.sum())
n_neg = int(mask_neg.sum())
assert n_pos == 4 and n_neg == 4, \
    f"Y_BL eigenvalue split {n_pos}+{n_neg}, expected 4+4"

Pi_Yplus = vec_Y[:, mask_pos] @ vec_Y[:, mask_pos].conj().T
Pi_Yminus = vec_Y[:, mask_neg] @ vec_Y[:, mask_neg].conj().T

assert np.linalg.norm(Pi_Yplus + Pi_Yminus - I_S) < TOL
assert abs(np.trace(Pi_Yplus).real - 4.0) < TOL
assert abs(np.trace(Pi_Yminus).real - 4.0) < TOL
assert np.linalg.norm(Pi_Yplus @ Pi_Yplus - Pi_Yplus) < TOL
assert np.linalg.norm(Pi_Yminus @ Pi_Yminus - Pi_Yminus) < TOL

# Refined species: charged-lepton projector picks out (e_L, e_R) within
# the B-L = -1 sector (lepton axis). Within Pi_Yminus (4-dim), the
# distinction nu vs e is by SU(2)_L iso-spin (T1 + T2)/2 sign on the
# SU(2)_L doublet, but T_L mixes T1 and T2; the simplest readout uses
# the diagonal Cartan T1 sign (which on the SU(2)_L doublet of leptons
# distinguishes nu from e). For the per-species refinement we project
# onto eigenstates of the joint Cartan (T1, T2, Y, G7) using the
# combined operator with incommensurate coefficients (B3 Step 2).

combined_S = T1 + 3.7 * T2 + 11.3 * Y_BL
ev_c, vec_c = np.linalg.eigh(combined_S)
species_label = []
for k in range(8):
    v = vec_c[:, k]
    t1 = round(2 * np.real(v.conj() @ T1 @ v))
    t2 = round(2 * np.real(v.conj() @ T2 @ v))
    y = round(2 * np.real(v.conj() @ Y_BL @ v))
    ch = round(np.real(v.conj() @ G7 @ v))
    # B3 Step 4 dictionary (one valid (Z/2)^3 convention; we adopt the
    # script's explicit table).
    # Determine chirality_sign as in B3 script Step 3 to match
    t1_0 = round(2 * np.real(vec_c[:, 0].conj() @ T1 @ vec_c[:, 0]))
    t2_0 = round(2 * np.real(vec_c[:, 0].conj() @ T2 @ vec_c[:, 0]))
    y_0 = round(2 * np.real(vec_c[:, 0].conj() @ Y_BL @ vec_c[:, 0]))
    ch_0 = round(np.real(vec_c[:, 0].conj() @ G7 @ vec_c[:, 0]))
    chirality_sign = ch_0 * (t1_0 * t2_0 * y_0)
    # SU(2) sector: t1 == t2 -> SU(2)_L sector; t1 == -t2 -> SU(2)_R sector
    sector = "SU2L" if t1 == t2 else "SU2R"
    # Lepton if (chirality_sign * y) == +1 (matches B3 script Step 4
    # convention via ps_content).
    species = "lepton" if (chirality_sign * y == +1) else "quark"
    iso_up = (t1 == +1)
    if species == "lepton":
        name = "nu" if iso_up else "e"
    else:
        name = "u" if iso_up else "d"
    chir_name = "L" if ch == +1 else "R"
    species_label.append(f"{name}_{chir_name}")

# Charged-lepton projector: pick out (e_L, e_R)
charged_lepton_indices = [k for k, s in enumerate(species_label) if s in ("e_L", "e_R")]
neutrino_indices = [k for k, s in enumerate(species_label) if s in ("nu_L", "nu_R")]
up_quark_indices = [k for k, s in enumerate(species_label) if s in ("u_L", "u_R")]
down_quark_indices = [k for k, s in enumerate(species_label) if s in ("d_L", "d_R")]

assert len(charged_lepton_indices) == 2, \
    f"Expected 2 charged-lepton states, got {len(charged_lepton_indices)}"
assert len(neutrino_indices) == 2
assert len(up_quark_indices) == 2
assert len(down_quark_indices) == 2

V_eL = vec_c[:, charged_lepton_indices]
Pi_e = V_eL @ V_eL.conj().T
V_nuL = vec_c[:, neutrino_indices]
Pi_nu = V_nuL @ V_nuL.conj().T
V_uL = vec_c[:, up_quark_indices]
Pi_u = V_uL @ V_uL.conj().T
V_dL = vec_c[:, down_quark_indices]
Pi_d = V_dL @ V_dL.conj().T

for name, P in [("Pi_e", Pi_e), ("Pi_nu", Pi_nu), ("Pi_u", Pi_u), ("Pi_d", Pi_d)]:
    assert np.linalg.norm(P @ P - P) < TOL, f"{name} not idempotent"
    assert abs(np.trace(P).real - 2.0) < TOL, f"Tr({name}) = {np.trace(P).real}"

assert np.linalg.norm(Pi_e + Pi_nu + Pi_u + Pi_d - I_S) < TOL, \
    "fine species projectors do not partition I_S"

# --------------------------------------------------------------------------
# Step B. Total Hamiltonian
# --------------------------------------------------------------------------

DIM = N_VIS * 8  # = 96
I_TOT = np.eye(DIM, dtype=complex)
H_spinor = np.zeros((8, 8), dtype=complex)  # H_spinor = 0 per task plan
H_full = np.kron(H_vis, I_S) + np.kron(I_VIS, H_spinor)
assert np.allclose(H_full, H_full.conj().T, atol=TOL), "H_full not Hermitian"
assert H_full.shape == (96, 96)

# --------------------------------------------------------------------------
# Step C. Jump operators
# --------------------------------------------------------------------------
# k* = 3, established upstream (predictions/k_star.py). Hard-code locally
# (the construction file does not import predictions/* to avoid circular
# import; the prediction-layer file will do that).
K_STAR = 3
gamma = 1.0 / K_STAR  # W4 cancellation rate per step (walker_dynamics Step 4)

# Family I: visible C_3-isotypic, on spinor identity
L_family_I = []
for name, P_a in [('triv', P_vis_triv), ('omega', P_vis_om), ('omegabar', P_vis_omb)]:
    L_family_I.append(np.sqrt(gamma) * np.kron(P_a, I_S))

# Family II: visible directed-edge tensor B-L species projector
# Rate scaling: there are 12 * 2 = 24 jump operators in family II. We use
# rate = 1/k* per channel, same as family I, to keep both families on the
# same axiomatic footing (rate = W4 cancellation rate per step). The
# combined dissipator total rate is then sum_{a,vis} = 3 * (1/k*) = 1/k*
# from family I (since sum_a P_a (x) I = I (x) I and 3 * (1/k*) factor
# from rate^2 sum, no wait, sum_a L_a^dag L_a = (1/k*) sum_a P_a (x) I =
# (1/k*) I_96), plus from family II:
# sum_{e,s} L^dag L = gamma * sum_{e,s} P_e (x) Pi_s = gamma * I (x) I.
# So total = 2 gamma * I, total rate 2/k*.

L_family_II = []
species_projs_II = [('Y_plus', Pi_Yplus), ('Y_minus', Pi_Yminus)]
for e in range(N_VIS):
    P_e = np.zeros((N_VIS, N_VIS), dtype=complex)
    P_e[e, e] = 1.0
    for sname, Pi_s in species_projs_II:
        L_family_II.append(np.sqrt(gamma) * np.kron(P_e, Pi_s))

L_all = L_family_I + L_family_II
print(f"Family I jump operators: {len(L_family_I)} (visible C_3-isotypic)")
print(f"Family II jump operators: {len(L_family_II)} (visible edge x B-L species)")
print(f"Total jump operators: {len(L_all)}")

# Verify unitality:
S_check = sum(L.conj().T @ L for L in L_all)
# Expected: gamma * I_96 (family I) + gamma * I_96 (family II) = 2 gamma I_96
S_expected = 2 * gamma * I_TOT
assert np.linalg.norm(S_check - S_expected) < 1e-10, \
    f"Total dissipator not unital: ||sum L^dL - 2 gamma I|| = " \
    f"{np.linalg.norm(S_check - S_expected)}"
print(f"Total dissipator: sum L^dL = {2 * gamma:.6f} I_96 (unital, OK)")

# Sanity check: family I commutes with U_C3 (x) I, family II individual
# jumps do not (they break C_3).
U_C3_full = np.kron(U_C3_vis, I_S)
for k, L in enumerate(L_family_I):
    err = np.linalg.norm(U_C3_full @ L - L @ U_C3_full)
    assert err < 1e-10, f"Family I jump {k} does not commute with U_C3 (x) I"
# Family II individual jumps DO NOT commute (this is the C_3-breaking content).
breaking_norms = []
for k, L in enumerate(L_family_II):
    err = np.linalg.norm(U_C3_full @ L - L @ U_C3_full)
    breaking_norms.append(err)
assert max(breaking_norms) > 0.5, \
    "Family II jumps unexpectedly preserve C_3 -- construction broken"
print(f"Family I jumps commute with U_C3 (x) I "
      f"(verified, max ||commutator|| = {max(np.linalg.norm(U_C3_full @ L - L @ U_C3_full) for L in L_family_I):.1e})")
print(f"Family II jumps individually break C_3 "
      f"(verified, max ||commutator|| = {max(breaking_norms):.3f})")


# --------------------------------------------------------------------------
# Verify that Pi_e + Pi_nu = Pi_Yminus on lepton axis (or = Pi_Yplus depending
# on B3 convention sign), so the fine projectors refine the coarse ones.
# --------------------------------------------------------------------------
lepton_coarse = Pi_e + Pi_nu
quark_coarse = Pi_u + Pi_d
# The lepton coarse projector should equal one of (Pi_Yplus, Pi_Yminus).
match_minus = np.linalg.norm(lepton_coarse - Pi_Yminus) < TOL
match_plus = np.linalg.norm(lepton_coarse - Pi_Yplus) < TOL
assert match_minus or match_plus, \
    "lepton (Pi_e + Pi_nu) does not match either Y-eigenspace projector"
if match_minus:
    print("Convention: lepton axis = Y < 0 eigenspace (B-L = -1 per PS)")
else:
    print("Convention: lepton axis = Y > 0 eigenspace")

# --------------------------------------------------------------------------
# Verify h-eigenspace C_3 content = (1, 1, 0) (theorem_B5_3_core Step 5)
# --------------------------------------------------------------------------
d_th = float(np.trace(P_vis_triv @ P_h).real)
d_oh = float(np.trace(P_vis_om @ P_h).real)
d_obh = float(np.trace(P_vis_omb @ P_h).real)
assert abs(d_th - 1.0) < TOL, f"Tr(P_triv P_h) = {d_th}, expected 1"
assert abs(d_oh - 1.0) < TOL, f"Tr(P_omega P_h) = {d_oh}, expected 1"
assert abs(d_obh - 0.0) < TOL, f"Tr(P_omegabar P_h) = {d_obh}, expected 0"
print(f"h-eigenspace C_3 content: (Tr(P_t P_h), Tr(P_o P_h), Tr(P_ob P_h)) = "
      f"({d_th:.0f}, {d_oh:.0f}, {d_obh:.0f})")

print()
print("OK: lindblad_spinor_coupled_construction algebraic verifications complete.")
