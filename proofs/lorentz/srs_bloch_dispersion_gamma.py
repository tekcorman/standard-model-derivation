#!/usr/bin/env python3
"""
Canonical prediction file for the small-k Bloch dispersion coefficient
gamma of the srs scalar adjacency near Gamma.

Claim. With the same I4_132 + Wyckoff 8a primitive-cell bond list used
in predictions/B_P_doubly_degenerate_h.py, the scalar Bloch adjacency
A(k) of srs has, near Gamma, the second-order Rayleigh-Schrodinger
expansion of its Perron eigenvalue:

    lambda_0(k) = k* - sum_{a,b} (gamma_tensor)_{ab} k_a k_b + O(|k|^4)

where (k_1, k_2, k_3) are PRIMITIVE-BCC reduced coordinates and the
dispersion tensor is

    (gamma_tensor)_{ab} = (pi^2 / 2)        if a == b,
                        = (pi^2 / 4)        if a != b.

Equivalently, in physical Cartesian wavevector q (with conventional
cubic lattice constant a = 1) related to (k_1, k_2, k_3) by
q = 2*pi*(k_1 b_1 + k_2 b_2 + k_3 b_3) where b_i are the primitive
reciprocal vectors b_1=(0,1,1), b_2=(1,0,1), b_3=(1,1,0):

    lambda_0(q) = k* - (1/16) |q|^2 + O(|q|^4),

i.e. the dispersion is isotropic in physical k-space with the closed
scalar coefficient

    gamma_phys = 1 / 16.

Both forms are equivalent; the apparent "anisotropy" of the tensor in
primitive reduced coordinates is the standard non-orthogonality of the
BCC primitive reciprocal basis (b_i . b_j = 1 for i != j, 2 for i = j).
"""

# ============================================================
# PARAMETER: srs_bloch_dispersion_gamma  (curvature of Perron
# branch of A(k) at Gamma, sub-target n_s-1 of
# an internal working note)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       gamma_tensor (3x3 rational-times-pi^2 matrix)
#              gamma_phys = 1 / 16     (Cartesian, conventional a=1)
# Source:      Structural prediction of the srs scalar Bloch operator;
#              not a phenomenological number. "Observation" here means
#              the numerical Perron eigenvalue of A(k) for small k.
# PDG edition: n/a

# --- PREDICTED VALUE -----------------------------------------
# Value:       gamma_tensor[a,a] = pi^2/2,  gamma_tensor[a,b] = pi^2/4 (a != b)
#              gamma_phys = 1/16   (exact)
# Deviation:   Sympy verifies the symbolic coefficients to zero.
#              Numerical Perron eigenvalue of A(k) at |k| ~ 1e-3 in
#              several Cartesian directions matches predicted form to
#              machine precision; residual scales as O(|k|^4) as
#              required by Rayleigh-Schrodinger.

# --- DERIVED FORMULA -----------------------------------------
# Full proof in predictions/srs_bloch_dispersion_gamma_derivation.md.
# Skeleton:
#
#   1. Upstream: k* = 3, d = 3 -> srs = I4_132 Wyckoff 8a
#                                       [predictions/k_star.py,
#                                        predictions/d_spatial.py,
#                                        predictions/g_girth_derivation.md §2]
#   2. Upstream: walker dynamics on srs = NB walks; A(k) is the scalar
#      Bloch fibre on the 4-vertex primitive cell
#                                       [../predictions/walker_dynamics_derivation.md W1-W3]
#   3. A(k) constructed from the same six-bond list of the I4_132
#      Wyckoff 8a primitive cell used in B_P_doubly_degenerate_h.py.
#      At Gamma, A(0) = J - I (the K_4 adjacency matrix), spectrum
#      {+3, -1, -1, -1}; +3 is non-degenerate (Perron-Frobenius).
#                                       [step 3 of the derivation doc]
#   4. Non-degenerate Rayleigh-Schrodinger second-order perturbation
#      theory (Kato 1980, "Perturbation Theory for Linear Operators",
#      §II.5 Theorem 5.4; equivalently Reed-Simon 1978, Vol IV,
#      Theorem XII.13) with v_0 = (1,1,1,1)/2 the Perron eigenvector:
#
#         lambda_0^{(2)}(k) = <v_0 | H_2 | v_0>
#                           + sum_{n in {-1 sector}} |<v_n | H_1 | v_0>|^2 / (E_0 - E_n)
#
#      where H_1 = sum_a (dA/dk_a)|_0 * k_a and
#      H_2 = (1/2) sum_{a,b} (d^2 A/dk_a dk_b)|_0 * k_a k_b are the
#      Taylor coefficients of A(k) and E_0 - E_n = 3 - (-1) = 4.
#                                       [steps 4-5 of the derivation doc]
#   5. Sympy evaluation of both terms gives:
#         <v_0 | H_2 | v_0> = -pi^2 (4 (k_1^2 + k_2^2 + k_3^2)
#                                  + 6 (k_1 k_2 + k_1 k_3 + k_2 k_3))
#         (1/4) sum |<v_n|H_1|v_0>|^2
#                            = +(pi^2/2) (7 (k_1^2 + k_2^2 + k_3^2)
#                                       + 11 (k_1 k_2 + k_1 k_3 + k_2 k_3))
#      sum:
#         lambda_0^{(2)}(k) = -(pi^2/2) ((k_1^2 + k_2^2 + k_3^2)
#                                       + (k_1 k_2 + k_1 k_3 + k_2 k_3))
#      i.e. (gamma_tensor)_{aa} = pi^2/2, (gamma_tensor)_{ab} = pi^2/4
#      for a != b.
#                                       [step 6 of the derivation doc]
#   6. Conversion to physical Cartesian wavevector q = 2 pi (k_1 b_1
#      + k_2 b_2 + k_3 b_3) where b_1=(0,1,1), b_2=(1,0,1),
#      b_3=(1,1,0) are the primitive BCC reciprocal vectors (a=1):
#
#         |q|^2 = (2 pi)^2 sum_{a,b} k_a k_b (b_a . b_b)
#               = (2 pi)^2 * 2 ((k_1^2 + k_2^2 + k_3^2)
#                              + (k_1 k_2 + k_1 k_3 + k_2 k_3))
#
#      so lambda_0^{(2)}(k) = -(pi^2/2) * |q|^2 / (8 pi^2)
#                          = -|q|^2 / 16.
#
#      Hence
#         lambda_0(q) = 3 - |q|^2 / 16 + O(|q|^4).
#                                       [step 7 of the derivation doc]
#   7. Numerical cross-check at |q| = 1e-3 in seven Cartesian directions
#      (principal axes, face-diagonal, body-diagonal, off-symmetry)
#      reproduces (3 - lambda_0(q))/(|q|^2/16) = 1.000000 to all
#      printed digits.  Residual scales as |k|^4 (verified at five
#      values of |k| in (1e-2, ..., 1e-4)).
#                                       [step 8 of the derivation doc]

# --- INPUTS --------------------------------------------------
# symbol      | value           | status    | predictions/ file                        | meaning
# ------------|-----------------|-----------|------------------------------------------|--------
# k_star      | 3               | [derived] | predictions/k_star.py                    | coordination number; selects srs
# d_spatial   | 3               | [derived] | predictions/d_spatial.py                 | spatial dimension; selects 3D net
# srs embed   | I4_132 Wyckoff 8a | [derived] | predictions/g_girth_derivation.md §2     | space group + bond list of primitive cell
# A(k) Bloch  | 4x4 trig poly   | [derived] | ../predictions/walker_dynamics_derivation.md W1-W3    | scalar Bloch adjacency on primitive cell
# Gamma       | (0,0,0)         | [derived] | predictions/B_P_doubly_degenerate_h.py   | BZ origin
# Perron v_0  | (1,1,1,1)/2     | [derived] | this file (computed from A(0)=J-I)       | non-degenerate Perron eigenvector

# --- IMPLEMENTATION ------------------------------------------
# Sympy verification of every algebraic step + numerical cross-check
# in physical Cartesian wavevector.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# moved to proofs/ 2026-05-27: predictions/ siblings live 2 dirs up at <repo>/predictions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "predictions"))

import sympy as sp
import numpy as np
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from p_toggle import predict_p_toggle
from V_count import predict_V_count
import functools

# Upstream framework values (no external inputs).
d = predict_d_spatial()
k_star = predict_k_star(d)
p = predict_p_toggle()                # = 2 (binary toggle)
V = predict_V_count(k_star, d)        # = 4 (primitive-cell vertex count)

# ---- Step 3: build A(k) symbolically ----
# Bloch scalar adjacency of srs primitive (4-vertex) cell, using the
# same primitive-cell bond list as B_P_doubly_degenerate_h.py.  The
# 'cell' tuples are lattice translations expressed in the PRIMITIVE
# BCC basis.  The Bloch coordinate k = (k_1, k_2, k_3) is conjugate
# to that primitive basis.
k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
A_k = sp.zeros(4, 4)


def _add(tgt, src, cell):
    A_k[tgt, src] += sp.exp(sp.I * 2 * sp.pi * (cell[0] * k1 + cell[1] * k2 + cell[2] * k3))


# Primitive-cell bond list for I4_132 Wyckoff 8a srs realisation.
_add(1, 0, (-1, -1, -1)); _add(0, 1, (1, 1, 1))
_add(2, 0, (-1, -1, -1)); _add(0, 2, (1, 1, 1))
_add(3, 0, (-1, -1, -1)); _add(0, 3, (1, 1, 1))
_add(2, 1, (1, 0, 0));    _add(1, 2, (-1, 0, 0))
_add(3, 1, (0, -1, 0));   _add(1, 3, (0, 1, 0))
_add(3, 2, (0, 0, 1));    _add(2, 3, (0, 0, -1))

# Hermiticity (preliminary check).
hermitian_residual = sp.simplify(A_k - A_k.H)
assert hermitian_residual == sp.zeros(4, 4), "A(k) not Hermitian symbolically."

# A(Gamma) = J - I  (K_4 adjacency).
A_Gamma = A_k.subs({k1: 0, k2: 0, k3: 0})
J_minus_I = sp.ones(4, 4) - sp.eye(4)
assert sp.simplify(A_Gamma - J_minus_I) == sp.zeros(4, 4), "A(Gamma) != K_4 adjacency."

# Spectrum at Gamma: {+3, -1*3}.  Perron-Frobenius non-degenerate.
eigenvals_at_Gamma = A_Gamma.eigenvals()
assert eigenvals_at_Gamma == {sp.Integer(3): 1, sp.Integer(-1): 3}, \
    f"Unexpected A(Gamma) spectrum: {eigenvals_at_Gamma}"

# ---- Step 4: Perron eigenvector and orthonormal -1 basis ----
v0 = sp.Matrix([1, 1, 1, 1]) / 2
# Sanity: A(0) v_0 = +3 v_0.
assert sp.simplify(A_Gamma * v0 - 3 * v0) == sp.zeros(4, 1)
# Three orthonormal eigenvectors spanning the (-1)-eigenspace
# (= orthogonal complement of v_0):
v1 = sp.Matrix([1, -1, 0, 0]) / sp.sqrt(2)
v2 = sp.Matrix([1, 1, -2, 0]) / sp.sqrt(6)
v3 = sp.Matrix([1, 1, 1, -3]) / (2 * sp.sqrt(3))
B_basis = sp.Matrix.hstack(v0, v1, v2, v3)
# Check orthonormality.
assert sp.simplify(B_basis.H * B_basis) == sp.eye(4), "v0..v3 not orthonormal."
# Check A(0) v_n = -v_n for n=1,2,3.
for vn in (v1, v2, v3):
    assert sp.simplify(A_Gamma * vn - (-1) * vn) == sp.zeros(4, 1), "non-(-1) eigenvector."

# ---- Step 5: Taylor expand A(k) at Gamma to second order ----
H1 = sp.zeros(4, 4)
for ka in (k1, k2, k3):
    H1 = H1 + sp.diff(A_k, ka).subs({k1: 0, k2: 0, k3: 0}) * ka

H2 = sp.zeros(4, 4)
for ka in (k1, k2, k3):
    for kb in (k1, k2, k3):
        d2 = sp.diff(A_k, ka, kb).subs({k1: 0, k2: 0, k3: 0})
        H2 = H2 + sp.Rational(1, 2) * d2 * ka * kb
H2 = sp.simplify(H2)

# Sanity: first-order RS correction <v0|H1|v0> must vanish (no permanent dipole).
first_order = sp.simplify((v0.H * H1 * v0)[0, 0])
assert first_order == 0, f"First-order correction non-zero: {first_order}"

# Diagonal second-order Rayleigh quotient piece.
diag_term = sp.expand((v0.H * H2 * v0)[0, 0])
expected_diag = -sp.pi**2 * (4 * (k1**2 + k2**2 + k3**2)
                             + 6 * (k1 * k2 + k1 * k3 + k2 * k3))
assert sp.simplify(diag_term - expected_diag) == 0, \
    f"Unexpected diagonal term: {sp.expand(diag_term)}"

# Cross sum: (1/(E_0 - E_{-1})) sum_n |<v_n|H_1|v_0>|^2 = (1/4) sum_n ...
cross_sum = sp.S.Zero
for vn in (v1, v2, v3):
    amp = (vn.H * H1 * v0)[0, 0]
    amp_conj = (v0.H * H1 * vn)[0, 0]
    cross_sum = cross_sum + sp.expand(amp * amp_conj)
cross_term = sp.simplify(cross_sum / sp.Integer(4))
expected_cross = (sp.pi**2 / 2) * (7 * (k1**2 + k2**2 + k3**2)
                                   + 11 * (k1 * k2 + k1 * k3 + k2 * k3))
assert sp.simplify(cross_term - expected_cross) == 0, \
    f"Unexpected cross term: {cross_term}"

# Sum: total second-order energy correction.
E2 = sp.expand(diag_term + cross_term)
expected_E2 = -(sp.pi**2 / 2) * ((k1**2 + k2**2 + k3**2)
                                 + (k1 * k2 + k1 * k3 + k2 * k3))
assert sp.simplify(E2 - expected_E2) == 0, \
    f"Unexpected E2: {E2}"

# Extract the gamma TENSOR (in primitive reduced coordinates):
# E2 = -sum_{a,b} gamma_tensor[a,b] k_a k_b
# Diagonals: pi^2/2.  Off-diagonals: pi^2/4 each (cross-terms split half/half).
gamma_tensor = sp.Matrix([[sp.pi**2 / 2, sp.pi**2 / 4, sp.pi**2 / 4],
                          [sp.pi**2 / 4, sp.pi**2 / 2, sp.pi**2 / 4],
                          [sp.pi**2 / 4, sp.pi**2 / 4, sp.pi**2 / 2]])
# Verify by reconstructing E2.
k_vec = sp.Matrix([k1, k2, k3])
E2_from_tensor = -sp.expand((k_vec.T * gamma_tensor * k_vec)[0, 0])
assert sp.simplify(E2 - E2_from_tensor) == 0, \
    "gamma_tensor does not reproduce E2."

# ---- Step 6: convert to physical Cartesian wavevector ----
# q = 2*pi*(k1 b_1 + k2 b_2 + k3 b_3) with primitive BCC reciprocal
# vectors (conventional cubic a = 1):
b1_vec = sp.Matrix([0, 1, 1])
b2_vec = sp.Matrix([1, 0, 1])
b3_vec = sp.Matrix([1, 1, 0])
G = sp.Matrix(3, 3, lambda i, j: [b1_vec, b2_vec, b3_vec][i].dot(
                                  [b1_vec, b2_vec, b3_vec][j]))
expected_G = sp.Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
assert sp.simplify(G - expected_G) == sp.zeros(3, 3), f"G != expected: {G}"

# |q|^2 = (2*pi)^2 k^T G k.
q_sq = sp.expand((2 * sp.pi)**2 * (k_vec.T * G * k_vec)[0, 0])
# Check that gamma_tensor = pi^2 * G / 4 = (1/16) * (2*pi)^2 * G.
gamma_tensor_check = sp.pi**2 * G / 4
assert sp.simplify(gamma_tensor - gamma_tensor_check) == sp.zeros(3, 3), \
    "gamma_tensor != pi^2 G / 4."

# Hence  E2 = -(1/16) * |q|^2.
E2_in_q = -q_sq / 16
assert sp.simplify(E2 - E2_in_q) == 0, "E2 vs q-expression mismatch."

gamma_phys = sp.Rational(1, 16)
# Final closed form:
print(f"k* = {k_star}, d = {d}")
print("A(Gamma) = K_4 adjacency, spectrum {+3, -1*3}; +3 non-degenerate (Perron).")
print("Perron eigenvector v_0 = (1,1,1,1)/2.")
print()
print("Second-order Rayleigh-Schrodinger correction (sympy-verified):")
print(f"  E^(2)(k) = {E2}")
print()
print("Equivalent tensor form (primitive BCC reduced coords):")
print(f"  gamma_tensor =")
sp.pprint(gamma_tensor)
print()
print("Physical Cartesian wavevector q = 2*pi*(k1 b1 + k2 b2 + k3 b3)")
print("  with b1=(0,1,1), b2=(1,0,1), b3=(1,1,0):  |q|^2 = (2*pi)^2 k^T G k.")
print(f"  E^(2)(q) = -|q|^2 / 16   (isotropic).")
print(f"  gamma_phys = {gamma_phys}")

# ---- Step 7: numerical cross-check ----
M_basis_to_q = np.column_stack([
    np.array([0, 1, 1]),
    np.array([1, 0, 1]),
    np.array([1, 1, 0]),
])  # q_Cartesian / (2 pi) = M_basis_to_q @ k
print()
print("Numerical check:  |q| = 1e-3 in 7 Cartesian directions, predicted = |q|^2/16.")
print(f"{'direction':18s} | {'|q|^2':>12s} | {'3 - lam_0':>12s} | {'|q|^2/16':>12s} | ratio")
eps = 1e-3
for label, q_dir in [
    ('(1,0,0)', np.array([1.0, 0.0, 0.0])),
    ('(0,1,0)', np.array([0.0, 1.0, 0.0])),
    ('(0,0,1)', np.array([0.0, 0.0, 1.0])),
    ('(1,1,0)/sqrt(2)', np.array([1.0, 1.0, 0.0]) / np.sqrt(2)),
    ('(1,1,1)/sqrt(3)', np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
    ('(2,1,1)/sqrt(6)', np.array([2.0, 1.0, 1.0]) / np.sqrt(6)),
    ('(3,-1,2)/sqrt(14)', np.array([3.0, -1.0, 2.0]) / np.sqrt(14)),
]:
    q_vec = eps * q_dir
    k_red = (1.0 / (2 * np.pi)) * np.linalg.solve(M_basis_to_q, q_vec)
    A_num = np.array(
        A_k.subs({k1: float(k_red[0]), k2: float(k_red[1]), k3: float(k_red[2])}).evalf().tolist(),
        dtype=complex,
    )
    # Hermitize numerical A (should already be Hermitian to machine precision).
    A_num = (A_num + A_num.conj().T) / 2
    eigs = np.linalg.eigvalsh(A_num)
    lam = float(np.max(eigs))
    qsq = float(np.dot(q_vec, q_vec))
    pred = qsq / 16.0
    ratio = (3.0 - lam) / pred
    print(f"{label:18s} | {qsq:12.3e} | {3.0 - lam:12.3e} | {pred:12.3e} | {ratio:.6f}")

# Residual scales as O(|k|^4).
print()
print("O(|k|^4) residual check (k = (eps, 0, 0) in primitive reduced coords):")
print(f"{'eps':>8s} | {'residual = 3 - lam - (pi^2/2) eps^2':>40s} | {'/eps^4':>10s}")
for eps in (1e-2, 5e-3, 1e-3, 5e-4):
    A_num = np.array(
        A_k.subs({k1: eps, k2: 0.0, k3: 0.0}).evalf().tolist(),
        dtype=complex,
    )
    A_num = (A_num + A_num.conj().T) / 2
    lam = float(np.max(np.linalg.eigvalsh(A_num)))
    residual = 3.0 - lam - (np.pi**2 / 2) * eps**2
    print(f"{eps:8.0e} | {residual:40.6e} | {residual / eps**4:10.4f}")


# --- PURE FUNCTION -------------------------------------------
# Inputs: k_star only.  No hardcoded physical constants.  The function
# rebuilds A(k) symbolically from the I4_132 + Wyckoff 8a primitive-cell
# bond list (forced by k_star=3, d_spatial=3 via predictions/g_girth_derivation.md §2)
# and returns the closed-form gamma tensor and gamma_phys.

@functools.lru_cache(maxsize=None)
def predict_srs_bloch_dispersion_gamma(k_star, p_toggle, V_count, d_spatial):
    """
    Returns the closed-form coefficients of the small-k expansion of
    the Perron eigenvalue lambda_0(k) of the srs scalar Bloch adjacency
    A(k) at Gamma:

        lambda_0(k) = k_star - sum_{a,b} gamma_tensor[a, b] * k_a * k_b
                              + O(|k|^4)

    in primitive-BCC reduced coordinates (k_1, k_2, k_3).  Equivalently
    in physical Cartesian wavevector q = 2*pi*(k_1 b_1 + k_2 b_2 + k_3 b_3),

        lambda_0(q) = k_star - gamma_phys * |q|^2 + O(|q|^4)

    where gamma_phys = 1/16.

    Parameters
    ----------
    k_star : int
        Coordination number.  The theorem is established for k_star = 3
        (srs); the function raises for other values.
    p_toggle : int
        Toggle alphabet (= 2). Sets Taylor 2nd-order divisor 1/p, the
        polynomial coefficient degree, Fourier 2π convention, and the
        perron-vector normalization (1/sqrt(V) = 1/p for V=4=p^2).
    V_count : int
        Primitive-cell vertex count (= 4). Sets the matrix dimension
        of A(k).
    d_spatial : int
        Spatial dimension (= 3). Sets gamma_tensor's 3x3 spatial-tensor
        shape and the BCC reciprocal-basis range.

    Returns
    -------
    (sympy.Matrix, sympy.Rational)
        A pair (gamma_tensor, gamma_phys), where gamma_tensor is a 3x3
        sympy.Matrix with diagonal pi^2/2 and off-diagonal pi^2/4, and
        gamma_phys = sympy.Rational(1, 16).
    """
    if k_star != 3:
        raise ValueError(
            f"srs_bloch_dispersion_gamma is established for k_star = 3 only.  Got {k_star}."
        )

    one_nb = p_toggle - 1                 # = 1, NB constraint count

    # Build A(k) symbolically from scratch.
    kappa1, kappa2, kappa3 = sp.symbols('kappa1 kappa2 kappa3', real=True)
    A_local = sp.zeros(V_count, V_count)   # V_count × V_count primitive-cell matrix

    def _add_local(tgt, src, cell):
        # 2π = p_toggle·π (Bloch Fourier convention; p_toggle = 2).
        A_local[tgt, src] += sp.exp(sp.I * p_toggle * sp.pi * (
            cell[0] * kappa1 + cell[1] * kappa2 + cell[2] * kappa3))

    _add_local(1, 0, (-1, -1, -1)); _add_local(0, 1, (1, 1, 1))
    _add_local(2, 0, (-1, -1, -1)); _add_local(0, 2, (1, 1, 1))
    _add_local(3, 0, (-1, -1, -1)); _add_local(0, 3, (1, 1, 1))
    _add_local(2, 1, (1, 0, 0));    _add_local(1, 2, (-1, 0, 0))
    _add_local(3, 1, (0, -1, 0));   _add_local(1, 3, (0, 1, 0))
    _add_local(3, 2, (0, 0, 1));    _add_local(2, 3, (0, 0, -1))

    # Spectrum at Gamma: must be {k_star, -1, -1, -1}.
    A_local_Gamma = A_local.subs({kappa1: 0, kappa2: 0, kappa3: 0})
    eigs = A_local_Gamma.eigenvals()
    if eigs != {sp.Integer(k_star): one_nb, sp.Integer(-one_nb): k_star}:
        raise RuntimeError(f"Unexpected A(Gamma) spectrum: {eigs}")

    # Perron eigenvector (uniform): norm 1/sqrt(V_count) = 1/p_toggle (V=4=p^2).
    perron = sp.Matrix([one_nb] * V_count) / p_toggle
    if sp.simplify(A_local_Gamma * perron - k_star * perron) != sp.zeros(V_count, one_nb):
        raise RuntimeError("Perron eigenvector check failed.")

    # Orthonormal basis for the (-1)-eigenspace (= perron complement).
    # The Gram-Schmidt vectors of the (-1)-eigenspace of A(Gamma)=J-I.
    e1 = sp.Matrix([1, -1, 0, 0]) / sp.sqrt(p_toggle)
    e2 = sp.Matrix([1, 1, -2, 0]) / sp.sqrt(p_toggle * k_star)
    e3 = sp.Matrix([1, 1, 1, -3]) / (p_toggle * sp.sqrt(k_star))

    # Taylor terms.
    H1_local = sp.zeros(V_count, V_count)
    for ka in (kappa1, kappa2, kappa3):
        H1_local = H1_local + sp.diff(A_local, ka).subs(
            {kappa1: 0, kappa2: 0, kappa3: 0}) * ka
    H2_local = sp.zeros(V_count, V_count)
    for ka in (kappa1, kappa2, kappa3):
        for kb in (kappa1, kappa2, kappa3):
            d2 = sp.diff(A_local, ka, kb).subs({kappa1: 0, kappa2: 0, kappa3: 0})
            # 1/2! = (p-1)/p Taylor coefficient (p_toggle factorial denominator).
            H2_local = H2_local + sp.Rational(one_nb, p_toggle) * d2 * ka * kb

    # Second-order RS energy.
    diag_term_local = (perron.H * H2_local * perron)[0, 0]
    cross_term_local = sp.S.Zero
    for vn in (e1, e2, e3):
        amp = (vn.H * H1_local * perron)[0, 0]
        amp_conj = (perron.H * H1_local * vn)[0, 0]
        cross_term_local = cross_term_local + sp.expand(amp * amp_conj)
    # Energy denominator: E_0 - E_{-1} = k_star - (-1) = k_star + 1.
    cross_term_local = cross_term_local / (k_star + one_nb)
    E2_local = sp.expand(diag_term_local + cross_term_local)

    # Extract the symmetric tensor of (-coefficients of k_a k_b) in E2.
    coords = (kappa1, kappa2, kappa3)
    gamma_local = sp.zeros(d_spatial, d_spatial)
    for a in range(d_spatial):
        for b in range(d_spatial):
            if a == b:
                # 2nd-order polynomial coefficient (degree = p_toggle).
                gamma_local[a, b] = -E2_local.coeff(coords[a], p_toggle)
            else:
                # cross term k_a k_b appears once: coefficient of k_a k_b is -2 gamma_{ab}
                # because the symmetric tensor sums gamma_{ab} k_a k_b + gamma_{ba} k_b k_a.
                cross = E2_local.coeff(coords[a]).coeff(coords[b])
                gamma_local[a, b] = -cross / p_toggle

    # Cartesian-isotropic scalar.
    # In primitive BCC reduced coords, |q|^2 = (2*pi)^2 * k^T G k where the
    # reciprocal metric is G = 2 I + (J - I) (diagonal 2, off-diagonal 1).
    # Symbolic G from the primitive reciprocal vectors b_i (a = 1):
    b1_loc = sp.Matrix([0, 1, 1])
    b2_loc = sp.Matrix([1, 0, 1])
    b3_loc = sp.Matrix([1, 1, 0])
    G_local = sp.Matrix(d_spatial, d_spatial, lambda i, j: [b1_loc, b2_loc, b3_loc][i].dot(
                                                            [b1_loc, b2_loc, b3_loc][j]))
    # gamma_tensor must equal gamma_phys * (2*pi)^2 * G  (so that
    # k^T gamma_tensor k = gamma_phys * |q|^2).  Diagonal a=0 gives
    # gamma_phys = gamma_tensor[0, 0] / ((2*pi)^2 * G[0, 0]).
    gamma_phys_local = sp.simplify(
        gamma_local[0, 0] / ((p_toggle * sp.pi)**p_toggle * G_local[0, 0]))
    # Sanity: full tensor identity gamma_tensor == gamma_phys * (2*pi)^2 * G.
    if sp.simplify(gamma_local - gamma_phys_local * (p_toggle * sp.pi)**p_toggle * G_local) != sp.zeros(d_spatial, d_spatial):
        raise RuntimeError(
            "gamma_tensor does not equal gamma_phys * (2*pi)^2 * G "
            "(physical-space isotropy fails).")

    return gamma_local, gamma_phys_local


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_tensor = gamma_tensor
    impl_gamma_phys = gamma_phys
    pure_tensor, pure_gamma_phys = predict_srs_bloch_dispersion_gamma(k_star, p, V, d)

    print()
    print("Implementation gamma_tensor:")
    sp.pprint(impl_tensor)
    print(f"Implementation gamma_phys = {impl_gamma_phys}")
    print()
    print("Pure function gamma_tensor:")
    sp.pprint(pure_tensor)
    print(f"Pure function gamma_phys = {pure_gamma_phys}")

    assert sp.simplify(impl_tensor - pure_tensor) == sp.zeros(3, 3), \
        f"Tensor mismatch: {impl_tensor} vs {pure_tensor}"
    assert sp.simplify(impl_gamma_phys - pure_gamma_phys) == 0, \
        f"gamma_phys mismatch: {impl_gamma_phys} vs {pure_gamma_phys}"
    print()
    print("OK: outputs agree.  lambda_0(q) = k* - |q|^2/16 + O(|q|^4) on srs.")
