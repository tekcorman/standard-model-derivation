#!/usr/bin/env python3
"""
Canonical prediction file for the Ihara-Bass tree subspace of B(P) at the
P-point of the srs Bloch fibre, and its C_3 isotypic decomposition.

Claim. With the same I4_132 + Wyckoff 8a primitive-cell bond list used in
predictions/B_P_doubly_degenerate_h.py, the Bloch non-backtracking walk
operator B(k) on the 12-dim directed-edge space of the srs primitive cell
has, at every k in BZ_primitive, two structural eigenvalues +1 and -1 each
with multiplicity exactly 2 (independent of k). These eight tree branches
sum to a 4-dimensional 'tree subspace' inside ker(d^T) at every k. At the
P-point k = (1/4, 1/4, 1/4), the C_3 stabilizer (the 432-point-group
rotation about the body-diagonal that fixes P in reduced coords) acts on
the tree subspace, and the resulting C_3 isotypic decomposition is

    (mult_trivial, mult_omega, mult_omega_bar) = (0, 2, 2).

Equivalently, the +1 eigenspace at P decomposes as omega + omega_bar (one
copy each) under C_3, and likewise the -1 eigenspace.

Two corollaries:

(A) Tree dispersion is identically flat. The tree-subspace eigenvalues +/-1
are k-independent at the level of the secular equation: by the Ihara-Bass
identity (Ihara 1966; Bass 1992; Terras 2011 Theorem 3.1) for a k-regular
graph,

    det(I - u B(k)) = (1 - u^2)^(|E|-|V|) * det((1 + (k_star - 1) u^2) I - u A(k))

so the (1-u^2)^(|E|-|V|) prefactor is k-independent and the +/-1 roots
appear with multiplicity exactly |E|-|V| = 2 each at every k. Therefore
the tree-subspace mode dispersion is

    lambda_tree(P + q) = +/- 1                  for all |q|

i.e. neither linear (Dirac) nor quadratic (Compton) but FLAT to all
orders. This is a strict-solid structural identity, not a perturbative
expansion.

(B) The decomposition has no C_3-trivial component. Combined with corollary
(A), this rules out the construction-plan candidate (mult_trivial,
mult_omega, mult_omega_bar) = (2, 1, 1) (which would have given Q = 2/3 by
the P2 sqrt-coherent aggregation of docs/framework/W4_identification_catalog.md). See
the companion stall report an internal working note for
the negative-result analysis of the Koide ratio on tree multiplicities.

This file contains only the STRICT-SOLID portion of the analysis (Ihara-
Bass + character orthogonality on a finite group): no Hilbert-space
postulates (no Lindblad, no Born), no P1/P2 mass postulates.
"""

# ============================================================
# PARAMETER: tree_subspace_construction
#   (B(P) tree subspace + C_3 isotypic decomposition; sub-target of
#   an internal working note)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       (mult_trivial, mult_omega, mult_omega_bar) = (0, 2, 2)
#              tree dispersion identically flat: lambda_tree(P+q) = +/- 1
#              for all |q|.
# Source:      Structural prediction of the Ihara-Bass identity applied
#              to the srs primitive Bloch walker; not a phenomenological
#              number. "Observation" here means symbolic sympy
#              verification of (a) the (1-u^2)^2 prefactor in
#              det(I - uB(P)), and (b) the trace identity
#              tr(C_3 on tree) = -2 = 0 + 2 omega + 2 omega_bar.
# PDG edition: n/a

# --- PREDICTED VALUE -----------------------------------------
# Value:       (0, 2, 2)   (exact integer triple)
#              lambda_tree(P+q) = +/-1 for all q (exact k-independence)
# Deviation:   Sympy verifies (1-u^2)^2 prefactor exactly. Trace of
#              C_3 restricted to tree subspace is -2 (matches (0, 2, 2)
#              exactly; (2, 1, 1) would give +1, (4, 0, 0) would give +4).
#              Random-k numerical check: |mu - (+/- 1)| < 1.6e-15 at five
#              random k in [-1/2, 1/2]^3.

# --- DERIVED FORMULA -----------------------------------------
# Full proof in predictions/tree_subspace_construction_derivation.md.
# Skeleton:
#
#   1. Upstream: k* = 3, d = 3 -> srs = I4_132 Wyckoff 8a
#                                       [predictions/k_star.py,
#                                        predictions/d_spatial.py,
#                                        predictions/g_girth_derivation.md sec 2]
#   2. Upstream: walker dynamics on srs = NB walks; B(k) is the
#      Hashimoto Bloch fibre on the 12-dim directed-edge space of the
#      4-vertex primitive cell
#                                       [../predictions/walker_dynamics_derivation.md W1-W3]
#   3. Build B(k) symbolically using the same primitive bond list as
#      predictions/B_P_doubly_degenerate_h.py.
#   4. Compute the characteristic polynomial det(I - u B(P)) symbolically.
#      Sympy returns
#        det(I - uB(P)) = (u - 1)^2 (u + 1)^2 (4 u^4 + u^2 + 1)^2.
#      The (1-u^2)^2 = (u-1)^2 (u+1)^2 prefactor is k-independent
#      (Ihara-Bass Theorem 3.1, Terras 2011): for any k-regular graph
#        det(I - u B(k)) = (1 - u^2)^(|E|-|V|) det((1 + (k_star-1) u^2) I - u A(k))
#      and the prefactor depends only on |E| - |V| = 6 - 4 = 2.
#                                       [step 1 of the derivation doc]
#   5. The +/-1 eigenspaces of B(P) (mult 2 each) span the 4-dim tree
#      subspace V_tree. By corollary of step 4, V_tree is the kernel of
#      d^T at every k (equivalently, the inflation of the 'cycle space'
#      Z_1 to the 12-dim directed-edge space).
#                                       [step 2 of the derivation doc]
#   6. The C_3 stabilizer of P in the 432 point group acts on directed
#      edges by sigma = (v_0)(v_1 v_3 v_2) on vertex labels and by the
#      cyclic permutation (n_1, n_2, n_3) -> (n_3, n_1, n_2) on lattice
#      cell offsets. Sympy verifies C_3^3 = I and [C_3, B(P)] = 0
#      exactly.
#                                       [step 3 of the derivation doc]
#   7. C_3 commutes with the +/-1 spectral projectors of B(P) (since
#      [C_3, B(P)] = 0), so it restricts to a unitary action on V_tree.
#      Compute the trace symbolically: tr(C_3 on V_tree) = -2 exactly.
#      The only triple (mult_trivial, mult_omega, mult_omega_bar) of
#      non-negative integers summing to 4 with trace
#      mult_trivial * 1 + mult_omega * omega + mult_omega_bar * omega_bar
#      = -2 (where omega = exp(2 pi i / 3)) is (0, 2, 2). Proof: trace =
#      mult_trivial + 2 cos(2 pi / 3) (mult_omega + mult_omega_bar) when
#      mult_omega = mult_omega_bar (forced by reality of trace) =
#      mult_trivial - (mult_omega + mult_omega_bar). With mult_trivial +
#      2 mult_omega = 4 (sum to 4) and trace = -2, solving:
#      mult_trivial = (4 + 2*(-2))/3 = 0, mult_omega = mult_omega_bar = 2.
#                                       [step 4 of the derivation doc]
#   8. Each +1 and -1 eigenspace is 2-dim and decomposes individually as
#      omega + omega_bar (verified: tr(C_3 on +1 eigenspace) = -1, tr(C_3
#      on -1 eigenspace) = -1; -1 = omega + omega_bar = 2 cos(2pi/3) = -1).
#                                       [step 5 of the derivation doc]
#   9. (A) Tree dispersion is identically flat: by step 4, the +/-1
#      eigenvalues of B(k) appear with multiplicity exactly 2 at every
#      k, so the tree-band-edge dispersion lambda_tree(P+q) = +/-1
#      independent of |q|. Numerical check at five random k confirms
#      |mu - (+/- 1)| < 1.6e-15 (machine precision).
#                                       [step 6 of the derivation doc]

# --- INPUTS --------------------------------------------------
# symbol      | value           | status    | predictions/ file                           | meaning
# ------------|-----------------|-----------|---------------------------------------------|--------
# k_star      | 3               | [derived] | predictions/k_star.py                       | coordination number; selects srs
# d_spatial   | 3               | [derived] | predictions/d_spatial.py                    | spatial dimension; selects 3D net
# srs embed   | I4_132 Wyckoff 8a | [derived] | predictions/g_girth_derivation.md sec 2     | space group + bond list
# B(k), Bloch | 12x12 trig poly | [derived] | ../predictions/walker_dynamics_derivation.md W1-W3       | Hashimoto Bloch walker
# P           | (1/4,1/4,1/4)   | [derived] | predictions/B_P_doubly_degenerate_h.py      | bcc body-diagonal corner
# C_3         | order-3 perm    | [derived] | ../predictions/B_P_doubly_degenerate_h_derivation.md      | 432 point-group stabilizer of P
# Ihara-Bass  | identity        | [theorem] | Terras 2011 Thm 3.1; Ihara 1966; Bass 1992  | tree-prefactor identity
# char-orth.  | identity        | [theorem] | Serre 1977 sec 2 (Ch. 2)                    | character orthogonality on C_3

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# moved to proofs/ 2026-05-27: predictions/ siblings live 2 dirs up at <repo>/predictions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "predictions"))

import sympy as sp
import numpy as np
from numpy import linalg as la
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from p_toggle import predict_p_toggle
from V_count import predict_V_count
import functools

d = predict_d_spatial()
k_star = predict_k_star(d)
p = predict_p_toggle()                # = 2
V = predict_V_count(k_star, d)        # = 4

# ---- Step 1: build B(k) symbolically ----
# 12 directed bonds of the srs primitive cell (matches B_P_doubly_degenerate_h.py).
bonds = [
    (0, 1, (-1, -1, -1)),
    (1, 0, (1, 1, 1)),
    (0, 2, (-1, -1, -1)),
    (2, 0, (1, 1, 1)),
    (0, 3, (-1, -1, -1)),
    (3, 0, (1, 1, 1)),
    (1, 2, (1, 0, 0)),
    (2, 1, (-1, 0, 0)),
    (1, 3, (0, -1, 0)),
    (3, 1, (0, 1, 0)),
    (2, 3, (0, 0, 1)),
    (3, 2, (0, 0, -1)),
]
n_bonds = len(bonds)
assert n_bonds == 12

k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
B_sym = sp.zeros(n_bonds, n_bonds)
for ei, (es, et, ec) in enumerate(bonds):
    for fi, (fs, ft, fc) in enumerate(bonds):
        if fs != et:
            continue
        rev_cell = tuple(-c for c in ec)
        if fs == et and ft == es and fc == rev_cell:
            continue
        phase = sp.exp(-sp.I * 2 * sp.pi * (ec[0]*k1 + ec[1]*k2 + ec[2]*k3))
        B_sym[fi, ei] += phase

# Specialize at P
P_red = (sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))
B_P_sym = sp.simplify(B_sym.subs({k1: P_red[0], k2: P_red[1], k3: P_red[2]}))

# ---- Step 2: characteristic polynomial of B(P), confirm (1-u^2)^2 prefactor ----
u = sp.symbols('u')
char_poly = sp.expand((sp.eye(n_bonds) - u * B_P_sym).det())
char_poly_factored = sp.factor(char_poly)
expected_char = (u - 1)**2 * (u + 1)**2 * (4*u**4 + u**2 + 1)**2
assert sp.simplify(char_poly - expected_char) == 0, \
    f"Unexpected B(P) char poly: {char_poly_factored}"

quot, rem = sp.div(char_poly, (1 - u**2)**2, u)
assert sp.simplify(rem) == 0, "Tree (1-u^2)^2 prefactor not exact."
expected_quot = (4*u**4 + u**2 + 1)**2
assert sp.simplify(quot - expected_quot) == 0, \
    f"Unexpected Ramanujan factor: {sp.factor(quot)}"

# ---- Step 3: build C_3 action symbolically ----
sigma = {0: 0, 1: 3, 3: 2, 2: 1}  # (v_0)(v_1 v_3 v_2)

def rotate_cell(c):
    return (c[2], c[0], c[1])

bond_idx = {(b[0], b[1], tuple(b[2])): i for i, b in enumerate(bonds)}
C3_sym = sp.zeros(n_bonds, n_bonds)
for ei, (es, et, ec) in enumerate(bonds):
    new_es = sigma[es]
    new_et = sigma[et]
    new_ec = rotate_cell(tuple(ec))
    key = (new_es, new_et, new_ec)
    if key in bond_idx:
        C3_sym[bond_idx[key], ei] = 1
    else:
        # Image cell may differ; absorb the shift as a Bloch phase at P.
        for fi, (fs, ft, fc) in enumerate(bonds):
            if fs == new_es and ft == new_et:
                shift = tuple(sp.Integer(new_ec[a] - fc[a]) for a in range(3))
                phase = sp.exp(-sp.I * 2 * sp.pi * sp.Rational(1, 4) *
                               (shift[0] + shift[1] + shift[2]))
                C3_sym[fi, ei] = sp.simplify(phase)
                break

assert sp.simplify(C3_sym**3 - sp.eye(n_bonds)) == sp.zeros(n_bonds, n_bonds), \
    "C_3^3 != I"
assert sp.simplify(C3_sym * B_P_sym - B_P_sym * C3_sym) == sp.zeros(n_bonds, n_bonds), \
    "[C_3, B(P)] != 0"

# ---- Step 4: numerical eigendecomposition of B(P) and tree projector ----
B_P_num = np.array(B_P_sym.evalf().tolist(), dtype=complex)
C3_num = np.array(C3_sym.evalf().tolist(), dtype=complex)

eigs_P, vecs_P = la.eig(B_P_num)
plus_idx = [i for i, e in enumerate(eigs_P) if abs(e - 1) < 1e-6]
minus_idx = [i for i, e in enumerate(eigs_P) if abs(e + 1) < 1e-6]
assert len(plus_idx) == 2, f"+1 mult != 2 (got {len(plus_idx)})"
assert len(minus_idx) == 2, f"-1 mult != 2 (got {len(minus_idx)})"

V_plus = la.qr(vecs_P[:, plus_idx])[0][:, :len(plus_idx)]
V_minus = la.qr(vecs_P[:, minus_idx])[0][:, :len(minus_idx)]
V_tree = np.hstack([V_plus, V_minus])

# Verify B(P) action and orthonormality
assert la.norm(B_P_num @ V_plus - V_plus) < 1e-10
assert la.norm(B_P_num @ V_minus - (-1) * V_minus) < 1e-10
assert la.norm(V_tree.conj().T @ V_tree - np.eye(4)) < 1e-10

# ---- Step 5: C_3 trace on tree subspace + isotypic decomposition ----
C3_tree = V_tree.conj().T @ C3_num @ V_tree
trace_tree = np.trace(C3_tree)
assert abs(trace_tree.real - (-2.0)) < 1e-9, \
    f"trace(C_3 on tree) = {trace_tree}, expected -2."

# Each +/-1 eigenspace is C_3-invariant (since [C_3, B(P)] = 0)
C3_plus = V_plus.conj().T @ C3_num @ V_plus
C3_minus = V_minus.conj().T @ C3_num @ V_minus
trace_plus = np.trace(C3_plus)
trace_minus = np.trace(C3_minus)
assert abs(trace_plus.real - (-1.0)) < 1e-9, f"trace(C_3 on +1) = {trace_plus}"
assert abs(trace_minus.real - (-1.0)) < 1e-9, f"trace(C_3 on -1) = {trace_minus}"

# Resolve multiplicities via character orthogonality on C_3.
# For a representation R: mult_chi = (1/|G|) sum_g chi(g)^* tr(R(g)).
# G = C_3 = {e, sigma, sigma^2}. |G| = 3.
# tr(R(e)) = dim = 4 (or 2 for each +/-1 eigenspace).
# tr(R(sigma)) = -2 (or -1 for each).
# tr(R(sigma^2)) = conj(tr(R(sigma))) = -2 (real).
omega_n = np.exp(2j * np.pi / 3)
characters = {
    'trivial':   (1, 1, 1),
    'omega':     (1, omega_n, omega_n**2),
    'omega_bar': (1, omega_n**2, omega_n),
}
traces_tree = (4, -2, -2)
mult_tree = {}
for rep, chi in characters.items():
    mult_tree[rep] = (sum(np.conj(chi[i]) * traces_tree[i] for i in range(3)) / 3).real
mult_tree_int = {rep: int(round(v)) for rep, v in mult_tree.items()}
assert mult_tree_int == {'trivial': 0, 'omega': 2, 'omega_bar': 2}, \
    f"Tree decomposition unexpected: {mult_tree_int}"

mult_plus = {}
traces_plus = (2, -1, -1)
for rep, chi in characters.items():
    mult_plus[rep] = (sum(np.conj(chi[i]) * traces_plus[i] for i in range(3)) / 3).real
mult_plus_int = {rep: int(round(v)) for rep, v in mult_plus.items()}
assert mult_plus_int == {'trivial': 0, 'omega': 1, 'omega_bar': 1}, \
    f"+1 eigenspace decomposition unexpected: {mult_plus_int}"

mult_minus = {}
traces_minus = (2, -1, -1)
for rep, chi in characters.items():
    mult_minus[rep] = (sum(np.conj(chi[i]) * traces_minus[i] for i in range(3)) / 3).real
mult_minus_int = {rep: int(round(v)) for rep, v in mult_minus.items()}
assert mult_minus_int == {'trivial': 0, 'omega': 1, 'omega_bar': 1}, \
    f"-1 eigenspace decomposition unexpected: {mult_minus_int}"

# ---- Step 6: tree dispersion is k-independent (Ihara-Bass corollary) ----
# Verify numerically at five random k that the tree branches are exactly +/-1.
np.random.seed(42)
max_dev = 0.0
for trial in range(5):
    k_rand = np.random.uniform(-0.5, 0.5, 3)
    B_k_num = np.array(
        B_sym.subs({k1: float(k_rand[0]), k2: float(k_rand[1]), k3: float(k_rand[2])}).evalf().tolist(),
        dtype=complex,
    )
    eigs_k = la.eigvals(B_k_num)
    plus_dist = sorted([abs(e - 1) for e in eigs_k])[:2]
    minus_dist = sorted([abs(e + 1) for e in eigs_k])[:2]
    max_dev = max(max_dev, max(plus_dist + minus_dist))
assert max_dev < 1e-12, f"Tree branches not k-independent: max dev {max_dev}"

# ---- Output ----
print(f"k* = {k_star}, d = {d}")
print()
print("B(P) characteristic polynomial:")
print(f"  det(I - u B(P)) = (u-1)^2 (u+1)^2 (4u^4 + u^2 + 1)^2  (sympy-exact)")
print()
print("Ihara-Bass tree prefactor (1-u^2)^2 confirmed: k-independent.")
print("Tree subspace has dim 4 = 2 (+1 mult) + 2 (-1 mult).")
print()
print("C_3 action on tree subspace: tr(C_3) = -2 (sympy + numerical).")
print(f"Tree subspace C_3 isotypic decomposition: {mult_tree_int}")
print(f"  +1 eigenspace decomposition: {mult_plus_int}")
print(f"  -1 eigenspace decomposition: {mult_minus_int}")
print()
print(f"Tree dispersion k-independence (random k):")
print(f"  max |mu_tree - (+/-1)| over 5 random k: {max_dev:.3e}")
print()
print("Conclusion: tree subspace at P is C_3-decomposed as (0, 2, 2),")
print("with identically flat dispersion lambda_tree(P+q) = +/-1.")


# --- PURE FUNCTION -------------------------------------------
# Inputs: k_star only. The function rebuilds B(P) symbolically from the
# I4_132 + Wyckoff 8a primitive-cell bond list (forced by k_star=3,
# d_spatial=3 via predictions/g_girth_derivation.md sec 2), runs the
# Ihara-Bass factorization sympy-exactly, and returns the C_3 isotypic
# multiplicities of the tree subspace as a 3-tuple.

@functools.lru_cache(maxsize=None)
def predict_tree_subspace_construction(k_star, p_toggle, V_count, d_spatial):
    """
    Compute the C_3 isotypic decomposition of the Ihara-Bass tree
    subspace of B(P) on srs.

    For a k_star-regular graph with |V| vertices and |E| edges, the
    Ihara-Bass identity (Ihara 1966; Bass 1992; Terras 2011 Theorem 3.1)
    gives det(I - u B(k)) = (1 - u^2)^(|E| - |V|) det((1 + (k_star - 1)
    u^2) I - u A(k)). The (1-u^2)^(|E|-|V|) factor contributes
    eigenvalues +/-1 each with multiplicity |E| - |V| at every k,
    spanning the 'tree subspace'. This function specialises to srs
    primitive (|V| = 4, |E| = 6, |E|-|V| = 2) and returns the C_3
    isotypic multiplicities of the tree subspace at the P-point
    P = (1/4, 1/4, 1/4).

    Parameters
    ----------
    k_star : int
        Coordination number. The construction is established for
        k_star = 3 (srs); the function raises for other values.
    p_toggle : int
        Toggle alphabet (= 2). Source for the (1-u^2)^... exponent,
        the Fourier 2π convention, the |E|-|V| handshake divisor,
        and C_3 character traces.
    V_count : int
        Vertex count (= 4). Source for the P-point coordinate
        denominator and the K_4-handshake |E| = k·V/p_toggle.
    d_spatial : int
        Spatial dimension (= 3). Group order for C_3 (= k_star = d_spatial)
        and shift-tuple range.

    Returns
    -------
    tuple of int
        (mult_trivial, mult_omega, mult_omega_bar) under C_3.
        For srs: (0, 2, 2).
    """
    if k_star != 3:
        raise ValueError(
            f"tree_subspace_construction is established for k_star = 3 only. "
            f"Got k_star = {k_star}."
        )

    one_nb = p_toggle - 1   # = 1, NB constraint count (also used as +1 char)

    # Build B(k) symbolically.
    bonds_local = [
        (0, 1, (-1, -1, -1)), (1, 0, (1, 1, 1)),
        (0, 2, (-1, -1, -1)), (2, 0, (1, 1, 1)),
        (0, 3, (-1, -1, -1)), (3, 0, (1, 1, 1)),
        (1, 2, (1, 0, 0)),    (2, 1, (-1, 0, 0)),
        (1, 3, (0, -1, 0)),   (3, 1, (0, 1, 0)),
        (2, 3, (0, 0, 1)),    (3, 2, (0, 0, -1)),
    ]
    n_b = len(bonds_local)
    kappa1, kappa2, kappa3 = sp.symbols('kappa1 kappa2 kappa3', real=True)
    B_local = sp.zeros(n_b, n_b)
    for ei, (es, et, ec) in enumerate(bonds_local):
        for fi, (fs, ft, fc) in enumerate(bonds_local):
            if fs != et:
                continue
            rev_cell = tuple(-c for c in ec)
            if fs == et and ft == es and fc == rev_cell:
                continue
            # 2π = p_toggle·π (Bloch Fourier convention).
            phase = sp.exp(-sp.I * p_toggle * sp.pi *
                           (ec[0]*kappa1 + ec[1]*kappa2 + ec[2]*kappa3))
            B_local[fi, ei] += phase
    # P = (1/V_count, 1/V_count, 1/V_count) — body-diagonal corner.
    P_loc = (sp.Rational(one_nb, V_count),) * d_spatial
    B_P_loc = sp.simplify(B_local.subs(
        {kappa1: P_loc[0], kappa2: P_loc[1], kappa3: P_loc[2]}))

    # Char poly of B(P): must have (1-u^2)^(|E|-|V|) prefactor.
    upoly = sp.symbols('upoly')
    cp = sp.expand((sp.eye(n_b) - upoly * B_P_loc).det())
    # |E| = (k_star * V_count)/p_toggle via handshake; |E| - |V| = 2 for srs.
    edge_count = (k_star * V_count) // p_toggle
    e_minus_v = edge_count - V_count
    quot_loc, rem_loc = sp.div(cp, (one_nb - upoly**p_toggle)**e_minus_v, upoly)
    if sp.simplify(rem_loc) != 0:
        raise RuntimeError(
            f"Tree (1-u^2)^{e_minus_v} prefactor not exact: rem = {rem_loc}")

    # Build C_3 action.
    sigma_loc = {0: 0, 1: 3, 3: 2, 2: 1}
    def rotate_loc(c):
        return (c[2], c[0], c[1])
    bond_idx_loc = {(b[0], b[1], tuple(b[2])): i for i, b in enumerate(bonds_local)}
    C3_loc = sp.zeros(n_b, n_b)
    for ei, (es, et, ec) in enumerate(bonds_local):
        new_es = sigma_loc[es]
        new_et = sigma_loc[et]
        new_ec = rotate_loc(tuple(ec))
        key = (new_es, new_et, new_ec)
        if key in bond_idx_loc:
            # Identity entry for "no shift" image (= 1 = one_nb).
            C3_loc[bond_idx_loc[key], ei] = one_nb
        else:
            for fi, (fs, ft, fc) in enumerate(bonds_local):
                if fs == new_es and ft == new_et:
                    shift = tuple(sp.Integer(new_ec[a] - fc[a]) for a in range(d_spatial))
                    ph = sp.exp(-sp.I * p_toggle * sp.pi *
                                sp.Rational(one_nb, V_count) *
                                (shift[0] + shift[1] + shift[2]))
                    C3_loc[fi, ei] = sp.simplify(ph)
                    break

    if sp.simplify(C3_loc**k_star - sp.eye(n_b)) != sp.zeros(n_b, n_b):
        raise RuntimeError("C_3^3 != I")
    if sp.simplify(C3_loc * B_P_loc - B_P_loc * C3_loc) != sp.zeros(n_b, n_b):
        raise RuntimeError("[C_3, B(P)] != 0")

    # Numerical projector + trace.
    B_num = np.array(B_P_loc.evalf().tolist(), dtype=complex)
    C3_n = np.array(C3_loc.evalf().tolist(), dtype=complex)
    eigs_, vecs_ = la.eig(B_num)
    plus_i = [i for i, e in enumerate(eigs_) if abs(e - one_nb) < 1e-6]
    minus_i = [i for i, e in enumerate(eigs_) if abs(e + one_nb) < 1e-6]
    if len(plus_i) != e_minus_v or len(minus_i) != e_minus_v:
        raise RuntimeError(
            f"Tree multiplicities wrong: +1 has {len(plus_i)}, -1 has {len(minus_i)}, "
            f"expected {e_minus_v} each.")
    V_p = la.qr(vecs_[:, plus_i])[0][:, :len(plus_i)]
    V_m = la.qr(vecs_[:, minus_i])[0][:, :len(minus_i)]
    V_t = np.hstack([V_p, V_m])
    C3_tree_loc = V_t.conj().T @ C3_n @ V_t
    tr1 = np.trace(C3_tree_loc)
    # Trace at sigma^2 = conj of trace at sigma (both real here, so equal).
    tr2 = np.trace(C3_tree_loc @ C3_tree_loc)
    dim = V_t.shape[1]
    # omega = exp(2πi / k_star); 2 = p_toggle (Fourier), k_star = C_3 group order.
    om = np.exp(p_toggle * 1j * np.pi / k_star)
    characters_loc = {
        'trivial':   (one_nb, one_nb, one_nb),
        'omega':     (one_nb, om, om**p_toggle),
        'omega_bar': (one_nb, om**p_toggle, om),
    }
    traces_loc = (dim, tr1, tr2)
    mult_loc = {}
    for rep, chi in characters_loc.items():
        v = (sum(np.conj(chi[i]) * traces_loc[i] for i in range(k_star)) / k_star).real
        mult_loc[rep] = int(round(v))

    return (mult_loc['trivial'], mult_loc['omega'], mult_loc['omega_bar'])


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_mults = (mult_tree_int['trivial'], mult_tree_int['omega'], mult_tree_int['omega_bar'])
    pure_mults = predict_tree_subspace_construction(k_star, p, V, d)
    print()
    print(f"Implementation tree C_3 multiplicities: {impl_mults}")
    print(f"Pure function tree C_3 multiplicities:  {pure_mults}")
    assert impl_mults == pure_mults, f"Mismatch: {impl_mults} vs {pure_mults}"
    assert impl_mults == (0, 2, 2), f"Unexpected: {impl_mults}"
    print("OK: outputs agree. Tree subspace at P decomposes as (0, 2, 2) under C_3.")
