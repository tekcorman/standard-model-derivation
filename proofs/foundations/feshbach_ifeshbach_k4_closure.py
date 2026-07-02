#!/usr/bin/env python3
"""
feshbach_ifeshbach_k4_closure.py
=================================

Explicit finite computation on the K_4 quotient (srs primitive cell)
attempting formal closure of I-Feshbach per the instructions of
../../predictions/Feshbach_coupling_strength_derivation.md §9.5.

CLOSURE TARGET (§9.5):

    C_{g-2} := P B(P) (Q B(P))^{g-2} Q B(P) P
             = alpha_1^bare * (k-1)^{g-2} * (orientation projector)

where
    g = 10, k = 3, g-2 = 8
    alpha_1^bare = (2/3)^8 = 256/6561
    B(P) = 12x12 Bloch Hashimoto at P = (1/4,1/4,1/4), srs primitive cell
    P    = projector onto h-eigenspace of B(P), h = (sqrt(3)+i*sqrt(5))/2
    Q    = I - P

RESULT:

    C_{g-2} = 0 identically. Not a numerical accident: algebraically exact.

REASON:

    P is the eigenspace projector of B(P) at eigenvalue h. By the eigenspace
    identity B P = h P, we get:

        Q B P = (I - P) B P = B P - P B P = h P - h P^2 = h P - h P = 0.

    Therefore C_n = P B (QB)^n (QBP) = P B (QB)^n * 0 = 0 for all n >= 0.

    The Feshbach self-energy of an eigenspace of its own Hamiltonian is
    identically zero. This is a structural obstruction, not a numeric failure.

    This script lives in proofs/foundations/ (not predictions/) because the
    result is BLOCKED — predictions/ is reserved for theorem-grade or
    Feshbach-pattern pairs per an internal note.

CONCLUSION:

    I-Feshbach DOES NOT CLOSE via the stated computation. The closure target
    is ill-posed: it requires C_{g-2} != 0, but C_{g-2} = 0 exactly because
    P projects onto an eigenspace of B(P) itself.

    See an internal working note for the full diagnosis.

FRAMEWORK INPUTS:
    predictions/k_star.py          -> k* = 3
    predictions/g_girth.py         -> g = 10
    predictions/alpha_1.py         -> alpha_1_bare = (2/3)^8
    predictions/B_P_doubly_degenerate_h.py  -> h eigenspace of B(P), mult 2
    proofs/cosmology/srs_photon_bloch_primitive.py -> nb_walk_operator()
    ../../predictions/Feshbach_coupling_strength_derivation.md §9.5 -> closure target
"""

import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'proofs', 'cosmology'))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    nb_walk_operator,
)

P_POINT = np.array([0.25, 0.25, 0.25])

# ============================================================
# Step 1: Build B(P), the 12x12 Bloch Hashimoto at P-point
# ============================================================

verts, lat_vecs = build_primitive_unit_cell()
bonds = find_primitive_connectivity(verts, lat_vecs)
assert len(bonds) == 12, f"Expected 12 directed bonds, got {len(bonds)}"
B = nb_walk_operator(P_POINT, bonds)
assert B.shape == (12, 12), f"Expected (12, 12), got {B.shape}"

# ============================================================
# Step 2: Algebraic proof that Q B P = 0 for any eigenspace P of B
# ============================================================
# Theorem: If B v = h v for all v in range(P), then QBP = 0.
# Proof: QBP = (I-P)BP = BP - PBP = hP - P(hP) = hP - hP^2 = hP - hP = 0.

h = (np.sqrt(3) + 1j * np.sqrt(5)) / 2

eigenvalues, eigenvectors = np.linalg.eig(B)

TOL = 1e-8
h_indices = [i for i, lam in enumerate(eigenvalues) if abs(lam - h) < TOL]
assert len(h_indices) == 2, (
    f"Expected 2 eigenvalues at h, found {len(h_indices)}. "
    f"Check B(P) construction and TOL."
)

V_h = eigenvectors[:, h_indices]
V_h_orth, _ = np.linalg.qr(V_h)
I12 = np.eye(12, dtype=complex)
P_proj = V_h_orth @ V_h_orth.conj().T
Q_proj = I12 - P_proj

# ============================================================
# Step 3: Verify B P = h P numerically
# ============================================================

BP = B @ P_proj
hP = h * P_proj
err_eigenspace = np.max(np.abs(BP - hP))

# ============================================================
# Step 4: Verify Q B P = 0 numerically
# ============================================================

QBP = Q_proj @ B @ P_proj
err_QBP = np.max(np.abs(QBP))

# ============================================================
# Step 5: Compute C_8 = P B (QB)^8 Q B P and verify it is 0
# ============================================================

QB = Q_proj @ B
QB8 = I12.copy()
for _ in range(8):
    QB8 = QB8 @ QB

C8 = P_proj @ B @ QB8 @ Q_proj @ B @ P_proj
err_C8 = np.max(np.abs(C8))

# ============================================================
# Step 6: Results
# ============================================================

alpha_1_bare = (2 / 3) ** 8
k_minus_1_8 = 2 ** 8   # (k-1)^8 = 2^8 = 256
target = alpha_1_bare * k_minus_1_8

print("=" * 70)
print("  I-Feshbach K_4 closure attempt")
print("=" * 70)
print()
print(f"  B(P):  12x12 Bloch Hashimoto at P = (1/4,1/4,1/4)")
print(f"  h:     ({np.sqrt(3):.8f} + {np.sqrt(5):.8f}i)/2")
print(f"  h-eigenspace dimension:  2  (expected from theorem)")
print()
print(f"  --- Algebraic identity check ---")
print(f"  max |B P - h P|:  {err_eigenspace:.3e}  (expected: 0, machine precision ~1e-14)")
print(f"  max |Q B P|:      {err_QBP:.3e}  (expected: 0, exact identity QBP=0)")
print()
print(f"  --- C_8 computation ---")
print(f"  max |C_8|:        {err_C8:.3e}  (expected: 0 by QBP=0)")
print()
print(f"  --- Closure target ---")
print(f"  target = alpha_1_bare * 2^8 = {alpha_1_bare:.8f} * {k_minus_1_8} = {target:.8f}")
print(f"  obtained: C_8 = 0 (zero matrix)")
print()
print("  RESULT: C_8 = 0 identically.")
print("  REASON: P projects onto an eigenspace of B(P) => QBP = 0 => C_8 = 0.")
print("  STATUS: I-Feshbach DOES NOT CLOSE. See an internal working note")
print()

# ============================================================
# Assertions for regression testing
# ============================================================

assert err_eigenspace < 1e-10, (
    f"B P != h P: max |BP - hP| = {err_eigenspace:.3e}. "
    f"h-eigenspace construction is incorrect."
)
assert err_QBP < 1e-10, (
    f"QBP != 0: max |QBP| = {err_QBP:.3e}. "
    f"Algebraic identity violated (should be exact)."
)
assert err_C8 < 1e-8, (
    f"C_8 != 0: max |C_8| = {err_C8:.3e}. "
    f"Should be zero because QBP = 0."
)
print("  All assertions pass. Zero result confirmed numerically.")
print("=" * 70)


# ============================================================
# Pure function interface (for linter)
# ============================================================

def predict_feshbach_ifeshbach_k4_closure(k_star, g_girth):
    """
    Computes C_{g-2} = P B(P) (Q B(P))^{g-2} Q B(P) P and checks whether
    it equals alpha_1_bare * (k-1)^{g-2} * (orientation projector).

    Parameters
    ----------
    k_star : int
        Coordination number. Must be 3 (srs case).
    g_girth : int
        Girth. Must be 10 (srs case).

    Returns
    -------
    dict with keys:
        'C_n_max_abs'  : float  max |C_{g-2}|
        'QBP_max_abs'  : float  max |QBP|
        'BP_err'       : float  max |BP - hP|
        'target'       : float  alpha_1_bare * (k-1)^{g-2}
        'closed'       : bool   False (C_{g-2} = 0 != target)
        'diagnosis'    : str    explanation
    """
    if k_star != 3 or g_girth != 10:
        raise ValueError(
            f"I-Feshbach K_4 closure is implemented for k_star=3, g=10 only. "
            f"Got k_star={k_star}, g_girth={g_girth}."
        )
    return {
        'C_n_max_abs': float(err_C8),
        'QBP_max_abs': float(err_QBP),
        'BP_err': float(err_eigenspace),
        'target': float(target),
        'closed': False,
        'diagnosis': (
            "C_{g-2} = 0 identically: P is an eigenspace projector of B(P), "
            "so QBP = (I-P)BP = hP - hP = 0 and therefore "
            "C_n = PB(QB)^n QBP = 0 for all n. "
            "The Feshbach self-energy of an eigenspace of its own Hamiltonian "
            "is identically zero. The closure target requires a non-trivial "
            "Feshbach setup where P is NOT an eigenspace of B. "
            "See an internal working note for full diagnosis."
        ),
    }


if __name__ == "__main__":
    result = predict_feshbach_ifeshbach_k4_closure(3, 10)
    print()
    print(f"Pure function result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
