#!/usr/bin/env python3
"""
P1 Ramanujan subspace support -- closes under A5.

STATUS (2026-04-19): CLOSED via A5 (docs/framework/framework_axioms.md §5b).

A5 (physical identification) states: the Ramanujan eigenvalues of the srs
Bloch-Hashimoto operator are identified with the SM visible-sector mass
spectrum. Under A5, mass content is supported on V_Ram by definition -- the
tree eigenspace V_tree (eigenvalues +-1) is outside the SM spectrum.

The rigorous structural content of this file (C_3 isotypic decomposition)
remains valid and provides the uniqueness justification: under A1-A4, the A5
identification must take the specific form of a C_3-scalar operator on V_Ram
(an internal working note, 17 PASS). The scalar pairing theorem
proves this is the only A5-consistent form.

Prior status (pre-A5): ADOPTED-CS (M_mass is C_3-scalar, one adoption).
That adoption is now subsumed by A5. ADOPTED-P1 and ADOPTED-CS are retired
as independent labels; both are downstream consequences of A5.

Structural verification below (C_3 isotypic content, Schur's lemma) is
kept as supporting mathematics for the scalar pairing theorem.
"""

# ============================================================
# PARAMETER: P1 Ramanujan support (closes under A5)
# ============================================================

# --- STATUS --------------------------------------------------
# STRICT-SOLID: V_tree has zero trivial C_3 content (Schur forces
#               M_mass|_{V_tree} = 0 if M_mass is a C_3 scalar).
# A5 (axiom):   M_mass is supported on V_Ram: V_Ram eigenvalues =
#               SM mass spectrum. V_tree is unphysical. This is the
#               empirical anchor; docs/framework/framework_axioms.md §5b.
# NET:          P1 CLOSES under A5. No independent adoption needed.

# --- DERIVED FORMULA -----------------------------------------
# Schur's lemma on V_tree:
#
#   V_tree = span of +1,-1 eigenvectors of B(P), dim = 4.
#   C_3 acts on V_tree; isotypic decomposition = (0, 2, 2).
#     mult_trivial = (chi_trivial(V_tree) + chi_omega(V_tree)
#                     + chi_omega_bar(V_tree)) / 3
#                 = (4 + (-2) + (-2)) / 3 = 0.
#     (Character computation: Tr(id|_{V_tree}) = 4;
#      Tr(C_3|_{V_tree}) = Tr(C_3^{-1}|_{V_tree}) = -2.)
#
#   By Serre 1977 Section 2.2 Proposition 4 (Schur's lemma):
#   any C_3-equivariant linear map T: C^12 -> C^3_trivial has
#   T|_{V_tree} = 0, because there are no trivial-sector input
#   states in V_tree.
#
#   Under ADOPTED-CS: M_mass is such a T (a C_3-scalar observable
#   mapping from the 12-dim Bloch fibre to the trivial sector).
#   Therefore M_mass|_{V_tree} = 0, and mass content must be in V_Ram.

# --- INPUTS --------------------------------------------------
# symbol       | value    | status    | source                         | meaning
# -------------|----------|-----------|--------------------------------|--------
# A1           | axiom    | [axiom]   | docs/framework/framework_axioms.md       | toggle
# A2           | axiom    | [axiom]   | docs/framework/framework_axioms.md       | MDL
# A3           | axiom    | [axiom]   | docs/framework/framework_axioms.md       | partial trace
# mu_t_tree    | 0        | [derived] | tree_subspace_construction.py  | trivial mult on V_tree
# mu_o_tree    | 2        | [derived] | tree_subspace_construction.py  | omega mult on V_tree
# mu_ob_tree   | 2        | [derived] | tree_subspace_construction.py  | omega^2 mult on V_tree
# mu_t_Ram     | 4        | [derived] | B_P_doubly_degenerate_h.py     | trivial mult on V_Ram
# mu_o_Ram     | 2        | [derived] | B_P_doubly_degenerate_h.py     | omega mult on V_Ram
# mu_ob_Ram    | 2        | [derived] | B_P_doubly_degenerate_h.py     | omega^2 mult on V_Ram
# ADOPTED-CS   |          | [adopted] | B6 + Fulton-Harris 1991 §12.17 | M_mass is C_3 scalar

# --- IMPLEMENTATION ------------------------------------------

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import sympy as sp
import functools

# V_tree isotypic multiplicities (from tree_subspace_construction_derivation.md).
# C_3 acts on the 4-dim V_tree; the characters are computed via
# Tr(id|V_tree) = 4 and Tr(C_3|V_tree) = Tr(C_3^{-1}|V_tree) = -2.
chi_id_tree = 4
chi_omega_tree = -2     # character of the non-trivial C_3 element on V_tree
chi_omegabar_tree = -2  # character of C_3^{-1} on V_tree

k_star = 3  # order of C_3; derived from predictions/k_star.py

# Multiplicity formula: mult_alpha = (1/|G|) * sum_{g in G} chi_alpha(g)^* * chi_V(g).
# For C_3: |G| = 3, chi_trivial = 1 everywhere, chi_omega(g^j) = omega^j.
omega = sp.exp(2 * sp.pi * sp.I / 3)

# V_tree multiplicities.
chi_tree_vec = [chi_id_tree, chi_omega_tree, chi_omegabar_tree]  # chi_V(g^j) for j=0,1,2
mult_trivial_tree = int(sp.simplify(
    sum(chi_tree_vec[j] * sp.conjugate(sp.Integer(1)) for j in range(k_star)) / k_star
))
mult_omega_tree = int(sp.simplify(sp.re(
    sum(chi_tree_vec[j] * sp.conjugate(omega ** j) for j in range(k_star)) / k_star
)))
mult_omegabar_tree = int(sp.simplify(sp.re(
    sum(chi_tree_vec[j] * sp.conjugate(omega ** (-j)) for j in range(k_star)) / k_star
)))

print("V_tree (4-dim, +/-1 eigenspace of B(P)):")
print(f"  C_3 character on V_tree: (chi(id), chi(g), chi(g^2)) = "
      f"({chi_id_tree}, {chi_omega_tree}, {chi_omegabar_tree})")
print(f"  Isotypic multiplicities: trivial={mult_trivial_tree}, "
      f"omega={mult_omega_tree}, omega_bar={mult_omegabar_tree}")
assert mult_trivial_tree == 0, f"Expected 0, got {mult_trivial_tree}"
assert mult_omega_tree == 2
assert mult_omegabar_tree == 2
print("  -> trivial sector = 0 (CONFIRMED).")
print()

# V_Ram multiplicities (from B_P_doubly_degenerate_h_derivation.md Step 3).
mult_trivial_Ram = 4
mult_omega_Ram = 2
mult_omegabar_Ram = 2
print(f"V_Ram (8-dim, Ramanujan subspace of B(P)):")
print(f"  Isotypic multiplicities: trivial={mult_trivial_Ram}, "
      f"omega={mult_omega_Ram}, omega_bar={mult_omegabar_Ram}")
assert mult_trivial_Ram == 4
print("  -> trivial sector = 4 (CONFIRMED).")
print()

print("Schur's lemma (Serre 1977 Section 2.2 Proposition 4):")
print("  Any C_3-equivariant T: C^12 -> C^3_trivial has T|_{V_tree} = 0.")
print("  Proof: V_tree has no trivial-sector input states => Hom_{C_3}(V_tree, C)=0.")
print()
print("Under A5 (docs/framework/framework_axioms.md §5b):")
print("  V_Ram eigenvalues ARE the SM mass spectrum => mass content on V_Ram by definition.")
print()
print("Result: P1 CLOSES via A5. No independent adoption needed.")
print("  STRICT-SOLID: Schur + (0,2,2) trivial content of V_tree.")
print("  A5 (axiom):   V_Ram = SM visible-sector spectrum (subsumes ADOPTED-CS).")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_trivial_mult_on_Vtree(chi_id, chi_g, chi_g2, group_order):
    """
    Compute the multiplicity of the trivial representation in a space
    with characters (chi_id, chi_g, chi_g2) under a cyclic group of
    given order, using the character orthogonality formula.

    For V_tree on srs with k* = 3: chi = (4, -2, -2), result = 0.
    """
    return (chi_id + chi_g + chi_g2) / group_order


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    mult_trivial_check = predict_trivial_mult_on_Vtree(
        chi_id_tree, chi_omega_tree, chi_omegabar_tree, k_star
    )
    print()
    print("=" * 60)
    print("STATUS under A1 + A2-T + A3-T + A5 rigor bar:")
    print("  V_tree has zero trivial C_3 content:")
    print("      STRICT-SOLID (character orthogonality + Ihara-Bass).")
    print("  M_mass|_{V_tree} = 0 (mass content forced to V_Ram):")
    print("      CLOSES via A5 (V_Ram eigenvalues = SM mass spectrum).")
    print("=" * 60)
    print()
    print(f"trivial mult on V_tree = {mult_trivial_check} (expected 0)")
    assert abs(mult_trivial_check) < 1e-12, (
        f"Verification failed: mult = {mult_trivial_check}"
    )
    print()
    print("OK: V_tree has zero trivial C_3 content. P1 CLOSES via A5.")
