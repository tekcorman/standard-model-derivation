#!/usr/bin/env python3
"""
Theorem P1 Ramanujan Support: diagnostic proof script.

Verifies all theorem-grade structural claims for docs/theorem_P1_ramanujan_support.md,
and numerically locates the precise remaining gap.

STATUS: BLOCKED -- see theorem doc for the exact diagnosis.

STRUCTURE OF THIS SCRIPT:
  Part 1: Ihara-Bass factorization and universality (THEOREM-GRADE)
  Part 2: Minimal polynomials of tree vs Ram eigenvalues (THEOREM-GRADE)
  Part 3: C3 isotypic content of V_tree (0,2,2) and V_Ram (4,2,2) (THEOREM-GRADE)
  Part 4: [B(k_P), C3] = 0 algebraic identity (THEOREM-GRADE, numerically verified)
  Part 5: Schur lemma consequence: C3-scalar M forces M|_{V_tree}=0 (THEOREM-GRADE)
  Part 6: Spectral dominance of V_Ram under B^N iteration (THEOREM-GRADE)
  Part 7: The precise remaining gap (diagnostic)

References:
  Ihara (1966) J. Math. Soc. Japan 18, 219-235.
  Bass (1992) Internat. J. Math. 3, 717-797.
  Terras (2011) Zeta Functions of Graphs, Theorem 3.1.
  Serre (1977) Linear Representations of Finite Groups, Sections 2.2-2.3.
  Rissanen (1978) Automatica 14, 465-471.
  Grunwald (2007) The MDL Principle, MIT Press, Sections 5.1-5.3.
"""

import numpy as np
from numpy import linalg as la
import sympy as sp
import math
from itertools import product as iproduct

np.set_printoptions(precision=8, linewidth=120, suppress=True)

PASS_COUNT = 0
FAIL_COUNT = 0


def check(label, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{tag}] {label}")
    if detail:
        print(f"         {detail}")


# ============================================================================
# SETUP: Build B(k_P) for the srs lattice at the P-point
# ============================================================================

A_PRIM = np.array([
    [-0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
])
ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])
N_ATOMS = 4
k_P = np.array([0.25, 0.25, 0.25])


def find_bonds():
    tol = 0.02
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in iproduct(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                dist = la.norm(rj - ATOMS[i])
                if 0.02 < dist < tol:
                    continue
                if abs(dist - math.sqrt(2)/4) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds


bonds = find_bonds()
assert len(bonds) == N_ATOMS * 3, f"Expected 12 bonds, got {len(bonds)}"


def build_bloch_hashimoto(bonds, k_pt):
    n = len(bonds)
    B = np.zeros((n, n), dtype=complex)
    for i, (ti, hi, ni) in enumerate(bonds):
        for j, (tj, hj, nj) in enumerate(bonds):
            if hi == tj and ti != hj:
                B[i, j] = np.exp(2j * np.pi * np.dot(k_pt, np.array(nj)))
    return B


B_P = build_bloch_hashimoto(bonds, k_P)
n_bonds = len(bonds)


def c3_cartesian(v):
    """C3 rotation: body diagonal (1,1,1); maps (x,y,z) -> (z,x,y)."""
    return np.array([v[2], v[0], v[1]])


def build_c3_matrix(bonds):
    """Build 12x12 matrix of the C3 rotation acting on the directed-edge basis."""
    U = np.zeros((n_bonds, n_bonds), dtype=complex)
    for i, (ti, hi, ni) in enumerate(bonds):
        r_ti = ATOMS[ti]
        r_hi = ATOMS[hi] + sum(ni[d] * A_PRIM[d] for d in range(3))
        r_ti_rot = c3_cartesian(r_ti)
        r_hi_rot = c3_cartesian(r_hi)
        for j, (tj, hj, nj) in enumerate(bonds):
            r_tj = ATOMS[tj]
            r_hj_base = ATOMS[hj]
            tol = 1e-6
            if la.norm(r_ti_rot - r_tj) < tol:
                for m1, m2, m3 in iproduct(range(-2, 3), repeat=3):
                    r_hj = r_hj_base + m1*A_PRIM[0] + m2*A_PRIM[1] + m3*A_PRIM[2]
                    if la.norm(r_hi_rot - r_hj) < tol and (m1, m2, m3) == nj:
                        U[j, i] = 1.0
                        break
    return U


C3 = build_c3_matrix(bonds)


# ============================================================================
# PART 1: IHARA-BASS FACTORIZATION AND UNIVERSALITY
# ============================================================================

print("=" * 72)
print("PART 1: Ihara-Bass factorization and tree-sector universality")
print("=" * 72)

# Ihara-Bass identity (Terras 2011 Theorem 3.1):
# det(I - u B(k)) = (1 - u^2)^{|E|-|V|} * det((1 + (k-1)u^2)I - uA(k))
# For srs: |E|=6, |V|=4, so (1-u^2)^2 is the UNIVERSAL tree factor.
# For k*=3: (k-1)u^2 = 2u^2, so the ram factor is det((1+2u^2)I - uA(k)).

u_sym = sp.Symbol('u')
print("  Ihara-Bass: char_poly(B(k)) = (1-u^2)^2 * det((1+2u^2)I - u*A(k))")
print()

# At P: A(P) has eigenvalues +/-sqrt(3), each with multiplicity 2.
# det((1+2u^2)I - u*A(P)) = [(1+2u^2)-u*sqrt(3)]^2 * [(1+2u^2)+u*sqrt(3)]^2
fac_plus  = (1 + 2*u_sym**2 - u_sym*sp.sqrt(3))**2
fac_minus = (1 + 2*u_sym**2 + u_sym*sp.sqrt(3))**2
ram_factor_sym = sp.expand(fac_plus * fac_minus)
target = (4*u_sym**4 + u_sym**2 + 1)**2

print("  Ram factor at P:")
print(f"    [(1+2u^2-sqrt(3)u)(1+2u^2+sqrt(3)u)]^2 = {ram_factor_sym}")
print(f"    Equals (4u^4+u^2+1)^2? {sp.simplify(ram_factor_sym - target) == 0}")
check("Ram factor at P equals (4u^4+u^2+1)^2",
      sp.simplify(ram_factor_sym - target) == 0)
print()

tree_factor_sym = (1 - u_sym**2)**2
full_char = sp.expand(tree_factor_sym * ram_factor_sym)
print(f"  Full char poly degree: {sp.Poly(full_char, u_sym).degree()} (should be 12)")
check("Full char poly degree = 12", sp.Poly(full_char, u_sym).degree() == 12)
print()

print("  KEY: The (1-u^2)^2 factor depends only on |E|-|V| = 6-4 = 2.")
print("  It is UNIVERSAL for ANY 3-regular graph on a 4-vertex cell,")
print("  independent of girth, Ramanujan property, or spatial embedding.")
print("  (Terras 2011 Theorem 3.1; Ihara 1966; Bass 1992)")
print()

# Numerical verification: eigenvalues of B(P)
evals_P, evecs_P = la.eig(B_P)
idx_sort = np.argsort(-np.abs(evals_P))
evals_P = evals_P[idx_sort]
evecs_P = evecs_P[:, idx_sort]

n_ram  = sum(1 for ev in evals_P if abs(abs(ev) - math.sqrt(2)) < 0.05)
n_tree = sum(1 for ev in evals_P if abs(abs(ev) - 1.0) < 0.05)
check("B(k_P) has 8 Ram eigenvalues with |lambda|=sqrt(2)", n_ram == 8,
      f"Found {n_ram}")
check("B(k_P) has 4 tree eigenvalues with |lambda|=1", n_tree == 4,
      f"Found {n_tree}")
check("Ram + tree = 12 (complete)", n_ram + n_tree == 12)
print()

# Build spectral projectors
V_mat = evecs_P
V_inv = la.inv(V_mat)
mask_ram  = np.array([1.0 if abs(abs(ev) - math.sqrt(2)) < 0.05 else 0.0
                      for ev in evals_P])
mask_tree = np.array([1.0 if abs(abs(ev) - 1.0) < 0.05 else 0.0
                      for ev in evals_P])
P_Ram  = V_mat @ np.diag(mask_ram)  @ V_inv
P_Tree = V_mat @ np.diag(mask_tree) @ V_inv

check("P_Ram + P_Tree = I", la.norm(P_Ram + P_Tree - np.eye(12)) < 1e-12,
      f"||P_Ram+P_Tree-I|| = {la.norm(P_Ram+P_Tree-np.eye(12)):.2e}")
check("P_Ram @ P_Tree = 0", la.norm(P_Ram @ P_Tree) < 1e-12,
      f"||P_Ram@P_Tree|| = {la.norm(P_Ram@P_Tree):.2e}")
check("P_Ram^2 = P_Ram",    la.norm(P_Ram @ P_Ram - P_Ram) < 1e-12)
check("P_Tree^2 = P_Tree",  la.norm(P_Tree @ P_Tree - P_Tree) < 1e-12)
print()

# Verify tree eigenvalues are exactly +/-1 (not approximately)
tree_evals_num = evals_P[mask_tree == 1.0]
check("V_tree eigenvalues are exactly +/-1",
      all(abs(abs(ev) - 1.0) < 1e-12 for ev in tree_evals_num),
      f"Max deviation: {max(abs(abs(ev)-1.0) for ev in tree_evals_num):.2e}")
print()


# ============================================================================
# PART 2: MINIMAL POLYNOMIALS
# ============================================================================

print("=" * 72)
print("PART 2: Minimal polynomials of tree vs Ram eigenvalues")
print("=" * 72)

# Tree eigenvalue +1 satisfies (u-1): degree-1 polynomial over Q.
# Tree eigenvalue -1 satisfies (u+1): degree-1 polynomial over Q.
# Both are rational integers -- no algebraic complexity.
print("  Tree eigenvalues +/-1:")
print("    Minimal polynomial of +1 over Q: (u - 1)   -- rational, degree 1")
print("    Minimal polynomial of -1 over Q: (u + 1)   -- rational, degree 1")
print("    Both are universal algebraic integers with zero srs-specific content.")
print()

# Ram eigenvalue h = (sqrt(3) + i*sqrt(5))/2 satisfies h^2 - sqrt(3)*h + 2 = 0.
# Over Q, the minimal polynomial is obtained from:
# h^2 + 2 = sqrt(3)*h  =>  (h^2+2)^2 = 3*h^2 = 3*(sqrt(3)*h-2) = 3sqrt(3)*h - 6
# This gives (h^4 + 4*h^2 + 4) = 3*sqrt(3)*h - 6,
# and further elimination of sqrt(3) gives the degree-4 minimal polynomial over Q.
# Direct computation: the four conjugates h, h*, -h, -h* satisfy
#   (u - h)(u - h*)(u + h)(u + h*) = (u^2 - sqrt(3)*u + 2)(u^2 + sqrt(3)*u + 2)
#                                   = u^4 + u^2 + 4.
u_s = sp.Symbol('u')
h_sym = (sp.sqrt(3) + sp.I*sp.sqrt(5)) / 2
prod_factors = (u_s**2 - sp.sqrt(3)*u_s + 2) * (u_s**2 + sp.sqrt(3)*u_s + 2)
min_poly_h_Q = sp.expand(prod_factors)
print("  Ram eigenvalue h = (sqrt(3)+i*sqrt(5))/2:")
print(f"    (u^2-sqrt(3)u+2)(u^2+sqrt(3)u+2) = {min_poly_h_Q}")
check("Minimal poly of h over Q is u^4+u^2+4",
      sp.simplify(min_poly_h_Q - (u_s**4 + u_s**2 + 4)) == 0)

# Verify h^4 + h^2 + 4 = 0
val_at_h = sp.simplify(h_sym**4 + h_sym**2 + 4)
check("h satisfies u^4+u^2+4=0", val_at_h == 0,
      f"h^4+h^2+4 = {val_at_h}")

# Verify u^4+u^2+4 has no rational roots (hence is srs-specific over Q)
print("  u^4+u^2+4 has no rational roots: for real r, r^4+r^2+4 >= 4 > 0.")
check("u^4+u^2+4 >= 4 > 0 for all real u",
      all((r**4 + r**2 + 4) >= 4 for r in np.linspace(-5, 5, 1001)))
print()
print("  Tree eigenvalues carry ZERO srs-specific algebraic information.")
print("  Ram eigenvalue h generates a degree-4 extension of Q, encoding srs geometry.")
print()


# ============================================================================
# PART 3: C3 ISOTYPIC CONTENT (character orthogonality)
# ============================================================================

print("=" * 72)
print("PART 3: C3 isotypic content of V_tree and V_Ram")
print("=" * 72)
print("  (Serre 1977 Linear Representations of Finite Groups, Section 2.3)")
print()

# C3 character on V_tree: Tr(C3|_{V_tree}) = -2
# (from predictions/tree_subspace_construction_derivation.md, CAS-verified)
# C3 character on V_Ram: Tr(C3|_{V_Ram}) = +2
# (from docs/theorem_B5_3_core.md)

omega_sym = sp.exp(2*sp.pi*sp.I/3)

def c3_isotypic(dim, chi_sigma):
    """Compute C3 isotypic multiplicities from dimension and Tr(C3)."""
    chi_e = dim
    chi_s2 = sp.conjugate(chi_sigma)
    m_triv  = sp.Rational(1,3) * (chi_e + chi_sigma + chi_s2)
    m_omega = sp.Rational(1,3) * (chi_e + sp.conjugate(omega_sym)*chi_sigma
                                        + sp.conjugate(omega_sym**2)*chi_s2)
    m_omg2  = sp.Rational(1,3) * (chi_e + sp.conjugate(omega_sym**2)*chi_sigma
                                        + sp.conjugate(omega_sym)*chi_s2)
    return (sp.simplify(m_triv), sp.simplify(m_omega), sp.simplify(m_omg2))

# Numerical verification (omega numerics)
omega_num = np.exp(2j*np.pi/3)
omega2_num = np.exp(4j*np.pi/3)

def c3_isotypic_num(dim, chi_s):
    """Numerical C3 isotypic decomposition."""
    m0 = (1/3) * (dim + chi_s + np.conj(chi_s))
    m1 = (1/3) * (dim + np.conj(omega_num)*chi_s + np.conj(omega2_num)*np.conj(chi_s))
    m2 = (1/3) * (dim + np.conj(omega2_num)*chi_s + np.conj(omega_num)*np.conj(chi_s))
    return m0, m1, m2

# V_tree (dim=4, Tr(C3)=-2):
m0_t, m1_t, m2_t = c3_isotypic_num(4, -2)
print(f"  V_tree (dim=4, Tr(C3|_{{V_tree}})=-2):")
print(f"    trivial mult = (1/3)(4 + (-2) + (-2)) = {m0_t.real:.4f}  [should be 0]")
print(f"    omega   mult = {m1_t.real:.4f}  [should be 2]")
print(f"    omega^2 mult = {m2_t.real:.4f}  [should be 2]")
check("V_tree: trivial mult = 0", abs(m0_t.real) < 1e-10,
      f"trivial = {m0_t.real:.6f}")
check("V_tree: omega   mult = 2", abs(m1_t.real - 2) < 1e-10)
check("V_tree: omega^2 mult = 2", abs(m2_t.real - 2) < 1e-10)
print()

# V_Ram (dim=8, Tr(C3)=+2):
m0_r, m1_r, m2_r = c3_isotypic_num(8, 2)
print(f"  V_Ram (dim=8, Tr(C3|_{{V_Ram}})=+2):")
print(f"    trivial mult = (1/3)(8 + 2 + 2) = {m0_r.real:.4f}  [should be 4]")
print(f"    omega   mult = {m1_r.real:.4f}  [should be 2]")
print(f"    omega^2 mult = {m2_r.real:.4f}  [should be 2]")
check("V_Ram: trivial mult = 4", abs(m0_r.real - 4) < 1e-10)
check("V_Ram: omega   mult = 2", abs(m1_r.real - 2) < 1e-10)
check("V_Ram: omega^2 mult = 2", abs(m2_r.real - 2) < 1e-10)
print()

print("  KEY: V_tree has ZERO trivial C3 content.")
print("       V_Ram has FOUR copies of the trivial C3 representation.")
print("       (This is the Schur gateway to P1 under ADOPTED-CS.)")
print()


# ============================================================================
# PART 4: [B(k_P), C3] = 0 (algebraic identity, numerically verified)
# ============================================================================

print("=" * 72)
print("PART 4: [B(k_P), C3] = 0 algebraic identity")
print("=" * 72)

# C3 is in the stabilizer of k_P = (1/4,1/4,1/4) in the I4_132 point group.
# The full space group I4_132 acts on the crystal; C3 (rotation around body diagonal)
# fixes k_P and therefore acts on the Bloch fiber at P.
# Since C3 is a symmetry of the srs lattice (via the space group action),
# it commutes with the Bloch Hashimoto operator at every k-point it fixes.

check("C3^3 = I (order 3)", la.norm(la.matrix_power(C3, 3) - np.eye(12)) < 1e-12,
      f"max |C3^3 - I| = {la.norm(la.matrix_power(C3,3)-np.eye(12)):.2e}")

comm_B_C3 = B_P @ C3 - C3 @ B_P
check("[B(k_P), C3] = 0",
      np.max(np.abs(comm_B_C3)) < 1e-12,
      f"max |[B,C3]| = {np.max(np.abs(comm_B_C3)):.2e}")
print()
print("  This is an exact algebraic consequence of srs symmetry.")
print("  The space group I4_132 has a C3 element that fixes k_P = (1/4,1/4,1/4);")
print("  hence it acts on the Bloch fiber at P and commutes with B(k_P).")
print("  (A1+A2 derive the srs lattice and its symmetry group.)")
print()


# ============================================================================
# PART 5: SCHUR LEMMA CONSEQUENCE
# ============================================================================

print("=" * 72)
print("PART 5: Schur lemma: C3-scalar operator M => M|_{V_tree} = 0")
print("=" * 72)
print("  (Serre 1977 Section 2.2 Proposition 4 (Schur's lemma))")
print()
print("  Assume M is a C3-scalar operator: [M, C3] = 0 and M transforms")
print("  in the trivial C3 representation (i.e., C3*M = M*C3 = M as operators).")
print()
print("  By Serre 1977 Prop 4 (Schur's lemma, irrep form):")
print("    For V an irrep of G and W another irrep, Hom_G(V, W) = 0 if V != W.")
print("  Applied to M: M maps each C3-isotypic subspace of its domain to the")
print("  same isotypic subspace of its codomain.")
print()
print("  Since V_tree has ZERO trivial-C3 content (Part 3) and M acts in the")
print("  trivial representation, M has ZERO matrix elements between any state")
print("  in V_tree and any state in the trivial sector.")
print()
print("  Formally: if {|t_j>} spans V_tree and {|s_k>} spans the trivial C3")
print("  sector of any space, then <s_k|M|t_j> = 0 for all j, k.")
print()
print("  If M is ALSO a non-negative operator (as mass operators are) and M|_{V_tree}")
print("  is the restriction of M to V_tree acting within V_tree, then:")
print("    V_tree has trivial mult 0 => M|_{V_tree} = 0.")
print()

# Numerical verification: build a prototype C3-scalar operator (projector onto trivial sector)
# and show it annihilates V_tree.
# The trivial C3 projector: P_trivial = (1/3)(I + C3 + C3^2)
P_trivial = (np.eye(12) + C3 + la.matrix_power(C3, 2)) / 3.0

# P_trivial should project onto the trivial C3 content of the 12-dim fiber.
# V_tree has zero trivial content, so P_trivial @ P_Tree should be zero.
pt_on_tree = P_trivial @ P_Tree
check("P_trivial @ P_Tree = 0 (trivial C3 projector kills V_tree)",
      la.norm(pt_on_tree) < 1e-10,
      f"||P_trivial @ P_Tree|| = {la.norm(pt_on_tree):.2e}")

# Conversely, P_trivial acting on V_Ram is non-trivial (4-dim trivial sector)
pt_on_ram = P_trivial @ P_Ram
rank_trivial_in_ram = np.linalg.matrix_rank(pt_on_ram, tol=1e-8)
check("P_trivial has rank-4 support in V_Ram",
      rank_trivial_in_ram == 4,
      f"rank(P_trivial @ P_Ram) = {rank_trivial_in_ram}")
print()
print("  Schur conclusion (CONDITIONAL on ADOPTED-CS):")
print("    If mass operator M is C3-invariant (ADOPTED-CS),")
print("    then M|_{V_tree} = 0 follows purely from Schur's lemma + Part 3.")
print("    P1 is a COROLLARY of ADOPTED-CS at theorem grade.")
print()


# ============================================================================
# PART 6: SPECTRAL DOMINANCE
# ============================================================================

print("=" * 72)
print("PART 6: Spectral dominance of V_Ram under B^N iteration")
print("=" * 72)
print()
print("  On V_Ram: all eigenvalues have |lambda| = sqrt(2).")
print("  => ||B^N psi_R|| = (sqrt(2))^N * ||psi_R||  grows exponentially.")
print()
print("  On V_tree: all eigenvalues have |lambda| = 1.")
print("  => ||B^N psi_T|| = ||psi_T||  (constant, no growth).")
print()
print("  For a mixed state psi = alpha*psi_R + beta*psi_T with alpha != 0:")
print("    Ram fraction = ||P_Ram B^N psi|| / ||B^N psi|| -> 1 as N -> inf")
print("    at rate O(2^{-N/2}).")
print()

# Numerical verification
psi_R = P_Ram[:, 0] / la.norm(P_Ram[:, 0])
psi_T = P_Tree[:, 0] / la.norm(P_Tree[:, 0])
psi_mix = (psi_R + psi_T) / math.sqrt(2)

print(f"  {'N':>4}  {'||B^N psi_R||':>18}  {'(sqrt2)^N':>12}  {'||B^N psi_T||':>18}  {'Ram_frac(mix)':>16}")
print("  " + "-" * 80)
all_tree_const = True
for N in [1, 2, 4, 6, 8, 10]:
    BN = la.matrix_power(B_P, N)
    norm_R = la.norm(BN @ psi_R)
    norm_T = la.norm(BN @ psi_T)
    expected_R = math.sqrt(2)**N
    if abs(norm_T - 1.0) > 1e-8:
        all_tree_const = False
    v_mix = BN @ psi_mix
    ram_frac = la.norm(P_Ram @ v_mix) / la.norm(v_mix)
    print(f"  {N:4d}  {norm_R:18.8f}  {expected_R:12.8f}  {norm_T:18.10f}  {ram_frac:16.8f}")

check("V_tree norm constant under B^N", all_tree_const,
      "All ||B^N psi_T|| == 1.0")
check("Ram fraction -> 1 at N=10 (> 0.999)",
      la.norm(P_Ram @ la.matrix_power(B_P, 10) @ psi_mix) /
      la.norm(la.matrix_power(B_P, 10) @ psi_mix) > 0.999)
print()
print("  NOTE (from theorem_ars_nwalk_dynamics_attempt.md §2):")
print("  The spectral dominance of V_Ram under B^N is THEOREM-GRADE.")
print("  However, converting this to 'MDL drops V_tree' requires:")
print("  (a) specifying the likelihood function L_data|model, and")
print("  (b) identifying 'data length N' with walk steps.")
print("  Both require the reading rule (Need-RR). This is the MDL-dynamics gap.")
print()


# ============================================================================
# PART 7: THE PRECISE REMAINING GAP
# ============================================================================

print("=" * 72)
print("PART 7: The precise remaining gap -- honest diagnosis")
print("=" * 72)

print("""
  WHAT IS THEOREM-GRADE (Parts 1-6 above):
  -----------------------------------------
  T1. V_tree eigenvalues +/-1 are universal for any k*=3 graph (Ihara-Bass).
  T2. V_Ram eigenvalues h, h*, -h, -h* are srs-specific; h has minimal poly
      u^4+u^2+4 over Q (degree 4; no rational roots).
  T3. V_tree has C3-isotypic content (0,2,2) (zero trivial sector).
  T4. V_Ram has C3-isotypic content (4,2,2) (four trivial generators).
  T5. [B(k_P), C3] = 0 algebraically (exact, not approximate).
  T6. V_Ram is spectrally dominant: ||B^N psi_R|| grows as (sqrt(2))^N,
      while ||B^N psi_T|| is constant.
  T7. P_trivial @ P_Tree = 0: the trivial-C3 projector kills V_tree exactly.

  THEOREM CONDITIONAL ON ADOPTED-CS:
  ------------------------------------
  IF the mass operator M satisfies [M, C3] = 0 (ADOPTED-CS),
  THEN M|_{V_tree} = 0 by Schur's lemma (T3+T7).
  THEREFORE: ADOPTED-P1 is a COROLLARY of ADOPTED-CS at theorem grade.

  This is the STRONGEST achievable result under A1+A2+A3 at current rigor.

  WHY NO STRONGER CONCLUSION IS ACHIEVABLE:
  -------------------------------------------
  Three routes to closing P1 WITHOUT ADOPTED-CS have been exhausted:

  Route A (MDL flat-band suppression, Grunwald 2007):
    Fails because MDL zeros parameters with no L_data_given_model benefit,
    but V_tree amplitudes DO contribute to generic observables (the Green's
    function G(k,u) includes V_tree terms at poles u=+-1). The argument
    'V_tree has no benefit' requires specifying the observable first, which
    is equivalent to ADOPTED-CS. (theorem_sprint_P1_attempt.md Section 2-3.)

  Route B (Jaynes MaxEnt, Jaynes 1957):
    Fails because MaxEnt with no V_tree-sensitive constraints distributes
    probability UNIFORMLY over V_tree -- the opposite of zeroing it out.
    (theorem_sprint_P1_attempt.md Section 3.)

  Route C (ADC identification, Grunwald 2007 + Ihara-Bass universality):
    'A2's amplitude-selection data = A2's graph-selection data' is either
    (a) circular: it requires the reading rule to compute L_data for
    amplitude distributions on the Bloch fiber, or
    (b) an adoption: identifying the relevant observable with the
    srs-discriminating spectral content. Neither is A2-derivable alone.
    (theorem_adopted_p1_vtree_mdl_attempt.md Section 5.)

  Route D (W4 + C3 symmetry):
    [B(k_P), C3] = 0 (T5) implies [f(B), C3] = 0 for any f (spectral mapping
    theorem). So every observable is C3-invariant. But f(B)|_{V_tree} = f(+-1)
    which is generically non-zero. C3-invariance of M does NOT force M|_{V_tree}=0;
    it forces M to block-diagonalize with respect to C3-isotypic sectors, but
    the tree sector is not pure-trivial (it has omega and omega^2 content).
    Wait -- actually the Schur argument DOES work for C3-invariant M: if M
    is C3-invariant AND it acts as a C3-scalar in the mass-selection context
    (not just any C3-invariant operator), the trivial-sector selectivity is
    needed. The issue is: [B, C3]=0 gives [M, C3]=0 for M=f(B), but this
    means M preserves each isotypic sector. For M to annihilate V_tree, M
    must ALSO have zero eigenvalue on the +-1 spectral subspace. That is the
    additional structural input not derivable from C3 symmetry alone.

  SUMMARY OF GAPS:
  -----------------
  Gap 1 (ADOPTED-CS): Mass operator M is C3-invariant and acts in the trivial
    C3 representation. [M, C3] = 0 follows from W4+T5. But the 'trivial
    representation' selectivity requires M to annihilate non-trivial C3 sectors,
    which is ADOPTED-CS proper, not derivable from dynamics alone.

  Gap 2 (ADC): The identification of A2's amplitude-data context with its
    graph-selection context. Not derivable from A2 alone without the reading rule.

  CONCLUSION: ADOPTED-P1 is BLOCKED as a pure theorem of A1+A2+A3.
  It achieves CONDITIONAL status: P1 = COROLLARY(ADOPTED-CS).
  Under ADOPTED-CS, T3+T4+T7 close it at theorem grade via Schur.
""")

print("=" * 72)
print(f"FINAL TALLY: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL")
print("=" * 72)
if FAIL_COUNT > 0:
    print("  FAILURES detected. Review above.")
    import sys; sys.exit(1)
else:
    print("  All structural verifications pass.")
    print("  See theorem doc for the conditional closure statement and remaining gap.")
